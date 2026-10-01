"""Prepare a diagnostic bilinear-streamfunction centroid control on native q.

The control shares the candidate's streamfunction and receiver liquid geometry.
The velocity is the exact liquid-volume mean of curl(piecewise-bilinear psi),
which equals that curl at each liquid polygon's centroid. No global momentum
correction is applied. Existing gas velocities are preserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from shapely import area, box, centroid, get_x, get_y, intersection

from campaign_integrity import sha256
from prepare_aphros_initial_state import (
    DOMAIN_LENGTH,
    VERTICAL_ORIGIN,
    periodic_liquid_geometry,
    target_mac_velocity_state,
)


def run(source: Path, input_dir: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    data = np.load(source)
    initial = json.loads((input_dir / "initial_state.json").read_text())
    nx, ny = int(initial["nx"]), int(initial["ny"])
    h = DOMAIN_LENGTH / nx
    x_edges = np.linspace(0, DOMAIN_LENGTH, nx + 1)
    y_edges = VERTICAL_ORIGIN + np.arange(ny + 1) * h
    q = np.fromfile(input_dir / "q.raw", dtype="<f8").reshape(ny, nx)
    current_u = np.fromfile(input_dir / "vx.raw", dtype="<f8").reshape(ny, nx)
    current_w = np.fromfile(input_dir / "vy.raw", dtype="<f8").reshape(ny, nx)
    bed = np.load(input_dir / "receiver_bed.npz")
    geometry = periodic_liquid_geometry(
        data["surface_x"],
        data["surface_z"] + initial["volume_gauge_surface_shift"],
        bed["x"], bed["z"],
        y_min=y_edges[0], y_max=y_edges[-1],
    )
    state = target_mac_velocity_state(
        data["mac_streamfunction"], h, h, x_edges, y_edges,
        source_x_origin=0.0, source_z_origin=VERTICAL_ORIGIN,
    )
    u, w = current_u.copy(), current_w.copy()
    liquid = q > 0
    u[liquid], w[liquid] = state["vx"][liquid], state["vy"][liquid]
    k, i = np.nonzero((q > 0) & (q < 1))
    pieces = intersection(
        geometry,
        box(x_edges[i], y_edges[k], x_edges[i + 1], y_edges[k + 1]),
    )
    areas = area(pieces)
    area_error = np.max(np.abs(areas - q[k, i] * h * h))
    if area_error > 1e-10 * h * h:
        raise ValueError(f"native q and reconstructed polygon differ: {area_error}")
    center = centroid(pieces)
    ax = (get_x(center) - x_edges[i]) / h
    az = (get_y(center) - y_edges[k]) / h
    if not np.isfinite(ax + az).all() or np.any((ax < 0) | (ax > 1) | (az < 0) | (az > 1)):
        raise ValueError("invalid liquid centroids")
    psi = state["target_streamfunction"]
    j = (i + 1) % nx
    p00, p10, p01, p11 = psi[k, i], psi[k, j], psi[k + 1, i], psi[k + 1, j]
    mixed = p11 - p10 - p01 + p00
    u[k, i] = (p01 - p00 + mixed * ax) / h
    w[k, i] = -(p10 - p00 + mixed * az) / h
    if not np.isfinite(u + w).all():
        raise ValueError("non-finite baseline velocity")
    np.asarray(u, dtype="<f8").tofile(output / "vx.raw")
    np.asarray(w, dtype="<f8").tofile(output / "vy.raw")
    baseline_momentum = np.array([np.sum(q * u), np.sum(q * w)]) * h * h
    target = np.array(initial["target_liquid_momentum"])
    report = {
        "schema": "native-bilinear-centroid-control-v1",
        "scope": "diagnostic baseline, not qualified production state",
        "source_sha256": sha256(source),
        "input_hashes": {name: sha256(input_dir / name) for name in (
            "initial_state.json", "q.raw", "vx.raw", "vy.raw", "receiver_bed.npz",
        )},
        "code_sha256": sha256(Path(__file__)),
        "nx": nx, "ny": ny, "mixed_cells": len(k),
        "maximum_mixed_area_error": float(area_error),
        "momentum": baseline_momentum.tolist(),
        "target_momentum": target.tolist(),
        "momentum_relative_error": float(np.linalg.norm(baseline_momentum - target) / np.linalg.norm(target)),
        "maximum_liquid_velocity_difference_from_fitted": float(
            np.max(np.hypot(u[liquid] - current_u[liquid], w[liquid] - current_w[liquid]))
        ),
        "maximum_gas_velocity_difference": float(
            np.max(np.hypot(u[~liquid] - current_u[~liquid], w[~liquid] - current_w[~liquid]))
        ),
        "output_hashes": {name: sha256(output / name) for name in ("vx.raw", "vy.raw")},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    control_initial = dict(initial)
    control_initial["qualification_scope"] = "diagnostic-bilinear-centroid-control"
    control_initial["boundary_fitted_moments"] = None
    control_initial["liquid_momentum"] = baseline_momentum.tolist()
    control_initial["relative_momentum_error"] = report["momentum_relative_error"]
    control_initial["control_report_sha256"] = sha256(output / "report.json")
    (output / "initial_state.json").write_text(json.dumps(control_initial, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.input_dir, args.output)
