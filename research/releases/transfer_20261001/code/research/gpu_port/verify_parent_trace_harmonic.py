"""Analytic phase-mean audit of the opt-in parent-chord Hermite trace.

The exact reference is a prescribed harmonic potential field on a flat liquid
strip.  Both transfer variants use identical source data and polygonal geometry;
each centroid comparator is the bilinear curl of that variant's own sampled psi.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import shapely

from liquid_dirichlet_transfer import boundary_traces, transfer
from prepare_aphros_initial_state import VERTICAL_ORIGIN, EMBEDDED_BOUNDARY_OFFSET


LENGTH = 64.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(bed: float, surface: float, *, mode: int, amplitude: float,
           ntrace: int) -> dict:
    x = LENGTH * np.arange(ntrace) / ntrace
    k = 2 * np.pi * mode / LENGTH
    depth = surface - bed
    norm = np.cosh(k * depth)
    potential = amplitude * np.cos(k * x)
    normal = amplitude * k * np.tanh(k * depth) * np.cos(k * x)
    return {
        "surface_x": x, "surface_z": np.full_like(x, surface),
        "bottom_x": x, "bottom_z": np.full_like(x, bed),
        "surface_potential": potential,
        "bottom_potential": potential / norm,
        "surface_normal_derivative": normal,
        "bottom_normal_derivative": np.zeros_like(x),
        "surface_velocity_x": -amplitude * k * np.sin(k * x),
        "surface_velocity_z": normal,
        "source_volume": LENGTH * depth,
        "background_current": 0.0,
    }


def exact_phase_means(x: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                      bed: float, surface: float, mode: int,
                      amplitude: float, h: float) -> np.ndarray:
    k = 2 * np.pi * mode / LENGTH
    mid = 0.5 * (lo + hi)
    dz = hi - lo
    vertical_sinc = np.sinh(0.5 * k * dz) / (0.5 * k * dz)
    horizontal_sinc = np.sinc(k * h / (2 * np.pi))
    coefficient = amplitude * k * horizontal_sinc / np.cosh(k * (surface - bed))
    u = -coefficient * np.cosh(k * (mid - bed))[:, None] * vertical_sinc[:, None] * np.sin(k * x)[None, :]
    w = coefficient * np.sinh(k * (mid - bed))[:, None] * vertical_sinc[:, None] * np.cos(k * x)[None, :]
    return np.array([u, w])


def centroid_curl(psi: np.ndarray, zedge: np.ndarray, lo: np.ndarray,
                  hi: np.ndarray, h: float) -> np.ndarray:
    az = (0.5 * (lo + hi) - zedge[:-1]) / h
    p00, p01 = psi[:-1], psi[1:]
    p10, p11 = np.roll(p00, -1, axis=1), np.roll(p01, -1, axis=1)
    mixed = p11 - p10 - p01 + p00
    return np.array([((p01 - p00) + 0.5 * mixed) / h,
                     -((p10 - p00) + mixed * az[:, None]) / h])


def relative_l2(found: np.ndarray, exact: np.ndarray, q: np.ndarray,
                mask: np.ndarray, weighted: bool) -> float:
    weight = np.where(mask, q if weighted else 1.0, 0.0)
    error = np.sum(weight * np.sum((found - exact) ** 2, axis=0))
    reference = np.sum(weight * np.sum(exact ** 2, axis=0))
    return float(np.sqrt(error / reference))


def specs(suite: str):
    manufactured = [("manufactured", n, q) for n in (128, 256)
                    for q in (1e-5, 0.001, 0.01, 0.1)]
    native = [("native", 512, q) for q in (0.0005, 0.001, 0.01, 0.1)]
    native += [("native", 1024, q) for q in (0.001, 0.01, 0.1)]
    return ([manufactured[1], native[0]] if suite == "pilot"
            else manufactured + native)


def run_case(family: str, nx: int, fraction: float, root: Path) -> dict:
    h = LENGTH / nx
    xedge = np.arange(nx) * h
    if family == "manufactured":
        bed = -4.0
        zedge = np.arange(-4.0 - 0.37 * h, 4.0 + h, h)
        surface = (-0.37 + fraction) * h
        mode, ntrace, factor = 4, 256, nx // 16
        k = 2 * np.pi * mode / LENGTH
        amplitude = np.cosh(k * (surface - bed)) / np.cosh(k * 4.0)
    else:
        bed = VERTICAL_ORIGIN + 0.53 + EMBEDDED_BOUNDARY_OFFSET
        zedge = VERTICAL_ORIGIN + np.arange(nx // 16 + 1) * h
        # The native free surface is within the fixed row containing 1.5000625.
        native_surface = VERTICAL_ORIGIN + 1.5000625
        top_row = int(np.floor((native_surface - VERTICAL_ORIGIN) / h))
        surface = zedge[top_row] + fraction * h
        mode, ntrace, factor, amplitude = 16, 512, 8, 0.02
    case = f"{family}_nx{nx}_q{fraction:g}"
    src = source(bed, surface, mode=mode, amplitude=amplitude, ntrace=ntrace)
    curves, source_report = boundary_traces(src, factor=factor)
    results = {}
    reports = {}
    timing = {}
    for label, option in (("linear", "linear"),
                          ("parent_hermite", "parent_hermite"),
                          ("parent_hermite_matched", "parent_hermite_matched")):
        started = time.perf_counter()
        data, report = transfer(
            curves, xedge, zedge, backend="numpy", gas=False,
            continuation_order=2, cutcell_moments=True,
            physical_trace=option,
        )
        timing[label] = time.perf_counter() - started
        results[label] = data
        reports[label] = report
    q = results["linear"]["volume_fraction"]
    if any(not np.array_equal(q, data["volume_fraction"])
           for data in results.values()):
        raise ValueError("physical trace option changed liquid geometry")
    actual_surface = float(results["linear"]["surface_z"][0])
    actual_bed = float(results["linear"]["bottom_z"][0])
    lo = np.maximum(zedge[:-1], actual_bed)
    hi = np.minimum(zedge[1:], actual_surface)
    wet_rows = hi > lo
    exact = np.zeros((2, len(lo), nx))
    exact[:, wet_rows] = exact_phase_means(
        results["linear"]["grid_x"], lo[wet_rows], hi[wet_rows],
        actual_bed, actual_surface, mode, amplitude, h,
    )
    expected_q = np.broadcast_to(np.maximum(hi - lo, 0)[:, None] / h, q.shape)
    if np.max(np.abs(q - expected_q)) > 1e-10:
        raise ValueError("analytic and receiver liquid fractions differ")
    top_mask = np.zeros_like(q, dtype=bool)
    top_mask[np.flatnonzero(wet_rows)[-1]] = q[np.flatnonzero(wet_rows)[-1]] > 0
    masks = {"all": q > 0, "cut": (q > 0) & (q < 1), "top": top_mask}
    arrays = {"q": q, "exact_phase_mean": exact}
    metrics = {}
    for label, data in results.items():
        fitted = np.array([data["cell_velocity_x_projected"],
                           data["cell_velocity_z_projected"]])
        centroid = centroid_curl(data["mac_streamfunction"], zedge, lo, hi, h)
        arrays[label + "_phase_mean"] = fitted
        arrays[label + "_same_psi_centroid"] = centroid
        metrics[label] = {}
        for method, field in (("fitted", fitted), ("same_psi_centroid", centroid)):
            metrics[label][method] = {
                tag + suffix: relative_l2(field, exact, q, mask, weighted)
                for tag, mask in masks.items()
                for suffix, weighted in (("_relative_l2", False),
                                         ("_q_weighted_relative_l2", True))
            }
            metrics[label][method]["top_max_absolute_error"] = float(
                np.max(np.abs((field - exact)[:, top_mask]))
            )
    array_path = root / (case + ".npz")
    np.savez_compressed(array_path, **arrays)
    return {
        "case": case, "family": family, "nx": nx,
        "surface_liquid_fraction_requested": fraction,
        "surface_liquid_fraction_actual": float(q[top_mask].mean()),
        "mode": mode, "amplitude": amplitude, "source_nodes": ntrace,
        "trace_factor": factor, "surface": actual_surface, "bed": actual_bed,
        "metrics": metrics, "timing_seconds": timing,
        "linear_momentum_closure": reports["linear"]["cutcell_moments"]["trace_closure_absolute"],
        "hermite_momentum_closure": reports["parent_hermite"]["cutcell_moments"]["trace_closure_absolute"],
        "matched_momentum_closure": reports["parent_hermite_matched"]["cutcell_moments"]["trace_closure_absolute"],
        "hermite_trace": reports["parent_hermite"]["physical_trace_report"],
        "matched_edge_report": reports["parent_hermite_matched"]["cutcell_moments"],
        "source_trace": source_report,
        "arrays": array_path.name, "arrays_sha256": digest(array_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--suite", choices=("pilot", "declared"), default="pilot")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    rows = []
    for family, nx, q in specs(args.suite):
        row = run_case(family, nx, q, args.root)
        rows.append(row)
        print(json.dumps({"case": row["case"],
                          "linear_top": row["metrics"]["linear"]["fitted"]["top_relative_l2"],
                          "hermite_top": row["metrics"]["parent_hermite"]["fitted"]["top_relative_l2"],
                          "matched_top": row["metrics"]["parent_hermite_matched"]["fitted"]["top_relative_l2"],
                          "timing_seconds": row["timing_seconds"]}), flush=True)
        (args.root / "result.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    files = ("physical_trace.py", "liquid_dirichlet_transfer.py",
             "cutcell_transfer.py", "verify_parent_trace_harmonic.py",
             "test_physical_trace.py")
    report = {
        "schema": "parent-chord-hermite-harmonic-audit-v2",
        "suite": args.suite, "rows": rows,
        "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in files},
        "protocol_sha256": digest(Path(__file__).parents[1] / "results/method_trace_hermite_20261001/PREDECLARED_PROTOCOL.md"),
        "matched_amendment_sha256": digest(Path(__file__).parents[1] / "results/method_trace_hermite_20261001/MATCHED_EDGE_AMENDMENT.md"),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "shapely": shapely.__version__},
    }
    (args.root / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
