"""Independently integrate harmonic velocities to audit native reference inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(folder: Path) -> dict:
    protocol_path = folder / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    geometry = protocol["geometry"]
    nx, ny = geometry["nx"], geometry["ny"]
    h = geometry["period"] / nx
    bed = geometry["bed_solver_y_effective"]
    surface = geometry["surface_solver_y"]
    k = geometry["wave_number"]
    amplitude = protocol["initial_field"]["potential_amplitude"]
    if protocol["initial_field"]["background_current"] != 0:
        raise ValueError("This audit is for the declared zero-current benchmark")
    points, weights = np.polynomial.legendre.leggauss(8)
    x = (np.arange(nx)[:, None] + 0.5) * h + points[None, :] * h / 2
    edges = np.arange(ny + 1) * h
    lo, hi = np.maximum(edges[:-1], bed), np.minimum(edges[1:], surface)
    wet_rows = hi > lo
    ys = (lo[wet_rows, None] + hi[wet_rows, None]) / 2 + (
        hi[wet_rows, None] - lo[wet_rows, None]) * points[None, :] / 2
    sin_mean = np.sin(k * x) @ weights / 2
    cos_mean = np.cos(k * x) @ weights / 2
    cosh_mean = np.cosh(k * (ys - bed)) @ weights / 2
    sinh_mean = np.sinh(k * (ys - bed)) @ weights / 2
    scale = amplitude * k / np.cosh(k * (surface - bed))
    gauss = np.zeros((2, ny, nx))
    gauss[0, wet_rows] = -scale * cosh_mean[:, None] * sin_mean[None, :]
    gauss[1, wet_rows] = scale * sinh_mean[:, None] * cos_mean[None, :]

    def raw(method: str, name: str) -> np.ndarray:
        value = np.fromfile(folder / method / f"{name}.raw", dtype="<f8").reshape(ny, nx)
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {method}/{name}")
        return value

    exact = np.stack([raw("exact_initial_means", name) for name in ("vx", "vy")])
    q = raw("exact_initial_means", "q")
    expected_q = np.broadcast_to(np.maximum(hi - lo, 0)[:, None] / h, q.shape)
    geometry_error = float(np.max(np.abs(q - expected_q)))
    if geometry_error > 1e-11:
        raise ValueError(f"Geometry mismatch: {geometry_error}")
    wet = q > 0
    qden = float(np.sum(q * np.sum(exact ** 2, axis=0)))
    gauss_error = float(np.sqrt(np.sum(q * np.sum((gauss - exact) ** 2, axis=0)) / qden))
    if gauss_error > 1e-9:
        raise ValueError(f"Independent quadrature mismatch: {gauss_error}")
    metrics = {}
    inputs = {"protocol.json": digest(protocol_path)}
    for method in ("fitted", "bilinear_centroid", "exact_initial_means"):
        field = np.stack([raw(method, name) for name in ("vx", "vy")])
        if not np.array_equal(q, raw(method, "q")):
            raise ValueError("Different liquid geometries between branches")
        gas_error = float(np.max(np.abs((field - exact)[:, ~wet])))
        if gas_error != 0:
            raise ValueError("Different gas fields between branches")
        metrics[method] = float(np.sqrt(np.sum(q * np.sum((field - exact) ** 2, axis=0)) / qden))
        for name in ("q", "vx", "vy"):
            key = f"{method}/{name}.raw"
            inputs[key] = digest(folder / key)
    return {"case": folder.name, "input_hashes": inputs,
            "max_liquid_fraction_geometry_difference": geometry_error,
            "independent_gauss_reference_relative_error": gauss_error,
            "global_liquid_weighted_initial_errors": metrics,
            "gas_fields_identical": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case_names = [f"flat_native_L{level}{suffix}"
                  for level in (9, 10) for suffix in ("", "_trace4")]
    case_names.append("flat_native_L11_resolved")
    cases = [check(args.root / name) for name in case_names]
    report = {"schema": "native-flat-input-independent-audit-v1", "passed": True,
              "scope": "independent eight-point tensor Gaussian phase means; no dynamic accuracy assertion",
              "code_sha256": digest(Path(__file__)), "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases": len(cases),
                      "max_gauss_relative_error": max(c["independent_gauss_reference_relative_error"] for c in cases)}))


if __name__ == "__main__":
    main()
