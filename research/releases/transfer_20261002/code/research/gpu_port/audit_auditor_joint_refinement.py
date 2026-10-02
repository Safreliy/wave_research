"""Recompute three-level initial errors directly from serialized native fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def field(folder: Path, component: str, shape: tuple[int, int]) -> np.ndarray:
    values = np.fromfile(folder / f"{component}.raw", dtype="<f8")
    if values.size != shape[0] * shape[1] or not np.isfinite(values).all():
        raise ValueError((folder, component))
    return values.reshape(shape)


def norms(candidate: np.ndarray, reference: np.ndarray, q: np.ndarray,
          mask: np.ndarray) -> dict:
    delta = np.sum((candidate-reference)**2, axis=0)
    base = np.sum(reference**2, axis=0)
    weighted = math.sqrt(float(np.sum(q[mask] * delta[mask]) /
                               np.sum(q[mask] * base[mask])))
    unweighted = math.sqrt(float(np.sum(delta[mask]) / np.sum(base[mask])))
    return {"liquid_volume_weighted_relative_l2": weighted,
            "cell_unweighted_relative_l2": unweighted,
            "max_absolute_velocity_vector_error": float(np.sqrt(delta[mask]).max())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    rows = []
    for nx in (512, 1024, 2048):
        folder = root / f"joint_L{nx.bit_length()-1}"
        shape = (nx//16, nx)
        protocol = json.loads((folder / "protocol.json").read_text())
        report = json.loads((folder / "preparation_report.json").read_text())
        if protocol["initial_field"]["boundary_trace_refinement_factor"] != nx//64:
            raise ValueError("joint trace policy changed")
        q = field(folder / "exact_initial_means", "q", shape)
        exact = np.stack([field(folder / "exact_initial_means", name, shape)
                          for name in ("vx", "vy")])
        cut = (q > 0) & (q < 1)
        wet = q > 0
        row = {"grid": f"{nx}x{nx//16}", "nx": nx,
               "trace_factor": nx//64, "trace_points": 512*(nx//64),
               "trace_spacing_over_h": 0.125,
               "cut_cell_count": int(cut.sum()), "wet_cell_count": int(wet.sum()),
               "protocol_sha256": sha(folder / "protocol.json"),
               "preparation_report_sha256": sha(folder / "preparation_report.json"),
               "methods": {}}
        for name, subdir in (("matched", "fitted"), ("bilinear", "bilinear_centroid")):
            velocity = np.stack([field(folder / subdir, component, shape)
                                 for component in ("vx", "vy")])
            result = {"all_wet": norms(velocity, exact, q, wet),
                      "cut_cells": norms(velocity, exact, q, cut)}
            old = report["branches"][subdir]["initial_global_liquid_velocity_error_against_exact_harmonic"]
            if abs(result["all_wet"]["liquid_volume_weighted_relative_l2"]-old) > 1e-12:
                raise ValueError("independent/global preparation norm mismatch")
            row["methods"][name] = result
        row["global_gain_bilinear_over_matched"] = (
            row["methods"]["bilinear"]["all_wet"]["liquid_volume_weighted_relative_l2"] /
            row["methods"]["matched"]["all_wet"]["liquid_volume_weighted_relative_l2"])
        rows.append(row)
    orders = {}
    for method in ("matched", "bilinear"):
        errors = [row["methods"][method]["all_wet"]["liquid_volume_weighted_relative_l2"]
                  for row in rows]
        orders[method] = [math.log(errors[i]/errors[i+1], 2) for i in (0, 1)]
    output = {"schema": "auditor-fixed-ratio-three-level-initial-v1",
              "family": "W16 flat harmonic wave initial data",
              "reference": "closed-form exact liquid-cell means of prescribed harmonic field",
              "claim_scope": "initial transfer consistency at fixed refined trace/grid ratio; no native time evolution or unconditional convergence theorem",
              "rows": rows, "observed_two_level_orders": orders,
              "analysis_code_sha256": sha(Path(__file__))}
    path = root / "joint_refinement_initial_audit.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"rows": [{"grid": r["grid"], "matched": r["methods"]["matched"]["all_wet"]["liquid_volume_weighted_relative_l2"],
                                "bilinear": r["methods"]["bilinear"]["all_wet"]["liquid_volume_weighted_relative_l2"]} for r in rows],
                      "orders": orders}, indent=2))


if __name__ == "__main__":
    main()
