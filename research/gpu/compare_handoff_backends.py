"""Compare two prepared BIE-to-VOF states without trusting summary metrics alone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROMOTION_GATES = {
    "relative_mass_error": 1.0e-10,
    "relative_momentum_error": 1.0e-8,
    "pre_projection_divergence_linf_full_liquid": 1.0e-10,
    "relative_bulk_velocity_fit_error": 0.10,
    "relative_surface_velocity_fit_error": 0.05,
    "maximum_liquid_to_surface_q99_speed_ratio": 1.5 * (1.0 + 1.0e-6),
}


def relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(reference)), np.finfo(float).eps)
    return float(np.linalg.norm(candidate - reference) / scale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_json", type=Path)
    parser.add_argument("reference_npz", type=Path)
    parser.add_argument("candidate_json", type=Path)
    parser.add_argument("candidate_npz", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    reference_metrics = json.loads(args.reference_json.read_text(encoding="utf-8"))
    candidate_metrics = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    with np.load(args.reference_npz) as loaded:
        reference = {key: loaded[key] for key in loaded.files}
    with np.load(args.candidate_npz) as loaded:
        candidate = {key: loaded[key] for key in loaded.files}

    array_comparison = {}
    for key in (
        "velocity_x_prepared",
        "velocity_z_prepared",
        "streamfunction_prepared",
        "volume_fraction_prepared",
    ):
        array_comparison[key] = {
            "relative_l2": relative_l2(reference[key], candidate[key]),
            "linf": float(np.max(np.abs(candidate[key] - reference[key]))),
        }
    reference_velocity = np.concatenate(
        (
            reference["velocity_x_prepared"].ravel(),
            reference["velocity_z_prepared"].ravel(),
        )
    )
    candidate_velocity = np.concatenate(
        (
            candidate["velocity_x_prepared"].ravel(),
            candidate["velocity_z_prepared"].ravel(),
        )
    )
    velocity_disagreement = relative_l2(reference_velocity, candidate_velocity)

    gate_results = {}
    for key, maximum in PROMOTION_GATES.items():
        value = abs(float(candidate_metrics[key]))
        gate_results[key] = {"value": value, "maximum": maximum, "passed": value <= maximum}
    speed_violations = int(candidate_metrics.get("final_speed_cap_violation_count", 0))
    gate_results["final_speed_cap_violation_count"] = {
        "value": speed_violations,
        "maximum": 0,
        "passed": speed_violations == 0,
    }
    gate_results["backend_velocity_relative_l2"] = {
        "value": velocity_disagreement,
        "maximum": 1.0e-4,
        "passed": velocity_disagreement <= 1.0e-4,
    }

    report = {
        "schema": "bie-to-vof-backend-comparison-v1",
        "reference_backend": reference_metrics.get("linear_solver_backend", "scipy_lsqr"),
        "candidate_backend": candidate_metrics.get("linear_solver_backend", "unknown"),
        "arrays": array_comparison,
        "combined_velocity_relative_l2": velocity_disagreement,
        "gates": gate_results,
        "all_gates_passed": all(item["passed"] for item in gate_results.values()),
    }
    rendered = json.dumps(report, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
