"""Summarize the refinement/integrator path toward plunging-jet contact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from topographic_wave_solver import reference_energy


ROOT = Path(__file__).resolve().parent


def summarize(path: Path, baseline: dict[str, np.ndarray]) -> dict[str, object]:
    with np.load(path) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    reference = reference_energy(
        baseline["bottom_x"],
        baseline["bottom_z"],
        float(baseline["length"]),
        float(baseline["gravity"]),
        float(baseline.get("background_current", np.asarray(0.0))),
    )
    scale = max(
        abs(float(baseline["energy"][0]) - reference), np.finfo(float).eps
    )
    energy_error = np.abs(data["energy"] - baseline["energy"][0]) / scale
    flux = np.abs(data["surface_flux_defect"])
    valid = (
        (flux <= 1.0e-3)
        & (energy_error <= 2.0e-3)
        & (data["bie_residual"] <= 1.0e-10)
        & (data["marker_cv"] <= 0.1)
    )
    valid_indices = np.where(valid)[0]
    finite_gap = np.isfinite(data.get(
        "normalized_impact_distance", np.full(len(data["time"]), np.inf)
    ))
    valid_gap_indices = np.where(valid & finite_gap)[0]
    latest = int(valid_indices[-1]) if len(valid_indices) else None
    closest = (
        int(valid_gap_indices[
            np.argmin(data["normalized_impact_distance"][valid_gap_indices])
        ])
        if len(valid_gap_indices)
        else None
    )
    return {
        "source": str(path.relative_to(ROOT)),
        "n": int(data["x"].shape[1]),
        "time_integrator": str(data.get("time_integrator", np.asarray("rk4"))),
        "reparameterize_every": int(data.get("reparameterize_every", np.asarray(0))),
        "termination_reason": str(data["termination_reason"]),
        "final_time_computed": float(data["time"][-1]),
        "latest_gate_passing_time": None if latest is None else float(data["time"][latest]),
        "closest_gate_passing_gap_over_ds": (
            None if closest is None else float(data["normalized_impact_distance"][closest])
        ),
        "closest_gate_passing_time": None if closest is None else float(data["time"][closest]),
        "latest_gate_passing_flux_defect": None if latest is None else float(flux[latest]),
        "latest_gate_passing_relative_energy_error": (
            None if latest is None else float(energy_error[latest])
        ),
        "maximum_saved_newton_iterations": int(
            np.max(data.get("nonlinear_iterations", np.asarray([0])))
        ),
        "total_saved_rhs_evaluations": int(
            np.sum(data.get("rhs_evaluations", np.asarray([0])))
        ),
        "contact_threshold_reached_with_gates": bool(
            closest is not None
            and data["normalized_impact_distance"][closest]
            <= float(data.get("impact_distance_factor", np.asarray(0.15)))
        ),
    }


def main() -> None:
    baseline_path = ROOT / "results" / "reef_exact_a06_n96_visual_jet.npz"
    with np.load(baseline_path) as loaded:
        baseline = {key: loaded[key] for key in loaded.files}
    paths = [
        ROOT / "results" / name
        for name in (
            "reef_exact_a06_n192_to_impact.npz",
            "reef_exact_a06_n384_to_impact.npz",
            "reef_exact_a06_n384_implicit_to_impact.npz",
            "reef_exact_a06_n384_remesh_to_impact_long.npz",
        )
    ]
    report = {
        "contact_definition": "minimum nonlocal panel distance / median panel length <= 0.15",
        "conservation_gates": {
            "surface_flux_defect": 1.0e-3,
            "relative_wave_energy_error": 2.0e-3,
            "bie_residual": 1.0e-10,
            "marker_spacing_cv": 0.1,
        },
        "cases": [summarize(path, baseline) for path in paths],
        "conclusion": (
            "Overturning is gate-passing, but no refinement/integrator branch "
            "reaches the first-contact threshold before the spatial gate fails."
        ),
    }
    output = ROOT / "results" / "publication_package" / "impact_resolution_ablation.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
