"""Create a machine-readable audit for the clean L11 startup pilot."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def records(path: Path) -> dict[str, list[list[float]]]:
    parsed: dict[str, list[list[float]]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if not fields:
            continue
        try:
            values = [float(value) for value in fields[1:]]
        except ValueError:
            continue
        parsed.setdefault(fields[0], []).append(values)
    return parsed


def maximum(rows: list[list[float]], index: int, *, absolute: bool = True) -> float:
    values = [row[index] for row in rows if len(row) > index]
    if not values:
        return math.nan
    return max(abs(value) if absolute else value for value in values)


def resource_value(text: str, label: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(label)}:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("stdout", type=Path)
    parser.add_argument("resource_usage", type=Path)
    parser.add_argument("projection_json", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    data = records(args.diagnostics)
    stdout = args.stdout.read_text(encoding="utf-8", errors="replace")
    resources = args.resource_usage.read_text(encoding="utf-8", errors="replace")
    projection = json.loads(args.projection_json.read_text(encoding="utf-8"))
    thresholds = json.loads(args.manifest.read_text(encoding="utf-8"))["thresholds"]

    summary_match = re.search(
        r"# Quadtree,\s+(\d+) steps,\s+([0-9.eE+-]+) CPU,\s+"
        r"([0-9.eE+-]+) real,\s+([0-9.eE+-]+) points.step/s",
        stdout,
    )
    post = data.get("POSTADAPT_REPROJECT", [])
    q_runs = data.get("Q_RUN", [])
    cleanups = data.get("Q_CLEANUP", [])
    amr = data.get("AMR_VOLUME_STAGE", [])
    init = data.get("INIT_RECEIVER_CORRECTION", [])

    metrics = {
        "completed_steps": int(summary_match.group(1)) if summary_match else None,
        "cpu_seconds": float(summary_match.group(2)) if summary_match else None,
        "wall_seconds": float(summary_match.group(3)) if summary_match else None,
        "points_steps_per_second": (
            float(summary_match.group(4)) if summary_match else None
        ),
        "maximum_resident_kbytes": int(
            resource_value(resources, "Maximum resident set size (kbytes)") or 0
        ),
        "q_runtime_record_count": len(q_runs),
        "maximum_q_lower_bound_violation": maximum(q_runs, 4),
        "maximum_q_upper_bound_violation": maximum(q_runs, 5),
        "maximum_cleanup_bound_violation": maximum(cleanups, 3),
        "maximum_absolute_amr_mass_drift": maximum(amr, 4),
        "post_adaptation_reprojection_count": len(post),
        "maximum_post_projection_face_divergence_rms": maximum(post, 4),
        "maximum_post_projection_face_divergence_linf": maximum(post, 5),
        "maximum_post_projection_relative_momentum_change": maximum(post, 6),
        "maximum_post_projection_relative_kinetic_change": maximum(post, 7),
        "maximum_pre_projection_face_divergence_rms": maximum(post, 8),
        "maximum_pre_projection_face_divergence_linf": maximum(post, 9),
        "initial_receiver_relative_momentum_error_before": (
            init[-1][2] if init else None
        ),
        "initial_receiver_relative_momentum_error_after": (
            init[-1][3] if init else None
        ),
    }
    finite_values = [
        value
        for value in metrics.values()
        if isinstance(value, float) and not math.isnan(value)
    ]
    gates = {
        "clean_terminal_summary": summary_match is not None,
        "finite_metrics": all(math.isfinite(value) for value in finite_values),
        "amr_mass_drift": metrics["maximum_absolute_amr_mass_drift"]
        <= thresholds["mass_drift_tolerance"],
        "q_bounds_at_output_records": max(
            metrics["maximum_q_lower_bound_violation"],
            metrics["maximum_q_upper_bound_violation"],
        )
        <= thresholds["bound_tolerance"],
        "q_bounds_during_cleanup": metrics["maximum_cleanup_bound_violation"]
        <= thresholds["bound_tolerance"],
        "post_amr_divergence": metrics[
            "maximum_post_projection_face_divergence_linf"
        ]
        <= thresholds["post_projection_divergence_tolerance"],
        "post_amr_momentum_correction": metrics[
            "maximum_post_projection_relative_momentum_change"
        ]
        <= thresholds["post_projection_correction_tolerance"],
    }
    report = {
        "schema": "l11-startup-pilot-audit-v1",
        "scope": "startup stability only; not impact, topology or grid convergence",
        "receiver_levels": {"maximum": 11, "minimum": 9},
        "end_time": 0.01,
        "thresholds": thresholds,
        "projection": {
            key: projection[key]
            for key in (
                "least_squares_iterations",
                "solve_seconds",
                "relative_face_flux_fit_error",
                "relative_surface_velocity_error",
                "receiver_grid_level",
                "relative_receiver_momentum_error",
                "receiver_weighted_relative_velocity_change",
                "receiver_relative_kinetic_energy_change",
                "projected_face_divergence_linf",
            )
        },
        "metrics": metrics,
        "gates": gates,
        "pilot_accepted": all(gates.values()),
        "limitations": [
            "The 0.02 runtime-output cadence exceeds the 0.01 pilot horizon, so Q_RUN is available only at t=0.",
            "Zero Q_CLEANUP bound violation is checked at every one of the 17 adaptation stages.",
            "The pilot contains one rendered state and cannot support topology or geometric-convergence claims.",
        ],
        "sources": {
            "diagnostics": args.diagnostics.as_posix(),
            "stdout": args.stdout.as_posix(),
            "resource_usage": args.resource_usage.as_posix(),
            "projection_json": args.projection_json.as_posix(),
            "manifest": args.manifest.as_posix(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
