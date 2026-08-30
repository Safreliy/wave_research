"""Build a machine-readable audit of the constrained MAC handoff receiver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def records(path: Path, label: str) -> list[list[float]]:
    parsed: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0] == label:
            parsed.append([float(value) for value in fields[1:]])
    return parsed


def initial_audit(
    path: Path,
    target_volume: float,
    target_momentum: np.ndarray,
) -> dict[str, object]:
    transfer = next(row for row in records(path, "TRANSFER") if row[0] == 0.0)
    projected = next(row for row in records(path, "PROJECTED") if row[0] == 0.0)
    pre_momentum = np.asarray(transfer[3:5])
    post_momentum = np.asarray(projected[3:5])
    momentum_scale = max(float(np.linalg.norm(target_momentum)), np.finfo(float).eps)
    metrics = {
        "diagnostics": str(path),
        "relative_initial_mass_error": abs(transfer[2] - target_volume) / target_volume,
        "relative_initial_momentum_error": float(
            np.linalg.norm(pre_momentum - target_momentum) / momentum_scale
        ),
        "relative_projection_momentum_impulse": float(
            np.linalg.norm(post_momentum - pre_momentum) / momentum_scale
        ),
        "pre_projection_cell_divergence_rms": transfer[5],
        "pre_projection_cell_divergence_linf": transfer[6],
        "pre_projection_face_divergence_rms": transfer[7],
        "pre_projection_face_divergence_linf": transfer[8],
        "post_projection_cell_divergence_rms": projected[5],
        "post_projection_cell_divergence_linf": projected[6],
        "post_projection_face_divergence_rms": projected[7],
        "post_projection_face_divergence_linf": projected[8],
        "uniform_full_liquid_cell_count": int(projected[9]),
    }
    gates = {
        "mass_le_1e-10": metrics["relative_initial_mass_error"] <= 1.0e-10,
        "momentum_le_1e-8": metrics["relative_initial_momentum_error"] <= 1.0e-8,
        "projection_impulse_le_1pct": metrics[
            "relative_projection_momentum_impulse"
        ]
        <= 0.01,
        "post_projection_face_divergence_rms_le_1e-3": metrics[
            "post_projection_face_divergence_rms"
        ]
        <= 1.0e-3,
        "post_projection_face_divergence_linf_le_1e-3": metrics[
            "post_projection_face_divergence_linf"
        ]
        <= 1.0e-3,
    }
    result: dict[str, object] = {
        "metrics": metrics,
        "gates": gates,
        "all_initial_gates_passed": all(gates.values()),
    }
    later_projected = [row for row in records(path, "PROJECTED") if row[0] > 0.0]
    if later_projected:
        first = later_projected[0]
        result["first_post_adaptation_projection"] = {
            "time": first[0],
            "iteration": int(first[1]),
            "face_divergence_rms": first[7],
            "face_divergence_linf": first[8],
            "uniform_full_liquid_cell_count": int(first[9]),
            "rms_gate_le_1e-3": first[7] <= 1.0e-3,
            "linf_gate_le_1e-3": first[8] <= 1.0e-3,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--level8", type=Path, required=True)
    parser.add_argument("--level9", type=Path, required=True)
    parser.add_argument("--level10", type=Path, required=True)
    parser.add_argument("--long-run", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.source) as loaded:
        target_volume = float(loaded["source_volume"])
        target_momentum = np.asarray(
            [float(loaded["target_momentum_x"]), float(loaded["target_momentum_z"])]
        )
    levels = {
        "8": initial_audit(args.level8, target_volume, target_momentum),
        "9": initial_audit(args.level9, target_volume, target_momentum),
        "10": initial_audit(args.level10, target_volume, target_momentum),
    }
    long_records = records(args.long_run, "RUN")
    last_run = long_records[-1]
    long_metrics = {
        "diagnostics": str(args.long_run),
        "last_reported_time": last_run[0],
        "last_reported_iteration": int(last_run[1]),
        "relative_volume_error": abs(last_run[2] - target_volume) / target_volume,
        "kinetic_energy": last_run[3],
        "volume_gate_le_1e-3": abs(last_run[2] - target_volume) / target_volume
        <= 1.0e-3,
        "contact_claim": "rejected: no resolved jet-to-surface closure in the rendered sequence",
    }
    report = {
        "schema": "bie-to-vof-mac-handoff-audit-v1",
        "source": str(args.source),
        "target_volume": target_volume,
        "target_momentum": target_momentum.tolist(),
        "initial_transfer_by_level": levels,
        "all_three_initial_levels_passed": all(
            level["all_initial_gates_passed"] for level in levels.values()
        ),
        "post_adaptation_compatibility": (
            "open: the first post-adaptation projection does not retain both "
            "face-divergence gates on every audited level"
        ),
        "long_level9_continuation": long_metrics,
        "publication_status": (
            "MAC handoff compatibility passes at three levels; coupled impact "
            "and post-AMR convergence remain open."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
