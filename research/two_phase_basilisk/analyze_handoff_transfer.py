"""Summarize conservative BIE-to-VOF transfer checks as machine-readable JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _records(
    path: Path,
) -> tuple[
    list[dict[str, float | int]],
    list[dict[str, float | int]],
    list[dict[str, float | int]],
]:
    transfers: list[dict[str, float | int]] = []
    projected: list[dict[str, float | int]] = []
    runs: list[dict[str, float | int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] in ("TRANSFER", "PROJECTED") and len(fields) == 11:
            target = transfers if fields[0] == "TRANSFER" else projected
            target.append(
                {
                    "time": float(fields[1]),
                    "iteration": int(fields[2]),
                    "volume": float(fields[3]),
                    "momentum_x": float(fields[4]),
                    "momentum_z": float(fields[5]),
                    "cell_divergence_l2": float(fields[6]),
                    "cell_divergence_linf": float(fields[7]),
                    "face_divergence_l2": float(fields[8]),
                    "face_divergence_linf": float(fields[9]),
                    "leaf_cells": int(fields[10]),
                }
            )
        elif fields[0] == "RUN" and len(fields) == 5:
            runs.append(
                {
                    "time": float(fields[1]),
                    "iteration": int(fields[2]),
                    "volume": float(fields[3]),
                    "kinetic_energy": float(fields[4]),
                }
            )
    if not transfers or (not projected and len(transfers) < 2) or not runs:
        raise ValueError(
            "diagnostics require an initial TRANSFER, a projected record and one RUN"
        )
    return transfers, projected, runs


def analyze(handoff_json: Path, preparation_json: Path, diagnostics: Path) -> dict[str, object]:
    handoff = json.loads(handoff_json.read_text(encoding="utf-8"))
    preparation = json.loads(preparation_json.read_text(encoding="utf-8"))
    transfers, projected_records, runs = _records(diagnostics)
    initial = transfers[0]
    projected = projected_records[0] if projected_records else transfers[1]
    target_volume = float(handoff["source_volume"])
    target_momentum = np.asarray(
        [handoff["target_momentum_x"], handoff["target_momentum_z"]], dtype=float
    )
    received_momentum = np.asarray(
        [initial["momentum_x"], initial["momentum_z"]], dtype=float
    )
    relative_mass_error = (float(initial["volume"]) - target_volume) / target_volume
    relative_momentum_error = float(
        np.linalg.norm(received_momentum - target_momentum)
        / max(float(np.linalg.norm(target_momentum)), np.finfo(float).eps)
    )
    projected_momentum = np.asarray(
        [projected["momentum_x"], projected["momentum_z"]], dtype=float
    )
    relative_projection_momentum_impulse = float(
        np.linalg.norm(projected_momentum - received_momentum)
        / max(float(np.linalg.norm(received_momentum)), np.finfo(float).eps)
    )
    extension_method = str(preparation.get("extension_method", "nearest_gaussian"))
    initialization_gates = {
        "relative_mass_error_le_1e-6": abs(relative_mass_error) <= 1.0e-6,
        "relative_momentum_error_le_1e-9": relative_momentum_error <= 1.0e-9,
    }
    extension_gates: dict[str, bool]
    if extension_method == "nearest_gaussian":
        velocity_change = float(
            preparation["relative_weighted_velocity_smoothing_change"]
        )
        extension_gates = {
            "velocity_smoothing_change_le_0p1": velocity_change <= 0.1,
        }
    else:
        extension_gates = {
            "relative_bulk_velocity_fit_error_le_0p1": (
                float(preparation["relative_bulk_velocity_fit_error"]) <= 0.1
            ),
            "relative_surface_velocity_fit_error_le_0p05": (
                float(preparation["relative_surface_velocity_fit_error"]) <= 0.05
            ),
            "maximum_liquid_to_surface_q99_speed_ratio_le_1p5": (
                float(preparation["maximum_liquid_to_surface_q99_speed_ratio"])
                <= 1.5
            ),
        }
    receiver_gates = {
        "first_projected_face_divergence_l2_le_1e-3": (
            float(projected["face_divergence_l2"]) <= 1.0e-3
        ),
        "relative_projection_momentum_impulse_le_1e-2": (
            relative_projection_momentum_impulse <= 1.0e-2
        ),
        "relative_volume_change_at_last_run_le_1e-3": abs(
            (float(runs[-1]["volume"]) - float(runs[0]["volume"]))
            / float(runs[0]["volume"])
        )
        <= 1.0e-3,
    }
    gates = {**initialization_gates, **extension_gates, **receiver_gates}
    rejection_reasons = [name for name, passed in gates.items() if not passed]
    accepted = all(gates.values())
    return {
        "schema": "bie-to-vof-transfer-audit-v2",
        "evidence_class": (
            "accepted conservative short coupled continuation"
            if accepted
            else "conservative initialization; coupled dynamics rejected"
        ),
        "source_bie_time": handoff["time"],
        "source_snapshot_index": handoff["snapshot_index"],
        "target_volume": target_volume,
        "received_volume": initial["volume"],
        "relative_mass_error": relative_mass_error,
        "target_momentum": target_momentum.tolist(),
        "received_momentum": received_momentum.tolist(),
        "relative_momentum_error": relative_momentum_error,
        "extension_method": extension_method,
        "extension_metrics": {
            key: value
            for key, value in preparation.items()
            if key.startswith("relative_")
            or key.startswith("maximum_")
            or key.endswith("divergence_l2_full_liquid")
        },
        "first_projected_step": projected,
        "relative_projection_momentum_impulse": (
            relative_projection_momentum_impulse
        ),
        "last_run_record": runs[-1],
        "relative_volume_change_at_last_run": (
            float(runs[-1]["volume"]) - float(runs[0]["volume"])
        ) / float(runs[0]["volume"]),
        "acceptance_gates": gates,
        "conservative_initialization_passed": all(initialization_gates.values()),
        "velocity_extension_passed": all(extension_gates.values()),
        "receiver_short_run_passed": all(receiver_gates.values()),
        "coupled_continuation_accepted": accepted,
        "rejection_reasons": rejection_reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("handoff_json", type=Path)
    parser.add_argument("preparation_json", type=Path)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(args.handoff_json, args.preparation_json, args.diagnostics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
