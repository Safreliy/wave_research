"""Summarize and plot the bounded three-hypothesis stabilization screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VOLUME_GATE = 1.0e-8
MOMENTUM_GATE = 1.0e-8
VELOCITY_CORRECTION_GATE = 5.0e-3
ENERGY_CORRECTION_GATE = 5.0e-3
TRANSFER_GATES = {
    "relative_face_flux_fit_error": 2.0e-2,
    "relative_cell_center_velocity_error": 1.0e-1,
    "relative_surface_velocity_error": 5.0e-2,
    "maximum_liquid_to_surface_q99_speed_ratio": 1.5 * (1.0 + 1.0e-6),
    "relative_momentum_error": 1.0e-8,
    "projected_face_divergence_linf": 1.0e-10,
}


def records(path: Path, label: str) -> list[list[float]]:
    result: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if fields and fields[0] == label:
            result.append([float(value) for value in fields[1:]])
    return result


def amr_run(path: Path) -> dict[str, object]:
    ledger = records(path, "AMR_VOLUME_STAGE")
    runtime = records(path, "RUN")
    initial = records(path, "INIT_RECEIVER_CORRECTION")[0]
    first_iteration = int(ledger[0][1])
    first = {int(row[2]): row[4] for row in ledger if int(row[1]) == first_iteration}
    final_relative_volume_error = abs(runtime[-1][2] - runtime[0][2]) / runtime[0][2]
    adapt_jump = first.get(1, first[0]) - first[0]
    return {
        "diagnostics": str(path),
        "first_adaptation_stage_relative_volume_error": {
            str(stage): value for stage, value in sorted(first.items())
        },
        "first_adaptation_relative_jump": adapt_jump,
        "short_prefix_final_relative_volume_error": final_relative_volume_error,
        "short_prefix_volume_gate_passed": final_relative_volume_error <= VOLUME_GATE,
        "initial_receiver": {
            "relative_momentum_error_before": initial[2],
            "relative_momentum_error_after": initial[3],
            "relative_velocity_correction": initial[6],
            "relative_kinetic_energy_change": initial[7],
        },
    }


def receiver_run(level: int, path: Path) -> dict[str, object]:
    row = records(path, "INIT_RECEIVER_CORRECTION")[0]
    result = {
        "level": level,
        "diagnostics": str(path),
        "relative_momentum_error_before": row[2],
        "relative_momentum_error_after": row[3],
        "shift_x": row[4],
        "shift_z": row[5],
        "relative_velocity_correction": row[6],
        "relative_kinetic_energy_change": row[7],
    }
    result["gates"] = {
        "momentum": result["relative_momentum_error_after"] <= MOMENTUM_GATE,
        "velocity_correction":
            result["relative_velocity_correction"] <= VELOCITY_CORRECTION_GATE,
        "kinetic_energy":
            result["relative_kinetic_energy_change"] <= ENERGY_CORRECTION_GATE,
    }
    result["all_gates_passed"] = all(result["gates"].values())
    return result


def gpu_run(time: float, path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    gates = {key: data[key] <= gate for key, gate in TRANSFER_GATES.items()}
    return {
        "time": time,
        "report": str(path),
        "solve_seconds": data["solve_seconds"],
        "least_squares_iterations": data["least_squares_iterations"],
        "metrics": {key: data[key] for key in TRANSFER_GATES},
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def monotone_decreasing(values: list[float]) -> bool:
    return bool(np.all(np.diff(np.asarray(values)) < 0.0))


def plot(report: dict[str, object], output: Path) -> None:
    colors = {"standard": "#B03A2E", "fixed": "#1F618D", "locked": "#148F77"}
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.5), constrained_layout=True)

    for policy, run in report["hypothesis_1_amr"].items():
        ledger = run["first_adaptation_stage_relative_volume_error"]
        stages = np.asarray(sorted(int(key) for key in ledger))
        values = np.asarray([ledger[str(stage)] for stage in stages])
        axes[0, 0].plot(stages, values, marker="o", lw=1.8,
                        color=colors[policy], label=policy)
    axes[0, 0].axhline(0.0, color="#666666", lw=0.8)
    axes[0, 0].set_xticks([0, 1, 2, 3], ["pre-AMR", "AMR", "cleanup", "projection"])
    axes[0, 0].tick_params(axis="x", rotation=15)
    axes[0, 0].set_ylabel("relative liquid-volume error")
    axes[0, 0].set_title("(a) L9 localization of the volume jump")
    axes[0, 0].legend(frameon=False)

    policies = list(report["hypothesis_1_amr"])
    final_errors = [
        report["hypothesis_1_amr"][policy]["short_prefix_final_relative_volume_error"]
        for policy in policies
    ]
    axes[0, 1].bar(policies, final_errors, color=[colors[p] for p in policies])
    axes[0, 1].axhline(VOLUME_GATE, color="#333333", ls=":", label=r"gate $10^{-8}$")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("final relative volume error")
    axes[0, 1].set_title("(b) Short-prefix conservation")
    axes[0, 1].legend(frameon=False)

    receiver = report["hypothesis_2_receiver_momentum"]
    levels = [run["level"] for run in receiver]
    velocity = [100.0 * run["relative_velocity_correction"] for run in receiver]
    energy = [100.0 * run["relative_kinetic_energy_change"] for run in receiver]
    axes[1, 0].plot(levels, velocity, marker="o", lw=1.8,
                    color="#7D3C98", label="velocity correction")
    axes[1, 0].plot(levels, energy, marker="s", lw=1.8,
                    color="#D68910", label="kinetic-energy change")
    axes[1, 0].axhline(100.0 * VELOCITY_CORRECTION_GATE,
                       color="#333333", ls=":", label="0.5% gate")
    axes[1, 0].set_xticks(levels)
    axes[1, 0].set_xlabel("receiver level")
    axes[1, 0].set_ylabel("relative correction (%)")
    axes[1, 0].set_title("(c) Cost of exact receiver momentum")
    axes[1, 0].legend(frameon=False, fontsize=8)

    gpu = report["hypothesis_3_handoff_time"]
    time_reference = 20.3
    selected = [
        "relative_face_flux_fit_error",
        "relative_cell_center_velocity_error",
        "relative_surface_velocity_error",
        "maximum_liquid_to_surface_q99_speed_ratio",
    ]
    labels = ["face fit", "bulk fit", "surface fit", "speed cap"]
    for key, label, color in zip(
        selected, labels, ["#1F77B4", "#D62728", "#2CA02C", "#9467BD"]
    ):
        axes[1, 1].plot(
            [run["time"] - time_reference for run in gpu],
            [run["metrics"][key] / TRANSFER_GATES[key] for run in gpu],
            marker="o", lw=1.7, color=color, label=label,
        )
    axes[1, 1].axhline(1.0, color="#333333", ls=":", label="accept/reject")
    axes[1, 1].set_xlabel(r"BIE handoff time $t-20.3$")
    axes[1, 1].set_ylabel("metric / fixed gate")
    axes[1, 1].set_title("(d) GPU transfer-time screen")
    axes[1, 1].legend(frameon=False, fontsize=8, ncol=2)

    fig.suptitle(
        "Three stabilization hypotheses: bounded negative/partial results",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_figure", type=Path)
    parser.add_argument("--standard-l9", type=Path, required=True)
    parser.add_argument("--fixed-l9", type=Path, required=True)
    parser.add_argument("--locked-l9", type=Path, required=True)
    parser.add_argument("--receiver", nargs=2, action="append", metavar=("LEVEL", "LOG"),
                        required=True)
    parser.add_argument("--gpu", nargs=2, action="append", metavar=("TIME", "JSON"),
                        required=True)
    args = parser.parse_args()

    amr = {
        "standard": amr_run(args.standard_l9),
        "fixed": amr_run(args.fixed_l9),
        "locked": amr_run(args.locked_l9),
    }
    receiver = [
        receiver_run(int(level), Path(path)) for level, path in args.receiver
    ]
    receiver.sort(key=lambda item: item["level"])
    gpu = [gpu_run(float(time), Path(path)) for time, path in args.gpu]
    gpu.sort(key=lambda item: item["time"])
    fixed_locked_difference = abs(
        amr["fixed"]["short_prefix_final_relative_volume_error"]
        - amr["locked"]["short_prefix_final_relative_volume_error"]
    )
    report: dict[str, object] = {
        "schema": "three-hypothesis-stabilization-screen-v1",
        "evidence_class": "bounded numerical diagnostic; not impact validation",
        "fixed_gates": {
            "relative_volume_error": VOLUME_GATE,
            "relative_receiver_momentum_error": MOMENTUM_GATE,
            "relative_velocity_correction": VELOCITY_CORRECTION_GATE,
            "relative_kinetic_energy_change": ENERGY_CORRECTION_GATE,
            "gpu_transfer": TRANSFER_GATES,
        },
        "hypothesis_1_amr": amr,
        "hypothesis_1_findings": {
            "standard_first_adapt_jump_detected":
                abs(amr["standard"]["first_adaptation_relative_jump"]) > 1.0e-6,
            "fixed_first_adapt_conservative":
                abs(amr["fixed"]["first_adaptation_relative_jump"]) <= 1.0e-14,
            "locked_first_adapt_conservative":
                abs(amr["locked"]["first_adaptation_relative_jump"]) <= 1.0e-14,
            "fixed_locked_final_error_difference": fixed_locked_difference,
            "any_policy_passes_short_prefix_volume_gate":
                any(run["short_prefix_volume_gate_passed"] for run in amr.values()),
        },
        "hypothesis_2_receiver_momentum": receiver,
        "hypothesis_2_findings": {
            "pre_correction_error_decreases_with_level": monotone_decreasing(
                [run["relative_momentum_error_before"] for run in receiver]
            ),
            "velocity_correction_decreases_with_level": monotone_decreasing(
                [run["relative_velocity_correction"] for run in receiver]
            ),
            "any_level_passes_all_gates": any(run["all_gates_passed"] for run in receiver),
        },
        "hypothesis_3_handoff_time": gpu,
        "hypothesis_3_findings": {
            "accepted_times": [run["time"] for run in gpu if run["all_gates_passed"]],
            "any_time_passes_all_transfer_gates": any(
                run["all_gates_passed"] for run in gpu
            ),
            "vof_prefix_started": False,
            "vof_prefix_reason":
                "all source times were rejected by predeclared transfer gates",
        },
        "promotion_decision": "reject all three candidates before impact continuation",
        "next_testable_step":
            "coupled VOF/embedded-boundary conservative advection for q=f*cs, then "
            "a receiver-grid constrained correction that is smaller than the global shift",
    }
    plot(report, args.output_figure)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
