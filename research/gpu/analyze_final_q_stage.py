"""Summarise the final conservative-q/receiver experiments.

The parser intentionally reads the solver logs rather than duplicating the
reported numbers.  All volume measures use the physical convention
``sum(q * Delta**2) == sum(f * dv())`` for Basilisk embedded cells.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def records(path: Path, tag: str) -> list[list[float]]:
    prefix = tag + " "
    result: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            result.append([float(value) for value in line[len(prefix) :].split()])
    return result


def final_record(path: Path) -> dict[str, float | int]:
    row = records(path, "Q_FINAL")[-1]
    return {
        "level": int(row[0]),
        "uses_amr": int(row[1]),
        "mass": row[2],
        "exact_mass": row[3],
        "relative_exact_mass_difference": row[4],
        "relative_l1_error": row[5],
    }


def maximum_mass_drift(path: Path) -> float:
    masses = [row[2] for row in records(path, "Q_MANUFACTURED")]
    if not masses:
        final = final_record(path)
        return abs(float(final["relative_exact_mass_difference"]))
    reference = masses[0]
    return max(abs(value - reference) for value in masses) / reference


def q_run_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = records(path, "Q_RUN")
    return (
        np.asarray([row[0] for row in rows]),
        np.asarray([row[3] for row in rows]),
    )


def build_summary(root: Path) -> dict[str, object]:
    validation = root / "results/q_transport/final_validation"
    fixed = [final_record(validation / f"fixed_l{level}.log") for level in (6, 7, 8)]
    for item, level in zip(fixed, (6, 7, 8)):
        item["maximum_relative_mass_drift"] = maximum_mass_drift(
            validation / f"fixed_l{level}.log"
        )
    errors = [float(item["relative_l1_error"]) for item in fixed]
    orders = [math.log(errors[index] / errors[index + 1], 2.0) for index in (0, 1)]

    frozen_path = validation / "frozen_l7.log"
    dynamic_path = validation / "dynamic_l7_t05.log"
    frozen = final_record(frozen_path)
    dynamic = final_record(dynamic_path)
    frozen["maximum_relative_mass_drift"] = maximum_mass_drift(frozen_path)
    dynamic["maximum_relative_mass_drift"] = maximum_mass_drift(dynamic_path)
    amr_rows = records(dynamic_path, "Q_AMR")
    dynamic["maximum_relative_amr_mass_jump"] = max(
        max(abs(row[3] - row[2]), abs(row[4] - row[3])) / row[2]
        for row in amr_rows
    )

    init_path = root / "two_phase_basilisk/q_remap_receiver_l9_init_v2/diagnostics.dat"
    remap = records(init_path, "Q_CONSERVATIVE_REMAP")[-1]
    repair = records(init_path, "Q_INITIAL_MASS_REPAIR")[-1]
    gpu = json.loads(
        (root / "gpu/receiver_conservative_q.json").read_text(encoding="utf-8")
    )

    coupled: dict[str, object] = {}
    for name, relative in (
        ("frozen_amr", "two_phase_basilisk/q_conservative_receiver_l9_t010/diagnostics.dat"),
        ("dynamic_amr", "two_phase_basilisk/q_conservative_receiver_l9_dynamic_t010/diagnostics.dat"),
    ):
        path = root / relative
        q_rows = records(path, "Q_RUN")
        projection_rows = records(path, "POSTADAPT_REPROJECT")
        coupled[name] = {
            "end_time": q_rows[-1][0],
            "final_relative_mass_drift": q_rows[-1][3],
            "maximum_relative_mass_drift": max(abs(row[3]) for row in q_rows),
            "maximum_q_bound_violation": max(
                max(abs(row[4]), abs(row[5])) for row in q_rows
            ),
            "maximum_post_amr_momentum_change": max(row[6] for row in projection_rows),
            "maximum_post_amr_energy_change": max(row[7] for row in projection_rows),
            "maximum_post_projection_face_divergence": max(
                row[5] for row in projection_rows
            ),
        }

    momentum_path = root / "two_phase_basilisk/q_conservative_momentum_l9_t002/diagnostics.dat"
    momentum_text = momentum_path.read_text(encoding="utf-8")
    momentum_transfer = records(momentum_path, "TRANSFER")[-1]
    coupled_momentum = {
        "status": "rejected",
        "last_time": momentum_transfer[0],
        "floating_point_exception": "Floating point exception" in momentum_text,
        "velocity_convergence_warning": "convergence for u.x not reached" in momentum_text,
        "pressure_convergence_warning": "convergence for p not reached" in momentum_text,
    }

    target = remap[2]
    return {
        "schema": "final-conservative-q-stage-v2",
        "measure_convention": "sum(q*Delta^2) = sum(f*dv); dv already includes cs",
        "manufactured_translation": {
            "fixed_grid": fixed,
            "observed_l1_orders": orders,
            "frozen_amr": frozen,
            "dynamic_amr": dynamic,
        },
        "receiver_handoff": {
            "source_volume": target,
            "conservative_remap_volume": remap[0],
            "relative_conservative_remap_error": (remap[0] - target) / target,
            "volume_after_bed_clip": remap[1],
            "relative_bed_reconciliation": repair[1],
            "maximum_q_minus_cs_before_clip": remap[4],
            "gpu_solve_seconds": gpu["solve_seconds"],
            "gpu_iterations": gpu["least_squares_iterations"],
            "relative_receiver_momentum_error": gpu[
                "relative_receiver_momentum_error"
            ],
            "receiver_weighted_relative_velocity_change": gpu[
                "receiver_weighted_relative_velocity_change"
            ],
            "receiver_relative_kinetic_energy_change": gpu[
                "receiver_relative_kinetic_energy_change"
            ],
            "relative_face_flux_fit_error": gpu["relative_face_flux_fit_error"],
            "relative_surface_velocity_error": gpu["relative_surface_velocity_error"],
            "projected_face_divergence_l2": gpu["projected_face_divergence_l2"],
        },
        "coupled_level9_prefix": coupled,
        "phase_momentum_ablation": coupled_momentum,
        "decision": {
            "conservative_q_transport": "pass for the stated 2-D manufactured fixed/dynamic-AMR tests",
            "receiver_remap": "pass conservation; disclose 0.0868% bed-aperture reconciliation",
            "receiver_constraint": "passes momentum/divergence; fails the provisional 0.5% field-change gate",
            "centered_momentum_coupling": "pass only as a stable level-9 prefix through t=0.1",
            "phase_momentum_coupling": "reject after solver divergence and floating-point exception",
            "impact_claim": "open: no coupled jet closure or grid-converged impact",
        },
    }


def create_figure(summary: dict[str, object], root: Path, output: Path) -> None:
    manufactured = summary["manufactured_translation"]
    fixed = manufactured["fixed_grid"]
    receiver = summary["receiver_handoff"]
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 8.0), constrained_layout=True)

    levels = np.asarray([item["level"] for item in fixed])
    h = 2.0 ** (-levels)
    error = np.asarray([item["relative_l1_error"] for item in fixed])
    axes[0, 0].loglog(h, error, "o-", color="#005F73", label="fixed grid")
    reference = error[-1] * (h / h[-1])
    axes[0, 0].loglog(h, reference, "--", color="#BB3E03", label=r"$O(h)$")
    axes[0, 0].invert_xaxis()
    axes[0, 0].set(
        xlabel=r"cell width $h$",
        ylabel=r"relative $L_1(q)$",
        title="(a) Curved-interface translation",
    )
    axes[0, 0].grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    labels = ["L6 fixed", "L7 fixed", "L8 fixed", "L7 frozen AMR", "L7 dynamic AMR"]
    drifts = [item["maximum_relative_mass_drift"] for item in fixed] + [
        manufactured["frozen_amr"]["maximum_relative_mass_drift"],
        manufactured["dynamic_amr"]["maximum_relative_mass_drift"],
    ]
    axes[0, 1].bar(labels, np.maximum(drifts, 1.0e-17), color="#0A9396")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(ylabel="maximum relative drift", title="(b) Physical-volume invariant")
    axes[0, 1].tick_params(axis="x", rotation=18)
    axes[0, 1].axhline(1.0e-12, color="black", linestyle="--", linewidth=0.9)
    axes[0, 1].grid(True, axis="y", which="both", alpha=0.25)

    remap_values = 100.0 * np.abs(
        [
            receiver["relative_conservative_remap_error"],
            receiver["relative_bed_reconciliation"],
            receiver["relative_receiver_momentum_error"],
            receiver["receiver_weighted_relative_velocity_change"],
        ]
    )
    remap_labels = ["volume remap", "bed reconciliation", "momentum residual", "velocity change"]
    axes[1, 0].barh(remap_labels, np.maximum(remap_values, 1.0e-12), color=["#005F73", "#EE9B00", "#0A9396", "#AE2012"])
    axes[1, 0].set_xscale("log")
    axes[1, 0].axvline(0.5, color="black", linestyle="--", linewidth=0.9, label="0.5% perturbation gate")
    axes[1, 0].set(xlabel="relative magnitude [%]", title="(c) Conservative receiver handoff")
    axes[1, 0].grid(True, axis="x", which="both", alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    for name, label, color in (
        ("q_conservative_receiver_l9_t010", "frozen AMR", "#005F73"),
        ("q_conservative_receiver_l9_dynamic_t010", "dynamic AMR", "#BB3E03"),
    ):
        time, drift = q_run_series(root / f"two_phase_basilisk/{name}/diagnostics.dat")
        axes[1, 1].semilogy(time, np.maximum(np.abs(drift), 1.0e-17), "o-", label=label, color=color)
    axes[1, 1].axhline(1.0e-6, color="black", linestyle="--", linewidth=0.9, label=r"$10^{-6}$ prefix gate")
    axes[1, 1].set(xlabel="receiver time", ylabel="relative volume drift", title="(d) Coupled level-9 prefix")
    axes[1, 1].grid(True, which="both", alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    figure.suptitle("Final audit of conservative liquid-aperture transport")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("research"))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("research/results/q_transport/final_q_stage"),
    )
    args = parser.parse_args()
    summary = build_summary(args.root)
    json_path = args.output_prefix.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    create_figure(summary, args.root, args.output_prefix.with_suffix(".png"))
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
