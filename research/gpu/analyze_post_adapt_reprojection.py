"""Audit the pure Helmholtz re-projection performed after quadtree AMR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def records(path: Path, label: str) -> list[list[float]]:
    parsed: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0] == label:
            parsed.append([float(value) for value in fields[1:]])
    return parsed


def level_report(path: Path) -> dict[str, object]:
    rows = records(path, "POSTADAPT_REPROJECT")
    if not rows:
        raise ValueError(f"no POSTADAPT_REPROJECT records in {path}")
    entries = [
        {
            "time": row[0],
            "iteration": int(row[1]),
            "poisson_iterations": int(row[2]),
            "poisson_residual": row[3],
            "face_divergence_rms": row[4],
            "face_divergence_linf": row[5],
            "relative_momentum_change": row[6],
            "relative_kinetic_energy_change": row[7],
        }
        for row in rows
    ]
    maxima = {
        key: max(float(entry[key]) for entry in entries)
        for key in (
            "face_divergence_rms",
            "face_divergence_linf",
            "relative_momentum_change",
            "relative_kinetic_energy_change",
        )
    }
    gates = {
        "face_divergence_rms_le_1e-3": maxima["face_divergence_rms"] <= 1.0e-3,
        "face_divergence_linf_le_1e-3": maxima["face_divergence_linf"] <= 1.0e-3,
        "momentum_change_le_1e-3": maxima["relative_momentum_change"] <= 1.0e-3,
        "kinetic_change_le_1e-3": maxima["relative_kinetic_energy_change"] <= 1.0e-3,
    }
    return {
        "diagnostics": str(path),
        "records": entries,
        "maxima": maxima,
        "gates": gates,
        "all_short_reprojection_gates_passed": all(gates.values()),
    }


def plot_report(report: dict[str, object], baseline: dict[str, object], output: Path) -> None:
    levels = [8, 9, 10]
    level_data = report["levels"]
    after_rms = np.array(
        [level_data[str(level)]["maxima"]["face_divergence_rms"] for level in levels]
    )
    after_linf = np.array(
        [level_data[str(level)]["maxima"]["face_divergence_linf"] for level in levels]
    )
    momentum = 100 * np.array(
        [level_data[str(level)]["maxima"]["relative_momentum_change"] for level in levels]
    )
    kinetic = 100 * np.array(
        [level_data[str(level)]["maxima"]["relative_kinetic_energy_change"] for level in levels]
    )
    before_levels = [9, 10]
    before_rms = np.array(
        [
            baseline["initial_transfer_by_level"][str(level)][
                "first_post_adaptation_projection"
            ]["face_divergence_rms"]
            for level in before_levels
        ]
    )
    before_linf = np.array(
        [
            baseline["initial_transfer_by_level"][str(level)][
                "first_post_adaptation_projection"
            ]["face_divergence_linf"]
            for level in before_levels
        ]
    )

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 170})
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), constrained_layout=True)
    navy, teal, orange, red = "#17365D", "#168AAD", "#F4A261", "#C44536"
    for ax, before, after, title in (
        (axes[0, 0], before_rms, after_rms, "(a) Face-divergence RMS"),
        (axes[0, 1], before_linf, after_linf, r"(b) Face-divergence $L_\infty$"),
    ):
        ax.semilogy(before_levels, before, "s--", color=orange, lw=1.8, label="ordinary AMR")
        ax.semilogy(levels, after, "o-", color=teal, lw=2, label="post-AMR projection")
        ax.axhline(1.0e-3, color=red, ls=":", lw=1.4, label=r"$10^{-3}$ gate")
        ax.set_xticks(levels)
        ax.set_xlabel("maximum quadtree level")
        ax.set_title(title)
        ax.legend(frameon=False)

    axes[1, 0].plot(levels, momentum, "o-", color=navy, lw=2)
    axes[1, 0].axhline(0.1, color=red, ls=":", lw=1.4, label="0.1% diagnostic gate")
    axes[1, 0].set_xticks(levels)
    axes[1, 0].set_xlabel("maximum quadtree level")
    axes[1, 0].set_ylabel("maximum correction (%)")
    axes[1, 0].set_title("(c) Liquid-momentum change")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(levels, kinetic, "o-", color=navy, lw=2)
    axes[1, 1].axhline(0.1, color=red, ls=":", lw=1.4, label="0.1% diagnostic gate")
    axes[1, 1].set_xticks(levels)
    axes[1, 1].set_xlabel("maximum quadtree level")
    axes[1, 1].set_ylabel("maximum correction (%)")
    axes[1, 1].set_title("(d) Liquid kinetic-energy change")
    axes[1, 1].legend(frameon=False)

    fig.suptitle(
        "MAC handoff: short post-AMR Helmholtz audit\n"
        "three iterations per level; coupled impact convergence not implied",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_figure", type=Path)
    parser.add_argument("--level8", type=Path, required=True)
    parser.add_argument("--level9", type=Path, required=True)
    parser.add_argument("--level10", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    levels = {
        "8": level_report(args.level8),
        "9": level_report(args.level9),
        "10": level_report(args.level10),
    }
    report: dict[str, object] = {
        "schema": "bie-to-vof-post-amr-reprojection-audit-v1",
        "method": (
            "reconstruct face flux from the adapted cell velocity, apply a pure "
            "unit-pseudotime Helmholtz projection, average the pressure-only "
            "correction back to cell centres, and do not reapply generic face "
            "boundary prolongation to the projected flux"
        ),
        "levels": levels,
        "all_three_levels_passed": all(
            level["all_short_reprojection_gates_passed"] for level in levels.values()
        ),
        "claim_limit": (
            "short three-iteration AMR compatibility only; a long coupled "
            "continuation and contact convergence remain open"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.baseline.open("r", encoding="utf-8") as stream:
        baseline = json.load(stream)
    plot_report(report, baseline, args.output_figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
