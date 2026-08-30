"""Audit bounded continuations using the post-AMR Helmholtz correction."""

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


def audit(path: Path, target_volume: float, intended_horizon: float) -> dict[str, object]:
    post = records(path, "POSTADAPT_REPROJECT")
    run = records(path, "RUN")
    if not post or not run:
        raise ValueError(f"missing reprojection or run records in {path}")
    volume_errors = [abs(row[2] - target_volume) / target_volume for row in run]
    maxima = {
        "face_divergence_rms": max(row[4] for row in post),
        "face_divergence_linf": max(row[5] for row in post),
        "relative_momentum_change_per_projection": max(row[6] for row in post),
        "relative_kinetic_energy_change_per_projection": max(row[7] for row in post),
        "relative_volume_error": max(volume_errors),
    }
    if all(len(row) >= 10 for row in post):
        maxima.update({
            "pre_projection_face_divergence_rms": max(row[8] for row in post),
            "pre_projection_face_divergence_linf": max(row[9] for row in post),
        })
    correction_upper_bounds = {
        "sum_absolute_relative_momentum_changes": sum(row[6] for row in post),
        "sum_absolute_relative_kinetic_energy_changes": sum(row[7] for row in post),
    }
    gates = {
        "completed_intended_horizon": run[-1][0] >= intended_horizon - 0.02 - 1.0e-12,
        "face_divergence_rms_le_1e-3": maxima["face_divergence_rms"] <= 1.0e-3,
        "face_divergence_linf_le_1e-3": maxima["face_divergence_linf"] <= 1.0e-3,
        "momentum_change_per_projection_le_1e-3": maxima[
            "relative_momentum_change_per_projection"
        ]
        <= 1.0e-3,
        "kinetic_change_per_projection_le_1e-3": maxima[
            "relative_kinetic_energy_change_per_projection"
        ]
        <= 1.0e-3,
        "relative_volume_error_le_1e-3": maxima["relative_volume_error"] <= 1.0e-3,
    }
    return {
        "diagnostics": str(path),
        "intended_horizon": intended_horizon,
        "last_reported_time": run[-1][0],
        "post_adaptation_projection_count": len(post),
        "final_relative_volume_error": volume_errors[-1],
        "maxima": maxima,
        "correction_upper_bounds": correction_upper_bounds,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "series": {
            "post_time": [row[0] for row in post],
            "face_divergence_rms": [row[4] for row in post],
            "face_divergence_linf": [row[5] for row in post],
            "relative_momentum_change": [row[6] for row in post],
            "relative_kinetic_energy_change": [row[7] for row in post],
            "run_time": [row[0] for row in run],
            "relative_volume_error": volume_errors,
        },
    }


def plot(report: dict[str, object], output: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 170})
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), constrained_layout=True)
    colors = {"level9_t1": "#17365D", "level10_t010": "#168AAD"}
    labels = {
        key: f"level {data['level']}, horizon {data['last_reported_time']:.2f}"
        for key, data in report["runs"].items()
    }

    ax = axes[0, 0]
    for key, data in report["runs"].items():
        series = data["series"]
        ax.semilogy(
            series["post_time"], series["face_divergence_rms"],
            color=colors[key], lw=1.8, label=f"{labels[key]} RMS",
        )
        ax.semilogy(
            series["post_time"], series["face_divergence_linf"],
            color=colors[key], lw=1.3, ls="--", label=f"{labels[key]} Linf",
        )
    ax.axhline(1e-3, color="#C44536", ls=":", label="1e-3 gate")
    ax.set_xlabel("continuation time")
    ax.set_ylabel("face-divergence norm")
    ax.set_title("(a) Post-AMR finite-volume compatibility")
    ax.legend(frameon=False, ncol=2, fontsize=7)

    for ax, field, title in (
        (axes[0, 1], "relative_momentum_change", "(b) Momentum correction"),
        (axes[1, 0], "relative_kinetic_energy_change", "(c) Kinetic-energy correction"),
    ):
        for key, data in report["runs"].items():
            series = data["series"]
            ax.plot(
                series["post_time"], 100 * np.asarray(series[field]),
                color=colors[key], lw=1.8, label=labels[key],
            )
        ax.axhline(0.1, color="#C44536", ls=":", label="0.1% gate")
        ax.set_xlabel("continuation time")
        ax.set_ylabel("per-projection change (%)")
        ax.set_title(title)
        ax.legend(frameon=False)

    ax = axes[1, 1]
    for key, data in report["runs"].items():
        series = data["series"]
        ax.semilogy(
            series["run_time"], np.maximum(series["relative_volume_error"], 1e-16),
            color=colors[key], lw=1.8, marker="o", ms=3, label=labels[key],
        )
    ax.axhline(1e-3, color="#C44536", ls=":", label="1e-3 gate")
    ax.set_xlabel("continuation time")
    ax.set_ylabel("relative liquid-volume error")
    ax.set_title("(d) Volume conservation")
    ax.legend(frameon=False)

    all_pass = all(data["all_gates_passed"] for data in report["runs"].values())
    gate_summary = (
        "reported numerical gates pass; coupled contact convergence is not implied"
        if all_pass else
        "at least one numerical gate fails; coupled contact claim remains rejected"
    )
    fig.suptitle(
        f"{report['method_label']} continuation audit\n{gate_summary}",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_figure", type=Path)
    parser.add_argument("--level9", type=Path, required=True)
    parser.add_argument("--level10", type=Path, required=True)
    parser.add_argument("--level9-horizon", type=float, default=1.0)
    parser.add_argument("--level10-horizon", type=float, default=0.1)
    parser.add_argument(
        "--method-label", default="Post-AMR Helmholtz-corrected",
        help="descriptive label used in the report and figure title",
    )
    args = parser.parse_args()
    with np.load(args.source) as loaded:
        target_volume = float(loaded["source_volume"])
    runs = {
        "level9_t1": audit(args.level9, target_volume, args.level9_horizon),
        "level10_t010": audit(args.level10, target_volume, args.level10_horizon),
    }
    runs["level9_t1"]["level"] = 9
    runs["level10_t010"]["level"] = 10
    all_pass = all(data["all_gates_passed"] for data in runs.values())
    report: dict[str, object] = {
        "schema": "bie-to-vof-corrected-amr-continuation-audit-v2",
        "method_label": args.method_label,
        "target_volume": target_volume,
        "runs": runs,
        "publication_status": (
            "reported numerical gates pass, but level 10 is only a short prefix "
            "and no coupled contact claim is accepted"
            if all_pass else
            "at least one reported numerical gate fails and no coupled contact "
            "claim is accepted"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot(report, args.output_figure)
    print(json.dumps({key: {k: v for k, v in value.items() if k != "series"}
                      for key, value in runs.items()}, indent=2))


if __name__ == "__main__":
    main()
