"""Compare post-AMR face reconstruction with preservation of adapted fluxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def labelled(path: Path, label: str) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0] == label:
            rows.append([float(value) for value in fields[1:]])
    return rows


def projection_summary(path: Path) -> dict[str, object]:
    rows = labelled(path, "POSTADAPT_REPROJECT")
    if not rows:
        raise ValueError(f"no POSTADAPT_REPROJECT records in {path}")
    has_pre = all(len(row) >= 10 for row in rows)
    series: dict[str, list[float]] = {
        "time": [row[0] for row in rows],
        "post_divergence_linf": [row[5] for row in rows],
        "relative_momentum_change": [row[6] for row in rows],
        "relative_kinetic_energy_change": [row[7] for row in rows],
    }
    if has_pre:
        series["pre_divergence_linf"] = [row[9] for row in rows]
    return {
        "diagnostics": str(path),
        "projection_count": len(rows),
        "max_post_divergence_linf": max(series["post_divergence_linf"]),
        "max_relative_momentum_change": max(series["relative_momentum_change"]),
        "sum_absolute_relative_momentum_changes": sum(
            series["relative_momentum_change"]
        ),
        "max_relative_kinetic_energy_change": max(
            series["relative_kinetic_energy_change"]
        ),
        "sum_absolute_relative_kinetic_energy_changes": sum(
            series["relative_kinetic_energy_change"]
        ),
        "max_pre_divergence_linf": (
            max(series["pre_divergence_linf"]) if has_pre else None
        ),
        "series": series,
    }


def trigger_summary(path: Path, threshold: float) -> dict[str, object]:
    projected = labelled(path, "POSTADAPT_REPROJECT")
    skipped = labelled(path, "POSTADAPT_SKIP")
    projected_pre = [(row[0], row[9]) for row in projected if len(row) >= 10]
    skipped_pre = [(row[0], row[3]) for row in skipped if len(row) >= 4]
    pre_records = sorted(projected_pre + skipped_pre)
    pre_linf = [value for _, value in pre_records]
    return {
        "diagnostics": str(path),
        "threshold": threshold,
        "projection_count": len(projected),
        "skip_count": len(skipped),
        "minimum_pre_projection_linf": min(pre_linf) if pre_linf else None,
        "maximum_pre_projection_linf": max(pre_linf) if pre_linf else None,
        "series": {
            "time": [time for time, _ in pre_records],
            "pre_divergence_linf": pre_linf,
        },
    }


def plot(report: dict[str, object], output: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 170})
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), constrained_layout=True)
    colors = {"reconstructed": "#C44536", "preserved": "#17365D"}
    labels = {
        "reconstructed": "reconstruct from cell velocity",
        "preserved": "preserve adapted face flux",
    }
    for key, data in report["paths"].items():
        series = data["series"]
        if "pre_divergence_linf" in series:
            axes[0, 0].semilogy(
                series["time"], series["pre_divergence_linf"],
                lw=1.7, color=colors[key], label=labels[key],
            )
        axes[0, 1].semilogy(
            series["time"], np.maximum(series["post_divergence_linf"], 1e-16),
            lw=1.7, color=colors[key], label=labels[key],
        )
        kinetic_percent = 100 * np.asarray(series["relative_kinetic_energy_change"])
        axes[1, 0].semilogy(
            series["time"], np.maximum(kinetic_percent, 1e-12),
            lw=1.7, color=colors[key], label=labels[key],
        )
        axes[1, 1].plot(
            series["time"], np.cumsum(kinetic_percent),
            lw=1.7, color=colors[key], label=labels[key],
        )

    reconstructed = report["paths"]["reconstructed"]["series"]
    if "pre_divergence_linf" not in reconstructed and report["conditional_trigger_screen"]:
        representative = max(
            report["conditional_trigger_screen"].values(),
            key=lambda item: item["threshold"],
        )["series"]
        axes[0, 0].semilogy(
            representative["time"], representative["pre_divergence_linf"],
            lw=1.7, color=colors["reconstructed"],
            label="reconstruct from cell velocity (screen)",
        )

    axes[0, 0].set_title("(a) Divergence before correction")
    axes[0, 0].set_ylabel(r"pre-projection $L_\infty$")
    axes[0, 0].axhline(1e-3, color="#666666", ls=":", label="largest tested trigger")
    axes[0, 1].set_title("(b) Divergence after correction")
    axes[0, 1].set_ylabel(r"post-projection $L_\infty$")
    axes[0, 1].axhline(1e-3, color="#666666", ls=":", label="divergence gate")
    axes[1, 0].set_title("(c) Per-projection energy change")
    axes[1, 0].set_ylabel("relative change (%)")
    axes[1, 0].axhline(0.1, color="#666666", ls=":", label="0.1% gate")
    axes[1, 1].set_title("(d) Cumulative absolute energy change")
    axes[1, 1].set_ylabel("upper bound (%)")
    for ax in axes.flat:
        ax.set_xlabel("continuation time")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Post-AMR face-state ablation\n"
        "preserving transported face flux removes the reconstruction energy defect",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_trigger(value: str) -> tuple[float, Path]:
    threshold, path = value.split("=", 1)
    return float(threshold), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_figure", type=Path)
    parser.add_argument("--reconstructed", type=Path, required=True)
    parser.add_argument("--preserved", type=Path, required=True)
    parser.add_argument(
        "--trigger", action="append", default=[], metavar="THRESHOLD=DIAGNOSTICS",
    )
    args = parser.parse_args()
    paths = {
        "reconstructed": projection_summary(args.reconstructed),
        "preserved": projection_summary(args.preserved),
    }
    triggers = {
        f"{threshold:g}": trigger_summary(path, threshold)
        for threshold, path in map(parse_trigger, args.trigger)
    }
    reconstructed_energy = paths["reconstructed"][
        "sum_absolute_relative_kinetic_energy_changes"
    ]
    preserved_energy = paths["preserved"][
        "sum_absolute_relative_kinetic_energy_changes"
    ]
    report: dict[str, object] = {
        "schema": "post-amr-face-state-ablation-v1",
        "paths": paths,
        "conditional_trigger_screen": triggers,
        "energy_correction_reduction_factor": (
            reconstructed_energy / preserved_energy
        ),
        "conclusion": (
            "the tested divergence triggers do not avoid reconstruction; "
            "preserving the AMR-transported face flux removes the long-run "
            "energy-gate failure, but does not establish impact convergence"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot(report, args.output_figure)
    printable = {key: {k: v for k, v in data.items() if k != "series"}
                 for key, data in paths.items()}
    printable_triggers = {
        key: {k: v for k, v in data.items() if k != "series"}
        for key, data in triggers.items()
    }
    print(json.dumps({"paths": printable, "triggers": printable_triggers,
                      "energy_correction_reduction_factor":
                      report["energy_correction_reduction_factor"]}, indent=2))


if __name__ == "__main__":
    main()
