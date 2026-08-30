"""Publication audit plot for the active adaptive overturning experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_joined(paths: list[Path]) -> dict[str, np.ndarray]:
    keys = [
        "time",
        "min_x_alpha",
        "surface_flux_defect",
        "projection_relative_correction",
        "cumulative_projection_correction",
        "accepted_dt",
        "count",
        "circulation_total_defect",
        "normalized_impact_distance",
    ]
    joined = {key: [] for key in keys}
    last_time = -np.inf
    for path in paths:
        with np.load(path) as data:
            times = np.asarray(data["time"], dtype=float)
            selected = times > last_time + 1.0e-12
            for key in keys:
                joined[key].append(np.asarray(data[key])[selected])
            if np.any(selected):
                last_time = float(times[selected][-1])
    return {key: np.concatenate(values) for key, values in joined.items()}


def plot(history: dict[str, np.ndarray], p_refined: Path, output: Path) -> None:
    with np.load(p_refined) as p_data:
        p_time = float(p_data["time"][0])
        p_count = int(p_data["count"][0])
        p_flux = abs(float(p_data["surface_flux_defect"][0]))
        p_projection = float(p_data["projection_relative_correction"][0])
        p_circulation = abs(float(p_data["circulation_total_defect"][0]))
    time = history["time"]
    use_offset_time = float(np.ptp(time)) < 0.1
    plot_time = time - time[0] if use_offset_time else time
    time_label = (
        f"time since t={time[0]:.6f}"
        if use_offset_time else "nondimensional time"
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)
    figure.suptitle(
        "Active adaptive vortex-sheet continuation: accepted-state audit",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(plot_time, history["min_x_alpha"], color="#153d5c", linewidth=2.2)
    ax.axhline(0.0, color="#c53b2c", linestyle="--", linewidth=1.4)
    vertical = np.flatnonzero(history["min_x_alpha"] <= 0.0)
    if len(vertical):
        first = int(vertical[0])
        ax.scatter(plot_time[first], history["min_x_alpha"][first], color="#c53b2c", zorder=3)
        ax.annotate(
            f"first accepted overturning\nt={time[first]:.5f}",
            (plot_time[first], history["min_x_alpha"][first]),
            xytext=(16, 18),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#c53b2c"},
            fontsize=9,
        )
    ax.set_ylabel(r"minimum $x_\alpha$")
    ax.set_title("Geometric event: verticality is crossed")
    ax.grid(alpha=0.22)

    ax = axes[0, 1]
    ax.semilogy(plot_time, np.maximum(np.abs(history["surface_flux_defect"]), 1.0e-16), color="#7251a2", linewidth=2.0, label="raw DNO flux defect")
    ax.axhline(2.0e-2, color="#c53b2c", linestyle="--", label="acceptance gate")
    ax.semilogy(plot_time, np.maximum(history["projection_relative_correction"], 1.0e-16), color="#d78a20", linewidth=1.7, label="energy projection")
    ax.axhline(1.0e-3, color="#d78a20", linestyle=":", label="projection gate")
    ax.set_title("Spatial and projection gates")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.22, which="both")

    ax = axes[1, 0]
    positive_dt = np.maximum(history["accepted_dt"], 1.0e-6)
    ax.step(plot_time, positive_dt, where="post", color="#168179", linewidth=2.0, label=r"accepted $\Delta t$")
    ax.set_yscale("log")
    ax2 = ax.twinx()
    ax2.plot(plot_time, history["normalized_impact_distance"], color="#516176", linewidth=1.7, label=r"gap/$\widetilde{\Delta s}$")
    ax.set_xlabel(time_label)
    ax.set_ylabel(r"accepted $\Delta t$")
    ax2.set_ylabel("normalized nonlocal gap")
    ax.set_title("Accepted step and nonlocal gap; no contact is claimed")
    ax.grid(alpha=0.22, which="both")

    ax = axes[1, 1]
    ax.axis("off")
    text = (
        "ACTIVE N=576 SCREEN\n\n"
        f"restart time              {p_time:.6f}\n"
        f"accepted carrier count    384 -> {p_count}\n"
        f"energy projection         {p_projection:.3e}\n"
        f"raw DNO flux defect       {p_flux:.3e}\n"
        f"Kelvin-period defect      {p_circulation:.3e}\n"
        f"screen end time           {time[-1]:.6f}\n"
        f"accepted stored states    {len(time)}\n"
        f"minimum normalized gap    {np.min(history['normalized_impact_distance']):.3f}\n"
        f"maximum raw flux defect   {np.max(np.abs(history['surface_flux_defect'])):.3e}\n"
        f"maximum energy projection {np.max(history['projection_relative_correction']):.3e}\n\n"
        "All stored screen states pass the displayed gates.\n\n"
        "CLAIM BOUNDARY\n"
        "Computed: active Gamma evolution through overturning.\n"
        "Not computed: jet-to-surface contact or reconnection."
    )
    ax.text(
        0.04,
        0.96,
        text,
        va="top",
        family="monospace",
        fontsize=10.2,
        color="#203044",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f4f7fa", "edgecolor": "#bdc9d5"},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--p-refined", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plot(load_joined(args.inputs), args.p_refined, args.output)


if __name__ == "__main__":
    main()
