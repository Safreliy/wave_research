"""Compact publication keyframe for the accepted active-sheet overhang."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(source: Path, output: Path) -> None:
    with np.load(source) as data:
        index = len(data["time"]) - 1
        count = int(data["count"][index])
        x = np.asarray(data["x"][index, :count], dtype=float)
        z = np.asarray(data["z"][index, :count], dtype=float)
        gamma = np.asarray(data["panel_circulation"][index, :count], dtype=float)
        time = float(data["time"][index])
        min_x_alpha = float(data["min_x_alpha"][index])
        gap = float(data["normalized_impact_distance"][index])
        flux = abs(float(data["surface_flux_defect"][index]))
    x_min, x_max = 31.2, 35.5
    z_min, z_max = -0.055, 0.86
    selected = np.flatnonzero((x >= x_min - 0.15) & (x <= x_max + 0.15))
    local_x, local_z, local_gamma = x[selected], z[selected], gamma[selected]
    gamma_scale = max(float(np.quantile(np.abs(gamma), 0.985)), 1.0e-14)

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 3.0))
    figure.subplots_adjust(left=0.065, right=0.98, bottom=0.23, top=0.88, wspace=0.13)
    for axis in axes:
        polygon_x = np.r_[local_x, local_x[-1], local_x[0]]
        polygon_z = np.r_[local_z, z_min, z_min]
        axis.fill(polygon_x, polygon_z, color="#55acd1", alpha=0.72)
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(z_min, z_max)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(r"$x/h_0$")
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].plot(local_x, local_z, color="#123a57", linewidth=2.6)
    axes[0].set_title(f"Smooth geometry, t={time:.6f}", fontsize=11.5)
    axes[0].set_ylabel(r"$z/h_0$")
    colors = np.where(local_gamma >= 0.0, "#e34a3b", "#136f8a")
    sizes = 7.0 + 54.0 * np.sqrt(
        np.clip(np.abs(local_gamma) / gamma_scale, 0.0, 1.0)
    )
    axes[1].plot(local_x, local_z, color="#8493a2", linewidth=0.7, alpha=0.6)
    axes[1].scatter(local_x, local_z, c=colors, s=sizes, edgecolors="none", alpha=0.9)
    axes[1].set_title(r"Active carriers: red $\Gamma_j>0$, blue $\Gamma_j<0$", fontsize=11.5)
    axes[1].text(
        0.98,
        0.06,
        (
            rf"$\min x_\alpha={min_x_alpha:+.3f}$" "\n"
            rf"$d_{{\rm nl}}/\widetilde{{\Delta s}}={gap:.2f}$" "\n"
            rf"raw flux defect $={flux:.2e}$"
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#c8d2dc", "alpha": 0.92},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=210)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.source, args.output)


if __name__ == "__main__":
    main()
