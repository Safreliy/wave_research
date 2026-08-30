"""Plot the three-level short physical BIE-to-VOF compatibility audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("level8", type=Path)
    parser.add_argument("level9", type=Path)
    parser.add_argument("level10", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.level8, args.level9, args.level10)
    ]
    levels = np.asarray([8, 9, 10])
    mass = np.abs([report["relative_mass_error"] for report in reports])
    face = np.asarray(
        [report["first_projected_step"]["face_divergence_l2"] for report in reports]
    )
    impulse = np.asarray(
        [report["relative_projection_momentum_impulse"] for report in reports]
    )
    volume = np.abs(
        [report["relative_volume_change_at_last_run"] for report in reports]
    )
    if not all(report["coupled_continuation_accepted"] for report in reports):
        raise ValueError("all three inputs must be accepted short-run audits")

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.6), constrained_layout=True)
    panels = (
        (axes[0, 0], mass, 1.0e-6, "Initialization mass identity", r"$|\Delta V|/V$"),
        (axes[0, 1], face, 1.0e-3, "First projected face divergence", r"$\|\nabla_h\cdot u_f\|_2$"),
        (axes[1, 0], impulse, 1.0e-2, "Projection momentum impulse", r"$\|\Delta\mathbf{P}\|/\|\mathbf{P}\|$"),
        (axes[1, 1], volume, 1.0e-3, r"Liquid-volume change by $\Delta t=0.2$", r"$|V(0.2)-V(0)|/V(0)$"),
    )
    for axis, values, gate, title, ylabel in panels:
        axis.semilogy(levels, np.maximum(values, 1.0e-15), "o-", color="#0f766e", lw=2.2, ms=7)
        axis.axhline(gate, color="#374151", ls="--", lw=1, label="acceptance gate")
        axis.set_xticks(levels)
        axis.set_xlim(7.7, 10.3)
        axis.set_xlabel("maximum quadtree level")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        for level, value in zip(levels, values):
            axis.annotate(f"{value:.2e}", (level, max(value, 1.0e-15)),
                          xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Physical N=256 early BIE-to-VOF handoff: three-level compatibility",
        fontsize=15,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
