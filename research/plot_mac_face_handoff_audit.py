"""Publication figure for the late-state MAC face-flux handoff audit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "results" / "publication_package"
CPU_JSON = ROOT / "gpu" / "mac_h200_m1e8_cpu.json"
GPU_JSON = ROOT / "gpu" / "mac_h200_m1e8_gpu.json"
EXPORT_JSON = ROOT / "gpu" / "mac_face_flux_export.json"
AUDIT_JSON = PACKAGE / "mac_face_handoff_audit.json"
METHOD_JSON = PACKAGE / "mac_face_handoff_method_summary.json"


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    cpu = load(CPU_JSON)
    gpu = load(GPU_JSON)
    export = load(EXPORT_JSON)
    audit = load(AUDIT_JSON)
    method = load(METHOD_JSON)
    levels = [8, 9, 10]
    rows = [audit["initial_transfer_by_level"][str(level)]["metrics"] for level in levels]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 170,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), constrained_layout=True)
    navy, teal, orange, red = "#17365D", "#168AAD", "#F4A261", "#C44536"

    ax = axes[0, 0]
    representation = method["representation_diagnostics"]
    backend = method["backend_reproducibility"]
    raw_cell_divergence = representation["direct_cell_centered_divergence_rms"]
    raw_face_divergence = export["face_flux_divergence_l2"]
    projected_divergence = cpu["projected_face_divergence_l2"]
    vals = [raw_cell_divergence, raw_face_divergence, projected_divergence]
    ax.bar(range(3), vals, color=[red, orange, teal], width=0.66)
    ax.set_yscale("log")
    ax.set_xticks(range(3), ["cell samples", "face averages", "MAC fit"])
    ax.set_ylabel(r"discrete divergence RMS")
    ax.set_title("(a) Representation compatibility")
    for i, val in enumerate(vals):
        ax.text(i, val * (1.7 if i < 2 else 3.2), f"{val:.2e}", ha="center", va="bottom")

    ax = axes[0, 1]
    metric_names = ["face fit", "surface fit", "speed ratio"]
    measured = [
        100 * cpu["relative_face_flux_fit_error"],
        100 * cpu["relative_surface_velocity_error"],
        cpu["maximum_liquid_to_surface_q99_speed_ratio"],
    ]
    gates = [2.0, 5.0, 1.5]
    x = np.arange(3)
    width = 0.36
    ax.bar(x - width / 2, measured, width, color=teal, label="measured")
    ax.bar(x + width / 2, gates, width, color=navy, alpha=0.72, label="gate")
    ax.set_xticks(x, metric_names)
    ax.set_ylabel("percent, percent, ratio")
    ax.set_title("(b) Bounded late-state reconstruction")
    ax.legend(frameon=False)
    for i, val in enumerate(measured):
        ax.text(i - width / 2, val + 0.09, f"{val:.3g}", ha="center", va="bottom")

    ax = axes[1, 0]
    impulses = 100 * np.array([row["relative_projection_momentum_impulse"] for row in rows])
    ax.plot(levels, impulses, "o-", color=navy, lw=2, ms=6)
    ax.axhline(1.0, color=red, ls="--", lw=1.3, label="1% gate")
    ax.set_xticks(levels)
    ax.set_xlabel("maximum quadtree level")
    ax.set_ylabel("projection impulse (%)")
    ax.set_ylim(0, 1.08)
    ax.set_title("(c) Receiver pressure-projection impulse")
    ax.legend(frameon=False)
    for level, val in zip(levels, impulses):
        ax.text(level, val + 0.045, f"{val:.3f}", ha="center")

    ax = axes[1, 1]
    rms = np.array([row["post_projection_face_divergence_rms"] for row in rows])
    linf = np.array([row["post_projection_face_divergence_linf"] for row in rows])
    ax.semilogy(levels, rms, "o-", color=teal, lw=2, ms=6, label=r"RMS")
    ax.semilogy(levels, linf, "s--", color=orange, lw=1.7, ms=5, label=r"$L_\infty$")
    ax.axhline(1e-3, color=red, ls=":", lw=1.5, label=r"$10^{-3}$ gate")
    ax.set_xticks(levels)
    ax.set_xlabel("maximum quadtree level")
    ax.set_ylabel("projected face divergence")
    ax.set_ylim(5e-9, 3e-3)
    ax.set_title("(d) Initial finite-volume compatibility")
    ax.legend(frameon=False, ncol=2)

    speedup = cpu["solve_seconds"] / gpu["solve_seconds"]
    fig.suptitle(
        "Late active N=576 BIE-to-VOF MAC handoff\n"
        "CPU/GPU field agreement "
        f"{backend['relative_trusted_face_field_difference']:.2e}; "
        f"GPU sparse-LSQR speedup {speedup:.2f}x",
        fontsize=11,
    )
    PACKAGE.mkdir(parents=True, exist_ok=True)
    fig.savefig(PACKAGE / "mac_face_handoff_audit.png", bbox_inches="tight")
    fig.savefig(PACKAGE / "mac_face_handoff_audit.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
