"""Three-grid geometric audit for conservative-q receiver continuations.

This is intentionally separate from the connected-component impact gate.  It
tests whether the visible upper-surface trajectory approaches a common limit;
it cannot establish contact or air entrainment by itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_coupled_morphology import roughness, surface_profile


def richardson_order(coarse_error: np.ndarray, fine_error: np.ndarray) -> np.ndarray:
    coarse = np.asarray(coarse_error, dtype=float)
    fine = np.asarray(fine_error, dtype=float)
    if coarse.shape != fine.shape:
        raise ValueError("paired error histories must have equal shape")
    order = np.full(coarse.shape, np.nan, dtype=float)
    valid = (coarse > 0.0) & (fine > 0.0)
    order[valid] = np.log2(coarse[valid] / fine[valid])
    return order


def analyze(manifest: dict[str, object]) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    cases = sorted(manifest["cases"], key=lambda case: int(case["level"]))
    if len(cases) != 3:
        raise ValueError("three distinct receiver levels are required")
    levels = [int(case["level"]) for case in cases]
    if levels[1] - levels[0] != 1 or levels[2] - levels[1] != 1:
        raise ValueError("receiver levels must be consecutive")
    frame_sets = [
        sorted(Path(case["frame_directory"]).glob("vof-*.ppm")) for case in cases
    ]
    count = min(map(len, frame_sets))
    if count < 2:
        raise ValueError("each receiver level must contain at least two frames")
    geometry = manifest["geometry"]
    y_min = float(geometry["y_min"])
    y_max = float(geometry["y_max"])
    profiles = np.asarray(
        [
            [surface_profile(path, y_min, y_max) for path in frames[:count]]
            for frames in frame_sets
        ]
    )
    if len({profile.shape for profile in profiles}) != 1:
        raise ValueError("all rendered profile arrays must have equal shape")
    dt = float(manifest["thresholds"]["simulation_dt"])
    times = dt * np.arange(count)
    scale = max(float(np.ptp(profiles[-1, 0])), np.finfo(float).eps)
    pair_rms = np.linalg.norm(np.diff(profiles, axis=0), axis=2)
    pair_rms /= np.sqrt(profiles.shape[2]) * scale
    orders = richardson_order(pair_rms[0], pair_rms[1])
    crests = np.max(profiles, axis=2)
    crest_pair = np.abs(np.diff(crests, axis=0)) / scale
    late = times >= max(0.0, times[-1] - 1.0)
    finite_late_order = orders[late & np.isfinite(orders)]
    positive_order_fraction = (
        float(np.mean(finite_late_order > 0.0)) if len(finite_late_order) else None
    )
    median_late_order = (
        float(np.median(finite_late_order)) if len(finite_late_order) else None
    )
    thresholds = {
        "fine_pair_profile_rms": 0.05,
        "fine_pair_crest_difference": 0.05,
        "late_positive_order_fraction": 0.7,
    }
    gates = {
        "final_fine_pair_profile": bool(pair_rms[1, -1] <= 0.05),
        "final_fine_pair_crest": bool(crest_pair[1, -1] <= 0.05),
        "late_positive_order": bool(
            positive_order_fraction is not None and positive_order_fraction >= 0.7
        ),
    }
    report: dict[str, object] = {
        "schema": "three-grid-upper-surface-audit-v1",
        "evidence_class": "geometric computation; not a contact theorem",
        "levels": levels,
        "matched_frame_count": count,
        "time_interval": [float(times[0]), float(times[-1])],
        "amplitude_scale": scale,
        "thresholds": thresholds,
        "coarse_pair_maximum_relative_profile_rms": float(np.max(pair_rms[0])),
        "fine_pair_maximum_relative_profile_rms": float(np.max(pair_rms[1])),
        "coarse_pair_final_relative_profile_rms": float(pair_rms[0, -1]),
        "fine_pair_final_relative_profile_rms": float(pair_rms[1, -1]),
        "fine_pair_maximum_relative_crest_difference": float(np.max(crest_pair[1])),
        "fine_pair_final_relative_crest_difference": float(crest_pair[1, -1]),
        "median_observed_order_over_last_unit_time": median_late_order,
        "positive_order_fraction_over_last_unit_time": positive_order_fraction,
        "gates": gates,
        "accepted_three_grid_upper_surface": all(gates.values()),
    }
    arrays = {
        "time": times,
        "profiles": profiles,
        "pair_rms": pair_rms,
        "observed_order": orders,
        "crests": crests,
        "crest_pair": crest_pair,
        "roughness": np.asarray(
            [[roughness(profile) for profile in level] for level in profiles]
        ),
    }
    return report, arrays


def plot(report: dict[str, object], arrays: dict[str, np.ndarray], output: Path) -> None:
    levels = report["levels"]
    time = arrays["time"]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.9), constrained_layout=True)
    axes[0, 0].plot(time, arrays["pair_rms"][0], label=f"L{levels[0]}/L{levels[1]}")
    axes[0, 0].plot(time, arrays["pair_rms"][1], label=f"L{levels[1]}/L{levels[2]}")
    axes[0, 0].axhline(0.05, color="#991b1b", ls="--", label="5% gate")
    axes[0, 0].set(title="Upper-profile pair differences", ylabel="RMS / initial amplitude")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(time, np.clip(arrays["observed_order"], -4, 4), color="#7c3aed")
    axes[0, 1].axhline(0.0, color="#991b1b", ls="--")
    axes[0, 1].set(title="Instantaneous Richardson indicator", ylabel=r"$\log_2(E_c/E_f)$")

    colors = ("#64748b", "#0f766e", "#d97706")
    for level, crest, color in zip(levels, arrays["crests"], colors):
        axes[1, 0].plot(time, crest, label=f"level {level}", color=color)
    axes[1, 0].set(title="Crest trajectory", ylabel=r"$\max z/h_0$")
    axes[1, 0].legend(frameon=False)

    for level, values, color in zip(levels, arrays["roughness"], colors):
        axes[1, 1].plot(time, values, label=f"level {level}", color=color)
    axes[1, 1].set(title="High-frequency profile content", ylabel="normalized high-pass RMS")
    axes[1, 1].legend(frameon=False)
    for axis in axes.ravel():
        axis.set_xlabel("continuation time")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Three-grid upper-surface comparison",
        fontsize=14, fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report, arrays = analyze(manifest)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output_prefix.with_suffix(".npz"), **arrays)
    plot(report, arrays, args.output_prefix.with_suffix(".png"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
