"""Quantify grid disagreement in the physical BIE-to-VOF continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage


def liquid_mask(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return rgb[..., 0].astype(int) > rgb[..., 2].astype(int)


def surface_profile(path: Path, y_min: float, y_max: float) -> np.ndarray:
    """Return the upper liquid boundary at every output-pixel column."""
    liquid = liquid_mask(path)
    has_liquid = np.any(liquid, axis=0)
    first = np.argmax(liquid, axis=0)
    profile = y_max - (first + 0.5) * (y_max - y_min) / liquid.shape[0]
    profile = profile.astype(float)
    profile[~has_liquid] = np.nan
    if np.any(~has_liquid):
        valid = np.flatnonzero(has_liquid)
        profile[~has_liquid] = np.interp(
            np.flatnonzero(~has_liquid), valid, profile[valid]
        )
    return profile


def roughness(profile: np.ndarray, sigma_pixels: float = 12.0) -> float:
    low_pass = ndimage.gaussian_filter1d(profile, sigma_pixels, mode="nearest")
    scale = max(float(np.ptp(profile)), np.finfo(float).eps)
    return float(np.linalg.norm(profile - low_pass) / np.sqrt(len(profile)) / scale)


def analyze(
    coarse_paths: list[Path],
    fine_paths: list[Path],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    frame_dt: float,
    case_label: str = "Physical N=256 BIE-to-VOF continuation",
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    count = min(len(coarse_paths), len(fine_paths))
    if count < 2:
        raise ValueError("need at least two matched frames")
    coarse = np.asarray(
        [surface_profile(path, y_min, y_max) for path in coarse_paths[:count]]
    )
    fine = np.asarray(
        [surface_profile(path, y_min, y_max) for path in fine_paths[:count]]
    )
    if coarse.shape != fine.shape:
        raise ValueError("matched rendered frames must have equal pixel shape")
    times = frame_dt * np.arange(count)
    amplitude_scale = max(float(np.ptp(fine[0])), np.finfo(float).eps)
    rms_difference = np.linalg.norm(coarse - fine, axis=1) / np.sqrt(coarse.shape[1])
    relative_rms_difference = rms_difference / amplitude_scale
    coarse_crest = np.max(coarse, axis=1)
    fine_crest = np.max(fine, axis=1)
    relative_crest_difference = np.abs(fine_crest - coarse_crest) / amplitude_scale
    coarse_roughness = np.asarray([roughness(row) for row in coarse])
    fine_roughness = np.asarray([roughness(row) for row in fine])
    final = count - 1
    report: dict[str, object] = {
        "schema": "coupled-vof-morphology-audit-v2",
        "evidence_class": "bounded two-grid morphology diagnostic",
        "case_label": case_label,
        "coarse_level": 9,
        "fine_level": 10,
        "matched_frame_count": count,
        "time_interval": [float(times[0]), float(times[-1])],
        "window": [x_min, x_max, y_min, y_max],
        "amplitude_scale": amplitude_scale,
        "maximum_relative_profile_rms_difference": float(
            np.max(relative_rms_difference)
        ),
        "final_relative_profile_rms_difference": float(
            relative_rms_difference[final]
        ),
        "final_crest_difference": float(fine_crest[final] - coarse_crest[final]),
        "maximum_relative_crest_difference": float(
            np.max(relative_crest_difference)
        ),
        "final_relative_crest_difference": float(
            relative_crest_difference[final]
        ),
        "final_coarse_crest": float(coarse_crest[final]),
        "final_fine_crest": float(fine_crest[final]),
        "final_coarse_roughness": float(coarse_roughness[final]),
        "final_fine_roughness": float(fine_roughness[final]),
        "profile_convergence_gate_relative_rms_le_0p05": bool(
            relative_rms_difference[final] <= 0.05
        ),
        "crest_convergence_gate_relative_difference_le_0p05": bool(
            relative_crest_difference[final] <= 0.05
        ),
        "accepted_long_coupled_morphology": bool(
            relative_rms_difference[final] <= 0.05
            and relative_crest_difference[final] <= 0.05
        ),
    }
    arrays = {
        "time": times,
        "coarse": coarse,
        "fine": fine,
        "relative_rms_difference": relative_rms_difference,
        "coarse_crest": coarse_crest,
        "fine_crest": fine_crest,
        "relative_crest_difference": relative_crest_difference,
        "coarse_roughness": coarse_roughness,
        "fine_roughness": fine_roughness,
    }
    return report, arrays


def plot(report: dict[str, object], arrays: dict[str, np.ndarray], output: Path,
         x_min: float, x_max: float) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.8), constrained_layout=True)
    x = np.linspace(x_min, x_max, arrays["coarse"].shape[1], endpoint=False)
    positions = np.linspace(0, len(arrays["time"]) - 1, 3, dtype=int)
    colors = ("#64748b", "#0f766e", "#b45309")
    for position, color in zip(positions, colors):
        time = arrays["time"][position]
        axes[0, 0].plot(
            x, arrays["coarse"][position], color=color, ls="--", lw=1.5,
            label=fr"L9, $\Delta t={time:.2f}$",
        )
        axes[0, 0].plot(
            x, arrays["fine"][position], color=color, lw=2,
            label=fr"L10, $\Delta t={time:.2f}$",
        )
    axes[0, 0].set_title("Matched free-surface profiles")
    axes[0, 0].set_xlabel(r"$x/h_0$")
    axes[0, 0].set_ylabel(r"$z/h_0$")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)

    axes[0, 1].plot(
        arrays["time"], arrays["relative_rms_difference"], color="#9f1239", lw=2
    )
    axes[0, 1].axhline(0.05, color="#374151", ls="--", lw=1)
    axes[0, 1].set_title("Level-9/10 profile disagreement")
    axes[0, 1].set_xlabel(r"VOF continuation time $\Delta t$")
    axes[0, 1].set_ylabel("RMS difference / initial amplitude")

    axes[1, 0].plot(arrays["time"], arrays["coarse_crest"], ls="--", lw=2,
                    color="#2563eb", label="level 9")
    axes[1, 0].plot(arrays["time"], arrays["fine_crest"], lw=2,
                    color="#2563eb", label="level 10")
    axes[1, 0].set_title("Crest elevation")
    axes[1, 0].set_xlabel(r"VOF continuation time $\Delta t$")
    axes[1, 0].set_ylabel(r"$\max z/h_0$")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(arrays["time"], arrays["coarse_roughness"], ls="--", lw=2,
                    color="#7c3aed", label="level 9")
    axes[1, 1].plot(arrays["time"], arrays["fine_roughness"], lw=2,
                    color="#7c3aed", label="level 10")
    axes[1, 1].set_title("High-frequency profile content")
    axes[1, 1].set_xlabel(r"VOF continuation time $\Delta t$")
    axes[1, 1].set_ylabel("normalized high-pass RMS")
    axes[1, 1].legend(frameon=False)
    for axis in axes.ravel():
        axis.grid(True, alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    status = "PASS" if report["accepted_long_coupled_morphology"] else "FAIL"
    fig.suptitle(
        f"{report['case_label']}: morphology gate {status}",
        fontsize=15,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("coarse_frames", type=Path)
    parser.add_argument("fine_frames", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--x-min", type=float, required=True)
    parser.add_argument("--x-max", type=float, required=True)
    parser.add_argument("--y-min", type=float, required=True)
    parser.add_argument("--y-max", type=float, required=True)
    parser.add_argument("--frame-dt", type=float, default=0.02)
    parser.add_argument(
        "--case-label", default="Physical N=256 BIE-to-VOF continuation",
    )
    args = parser.parse_args()
    coarse_paths = sorted(args.coarse_frames.glob("vof-*.ppm"))
    fine_paths = sorted(args.fine_frames.glob("vof-*.ppm"))
    report, arrays = analyze(
        coarse_paths, fine_paths, args.x_min, args.x_max,
        args.y_min, args.y_max, args.frame_dt, args.case_label,
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output_prefix.with_suffix(".npz"), **arrays)
    plot(
        report, arrays, args.output_prefix.with_suffix(".png"),
        args.x_min, args.x_max,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
