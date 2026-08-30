"""Compare two accepted BIE handoff times on the same level-9 VOF grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_corrected_amr_continuation import records


def surface_profile(path: Path, y_min: float, y_max: float) -> np.ndarray:
    from PIL import Image

    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    liquid = rgb[..., 0].astype(int) > rgb[..., 2].astype(int)
    valid = np.any(liquid, axis=0)
    first = np.argmax(liquid, axis=0)
    profile = y_max - (first + 0.5) * (y_max - y_min) / liquid.shape[0]
    if np.any(~valid):
        indices = np.flatnonzero(valid)
        profile[~valid] = np.interp(np.flatnonzero(~valid), indices, profile[valid])
    return profile


def load_run(run_directory: Path, source: Path, y_min: float, y_max: float) -> dict[str, object]:
    frames = sorted((run_directory / "handoff_frames").glob("vof-*.ppm"))
    run = records(run_directory / "diagnostics.dat", "RUN")
    post = records(run_directory / "diagnostics.dat", "POSTADAPT_REPROJECT")
    if not frames or not run or not post:
        raise ValueError(f"incomplete run in {run_directory}")
    with np.load(source) as data:
        target_volume = float(data["source_volume"])
        source_time = float(data["time"])
    volume_error = [abs(row[2] - target_volume) / target_volume for row in run]
    return {
        "run_directory": str(run_directory),
        "source": str(source),
        "source_time": source_time,
        "target_volume": target_volume,
        "frames": frames,
        "profiles": np.asarray([surface_profile(path, y_min, y_max) for path in frames]),
        "run_time": np.asarray([row[0] for row in run]),
        "relative_volume_error": np.asarray(volume_error),
        "maximum_relative_volume_error": max(volume_error),
        "final_relative_volume_error": volume_error[-1],
        "maximum_post_divergence_linf": max(row[5] for row in post),
        "maximum_relative_momentum_change": max(row[6] for row in post),
        "maximum_relative_kinetic_energy_change": max(row[7] for row in post),
    }


def serializable(run: dict[str, object]) -> dict[str, object]:
    return {
        key: value for key, value in run.items()
        if key not in {"frames", "profiles", "run_time", "relative_volume_error"}
    }


def plot(report: dict[str, object], early: dict[str, object], late: dict[str, object],
         output: Path, x_min: float, x_max: float, frame_dt: float) -> None:
    count = min(len(early["profiles"]), len(late["profiles"]))
    early_profiles = early["profiles"][:count]
    late_profiles = late["profiles"][:count]
    times = frame_dt * np.arange(count)
    amplitude = max(float(np.ptp(early_profiles[0])), np.finfo(float).eps)
    rms = np.linalg.norm(late_profiles - early_profiles, axis=1) / np.sqrt(
        early_profiles.shape[1]
    ) / amplitude
    early_crest = np.max(early_profiles, axis=1)
    late_crest = np.max(late_profiles, axis=1)
    x = np.linspace(x_min, x_max, early_profiles.shape[1], endpoint=False)
    labels = {
        "early": f"handoff t={early['source_time']:.6f}",
        "late": f"handoff t={late['source_time']:.6f}",
    }
    colors = {"early": "#17365D", "late": "#C44536"}
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), constrained_layout=True)
    axes[0, 0].plot(x, early_profiles[-1], color=colors["early"], lw=1.8,
                    label=labels["early"])
    axes[0, 0].plot(x, late_profiles[-1], color=colors["late"], lw=1.8,
                    label=labels["late"])
    axes[0, 0].set_title(f"(a) Matched profiles at continuation time {times[-1]:.2f}")
    axes[0, 0].set_xlabel(r"$x/h_0$")
    axes[0, 0].set_ylabel(r"$z/h_0$")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(times, rms, color="#7C3AED", lw=1.8)
    axes[0, 1].axhline(0.05, color="#666666", ls=":", label="5% screen")
    axes[0, 1].set_title("(b) Handoff-time profile sensitivity")
    axes[0, 1].set_xlabel("VOF continuation time")
    axes[0, 1].set_ylabel("RMS difference / initial amplitude")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(times, early_crest, color=colors["early"], lw=1.8,
                    label=labels["early"])
    axes[1, 0].plot(times, late_crest, color=colors["late"], lw=1.8,
                    label=labels["late"])
    axes[1, 0].set_title("(c) Crest history")
    axes[1, 0].set_xlabel("VOF continuation time")
    axes[1, 0].set_ylabel(r"crest $z/h_0$")
    axes[1, 0].legend(frameon=False)
    for key, run_data in (("early", early), ("late", late)):
        axes[1, 1].semilogy(
            run_data["run_time"],
            np.maximum(run_data["relative_volume_error"], 1e-16),
            color=colors[key], marker="o", ms=3, lw=1.6, label=labels[key],
        )
    axes[1, 1].axhline(1e-3, color="#666666", ls=":", label="volume gate")
    axes[1, 1].set_title("(d) Liquid-volume conservation")
    axes[1, 1].set_xlabel("VOF continuation time")
    axes[1, 1].set_ylabel("relative volume error")
    axes[1, 1].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Accepted BIE handoff-time screen on level 9\n"
        "the later source remains stable but fails the fixed volume gate",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    report["matched_frame_count"] = count
    report["matched_time_interval"] = [float(times[0]), float(times[-1])]
    report["maximum_relative_profile_rms_difference"] = float(np.max(rms))
    report["final_relative_profile_rms_difference"] = float(rms[-1])
    report["maximum_relative_crest_difference"] = float(
        np.max(np.abs(late_crest - early_crest)) / amplitude
    )
    report["final_relative_crest_difference"] = float(
        abs(late_crest[-1] - early_crest[-1]) / amplitude
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_figure", type=Path)
    parser.add_argument("--early-run", type=Path, required=True)
    parser.add_argument("--late-run", type=Path, required=True)
    parser.add_argument("--early-source", type=Path, required=True)
    parser.add_argument("--late-source", type=Path, required=True)
    parser.add_argument("--gpu-report", type=Path, required=True)
    parser.add_argument("--x-min", type=float, default=28.0)
    parser.add_argument("--x-max", type=float, default=39.0)
    parser.add_argument("--y-min", type=float, default=-1.28)
    parser.add_argument("--y-max", type=float, default=1.25)
    parser.add_argument("--frame-dt", type=float, default=0.02)
    args = parser.parse_args()
    early = load_run(args.early_run, args.early_source, args.y_min, args.y_max)
    late = load_run(args.late_run, args.late_source, args.y_min, args.y_max)
    gpu = json.loads(args.gpu_report.read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "schema": "bie-handoff-time-screen-v1",
        "evidence_class": "bounded handoff-time sensitivity diagnostic",
        "early": serializable(early),
        "late": serializable(late),
        "late_gpu_reconstruction": gpu,
        "gates": {
            "late_face_fit_le_0p02": gpu["relative_face_flux_fit_error"] <= 0.02,
            "late_surface_fit_le_0p05": gpu["relative_surface_velocity_error"] <= 0.05,
            "late_momentum_error_le_1e-8": gpu["relative_momentum_error"] <= 1e-8,
            "late_projected_divergence_linf_le_1e-10":
                gpu["projected_face_divergence_linf"] <= 1e-10,
            "late_vof_volume_error_le_1e-3":
                late["maximum_relative_volume_error"] <= 1e-3,
        },
    }
    plot(report, early, late, args.output_figure, args.x_min, args.x_max,
         args.frame_dt)
    report["all_gates_passed"] = all(report["gates"].values())
    report["conclusion"] = (
        "late MAC reconstruction and post-AMR correction pass, but the level-9 "
        "VOF volume gate fails; no handoff-time or impact claim is promoted"
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
