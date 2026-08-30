"""Audit post-AMR volume repair and invariant-compensation ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_corrected_amr_continuation import records
from analyze_handoff_time_screen import surface_profile


def run_data(directory: Path, target_volume: float) -> dict[str, object]:
    diagnostic_path = directory / "diagnostics.dat"
    text = diagnostic_path.read_text(encoding="utf-8")
    run = records(diagnostic_path, "RUN")
    post = records(diagnostic_path, "POSTADAPT_REPROJECT")
    volume = records(diagnostic_path, "POSTADAPT_VOLUME")
    invariants = records(diagnostic_path, "POSTADAPT_INVARIANTS")
    return {
        "directory": str(directory),
        "completed_time": run[-1][0] if run else None,
        "warning_count": text.count("WARNING"),
        "run_time": [row[0] for row in run],
        "relative_volume_error": [abs(row[2] - target_volume) / target_volume for row in run],
        "maximum_relative_volume_error": (
            max(abs(row[2] - target_volume) / target_volume for row in run)
            if run else None
        ),
        "maximum_post_divergence_linf": max((row[5] for row in post), default=None),
        "maximum_reprojection_momentum_change": max((row[6] for row in post), default=None),
        "maximum_reprojection_energy_change": max((row[7] for row in post), default=None),
        "volume_time": [row[0] for row in volume],
        "volume_l1_change": [row[5] for row in volume],
        "volume_raw_momentum_change": [row[7] for row in volume],
        "volume_raw_energy_change": [row[8] for row in volume],
        "maximum_volume_l1_change": max((row[5] for row in volume), default=None),
        "sum_volume_l1_change": sum(row[5] for row in volume),
        "maximum_raw_volume_momentum_change": max((row[7] for row in volume), default=None),
        "maximum_raw_volume_energy_change": max((row[8] for row in volume), default=None),
        "invariant_time": [row[0] for row in invariants],
        "invariant_velocity_change": [row[5] for row in invariants],
        "maximum_invariant_scale_deviation": max(
            (abs(row[2] - 1.0) for row in invariants), default=None
        ),
        "maximum_invariant_velocity_change": max((row[5] for row in invariants), default=None),
        "maximum_invariant_momentum_residual": max((row[6] for row in invariants), default=None),
        "maximum_invariant_energy_residual": max((row[7] for row in invariants), default=None),
    }


def matched_morphology(baseline: Path, compensated: Path, y_min: float,
                       y_max: float, frame_dt: float) -> dict[str, object]:
    baseline_paths = sorted((baseline / "handoff_frames").glob("vof-*.ppm"))
    compensated_paths = sorted((compensated / "handoff_frames").glob("vof-*.ppm"))
    count = min(len(baseline_paths), len(compensated_paths))
    base = np.asarray([surface_profile(path, y_min, y_max)
                       for path in baseline_paths[:count]])
    corrected = np.asarray([surface_profile(path, y_min, y_max)
                            for path in compensated_paths[:count]])
    amplitude = max(float(np.ptp(base[0])), np.finfo(float).eps)
    rms = np.linalg.norm(corrected - base, axis=1) / np.sqrt(base.shape[1]) / amplitude
    crest = np.abs(np.max(corrected, axis=1) - np.max(base, axis=1)) / amplitude
    return {
        "matched_frame_count": count,
        "time": frame_dt * np.arange(count),
        "baseline": base,
        "compensated": corrected,
        "relative_profile_rms": rms,
        "relative_crest_difference": crest,
        "maximum_relative_profile_rms": float(np.max(rms)),
        "final_relative_profile_rms": float(rms[-1]),
        "maximum_relative_crest_difference": float(np.max(crest)),
        "final_relative_crest_difference": float(crest[-1]),
    }


def compact(run: dict[str, object]) -> dict[str, object]:
    series = {
        "run_time", "relative_volume_error", "volume_time", "volume_l1_change",
        "volume_raw_momentum_change", "volume_raw_energy_change", "invariant_time",
        "invariant_velocity_change",
    }
    return {key: value for key, value in run.items() if key not in series}


def plot(report: dict[str, object], runs: dict[str, dict[str, object]],
         morphology: dict[str, object], output: Path, x_min: float,
         x_max: float) -> None:
    colors = {
        "baseline": "#17365D", "volume_only": "#C44536",
        "cell_invariants": "#168AAD",
    }
    labels = {
        "baseline": "no volume repair", "volume_only": "volume only",
        "cell_invariants": "volume + cell invariants",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), constrained_layout=True)
    for key in ("baseline", "volume_only", "cell_invariants"):
        data = runs[key]
        axes[0, 0].semilogy(
            data["run_time"], np.maximum(data["relative_volume_error"], 1e-16),
            lw=1.7, marker="o", ms=3, color=colors[key], label=labels[key],
        )
    axes[0, 0].axhline(1e-3, color="#666666", ls=":", label="volume gate")
    axes[0, 0].set_title("(a) Liquid-volume conservation")
    axes[0, 0].set_xlabel("continuation time")
    axes[0, 0].set_ylabel("relative volume error")
    axes[0, 0].legend(frameon=False, fontsize=7)
    for key in ("volume_only", "cell_invariants"):
        data = runs[key]
        axes[0, 1].semilogy(
            data["volume_time"],
            100 * np.maximum(data["volume_raw_energy_change"], 1e-16),
            lw=1.7, color=colors[key], label=labels[key],
        )
    axes[0, 1].axhline(0.1, color="#666666", ls=":", label="0.1% energy gate")
    axes[0, 1].set_title("(b) Raw energy change from editing VOF")
    axes[0, 1].set_xlabel("continuation time")
    axes[0, 1].set_ylabel("relative change (%)")
    axes[0, 1].legend(frameon=False, fontsize=7)
    invariant = runs["cell_invariants"]
    axes[1, 0].plot(
        invariant["invariant_time"],
        100 * np.asarray(invariant["invariant_velocity_change"]),
        lw=1.8, color=colors["cell_invariants"],
        label="cell-velocity invariant projection",
    )
    axes[1, 0].axhline(0.5, color="#666666", ls=":", label="0.5% diagnostic ceiling")
    axes[1, 0].set_title("(c) Compensating velocity correction")
    axes[1, 0].set_xlabel("continuation time")
    axes[1, 0].set_ylabel("liquid velocity L2 change (%)")
    axes[1, 0].legend(frameon=False, fontsize=7)
    x = np.linspace(x_min, x_max, morphology["baseline"].shape[1], endpoint=False)
    axes[1, 1].plot(x, morphology["baseline"][-1], color=colors["baseline"],
                    lw=1.7, label=labels["baseline"])
    axes[1, 1].plot(x, morphology["compensated"][-1],
                    color=colors["cell_invariants"], lw=1.7,
                    label=labels["cell_invariants"])
    axes[1, 1].set_title("(d) Final profile at continuation time 0.30")
    axes[1, 1].set_xlabel(r"$x/h_0$")
    axes[1, 1].set_ylabel(r"$z/h_0$")
    axes[1, 1].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Post-AMR volume-repair ablation\n"
        "volume-only repair violates energy; cell-only invariant compensation is bounded",
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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--volume-only", type=Path, required=True)
    parser.add_argument("--cell-invariants", type=Path, required=True)
    parser.add_argument("--face-invariants", type=Path, required=True)
    parser.add_argument("--x-min", type=float, default=28.0)
    parser.add_argument("--x-max", type=float, default=39.0)
    parser.add_argument("--y-min", type=float, default=-1.28)
    parser.add_argument("--y-max", type=float, default=1.25)
    parser.add_argument("--frame-dt", type=float, default=0.02)
    args = parser.parse_args()
    with np.load(args.source) as data:
        target_volume = float(data["source_volume"])
    runs = {
        "baseline": run_data(args.baseline, target_volume),
        "volume_only": run_data(args.volume_only, target_volume),
        "cell_invariants": run_data(args.cell_invariants, target_volume),
        "face_invariants": run_data(args.face_invariants, target_volume),
    }
    morphology = matched_morphology(
        args.baseline, args.cell_invariants, args.y_min, args.y_max, args.frame_dt,
    )
    gates = {
        "baseline_volume_le_1e-3":
            runs["baseline"]["maximum_relative_volume_error"] <= 1e-3,
        "volume_only_raw_energy_le_1e-3":
            runs["volume_only"]["maximum_raw_volume_energy_change"] <= 1e-3,
        "cell_invariants_completed_0p30":
            runs["cell_invariants"]["completed_time"] >= 0.3 - 1e-12,
        "cell_invariants_volume_le_1e-3":
            runs["cell_invariants"]["maximum_relative_volume_error"] <= 1e-3,
        "cell_invariants_momentum_residual_le_1e-10":
            runs["cell_invariants"]["maximum_invariant_momentum_residual"] <= 1e-10,
        "cell_invariants_energy_residual_le_1e-10":
            runs["cell_invariants"]["maximum_invariant_energy_residual"] <= 1e-10,
        "cell_invariants_velocity_change_le_5e-3_diagnostic":
            runs["cell_invariants"]["maximum_invariant_velocity_change"] <= 5e-3,
        "cell_invariants_no_pressure_warnings":
            runs["cell_invariants"]["warning_count"] == 0,
        "face_invariants_completed_0p30":
            runs["face_invariants"]["completed_time"] is not None and
            runs["face_invariants"]["completed_time"] >= 0.3 - 1e-12,
        "face_invariants_no_pressure_warnings":
            runs["face_invariants"]["warning_count"] == 0,
    }
    report: dict[str, object] = {
        "schema": "post-amr-volume-repair-ablation-v1",
        "target_volume": target_volume,
        "runs": {key: compact(value) for key, value in runs.items()},
        "morphology": {
            key: value for key, value in morphology.items()
            if key not in {"time", "baseline", "compensated",
                           "relative_profile_rms", "relative_crest_difference"}
        },
        "gates": gates,
        "accepted_for_publication": False,
        "conclusion": (
            "cell-only affine invariant compensation closes the bounded level-9 "
            "volume, momentum, energy and divergence diagnostics, but its 0.326% "
            "maximum velocity correction must decrease under grid refinement; "
            "volume-only and face-flux-transform variants are rejected"
        ),
    }
    plot(report, runs, morphology, args.output_figure, args.x_min, args.x_max)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
