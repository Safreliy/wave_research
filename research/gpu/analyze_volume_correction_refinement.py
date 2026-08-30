"""Three-level diagnostic for the provisional post-AMR volume repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_corrected_amr_continuation import records
from analyze_handoff_time_screen import surface_profile


def level_metrics(directory: Path, target_volume: float,
                  target_momentum: np.ndarray) -> dict[str, object]:
    diagnostic_path = directory / "diagnostics.dat"
    text = diagnostic_path.read_text(encoding="utf-8")
    run = records(diagnostic_path, "RUN")
    transfer = records(diagnostic_path, "TRANSFER")
    volume = records(diagnostic_path, "POSTADAPT_VOLUME")
    invariant = records(diagnostic_path, "POSTADAPT_INVARIANTS")
    reproject = records(diagnostic_path, "POSTADAPT_REPROJECT")
    initial_momentum = np.asarray(transfer[0][3:5], dtype=float)
    target_norm = max(float(np.linalg.norm(target_momentum)), np.finfo(float).eps)
    return {
        "directory": str(directory),
        "completed_time": run[-1][0],
        "warning_count": text.count("WARNING"),
        "run_time": [row[0] for row in run],
        "relative_volume_error": [
            abs(row[2] - target_volume)/target_volume for row in run
        ],
        "maximum_relative_volume_error": max(
            abs(row[2] - target_volume)/target_volume for row in run
        ),
        "initial_receiver_momentum": initial_momentum.tolist(),
        "initial_receiver_momentum_relative_error": float(
            np.linalg.norm(initial_momentum - target_momentum)/target_norm
        ),
        "invariant_time": [row[0] for row in invariant],
        "invariant_velocity_change": [row[5] for row in invariant],
        "maximum_invariant_velocity_change": max(row[5] for row in invariant),
        "maximum_invariant_scale_deviation": max(abs(row[2] - 1.) for row in invariant),
        "maximum_invariant_momentum_residual": max(row[6] for row in invariant),
        "maximum_invariant_energy_residual": max(row[7] for row in invariant),
        "maximum_volume_l1_change": max(row[5] for row in volume),
        "maximum_raw_volume_momentum_change": max(row[7] for row in volume),
        "maximum_raw_volume_energy_change": max(row[8] for row in volume),
        "maximum_post_divergence_linf": max(row[5] for row in reproject),
        "maximum_reprojection_momentum_change": max(row[6] for row in reproject),
        "maximum_reprojection_energy_change": max(row[7] for row in reproject),
    }


def morphology(levels: dict[int, Path], y_min: float, y_max: float,
               frame_dt: float) -> dict[str, object]:
    paths = {
        level: sorted((directory / "handoff_frames").glob("vof-*.ppm"))
        for level, directory in levels.items()
    }
    count = min(len(value) for value in paths.values())
    profiles = {
        level: np.asarray([
            surface_profile(path, y_min, y_max) for path in value[:count]
        ])
        for level, value in paths.items()
    }
    amplitude = max(float(np.ptp(profiles[10][0])), np.finfo(float).eps)
    pairs: dict[str, object] = {}
    for coarse, fine in ((8, 9), (9, 10)):
        delta = profiles[coarse] - profiles[fine]
        rms = np.linalg.norm(delta, axis=1)/np.sqrt(delta.shape[1])/amplitude
        crest = np.abs(
            np.max(profiles[coarse], axis=1) - np.max(profiles[fine], axis=1)
        )/amplitude
        pairs[f"level_{coarse}_{fine}"] = {
            "maximum_relative_profile_rms": float(np.max(rms)),
            "final_relative_profile_rms": float(rms[-1]),
            "maximum_relative_crest_difference": float(np.max(crest)),
            "final_relative_crest_difference": float(crest[-1]),
        }
    return {
        "matched_frame_count": count,
        "matched_final_time": (count - 1)*frame_dt,
        "amplitude": amplitude,
        "profiles": profiles,
        "pairs": pairs,
    }


def compact(metrics: dict[str, object]) -> dict[str, object]:
    excluded = {"run_time", "relative_volume_error", "invariant_time",
                "invariant_velocity_change"}
    return {key: value for key, value in metrics.items() if key not in excluded}


def plot(metrics: dict[int, dict[str, object]], geometry: dict[str, object],
         output: Path, x_min: float, x_max: float) -> None:
    colors = {8: "#6A4C93", 9: "#168AAD", 10: "#C44536"}
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), constrained_layout=True)
    level_array = np.asarray([8, 9, 10])
    corrections = 100*np.asarray([
        metrics[level]["maximum_invariant_velocity_change"] for level in level_array
    ])
    axes[0, 0].semilogy(level_array, corrections, marker="o", lw=1.8, color="#17365D")
    axes[0, 0].axhline(0.5, color="#666666", ls=":", label="0.5% diagnostic ceiling")
    axes[0, 0].set_xticks(level_array)
    axes[0, 0].set_title("(a) Maximum compensating correction")
    axes[0, 0].set_xlabel("maximum quadtree level")
    axes[0, 0].set_ylabel("liquid velocity L2 change (%)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    for level in level_array:
        axes[0, 1].plot(
            metrics[level]["invariant_time"],
            100*np.asarray(metrics[level]["invariant_velocity_change"]),
            color=colors[level], lw=1.6, label=f"level {level}",
        )
    axes[0, 1].set_title("(b) Correction history")
    axes[0, 1].set_xlabel("continuation time")
    axes[0, 1].set_ylabel("liquid velocity L2 change (%)")
    axes[0, 1].legend(frameon=False, fontsize=7)
    for level in level_array:
        axes[1, 0].semilogy(
            metrics[level]["run_time"],
            np.maximum(metrics[level]["relative_volume_error"], 1e-16),
            color=colors[level], lw=1.6, marker="o", ms=3,
            label=f"level {level}",
        )
    axes[1, 0].axhline(1e-3, color="#666666", ls=":", label="volume gate")
    axes[1, 0].set_title("(c) Corrected liquid volume")
    axes[1, 0].set_xlabel("continuation time")
    axes[1, 0].set_ylabel("relative volume error")
    axes[1, 0].legend(frameon=False, fontsize=7)
    profiles = geometry["profiles"]
    x = np.linspace(x_min, x_max, profiles[10].shape[1], endpoint=False)
    for level in level_array:
        axes[1, 1].plot(
            x, profiles[level][-1], color=colors[level], lw=1.5,
            label=f"level {level}",
        )
    axes[1, 1].set_title(
        f"(d) Profiles at continuation time {geometry['matched_final_time']:.2f}"
    )
    axes[1, 1].set_xlabel(r"$x/h_0$")
    axes[1, 1].set_ylabel(r"$z/h_0$")
    axes[1, 1].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Three-level post-AMR repair diagnostic\n"
        "correction is non-monotone and level 10 has a pressure warning",
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
    parser.add_argument("--level8", type=Path, required=True)
    parser.add_argument("--level9", type=Path, required=True)
    parser.add_argument("--level10", type=Path, required=True)
    parser.add_argument("--x-min", type=float, default=28.)
    parser.add_argument("--x-max", type=float, default=39.)
    parser.add_argument("--y-min", type=float, default=-1.28)
    parser.add_argument("--y-max", type=float, default=1.25)
    parser.add_argument("--frame-dt", type=float, default=0.02)
    args = parser.parse_args()
    with np.load(args.source) as source:
        target_volume = float(source["source_volume"])
        target_momentum = np.asarray([
            float(source["target_momentum_x"]), float(source["target_momentum_z"])
        ])
    directories = {8: args.level8, 9: args.level9, 10: args.level10}
    metrics = {
        level: level_metrics(directory, target_volume, target_momentum)
        for level, directory in directories.items()
    }
    geometry = morphology(
        directories, args.y_min, args.y_max, args.frame_dt,
    )
    corrections = {
        str(level): metrics[level]["maximum_invariant_velocity_change"]
        for level in (8, 9, 10)
    }
    report = {
        "schema": "post-amr-volume-repair-refinement-v1",
        "target_volume": target_volume,
        "target_momentum": target_momentum.tolist(),
        "levels": {str(level): compact(value) for level, value in metrics.items()},
        "correction_maxima": corrections,
        "correction_ratio_level10_to_level9": (
            corrections["10"]/corrections["9"]
        ),
        "correction_monotone_over_levels_8_9_10": (
            corrections["10"] < corrections["9"] < corrections["8"]
        ),
        "morphology": {
            "matched_frame_count": geometry["matched_frame_count"],
            "matched_final_time": geometry["matched_final_time"],
            "pairs": geometry["pairs"],
        },
        "gates": {
            "all_completed_to_0p10": all(
                metrics[level]["completed_time"] >= 0.1 - 1e-12
                for level in (8, 9, 10)
            ),
            "all_volume_le_1e-3": all(
                metrics[level]["maximum_relative_volume_error"] <= 1e-3
                for level in (8, 9, 10)
            ),
            "all_invariant_residuals_le_1e-10": all(
                metrics[level]["maximum_invariant_momentum_residual"] <= 1e-10 and
                metrics[level]["maximum_invariant_energy_residual"] <= 1e-10
                for level in (8, 9, 10)
            ),
            "all_post_divergence_linf_le_1e-8": all(
                metrics[level]["maximum_post_divergence_linf"] <= 1e-8
                for level in (8, 9, 10)
            ),
            "all_pressure_warning_free": all(
                metrics[level]["warning_count"] == 0 for level in (8, 9, 10)
            ),
            "all_initial_receiver_momentum_error_le_1e-3": all(
                metrics[level]["initial_receiver_momentum_relative_error"] <= 1e-3
                for level in (8, 9, 10)
            ),
            "correction_monotone_with_refinement": (
                corrections["10"] < corrections["9"] < corrections["8"]
            ),
        },
        "accepted_for_publication": False,
        "conclusion": (
            "the repaired volume and liquid invariant residuals pass over the "
            "bounded three-level prefix, but the correction is non-monotone, "
            "level 10 has a pressure-convergence warning, and receiver-cell "
            "momentum is not preserved across levels; repair the receiver "
            "momentum projection before repeating this convergence audit"
        ),
    }
    plot(metrics, geometry, args.output_figure, args.x_min, args.x_max)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
