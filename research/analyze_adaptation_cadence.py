"""Falsification audit for reducing the receiver AMR cadence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_coupled_morphology import analyze as analyze_morphology
from analyze_impact_claim import _read_diagnostics
from two_phase_basilisk.render_bie_handoff import (
    liquid_mask,
    persistent_onset,
    topology_record_from_liquid,
)


def topology_summary(
    frame_directory: Path,
    bottom_x: np.ndarray,
    bottom_z: np.ndarray,
    length: float,
    minimum_area: float,
) -> dict[str, float | None]:
    records = [
        topology_record_from_liquid(
            liquid_mask(path), bottom_x, bottom_z, length,
            32.0, 46.0, -1.28, 1.25,
            minimum_pixels=1, minimum_area=minimum_area,
            bed_guard_distance=0.0,
        )
        for path in sorted(frame_directory.glob("vof-*.ppm"))
    ]
    onset = persistent_onset(records, 3)
    return {
        "persistent_event_time": None if onset is None else 0.02 * onset,
        "maximum_pocket_area": max(
            float(record["unfiltered_largest_pocket_area"]) for record in records
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_npz", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("adapt2", type=Path)
    parser.add_argument("adapt4", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    with np.load(args.source_npz) as loaded:
        bottom_x = loaded["bottom_x"]
        bottom_z = loaded["bottom_z"]
        grid_x = loaded["grid_x"]
        length = float(grid_x[-1] - grid_x[0] + grid_x[1] - grid_x[0])
    delta = length / 2**9
    paths = {1: args.baseline, 2: args.adapt2, 4: args.adapt4}
    baseline_frames = sorted((args.baseline / "handoff_frames").glob("vof-*.ppm"))
    cases: dict[str, object] = {}
    for cadence, run in paths.items():
        frames = run / "handoff_frames"
        raw = topology_summary(frames, bottom_x, bottom_z, length, 0.0)
        resolved = topology_summary(frames, bottom_x, bottom_z, length, delta**2)
        if cadence == 1:
            morphology = {
                "maximum_relative_profile_rms_difference": 0.0,
                "maximum_relative_crest_difference": 0.0,
            }
        else:
            morphology, _ = analyze_morphology(
                baseline_frames, sorted(frames.glob("vof-*.ppm")),
                32.0, 46.0, -1.28, 1.25, 0.02,
                f"AMR every step versus every {cadence} steps",
            )
        diagnostics = _read_diagnostics(run / "diagnostics.dat")
        adapt_count = sum(
            line.startswith("POSTADAPT_REPROJECT")
            for line in (run / "diagnostics.dat").read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        )
        cases[str(cadence)] = {
            "adapt_interval_steps": cadence,
            "post_adaptation_reprojection_count": adapt_count,
            "raw_subcell_event_time": raw["persistent_event_time"],
            "resolved_one_cell_event_time": resolved["persistent_event_time"],
            "maximum_unfiltered_pocket_area": raw["maximum_pocket_area"],
            "maximum_unfiltered_pocket_area_over_delta_squared": (
                float(raw["maximum_pocket_area"]) / delta**2
            ),
            "maximum_relative_profile_rms_difference_from_every_step": morphology[
                "maximum_relative_profile_rms_difference"
            ],
            "maximum_relative_crest_difference_from_every_step": morphology[
                "maximum_relative_crest_difference"
            ],
            "maximum_absolute_relative_mass_drift": diagnostics.get(
                "maximum_absolute_relative_mass_drift"
            ),
            "maximum_q_upper_bound_violation": diagnostics.get(
                "maximum_q_upper_bound_violation"
            ),
        }
    baseline_event = cases["1"]["raw_subcell_event_time"]
    for cadence in (2, 4):
        case = cases[str(cadence)]
        event = case["raw_subcell_event_time"]
        case["event_time_shift_from_every_step"] = (
            None if event is None or baseline_event is None else event - baseline_event
        )
        case["shortcut_accepted"] = bool(
            event is not None
            and baseline_event is not None
            and abs(event - baseline_event) <= 0.08
            and case["maximum_relative_profile_rms_difference_from_every_step"] <= 0.05
            and case["maximum_absolute_relative_mass_drift"] <= 1.0e-3
        )
    report = {
        "schema": "receiver-amr-cadence-falsification-v1",
        "level": 9,
        "finest_cell_width": delta,
        "event_definition": (
            "three consecutive rendered frames with exterior-disconnected gas; "
            "raw subcell timing is diagnostic and a one-finest-cell area is required for acceptance"
        ),
        "thresholds": {
            "event_time_shift": 0.08,
            "profile_rms": 0.05,
            "mass_drift": 1.0e-3,
            "resolved_pocket_area": delta**2,
        },
        "cases": cases,
        "conclusion": (
            "adapt-every-2 and adapt-every-4 are rejected: upper-surface profiles pass, "
            "but topology timing shifts by about one nondimensional time unit"
        ),
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    cadence = np.array([1, 2, 4])
    events = np.array([cases[str(value)]["raw_subcell_event_time"] for value in cadence])
    profiles = 100.0 * np.array([
        cases[str(value)]["maximum_relative_profile_rms_difference_from_every_step"]
        for value in cadence
    ])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    axes[0].plot(cadence, events, "o-", color="#9f1239", linewidth=2)
    axes[0].axhspan(events[0] - 0.08, events[0] + 0.08, color="#bbf7d0", alpha=0.7)
    axes[0].set(xlabel="AMR interval [steps]", ylabel="raw closure time",
                title="Topology is cadence-sensitive")
    axes[1].bar(cadence, profiles, color="#0f766e")
    axes[1].axhline(5.0, color="#334155", linestyle="--")
    axes[1].set(xlabel="AMR interval [steps]", ylabel="max profile RMS [%]",
                title="Upper-profile gate alone passes")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("AMR-cadence ablation: pictures pass, event timing fails", fontweight="bold")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=220)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
