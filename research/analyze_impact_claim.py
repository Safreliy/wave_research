"""Audit the numerical evidence needed for a coupled impact claim.

The script deliberately separates three kinds of evidence:

* finite-volume evidence: mass/bounds diagnostics from the receiver log;
* graph evidence: persistent connected components of enclosed gas;
* geometric evidence: event-time and pocket-area changes across mesh levels.

It consumes a JSON manifest so every threshold is recorded next to the result.
No single rendered frame can pass the audit: closure must persist, survive a
diagnostic-raster sensitivity check, and be reproduced across mesh levels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from two_phase_basilisk.render_bie_handoff import (
    liquid_mask,
    persistent_onset,
    topology_record_from_liquid,
)


def _downsample(
    mask: np.ndarray, divisor: int, *, liquid_threshold: float = 0.5
) -> np.ndarray:
    """Coarsen a binary liquid mask through local area fractions.

    Nearest-neighbour resizing makes a connectivity event depend on the phase
    of one diagnostic pixel.  Exact block averages are used whenever the
    raster dimensions are divisible by ``divisor``; the BOX filter provides
    the same area-average semantics for the general case.  Threshold variation
    is part of the declared topology sensitivity audit, not a hidden tuning
    parameter.
    """
    if divisor < 1:
        raise ValueError("raster divisors must be positive")
    if not 0.0 < liquid_threshold < 1.0:
        raise ValueError("liquid threshold must lie strictly between zero and one")
    if divisor == 1:
        return mask.copy()
    height, width = mask.shape
    if height % divisor == 0 and width % divisor == 0:
        fractions = mask.reshape(
            height // divisor, divisor, width // divisor, divisor
        ).mean(axis=(1, 3))
    else:
        target = (max(2, width // divisor), max(2, height // divisor))
        encoded = Image.fromarray(mask.astype(np.float32), mode="F")
        fractions = np.asarray(
            encoded.resize(target, Image.Resampling.BOX), dtype=float
        )
    return fractions >= liquid_threshold


def _read_diagnostics(path: Path | None) -> dict[str, Any]:
    runs: list[tuple[float, float, float]] = []
    q_runs: list[tuple[float, float, float, float]] = []
    shores: list[tuple[float, float, float]] = []
    corrections: list[list[float]] = []
    reprojections: list[list[float]] = []
    if path is None or not path.exists():
        return {"available": False}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        try:
            if fields and fields[0] == "RUN" and len(fields) >= 5:
                runs.append((float(fields[1]), float(fields[3]), float(fields[4])))
            elif fields and fields[0] == "Q_RUN" and len(fields) >= 7:
                q_runs.append(
                    (float(fields[1]), float(fields[4]), float(fields[5]), float(fields[6]))
                )
            elif fields and fields[0] == "SHORE" and len(fields) >= 6:
                shores.append((float(fields[1]), float(fields[4]), float(fields[5])))
            elif fields and fields[0] == "INIT_RECEIVER_CORRECTION":
                corrections.append([float(value) for value in fields[1:]])
            elif fields and fields[0] == "POSTADAPT_REPROJECT" and len(fields) >= 11:
                reprojections.append([float(value) for value in fields[1:]])
        except ValueError:
            continue
    result: dict[str, Any] = {"available": bool(runs), "run_record_count": len(runs)}
    if runs:
        initial_mass = runs[0][1]
        mass_drift = [(mass - initial_mass) / max(abs(initial_mass), 1.0e-30) for _, mass, _ in runs]
        result.update(
            {
                "initial_mass": initial_mass,
                "final_mass": runs[-1][1],
                "final_relative_mass_drift": mass_drift[-1],
                "maximum_absolute_relative_mass_drift": max(map(abs, mass_drift)),
                "maximum_kinetic_energy": max(record[2] for record in runs),
            }
        )
    if q_runs:
        result.update(
            {
                "maximum_q_lower_bound_violation": max(record[2] for record in q_runs),
                "maximum_q_upper_bound_violation": max(record[3] for record in q_runs),
            }
        )
    if shores:
        result.update(
            {
                "maximum_bed_pressure_magnitude": max(record[1] for record in shores),
                "maximum_near_bed_speed": max(record[2] for record in shores),
            }
        )
    if corrections:
        # INIT_RECEIVER_CORRECTION: t, i, relative error before/after,
        # shifts, correction norm ratio, relative energy change, momenta.
        latest = corrections[-1]
        result["initial_receiver_relative_momentum_error_before"] = latest[2]
        result["initial_receiver_relative_momentum_error_after"] = latest[3]
        result["initial_receiver_correction_norm_ratio"] = latest[6]
        result["initial_receiver_relative_energy_change"] = latest[7]
    if reprojections:
        result.update(
            {
                "post_adaptation_reprojection_count": len(reprojections),
                "maximum_post_projection_face_divergence_rms": max(
                    abs(record[4]) for record in reprojections
                ),
                "maximum_post_projection_face_divergence_linf": max(
                    abs(record[5]) for record in reprojections
                ),
                "maximum_post_projection_relative_momentum_change": max(
                    abs(record[6]) for record in reprojections
                ),
                "maximum_post_projection_relative_kinetic_change": max(
                    abs(record[7]) for record in reprojections
                ),
                "maximum_pre_projection_face_divergence_rms": max(
                    abs(record[8]) for record in reprojections
                ),
                "maximum_pre_projection_face_divergence_linf": max(
                    abs(record[9]) for record in reprojections
                ),
            }
        )
    return result


def _event_for_raster(
    paths: list[Path],
    divisor: int,
    liquid_threshold: float,
    *,
    bottom_x: np.ndarray,
    bottom_z: np.ndarray,
    length: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    minimum_area: float,
    bed_guard_distance: float,
    persistence_frames: int,
    dt: float,
) -> dict[str, Any]:
    records: list[dict[str, float | int]] = []
    for frame, path in enumerate(paths):
        liquid = _downsample(
            liquid_mask(path), divisor, liquid_threshold=liquid_threshold
        )
        record = topology_record_from_liquid(
            liquid,
            bottom_x,
            bottom_z,
            length,
            x_min,
            x_max,
            y_min,
            y_max,
            minimum_pixels=1,
            minimum_area=minimum_area,
            bed_guard_distance=bed_guard_distance,
        )
        record.update({"frame": frame, "continuation_time": frame * dt})
        records.append(record)
    onset = persistent_onset(records, persistence_frames)
    unfiltered_records = [
        {"pocket_count": record["unfiltered_pocket_count"]}
        for record in records
    ]
    unfiltered_onset = persistent_onset(unfiltered_records, persistence_frames)
    maximum_unfiltered_area = max(
        float(record["unfiltered_largest_pocket_area"]) for record in records
    )
    return {
        "raster_divisor": divisor,
        "liquid_fraction_threshold": liquid_threshold,
        "raster_shape": list(
            _downsample(
                liquid_mask(paths[0]),
                divisor,
                liquid_threshold=liquid_threshold,
            ).shape
        ),
        "impact_detected": onset is not None,
        "impact_frame": onset,
        "unfiltered_subcell_closure_frame": unfiltered_onset,
        "unfiltered_subcell_closure_time": (
            unfiltered_onset * dt if unfiltered_onset is not None else None
        ),
        "maximum_unfiltered_pocket_area": maximum_unfiltered_area,
        "maximum_unfiltered_area_relative_to_area_gate": (
            maximum_unfiltered_area / minimum_area if minimum_area > 0.0 else None
        ),
        "last_open_time": (onset - 1) * dt if onset is not None and onset > 0 else None,
        "first_persistent_closed_time": onset * dt if onset is not None else None,
        "event_bracket_width": dt if onset is not None and onset > 0 else None,
        "records": records,
    }


def analyze_case(case: dict[str, Any], common: dict[str, Any]) -> dict[str, Any]:
    directory = Path(case["frame_directory"])
    paths = sorted(directory.glob("vof-*.ppm"))
    if len(paths) < int(common["persistence_frames"]):
        raise ValueError(f"insufficient frames in {directory}")
    handoff_path = Path(case.get("handoff_npz", common["handoff_npz"]))
    with np.load(handoff_path) as loaded:
        bottom_x = np.asarray(loaded["bottom_x"], dtype=float)
        bottom_z = np.asarray(loaded["bottom_z"], dtype=float)
        grid_x = np.asarray(loaded["grid_x"], dtype=float)
        length = float(grid_x[-1] - grid_x[0] + grid_x[1] - grid_x[0])
        source_time = float(loaded["time"])
    level = int(case["level"])
    delta = length / (2**level)
    minimum_area = float(common["minimum_pocket_cells"]) * delta**2
    bed_guard_distance = float(common["bed_guard_cells"]) * delta
    raster_thresholds = [
        float(value) for value in common.get("raster_liquid_thresholds", [0.5])
    ]
    raster_specs = [
        (int(divisor), threshold)
        for divisor in common["raster_divisors"]
        for threshold in (raster_thresholds if int(divisor) > 1 else [0.5])
    ]
    raster_results = [
        _event_for_raster(
            paths,
            divisor,
            threshold,
            bottom_x=bottom_x,
            bottom_z=bottom_z,
            length=length,
            x_min=float(common["x_min"]),
            x_max=float(common["x_max"]),
            y_min=float(common["y_min"]),
            y_max=float(common["y_max"]),
            minimum_area=minimum_area,
            bed_guard_distance=bed_guard_distance,
            persistence_frames=int(common["persistence_frames"]),
            dt=float(common["simulation_dt"]),
        )
        for divisor, threshold in raster_specs
    ]
    event_times = [
        result["first_persistent_closed_time"]
        for result in raster_results
        if result["impact_detected"]
    ]
    raw_event_times = [
        result["unfiltered_subcell_closure_time"]
        for result in raster_results
        if result["unfiltered_subcell_closure_time"] is not None
    ]
    all_rasters_detect = len(event_times) == len(raster_results)
    raster_spread = max(event_times) - min(event_times) if event_times else None
    primary = raster_results[0]
    event_index = primary["impact_frame"]
    lag_frames = round(float(common["pocket_area_lag"]) / float(common["simulation_dt"]))
    lag_area = None
    if event_index is not None:
        lag_index = min(len(primary["records"]) - 1, event_index + lag_frames)
        lag_area = primary["records"][lag_index]["total_pocket_area"]
    diagnostics_path = case.get("diagnostics")
    diagnostics = _read_diagnostics(Path(diagnostics_path) if diagnostics_path else None)
    mass_ok = bool(diagnostics.get("available")) and (
        diagnostics.get("maximum_absolute_relative_mass_drift", math.inf)
        <= float(common["mass_drift_tolerance"])
    )
    q_bounds_available = (
        "maximum_q_lower_bound_violation" in diagnostics
        and "maximum_q_upper_bound_violation" in diagnostics
    )
    bounds_ok = (q_bounds_available or not bool(common["require_q_bounds"])) and (
        diagnostics.get("maximum_q_lower_bound_violation", 0.0)
        <= float(common["bound_tolerance"])
        and diagnostics.get("maximum_q_upper_bound_violation", 0.0)
        <= float(common["bound_tolerance"])
    )
    raster_ok = all_rasters_detect and raster_spread is not None and (
        raster_spread <= float(common["raster_time_tolerance"])
    )
    compatibility_available = (
        "maximum_post_projection_face_divergence_linf" in diagnostics
        and "maximum_post_projection_relative_momentum_change" in diagnostics
        and "maximum_post_projection_relative_kinetic_change" in diagnostics
    )
    compatibility_required = bool(
        common.get("require_postadapt_compatibility", False)
    )
    compatibility_ok = (
        compatibility_available or not compatibility_required
    ) and (
        diagnostics.get("maximum_post_projection_face_divergence_linf", 0.0)
        <= float(common.get("post_projection_divergence_tolerance", math.inf))
        and diagnostics.get(
            "maximum_post_projection_relative_momentum_change", 0.0
        ) <= float(common.get("post_projection_correction_tolerance", math.inf))
        and diagnostics.get(
            "maximum_post_projection_relative_kinetic_change", 0.0
        ) <= float(common.get("post_projection_correction_tolerance", math.inf))
    )
    return {
        "label": case["label"],
        "level": level,
        "delta": delta,
        "frame_count": len(paths),
        "source_bie_time": source_time,
        "minimum_pocket_area": minimum_area,
        "bed_guard_distance": bed_guard_distance,
        "primary_impact_continuation_time": primary["first_persistent_closed_time"],
        "primary_impact_total_time": (
            source_time + primary["first_persistent_closed_time"]
            if primary["first_persistent_closed_time"] is not None else None
        ),
        "pocket_area_at_fixed_lag": lag_area,
        "maximum_unfiltered_pocket_area": primary[
            "maximum_unfiltered_pocket_area"
        ],
        "maximum_unfiltered_pocket_area_over_delta_squared": primary[
            "maximum_unfiltered_pocket_area"
        ] / delta**2,
        "raster_event_time_spread": raster_spread,
        "raw_raster_detection_count": len(raw_event_times),
        "raw_raster_probe_count": len(raster_results),
        "raw_raster_event_time_spread": (
            max(raw_event_times) - min(raw_event_times)
            if raw_event_times else None
        ),
        "raster_results": raster_results,
        "diagnostics": diagnostics,
        "gates": {
            "persistent_topology": primary["impact_detected"],
            "raster_sensitivity": raster_ok,
            "mass": mass_ok,
            "q_bounds": bounds_ok,
            "postadapt_compatibility": compatibility_ok,
        },
        "case_accepted": bool(
            primary["impact_detected"]
            and raster_ok
            and mass_ok
            and bounds_ok
            and compatibility_ok
        ),
    }


def _grid_audit(cases: list[dict[str, Any]], common: dict[str, Any]) -> dict[str, Any]:
    ordered = sorted(cases, key=lambda case: case["delta"], reverse=True)
    times = [case["primary_impact_continuation_time"] for case in ordered]
    areas = [case["pocket_area_at_fixed_lag"] for case in ordered]
    detected = all(time is not None for time in times)
    time_spread = max(times) - min(times) if detected else None
    observed_order = None
    if len(times) >= 3 and detected:
        coarse_difference = abs(times[-3] - times[-2])
        fine_difference = abs(times[-2] - times[-1])
        if coarse_difference > 0.0 and fine_difference > 0.0:
            observed_order = math.log(coarse_difference / fine_difference, 2.0)
    finest_area_relative_difference = None
    if len(areas) >= 2 and areas[-1] is not None and areas[-2] is not None:
        finest_area_relative_difference = abs(areas[-1] - areas[-2]) / max(
            abs(areas[-1]), abs(areas[-2]), 1.0e-30
        )
    enough_levels = len({case["level"] for case in cases}) >= 3
    time_ok = detected and time_spread is not None and (
        time_spread <= float(common["grid_time_tolerance"])
    )
    area_ok = finest_area_relative_difference is not None and (
        finest_area_relative_difference <= float(common["pocket_area_relative_tolerance"])
    )
    return {
        "levels": [case["level"] for case in ordered],
        "impact_times": times,
        "event_time_spread": time_spread,
        "observed_event_time_order": observed_order,
        "pocket_areas_at_fixed_lag": areas,
        "finest_pair_pocket_area_relative_difference": finest_area_relative_difference,
        "gates": {
            "at_least_three_levels": enough_levels,
            "event_detected_at_all_levels": detected,
            "event_time_convergence": time_ok,
            "pocket_area_convergence": area_ok,
        },
        "accepted": bool(enough_levels and time_ok and area_ok),
    }


def _plot(summary: dict[str, Any], output: Path) -> None:
    cases = sorted(summary["cases"], key=lambda case: case["delta"], reverse=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    levels = [str(case["level"]) for case in cases]
    probe_keys = []
    for case in cases:
        for record in case["raster_results"]:
            key = (
                int(record["raster_divisor"]),
                float(record["liquid_fraction_threshold"]),
            )
            if key not in probe_keys:
                probe_keys.append(key)
    probe_labels = [f"/{divisor}\n{threshold:.2g}" for divisor, threshold in probe_keys]
    for case in cases:
        lookup = {
            (
                int(record["raster_divisor"]),
                float(record["liquid_fraction_threshold"]),
            ): record["unfiltered_subcell_closure_time"]
            for record in case["raster_results"]
        }
        raw_times = [
            np.nan if lookup.get(key) is None else float(lookup[key])
            for key in probe_keys
        ]
        axes[0].plot(
            range(len(probe_keys)), raw_times, "o-", linewidth=1.8,
            label=f"level {case['level']}",
        )
    axes[0].set_xticks(range(len(probe_keys)), probe_labels)
    axes[0].set_xlabel("raster divisor / liquid threshold")
    axes[0].set_ylabel("raw persistent closure time")
    axes[0].set_title("Graph/raster sensitivity")
    axes[0].legend(frameon=False)
    peak_areas = [
        case["maximum_unfiltered_pocket_area_over_delta_squared"]
        for case in cases
    ]
    axes[1].bar(levels, peak_areas, color="#0f766e")
    axes[1].axhline(1.0, color="#991b1b", linestyle="--", linewidth=1.5,
                    label=r"one-cell gate $\Delta^2$")
    axes[1].set_xlabel("maximum quadtree level")
    axes[1].set_ylabel(r"native-raster peak pocket area / $\Delta^2$")
    axes[1].set_title("Physical topology scale")
    axes[1].legend(frameon=False)
    thresholds = summary["thresholds"]
    metrics = [
        ("mass", "maximum_absolute_relative_mass_drift",
         float(thresholds["mass_drift_tolerance"])),
        ("q bound", "maximum_q_upper_bound_violation",
         float(thresholds["bound_tolerance"])),
    ]
    if "post_projection_divergence_tolerance" in thresholds:
        metrics.extend(
            [
                ("div", "maximum_post_projection_face_divergence_linf",
                 float(thresholds["post_projection_divergence_tolerance"])),
                ("momentum", "maximum_post_projection_relative_momentum_change",
                 float(thresholds["post_projection_correction_tolerance"])),
                ("energy", "maximum_post_projection_relative_kinetic_change",
                 float(thresholds["post_projection_correction_tolerance"])),
            ]
        )
    positions = np.arange(len(metrics), dtype=float)
    width = 0.78 / max(len(cases), 1)
    for index, case in enumerate(cases):
        normalized = [
            max(float(case["diagnostics"].get(key, np.nan)) / gate, 1.0e-16)
            for _, key, gate in metrics
        ]
        axes[2].bar(
            positions + (index - (len(cases) - 1) / 2) * width,
            normalized, width=width, label=f"level {case['level']}",
        )
    axes[2].axhline(1.0, color="#991b1b", linestyle="--", linewidth=1.5,
                    label="acceptance boundary")
    axes[2].set_yscale("log")
    axes[2].set_xticks(positions, [label for label, _, _ in metrics], rotation=20)
    axes[2].set_ylabel("observed / declared gate")
    axes[2].set_title("Finite-volume/MAC gates")
    axes[2].legend(
        frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
    )
    fig.suptitle(
        "Three-level invariant and topology results",
        fontsize=13, fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_plot", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    common = manifest["thresholds"] | manifest["geometry"] | {
        "handoff_npz": manifest["handoff_npz"]
    }
    cases = [analyze_case(case, common) for case in manifest["cases"]]
    grid = _grid_audit(cases, common)
    accepted = all(case["case_accepted"] for case in cases) and grid["accepted"]
    summary = {
        "schema": "persistent-impact-claim-audit-v1",
        "evidence_semantics": {
            "finite_volume": "mass and q-bound observations from solver diagnostics",
            "graph": "persistent exterior-connectivity change of the gas phase",
            "geometric": "event-time and pocket-area changes across receiver levels",
            "theorem_status": "none; all reported evidence is computational",
        },
        "thresholds": manifest["thresholds"],
        "geometry": manifest["geometry"],
        "cases": cases,
        "grid_audit": grid,
        "impact_claim_accepted": accepted,
        "publication_statement": (
            "accepted for the scoped numerical impact claim"
            if accepted
            else "not accepted: retain only as qualitative or negative evidence"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _plot(summary, args.output_plot)
    print(json.dumps({"impact_claim_accepted": accepted, "grid_audit": grid}, indent=2))


if __name__ == "__main__":
    main()
