"""Publication-oriented metrics and stop criteria for parametric wave archives."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from animate_lagrangian_bie import spectral_curve_resample
from topographic_wave_solver import reference_energy
from topography_bie import TopographyBIE
from vof_handoff import interface_self_intersects, minimum_nonlocal_marker_distance


Array = np.ndarray


def _spectral_derivative(values: Array, order: int) -> Array:
    n = len(values)
    modes = np.fft.fftfreq(n, d=1.0 / n)
    return np.fft.ifft((1j * modes) ** order * np.fft.fft(values)).real


def _geometry(x: Array, z: Array, length: float) -> tuple[Array, Array, Array]:
    n = len(x)
    base = length * np.arange(n) / n
    x_alpha = length / (2.0 * math.pi) + _spectral_derivative(x - base, 1)
    z_alpha = _spectral_derivative(z, 1)
    metric = np.sqrt(x_alpha**2 + z_alpha**2)
    return x_alpha, z_alpha, metric


def _crest_series(data: dict[str, Array]) -> tuple[Array, Array]:
    length = float(data["length"])
    crest_x, crest_z = [], []
    for x, z in zip(data["x"], data["z"]):
        dense_x, dense_z = spectral_curve_resample(x, z, length, refinement=16)
        index = int(np.argmax(dense_z))
        crest_x.append(float(dense_x[index]))
        crest_z.append(float(dense_z[index]))
    phase = np.unwrap(2.0 * math.pi * np.asarray(crest_x) / length)
    return phase * length / (2.0 * math.pi), np.asarray(crest_z)


def _local_linear_speed(time: Array, position: Array, half_window: int = 4) -> Array:
    speed = np.empty_like(position)
    for index in range(len(position)):
        start = max(0, index - half_window)
        stop = min(len(position), index + half_window + 1)
        if stop - start < 2:
            speed[index] = 0.0
        else:
            speed[index] = np.polyfit(time[start:stop], position[start:stop], 1)[0]
    return speed


def _surface_velocity_at_crest(
    data: dict[str, Array], index: int, compute_condition_number: bool = True
) -> tuple[float, float, float, float]:
    length = float(data["length"])
    background = float(data.get("background_current", np.asarray(0.0)))
    bie = TopographyBIE(
        data["x"][index],
        data["z"][index],
        data["bottom_x"],
        data["bottom_z"],
        length,
    )
    bottom_normal_x = bie.bottom.z_alpha / bie.bottom.metric
    bottom_flux = -background * bottom_normal_x
    result = bie.solve(
        data["potential"][index],
        bottom_flux,
        compute_condition_number=compute_condition_number,
    )
    tangent_x = bie.surface.x_alpha / bie.surface.metric
    tangent_z = bie.surface.z_alpha / bie.surface.metric
    normal_x = -tangent_z
    normal_z = tangent_x
    tangential = bie.surface_derivative(data["potential"][index]) + background * tangent_x
    normal = result.surface_normal_derivative + background * normal_x
    velocity_x = tangential * tangent_x + normal * normal_x
    velocity_z = tangential * tangent_z + normal * normal_z
    crest = int(np.argmax(data["z"][index]))
    total_surface_flux = float(bie.dalpha * np.sum(normal * bie.surface.metric))
    return (
        float(velocity_x[crest]),
        float(velocity_z[crest]),
        total_surface_flux,
        result.condition_number,
    )


def _event_index(values: Array, predicate) -> int | None:
    indices = np.where(predicate(values))[0]
    return int(indices[0]) if len(indices) else None


def _local_depth(data: dict[str, Array], x_value: float) -> float:
    length = float(data["length"])
    wrapped = x_value % length
    bottom_x = np.asarray(data["bottom_x"])
    bottom_z = np.asarray(data["bottom_z"])
    extended_x = np.concatenate((bottom_x, [bottom_x[0] + length]))
    extended_z = np.concatenate((bottom_z, [bottom_z[0]]))
    bed = float(np.interp(wrapped, extended_x, extended_z))
    return -bed


def analyze_archive(
    input_path: Path, compute_condition_number: bool = True
) -> dict[str, object]:
    with np.load(input_path) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    length = float(data["length"])
    time = data["time"]
    baseline_data = data
    visited = {input_path.resolve()}
    for _ in range(6):
        if "restart_source" not in baseline_data:
            break
        candidate = Path(str(baseline_data["restart_source"]))
        if not candidate.is_absolute():
            direct = Path.cwd() / candidate
            candidate = direct if direct.exists() else input_path.parent / candidate
        if not candidate.exists() or candidate.resolve() in visited:
            break
        visited.add(candidate.resolve())
        with np.load(candidate) as loaded:
            baseline_data = {key: loaded[key] for key in loaded.files}
    baseline_energy = float(
        baseline_data.get("global_initial_energy", baseline_data["energy"][0])
    )
    baseline_volume = float(
        baseline_data.get("global_initial_volume", baseline_data["volume"][0])
    )
    crest_x, crest_z = _crest_series(data)
    crest_speed = _local_linear_speed(time, crest_x)
    breaking_index = _event_index(data["min_x_alpha"], lambda values: values <= 0.0)
    breaking_output_index = None if breaking_index is None else max(breaking_index - 1, 0)
    # A strictly single-valued graph cannot self-intersect.  The O(N^2)
    # segment test is needed only after x_alpha has become non-positive.
    self_intersections = np.zeros(len(time), dtype=bool)
    overhanging_frames = np.where(data["min_x_alpha"] <= 0.0)[0]
    for frame in overhanging_frames:
        self_intersections[frame] = interface_self_intersects(
            data["x"][frame], data["z"][frame], length
        )
    intersection_index = _event_index(self_intersections, lambda values: values)
    impact_output_index = None if intersection_index is None else max(intersection_index - 1, 0)
    toe = float(data.get("reef_toe", np.asarray(float("nan"))))
    toe_index = None
    if math.isfinite(toe):
        toe_index = _event_index(crest_x, lambda values: values >= toe)

    initial_x_alpha, _, _ = _geometry(
        baseline_data["x"][0], baseline_data["z"][0], length
    )
    wave_volume = float(
        2.0 * math.pi / baseline_data["x"].shape[1]
        * np.sum(baseline_data["z"][0] * initial_x_alpha)
    )
    energy_reference = float(
        data.get(
            "energy_reference",
            np.asarray(
                reference_energy(
                    data["bottom_x"],
                    data["bottom_z"],
                    length,
                    float(data["gravity"]),
                    float(data.get("background_current", np.asarray(0.0))),
                )
            ),
        )
    )
    energy_scale = max(
        abs(baseline_energy - energy_reference),
        np.finfo(float).eps,
    )
    volume_scale = max(abs(wave_volume), np.finfo(float).eps)
    event_index = breaking_output_index if breaking_output_index is not None else len(time) - 1
    valid_slice = slice(0, event_index + 1)
    relative_energy = (data["energy"] - baseline_energy) / energy_scale
    normalized_volume = (data["volume"] - baseline_volume) / volume_scale
    crest_u, crest_w, flux_defect, condition_number = _surface_velocity_at_crest(
        data, event_index, compute_condition_number
    )
    front_slopes, rear_slopes = [], []
    for frame_x, frame_z, peak_x, peak_z in zip(data["x"], data["z"], crest_x, crest_z):
        x_alpha, z_alpha, _ = _geometry(frame_x, frame_z, length)
        active = frame_z > max(0.01 * peak_z, 1.0e-8)
        single_valued = x_alpha > 1.0e-10
        slope = np.full_like(x_alpha, np.nan)
        slope[single_valued] = z_alpha[single_valued] / x_alpha[single_valued]
        front = active & (frame_x >= peak_x) & single_valued
        rear = active & (frame_x <= peak_x) & single_valued
        front_slopes.append(float(np.nanmin(slope[front])) if np.any(front) else float("nan"))
        rear_slopes.append(float(np.nanmax(slope[rear])) if np.any(rear) else float("nan"))
    nonlocal_distance = minimum_nonlocal_marker_distance(
        data["x"][event_index], data["z"][event_index]
    )
    local_depth = _local_depth(data, crest_x[event_index])
    breaking_height = float(crest_z[event_index])
    precursor = (
        float(crest_u / crest_speed[event_index])
        if abs(float(crest_speed[event_index])) > 1.0e-12
        else float("nan")
    )
    summary: dict[str, object] = {
        "source": str(input_path),
        "n": int(data["x"].shape[1]),
        "initial_condition": str(data.get("initial_condition", np.asarray("unknown"))),
        "bathymetry": str(data.get("bathymetry", np.asarray("unknown"))),
        "final_time": float(time[-1]),
        "toe_crossing_time": None if toe_index is None else float(time[toe_index]),
        "breaking_detected": breaking_index is not None,
        "breaking_time_bracket": (
            None
            if breaking_index is None
            else [float(time[breaking_index - 1]), float(time[breaking_index])]
        ),
        "impact_detected_by_self_intersection": intersection_index is not None,
        "impact_time_bracket": (
            None
            if intersection_index is None
            else [float(time[intersection_index - 1]), float(time[intersection_index])]
        ),
        "event_output_index": event_index,
        "event_time": float(time[event_index]),
        "crest_x": float(crest_x[event_index]),
        "crest_height": breaking_height,
        "local_depth": local_depth,
        "breaker_depth_index": breaking_height / local_depth,
        "front_slope": front_slopes[event_index],
        "rear_slope": rear_slopes[event_index],
        "crest_horizontal_velocity": crest_u,
        "crest_vertical_velocity": crest_w,
        "crest_speed": float(crest_speed[event_index]),
        "breaking_precursor_B": precursor,
        "surface_flux_defect": flux_defect,
        "bie_condition_number": condition_number,
        "minimum_nonlocal_marker_distance": nonlocal_distance,
        "maximum_relative_energy_drift": float(
            np.max(np.abs(relative_energy[valid_slice]))
        ),
        "maximum_wave_volume_normalized_drift": float(
            np.max(np.abs(normalized_volume[valid_slice]))
        ),
        "maximum_marker_spacing_cv": float(
            np.max(data["marker_cv"][valid_slice])
        ),
        "maximum_bie_residual": float(
            np.max(data["bie_residual"][valid_slice])
        ),
        "maximum_surface_flux_defect": float(
            np.max(np.abs(data["surface_flux_defect"][valid_slice]))
        ),
        "global_maximum_relative_energy_drift": float(
            np.nanmax(np.abs(relative_energy))
        ),
        "global_maximum_wave_volume_normalized_drift": float(
            np.nanmax(np.abs(normalized_volume))
        ),
    }
    p0_pass = (
        summary["maximum_surface_flux_defect"] < 1.0e-5
        and summary["maximum_bie_residual"] < 1.0e-10
        and summary["maximum_relative_energy_drift"] < 2.0e-3
        and summary["maximum_wave_volume_normalized_drift"] < 2.0e-3
        and summary["maximum_marker_spacing_cv"] < 0.1
    )
    summary["gate_P0_trajectory"] = "pass" if p0_pass else "fail"
    summary["publication_status"] = (
        "requires_grid_time_convergence"
        if p0_pass and breaking_index is not None
        else "validated_nonbreaking_trajectory"
        if p0_pass
        else "fails_correctness_gate"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-condition-number", action="store_true")
    args = parser.parse_args()
    summary = analyze_archive(
        args.input, compute_condition_number=not args.skip_condition_number
    )
    output = args.output or args.input.with_name(args.input.stem + "_publication.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
