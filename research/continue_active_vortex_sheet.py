"""Continue a pre-contact breaker with an active adaptive vortex-sheet state.

This driver is deliberately conservative about its claims.  The evolved
unknown is the integrated panel circulation, not a marker-only rendering.
Every accepted r/p-remesh is fed back into the next Euler--BIE step.  The two
continuous invariants which are most sensitive to interpolation, volume and
energy, are restored by documented scalar projections; Kelvin circulation is
preserved algebraically by the cumulative-circulation remap.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time as wall_clock

import numpy as np

from active_vortex_sheet import (
    ActiveVortexSheetState,
    panel_circulation_to_potential,
    potential_to_panel_circulation,
    step_active_vortex_sheet_rk4,
)
from adaptive_vortex_mesh import (
    project_vortex_sheet_invariants,
    remesh_active_vortex_sheet,
    resample_periodic_bottom,
    suggest_vortex_count,
    vortex_mesh_monitor,
)
from impact_detection import periodic_impact_diagnostics
from topographic_wave_solver import diagnostics, minimum_horizontal_mapping


Array = np.ndarray


def _scalar(data: dict[str, Array], key: str, default: float | str) -> float | str:
    if key not in data:
        return default
    return np.asarray(data[key]).item()


def _pad(vectors: list[Array], maximum_count: int) -> Array:
    result = np.full((len(vectors), maximum_count), np.nan)
    for row, values in enumerate(vectors):
        result[row, : len(values)] = values
    return result


def _state_values(
    state: ActiveVortexSheetState,
    bottom_x: Array,
    bottom_z: Array,
    length: float,
    gravity: float,
    background_current: float,
) -> tuple[Array, tuple[float, float, float, float, float]]:
    potential = panel_circulation_to_potential(
        state.x,
        state.panel_circulation,
        length,
        background_current,
        state.potential_mean,
    )
    values = diagnostics(
        state.x,
        state.z,
        potential,
        bottom_x,
        bottom_z,
        length,
        gravity,
        background_current,
    )
    return potential, values


def _geometric_shape_metrics(x: Array, z: Array, length: float) -> tuple[float, float]:
    """Return min(x_alpha) and marker-metric CV without assembling a BIE."""
    n = len(x)
    modes = np.fft.fftfreq(n, d=1.0 / n)
    base = length * np.arange(n) / n
    x_alpha = length / (2.0 * math.pi) + np.fft.ifft(
        1j * modes * np.fft.fft(x - base)
    ).real
    z_alpha = np.fft.ifft(1j * modes * np.fft.fft(z)).real
    metric = np.hypot(x_alpha, z_alpha)
    return float(np.min(x_alpha)), float(np.std(metric) / np.mean(metric))


def continue_active(
    source_path: Path,
    output_prefix: Path,
    *,
    start_time: float,
    final_time: float,
    dt: float,
    minimum_dt: float,
    initial_count: int,
    minimum_count: int,
    maximum_count: int,
    adapt_every: int,
    remesh_cv_gate: float,
    maximum_projection_correction: float,
    maximum_cumulative_projection: float,
    maximum_global_energy_drift: float,
    maximum_flux_defect: float,
    contact_distance_factor: float,
) -> dict[str, object]:
    with np.load(source_path) as loaded:
        source = {key: loaded[key] for key in loaded.files}
    source_times = np.asarray(source["time"], dtype=float)
    source_index = int(np.argmin(np.abs(source_times - start_time)))
    selected_time = float(source_times[source_index])
    if final_time <= selected_time:
        raise ValueError("final_time must be later than the selected restart time")
    if dt <= 0.0 or minimum_dt <= 0.0 or minimum_dt > dt or adapt_every < 1:
        raise ValueError("require 0 < minimum_dt <= dt and positive adapt_every")

    length = float(_scalar(source, "length", 0.0))
    gravity = float(_scalar(source, "gravity", 1.0))
    background = float(_scalar(source, "background_current", 0.0))
    gauge = str(_scalar(source, "tangential_gauge", "theta_s"))
    source_count = (
        int(np.asarray(source["count"])[source_index])
        if "count" in source
        else int(np.asarray(source["x"]).shape[1])
    )

    def selected_vector(key: str) -> Array:
        values = np.asarray(source[key], dtype=float)
        selected = values[source_index] if values.ndim == 2 else values
        return np.asarray(selected[:source_count], dtype=float)

    source_bottom_x = selected_vector("bottom_x")
    source_bottom_z = selected_vector("bottom_z")
    source_x = selected_vector("x")
    source_z = selected_vector("z")
    source_potential = selected_vector("potential")
    global_initial_energy = float(
        _scalar(source, "global_initial_energy", float(source["energy"][0]))
    )
    reference_energy = float(
        _scalar(source, "energy_reference", global_initial_energy - 1.0)
    )
    energy_scale = max(
        abs(global_initial_energy - reference_energy), np.finfo(float).eps
    )
    target_energy = float(source["energy"][source_index])
    if "global_target_volume" in source:
        target_volume = float(np.asarray(source["global_target_volume"]).item())
    elif "target_volume" in source:
        target_volume = float(np.asarray(source["target_volume"]).item())
    else:
        target_volume = float(source["volume"][source_index])
    total_circulation_target = background * length

    source_state = ActiveVortexSheetState(
        source_x,
        source_z,
        potential_to_panel_circulation(
            source_x, source_potential, length, background
        ),
        float(np.mean(source_potential)),
    )
    state, initial_remesh = remesh_active_vortex_sheet(
        source_state,
        length,
        background,
        target_count=initial_count,
    )
    bottom_x, bottom_z = resample_periodic_bottom(
        source_bottom_x, source_bottom_z, length, initial_count
    )
    state, initial_projection = project_vortex_sheet_invariants(
        state,
        bottom_x,
        bottom_z,
        length,
        gravity,
        background,
        target_volume,
        target_energy,
        energy_scale,
    )
    initial_global_drift = abs(target_energy - global_initial_energy) / energy_scale
    initial_gates = {
        "circulation_defect_le_1e-12": abs(
            state.total_circulation - total_circulation_target
        )
        <= 1.0e-12,
        "projection_correction_le_gate": initial_projection.relative_circulation_correction
        <= maximum_projection_correction,
        "global_energy_drift_le_gate": initial_global_drift
        <= maximum_global_energy_drift,
        "relative_energy_projection_error_le_1e-10": initial_projection.relative_energy_error
        <= 1.0e-10,
        "relative_volume_error_le_5e-5": initial_projection.relative_volume_error
        <= 5.0e-5,
        "raw_flux_defect_le_gate": initial_projection.raw_dno_flux_defect
        <= maximum_flux_defect,
    }
    if not all(initial_gates.values()):
        raise RuntimeError(f"restart projection rejected: {initial_gates}")

    history: dict[str, list[object]] = {
        "time": [],
        "x": [],
        "z": [],
        "panel_circulation": [],
        "potential": [],
        "bottom_x": [],
        "bottom_z": [],
        "count": [],
        "volume": [],
        "energy": [],
        "min_x_alpha": [],
        "marker_cv": [],
        "surface_flux_defect": [],
        "maximum_bie_residual": [],
        "circulation_derivative_defect": [],
        "reconstruction_relative_error": [],
        "circulation_total_defect": [],
        "projection_relative_correction": [],
        "projection_energy_input_drift": [],
        "projection_volume_shift": [],
        "cumulative_projection_correction": [],
        "monitor_mass_cv": [],
        "maximum_turning_angle": [],
        "minimum_nonlocal_gap_ratio": [],
        "maximum_circulation_fraction": [],
        "impact_distance": [],
        "normalized_impact_distance": [],
        "self_intersection": [],
        "accepted_dt": [],
        "step_retries": [],
    }
    adaptation_events: list[dict[str, object]] = []
    cumulative_projection_offset = (
        float(np.asarray(source["cumulative_projection_correction"])[source_index])
        if "cumulative_projection_correction" in source
        else 0.0
    )
    cumulative_projection = (
        cumulative_projection_offset
        + initial_projection.relative_circulation_correction
    )
    first_vertical_time = math.nan
    contact_time = math.nan
    termination_reason = "final_time"
    start_wall = wall_clock.perf_counter()

    def record(
        current_time: float,
        step_bie_residual: float,
        derivative_defect: float,
        reconstruction_error: float,
        projection_correction: float,
        projection_energy_input_drift: float,
        projection_volume_shift: float,
        projected_volume: float,
        projected_energy: float,
        projected_flux: float,
        accepted_dt: float,
        step_retries: int,
    ) -> tuple[float, bool]:
        nonlocal first_vertical_time, contact_time
        potential = panel_circulation_to_potential(
            state.x,
            state.panel_circulation,
            length,
            background,
            state.potential_mean,
        )
        horizontal_mapping, marker_cv = _geometric_shape_metrics(
            state.x, state.z, length
        )
        monitor = vortex_mesh_monitor(state, length)
        monitor_mass = 0.5 * (
            monitor.value + np.roll(monitor.value, -1)
        ) * monitor.panel_length
        monitor_cv = float(np.std(monitor_mass) / np.mean(monitor_mass))
        if horizontal_mapping <= 0.0 and not math.isfinite(first_vertical_time):
            first_vertical_time = current_time
        impact = periodic_impact_diagnostics(
            state.x,
            state.z,
            length,
            minimum_arc_separation_panels=12.0,
        )
        contact = impact.self_intersection or (
            horizontal_mapping <= 0.0
            and impact.normalized_distance <= contact_distance_factor
        )
        if contact and not math.isfinite(contact_time):
            contact_time = current_time
        history["time"].append(current_time)
        history["x"].append(state.x.copy())
        history["z"].append(state.z.copy())
        history["panel_circulation"].append(state.panel_circulation.copy())
        history["potential"].append(potential.copy())
        history["bottom_x"].append(bottom_x.copy())
        history["bottom_z"].append(bottom_z.copy())
        history["count"].append(len(state.x))
        history["volume"].append(projected_volume)
        history["energy"].append(projected_energy)
        history["min_x_alpha"].append(horizontal_mapping)
        history["marker_cv"].append(marker_cv)
        history["surface_flux_defect"].append(projected_flux)
        history["maximum_bie_residual"].append(step_bie_residual)
        history["circulation_derivative_defect"].append(derivative_defect)
        history["reconstruction_relative_error"].append(reconstruction_error)
        history["circulation_total_defect"].append(
            state.total_circulation - total_circulation_target
        )
        history["projection_relative_correction"].append(projection_correction)
        history["projection_energy_input_drift"].append(
            projection_energy_input_drift
        )
        history["projection_volume_shift"].append(projection_volume_shift)
        history["cumulative_projection_correction"].append(cumulative_projection)
        history["monitor_mass_cv"].append(monitor_cv)
        history["maximum_turning_angle"].append(monitor.maximum_turning_angle)
        history["minimum_nonlocal_gap_ratio"].append(
            monitor.minimum_nonlocal_gap_ratio
        )
        history["maximum_circulation_fraction"].append(
            monitor.maximum_circulation_fraction
        )
        history["impact_distance"].append(impact.minimum_distance)
        history["normalized_impact_distance"].append(impact.normalized_distance)
        history["self_intersection"].append(impact.self_intersection)
        history["accepted_dt"].append(accepted_dt)
        history["step_retries"].append(step_retries)
        return horizontal_mapping, contact

    record(
        selected_time,
        0.0,
        0.0,
        0.0,
        initial_projection.relative_circulation_correction,
        (initial_projection.energy_before - target_energy) / energy_scale,
        initial_projection.volume_shift,
        target_volume,
        target_energy,
        initial_projection.raw_dno_flux_defect,
        0.0,
        0,
    )
    current_time = selected_time
    step_index = 0
    while current_time < final_time - 1.0e-10 * dt:
        actual_dt = min(dt, final_time - current_time)
        retry_count = 0
        failed_step = False
        while True:
            step_error: Exception | None = None
            try:
                candidate, step_diagnostics = step_active_vortex_sheet_rk4(
                    state,
                    bottom_x,
                    bottom_z,
                    length,
                    gravity,
                    actual_dt,
                    background,
                    gauge,
                )
                candidate, projection = project_vortex_sheet_invariants(
                    candidate,
                    bottom_x,
                    bottom_z,
                    length,
                    gravity,
                    background,
                    target_volume,
                    target_energy,
                    energy_scale,
                )
                candidate_cumulative = cumulative_projection + (
                    projection.relative_circulation_correction
                )
                step_gates = {
                    "projection": projection.relative_circulation_correction
                    <= maximum_projection_correction,
                    "cumulative_projection": candidate_cumulative
                    <= maximum_cumulative_projection,
                    "energy": projection.relative_energy_error <= 1.0e-10,
                    "volume": projection.relative_volume_error <= 5.0e-5,
                    "flux": projection.raw_dno_flux_defect <= maximum_flux_defect,
                    "circulation": abs(
                        candidate.total_circulation - total_circulation_target
                    )
                    <= 1.0e-12,
                    "bie": step_diagnostics.maximum_bie_residual <= 1.0e-8,
                    "reconstruction": step_diagnostics.maximum_reconstruction_relative_error
                    <= 1.0e-10,
                }
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                step_error = error
                step_gates = {}
            if step_error is None and all(step_gates.values()):
                break
            next_dt = 0.5 * actual_dt
            if next_dt >= minimum_dt * (1.0 - 1.0e-12):
                actual_dt = next_dt
                retry_count += 1
                continue
            if step_error is not None:
                termination_reason = (
                    f"minimum_dt_step_failure:{type(step_error).__name__}:"
                    f"{step_error}"
                )
            else:
                failed = ",".join(
                    key for key, value in step_gates.items() if not value
                )
                termination_reason = f"minimum_dt_acceptance_gate_failure:{failed}"
            failed_step = True
            break
        if failed_step:
            break
        step_index += 1
        state = candidate
        cumulative_projection = candidate_cumulative
        current_time += actual_dt
        state_projection = projection

        spatial_quality_trigger = (
            state_projection.raw_dno_flux_defect > 0.5 * maximum_flux_defect
        )
        if step_index % adapt_every == 0 or spatial_quality_trigger:
            monitor = vortex_mesh_monitor(state, length)
            current_min_x_alpha, _ = _geometric_shape_metrics(
                state.x, state.z, length
            )
            vertical_front_trigger = current_min_x_alpha < 0.25
            panel_mass = 0.5 * (
                monitor.value + np.roll(monitor.value, -1)
            ) * monitor.panel_length
            cv_before = float(np.std(panel_mass) / np.mean(panel_mass))
            decision = suggest_vortex_count(
                monitor,
                len(state.x),
                minimum_count,
                maximum_count,
            )
            should_remesh = decision.target_count != len(state.x) or (
                cv_before > remesh_cv_gate
            ) or vertical_front_trigger or spatial_quality_trigger
            if should_remesh:
                event: dict[str, object] = {
                    "time": current_time,
                    "decision": asdict(decision),
                    "monitor_cv_before": cv_before,
                    "vertical_front_trigger": vertical_front_trigger,
                    "spatial_quality_trigger": spatial_quality_trigger,
                }
                try:
                    remeshed, remesh_diagnostics = remesh_active_vortex_sheet(
                        state,
                        length,
                        background,
                        target_count=decision.target_count,
                    )
                    if decision.target_count != len(state.x):
                        candidate_bottom_x, candidate_bottom_z = (
                            resample_periodic_bottom(
                                source_bottom_x,
                                source_bottom_z,
                                length,
                                decision.target_count,
                            )
                        )
                    else:
                        candidate_bottom_x, candidate_bottom_z = bottom_x, bottom_z
                    remeshed, remesh_projection = project_vortex_sheet_invariants(
                        remeshed,
                        candidate_bottom_x,
                        candidate_bottom_z,
                        length,
                        gravity,
                        background,
                        target_volume,
                        target_energy,
                        energy_scale,
                    )
                    remesh_cumulative = cumulative_projection + (
                        remesh_projection.relative_circulation_correction
                    )
                    remesh_gates = {
                        "circulation": abs(
                            remeshed.total_circulation - total_circulation_target
                        )
                        <= 1.0e-12,
                        "roundtrip": remesh_diagnostics.potential_roundtrip_relative_error
                        <= 1.0e-10,
                        "monitor": remesh_diagnostics.monitor_mass_cv_after
                        <= remesh_cv_gate,
                        "projection": remesh_projection.relative_circulation_correction
                        <= maximum_projection_correction,
                        "cumulative_projection": remesh_cumulative
                        <= maximum_cumulative_projection,
                        "energy": remesh_projection.relative_energy_error <= 1.0e-10,
                        "volume": remesh_projection.relative_volume_error <= 5.0e-5,
                        "flux": remesh_projection.raw_dno_flux_defect
                        <= maximum_flux_defect,
                    }
                    event["remesh"] = asdict(remesh_diagnostics)
                    event["projection"] = asdict(remesh_projection)
                    event["gates"] = remesh_gates
                    event["accepted"] = all(remesh_gates.values())
                    if event["accepted"]:
                        state = remeshed
                        bottom_x, bottom_z = candidate_bottom_x, candidate_bottom_z
                        cumulative_projection = remesh_cumulative
                        state_projection = remesh_projection
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    event["accepted"] = False
                    event["error"] = f"{type(error).__name__}:{error}"
                adaptation_events.append(event)

        horizontal_mapping, contact = record(
            current_time,
            step_diagnostics.maximum_bie_residual,
            step_diagnostics.maximum_circulation_derivative_defect,
            step_diagnostics.maximum_reconstruction_relative_error,
            projection.relative_circulation_correction,
            (projection.energy_before - target_energy) / energy_scale,
            projection.volume_shift,
            target_volume,
            target_energy,
            state_projection.raw_dno_flux_defect,
            actual_dt,
            retry_count,
        )
        if contact:
            termination_reason = "first_geometric_contact"
            break
        if step_index % 10 == 0:
            print(
                f"step={step_index} t={current_time:.4f} N={len(state.x)} "
                f"dt={actual_dt:.3e} retries={retry_count} "
                f"min_x_alpha={horizontal_mapping:.3e} "
                f"gap/ds={history['normalized_impact_distance'][-1]:.3e} "
                f"proj={projection.relative_circulation_correction:.3e}",
                flush=True,
            )

    elapsed = wall_clock.perf_counter() - start_wall
    maximum_history_count = max(int(value) for value in history["count"])
    array_history: dict[str, Array] = {}
    vector_keys = {
        "x",
        "z",
        "panel_circulation",
        "potential",
        "bottom_x",
        "bottom_z",
    }
    for key, values in history.items():
        if key in vector_keys:
            array_history[key] = _pad(values, maximum_history_count)  # type: ignore[arg-type]
        else:
            array_history[key] = np.asarray(values)
    global_energy_drift = np.abs(
        (array_history["energy"] - global_initial_energy) / energy_scale
    )
    final_gates = {
        "valid_termination": termination_reason
        in {"final_time", "first_geometric_contact"},
        "global_energy_drift_le_gate": float(np.max(global_energy_drift))
        <= maximum_global_energy_drift,
        "volume_error_le_5e-5": float(
            np.max(np.abs(array_history["volume"] - target_volume) / target_volume)
        )
        <= 5.0e-5,
        "flux_defect_le_gate": float(
            np.max(np.abs(array_history["surface_flux_defect"]))
        )
        <= maximum_flux_defect,
        "circulation_defect_le_1e-12": float(
            np.max(np.abs(array_history["circulation_total_defect"]))
        )
        <= 1.0e-12,
        "per_projection_correction_le_gate": float(
            np.max(array_history["projection_relative_correction"])
        )
        <= maximum_projection_correction,
        "cumulative_projection_le_gate": float(
            array_history["cumulative_projection_correction"][-1]
        )
        <= maximum_cumulative_projection,
        "bie_residual_le_1e-8": float(
            np.max(array_history["maximum_bie_residual"])
        )
        <= 1.0e-8,
        "finite_state": all(np.all(np.isfinite(values)) for values in history["x"])
        and all(np.all(np.isfinite(values)) for values in history["z"])
        and all(
            np.all(np.isfinite(values)) for values in history["panel_circulation"]
        ),
    }
    report: dict[str, object] = {
        "schema": "active-adaptive-vortex-sheet-continuation-v1",
        "source": str(source_path),
        "restart_index": source_index,
        "restart_time": selected_time,
        "requested_final_time": final_time,
        "reached_time": float(array_history["time"][-1]),
        "maximum_dt": dt,
        "minimum_dt": minimum_dt,
        "time_integrator": "explicit RK4 with post-step volume/energy projection",
        "state_variable": "integrated panel circulation Gamma",
        "restart_count": len(source_x),
        "initial_adapted_count": initial_count,
        "final_count": int(array_history["count"][-1]),
        "maximum_count_reached": maximum_history_count,
        "termination_reason": termination_reason,
        "first_vertical_front_time": first_vertical_time,
        "first_contact_time": contact_time,
        "target_volume": target_volume,
        "carried_restart_energy": target_energy,
        "global_initial_energy": global_initial_energy,
        "energy_scale": energy_scale,
        "initial_global_energy_drift": initial_global_drift,
        "total_circulation_target": total_circulation_target,
        "cumulative_projection_correction_offset": cumulative_projection_offset,
        "elapsed_seconds": elapsed,
        "initial_remesh": asdict(initial_remesh),
        "initial_projection": asdict(initial_projection),
        "initial_acceptance_gates": initial_gates,
        "adaptation_events": adaptation_events,
        "summary_metrics": {
            "maximum_global_energy_drift": float(np.max(global_energy_drift)),
            "maximum_relative_volume_error": float(
                np.max(
                    np.abs(array_history["volume"] - target_volume) / target_volume
                )
            ),
            "maximum_raw_dno_flux_defect": float(
                np.max(np.abs(array_history["surface_flux_defect"]))
            ),
            "maximum_bie_residual": float(
                np.max(array_history["maximum_bie_residual"])
            ),
            "maximum_projection_relative_correction": float(
                np.max(array_history["projection_relative_correction"])
            ),
            "cumulative_projection_relative_correction": float(
                array_history["cumulative_projection_correction"][-1]
            ),
            "minimum_x_alpha": float(np.min(array_history["min_x_alpha"])),
            "minimum_normalized_impact_distance": float(
                np.min(array_history["normalized_impact_distance"])
            ),
            "maximum_monitor_mass_cv": float(
                np.max(array_history["monitor_mass_cv"])
            ),
            "minimum_accepted_dt": float(
                np.min(array_history["accepted_dt"][1:])
            )
            if len(array_history["accepted_dt"]) > 1
            else 0.0,
            "total_step_retries": int(np.sum(array_history["step_retries"])),
        },
        "acceptance_gates": final_gates,
        "accepted": all(final_gates.values()),
        "claim_boundary": {
            "computation": "Gamma is advanced at every pre-contact step and every accepted remesh is fed back into the next BIE solve",
            "picture": "smooth curves and signed point vortices are two renderings of the same stored circulation state",
            "not_proved": "post-contact reconnection, entrained air and viscous vorticity generation are outside this irrotational pre-contact model",
        },
    }
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        **array_history,
        length=np.asarray(length),
        gravity=np.asarray(gravity),
        background_current=np.asarray(background),
        target_volume=np.asarray(target_volume),
        carried_restart_energy=np.asarray(target_energy),
        global_initial_energy=np.asarray(global_initial_energy),
        energy_reference=np.asarray(reference_energy),
        energy_scale=np.asarray(energy_scale),
        termination_reason=np.asarray(termination_reason),
        first_vertical_front_time=np.asarray(first_vertical_time),
        first_contact_time=np.asarray(contact_time),
        source=np.asarray(str(source_path)),
    )
    output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--start-time", type=float, default=19.95)
    parser.add_argument("--final-time", type=float, default=20.30)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--minimum-dt", type=float, default=0.00125)
    parser.add_argument("--initial-count", type=int, default=384)
    parser.add_argument("--minimum-count", type=int, default=256)
    parser.add_argument("--maximum-count", type=int, default=768)
    parser.add_argument("--adapt-every", type=int, default=10)
    parser.add_argument("--remesh-cv-gate", type=float, default=0.75)
    parser.add_argument("--maximum-projection-correction", type=float, default=1.0e-3)
    parser.add_argument("--maximum-cumulative-projection", type=float, default=5.0e-2)
    parser.add_argument("--maximum-global-energy-drift", type=float, default=5.0e-3)
    parser.add_argument("--maximum-flux-defect", type=float, default=2.0e-2)
    parser.add_argument("--contact-distance-factor", type=float, default=0.15)
    args = parser.parse_args()
    report = continue_active(
        args.source,
        args.output_prefix,
        start_time=args.start_time,
        final_time=args.final_time,
        dt=args.dt,
        minimum_dt=args.minimum_dt,
        initial_count=args.initial_count,
        minimum_count=args.minimum_count,
        maximum_count=args.maximum_count,
        adapt_every=args.adapt_every,
        remesh_cv_gate=args.remesh_cv_gate,
        maximum_projection_correction=args.maximum_projection_correction,
        maximum_cumulative_projection=args.maximum_cumulative_projection,
        maximum_global_energy_drift=args.maximum_global_energy_drift,
        maximum_flux_defect=args.maximum_flux_defect,
        contact_distance_factor=args.contact_distance_factor,
    )
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
