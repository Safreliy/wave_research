"""Continue an accepted topographic Euler--BIE trajectory without reinitialising it."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from jfm2026_solver_benchmark import _breaking_metrics
from topographic_wave_solver import (
    diagnostics,
    geometric_volume,
    minimum_horizontal_mapping,
    monitor_equidistribution_cv,
    periodic_impact_diagnostics,
    spectral_viscosity_filter,
    step_implicit_midpoint,
    step_rk4,
)


HISTORY_KEYS = (
    "x",
    "z",
    "potential",
    "time",
    "volume",
    "energy",
    "min_x_alpha",
    "marker_cv",
    "bie_residual",
    "surface_flux_defect",
    "nonlinear_iterations",
    "nonlinear_residual",
    "krylov_iterations",
    "rhs_evaluations",
    "implicit_fallback",
    "monitor_equidistribution_cv",
    "impact_distance",
    "normalized_impact_distance",
    "volume_projection_shifts",
    "spectral_filter_relative_corrections",
)


def continue_trajectory(
    source: dict[str, np.ndarray],
    final_time: float,
    stop_at_contact: bool = True,
    stop_on_conservation_failure: bool = True,
) -> dict[str, np.ndarray]:
    """Continue from the last archived state with identical discretisation.

    The complete accepted prefix is retained in the returned archive.  No
    translation, remapping or re-projection of the restart state is performed.
    """
    missing = set(HISTORY_KEYS).difference(source)
    if missing:
        raise ValueError(f"source trajectory is missing fields: {sorted(missing)}")
    time0 = float(source["time"][-1])
    if final_time <= time0:
        raise ValueError("final_time must exceed the last source time")
    dt = float(source["dt"])
    steps = int(math.ceil((final_time - time0) / dt))
    actual_dt = (final_time - time0) / steps
    length = float(source["length"])
    gravity = float(source["gravity"])
    background_current = float(source.get("background_current", np.asarray(0.0)))
    bottom_x = np.asarray(source["bottom_x"], dtype=float)
    bottom_z = np.asarray(source["bottom_z"], dtype=float)
    tangential_gauge = str(source.get("tangential_gauge", np.asarray("theta_s")))
    time_integrator = str(source.get("time_integrator", np.asarray("rk4")))
    implicit_tolerance = float(source.get("implicit_tolerance", np.asarray(2.0e-10)))
    spectral_strength = float(source.get("spectral_filter_strength", np.asarray(0.0)))
    spectral_order = int(source.get("spectral_filter_order", np.asarray(16)))
    project_volume = bool(source.get("project_volume", np.asarray(True)))
    target_volume = float(
        source.get("global_target_volume", np.asarray(source["volume"][0]))
    )
    contact_factor = float(source.get("impact_distance_factor", np.asarray(0.15)))
    contact_exclusion = float(source.get("contact_arc_exclusion", np.asarray(12.0)))
    monitor_strength = float(source.get("reparameterization_strength", np.asarray(0.0)))
    monitor_power = float(source.get("reparameterization_power", np.asarray(0.5)))

    records = {key: list(np.asarray(source[key])) for key in HISTORY_KEYS}
    x = np.asarray(source["x"][-1], dtype=float).copy()
    z = np.asarray(source["z"][-1], dtype=float).copy()
    potential = np.asarray(source["potential"][-1], dtype=float).copy()
    initial_energy = float(
        source.get("global_initial_energy", np.asarray(source["energy"][0]))
    )
    energy_scale = max(
        abs(float(initial_energy - source["energy_reference"])),
        np.finfo(float).eps,
    )
    cumulative_filter = float(
        source.get("cumulative_filter_correction_offset", np.asarray(0.0))
    ) + float(
        np.sum(source["spectral_filter_relative_corrections"])
    )
    termination_reason = "completed"
    first_vertical_time: float | None = None

    for local_step in range(1, steps + 1):
        previous = (x.copy(), z.copy(), potential.copy())
        step_diagnostics = None
        try:
            if time_integrator == "rk4":
                x, z, potential, residual = step_rk4(
                    x,
                    z,
                    potential,
                    bottom_x,
                    bottom_z,
                    length,
                    gravity,
                    actual_dt,
                    background_current,
                    tangential_gauge,
                )
            elif time_integrator == "implicit_midpoint":
                x, z, potential, residual, step_diagnostics = step_implicit_midpoint(
                    x,
                    z,
                    potential,
                    bottom_x,
                    bottom_z,
                    length,
                    gravity,
                    actual_dt,
                    background_current,
                    tangential_gauge,
                    nonlinear_tolerance=implicit_tolerance,
                    allow_explicit_fallback=False,
                )
            else:
                raise ValueError(f"unsupported time integrator: {time_integrator}")
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as error:
            x, z, potential = previous
            termination_reason = f"solver_failure: {type(error).__name__}"
            break
        if not (
            np.all(np.isfinite(x))
            and np.all(np.isfinite(z))
            and np.all(np.isfinite(potential))
        ):
            x, z, potential = previous
            termination_reason = "non_finite_candidate"
            break

        if spectral_strength:
            x, z, potential, filter_correction = spectral_viscosity_filter(
                x,
                z,
                potential,
                length,
                spectral_strength,
                spectral_order,
            )
        else:
            filter_correction = 0.0
        cumulative_filter += filter_correction
        if project_volume:
            current_volume = geometric_volume(x, z, bottom_x, bottom_z, length)
            volume_shift = (target_volume - current_volume) / length
            z += volume_shift
        else:
            volume_shift = 0.0

        state_time = time0 + local_step * actual_dt
        volume, energy, mapping, spacing, flux = diagnostics(
            x,
            z,
            potential,
            bottom_x,
            bottom_z,
            length,
            gravity,
            background_current,
        )
        impact = (
            periodic_impact_diagnostics(
                x,
                z,
                length,
                minimum_arc_separation_panels=contact_exclusion,
            )
            if mapping <= 0.0
            else None
        )
        if mapping <= 0.0 and first_vertical_time is None:
            first_vertical_time = state_time

        records["x"].append(x.copy())
        records["z"].append(z.copy())
        records["potential"].append(potential.copy())
        records["time"].append(state_time)
        records["volume"].append(volume)
        records["energy"].append(energy)
        records["min_x_alpha"].append(mapping)
        records["marker_cv"].append(spacing)
        records["bie_residual"].append(residual)
        records["surface_flux_defect"].append(flux)
        records["monitor_equidistribution_cv"].append(
            monitor_equidistribution_cv(
                x,
                z,
                length,
                monitor_strength,
                monitor_power,
            )
        )
        records["impact_distance"].append(
            impact.minimum_distance if impact is not None else float("inf")
        )
        records["normalized_impact_distance"].append(
            impact.normalized_distance if impact is not None else float("inf")
        )
        records["volume_projection_shifts"].append(volume_shift)
        records["spectral_filter_relative_corrections"].append(filter_correction)
        if step_diagnostics is None:
            records["nonlinear_iterations"].append(0)
            records["nonlinear_residual"].append(0.0)
            records["krylov_iterations"].append(0)
            records["rhs_evaluations"].append(4)
            records["implicit_fallback"].append(False)
        else:
            records["nonlinear_iterations"].append(
                step_diagnostics.nonlinear.newton_iterations
            )
            records["nonlinear_residual"].append(
                step_diagnostics.nonlinear.residual
            )
            records["krylov_iterations"].append(
                step_diagnostics.nonlinear.krylov_iterations
            )
            records["rhs_evaluations"].append(
                step_diagnostics.nonlinear.rhs_evaluations
            )
            records["implicit_fallback"].append(
                step_diagnostics.used_explicit_fallback
            )

        if impact is not None and stop_at_contact and (
            impact.self_intersection or impact.normalized_distance <= contact_factor
        ):
            termination_reason = (
                "self_intersection"
                if impact.self_intersection
                else "first_contact_threshold"
            )
            break
        if stop_on_conservation_failure:
            relative_energy = abs(energy - initial_energy) / energy_scale
            relative_volume = abs(volume - target_volume) / max(
                abs(target_volume), np.finfo(float).eps
            )
            if (
                relative_energy > 5.0e-3
                or relative_volume > 5.0e-5
                or abs(flux) > 2.0e-2
                or spacing > 1.0e-1
                or cumulative_filter > 5.0e-2
            ):
                termination_reason = "conservation_gate_failure"
                break

    result = {key: np.asarray(value) for key, value in source.items()}
    for key, values in records.items():
        result[key] = np.asarray(values)
    result["dt"] = np.asarray(actual_dt)
    result["termination_reason"] = np.asarray(termination_reason)
    result["termination_step"] = np.asarray(len(result["time"]) - 1)
    result["restart_source_time"] = np.asarray(time0)
    result["first_vertical_front_time"] = np.asarray(
        first_vertical_time if first_vertical_time is not None else float("nan")
    )
    return result


def continuation_report(data: dict[str, np.ndarray], source_path: Path) -> dict[str, object]:
    contact = np.flatnonzero(
        np.isfinite(data["normalized_impact_distance"])
        & (
            data["normalized_impact_distance"]
            <= float(data.get("impact_distance_factor", np.asarray(0.15)))
        )
    )
    contact_index = int(contact[0]) if len(contact) else None
    return {
        "schema": "topographic-bie-continuation-v1",
        "source": str(source_path),
        "restart_time": float(data["restart_source_time"]),
        "final_time": float(data["time"][-1]),
        "termination_reason": str(data["termination_reason"]),
        "first_vertical_front_time": (
            float(data["first_vertical_front_time"])
            if np.isfinite(data["first_vertical_front_time"])
            else None
        ),
        "contact_reached": contact_index is not None,
        "contact_time": (
            float(data["time"][contact_index]) if contact_index is not None else None
        ),
        "contact_normalized_gap": (
            float(data["normalized_impact_distance"][contact_index])
            if contact_index is not None
            else None
        ),
        **_breaking_metrics(data),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--final-time", type=float, required=True)
    parser.add_argument("--ignore-contact", action="store_true")
    parser.add_argument("--ignore-conservation-gates", action="store_true")
    args = parser.parse_args()
    with np.load(args.source) as loaded:
        source = {key: loaded[key] for key in loaded.files}
    result = continue_trajectory(
        source,
        args.final_time,
        stop_at_contact=not args.ignore_contact,
        stop_on_conservation_failure=not args.ignore_conservation_gates,
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_prefix.with_suffix(".npz"), **result)
    report = continuation_report(result, args.source)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
