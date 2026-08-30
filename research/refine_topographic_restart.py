"""Spectrally refine one accepted Euler--BIE state for event-local continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample

from topographic_wave_solver import (
    diagnostics,
    geometric_volume,
    monitor_equidistribution_cv,
)


def periodic_refine(values: np.ndarray, new_n: int) -> np.ndarray:
    return np.asarray(resample(np.asarray(values, dtype=float), new_n), dtype=float)


def refine_restart(
    source: dict[str, np.ndarray],
    new_n: int,
    restart_time: float | None = None,
    new_dt: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    old_n = int(source["x"].shape[1])
    if new_n <= old_n or new_n % 2:
        raise ValueError("new_n must be an even integer larger than the source N")
    index = (
        len(source["time"]) - 1
        if restart_time is None
        else int(np.argmin(np.abs(source["time"] - restart_time)))
    )
    selected_time = float(source["time"][index])
    length = float(source["length"])
    old_base = length * np.arange(old_n) / old_n
    new_base = length * np.arange(new_n) / new_n
    x = new_base + periodic_refine(source["x"][index] - old_base, new_n)
    z = periodic_refine(source["z"][index], new_n)
    potential = periodic_refine(source["potential"][index], new_n)
    potential -= np.mean(potential)
    bottom_x = new_base + periodic_refine(source["bottom_x"] - old_base, new_n)
    bottom_z = periodic_refine(source["bottom_z"], new_n)
    # Measure interpolation consistency before the invariant projection below;
    # these are distinct numerical operations and must not share one gate.
    downsampled_x = old_base + periodic_refine(x - new_base, old_n)
    downsampled_z = periodic_refine(z, old_n)
    downsampled_potential = periodic_refine(potential, old_n)
    state_scale = max(
        float(
            np.linalg.norm(
                np.concatenate(
                    (
                        source["x"][index] - old_base,
                        source["z"][index],
                        source["potential"][index],
                    )
                )
            )
        ),
        np.finfo(float).eps,
    )
    roundtrip_error = float(
        np.linalg.norm(
            np.concatenate(
                (
                    downsampled_x - source["x"][index],
                    downsampled_z - source["z"][index],
                    downsampled_potential - source["potential"][index],
                )
            )
        )
        / state_scale
    )
    target_volume = float(
        source.get("global_target_volume", np.asarray(source["volume"][0]))
    )
    raw_volume = geometric_volume(x, z, bottom_x, bottom_z, length)
    volume_shift = (target_volume - raw_volume) / length
    z += volume_shift
    volume, energy, mapping, spacing, flux = diagnostics(
        x,
        z,
        potential,
        bottom_x,
        bottom_z,
        length,
        float(source["gravity"]),
        float(source.get("background_current", np.asarray(0.0))),
    )
    initial_energy = float(
        source.get("global_initial_energy", np.asarray(source["energy"][0]))
    )
    energy_reference = float(source["energy_reference"])
    energy_scale = max(abs(initial_energy - energy_reference), np.finfo(float).eps)
    source_energy = float(source["energy"][index])
    relative_energy_jump = abs(energy - source_energy) / energy_scale
    offset = float(
        source.get("cumulative_filter_correction_offset", np.asarray(0.0))
    ) + float(np.sum(source["spectral_filter_relative_corrections"][: index + 1]))

    result = {key: np.asarray(value) for key, value in source.items()}
    result.update(
        {
            "x": x[np.newaxis, :],
            "z": z[np.newaxis, :],
            "potential": potential[np.newaxis, :],
            "time": np.asarray([selected_time]),
            "volume": np.asarray([volume]),
            "energy": np.asarray([energy]),
            "min_x_alpha": np.asarray([mapping]),
            "marker_cv": np.asarray([spacing]),
            "bie_residual": np.asarray([0.0]),
            "surface_flux_defect": np.asarray([flux]),
            "nonlinear_iterations": np.asarray([0]),
            "nonlinear_residual": np.asarray([0.0]),
            "krylov_iterations": np.asarray([0]),
            "rhs_evaluations": np.asarray([0]),
            "implicit_fallback": np.asarray([False]),
            "monitor_equidistribution_cv": np.asarray(
                [
                    monitor_equidistribution_cv(
                        x,
                        z,
                        length,
                        float(source.get("reparameterization_strength", np.asarray(0.0))),
                        float(source.get("reparameterization_power", np.asarray(0.5))),
                    )
                ]
            ),
            "impact_distance": np.asarray([float("inf")]),
            "normalized_impact_distance": np.asarray([float("inf")]),
            "volume_projection_shifts": np.asarray([volume_shift]),
            "spectral_filter_relative_corrections": np.asarray([0.0]),
            "bottom_x": bottom_x,
            "bottom_z": bottom_z,
            "dt": np.asarray(new_dt if new_dt is not None else float(source["dt"])),
            "termination_reason": np.asarray("refined_restart_ready"),
            "termination_step": np.asarray(0),
            "global_initial_energy": np.asarray(initial_energy),
            "global_target_volume": np.asarray(target_volume),
            "cumulative_filter_correction_offset": np.asarray(offset),
            "refined_from_n": np.asarray(old_n),
            "refinement_source_time": np.asarray(selected_time),
            "refinement_relative_energy_jump": np.asarray(relative_energy_jump),
            "refinement_roundtrip_state_error": np.asarray(roundtrip_error),
        }
    )
    report: dict[str, object] = {
        "schema": "topographic-bie-spectral-refinement-v1",
        "source_n": old_n,
        "target_n": new_n,
        "source_time": selected_time,
        "new_dt": float(result["dt"]),
        "relative_energy_jump": relative_energy_jump,
        "roundtrip_state_error": roundtrip_error,
        "volume_projection_shift": volume_shift,
        "relative_volume_error": (volume - target_volume) / target_volume,
        "raw_dno_flux_defect": abs(flux),
        "minimum_x_alpha": mapping,
        "marker_spacing_cv": spacing,
        "cumulative_filter_correction_offset": offset,
        "acceptance_gates": {
            "relative_energy_jump_le_1e-3": relative_energy_jump <= 1.0e-3,
            "roundtrip_state_error_le_1e-10": roundtrip_error <= 1.0e-10,
            "relative_volume_error_le_5e-5": abs(volume - target_volume) / target_volume
            <= 5.0e-5,
            "raw_dno_flux_defect_le_2e-2": abs(flux) <= 2.0e-2,
            "marker_spacing_cv_le_1e-1": spacing <= 1.0e-1,
            "cumulative_filter_correction_le_5e-2": offset <= 5.0e-2,
        },
    }
    report["accepted"] = all(report["acceptance_gates"].values())
    return result, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--time", type=float)
    parser.add_argument("--dt", type=float)
    args = parser.parse_args()
    with np.load(args.source) as archive:
        source = {key: archive[key] for key in archive.files}
    result, report = refine_restart(source, args.n, args.time, args.dt)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_prefix.with_suffix(".npz"), **result)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
