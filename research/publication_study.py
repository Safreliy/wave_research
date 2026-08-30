"""Reproducible convergence and close-quadrature ablation studies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from animate_lagrangian_bie import spectral_curve_resample
from solitary_wave import periodic_solitary_wave
from topographic_wave_solver import simulate, smooth_periodic_shoal
from topography_bie import TopographyBIE


def _crest_x(x: np.ndarray, z: np.ndarray, length: float) -> float:
    dense_x, dense_z = spectral_curve_resample(x, z, length, refinement=24)
    index = int(np.argmax(dense_z))
    previous = (index - 1) % len(dense_z)
    following = (index + 1) % len(dense_z)
    denominator = dense_z[previous] - 2.0 * dense_z[index] + dense_z[following]
    offset = (
        0.0
        if abs(float(denominator)) < np.finfo(float).eps
        else 0.5
        * float(dense_z[previous] - dense_z[following])
        / float(denominator)
    )
    spacing = 0.5 * (
        (dense_x[following] - dense_x[index]) % length
        + (dense_x[index] - dense_x[previous]) % length
    )
    return float((dense_x[index] + offset * spacing) % length)


def _translation_aligned_shape_error(
    initial_x: np.ndarray,
    initial_z: np.ndarray,
    final_x: np.ndarray,
    final_z: np.ndarray,
    length: float,
) -> float:
    common = np.linspace(-0.5 * length, 0.5 * length, 4096, endpoint=False)

    def crest_aligned(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        dense_x, dense_z = spectral_curve_resample(
            x, z, length, refinement=64
        )
        crest = _crest_x(x, z, length)
        relative_x = (dense_x - crest + 0.5 * length) % length - 0.5 * length
        order = np.argsort(relative_x)
        sorted_x = relative_x[order]
        sorted_z = dense_z[order]
        extended_x = np.concatenate(
            ([sorted_x[-1] - length], sorted_x, [sorted_x[0] + length])
        )
        extended_z = np.concatenate(
            ([sorted_z[-1]], sorted_z, [sorted_z[0]])
        )
        return np.interp(common, extended_x, extended_z)

    initial = crest_aligned(initial_x, initial_z)
    final = crest_aligned(final_x, final_z)
    return float(np.linalg.norm(final - initial) / np.linalg.norm(initial))


def _flat_run(
    n: int,
    dt: float,
    final_time: float,
    initial_condition: str = "exact_solitary",
) -> tuple[dict, dict]:
    length = 20.0
    center = 6.0
    amplitude = 0.2
    depth = 1.0
    result = simulate(
        n=n,
        length=length,
        depth=depth,
        shoal_height=0.0,
        shoal_center=0.0,
        crest_center=center,
        amplitude=amplitude,
        gravity=1.0,
        dt=dt,
        final_time=final_time,
        snapshots=11,
        initial_condition=initial_condition,
    )
    target_speed = float(result["initial_wave_speed"])
    shape_error = _translation_aligned_shape_error(
        result["x"][0],
        result["z"][0],
        result["x"][-1],
        result["z"][-1],
        length,
    )
    initial_crest = _crest_x(result["x"][0], result["z"][0], length)
    final_crest = _crest_x(result["x"][-1], result["z"][-1], length)
    measured_speed = (final_crest - initial_crest) / final_time
    energy_scale = max(
        abs(float(result["energy"][0] - result["energy_reference"])),
        np.finfo(float).eps,
    )
    metrics = {
        "n": n,
        "dt": dt,
        "final_time": final_time,
        "initial_condition": initial_condition,
        "initial_equation_residual": float(
            result["initial_equation_residual"]
        ),
        "shape_relative_l2_error": shape_error,
        "crest_height_error": float(abs(np.max(result["z"][-1]) - amplitude)),
        "measured_speed": measured_speed,
        "target_speed": target_speed,
        "relative_speed_error": abs(measured_speed - target_speed) / target_speed,
        "max_relative_energy_drift": float(
            np.max(np.abs(result["energy"] - result["energy"][0])) / energy_scale
        ),
        "max_absolute_volume_drift": float(
            np.max(np.abs(result["volume"] - result["volume"][0]))
        ),
        "max_surface_flux_defect": float(
            np.max(np.abs(result["surface_flux_defect"]))
        ),
    }
    return result, metrics


def flat_spatial_study(output_directory: Path) -> list[dict]:
    records = []
    for n in (32, 48, 64):
        result, metrics = _flat_run(n, dt=0.002, final_time=0.5)
        records.append(metrics)
        np.savez_compressed(
            output_directory / f"flat_solitary_n{n}.npz", **result
        )
    return records


def flat_temporal_study(output_directory: Path) -> list[dict]:
    records = []
    for dt in (0.008, 0.004, 0.002, 0.001):
        result, metrics = _flat_run(48, dt=dt, final_time=0.25)
        records.append(metrics)
        token = str(dt).replace(".", "p")
        np.savez_compressed(
            output_directory / f"flat_solitary_dt{token}.npz", **result
        )
    reference_error = records[-1]["shape_relative_l2_error"]
    for record in records:
        record["shape_error_above_finest"] = abs(
            record["shape_relative_l2_error"] - reference_error
        )
    return records


def initializer_ablation(output_directory: Path) -> list[dict]:
    records = []
    for initial_condition in ("solitary", "exact_solitary"):
        result, metrics = _flat_run(
            64,
            dt=0.002,
            final_time=0.5,
            initial_condition=initial_condition,
        )
        records.append(metrics)
        np.savez_compressed(
            output_directory / f"initializer_{initial_condition}.npz",
            **result,
        )
    return records


def close_quadrature_ablation() -> list[dict]:
    records = []
    length = 20.0
    for n in (24, 32, 40, 48, 64, 80):
        x = np.arange(n) * length / n
        bottom_x, bottom_z = smooth_periodic_shoal(
            n, length, depth=1.0, height=0.75, center=13.0
        )
        state = periodic_solitary_wave(
            x, length, center=5.0, amplitude=0.4, depth=1.0
        )
        reference = TopographyBIE(
            x,
            state.elevation,
            bottom_x,
            bottom_z,
            length,
            cross_oversampling=16,
        )
        reference_bottom_flux = (
            -state.background_current
            * reference.bottom.z_alpha
            / reference.bottom.metric
        )
        reference_q = reference.solve(
            state.periodic_potential, reference_bottom_flux
        ).surface_normal_derivative
        for method, factor in (("plain", 1), ("adaptive", None)):
            start = time.perf_counter()
            bie = TopographyBIE(
                x,
                state.elevation,
                bottom_x,
                bottom_z,
                length,
                cross_oversampling=factor,
            )
            bottom_flux = (
                -state.background_current
                * bie.bottom.z_alpha
                / bie.bottom.metric
            )
            solution = bie.solve(state.periodic_potential, bottom_flux)
            elapsed = time.perf_counter() - start
            normal_x = -bie.surface.z_alpha / bie.surface.metric
            total_normal = (
                solution.surface_normal_derivative
                + state.background_current * normal_x
            )
            flux = float(bie.dalpha * np.sum(total_normal * bie.surface.metric))
            q_error = float(
                np.linalg.norm(solution.surface_normal_derivative - reference_q)
                / np.linalg.norm(reference_q)
            )
            records.append(
                {
                    "n": n,
                    "method": method,
                    "oversampling_factor": bie.cross_oversampling,
                    "absolute_flux_defect": abs(flux),
                    "relative_q_error_vs_factor16": q_error,
                    "wall_seconds": elapsed,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study", choices=("flat", "close", "all"), default="all"
    )
    parser.add_argument(
        "--output-directory", type=Path, default=Path("results/publication_study")
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {}
    if args.study in ("flat", "all"):
        report["flat_spatial"] = flat_spatial_study(args.output_directory)
        report["flat_temporal"] = flat_temporal_study(args.output_directory)
        report["initializer_ablation"] = initializer_ablation(
            args.output_directory
        )
    if args.study in ("close", "all"):
        report["close_quadrature_ablation"] = close_quadrature_ablation()
    output = args.output_directory / "publication_study.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
