"""Spectrally refine one validated Euler--BIE snapshot for continuation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from impact_detection import periodic_impact_diagnostics
from lagrangian_bie_solver import monitor_equidistribution_cv
from topographic_wave_solver import diagnostics, reference_energy


def _refine_periodic(values: np.ndarray, count: int) -> np.ndarray:
    source_count = len(values)
    modes = np.fft.fftfreq(source_count, d=1.0 / source_count)
    target_alpha = 2.0 * np.pi * np.arange(count) / count
    return (
        np.exp(1j * target_alpha[:, None] * modes[None, :])
        @ (np.fft.fft(values) / source_count)
    ).real


def _refine_x(values: np.ndarray, count: int, length: float) -> np.ndarray:
    source_base = length * np.arange(len(values)) / len(values)
    target_base = length * np.arange(count) / count
    return target_base + _refine_periodic(values - source_base, count)


def refine_snapshot(
    input_path: Path, count: int, index: int
) -> dict[str, np.ndarray]:
    with np.load(input_path) as loaded:
        source = {key: loaded[key] for key in loaded.files}
    if count < source["x"].shape[1] or count % 2:
        raise ValueError("target marker count must be even and not smaller")
    resolved = index if index >= 0 else len(source["time"]) - 1
    length = float(source["length"])
    x = _refine_x(source["x"][resolved], count, length)
    z = _refine_periodic(source["z"][resolved], count)
    potential = _refine_periodic(source["potential"][resolved], count)
    bottom_x = _refine_x(source["bottom_x"], count, length)
    bottom_z = _refine_periodic(source["bottom_z"], count)
    gravity = float(source["gravity"])
    background = float(source.get("background_current", np.asarray(0.0)))
    volume, energy, mapping, spacing, flux = diagnostics(
        x,
        z,
        potential,
        bottom_x,
        bottom_z,
        length,
        gravity,
        background,
    )
    impact = (
        periodic_impact_diagnostics(
            x,
            z,
            length,
            minimum_arc_separation_panels=12.0,
        )
        if mapping <= 0.0
        else None
    )
    scalar_keys = {
        key: value for key, value in source.items()
        if np.asarray(value).ndim == 0
    }
    result = dict(scalar_keys)
    result.update(
        {
            "x": x[None, :],
            "z": z[None, :],
            "potential": potential[None, :],
            "time": np.asarray([source["time"][resolved]]),
            "volume": np.asarray([volume]),
            "energy": np.asarray([energy]),
            "energy_reference": np.asarray(
                reference_energy(
                    bottom_x, bottom_z, length, gravity, background
                )
            ),
            "min_x_alpha": np.asarray([mapping]),
            "marker_cv": np.asarray([spacing]),
            "bie_residual": np.asarray([0.0]),
            "surface_flux_defect": np.asarray([flux]),
            "monitor_equidistribution_cv": np.asarray([
                monitor_equidistribution_cv(x, z, length, 0.0)
            ]),
            "impact_distance": np.asarray([
                np.inf if impact is None else impact.minimum_distance
            ]),
            "normalized_impact_distance": np.asarray([
                np.inf if impact is None else impact.normalized_distance
            ]),
            "nonlinear_iterations": np.asarray([0]),
            "nonlinear_residual": np.asarray([0.0]),
            "krylov_iterations": np.asarray([0]),
            "rhs_evaluations": np.asarray([0]),
            "implicit_fallback": np.asarray([False]),
            "bottom_x": bottom_x,
            "bottom_z": bottom_z,
            "restart_source": np.asarray(str(input_path)),
            "restart_source_index": np.asarray(resolved),
            "restart_source_marker_count": np.asarray(source["x"].shape[1]),
            "global_initial_energy": np.asarray(
                float(source.get("global_initial_energy", source["energy"][0]))
            ),
            "global_initial_volume": np.asarray(
                float(source.get("global_initial_volume", source["volume"][0]))
            ),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--index", type=int, default=-1)
    args = parser.parse_args()
    result = refine_snapshot(args.input, args.n, args.index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(
        f"refined N={result['restart_source_marker_count']} -> {args.n}; "
        f"t={float(result['time'][0]):.6f}; "
        f"|flux|={abs(float(result['surface_flux_defect'][0])):.3e}"
    )


if __name__ == "__main__":
    main()
