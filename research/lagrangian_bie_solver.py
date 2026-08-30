"""Experimental material-surface Euler integrator using ParametricMirrorBIE.

This is a research prototype for pre-impact overturning, not yet a validated
publication solver.  It integrates the exact material free-surface equations
with an approximate dense BIE Dirichlet-to-Neumann map.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from parametric_bie import ParametricMirrorBIE


Array = np.ndarray
_DENSE_PHASE_CACHE: dict[tuple[int, int], tuple[Array, Array, Array]] = {}


def rhs(
    x: Array,
    z: Array,
    potential: Array,
    length: float,
    depth: float,
    gravity: float,
    tangential_mode: str,
) -> tuple[Array, Array, Array, float]:
    bie = ParametricMirrorBIE(x, z, length, depth)
    physical_velocity_x, physical_velocity_z, result = bie.surface_velocity(potential)
    if tangential_mode == "lagrangian":
        velocity_x = physical_velocity_x
        velocity_z = physical_velocity_z
        potential_t = (
            0.5 * (physical_velocity_x**2 + physical_velocity_z**2) - gravity * z
        )
    elif tangential_mode == "theta_s":
        normal_velocity = result.unit_normal_derivative
        tangential_fluid_velocity = bie.surface_derivative(potential)
        cross = (
            bie.x_alpha * bie.z_alpha_alpha
            - bie.z_alpha * bie.x_alpha_alpha
        )
        curvature = cross / bie.metric**3
        target_derivative = curvature * normal_velocity * bie.metric
        target_derivative -= np.mean(target_derivative)
        coefficients = np.fft.fft(target_derivative)
        inverse_derivative = np.zeros(bie.n, dtype=complex)
        nonzero = bie.alpha_modes != 0.0
        inverse_derivative[nonzero] = 1.0 / (1j * bie.alpha_modes[nonzero])
        tangential_grid_velocity = np.fft.ifft(
            inverse_derivative * coefficients
        ).real
        tangential_grid_velocity += np.mean(tangential_fluid_velocity)
        velocity_x = (
            normal_velocity * bie.normal_x
            + tangential_grid_velocity * bie.tangent_x
        )
        velocity_z = (
            normal_velocity * bie.normal_z
            + tangential_grid_velocity * bie.tangent_z
        )
        # ALE surface-potential equation.  Only the normal grid velocity is
        # physical; the chosen tangential velocity changes parameterization.
        potential_t = (
            0.5 * normal_velocity**2
            - 0.5 * tangential_fluid_velocity**2
            + tangential_grid_velocity * tangential_fluid_velocity
            - gravity * z
        )
    else:
        raise ValueError("tangential_mode must be 'lagrangian' or 'theta_s'")
    potential_t -= np.mean(potential_t)
    return velocity_x, velocity_z, potential_t, result.boundary_residual


def step_rk4(
    x: Array,
    z: Array,
    potential: Array,
    dt: float,
    length: float,
    depth: float,
    gravity: float,
    tangential_mode: str,
) -> tuple[Array, Array, Array, float]:
    k1 = rhs(x, z, potential, length, depth, gravity, tangential_mode)
    k2 = rhs(
        x + 0.5 * dt * k1[0],
        z + 0.5 * dt * k1[1],
        potential + 0.5 * dt * k1[2],
        length,
        depth,
        gravity,
        tangential_mode,
    )
    k3 = rhs(
        x + 0.5 * dt * k2[0],
        z + 0.5 * dt * k2[1],
        potential + 0.5 * dt * k2[2],
        length,
        depth,
        gravity,
        tangential_mode,
    )
    k4 = rhs(
        x + dt * k3[0],
        z + dt * k3[1],
        potential + dt * k3[2],
        length,
        depth,
        gravity,
        tangential_mode,
    )
    next_x = x + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
    next_z = z + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
    next_potential = potential + dt * (
        k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]
    ) / 6.0
    next_potential -= np.mean(next_potential)
    return next_x, next_z, next_potential, max(k1[3], k2[3], k3[3], k4[3])


def invariants(
    x: Array,
    z: Array,
    potential: Array,
    length: float,
    depth: float,
    gravity: float,
) -> tuple[float, float, float, float]:
    bie = ParametricMirrorBIE(x, z, length, depth)
    result = bie.dirichlet_to_neumann(potential)
    volume = float(bie.dalpha * np.sum(z * bie.x_alpha))
    kinetic = 0.5 * bie.dalpha * np.sum(
        potential * result.unit_normal_derivative * bie.metric
    )
    potential_energy = 0.5 * gravity * bie.dalpha * np.sum(z**2 * bie.x_alpha)
    energy = float(kinetic + potential_energy)
    min_x_alpha = float(np.min(bie.x_alpha))
    marker_cv = float(np.std(bie.metric) / np.mean(bie.metric))
    return volume, energy, min_x_alpha, marker_cv


def uniform_arclength_resample(
    x: Array, z: Array, potential: Array, length: float
) -> tuple[Array, Array, Array]:
    """One-time periodic resampling onto equal arclength markers."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    potential = np.asarray(potential, dtype=float)
    x_extended = np.concatenate((x, [x[0] + length]))
    z_extended = np.concatenate((z, [z[0]]))
    potential_extended = np.concatenate((potential, [potential[0]]))
    segment_length = np.sqrt(
        np.diff(x_extended) ** 2 + np.diff(z_extended) ** 2
    )
    cumulative = np.concatenate(([0.0], np.cumsum(segment_length)))
    targets = np.arange(len(x), dtype=float) * cumulative[-1] / len(x)
    new_x = np.interp(targets, cumulative, x_extended)
    new_z = np.interp(targets, cumulative, z_extended)
    new_potential = np.interp(targets, cumulative, potential_extended)
    new_potential -= np.mean(new_potential)
    return new_x, new_z, new_potential


def spectral_arclength_resample(
    x: Array,
    z: Array,
    potential: Array,
    length: float,
    refinement: int = 8,
    curvature_strength: float = 0.0,
    curvature_power: float = 0.5,
) -> tuple[Array, Array, Array]:
    """Spectrally reparameterize a periodic curve using a curvature monitor.

    ``curvature_strength=0`` gives equal arclength.  Positive strength makes
    ``(1 + strength * normalized_curvature**power) * ds`` approximately
    uniform and therefore places more markers near a tight overturning jet.
    Dense Fourier evaluation is used only to invert the monitor integral;
    the returned coordinates and trace are evaluated directly from the
    original Fourier coefficients.  This avoids the repeated piecewise-linear
    diffusion that would otherwise corrupt energy near a steep crest.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    potential = np.asarray(potential, dtype=float)
    if not (x.shape == z.shape == potential.shape) or x.ndim != 1:
        raise ValueError("x, z and potential must be matching vectors")
    if refinement < 4:
        raise ValueError("arclength refinement must be at least four")
    if curvature_strength < 0.0 or curvature_power <= 0.0:
        raise ValueError("curvature monitor parameters are invalid")
    n = len(x)
    dense_count = refinement * n
    cache_key = (n, refinement)
    cached = _DENSE_PHASE_CACHE.get(cache_key)
    if cached is None:
        dense_alpha = 2.0 * math.pi * np.arange(dense_count) / dense_count
        modes = np.fft.fftfreq(n, d=1.0 / n)
        dense_phase = np.exp(1j * dense_alpha[:, None] * modes[None, :])
        _DENSE_PHASE_CACHE[cache_key] = (dense_alpha, modes, dense_phase)
    else:
        dense_alpha, modes, dense_phase = cached
    base = length * np.arange(n) / n
    displacement_coefficients = np.fft.fft(x - base) / n
    z_coefficients = np.fft.fft(z) / n
    potential_coefficients = np.fft.fft(potential) / n
    dense_x = length * dense_alpha / (2.0 * math.pi) + (
        dense_phase @ displacement_coefficients
    ).real
    dense_z = (dense_phase @ z_coefficients).real
    if curvature_strength:
        dense_x_alpha = length / (2.0 * math.pi) + (
            dense_phase @ (1j * modes * displacement_coefficients)
        ).real
        dense_z_alpha = (
            dense_phase @ (1j * modes * z_coefficients)
        ).real
        dense_x_alpha_alpha = (
            dense_phase @ (-(modes**2) * displacement_coefficients)
        ).real
        dense_z_alpha_alpha = (
            dense_phase @ (-(modes**2) * z_coefficients)
        ).real
        dense_metric = np.sqrt(dense_x_alpha**2 + dense_z_alpha**2)
        dense_curvature = np.abs(
            dense_x_alpha * dense_z_alpha_alpha
            - dense_z_alpha * dense_x_alpha_alpha
        ) / np.maximum(dense_metric**3, 1.0e-14)
        curvature_scale = max(
            float(np.quantile(dense_curvature, 0.995)),
            1.0 / length,
        )
        monitor = 1.0 + curvature_strength * np.minimum(
            dense_curvature / curvature_scale, 1.0
        ) ** curvature_power
    else:
        monitor = np.ones(dense_count)
    extended_x = np.concatenate((dense_x, [dense_x[0] + length]))
    extended_z = np.concatenate((dense_z, [dense_z[0]]))
    segment_length = np.sqrt(
        np.diff(extended_x) ** 2 + np.diff(extended_z) ** 2
    )
    extended_monitor = np.concatenate((monitor, [monitor[0]]))
    weighted_segment = segment_length * 0.5 * (
        extended_monitor[:-1] + extended_monitor[1:]
    )
    cumulative = np.concatenate(([0.0], np.cumsum(weighted_segment)))
    targets = np.arange(n, dtype=float) * cumulative[-1] / n
    alpha_extended = np.concatenate((dense_alpha, [2.0 * math.pi]))
    target_alpha = np.interp(targets, cumulative, alpha_extended)
    target_phase = np.exp(1j * target_alpha[:, None] * modes[None, :])
    new_x = length * target_alpha / (2.0 * math.pi) + (
        target_phase @ displacement_coefficients
    ).real
    new_z = (target_phase @ z_coefficients).real
    new_potential = (target_phase @ potential_coefficients).real
    new_potential -= np.mean(new_potential)
    return new_x, new_z, new_potential


def monitor_equidistribution_cv(
    x: Array,
    z: Array,
    length: float,
    curvature_strength: float,
    curvature_power: float = 0.5,
) -> float:
    """Coefficient of variation of the curvature-monitor panel mass."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    n = len(x)
    modes = np.fft.fftfreq(n, d=1.0 / n)
    base = length * np.arange(n) / n
    displacement = np.fft.fft(x - base)
    z_coefficients = np.fft.fft(z)
    x_alpha = length / (2.0 * math.pi) + np.fft.ifft(
        1j * modes * displacement
    ).real
    z_alpha = np.fft.ifft(1j * modes * z_coefficients).real
    x_alpha_alpha = np.fft.ifft(-(modes**2) * displacement).real
    z_alpha_alpha = np.fft.ifft(-(modes**2) * z_coefficients).real
    metric = np.sqrt(x_alpha**2 + z_alpha**2)
    if curvature_strength:
        curvature = np.abs(
            x_alpha * z_alpha_alpha - z_alpha * x_alpha_alpha
        ) / np.maximum(metric**3, 1.0e-14)
        scale = max(float(np.quantile(curvature, 0.995)), 1.0 / length)
        monitor = 1.0 + curvature_strength * np.minimum(
            curvature / scale, 1.0
        ) ** curvature_power
    else:
        monitor = np.ones(n)
    next_x = np.roll(x, -1)
    next_x[-1] += length
    panel_length = np.sqrt((next_x - x) ** 2 + (np.roll(z, -1) - z) ** 2)
    panel_mass = panel_length * 0.5 * (monitor + np.roll(monitor, -1))
    return float(np.std(panel_mass) / np.mean(panel_mass))


def initial_progressive_wave(
    n: int,
    length: float,
    depth: float,
    gravity: float,
    amplitude: float,
    velocity_factor: float,
) -> tuple[Array, Array, Array]:
    alpha = np.arange(n) * 2.0 * math.pi / n
    x = length * alpha / (2.0 * math.pi)
    k = 2.0 * math.pi / length
    z = amplitude * np.cos(k * x) + 0.5 * k * amplitude**2 * np.cos(2.0 * k * x)
    z -= np.mean(z)
    omega = math.sqrt(gravity * k * math.tanh(k * depth))
    potential = (
        velocity_factor
        * amplitude
        * omega
        / (k * math.tanh(k * depth))
        * np.sin(k * x)
    )
    potential -= np.mean(potential)
    return x, z, potential


def initial_localized_breaker(
    n: int,
    length: float,
    amplitude: float,
    jet_speed: float,
    width: float,
) -> tuple[Array, Array, Array]:
    """Smooth Euler initial data with a fast crest-localized forward current."""
    if width <= 0.0 or jet_speed <= 0.0:
        raise ValueError("width and jet_speed must be positive")
    x = np.arange(n) * length / n
    center = 0.5 * length
    distance = (x - center + 0.5 * length) % length - 0.5 * length
    z = amplitude * np.exp(-(distance / width) ** 2)
    z -= np.mean(z)
    target_surface_u = jet_speed * np.exp(-(distance / (0.72 * width)) ** 2)
    target_surface_u -= np.mean(target_surface_u)
    modes = 2.0 * math.pi * np.fft.fftfreq(n, d=length / n)
    coefficients = np.fft.fft(target_surface_u)
    multiplier = np.zeros(n, dtype=complex)
    nonzero = modes != 0.0
    multiplier[nonzero] = 1.0 / (1j * modes[nonzero])
    potential = np.fft.ifft(multiplier * coefficients).real
    potential -= np.mean(potential)
    return x, z, potential


def simulate(
    n: int = 48,
    length: float = 2.0 * math.pi,
    depth: float = 1.0,
    gravity: float = 1.0,
    amplitude: float = 0.16,
    velocity_factor: float = 2.4,
    dt: float = 0.0025,
    final_time: float = 1.2,
    snapshots: int = 121,
    scenario: str = "progressive",
    jet_speed: float = 1.5,
    width: float = 0.75,
    tangential_mode: str = "theta_s",
) -> dict[str, Array]:
    if scenario == "progressive":
        x, z, potential = initial_progressive_wave(
            n, length, depth, gravity, amplitude, velocity_factor
        )
    elif scenario == "localized":
        x, z, potential = initial_localized_breaker(
            n, length, amplitude, jet_speed, width
        )
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    if tangential_mode == "theta_s":
        x, z, potential = uniform_arclength_resample(x, z, potential, length)
    steps = int(math.ceil(final_time / dt))
    actual_dt = final_time / steps
    save_steps = set(np.linspace(0, steps, snapshots, dtype=int).tolist())
    history_x, history_z, times = [], [], []
    volumes, energies, minimum_x_alpha, marker_cv, residuals = [], [], [], [], []

    def save(step: int, residual: float) -> None:
        volume, energy, min_derivative, spacing_cv = invariants(
            x, z, potential, length, depth, gravity
        )
        history_x.append(x.copy())
        history_z.append(z.copy())
        times.append(step * actual_dt)
        volumes.append(volume)
        energies.append(energy)
        minimum_x_alpha.append(min_derivative)
        marker_cv.append(spacing_cv)
        residuals.append(residual)

    save(0, 0.0)
    termination_reason = "completed"
    termination_step = steps
    for step in range(1, steps + 1):
        previous_x, previous_z, previous_potential = x, z, potential
        try:
            candidate = step_rk4(
                x,
                z,
                potential,
                actual_dt,
                length,
                depth,
                gravity,
                tangential_mode,
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
            termination_reason = f"solver_failure: {type(error).__name__}"
            termination_step = step - 1
            break
        x, z, potential, residual = candidate
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(z))):
            x, z, potential = previous_x, previous_z, previous_potential
            termination_reason = "non_finite_candidate"
            termination_step = step - 1
            break
        if np.min(z) <= -0.95 * depth:
            x, z, potential = previous_x, previous_z, previous_potential
            termination_reason = "surface_approached_mirror_bed"
            termination_step = step - 1
            break
        if step in save_steps:
            save(step, residual)
    if termination_reason != "completed" and (
        not times or times[-1] < termination_step * actual_dt
    ):
        save(termination_step, 0.0)
    return {
        "x": np.asarray(history_x),
        "z": np.asarray(history_z),
        "time": np.asarray(times),
        "volume": np.asarray(volumes),
        "energy": np.asarray(energies),
        "min_x_alpha": np.asarray(minimum_x_alpha),
        "marker_cv": np.asarray(marker_cv),
        "bie_residual": np.asarray(residuals),
        "length": np.asarray(length),
        "depth": np.asarray(depth),
        "gravity": np.asarray(gravity),
        "dt": np.asarray(actual_dt),
        "termination_reason": np.asarray(termination_reason),
        "termination_step": np.asarray(termination_step),
        "tangential_mode": np.asarray(tangential_mode),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--amplitude", type=float, default=0.16)
    parser.add_argument("--velocity-factor", type=float, default=2.4)
    parser.add_argument("--scenario", choices=("progressive", "localized"), default="progressive")
    parser.add_argument("--jet-speed", type=float, default=1.5)
    parser.add_argument("--width", type=float, default=0.75)
    parser.add_argument(
        "--tangential-mode",
        choices=("lagrangian", "theta_s"),
        default="theta_s",
    )
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--final-time", type=float, default=1.2)
    parser.add_argument("--snapshots", type=int, default=121)
    parser.add_argument(
        "--output-prefix", type=Path, default=Path("results/lagrangian_breaker")
    )
    args = parser.parse_args()
    result = simulate(
        n=args.n,
        amplitude=args.amplitude,
        velocity_factor=args.velocity_factor,
        dt=args.dt,
        final_time=args.final_time,
        snapshots=args.snapshots,
        scenario=args.scenario,
        jet_speed=args.jet_speed,
        width=args.width,
        tangential_mode=args.tangential_mode,
    )
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prefix.with_suffix(".npz"), **result)
    energy_scale = max(abs(float(result["energy"][0])), np.finfo(float).eps)
    metrics = {
        "n": args.n,
        "amplitude": args.amplitude,
        "velocity_factor": args.velocity_factor,
        "scenario": args.scenario,
        "jet_speed": args.jet_speed,
        "width": args.width,
        "tangential_mode": args.tangential_mode,
        "dt": float(result["dt"]),
        "final_time": float(result["time"][-1]),
        "minimum_x_alpha": float(np.min(result["min_x_alpha"])),
        "overturned": bool(np.min(result["min_x_alpha"]) < 0.0),
        "max_relative_energy_drift": float(
            np.max(np.abs(result["energy"] - result["energy"][0])) / energy_scale
        ),
        "max_absolute_volume_drift": float(
            np.max(np.abs(result["volume"] - result["volume"][0]))
        ),
        "max_bie_residual": float(np.max(result["bie_residual"])),
        "max_marker_spacing_cv": float(np.max(result["marker_cv"])),
        "termination_reason": str(result["termination_reason"]),
        "termination_step": int(result["termination_step"]),
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
