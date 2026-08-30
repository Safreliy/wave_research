"""Solitary-wave initial data for publication benchmarks.

The third-order free-surface and surface-potential formula follows the open
surftank implementation used by Pick & Feddersen (JFM, 2026), itself based on
the classical fully nonlinear solitary-wave benchmark literature.  A linear
potential jump is separated as a uniform background current so that the
remaining trace is periodic and can be represented spectrally.

The fully nonlinear option solves Babenko's equation with the Petviashvili
iteration described by Dutykh & Clamond, Wave Motion 51 (2014), 86--99,
https://doi.org/10.1016/j.wavemoti.2013.06.007.  It reconstructs both the
physical surface coordinate and the laboratory-frame surface potential from
the same conformal solution.  Thus it is an Euler travelling wave, rather
than an asymptotic profile paired with an independently approximated velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SolitaryWaveInitialState:
    elevation: Array
    periodic_potential: Array
    full_surface_potential: Array
    background_current: float
    speed: float
    inverse_width: float
    method: str = "third_order"
    equation_residual: float = float("nan")
    iterations: int = 0


@dataclass(frozen=True)
class _BabenkoSolution:
    eta: Array
    c_eta: Array
    speed: float
    conformal_length: float
    residual: float
    iterations: int


def _babenko_symbols(
    n: int, conformal_length: float, depth: float
) -> tuple[Array, Array]:
    modes = 2.0 * math.pi * np.fft.fftfreq(
        n, d=conformal_length / n
    )
    c_symbol = np.empty(n, dtype=float)
    nonzero = modes != 0.0
    c_symbol[nonzero] = modes[nonzero] / np.tanh(
        modes[nonzero] * depth
    )
    c_symbol[~nonzero] = 1.0 / depth
    return c_symbol, 1.0 / c_symbol


def _solve_babenko_at_speed(
    speed: float,
    conformal_length: float,
    depth: float,
    gravity: float,
    internal_points: int,
    tolerance: float,
    max_iterations: int,
) -> _BabenkoSolution:
    long_wave_speed = math.sqrt(gravity * depth)
    if speed <= long_wave_speed:
        raise ValueError("solitary-wave speed must exceed sqrt(g*depth)")
    alpha = (
        np.arange(internal_points, dtype=float) - internal_points // 2
    ) * conformal_length / internal_points
    c_symbol, c_inverse_symbol = _babenko_symbols(
        internal_points, conformal_length, depth
    )
    linear_symbol = speed**2 - gravity * c_inverse_symbol
    amplitude_guess = speed**2 / gravity - depth
    inverse_width = math.sqrt(
        3.0 * amplitude_guess / (4.0 * depth**3)
    )
    eta = amplitude_guess / np.cosh(inverse_width * alpha) ** 2
    error = float("inf")
    stabilizer = float("nan")
    for iteration in range(1, max_iterations + 1):
        c_eta = np.fft.ifft(c_symbol * np.fft.fft(eta)).real
        nonlinear = gravity * np.fft.ifft(
            c_inverse_symbol * np.fft.fft(eta * c_eta)
        ).real + 0.5 * gravity * eta**2
        linear_eta = np.fft.ifft(
            linear_symbol * np.fft.fft(eta)
        ).real
        denominator = float(np.vdot(eta, nonlinear).real)
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise FloatingPointError("invalid Petviashvili denominator")
        stabilizer = float(np.vdot(eta, linear_eta).real / denominator)
        next_eta = np.fft.ifft(
            np.fft.fft(nonlinear) / linear_symbol
        ).real * stabilizer**2
        # The desired branch is an even positive elevation wave.  Enforcing
        # its exact discrete symmetry removes round-off drift of the crest.
        next_eta = 0.5 * (
            next_eta + np.roll(next_eta[::-1], 1)
        )
        error = float(np.max(np.abs(next_eta - eta)))
        eta = next_eta
        if error < tolerance and abs(stabilizer - 1.0) < 10.0 * tolerance:
            break
    else:
        raise RuntimeError(
            "Petviashvili iteration did not converge: "
            f"last update={error:.3e}, stabilizer={stabilizer:.12g}"
        )
    c_eta = np.fft.ifft(c_symbol * np.fft.fft(eta)).real
    nonlinear = gravity * np.fft.ifft(
        c_inverse_symbol * np.fft.fft(eta * c_eta)
    ).real + 0.5 * gravity * eta**2
    linear_eta = np.fft.ifft(
        linear_symbol * np.fft.fft(eta)
    ).real
    residual = float(
        np.max(np.abs(linear_eta - nonlinear))
        / max(gravity * float(np.max(eta)), np.finfo(float).eps)
    )
    return _BabenkoSolution(
        eta=eta,
        c_eta=c_eta,
        speed=speed,
        conformal_length=conformal_length,
        residual=residual,
        iterations=iteration,
    )


def _babenko_for_amplitude(
    amplitude: float,
    conformal_length: float,
    depth: float,
    gravity: float,
    internal_points: int,
    tolerance: float,
    max_iterations: int,
) -> _BabenkoSolution:
    epsilon = amplitude / depth
    if not 0.0 < epsilon <= 0.65:
        raise ValueError(
            "fully nonlinear initializer requires 0 < amplitude/depth <= 0.65"
        )
    base_speed = math.sqrt(gravity * depth)
    lower = base_speed * math.sqrt(1.0 + 0.55 * epsilon)
    upper = base_speed * math.sqrt(1.0 + 1.10 * epsilon)

    def solve(speed: float) -> _BabenkoSolution:
        return _solve_babenko_at_speed(
            speed,
            conformal_length,
            depth,
            gravity,
            internal_points,
            tolerance,
            max_iterations,
        )

    lower_solution = solve(lower)
    upper_solution = solve(upper)
    lower_amplitude = float(np.max(lower_solution.eta))
    upper_amplitude = float(np.max(upper_solution.eta))
    if lower_amplitude >= amplitude:
        raise RuntimeError("lower speed does not bracket the requested amplitude")
    if upper_amplitude <= amplitude:
        raise RuntimeError("upper speed does not bracket the requested amplitude")
    solution = upper_solution
    amplitude_tolerance = max(tolerance, tolerance * amplitude)
    for _ in range(24):
        # A safeguarded secant step is substantially cheaper than pure
        # bisection because every trial entails a nonlinear FFT solve.
        fraction = (amplitude - lower_amplitude) / (
            upper_amplitude - lower_amplitude
        )
        fraction = min(max(fraction, 0.1), 0.9)
        midpoint = lower + fraction * (upper - lower)
        solution = solve(midpoint)
        midpoint_amplitude = float(np.max(solution.eta))
        difference = midpoint_amplitude - amplitude
        if abs(difference) < amplitude_tolerance:
            break
        if difference < 0.0:
            lower = midpoint
            lower_amplitude = midpoint_amplitude
        else:
            upper = midpoint
            upper_amplitude = midpoint_amplitude
    return solution


def fully_nonlinear_periodic_solitary_wave(
    x: Array,
    length: float,
    center: float,
    amplitude: float,
    depth: float,
    gravity: float = 1.0,
    internal_points: int = 2048,
    tolerance: float = 2.0e-13,
    max_iterations: int = 4000,
) -> SolitaryWaveInitialState:
    """Return a Babenko--Petviashvili Euler solitary-wave initial state.

    The infinite line is represented by one long periodic cell.  Its
    conformal length is iterated so that the reconstructed *physical* period
    equals ``length``.  ``x`` may be any physical sampling grid in that cell.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("x must be a one-dimensional grid")
    if length <= 0.0 or depth <= 0.0 or gravity <= 0.0:
        raise ValueError("length, depth and gravity must be positive")
    if internal_points < 256 or internal_points % 2:
        raise ValueError("internal_points must be an even integer >= 256")
    if length < 8.0 * depth:
        raise ValueError(
            "the Babenko solitary-wave cell must be at least eight depths long"
        )

    conformal_length = length
    solution: _BabenkoSolution | None = None
    for _ in range(16):
        solution = _babenko_for_amplitude(
            amplitude,
            conformal_length,
            depth,
            gravity,
            internal_points,
            tolerance,
            max_iterations,
        )
        physical_length = conformal_length * (
            1.0 + float(np.mean(solution.c_eta))
        )
        mismatch = physical_length - length
        if abs(mismatch) < 5.0e-12 * length:
            break
        # Since mean(C eta) = mean(eta)/depth, the physical period is the
        # conformal period plus the wave mass divided by depth.  Subtracting
        # that mass is the direct fixed-point update and converges in 2--4
        # iterations for the benchmark amplitudes.
        conformal_length = length - conformal_length * float(
            np.mean(solution.c_eta)
        )
    else:
        raise RuntimeError("physical-period iteration did not converge")
    assert solution is not None

    n_internal = len(solution.eta)
    alpha = (
        np.arange(n_internal, dtype=float) - n_internal // 2
    ) * conformal_length / n_internal
    mean_c_eta = float(np.mean(solution.c_eta))
    fluctuating = solution.c_eta - mean_c_eta
    modes = 2.0 * math.pi * np.fft.fftfreq(
        n_internal, d=conformal_length / n_internal
    )
    inverse_derivative = np.zeros(n_internal, dtype=complex)
    nonzero = modes != 0.0
    inverse_derivative[nonzero] = 1.0 / (1j * modes[nonzero])
    periodic_x_shift = np.fft.ifft(
        inverse_derivative * np.fft.fft(fluctuating)
    ).real
    crest_index = int(np.argmax(solution.eta))
    periodic_x_shift -= periodic_x_shift[crest_index]
    physical_relative_x = (
        alpha + mean_c_eta * alpha + periodic_x_shift
    )
    reconstructed_length = conformal_length * (1.0 + mean_c_eta)
    if abs(reconstructed_length - length) > 2.0e-10 * length:
        raise RuntimeError("reconstructed physical period is inconsistent")

    full_relative_potential = solution.speed * (
        mean_c_eta * alpha + periodic_x_shift
    )
    potential_jump = solution.speed * mean_c_eta * conformal_length
    background_current = potential_jump / length
    physical_coordinate = center + physical_relative_x
    periodic_potential = (
        full_relative_potential
        - background_current * physical_relative_x
    )

    wrapped = physical_coordinate % length
    order = np.argsort(wrapped)
    wrapped = wrapped[order]
    eta_sorted = solution.eta[order]
    potential_sorted = periodic_potential[order]
    wrapped_extended = np.concatenate((wrapped, [wrapped[0] + length]))
    eta_extended = np.concatenate((eta_sorted, [eta_sorted[0]]))
    potential_extended = np.concatenate(
        (potential_sorted, [potential_sorted[0]])
    )
    query = x % length
    query = np.where(query < wrapped[0], query + length, query)
    elevation = np.interp(query, wrapped_extended, eta_extended)
    sampled_periodic_potential = np.interp(
        query, wrapped_extended, potential_extended
    )
    sampled_periodic_potential -= np.mean(sampled_periodic_potential)
    full_surface_potential = (
        sampled_periodic_potential + background_current * x
    )
    inverse_width = math.sqrt(
        3.0 * amplitude / (4.0 * depth**3)
    )
    return SolitaryWaveInitialState(
        elevation=elevation,
        periodic_potential=sampled_periodic_potential,
        full_surface_potential=full_surface_potential,
        background_current=float(background_current),
        speed=float(solution.speed),
        inverse_width=inverse_width,
        method="babenko_petviashvili",
        equation_residual=solution.residual,
        iterations=solution.iterations,
    )


def _profile_and_potential(
    x: Array,
    center: float,
    amplitude: float,
    depth: float,
    gravity: float,
) -> tuple[Array, Array, float, float]:
    if amplitude <= 0.0:
        raise ValueError("solitary-wave amplitude must be positive")
    if depth <= 0.0 or gravity <= 0.0:
        raise ValueError("depth and gravity must be positive")
    epsilon = amplitude / depth
    if epsilon > 0.65:
        raise ValueError("third-order initializer is restricted to amplitude/depth <= 0.65")
    leading_inverse_width = math.sqrt(3.0 * amplitude / (4.0 * depth**3))
    inverse_width = leading_inverse_width * (
        1.0 - 5.0 * epsilon / 8.0 + 71.0 * epsilon**2 / 128.0
    )
    argument = inverse_width * (np.asarray(x, dtype=float) - center)
    sech_squared = 1.0 / np.cosh(argument) ** 2
    tanh = np.tanh(argument)
    elevation = depth * (
        epsilon * sech_squared
        - 0.75 * epsilon**2 * (sech_squared - sech_squared**2)
        + epsilon**3
        * (
            5.0 * sech_squared / 8.0
            - 151.0 * sech_squared**2 / 80.0
            + 101.0 * sech_squared**3 / 80.0
        )
    )
    sech2_tanh = sech_squared * tanh
    sech4_tanh = sech_squared**2 * tanh
    potential = (
        epsilon
        * math.sqrt(gravity * depth)
        / leading_inverse_width
        * (
            tanh
            + epsilon
            * (
                5.0 * tanh / 24.0
                - sech2_tanh / 3.0
                + 0.75 * (1.0 + elevation / depth) ** 2 * sech2_tanh
            )
            + epsilon**2
            * (
                -1257.0 * tanh / 3200.0
                + 9.0 * sech2_tanh / 200.0
                + 6.0 * sech4_tanh / 25.0
                + (1.0 + elevation / depth) ** 2
                * (-9.0 * sech2_tanh / 32.0 - 1.5 * sech4_tanh)
                + (1.0 + elevation / depth) ** 4
                * (-3.0 * sech2_tanh / 16.0 + 9.0 * sech4_tanh / 16.0)
            )
        )
    )
    speed = math.sqrt(gravity * depth * (1.0 + epsilon))
    return elevation, potential, speed, inverse_width


def periodic_solitary_wave(
    x: Array,
    length: float,
    center: float,
    amplitude: float,
    depth: float,
    gravity: float = 1.0,
) -> SolitaryWaveInitialState:
    """Return a localized right-going wave plus an exact uniform-flux split."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("x must be a one-dimensional grid")
    if length <= 0.0:
        raise ValueError("length must be positive")
    elevation, full_potential, speed, inverse_width = _profile_and_potential(
        x, center, amplitude, depth, gravity
    )
    endpoint_x = np.asarray([0.0, length])
    _, endpoint_potential, _, _ = _profile_and_potential(
        endpoint_x, center, amplitude, depth, gravity
    )
    background_current = float(
        (endpoint_potential[1] - endpoint_potential[0]) / length
    )
    periodic_potential = full_potential - background_current * x
    periodic_potential -= np.mean(periodic_potential)
    return SolitaryWaveInitialState(
        elevation=elevation,
        periodic_potential=periodic_potential,
        full_surface_potential=full_potential,
        background_current=background_current,
        speed=speed,
        inverse_width=inverse_width,
    )


def full_width_above_threshold(
    amplitude: float,
    depth: float,
    relative_threshold: float = 1.0e-3,
) -> float:
    """Leading-order estimate used only to size the offshore flat section."""
    if not 0.0 < relative_threshold < amplitude / depth:
        raise ValueError("threshold must lie below the dimensionless amplitude")
    inverse_width = math.sqrt(3.0 * amplitude / (4.0 * depth**3)) * (
        1.0
        - 5.0 * amplitude / (8.0 * depth)
        + 71.0 * amplitude**2 / (128.0 * depth**2)
    )
    return 2.0 * math.acosh(math.sqrt((amplitude / depth) / relative_threshold)) / inverse_width
