"""Pre-impact parametric Euler solver over a smooth periodic shoal."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from impact_detection import periodic_impact_diagnostics
from lagrangian_bie_solver import (
    monitor_equidistribution_cv,
    spectral_arclength_resample,
)
from implicit_midpoint import (
    ImplicitDiagnostics,
    implicit_midpoint_step as advance_implicit_midpoint,
)
from solitary_wave import (
    fully_nonlinear_periodic_solitary_wave,
    periodic_solitary_wave,
)
from topography_bie import TopographyBIE


Array = np.ndarray


@dataclass(frozen=True)
class WaveStepDiagnostics:
    nonlinear: ImplicitDiagnostics
    maximum_bie_residual: float
    used_explicit_fallback: bool


def periodic_distance(x: Array, center: float, length: float) -> Array:
    return (x - center + 0.5 * length) % length - 0.5 * length


def minimum_horizontal_mapping(x: Array, length: float) -> float:
    """Minimum x_alpha for a curve sampled uniformly in its parameter."""
    n = len(x)
    base = length * np.arange(n) / n
    modes = np.fft.fftfreq(n, d=1.0 / n)
    derivative = np.fft.ifft(
        1j * modes * np.fft.fft(np.asarray(x) - base)
    ).real
    return float(np.min(length / (2.0 * math.pi) + derivative))


def smooth_periodic_shoal(
    n: int,
    length: float,
    depth: float,
    height: float,
    center: float,
) -> tuple[Array, Array]:
    if not 0.0 <= height < depth:
        raise ValueError("shoal height must lie between zero (inclusive) and depth")
    x = np.arange(n) * length / n
    if height == 0.0:
        return x, -depth * np.ones(n)
    phase = 2.0 * math.pi * (x - center) / length
    z = -depth + 0.5 * height * (1.0 + np.cos(phase))
    return x, z


def smooth_reef_bathymetry(
    n: int,
    length: float,
    depth: float,
    shallow_depth: float,
    slope: float,
    toe: float,
    shelf_length: float,
    return_length: float,
    corner_width: float,
    cluster_strength: float = 0.0,
    cluster_width: float | None = None,
) -> tuple[Array, Array]:
    """Smooth periodic surrogate of the flat--slope--reef JFM benchmark.

    The event of interest occurs on the upslope section.  A smooth downslope is
    placed far downstream only to close the periodic computational cell.
    """
    if not 0.0 < shallow_depth < depth:
        raise ValueError("shallow_depth must lie between zero and depth")
    if min(slope, shelf_length, return_length, corner_width) <= 0.0:
        raise ValueError("reef lengths, slope and corner_width must be positive")
    rise = depth - shallow_depth
    slope_length = rise / slope
    shelf_start = toe + slope_length
    shelf_end = shelf_start + shelf_length
    return_end = shelf_end + return_length
    margin = 5.0 * corner_width
    if toe < margin or return_end > length - margin:
        raise ValueError("reef transitions need five corner widths of periodic margin")
    if cluster_strength < 0.0:
        raise ValueError("cluster_strength cannot be negative")
    if cluster_width is None:
        cluster_width = max(4.0 * corner_width, 0.25)
    if cluster_width <= 0.0:
        raise ValueError("cluster_width must be positive")
    if cluster_strength:
        # The fixed and free boundaries have independent smooth periodic
        # parameterizations.  Concentrating bottom nodes at the four rounded
        # reef transitions resolves the geometry and the thin shelf gap
        # without wasting free-surface markers over long flat reaches.
        dense_count = max(16384, 128 * n)
        dense_x = np.linspace(0.0, length, dense_count + 1)
        monitor = np.ones_like(dense_x)
        for transition in (toe, shelf_start, shelf_end, return_end):
            monitor += cluster_strength * np.exp(
                -((dense_x - transition) / cluster_width) ** 2
            )
        cumulative = np.zeros_like(dense_x)
        cumulative[1:] = np.cumsum(
            0.5 * (monitor[1:] + monitor[:-1]) * np.diff(dense_x)
        )
        targets = np.arange(n) * cumulative[-1] / n
        x = np.interp(targets, cumulative, dense_x)
    else:
        x = np.arange(n) * length / n

    def softplus(value: Array) -> Array:
        return corner_width * np.logaddexp(0.0, value / corner_width)

    upslope = (
        softplus(x - toe) - softplus(x - shelf_start)
    ) / slope_length
    downslope = (
        softplus(x - shelf_end) - softplus(x - return_end)
    ) / return_length
    elevation = rise * (upslope - downslope)
    return x, -depth + elevation


def localized_initial_state(
    n: int,
    length: float,
    amplitude: float,
    jet_speed: float,
    width: float,
    center: float,
) -> tuple[Array, Array, Array]:
    x = np.arange(n) * length / n
    distance = periodic_distance(x, center, length)
    z = amplitude * np.exp(-(distance / width) ** 2)
    z -= np.mean(z)
    target_u = jet_speed * np.exp(-(distance / (0.72 * width)) ** 2)
    target_u -= np.mean(target_u)
    modes = 2.0 * math.pi * np.fft.fftfreq(n, d=length / n)
    multiplier = np.zeros(n, dtype=complex)
    nonzero = modes != 0.0
    multiplier[nonzero] = 1.0 / (1j * modes[nonzero])
    potential = np.fft.ifft(multiplier * np.fft.fft(target_u)).real
    potential -= np.mean(potential)
    return x, z, potential


def theta_s_rhs(
    x: Array,
    z: Array,
    potential: Array,
    bottom_x: Array,
    bottom_z: Array,
    length: float,
    gravity: float,
    background_current: float = 0.0,
    tangential_gauge: str = "theta_s",
) -> tuple[Array, Array, Array, float]:
    bie = TopographyBIE(x, z, bottom_x, bottom_z, length)
    geometry = bie.surface
    tangent_x = geometry.x_alpha / geometry.metric
    tangent_z = geometry.z_alpha / geometry.metric
    normal_x = -tangent_z
    normal_z = tangent_x
    bottom_normal_x = bie.bottom.z_alpha / bie.bottom.metric
    bottom_perturbation_flux = -background_current * bottom_normal_x
    result = bie.solve(potential, bottom_perturbation_flux)
    # The continuous mixed DNO maps Dirichlet traces into the globally
    # compatible Neumann subspace.  Collocation and close-evaluation errors can
    # leave a small constant-flux component which is then integrated as a
    # spurious volume mode.  Remove that single nonphysical component before
    # evaluating the kinematic and Bernoulli equations; diagnostics retain the
    # unprojected defect as a spatial-accuracy gate.
    perturbation_flux = bie.dalpha * (
        np.dot(bie.surface.metric, result.surface_normal_derivative)
        + np.dot(bie.bottom.metric, bottom_perturbation_flux)
    )
    flux_projection = perturbation_flux / (
        bie.dalpha * np.sum(bie.surface.metric)
    )
    normal_velocity = (
        result.surface_normal_derivative
        - flux_projection
        + background_current * normal_x
    )
    tangential_fluid_velocity = (
        bie.surface_derivative(potential) + background_current * tangent_x
    )
    if tangential_gauge == "theta_s":
        cross = (
            geometry.x_alpha * geometry.z_alpha_alpha
            - geometry.z_alpha * geometry.x_alpha_alpha
        )
        curvature = cross / geometry.metric**3
        target_derivative = curvature * normal_velocity * geometry.metric
        target_derivative -= np.mean(target_derivative)
        coefficients = np.fft.fft(target_derivative)
        inverse = np.zeros(bie.n, dtype=complex)
        nonzero = bie.alpha_modes != 0.0
        inverse[nonzero] = 1.0 / (1j * bie.alpha_modes[nonzero])
        tangential_grid_velocity = np.fft.ifft(inverse * coefficients).real
        tangential_grid_velocity += np.mean(tangential_fluid_velocity)
    elif tangential_gauge == "lagrangian":
        tangential_grid_velocity = tangential_fluid_velocity
    else:
        raise ValueError("tangential_gauge must be 'theta_s' or 'lagrangian'")
    x_t = normal_velocity * normal_x + tangential_grid_velocity * tangent_x
    z_t = normal_velocity * normal_z + tangential_grid_velocity * tangent_z
    potential_t = (
        0.5 * normal_velocity**2
        - 0.5 * tangential_fluid_velocity**2
        - background_current * normal_velocity * normal_x
        + tangential_grid_velocity * tangential_fluid_velocity
        - background_current * tangential_grid_velocity * tangent_x
        - gravity * z
    )
    potential_t -= np.mean(potential_t)
    return x_t, z_t, potential_t, result.residual


def step_rk4(
    x: Array,
    z: Array,
    potential: Array,
    bottom_x: Array,
    bottom_z: Array,
    length: float,
    gravity: float,
    dt: float,
    background_current: float = 0.0,
    tangential_gauge: str = "theta_s",
) -> tuple[Array, Array, Array, float]:
    def evaluate(stage_x: Array, stage_z: Array, stage_potential: Array):
        return theta_s_rhs(
            stage_x,
            stage_z,
            stage_potential,
            bottom_x,
            bottom_z,
            length,
            gravity,
            background_current,
            tangential_gauge,
        )

    k1 = evaluate(x, z, potential)
    k2 = evaluate(
        x + 0.5 * dt * k1[0],
        z + 0.5 * dt * k1[1],
        potential + 0.5 * dt * k1[2],
    )
    k3 = evaluate(
        x + 0.5 * dt * k2[0],
        z + 0.5 * dt * k2[1],
        potential + 0.5 * dt * k2[2],
    )
    k4 = evaluate(
        x + dt * k3[0],
        z + dt * k3[1],
        potential + dt * k3[2],
    )
    next_x = x + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
    next_z = z + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
    next_potential = potential + dt * (
        k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]
    ) / 6.0
    next_potential -= np.mean(next_potential)
    return next_x, next_z, next_potential, max(k1[3], k2[3], k3[3], k4[3])


def step_implicit_midpoint(
    x: Array,
    z: Array,
    potential: Array,
    bottom_x: Array,
    bottom_z: Array,
    length: float,
    gravity: float,
    dt: float,
    background_current: float = 0.0,
    tangential_gauge: str = "theta_s",
    nonlinear_tolerance: float = 2.0e-10,
    maximum_newton_iterations: int = 7,
    maximum_krylov_iterations: int = 20,
    allow_explicit_fallback: bool = True,
) -> tuple[Array, Array, Array, float, WaveStepDiagnostics]:
    """Fully implicit midpoint step solved by Jacobian-free Newton--GMRES."""
    n = len(x)
    initial = np.concatenate((x, z, potential))
    maximum_bie_residual = 0.0

    def vector_rhs(state: Array) -> Array:
        nonlocal maximum_bie_residual
        velocity = theta_s_rhs(
            state[:n],
            state[n : 2 * n],
            state[2 * n :],
            bottom_x,
            bottom_z,
            length,
            gravity,
            background_current,
            tangential_gauge,
        )
        maximum_bie_residual = max(maximum_bie_residual, velocity[3])
        return np.concatenate(velocity[:3])

    try:
        candidate, nonlinear = advance_implicit_midpoint(
            initial,
            vector_rhs,
            dt,
            nonlinear_tolerance=nonlinear_tolerance,
            maximum_newton_iterations=maximum_newton_iterations,
            maximum_krylov_iterations=maximum_krylov_iterations,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        candidate = initial
        nonlinear = ImplicitDiagnostics(
            False, 0, 0, 0, float("inf"), 0
        )
    used_fallback = not nonlinear.converged
    if used_fallback:
        if not allow_explicit_fallback:
            raise RuntimeError("implicit midpoint nonlinear solve did not converge")
        fallback = step_rk4(
            x,
            z,
            potential,
            bottom_x,
            bottom_z,
            length,
            gravity,
            dt,
            background_current,
            tangential_gauge,
        )
        next_x, next_z, next_potential, fallback_residual = fallback
        maximum_bie_residual = max(maximum_bie_residual, fallback_residual)
    else:
        next_x = candidate[:n]
        next_z = candidate[n : 2 * n]
        next_potential = candidate[2 * n :]
        next_potential -= np.mean(next_potential)
    return (
        next_x,
        next_z,
        next_potential,
        maximum_bie_residual,
        WaveStepDiagnostics(nonlinear, maximum_bie_residual, used_fallback),
    )


def diagnostics(
    x: Array,
    z: Array,
    potential: Array,
    bottom_x: Array,
    bottom_z: Array,
    length: float,
    gravity: float,
    background_current: float = 0.0,
) -> tuple[float, float, float, float, float]:
    bie = TopographyBIE(x, z, bottom_x, bottom_z, length)
    bottom_normal_x = bie.bottom.z_alpha / bie.bottom.metric
    bottom_flux = -background_current * bottom_normal_x
    result = bie.solve(potential, bottom_flux)
    volume = bie.dalpha * np.sum(
        z * bie.surface.x_alpha - bottom_z * bie.bottom.x_alpha
    )
    perturbation_kinetic = 0.5 * bie.dalpha * (
        np.sum(
            potential * result.surface_normal_derivative * bie.surface.metric
        )
        + np.sum(result.bottom_potential * bottom_flux * bie.bottom.metric)
    )
    cross_kinetic = background_current * bie.dalpha * (
        np.sum(potential * (-bie.surface.z_alpha))
        + np.sum(result.bottom_potential * bie.bottom.z_alpha)
    )
    kinetic = (
        perturbation_kinetic
        + cross_kinetic
        + 0.5 * background_current**2 * volume
    )
    potential_energy = 0.5 * gravity * bie.dalpha * np.sum(
        z**2 * bie.surface.x_alpha - bottom_z**2 * bie.bottom.x_alpha
    )
    min_x_alpha = float(np.min(bie.surface.x_alpha))
    marker_cv = float(np.std(bie.surface.metric) / np.mean(bie.surface.metric))
    surface_normal_x = -bie.surface.z_alpha / bie.surface.metric
    total_surface_flux = bie.dalpha * np.sum(
        (
            result.surface_normal_derivative
            + background_current * surface_normal_x
        )
        * bie.surface.metric
    )
    return (
        float(volume),
        float(kinetic + potential_energy),
        min_x_alpha,
        marker_cv,
        float(total_surface_flux),
    )


def geometric_volume(
    x: Array,
    z: Array,
    bottom_x: Array,
    bottom_z: Array,
    length: float,
) -> float:
    """Liquid area from the two periodic parametric boundaries only."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    bottom_x = np.asarray(bottom_x, dtype=float)
    bottom_z = np.asarray(bottom_z, dtype=float)
    if not (x.shape == z.shape == bottom_x.shape == bottom_z.shape):
        raise ValueError("surface and bottom arrays must have identical shapes")
    n = len(x)
    modes = np.fft.fftfreq(n, d=1.0 / n)
    base = length * np.arange(n) / n
    x_alpha = length / (2.0 * math.pi) + np.fft.ifft(
        1j * modes * np.fft.fft(x - base)
    ).real
    bottom_x_alpha = length / (2.0 * math.pi) + np.fft.ifft(
        1j * modes * np.fft.fft(bottom_x - base)
    ).real
    dalpha = 2.0 * math.pi / n
    return float(
        dalpha * np.sum(z * x_alpha - bottom_z * bottom_x_alpha)
    )


def spectral_viscosity_filter(
    x: Array,
    z: Array,
    potential: Array,
    length: float,
    strength: float,
    order: int = 16,
) -> tuple[Array, Array, Array, float]:
    """Damp only the top of the Fourier spectrum of a periodic surface.

    The map displacement ``x-L alpha/(2 pi)``, elevation and potential trace
    receive the same exponential multiplier.  The zero mode is unchanged;
    volume is handled separately by the geometric projection.  The returned
    correction is the relative Euclidean change of the three state vectors
    and is recorded as a numerical-dissipation diagnostic.
    """
    if strength < 0.0:
        raise ValueError("spectral filter strength cannot be negative")
    if order < 2 or order % 2:
        raise ValueError("spectral filter order must be a positive even integer")
    arrays = [np.asarray(values, dtype=float) for values in (x, z, potential)]
    if any(values.ndim != 1 for values in arrays):
        raise ValueError("x, z and potential must be one-dimensional")
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError("x, z and potential must have matching shapes")
    if strength == 0.0:
        return arrays[0].copy(), arrays[1].copy(), arrays[2].copy(), 0.0
    n = len(arrays[0])
    base = length * np.arange(n) / n
    scaled_mode = np.abs(np.fft.fftfreq(n, d=1.0 / n)) / (n / 2.0)
    multiplier = np.exp(-strength * scaled_mode**order)
    filtered = []
    for values, reference in zip(arrays, (base, 0.0, 0.0)):
        periodic_part = values - reference
        candidate = np.fft.ifft(
            multiplier * np.fft.fft(periodic_part)
        ).real + reference
        filtered.append(candidate)
    filtered[2] -= np.mean(filtered[2])
    before = np.concatenate((arrays[0] - base, arrays[1], arrays[2]))
    after = np.concatenate((filtered[0] - base, filtered[1], filtered[2]))
    correction = float(
        np.linalg.norm(after - before)
        / max(float(np.linalg.norm(before)), np.finfo(float).eps)
    )
    return filtered[0], filtered[1], filtered[2], correction


def reference_energy(
    bottom_x: Array,
    bottom_z: Array,
    length: float,
    gravity: float,
    background_current: float = 0.0,
) -> float:
    """Energy of the flat-surface hydrostatic/background-current reference.

    Subtracting this constant before forming a *relative* conservation error
    avoids hiding wave-energy drift behind the much larger bed datum energy.
    """
    bottom_x = np.asarray(bottom_x, dtype=float)
    bottom_z = np.asarray(bottom_z, dtype=float)
    n = len(bottom_x)
    modes = np.fft.fftfreq(n, d=1.0 / n)
    base = length * np.arange(n) / n
    bottom_x_alpha = length / (2.0 * math.pi) + np.fft.ifft(
        1j * modes * np.fft.fft(bottom_x - base)
    ).real
    dalpha = 2.0 * math.pi / n
    rest_volume = -dalpha * np.sum(bottom_z * bottom_x_alpha)
    rest_potential = -0.5 * gravity * dalpha * np.sum(
        bottom_z**2 * bottom_x_alpha
    )
    return float(rest_potential + 0.5 * background_current**2 * rest_volume)


def simulate(
    n: int = 48,
    length: float = 2.0 * math.pi,
    depth: float = 1.2,
    shoal_height: float = 0.70,
    shoal_center: float = 4.7,
    crest_center: float = 3.0,
    amplitude: float = 0.18,
    jet_speed: float = 1.1,
    width: float = 0.72,
    gravity: float = 1.0,
    dt: float = 0.001,
    final_time: float = 1.0,
    snapshots: int = 121,
    initial_condition: str = "localized",
    bathymetry: str = "cosine",
    shallow_depth: float = 0.25,
    reef_slope: float = 0.1,
    reef_toe: float = 10.0,
    shelf_length: float = 3.0,
    return_length: float = 3.0,
    corner_width: float = 0.12,
    reef_cluster_strength: float = 0.0,
    reef_cluster_width: float | None = None,
    exact_internal_points: int = 2048,
    stop_at_overturning: bool = False,
    stop_at_impact: bool = False,
    impact_distance_factor: float = 0.15,
    contact_arc_exclusion: float = 12.0,
    reparameterize_every: int = 0,
    reparameterization_strength: float = 0.0,
    reparameterization_power: float = 0.5,
    time_integrator: str = "rk4",
    tangential_gauge: str = "theta_s",
    implicit_tolerance: float = 2.0e-10,
    implicit_fallback: bool = True,
    project_volume: bool = False,
    spectral_filter_strength: float = 0.0,
    spectral_filter_order: int = 16,
    initial_reparameterize: bool = True,
) -> dict[str, Array]:
    if reparameterize_every < 0:
        raise ValueError("reparameterize_every cannot be negative")
    if impact_distance_factor <= 0.0:
        raise ValueError("impact_distance_factor must be positive")
    if contact_arc_exclusion <= 0.0:
        raise ValueError("contact_arc_exclusion must be positive")
    if reparameterization_strength < 0.0 or reparameterization_power <= 0.0:
        raise ValueError("invalid reparameterization monitor")
    if time_integrator not in ("rk4", "implicit_midpoint"):
        raise ValueError("time_integrator must be 'rk4' or 'implicit_midpoint'")
    if tangential_gauge not in ("theta_s", "lagrangian"):
        raise ValueError("tangential_gauge must be 'theta_s' or 'lagrangian'")
    if spectral_filter_strength < 0.0:
        raise ValueError("spectral_filter_strength cannot be negative")
    if spectral_filter_order < 2 or spectral_filter_order % 2:
        raise ValueError("spectral_filter_order must be a positive even integer")
    if bathymetry == "cosine":
        bottom_x, bottom_z = smooth_periodic_shoal(
            n, length, depth, shoal_height, shoal_center
        )
    elif bathymetry == "reef":
        bottom_x, bottom_z = smooth_reef_bathymetry(
            n,
            length,
            depth,
            shallow_depth,
            reef_slope,
            reef_toe,
            shelf_length,
            return_length,
            corner_width,
            reef_cluster_strength,
            reef_cluster_width,
        )
    else:
        raise ValueError("bathymetry must be 'cosine' or 'reef'")
    if initial_condition == "localized":
        x, z, potential = localized_initial_state(
            n, length, amplitude, jet_speed, width, crest_center
        )
        background_current = 0.0
        initial_wave_speed = float("nan")
        initial_equation_residual = float("nan")
        initial_wave_method = "localized_prescribed"
    elif initial_condition in (
        "solitary",
        "exact_solitary",
        "projected_exact_solitary",
    ):
        x = np.arange(n) * length / n
        initializer = (
            fully_nonlinear_periodic_solitary_wave
            if initial_condition in ("exact_solitary", "projected_exact_solitary")
            else periodic_solitary_wave
        )
        initializer_arguments = {}
        if initial_condition in ("exact_solitary", "projected_exact_solitary"):
            initializer_arguments["internal_points"] = exact_internal_points
        solitary = initializer(
            x, length, crest_center, amplitude, depth, gravity,
            **initializer_arguments,
        )
        z = solitary.elevation
        potential = solitary.periodic_potential
        background_current = solitary.background_current
        initial_wave_speed = solitary.speed
        initial_equation_residual = solitary.equation_residual
        initial_wave_method = solitary.method
        initial_projection_error = float("nan")
        initial_projection_residual = float("nan")
        initial_projection_flux_correction = float("nan")
    else:
        raise ValueError(
            "initial_condition must be 'localized', 'solitary', "
            "'exact_solitary' or 'projected_exact_solitary'"
        )
    if initial_condition == "localized":
        initial_projection_error = float("nan")
        initial_projection_residual = float("nan")
        initial_projection_flux_correction = float("nan")
    if initial_reparameterize:
        x, z, potential = spectral_arclength_resample(
            x,
            z,
            potential,
            length,
            curvature_strength=reparameterization_strength,
            curvature_power=reparameterization_power,
        )
    if initial_condition == "projected_exact_solitary":
        # Projection and reparameterization do not commute discretely.  Form
        # the target normal velocity and solve the topographic N-to-D problem
        # only after the final initial marker distribution has been chosen.
        flat_bottom_x = np.arange(n) * length / n
        flat_bottom_z = -depth * np.ones(n)
        flat_bie = TopographyBIE(
            x, z, flat_bottom_x, flat_bottom_z, length
        )
        flat_solution = flat_bie.solve(potential)
        flat_normal_x = -flat_bie.surface.z_alpha / flat_bie.surface.metric
        target_normal_velocity = (
            flat_solution.surface_normal_derivative
            + background_current * flat_normal_x
        )
        raw_target = target_normal_velocity.copy()
        flux = flat_bie.dalpha * np.dot(
            flat_bie.surface.metric, target_normal_velocity
        )
        target_normal_velocity -= flux / (
            flat_bie.dalpha * np.sum(flat_bie.surface.metric)
        )
        initial_projection_flux_correction = float(
            np.linalg.norm(target_normal_velocity - raw_target)
            / max(float(np.linalg.norm(raw_target)), np.finfo(float).eps)
        )
        topographic_bie = TopographyBIE(
            x, z, bottom_x, bottom_z, length
        )
        projection = topographic_bie.solve_neumann_to_dirichlet(
            target_normal_velocity, np.zeros(n)
        )
        potential = projection.surface_potential
        projected_check = topographic_bie.solve(potential)
        projection_scale = max(
            float(np.linalg.norm(target_normal_velocity)),
            np.finfo(float).eps,
        )
        initial_projection_error = float(
            np.linalg.norm(
                projected_check.surface_normal_derivative
                - target_normal_velocity
            )
            / projection_scale
        )
        initial_projection_residual = projection.residual
        background_current = 0.0
        initial_wave_method += "+topography_neumann_projection"
    energy_reference = reference_energy(
        bottom_x, bottom_z, length, gravity, background_current
    )
    target_volume = geometric_volume(x, z, bottom_x, bottom_z, length)
    steps = int(math.ceil(final_time / dt))
    actual_dt = final_time / steps
    save_steps = set(np.linspace(0, steps, snapshots, dtype=int).tolist())
    history_x, history_z, history_potential, times = [], [], [], []
    volumes, energies, minimum_x_alpha, marker_cv, residuals, flux_defects = (
        [], [], [], [], [], []
    )
    nonlinear_iterations, nonlinear_residuals, krylov_iterations = [], [], []
    rhs_evaluations, implicit_fallbacks = [], []
    monitor_cv, impact_distance, normalized_impact_distance = [], [], []
    volume_projection_shifts: list[float] = [0.0]
    spectral_filter_relative_corrections: list[float] = [0.0]

    def save(
        step: int,
        residual: float,
        step_diagnostics: WaveStepDiagnostics | None = None,
    ) -> None:
        volume, energy, min_mapping, spacing, flux_defect = diagnostics(
            x,
            z,
            potential,
            bottom_x,
            bottom_z,
            length,
            gravity,
            background_current,
        )
        history_x.append(x.copy())
        history_z.append(z.copy())
        history_potential.append(potential.copy())
        times.append(step * actual_dt)
        volumes.append(volume)
        energies.append(energy)
        minimum_x_alpha.append(min_mapping)
        marker_cv.append(spacing)
        residuals.append(residual)
        flux_defects.append(flux_defect)
        monitor_cv.append(
            monitor_equidistribution_cv(
                x,
                z,
                length,
                reparameterization_strength,
                reparameterization_power,
            )
        )
        if min_mapping <= 0.0:
            impact = periodic_impact_diagnostics(
                x,
                z,
                length,
                minimum_arc_separation_panels=contact_arc_exclusion,
            )
            impact_distance.append(impact.minimum_distance)
            normalized_impact_distance.append(impact.normalized_distance)
        else:
            impact_distance.append(float("inf"))
            normalized_impact_distance.append(float("inf"))
        if step_diagnostics is None:
            nonlinear_iterations.append(0)
            nonlinear_residuals.append(0.0)
            krylov_iterations.append(0)
            rhs_evaluations.append(4 if step else 0)
            implicit_fallbacks.append(False)
        else:
            nonlinear_iterations.append(
                step_diagnostics.nonlinear.newton_iterations
            )
            nonlinear_residuals.append(step_diagnostics.nonlinear.residual)
            krylov_iterations.append(
                step_diagnostics.nonlinear.krylov_iterations
            )
            rhs_evaluations.append(step_diagnostics.nonlinear.rhs_evaluations)
            implicit_fallbacks.append(step_diagnostics.used_explicit_fallback)

    save(0, 0.0)
    termination_reason = "completed"
    termination_step = steps
    for step in range(1, steps + 1):
        previous = (x, z, potential)
        step_diagnostics = None
        try:
            if time_integrator == "rk4":
                candidate = step_rk4(
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
            else:
                implicit_candidate = step_implicit_midpoint(
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
                    allow_explicit_fallback=implicit_fallback,
                )
                candidate = implicit_candidate[:4]
                step_diagnostics = implicit_candidate[4]
        except (
            ValueError,
            FloatingPointError,
            np.linalg.LinAlgError,
            RuntimeError,
        ) as error:
            termination_reason = f"solver_failure: {type(error).__name__}"
            termination_step = step - 1
            break
        x, z, potential, residual = candidate
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(z))):
            x, z, potential = previous
            termination_reason = "non_finite_candidate"
            termination_step = step - 1
            break
        if reparameterize_every and step % reparameterize_every == 0:
            x, z, potential = spectral_arclength_resample(
                x,
                z,
                potential,
                length,
                curvature_strength=reparameterization_strength,
                curvature_power=reparameterization_power,
            )
        if spectral_filter_strength:
            x, z, potential, filter_correction = spectral_viscosity_filter(
                x,
                z,
                potential,
                length,
                spectral_filter_strength,
                spectral_filter_order,
            )
            spectral_filter_relative_corrections.append(filter_correction)
        else:
            spectral_filter_relative_corrections.append(0.0)
        if project_volume:
            current_volume = geometric_volume(
                x, z, bottom_x, bottom_z, length
            )
            volume_shift = (target_volume - current_volume) / length
            z += volume_shift
            volume_projection_shifts.append(float(volume_shift))
        else:
            volume_projection_shifts.append(0.0)
        if stop_at_overturning and minimum_horizontal_mapping(x, length) <= 0.0:
            termination_reason = "first_overturning"
            termination_step = step
            save(step, residual, step_diagnostics)
            break
        if stop_at_impact and minimum_horizontal_mapping(x, length) <= 0.0:
            impact = periodic_impact_diagnostics(
                x,
                z,
                length,
                minimum_arc_separation_panels=contact_arc_exclusion,
            )
            if (
                impact.self_intersection
                or impact.normalized_distance <= impact_distance_factor
            ):
                termination_reason = (
                    "self_intersection"
                    if impact.self_intersection
                    else "first_contact_threshold"
                )
                termination_step = step
                save(step, residual, step_diagnostics)
                break
        if step in save_steps:
            save(step, residual, step_diagnostics)
    if termination_reason != "completed" and times[-1] < termination_step * actual_dt:
        save(termination_step, 0.0)
    return {
        "x": np.asarray(history_x),
        "z": np.asarray(history_z),
        "potential": np.asarray(history_potential),
        "time": np.asarray(times),
        "volume": np.asarray(volumes),
        "energy": np.asarray(energies),
        "energy_reference": np.asarray(energy_reference),
        "min_x_alpha": np.asarray(minimum_x_alpha),
        "marker_cv": np.asarray(marker_cv),
        "bie_residual": np.asarray(residuals),
        "surface_flux_defect": np.asarray(flux_defects),
        "nonlinear_iterations": np.asarray(nonlinear_iterations),
        "nonlinear_residual": np.asarray(nonlinear_residuals),
        "krylov_iterations": np.asarray(krylov_iterations),
        "rhs_evaluations": np.asarray(rhs_evaluations),
        "implicit_fallback": np.asarray(implicit_fallbacks),
        "monitor_equidistribution_cv": np.asarray(monitor_cv),
        "impact_distance": np.asarray(impact_distance),
        "normalized_impact_distance": np.asarray(normalized_impact_distance),
        "bottom_x": bottom_x,
        "bottom_z": bottom_z,
        "length": np.asarray(length),
        "depth": np.asarray(depth),
        "gravity": np.asarray(gravity),
        "dt": np.asarray(actual_dt),
        "termination_reason": np.asarray(termination_reason),
        "termination_step": np.asarray(termination_step),
        "tangential_mode": np.asarray("theta_s"),
        "initial_condition": np.asarray(initial_condition),
        "initial_wave_method": np.asarray(initial_wave_method),
        "initial_wave_speed": np.asarray(initial_wave_speed),
        "initial_equation_residual": np.asarray(initial_equation_residual),
        "initial_projection_error": np.asarray(initial_projection_error),
        "initial_projection_residual": np.asarray(initial_projection_residual),
        "initial_projection_flux_correction": np.asarray(
            initial_projection_flux_correction
        ),
        "background_current": np.asarray(background_current),
        "bathymetry": np.asarray(bathymetry),
        "reef_toe": np.asarray(reef_toe),
        "reef_slope": np.asarray(reef_slope),
        "shallow_depth": np.asarray(shallow_depth),
        "reef_cluster_strength": np.asarray(reef_cluster_strength),
        "reef_cluster_width": np.asarray(
            float("nan") if reef_cluster_width is None else reef_cluster_width
        ),
        "reparameterize_every": np.asarray(reparameterize_every),
        "reparameterization_strength": np.asarray(reparameterization_strength),
        "reparameterization_power": np.asarray(reparameterization_power),
        "impact_distance_factor": np.asarray(impact_distance_factor),
        "contact_arc_exclusion": np.asarray(contact_arc_exclusion),
        "time_integrator": np.asarray(time_integrator),
        "tangential_gauge": np.asarray(tangential_gauge),
        "implicit_tolerance": np.asarray(implicit_tolerance),
        "project_volume": np.asarray(project_volume),
        "volume_projection_shifts": np.asarray(volume_projection_shifts),
        "spectral_filter_strength": np.asarray(spectral_filter_strength),
        "spectral_filter_order": np.asarray(spectral_filter_order),
        "spectral_filter_relative_corrections": np.asarray(
            spectral_filter_relative_corrections
        ),
        "initial_reparameterize": np.asarray(initial_reparameterize),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--length", type=float, default=2.0 * math.pi)
    parser.add_argument("--depth", type=float, default=1.2)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--shoal-height", type=float, default=0.70)
    parser.add_argument("--shoal-center", type=float, default=4.7)
    parser.add_argument(
        "--bathymetry", choices=("cosine", "reef"), default="cosine"
    )
    parser.add_argument("--shallow-depth", type=float, default=0.25)
    parser.add_argument("--reef-slope", type=float, default=0.1)
    parser.add_argument("--reef-toe", type=float, default=10.0)
    parser.add_argument("--shelf-length", type=float, default=3.0)
    parser.add_argument("--return-length", type=float, default=3.0)
    parser.add_argument("--corner-width", type=float, default=0.12)
    parser.add_argument("--reef-cluster-strength", type=float, default=0.0)
    parser.add_argument("--reef-cluster-width", type=float, default=None)
    parser.add_argument("--exact-internal-points", type=int, default=2048)
    parser.add_argument("--stop-at-overturning", action="store_true")
    parser.add_argument("--stop-at-impact", action="store_true")
    parser.add_argument("--impact-distance-factor", type=float, default=0.15)
    parser.add_argument("--contact-arc-exclusion", type=float, default=12.0)
    parser.add_argument("--reparameterize-every", type=int, default=0)
    parser.add_argument("--reparameterization-strength", type=float, default=0.0)
    parser.add_argument("--reparameterization-power", type=float, default=0.5)
    parser.add_argument(
        "--time-integrator",
        choices=("rk4", "implicit_midpoint"),
        default="rk4",
    )
    parser.add_argument(
        "--tangential-gauge",
        choices=("theta_s", "lagrangian"),
        default="theta_s",
    )
    parser.add_argument("--implicit-tolerance", type=float, default=2.0e-10)
    parser.add_argument("--no-implicit-fallback", action="store_true")
    parser.add_argument("--project-volume", action="store_true")
    parser.add_argument("--spectral-filter-strength", type=float, default=0.0)
    parser.add_argument("--spectral-filter-order", type=int, default=16)
    parser.add_argument("--no-initial-reparameterization", action="store_true")
    parser.add_argument("--crest-center", type=float, default=3.0)
    parser.add_argument("--amplitude", type=float, default=0.18)
    parser.add_argument("--jet-speed", type=float, default=1.1)
    parser.add_argument("--width", type=float, default=0.72)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--snapshots", type=int, default=121)
    parser.add_argument(
        "--initial-condition",
        choices=(
            "localized",
            "solitary",
            "exact_solitary",
            "projected_exact_solitary",
        ),
        default="localized",
    )
    parser.add_argument(
        "--output-prefix", type=Path, default=Path("results/topographic_breaker")
    )
    args = parser.parse_args()
    result = simulate(
        n=args.n,
        length=args.length,
        depth=args.depth,
        shoal_height=args.shoal_height,
        shoal_center=args.shoal_center,
        bathymetry=args.bathymetry,
        shallow_depth=args.shallow_depth,
        reef_slope=args.reef_slope,
        reef_toe=args.reef_toe,
        shelf_length=args.shelf_length,
        return_length=args.return_length,
        corner_width=args.corner_width,
        reef_cluster_strength=args.reef_cluster_strength,
        reef_cluster_width=args.reef_cluster_width,
        crest_center=args.crest_center,
        amplitude=args.amplitude,
        jet_speed=args.jet_speed,
        width=args.width,
        dt=args.dt,
        final_time=args.final_time,
        snapshots=args.snapshots,
        gravity=args.gravity,
        initial_condition=args.initial_condition,
        exact_internal_points=args.exact_internal_points,
        stop_at_overturning=args.stop_at_overturning,
        stop_at_impact=args.stop_at_impact,
        impact_distance_factor=args.impact_distance_factor,
        contact_arc_exclusion=args.contact_arc_exclusion,
        reparameterize_every=args.reparameterize_every,
        reparameterization_strength=args.reparameterization_strength,
        reparameterization_power=args.reparameterization_power,
        time_integrator=args.time_integrator,
        tangential_gauge=args.tangential_gauge,
        implicit_tolerance=args.implicit_tolerance,
        implicit_fallback=not args.no_implicit_fallback,
        project_volume=args.project_volume,
        spectral_filter_strength=args.spectral_filter_strength,
        spectral_filter_order=args.spectral_filter_order,
        initial_reparameterize=not args.no_initial_reparameterization,
    )
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prefix.with_suffix(".npz"), **result)
    energy_scale = max(
        abs(float(result["energy"][0] - result["energy_reference"])),
        np.finfo(float).eps,
    )
    metrics = {
        "n": args.n,
        "initial_condition": args.initial_condition,
        "initial_wave_method": str(result["initial_wave_method"]),
        "initial_wave_speed": float(result["initial_wave_speed"]),
        "initial_equation_residual": float(
            result["initial_equation_residual"]
        ),
        "time_integrator": str(result["time_integrator"]),
        "tangential_gauge": str(result["tangential_gauge"]),
        "implicit_fallback_count": int(np.sum(result["implicit_fallback"])),
        "maximum_nonlinear_residual": float(
            np.max(result["nonlinear_residual"])
        ),
        "background_current": float(result["background_current"]),
        "bathymetry": args.bathymetry,
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
        "max_marker_spacing_cv": float(np.max(result["marker_cv"])),
        "max_monitor_equidistribution_cv": float(
            np.max(result["monitor_equidistribution_cv"])
        ),
        "minimum_normalized_impact_distance": float(
            np.min(result["normalized_impact_distance"])
        ),
        "max_bie_residual": float(np.max(result["bie_residual"])),
        "max_surface_flux_defect": float(
            np.max(np.abs(result["surface_flux_defect"]))
        ),
        "termination_reason": str(result["termination_reason"]),
        "project_volume": bool(result["project_volume"]),
        "maximum_absolute_volume_projection_shift": float(
            np.max(np.abs(result["volume_projection_shifts"]))
        ),
        "cumulative_absolute_volume_projection_shift": float(
            np.sum(np.abs(result["volume_projection_shifts"]))
        ),
        "maximum_spectral_filter_relative_correction": float(
            np.max(result["spectral_filter_relative_corrections"])
        ),
        "cumulative_spectral_filter_relative_correction": float(
            np.sum(result["spectral_filter_relative_corrections"])
        ),
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
