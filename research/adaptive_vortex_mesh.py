"""Conservative r/p-adaptation for an active free-surface vortex sheet.

Nodes are equidistributed in a curvature/strength/proximity monitor.  Panel
circulation is remapped through its cumulative primitive (the potential
trace), so insertion and deletion preserve the Kelvin period exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample

from active_vortex_sheet import (
    ActiveVortexSheetState,
    panel_circulation_to_potential,
    potential_to_panel_circulation,
)


Array = np.ndarray


@dataclass(frozen=True)
class VortexMeshMonitor:
    value: Array
    curvature: Array
    strength_density: Array
    nonlocal_gap: Array
    panel_length: Array
    maximum_turning_angle: float
    minimum_nonlocal_gap_ratio: float
    maximum_circulation_fraction: float


@dataclass(frozen=True)
class VortexMeshDecision:
    current_count: int
    target_count: int
    action: str
    reason: str


@dataclass(frozen=True)
class VortexRemeshDiagnostics:
    source_count: int
    target_count: int
    total_circulation_defect: float
    potential_roundtrip_relative_error: float
    monitor_mass_cv_before: float
    monitor_mass_cv_after: float
    minimum_panel_length: float
    maximum_panel_length: float


@dataclass(frozen=True)
class VortexInvariantProjectionDiagnostics:
    volume_shift: float
    circulation_scale: float
    relative_circulation_correction: float
    energy_before: float
    energy_after: float
    energy_target: float
    relative_energy_error: float
    relative_volume_error: float
    raw_dno_flux_defect: float


def resample_periodic_bottom(
    bottom_x: Array,
    bottom_z: Array,
    length: float,
    target_count: int,
) -> tuple[Array, Array]:
    """Spectrally change fixed-boundary count while preserving its periodic map."""
    bottom_x = np.asarray(bottom_x, dtype=float)
    bottom_z = np.asarray(bottom_z, dtype=float)
    if bottom_x.ndim != 1 or bottom_x.shape != bottom_z.shape:
        raise ValueError("bottom_x and bottom_z must be matching vectors")
    if target_count < 8:
        raise ValueError("target_count must be at least eight")
    source_count = len(bottom_x)
    source_base = length * np.arange(source_count) / source_count
    target_base = length * np.arange(target_count) / target_count
    target_x = target_base + np.asarray(
        resample(bottom_x - source_base, target_count), dtype=float
    )
    target_z = np.asarray(resample(bottom_z, target_count), dtype=float)
    return target_x, target_z


def project_vortex_sheet_invariants(
    state: ActiveVortexSheetState,
    bottom_x: Array,
    bottom_z: Array,
    length: float,
    gravity: float,
    background_current: float,
    target_volume: float,
    target_energy: float,
    energy_scale: float,
) -> tuple[ActiveVortexSheetState, VortexInvariantProjectionDiagnostics]:
    """Project volume and energy while preserving the Kelvin circulation period.

    After the scalar volume shift, scale only the zero-period circulation
    component.  At fixed geometry the total energy is quadratic in this scale,
    including the cross term with a non-zero background current.  Three energy
    evaluations determine the quadratic; the real root nearest one is used.
    """
    from topographic_wave_solver import diagnostics, geometric_volume

    if target_volume <= 0.0 or energy_scale <= 0.0:
        raise ValueError("target_volume and energy_scale must be positive")
    raw_volume = geometric_volume(
        state.x, state.z, bottom_x, bottom_z, length
    )
    volume_shift = (target_volume - raw_volume) / length
    shifted = ActiveVortexSheetState(
        state.x,
        state.z + volume_shift,
        state.panel_circulation,
        state.potential_mean,
    )
    background_gamma = potential_to_panel_circulation(
        shifted.x,
        np.zeros_like(shifted.x),
        length,
        background_current,
    )
    perturbation_gamma = shifted.panel_circulation - background_gamma

    def state_at(scale: float) -> tuple[ActiveVortexSheetState, tuple[float, ...]]:
        candidate = ActiveVortexSheetState(
            shifted.x,
            shifted.z,
            background_gamma + scale * perturbation_gamma,
            shifted.potential_mean,
        )
        potential = panel_circulation_to_potential(
            candidate.x,
            candidate.panel_circulation,
            length,
            background_current,
            candidate.potential_mean,
        )
        values = diagnostics(
            candidate.x,
            candidate.z,
            potential,
            bottom_x,
            bottom_z,
            length,
            gravity,
            background_current,
        )
        return candidate, values

    _, at_one = state_at(1.0)
    if abs(background_current) <= 10.0 * np.finfo(float).eps:
        # For U=0 the zero-circulation state has no kinetic energy.  Evaluate
        # its potential energy geometrically instead of constructing and
        # solving a second dense BIE.  Linearity of the DNO then gives both
        # the projected energy and flux without a third solve.
        n = len(shifted.x)
        modes = np.fft.fftfreq(n, d=1.0 / n)
        base = length * np.arange(n) / n
        surface_x_alpha = length / (2.0 * math.pi) + np.fft.ifft(
            1j * modes * np.fft.fft(shifted.x - base)
        ).real
        bottom_x_alpha = length / (2.0 * math.pi) + np.fft.ifft(
            1j * modes * np.fft.fft(np.asarray(bottom_x) - base)
        ).real
        dalpha = 2.0 * math.pi / n
        zero_energy = 0.5 * gravity * dalpha * float(
            np.sum(
                shifted.z**2 * surface_x_alpha
                - np.asarray(bottom_z) ** 2 * bottom_x_alpha
            )
        )
        quadratic = at_one[1] - zero_energy
        if quadratic <= 0.0 or target_energy <= zero_energy:
            raise ValueError("energy target is incompatible with circulation scaling")
        circulation_scale = math.sqrt((target_energy - zero_energy) / quadratic)
        projected = ActiveVortexSheetState(
            shifted.x,
            shifted.z,
            background_gamma + circulation_scale * perturbation_gamma,
            shifted.potential_mean,
        )
        final_values = (
            at_one[0],
            target_energy,
            at_one[2],
            at_one[3],
            circulation_scale * at_one[4],
        )
    else:
        _, at_zero = state_at(0.0)
        _, at_minus_one = state_at(-1.0)
        coefficient_a = 0.5 * (at_one[1] + at_minus_one[1]) - at_zero[1]
        coefficient_b = 0.5 * (at_one[1] - at_minus_one[1])
        coefficient_c = at_zero[1] - target_energy
        roots = np.roots([coefficient_a, coefficient_b, coefficient_c])
        real_roots = [float(root.real) for root in roots if abs(root.imag) <= 1.0e-10]
        if not real_roots:
            raise ValueError("energy projection quadratic has no real root")
        circulation_scale = min(real_roots, key=lambda value: abs(value - 1.0))
        projected, final_values = state_at(circulation_scale)
    correction = float(
        np.linalg.norm(projected.panel_circulation - shifted.panel_circulation)
        / max(
            float(np.linalg.norm(shifted.panel_circulation)),
            np.finfo(float).eps,
        )
    )
    projection = VortexInvariantProjectionDiagnostics(
        volume_shift,
        float(circulation_scale),
        correction,
        float(at_one[1]),
        float(final_values[1]),
        float(target_energy),
        abs(float(final_values[1]) - target_energy) / energy_scale,
        abs(float(final_values[0]) - target_volume) / target_volume,
        abs(float(final_values[4])),
    )
    return projected, projection


def _periodic_panel_geometry(x: Array, z: Array, length: float) -> tuple[Array, Array, Array]:
    following_x = np.roll(x, -1)
    following_x[-1] += length
    following_z = np.roll(z, -1)
    dx = following_x - x
    dz = following_z - z
    panel_length = np.hypot(dx, dz)
    if np.any(panel_length <= 10.0 * np.finfo(float).eps):
        raise ValueError("vortex sheet contains a zero-length panel")
    s = np.zeros(len(x) + 1)
    s[1:] = np.cumsum(panel_length)
    return panel_length, s, np.column_stack((dx, dz))


def _normalized(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    positive = np.abs(values[np.isfinite(values)])
    if not len(positive):
        return np.zeros_like(values)
    scale = float(np.quantile(positive, 0.90))
    if scale <= np.finfo(float).eps:
        return np.zeros_like(values)
    return np.clip(np.abs(values) / scale, 0.0, 8.0)


def vortex_mesh_monitor(
    state: ActiveVortexSheetState,
    length: float,
    curvature_weight: float = 4.0,
    strength_weight: float = 1.5,
    proximity_weight: float = 8.0,
    arc_exclusion_panels: int = 10,
) -> VortexMeshMonitor:
    """Build a dimensionless monitor for overturning and near-contact zones."""
    if min(curvature_weight, strength_weight, proximity_weight) < 0.0:
        raise ValueError("monitor weights cannot be negative")
    x = np.asarray(state.x, dtype=float)
    z = np.asarray(state.z, dtype=float)
    circulation = np.asarray(state.panel_circulation, dtype=float)
    n = len(x)
    if n < 2 * arc_exclusion_panels + 3:
        raise ValueError("too few nodes for the requested nonlocal arc exclusion")
    panel_length, s, _ = _periodic_panel_geometry(x, z, length)
    total_arc = float(s[-1])
    x_residual = x - x[0] - length * s[:-1] / total_arc
    x_spline = CubicSpline(
        s, np.r_[x_residual, x_residual[0]], bc_type="periodic"
    )
    z_spline = CubicSpline(s, np.r_[z, z[0]], bc_type="periodic")
    x_first = length / total_arc + x_spline(s[:-1], 1)
    x_second = x_spline(s[:-1], 2)
    z_first = z_spline(s[:-1], 1)
    z_second = z_spline(s[:-1], 2)
    curvature = np.abs(x_first * z_second - z_first * x_second) / np.maximum(
        (x_first**2 + z_first**2) ** 1.5, np.finfo(float).eps
    )
    panel_density = circulation / panel_length
    strength_density = 0.5 * (
        np.abs(panel_density) + np.abs(np.roll(panel_density, 1))
    )

    node_s = s[:-1]
    arc_distance = np.abs(node_s[:, None] - node_s[None, :])
    arc_distance = np.minimum(arc_distance, total_arc - arc_distance)
    dx = x[:, None] - x[None, :]
    dx = (dx + 0.5 * length) % length - 0.5 * length
    dz = z[:, None] - z[None, :]
    distance = np.hypot(dx, dz)
    median_panel = float(np.median(panel_length))
    distance[arc_distance <= arc_exclusion_panels * median_panel] = np.inf
    nonlocal_gap = np.min(distance, axis=1)
    proximity = np.clip((2.0 * median_panel / nonlocal_gap) ** 2, 0.0, 25.0)
    monitor = (
        1.0
        + curvature_weight * _normalized(curvature * median_panel)
        + strength_weight * _normalized(strength_density)
        + proximity_weight * proximity
    )
    turning = curvature * 0.5 * (panel_length + np.roll(panel_length, 1))
    circulation_scale = max(float(np.sum(np.abs(circulation))), np.finfo(float).eps)
    return VortexMeshMonitor(
        monitor,
        curvature,
        strength_density,
        nonlocal_gap,
        panel_length,
        float(np.max(turning)),
        float(np.min(nonlocal_gap) / median_panel),
        float(np.max(np.abs(circulation)) / circulation_scale),
    )


def suggest_vortex_count(
    monitor: VortexMeshMonitor,
    current_count: int,
    minimum_count: int,
    maximum_count: int,
    allow_coarsening: bool = False,
) -> VortexMeshDecision:
    """Hysteretic p-adaptation rule; r-adaptation remains available every step."""
    if not minimum_count <= current_count <= maximum_count:
        raise ValueError("current_count must lie inside adaptation bounds")
    refine = (
        monitor.maximum_turning_angle > 0.18
        or monitor.minimum_nonlocal_gap_ratio < 4.0
        or monitor.maximum_circulation_fraction > 0.04
    )
    coarsen = (
        allow_coarsening
        and monitor.maximum_turning_angle < 0.055
        and monitor.minimum_nonlocal_gap_ratio > 12.0
        and monitor.maximum_circulation_fraction < 0.015
    )
    if refine and current_count < maximum_count:
        target = min(maximum_count, int(math.ceil(1.5 * current_count / 8.0)) * 8)
        return VortexMeshDecision(
            current_count,
            target,
            "refine",
            "turning, nonlocal gap or circulation concentration crossed refine gate",
        )
    if coarsen and current_count > minimum_count:
        target = max(minimum_count, int(math.floor(current_count / 1.25 / 8.0)) * 8)
        return VortexMeshDecision(
            current_count,
            target,
            "coarsen",
            "all three indicators remained inside the coarsening hysteresis gate",
        )
    return VortexMeshDecision(current_count, current_count, "keep", "inside hysteresis band")


def _monitor_mass_cv(monitor: VortexMeshMonitor) -> float:
    following = np.roll(monitor.value, -1)
    mass = 0.5 * (monitor.value + following) * monitor.panel_length
    return float(np.std(mass) / np.mean(mass))


def remesh_active_vortex_sheet(
    state: ActiveVortexSheetState,
    length: float,
    background_current: float = 0.0,
    target_count: int | None = None,
    curvature_weight: float = 4.0,
    strength_weight: float = 1.5,
    proximity_weight: float = 8.0,
    arc_exclusion_panels: int = 10,
    maximum_monitor_ratio: float = 6.0,
    monitor_smoothing_panels: float = 4.0,
) -> tuple[ActiveVortexSheetState, VortexRemeshDiagnostics]:
    """Equidistribute the sheet and conservatively remap its circulation.

    The raw physics monitor is smoothed and ratio-limited before inversion.
    This mesh-quality regularization is essential for a spectral BIE: an
    unconstrained curvature spike can otherwise collapse one panel by an
    order of magnitude and make the nominal global time step violate its
    local CFL limit.  A monotone cubic inverse replaces the piecewise-linear
    map formerly used here.
    """
    source_count = len(state.x)
    if target_count is None:
        target_count = source_count
    if target_count < 2 * arc_exclusion_panels + 3:
        raise ValueError("target_count is too small for the proximity monitor")
    if maximum_monitor_ratio < 1.0 or monitor_smoothing_panels < 0.0:
        raise ValueError("invalid monitor regularization")
    source_monitor = vortex_mesh_monitor(
        state,
        length,
        curvature_weight,
        strength_weight,
        proximity_weight,
        arc_exclusion_panels,
    )
    panel_length, s, _ = _periodic_panel_geometry(state.x, state.z, length)
    regularized_monitor = np.asarray(source_monitor.value, dtype=float)
    if monitor_smoothing_panels > 0.0:
        regularized_monitor = gaussian_filter1d(
            regularized_monitor,
            sigma=monitor_smoothing_panels,
            mode="wrap",
        )
    monitor_floor = max(
        float(np.quantile(regularized_monitor, 0.10)),
        np.finfo(float).eps,
    )
    regularized_monitor = np.clip(
        regularized_monitor,
        monitor_floor,
        maximum_monitor_ratio * monitor_floor,
    )
    monitor_panel_mass = 0.5 * (
        regularized_monitor + np.roll(regularized_monitor, -1)
    ) * panel_length
    cumulative_monitor = np.zeros(source_count + 1)
    cumulative_monitor[1:] = np.cumsum(monitor_panel_mass)
    targets = np.arange(target_count) * cumulative_monitor[-1] / target_count
    target_s = np.asarray(
        PchipInterpolator(cumulative_monitor, s)(targets), dtype=float
    )
    total_arc = float(s[-1])

    potential = panel_circulation_to_potential(
        state.x,
        state.panel_circulation,
        length,
        background_current,
        state.potential_mean,
    )
    x_residual = state.x - state.x[0] - length * s[:-1] / total_arc
    x_spline = CubicSpline(
        s, np.r_[x_residual, x_residual[0]], bc_type="periodic"
    )
    z_spline = CubicSpline(s, np.r_[state.z, state.z[0]], bc_type="periodic")
    potential_spline = CubicSpline(
        s, np.r_[potential, potential[0]], bc_type="periodic"
    )
    new_x = state.x[0] + length * target_s / total_arc + x_spline(target_s)
    new_z = z_spline(target_s)
    new_potential = potential_spline(target_s)
    new_potential += state.potential_mean - float(np.mean(new_potential))
    new_circulation = potential_to_panel_circulation(
        new_x, new_potential, length, background_current
    )
    remeshed = ActiveVortexSheetState(
        new_x,
        new_z,
        new_circulation,
        state.potential_mean,
    )
    reconstructed = panel_circulation_to_potential(
        new_x,
        new_circulation,
        length,
        background_current,
        state.potential_mean,
    )
    roundtrip = float(
        np.linalg.norm(reconstructed - new_potential)
        / max(float(np.linalg.norm(new_potential)), np.finfo(float).eps)
    )
    target_monitor = vortex_mesh_monitor(
        remeshed,
        length,
        curvature_weight,
        strength_weight,
        proximity_weight,
        arc_exclusion_panels,
    )
    diagnostics = VortexRemeshDiagnostics(
        source_count,
        target_count,
        float(remeshed.total_circulation - state.total_circulation),
        roundtrip,
        _monitor_mass_cv(source_monitor),
        _monitor_mass_cv(target_monitor),
        float(np.min(target_monitor.panel_length)),
        float(np.max(target_monitor.panel_length)),
    )
    return remeshed, diagnostics
