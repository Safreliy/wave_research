"""Export a conservative pre-impact Euler--BIE state for a future VOF solver.

This module does not pretend to continue the overturning wave through impact.
It validates a pre-impact snapshot, rasterizes the liquid region, and exports
boundary data plus a bulk potential velocity away from the singular boundary.
A two-phase VOF code must still reconstruct the interface and perform its own
divergence-free pressure projection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from impact_detection import interface_self_intersects as _periodic_self_intersection
from selective_physics_controller import (
    Action,
    ControllerThresholds,
    DiagnosticState,
    decide,
)
from topography_bie import MixedBIEResult, TopographyBIE
from topographic_wave_solver import reference_energy


Array = np.ndarray


def _snapshot_count(data: dict[str, Array], index: int) -> int:
    """Return the active marker count for fixed- or variable-order archives."""
    if "count" in data:
        return int(np.asarray(data["count"])[index])
    return int(np.asarray(data["x"]).shape[1])


def _snapshot_vector(data: dict[str, Array], key: str, index: int) -> Array:
    """Extract one unpadded boundary vector from either archive schema."""
    values = np.asarray(data[key])
    count = _snapshot_count(data, index)
    if values.ndim == 1:
        return np.asarray(values[:count], dtype=float)
    return np.asarray(values[index, :count], dtype=float)


def _orientation(a: Array, b: Array, c: Array) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _strict_segment_intersection(a: Array, b: Array, c: Array, d: Array) -> bool:
    scale = max(np.linalg.norm(b - a), np.linalg.norm(d - c), 1.0)
    tolerance = 1.0e-12 * scale * scale
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    return ab_c * ab_d < -tolerance and cd_a * cd_b < -tolerance


def interface_self_intersects(
    x: Array, z: Array, length: float | None = None
) -> bool:
    """Compatibility wrapper around the periodic all-panel detector."""
    return _periodic_self_intersection(x, z, length)


def minimum_nonlocal_marker_distance(x: Array, z: Array, exclusion: int = 3) -> float:
    points = np.column_stack((x, z))
    delta = points[:, None, :] - points[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    n = len(points)
    index = np.arange(n)
    separation = np.abs(index[:, None] - index[None, :])
    separation = np.minimum(separation, n - separation)
    distance[separation <= exclusion] = np.inf
    return float(np.min(distance))


def choose_snapshot(
    data: dict[str, Array],
    selection: str,
    max_energy_drift: float,
    max_volume_drift: float,
    max_surface_flux_defect: float = 1.0e-3,
    max_marker_spacing_cv: float = 7.5e-2,
) -> tuple[int, dict[str, float | bool | str]]:
    first_bottom_x = _snapshot_vector(data, "bottom_x", 0)
    first_bottom_z = _snapshot_vector(data, "bottom_z", 0)
    energy_reference = float(
        data.get(
            "energy_reference",
            np.asarray(
                reference_energy(
                    first_bottom_x,
                    first_bottom_z,
                    float(data["length"]),
                    float(data["gravity"]),
                    float(data.get("background_current", np.asarray(0.0))),
                )
            ),
        )
    )
    initial_energy = float(
        data.get("global_initial_energy", np.asarray(data["energy"][0]))
    )
    target_volume = float(
        data.get(
            "global_target_volume",
            data.get("target_volume", np.asarray(data["volume"][0])),
        )
    )
    energy_scale = max(
        abs(initial_energy - energy_reference),
        np.finfo(float).eps,
    )
    relative_energy = (data["energy"] - initial_energy) / energy_scale
    volume_drift = data["volume"] - target_volume
    surface_flux = np.abs(
        data.get("surface_flux_defect", np.zeros_like(data["time"]))
    )
    if "cumulative_projection_correction" in data:
        cumulative_correction = np.asarray(
            data["cumulative_projection_correction"], dtype=float
        )
    else:
        cumulative_correction = float(
            data.get("cumulative_filter_correction_offset", np.asarray(0.0))
        ) + np.cumsum(
            data.get(
                "spectral_filter_relative_corrections",
                np.zeros_like(data["time"]),
            )
        )
    marker_quality_field = (
        "monitor_mass_cv" if "monitor_mass_cv" in data else "marker_cv"
    )
    marker_quality = np.asarray(data[marker_quality_field], dtype=float)
    residual_field = (
        "maximum_bie_residual"
        if "maximum_bie_residual" in data
        else "bie_residual"
    )
    residual_history = np.asarray(data[residual_field], dtype=float)
    length = float(data["length"])
    snapshot_count = len(np.asarray(data["time"]))
    x_alpha_history: list[Array] = []
    z_alpha_history: list[Array] = []
    maximum_surface_slope = np.zeros(snapshot_count)
    mean_x_alpha = np.zeros(snapshot_count)
    boundary_spacing = np.zeros(snapshot_count)
    for snapshot in range(snapshot_count):
        surface_x = _snapshot_vector(data, "x", snapshot)
        surface_z = _snapshot_vector(data, "z", snapshot)
        n = len(surface_x)
        base = length * np.arange(n) / n
        modes = np.fft.fftfreq(n, d=1.0 / n)
        x_alpha = length / (2.0 * math.pi) + np.fft.ifft(
            1j * modes * np.fft.fft(surface_x - base)
        ).real
        z_alpha = np.fft.ifft(1j * modes * np.fft.fft(surface_z)).real
        metric = np.hypot(x_alpha, z_alpha)
        slope = np.divide(
            z_alpha,
            x_alpha,
            out=np.full_like(z_alpha, np.inf),
            where=np.abs(x_alpha) > 1.0e-12,
        )
        x_alpha_history.append(x_alpha)
        z_alpha_history.append(z_alpha)
        maximum_surface_slope[snapshot] = float(np.max(np.abs(slope)))
        mean_x_alpha[snapshot] = float(np.mean(x_alpha))
        boundary_spacing[snapshot] = float(2.0 * math.pi / n * np.mean(metric))
    admissible = (
        np.abs(relative_energy) <= max_energy_drift
    ) & (
        np.abs(volume_drift) <= max_volume_drift
    ) & (
        surface_flux <= max_surface_flux_defect
    ) & (
        marker_quality <= max_marker_spacing_cv
    ) & (
        np.abs(cumulative_correction) <= 5.0e-2
    )
    overturned = data["min_x_alpha"] <= 0.0
    if selection == "last_pre_event":
        candidates = np.where(
            admissible & ~overturned & (maximum_surface_slope >= 1.0)
        )[0]
    else:
        candidates = np.where(admissible & overturned)[0]
    if not len(candidates):
        event_kind = "steep pre-event" if selection == "last_pre_event" else "overturned"
        raise ValueError(
            f"no {event_kind} snapshot satisfies the conservation thresholds"
        )
    index = int(candidates[0] if selection == "first_overhang" else candidates[-1])
    surface_x = _snapshot_vector(data, "x", index)
    surface_z = _snapshot_vector(data, "z", index)
    bottom_x = _snapshot_vector(data, "bottom_x", index)
    bottom_z = _snapshot_vector(data, "bottom_z", index)
    self_intersection = interface_self_intersects(surface_x, surface_z, length)
    if self_intersection:
        raise ValueError("selected interface already self-intersects; hand off an earlier state")
    x_alpha = x_alpha_history[index]
    z_alpha = z_alpha_history[index]
    metric = np.sqrt(x_alpha**2 + z_alpha**2)
    bottom_points = np.column_stack((bottom_x, bottom_z))
    surface_points = np.column_stack((surface_x, surface_z))
    minimum_separation = float(
        np.min(
            np.sqrt(
                np.sum(
                    (surface_points[:, None, :] - bottom_points[None, :, :]) ** 2,
                    axis=2,
                )
            )
        )
    )
    decision = decide(
        DiagnosticState(
            relative_energy_drift=float(relative_energy[index]),
            wave_volume_normalized_drift=float(volume_drift[index]),
            surface_flux_defect=float(surface_flux[index]),
            bie_residual=float(residual_history[index]),
            marker_spacing_cv=float(marker_quality[index]),
            minimum_x_alpha=float(data["min_x_alpha"][index]),
            mean_x_alpha=float(mean_x_alpha[index]),
            minimum_boundary_separation=minimum_separation,
            boundary_spacing=float(boundary_spacing[index]),
            maximum_surface_slope=float(maximum_surface_slope[index]),
            self_intersection=self_intersection,
        ),
        ControllerThresholds(
            energy_drift=max_energy_drift,
            volume_drift=max_volume_drift,
            handoff_surface_flux=max_surface_flux_defect,
            handoff_marker_cv=max_marker_spacing_cv,
        ),
    )
    if decision.action != Action.HANDOFF_TWO_PHASE:
        raise ValueError(
            "selective-physics controller rejected handoff: "
            f"{decision.action.value}: {', '.join(decision.reasons)}"
        )
    checks: dict[str, float | bool | str] = {
        "relative_energy_drift": float(relative_energy[index]),
        "absolute_volume_drift": float(volume_drift[index]),
        "minimum_x_alpha": float(data["min_x_alpha"][index]),
        "marker_spacing_cv": float(marker_quality[index]),
        "marker_quality_field": marker_quality_field,
        "minimum_nonlocal_marker_distance": minimum_nonlocal_marker_distance(
            surface_x, surface_z
        ),
        "self_intersection": self_intersection,
        "surface_flux_defect": float(surface_flux[index]),
        "cumulative_numerical_correction": float(cumulative_correction[index]),
        "cumulative_filter_correction": float(cumulative_correction[index]),
        "maximum_surface_slope": float(maximum_surface_slope[index]),
        "controller_action": decision.action.value,
        "controller_evidence_label": decision.evidence_label,
    }
    return index, checks


def rasterize_volume_fraction(
    surface_x: Array,
    surface_z: Array,
    bottom_x: Array,
    bottom_z: Array,
    length: float,
    nx: int,
    nz: int,
    supersample: int = 4,
) -> tuple[Array, Array, Array]:
    z_min = float(np.min(bottom_z)) - 0.04
    z_max = float(np.max(surface_z)) + 0.12
    width, height = nx * supersample, nz * supersample
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    def pixel(x_value: float, z_value: float) -> tuple[float, float]:
        px = x_value / length * width
        pz = (z_max - z_value) / (z_max - z_min) * height
        return px, pz

    # Close the periodic panels explicitly.  During Lagrangian evolution the
    # first surface marker may drift slightly below x=0; clipping the open
    # N-marker polygon then loses the strip at x=L.  An unwrapped one-period
    # polygon supplies both copies of that seam before image clipping.
    start_x = float(surface_x[0])
    end_x = start_x + length
    top_x = np.concatenate((surface_x, [end_x]))
    top_z = np.concatenate((surface_z, [surface_z[0]]))
    interior_bottom = (bottom_x > start_x) & (bottom_x < end_x)
    bed_x = np.concatenate(
        ([start_x], bottom_x[interior_bottom], [end_x])
    )
    bed_z = np.interp(bed_x, bottom_x, bottom_z, period=length)
    polygon = [pixel(float(x), float(z)) for x, z in zip(top_x, top_z)]
    polygon += [
        pixel(float(x), float(z)) for x, z in zip(bed_x[::-1], bed_z[::-1])
    ]
    draw.polygon(polygon, fill=255)
    resampling = getattr(Image, "Resampling", Image).BOX
    fraction = np.asarray(mask.resize((nx, nz), resample=resampling), dtype=float) / 255.0
    fraction = np.flipud(fraction)
    grid_x = (np.arange(nx) + 0.5) * length / nx
    grid_z = z_min + (np.arange(nz) + 0.5) * (z_max - z_min) / nz
    return grid_x, grid_z, fraction


def _interior_potential(
    bie: TopographyBIE,
    surface_trace: Array,
    solution: MixedBIEResult,
    target_x: Array,
    target_z: Array,
    bottom_normal_derivative: Array | None = None,
) -> Array:
    if bottom_normal_derivative is None:
        bottom_normal_derivative = np.zeros(bie.n)
    density = np.concatenate(
        (solution.surface_normal_derivative, bottom_normal_derivative)
    )
    trace = np.concatenate((surface_trace, solution.bottom_potential))
    single_layer = bie._periodic_green(target_x, target_z)
    single_layer *= bie.dalpha * bie.metrics.reshape(1, -1)
    double_layer = bie._double_layer_kernel(target_x, target_z)
    return single_layer @ density - double_layer @ trace


def bulk_velocity(
    bie: TopographyBIE,
    surface_trace: Array,
    solution: MixedBIEResult,
    grid_x: Array,
    grid_z: Array,
    volume_fraction: Array,
    bottom_normal_derivative: Array | None = None,
    background_current: float = 0.0,
) -> tuple[Array, Array, Array]:
    xx, zz = np.meshgrid(grid_x, grid_z)
    dx = float(grid_x[1] - grid_x[0])
    dz = float(grid_z[1] - grid_z[0])
    clearance = 1.75 * max(dx, dz)
    targets = np.column_stack((xx.ravel(), zz.ravel()))
    boundary = np.column_stack(
        (
            np.concatenate((bie.surface.x, bie.bottom.x)),
            np.concatenate((bie.surface.z, bie.bottom.z)),
        )
    )
    distance = np.sqrt(
        np.min(np.sum((targets[:, None, :] - boundary[None, :, :]) ** 2, axis=2), axis=1)
    )
    valid = (volume_fraction.ravel() > 0.999) & (distance > clearance)
    velocity_x = np.full(len(targets), np.nan)
    velocity_z = np.full(len(targets), np.nan)
    derivative_step = 0.2 * min(dx, dz)
    selected = np.where(valid)[0]
    for start in range(0, len(selected), 4096):
        indices = selected[start : start + 4096]
        tx = targets[indices, 0]
        tz = targets[indices, 1]
        phi_x_plus = _interior_potential(
            bie, surface_trace, solution, tx + derivative_step, tz,
            bottom_normal_derivative,
        )
        phi_x_minus = _interior_potential(
            bie, surface_trace, solution, tx - derivative_step, tz,
            bottom_normal_derivative,
        )
        phi_z_plus = _interior_potential(
            bie, surface_trace, solution, tx, tz + derivative_step,
            bottom_normal_derivative,
        )
        phi_z_minus = _interior_potential(
            bie, surface_trace, solution, tx, tz - derivative_step,
            bottom_normal_derivative,
        )
        velocity_x[indices] = (
            (phi_x_plus - phi_x_minus) / (2.0 * derivative_step)
            + background_current
        )
        velocity_z[indices] = (phi_z_plus - phi_z_minus) / (2.0 * derivative_step)
    shape = volume_fraction.shape
    return velocity_x.reshape(shape), velocity_z.reshape(shape), valid.reshape(shape)


def bulk_potential(
    bie: TopographyBIE,
    surface_trace: Array,
    solution: MixedBIEResult,
    grid_x: Array,
    grid_z: Array,
    valid: Array,
    bottom_normal_derivative: Array | None = None,
    background_current: float = 0.0,
) -> Array:
    """Evaluate the scalar potential at the accepted interior grid cells.

    Passing scalar samples to the receiver is preferable to reconstructing a
    potential from separately differentiated velocity components: one scalar
    field enforces discrete curl-freeness by construction and gives the
    harmonic extension an absolute trace to fit.
    """
    xx, zz = np.meshgrid(grid_x, grid_z)
    targets = np.column_stack((xx.ravel(), zz.ravel()))
    selected = np.flatnonzero(np.asarray(valid, dtype=bool).ravel())
    values = np.full(len(targets), np.nan)
    for start in range(0, len(selected), 4096):
        indices = selected[start : start + 4096]
        values[indices] = _interior_potential(
            bie,
            surface_trace,
            solution,
            targets[indices, 0],
            targets[indices, 1],
            bottom_normal_derivative,
        ) + background_current * targets[indices, 0]
    return values.reshape(np.asarray(valid).shape)


def bulk_face_fluxes(
    bie: TopographyBIE,
    surface_trace: Array,
    solution: MixedBIEResult,
    grid_x: Array,
    grid_z: Array,
    velocity_valid: Array,
    bottom_normal_derivative: Array | None = None,
    background_current: float = 0.0,
) -> tuple[Array, Array, Array, Array, dict[str, float | int]]:
    """Evaluate shared MAC-face velocities with two-point Gauss quadrature.

    The returned values are face averages, hence they represent the finite-
    volume fluxes used by the receiving projection rather than another set of
    cell-centred point samples.  A vertical face is stored at the left edge of
    cell ``i``; horizontal faces include the two outer grid edges.
    """
    valid = np.asarray(velocity_valid, dtype=bool)
    nz, nx = valid.shape
    dx = float(grid_x[1] - grid_x[0])
    dz = float(grid_z[1] - grid_z[0])
    x_edges = np.asarray(grid_x, dtype=float) - 0.5 * dx
    z_edges = np.concatenate(
        (
            [float(grid_z[0]) - 0.5 * dz],
            np.asarray(grid_z, dtype=float) + 0.5 * dz,
        )
    )
    vertical_valid = valid & np.roll(valid, 1, axis=1)
    horizontal_valid = np.zeros((nz + 1, nx), dtype=bool)
    horizontal_valid[1:nz] = valid[:-1] & valid[1:]
    vertical_velocity = np.full((nz, nx), np.nan)
    horizontal_velocity = np.full((nz + 1, nx), np.nan)
    derivative_step = 0.2 * min(dx, dz)
    gauss_nodes = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))

    vertical_indices = np.flatnonzero(vertical_valid.ravel())
    for start in range(0, len(vertical_indices), 4096):
        indices = vertical_indices[start : start + 4096]
        k, i = np.unravel_index(indices, vertical_valid.shape)
        accumulated = np.zeros(len(indices))
        for node in gauss_nodes:
            tx = x_edges[i]
            tz = np.asarray(grid_z)[k] + 0.5 * dz * node
            phi_plus = _interior_potential(
                bie,
                surface_trace,
                solution,
                tx + derivative_step,
                tz,
                bottom_normal_derivative,
            )
            phi_minus = _interior_potential(
                bie,
                surface_trace,
                solution,
                tx - derivative_step,
                tz,
                bottom_normal_derivative,
            )
            accumulated += (phi_plus - phi_minus) / (2.0 * derivative_step)
        vertical_velocity.ravel()[indices] = 0.5 * accumulated + background_current

    horizontal_indices = np.flatnonzero(horizontal_valid.ravel())
    for start in range(0, len(horizontal_indices), 4096):
        indices = horizontal_indices[start : start + 4096]
        k, i = np.unravel_index(indices, horizontal_valid.shape)
        accumulated = np.zeros(len(indices))
        for node in gauss_nodes:
            tx = np.asarray(grid_x)[i] + 0.5 * dx * node
            tz = z_edges[k]
            phi_plus = _interior_potential(
                bie,
                surface_trace,
                solution,
                tx,
                tz + derivative_step,
                bottom_normal_derivative,
            )
            phi_minus = _interior_potential(
                bie,
                surface_trace,
                solution,
                tx,
                tz - derivative_step,
                bottom_normal_derivative,
            )
            accumulated += (phi_plus - phi_minus) / (2.0 * derivative_step)
        horizontal_velocity.ravel()[indices] = 0.5 * accumulated

    cell_valid = (
        vertical_valid
        & np.roll(vertical_valid, -1, axis=1)
        & horizontal_valid[:-1]
        & horizontal_valid[1:]
    )
    divergence = np.full((nz, nx), np.nan)
    divergence[cell_valid] = (
        (
            np.roll(vertical_velocity, -1, axis=1)[cell_valid]
            - vertical_velocity[cell_valid]
        )
        / dx
        + (horizontal_velocity[1:][cell_valid] - horizontal_velocity[:-1][cell_valid])
        / dz
    )
    diagnostics: dict[str, float | int] = {
        "valid_vertical_face_count": int(np.count_nonzero(vertical_valid)),
        "valid_horizontal_face_count": int(np.count_nonzero(horizontal_valid)),
        "valid_flux_cell_count": int(np.count_nonzero(cell_valid)),
        "face_flux_divergence_l2": float(
            np.sqrt(np.mean(divergence[cell_valid] ** 2))
            if np.any(cell_valid)
            else float("nan")
        ),
        "face_flux_divergence_linf": float(
            np.max(np.abs(divergence[cell_valid]))
            if np.any(cell_valid)
            else float("nan")
        ),
    }
    return (
        vertical_velocity,
        horizontal_velocity,
        vertical_valid,
        horizontal_valid,
        diagnostics,
    )


def export_handoff(
    input_path: Path,
    output_prefix: Path,
    selection: str = "first_overhang",
    max_energy_drift: float = 5.0e-3,
    max_volume_drift: float = 1.0e-3,
    max_surface_flux_defect: float = 1.0e-3,
    max_marker_spacing_cv: float = 7.5e-2,
    nx: int = 256,
    nz: int = 144,
    export_face_fluxes: bool = False,
) -> dict[str, float | int | str | bool]:
    with np.load(input_path) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    required = {"potential", "bottom_x", "bottom_z"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"input is missing handoff fields: {sorted(missing)}")
    index, checks = choose_snapshot(
        data,
        selection,
        max_energy_drift,
        max_volume_drift,
        max_surface_flux_defect,
        max_marker_spacing_cv,
    )
    surface_x = _snapshot_vector(data, "x", index)
    surface_z = _snapshot_vector(data, "z", index)
    potential = _snapshot_vector(data, "potential", index)
    bottom_x = _snapshot_vector(data, "bottom_x", index)
    bottom_z = _snapshot_vector(data, "bottom_z", index)
    length = float(data["length"])
    bie = TopographyBIE(
        surface_x, surface_z, bottom_x, bottom_z, length
    )
    background_current = float(
        data.get("background_current", np.asarray(0.0))
    )
    bottom_normal_x = bie.bottom.z_alpha / bie.bottom.metric
    bottom_flux = -background_current * bottom_normal_x
    # The condition number requires a dense SVD and is not needed for the
    # already-gated high-order handoff.  Residual and conservation checks stay
    # active; small research fixtures retain the additional diagnostic.
    solution = bie.solve(
        potential,
        bottom_flux,
        compute_condition_number=len(surface_x) <= 384,
    )
    tangent_x = bie.surface.x_alpha / bie.surface.metric
    tangent_z = bie.surface.z_alpha / bie.surface.metric
    normal_x = -tangent_z
    normal_z = tangent_x
    tangential = bie.surface_derivative(potential) + background_current * tangent_x
    normal = solution.surface_normal_derivative + background_current * normal_x
    surface_u = tangential * tangent_x + normal * normal_x
    surface_w = tangential * tangent_z + normal * normal_z
    dalpha = bie.dalpha
    target_momentum_x = float(
        dalpha
        * (
            np.sum(-potential * bie.surface.z_alpha)
            + np.sum(solution.bottom_potential * bie.bottom.z_alpha)
        )
        + background_current * float(data["volume"][index])
    )
    target_momentum_z = float(
        dalpha
        * (
            np.sum(potential * bie.surface.x_alpha)
            - np.sum(solution.bottom_potential * bie.bottom.x_alpha)
        )
    )
    grid_x, grid_z, fraction = rasterize_volume_fraction(
        surface_x,
        surface_z,
        bottom_x,
        bottom_z,
        length,
        nx,
        nz,
    )
    velocity_x, velocity_z, velocity_valid = bulk_velocity(
        bie,
        potential,
        solution,
        grid_x,
        grid_z,
        fraction,
        bottom_flux,
        background_current,
    )
    surface_speed_reference = float(
        np.quantile(np.hypot(surface_u, surface_w), 0.99)
    )
    bulk_speed_limit = max(1.5 * surface_speed_reference, 1.0e-8)
    bulk_speed = np.hypot(velocity_x, velocity_z)
    close_evaluation_outlier = velocity_valid & (bulk_speed > bulk_speed_limit)
    velocity_valid[close_evaluation_outlier] = False
    velocity_x[close_evaluation_outlier] = np.nan
    velocity_z[close_evaluation_outlier] = np.nan
    interior_potential = bulk_potential(
        bie,
        potential,
        solution,
        grid_x,
        grid_z,
        velocity_valid,
        bottom_flux,
        background_current,
    )
    face_flux_arrays: dict[str, Array] = {}
    face_flux_diagnostics: dict[str, float | int] = {}
    if export_face_fluxes:
        (
            face_velocity_x,
            face_velocity_z,
            face_velocity_x_valid,
            face_velocity_z_valid,
            face_flux_diagnostics,
        ) = bulk_face_fluxes(
            bie,
            potential,
            solution,
            grid_x,
            grid_z,
            velocity_valid,
            bottom_flux,
            background_current,
        )
        face_flux_arrays = {
            "face_velocity_x": face_velocity_x,
            "face_velocity_z": face_velocity_z,
            "face_velocity_x_valid": face_velocity_x_valid,
            "face_velocity_z_valid": face_velocity_z_valid,
        }
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_arrays: dict[str, Array] = dict(
        grid_x=grid_x,
        grid_z=grid_z,
        volume_fraction=fraction,
        velocity_x=velocity_x,
        velocity_z=velocity_z,
        velocity_valid=velocity_valid,
        bulk_potential=interior_potential,
        potential_valid=velocity_valid,
        surface_x=surface_x,
        surface_z=surface_z,
        surface_potential=potential,
        surface_full_potential=potential + background_current * surface_x,
        surface_normal_derivative=solution.surface_normal_derivative,
        surface_velocity_x=surface_u,
        surface_velocity_z=surface_w,
        bottom_x=bottom_x,
        bottom_z=bottom_z,
        bottom_potential=solution.bottom_potential,
        bottom_normal_derivative=bottom_flux,
        background_current=np.asarray(background_current),
        source_energy=data["energy"][index],
        source_volume=data["volume"][index],
        target_momentum_x=np.asarray(target_momentum_x),
        target_momentum_z=np.asarray(target_momentum_z),
        time=data["time"][index],
        source_file=np.asarray(str(input_path)),
        velocity_note=np.asarray(
            "Bulk velocity is exported only in full liquid cells away from boundaries; "
            "the receiving VOF solver must reconstruct and project it."
        ),
        **face_flux_arrays,
    )
    active_circulation_state = "panel_circulation" in data
    if active_circulation_state:
        panel_circulation = _snapshot_vector(data, "panel_circulation", index)
        output_arrays["surface_panel_circulation"] = panel_circulation
        output_arrays["surface_vortex_sheet_strength"] = (
            panel_circulation / (bie.surface.metric * bie.dalpha)
        )
    np.savez_compressed(output_prefix.with_suffix(".npz"), **output_arrays)
    metadata: dict[str, float | int | str | bool] = {
        "source": str(input_path),
        "snapshot_index": index,
        "time": float(data["time"][index]),
        "selection": selection,
        "grid_nx": nx,
        "grid_nz": nz,
        "valid_bulk_velocity_fraction": float(np.mean(velocity_valid)),
        "surface_speed_reference_q99": surface_speed_reference,
        "bulk_speed_limit": bulk_speed_limit,
        "discarded_close_evaluation_fraction": float(
            np.mean(close_evaluation_outlier)
        ),
        "face_fluxes_exported": export_face_fluxes,
        **face_flux_diagnostics,
        "bie_residual": solution.residual,
        "bie_condition_number": solution.condition_number,
        "background_current": background_current,
        "surface_circulation": float(background_current * length),
        "active_circulation_state": active_circulation_state,
        "total_panel_circulation": float(
            np.sum(output_arrays.get("surface_panel_circulation", np.asarray([0.0])))
        ),
        "panel_circulation_l1": float(
            np.sum(np.abs(output_arrays.get("surface_panel_circulation", np.asarray([0.0]))))
        ),
        "source_energy": float(data["energy"][index]),
        "source_volume": float(data["volume"][index]),
        "target_momentum_x": target_momentum_x,
        "target_momentum_z": target_momentum_z,
        "status": "validated_pre_impact_handoff",
        **checks,
    }
    output_prefix.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument(
        "--selection",
        choices=("first_overhang", "last_admissible", "last_pre_event"),
        default="first_overhang",
    )
    parser.add_argument("--max-energy-drift", type=float, default=5.0e-3)
    parser.add_argument("--max-volume-drift", type=float, default=1.0e-3)
    parser.add_argument("--max-surface-flux-defect", type=float, default=1.0e-3)
    parser.add_argument("--max-marker-spacing-cv", type=float, default=7.5e-2)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--nz", type=int, default=144)
    parser.add_argument(
        "--export-face-fluxes",
        action="store_true",
        help="also evaluate shared finite-volume face-average velocities",
    )
    args = parser.parse_args()
    metadata = export_handoff(
        args.input,
        args.output_prefix,
        selection=args.selection,
        max_energy_drift=args.max_energy_drift,
        max_volume_drift=args.max_volume_drift,
        max_surface_flux_defect=args.max_surface_flux_defect,
        max_marker_spacing_cv=args.max_marker_spacing_cv,
        nx=args.nx,
        nz=args.nz,
        export_face_fluxes=args.export_face_fluxes,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
