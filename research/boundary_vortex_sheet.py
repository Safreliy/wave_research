"""Discrete circulation carriers for an Euler--BIE boundary state.

These points visualize the tangential-velocity jump obtained by extending the
liquid velocity as zero outside the liquid.  They are a boundary vortex sheet,
not pre-existing bulk vortices and not an independent post-impact solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topography_bie import TopographyBIE


Array = np.ndarray


@dataclass(frozen=True)
class BoundaryVortexSheet:
    surface_x: Array
    surface_z: Array
    surface_circulation: Array
    bottom_x: Array
    bottom_z: Array
    bottom_circulation: Array
    closed_circulation_defect: float


def _panel_circulation(
    x: Array,
    z: Array,
    periodic_potential: Array,
    length: float,
    background_current: float,
) -> tuple[Array, Array, Array]:
    total_potential = np.asarray(periodic_potential) + background_current * np.asarray(x)
    next_potential = np.roll(total_potential, -1)
    next_potential[-1] += background_current * length
    next_x = np.roll(x, -1)
    next_x[-1] += length
    return (
        0.5 * (np.asarray(x) + next_x),
        0.5 * (np.asarray(z) + np.roll(z, -1)),
        next_potential - total_potential,
    )


def surface_vortex_points(
    x: Array,
    z: Array,
    periodic_potential: Array,
    length: float,
    background_current: float = 0.0,
) -> tuple[Array, Array, Array]:
    """Return panel midpoints and integrated circulation ``Gamma=Delta Phi``."""
    return _panel_circulation(
        x, z, periodic_potential, length, background_current
    )


def closed_boundary_vortex_sheet(
    x: Array,
    z: Array,
    periodic_potential: Array,
    bottom_x: Array,
    bottom_z: Array,
    length: float,
    background_current: float = 0.0,
) -> BoundaryVortexSheet:
    """Construct surface and oppositely oriented bed circulation panels."""
    bie = TopographyBIE(x, z, bottom_x, bottom_z, length)
    bottom_normal_x = bie.bottom.z_alpha / bie.bottom.metric
    result = bie.solve(
        periodic_potential, -background_current * bottom_normal_x
    )
    surface = _panel_circulation(
        x, z, periodic_potential, length, background_current
    )
    bottom_left_to_right = _panel_circulation(
        bottom_x,
        bottom_z,
        result.bottom_potential,
        length,
        background_current,
    )
    bottom_circulation = -bottom_left_to_right[2]
    defect = float(np.sum(surface[2]) + np.sum(bottom_circulation))
    return BoundaryVortexSheet(
        surface[0],
        surface[1],
        surface[2],
        bottom_left_to_right[0],
        bottom_left_to_right[1],
        bottom_circulation,
        defect,
    )
