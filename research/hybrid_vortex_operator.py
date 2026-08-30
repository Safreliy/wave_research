"""First operator prototype for a selective-vorticity water-wave method.

This module does *not* yet solve the coupled vortical free-surface problem.  It
isolates one falsifiable building block: the velocity induced by regularized
point vortices in a periodic half-plane above a flat impermeable bed.  Opposite
image vortices enforce zero normal velocity at the bed analytically.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class VortexCloud:
    x: Array
    z: Array
    circulation: Array
    core_radius: float = 0.0

    def __post_init__(self) -> None:
        sizes = {np.size(self.x), np.size(self.z), np.size(self.circulation)}
        if len(sizes) != 1:
            raise ValueError("x, z and circulation must have equal sizes")
        if self.core_radius < 0.0:
            raise ValueError("core_radius must be non-negative")

    @property
    def total_circulation(self) -> float:
        return float(np.sum(self.circulation))


def _periodic_single_vortex_velocity(
    evaluation_x: Array,
    evaluation_z: Array,
    vortex_x: float,
    vortex_z: float,
    circulation: float,
    length: float,
    core_radius: float,
) -> tuple[Array, Array]:
    """Velocity of an infinite x-periodic row of equal point vortices."""
    dx = evaluation_x - vortex_x
    dz = evaluation_z - vortex_z
    scale = 2.0 * math.pi / length
    denominator = np.cosh(scale * dz) - np.cos(scale * dx)
    if core_radius:
        # This is a smooth numerical core, not a derived viscous closure.  The
        # added dimensionless term tends to the Rosenhead core locally.
        denominator = denominator + 0.5 * (scale * core_radius) ** 2
    prefactor = circulation / (2.0 * length)
    u = -prefactor * np.sinh(scale * dz) / denominator
    w = prefactor * np.sin(scale * dx) / denominator
    return u, w


def velocity_above_flat_bed(
    evaluation_x: Array,
    evaluation_z: Array,
    vortices: VortexCloud,
    length: float,
    bed_z: float,
) -> tuple[Array, Array]:
    """Return periodic vortex velocity with exact flat-bed impermeability.

    Every physical vortex has an opposite-sign image reflected across
    ``z=bed_z``.  Evaluation exactly at a physical vortex remains singular when
    ``core_radius == 0`` and is intentionally not special-cased.
    """
    if length <= 0.0:
        raise ValueError("length must be positive")
    x = np.asarray(evaluation_x, dtype=float)
    z = np.asarray(evaluation_z, dtype=float)
    x, z = np.broadcast_arrays(x, z)
    u = np.zeros_like(x)
    w = np.zeros_like(x)

    for vortex_x, vortex_z, circulation in zip(
        np.ravel(vortices.x),
        np.ravel(vortices.z),
        np.ravel(vortices.circulation),
    ):
        if vortex_z <= bed_z:
            raise ValueError("physical vortices must lie above the bed")
        physical = _periodic_single_vortex_velocity(
            x,
            z,
            float(vortex_x),
            float(vortex_z),
            float(circulation),
            length,
            vortices.core_radius,
        )
        image_z = 2.0 * bed_z - float(vortex_z)
        image = _periodic_single_vortex_velocity(
            x,
            z,
            float(vortex_x),
            image_z,
            -float(circulation),
            length,
            vortices.core_radius,
        )
        u += physical[0] + image[0]
        w += physical[1] + image[1]
    return u, w


def vortex_dipole(
    center_x: float,
    center_z: float,
    separation: float,
    circulation: float,
    core_radius: float = 0.0,
) -> VortexCloud:
    """A zero-net-circulation horizontal vortex pair."""
    if separation <= 0.0:
        raise ValueError("separation must be positive")
    return VortexCloud(
        x=np.asarray([center_x - 0.5 * separation, center_x + 0.5 * separation]),
        z=np.asarray([center_z, center_z]),
        circulation=np.asarray([circulation, -circulation]),
        core_radius=core_radius,
    )
