"""Dense second-kind periodic BIE prototype for a water-wave strip.

The fluid is periodic in x, bounded above by ``z=eta(x)`` and below by the
flat bed ``z=-depth``.  Even reflection across the bed converts the mixed
Dirichlet/Neumann problem into a Dirichlet problem between the free surface and
its mirror ``z=-2*depth-eta(x)``.  A double-layer potential then gives the
second-kind equation

    (-1/2 I + K) mu = prescribed_trace.

This is a verification-oriented dense Nyström implementation.  Near-boundary
normal derivatives are evaluated by a second-order inward extrapolation; a
publication solver should replace that step by singularity subtraction or a
high-order close-evaluation scheme and replace the dense solve by GMRES/FMM.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class BIEResult:
    density: Array
    dno: Array
    boundary_residual: float
    offset: float


class PeriodicMirrorBIE:
    """Second-kind double-layer BIE with a flat Neumann bed by reflection."""

    def __init__(
        self,
        eta: Array,
        length: float = 2.0 * math.pi,
        depth: float = 1.0,
    ) -> None:
        eta = np.asarray(eta, dtype=float)
        if eta.ndim != 1 or len(eta) < 16 or len(eta) % 2:
            raise ValueError("eta must be a one-dimensional even grid with N >= 16")
        if length <= 0.0 or depth <= 0.0:
            raise ValueError("length and depth must be positive")
        if np.max(eta) <= -depth:
            raise ValueError("the free surface must lie above the bed")

        self.eta = eta.copy()
        self.n = len(eta)
        self.length = float(length)
        self.depth = float(depth)
        self.dx = self.length / self.n
        self.x = np.arange(self.n, dtype=float) * self.dx
        self.wavenumbers = 2.0 * math.pi * np.fft.fftfreq(self.n, d=self.dx)
        self.eta_x = self._derivative(eta, 1)
        self.eta_xx = self._derivative(eta, 2)

        top_z = eta
        bottom_z = -2.0 * depth - eta
        self.source_x = np.concatenate((self.x, self.x))
        self.source_z = np.concatenate((top_z, bottom_z))

        # Outward normal times ds/dx.  This avoids square roots in the integral.
        top_normal_ds_x = -self.eta_x
        top_normal_ds_z = np.ones(self.n)
        bottom_slope = -self.eta_x
        bottom_normal_ds_x = bottom_slope
        bottom_normal_ds_z = -np.ones(self.n)
        self.normal_ds_x = np.concatenate(
            (top_normal_ds_x, bottom_normal_ds_x)
        )
        self.normal_ds_z = np.concatenate(
            (top_normal_ds_z, bottom_normal_ds_z)
        )
        self.matrix = self._boundary_matrix()

    def _derivative(self, values: Array, order: int) -> Array:
        return np.fft.ifft(
            (1j * self.wavenumbers) ** order * np.fft.fft(values)
        ).real

    def _double_layer_kernel(
        self, target_x: Array, target_z: Array
    ) -> Array:
        """Return K(target, source) including the trapezoidal source weight."""
        target_x = np.asarray(target_x, dtype=float).reshape(-1, 1)
        target_z = np.asarray(target_z, dtype=float).reshape(-1, 1)
        delta_x = target_x - self.source_x.reshape(1, -1)
        delta_z = target_z - self.source_z.reshape(1, -1)
        scale = 2.0 * math.pi / self.length
        denominator = np.cosh(scale * delta_z) - np.cos(scale * delta_x)
        numerator = (
            np.sin(scale * delta_x) * self.normal_ds_x.reshape(1, -1)
            + np.sinh(scale * delta_z) * self.normal_ds_z.reshape(1, -1)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            kernel = scale * numerator / (4.0 * math.pi * denominator)
        return kernel * self.dx

    def _boundary_matrix(self) -> Array:
        kernel = self._double_layer_kernel(self.source_x, self.source_z)

        # Analytic diagonal of K dx for a graph.  Top and reflected-bottom
        # contributions have the same value with the chosen outward normals.
        diagonal = (
            self.eta_xx / (4.0 * math.pi * (1.0 + self.eta_x**2)) * self.dx
        )
        indices = np.arange(2 * self.n)
        kernel[indices, indices] = np.concatenate((diagonal, diagonal))
        return kernel - 0.5 * np.eye(2 * self.n)

    def solve_density(self, trace: Array) -> tuple[Array, float]:
        trace = np.asarray(trace, dtype=float)
        if trace.shape != (self.n,):
            raise ValueError(f"trace must have shape ({self.n},)")
        # Even reflection gives the same Dirichlet data on the mirror surface.
        right_hand_side = np.concatenate((trace, trace))
        density = np.linalg.solve(self.matrix, right_hand_side)
        residual = self.matrix @ density - right_hand_side
        scale = max(float(np.linalg.norm(right_hand_side, ord=np.inf)), 1.0)
        return density, float(np.linalg.norm(residual, ord=np.inf) / scale)

    def evaluate_potential(
        self, target_x: Array, target_z: Array, density: Array
    ) -> Array:
        target_x, target_z = np.broadcast_arrays(target_x, target_z)
        density = np.asarray(density, dtype=float)
        if density.shape != (2 * self.n,):
            raise ValueError(f"density must have shape ({2 * self.n},)")
        kernel = self._double_layer_kernel(target_x.ravel(), target_z.ravel())
        return (kernel @ density).reshape(target_x.shape)

    def dirichlet_to_neumann(
        self, trace: Array, offset_in_grid_steps: float = 2.5
    ) -> BIEResult:
        """Approximate ``phi_z - eta_x*phi_x`` on the upper boundary.

        The potential is sampled one and two offsets into the fluid.  Keeping
        the offset a few grid steps wide makes the periodic trapezoidal close
        evaluation resolved; the extrapolation error is second order.
        """
        if offset_in_grid_steps < 1.0:
            raise ValueError("offset_in_grid_steps must be >= 1")
        trace = np.asarray(trace, dtype=float)
        density, residual = self.solve_density(trace)
        metric = np.sqrt(1.0 + self.eta_x**2)
        normal_x = -self.eta_x / metric
        normal_z = 1.0 / metric
        offset = offset_in_grid_steps * self.dx

        first_x = self.x - offset * normal_x
        first_z = self.eta - offset * normal_z
        second_x = self.x - 2.0 * offset * normal_x
        second_z = self.eta - 2.0 * offset * normal_z
        first_value = self.evaluate_potential(first_x, first_z, density)
        second_value = self.evaluate_potential(second_x, second_z, density)
        outward_normal_derivative = (
            3.0 * trace - 4.0 * first_value + second_value
        ) / (2.0 * offset)
        dno = metric * outward_normal_derivative
        return BIEResult(
            density=density,
            dno=dno,
            boundary_residual=residual,
            offset=offset,
        )


def manufactured_harmonic_solution(
    solver: PeriodicMirrorBIE, mode: int
) -> tuple[Array, Array]:
    """Return surface trace and exact DNO for a bed-Neumann harmonic mode."""
    if mode < 1:
        raise ValueError("mode must be positive")
    k = 2.0 * math.pi * mode / solver.length
    height = solver.eta + solver.depth
    normalization = math.cosh(k * solver.depth)
    trace = np.cos(k * solver.x) * np.cosh(k * height) / normalization
    phi_x = -k * np.sin(k * solver.x) * np.cosh(k * height) / normalization
    phi_z = k * np.cos(k * solver.x) * np.sinh(k * height) / normalization
    exact_dno = phi_z - solver.eta_x * phi_x
    return trace, exact_dno
