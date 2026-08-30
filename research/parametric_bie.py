"""Parametric second-kind BIE for an overhanging periodic free surface.

The bed is flat and enforced by even reflection.  Unlike ``periodic_bie.py``,
the upper boundary is stored as ``(x(alpha), z(alpha))`` and may have vertical
tangents or overhangs, provided it is smooth and does not self-intersect.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ParametricDNOResult:
    density: Array
    unit_normal_derivative: Array
    boundary_residual: float
    offset: float


class ParametricMirrorBIE:
    def __init__(
        self,
        surface_x: Array,
        surface_z: Array,
        length: float,
        depth: float,
    ) -> None:
        surface_x = np.asarray(surface_x, dtype=float)
        surface_z = np.asarray(surface_z, dtype=float)
        if surface_x.shape != surface_z.shape or surface_x.ndim != 1:
            raise ValueError("surface_x and surface_z must be equal one-dimensional arrays")
        if len(surface_x) < 24 or len(surface_x) % 2:
            raise ValueError("an even number N >= 24 of surface markers is required")
        if length <= 0.0 or depth <= 0.0:
            raise ValueError("length and depth must be positive")
        self.x = surface_x.copy()
        self.z = surface_z.copy()
        self.n = len(surface_x)
        self.length = float(length)
        self.depth = float(depth)
        self.dalpha = 2.0 * math.pi / self.n
        self.alpha = np.arange(self.n) * self.dalpha
        self.alpha_modes = 2.0 * math.pi * np.fft.fftfreq(
            self.n, d=self.dalpha
        )

        base_x = self.length * self.alpha / (2.0 * math.pi)
        periodic_displacement = self.x - base_x
        self.x_alpha = self.length / (2.0 * math.pi) + self._derivative(
            periodic_displacement, 1
        )
        self.z_alpha = self._derivative(self.z, 1)
        self.x_alpha_alpha = self._derivative(periodic_displacement, 2)
        self.z_alpha_alpha = self._derivative(self.z, 2)
        self.metric = np.sqrt(self.x_alpha**2 + self.z_alpha**2)
        if np.min(self.metric) <= 1.0e-9:
            raise ValueError("degenerate surface parameterization")
        self.tangent_x = self.x_alpha / self.metric
        self.tangent_z = self.z_alpha / self.metric
        self.normal_x = -self.z_alpha / self.metric
        self.normal_z = self.x_alpha / self.metric

        mirror_z = -2.0 * self.depth - self.z
        self.source_x = np.concatenate((self.x, self.x))
        self.source_z = np.concatenate((self.z, mirror_z))
        self.normal_ds_x = np.concatenate((-self.z_alpha, -self.z_alpha))
        self.normal_ds_z = np.concatenate((self.x_alpha, -self.x_alpha))
        self.double_layer = self._double_layer_matrix()
        self.matrix = self.double_layer - 0.5 * np.eye(2 * self.n)
        self.single_layer = self._single_layer_matrix()

    def _derivative(self, values: Array, order: int) -> Array:
        return np.fft.ifft(
            (1j * self.alpha_modes) ** order * np.fft.fft(values)
        ).real

    def surface_derivative(self, values: Array) -> Array:
        return self._derivative(np.asarray(values, dtype=float), 1) / self.metric

    def _kernel(self, target_x: Array, target_z: Array) -> Array:
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
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            kernel = scale * numerator / (4.0 * math.pi * denominator)
        return kernel * self.dalpha

    def _double_layer_matrix(self) -> Array:
        matrix = self._kernel(self.source_x, self.source_z)
        cross = self.x_alpha * self.z_alpha_alpha - self.z_alpha * self.x_alpha_alpha
        diagonal = cross / (4.0 * math.pi * self.metric**2) * self.dalpha
        indices = np.arange(2 * self.n)
        matrix[indices, indices] = np.concatenate((diagonal, diagonal))
        return matrix

    def _kress_log_weights(self) -> Array:
        """Weights for integral log(4 sin^2((a-b)/2)) f(b) db."""
        delta = self.alpha.reshape(-1, 1) - self.alpha.reshape(1, -1)
        modes = np.arange(1, self.n // 2, dtype=float)
        series = np.sum(
            np.cos(delta[:, :, None] * modes.reshape(1, 1, -1))
            / modes.reshape(1, 1, -1),
            axis=2,
        )
        return (
            -4.0 * math.pi / self.n * series
            - 4.0
            * math.pi
            / self.n**2
            * np.cos((self.n / 2.0) * delta)
        )

    def _periodic_green(self, target_x: Array, target_z: Array) -> Array:
        target_x = np.asarray(target_x, dtype=float).reshape(-1, 1)
        target_z = np.asarray(target_z, dtype=float).reshape(-1, 1)
        delta_x = target_x - self.source_x.reshape(1, -1)
        delta_z = target_z - self.source_z.reshape(1, -1)
        scale = 2.0 * math.pi / self.length
        denominator = 2.0 * (
            np.cosh(scale * delta_z) - np.cos(scale * delta_x)
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return -np.log(denominator) / (4.0 * math.pi)

    def _single_layer_matrix(self) -> Array:
        """Kress product quadrature for the periodic logarithmic kernel."""
        total = 2 * self.n
        matrix = self._periodic_green(self.source_x, self.source_z)
        source_metric = np.concatenate((self.metric, self.metric))
        matrix = matrix * self.dalpha * source_metric.reshape(1, -1)
        log_weights = self._kress_log_weights()
        scale = 2.0 * math.pi / self.length
        delta = self.alpha.reshape(-1, 1) - self.alpha.reshape(1, -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_sine = np.log(4.0 * np.sin(0.5 * delta) ** 2)

        for block in (0, 1):
            row = slice(block * self.n, (block + 1) * self.n)
            column = slice(block * self.n, (block + 1) * self.n)
            green_block = self._periodic_green(
                self.source_x[row], self.source_z[row]
            )[:, column]
            with np.errstate(divide="ignore", invalid="ignore"):
                smooth = green_block + log_sine / (4.0 * math.pi)
            diagonal = -np.log((scale * self.metric) ** 2) / (4.0 * math.pi)
            indices = np.arange(self.n)
            smooth[indices, indices] = diagonal
            singular = -log_weights / (4.0 * math.pi)
            matrix[row, column] = (
                singular + self.dalpha * smooth
            ) * self.metric.reshape(1, -1)
        if matrix.shape != (total, total):
            raise AssertionError("invalid single-layer matrix shape")
        return matrix

    def solve_density(self, trace: Array) -> tuple[Array, float]:
        trace = np.asarray(trace, dtype=float)
        if trace.shape != (self.n,):
            raise ValueError(f"trace must have shape ({self.n},)")
        rhs = np.concatenate((trace, trace))
        density = np.linalg.solve(self.matrix, rhs)
        residual = self.matrix @ density - rhs
        scale = max(float(np.linalg.norm(rhs, ord=np.inf)), 1.0)
        return density, float(np.linalg.norm(residual, ord=np.inf) / scale)

    def evaluate(self, target_x: Array, target_z: Array, density: Array) -> Array:
        target_x, target_z = np.broadcast_arrays(target_x, target_z)
        values = self._kernel(target_x.ravel(), target_z.ravel()) @ density
        return values.reshape(target_x.shape)

    def dirichlet_to_neumann_close(
        self, trace: Array, offset_in_mean_marker_steps: float = 2.25
    ) -> ParametricDNOResult:
        trace = np.asarray(trace, dtype=float)
        density, residual = self.solve_density(trace)
        offset = (
            offset_in_mean_marker_steps * float(np.mean(self.metric)) * self.dalpha
        )
        first = self.evaluate(
            self.x - offset * self.normal_x,
            self.z - offset * self.normal_z,
            density,
        )
        second = self.evaluate(
            self.x - 2.0 * offset * self.normal_x,
            self.z - 2.0 * offset * self.normal_z,
            density,
        )
        q = (3.0 * trace - 4.0 * first + second) / (2.0 * offset)
        return ParametricDNOResult(density, q, residual, offset)

    def dirichlet_to_neumann_singular(self, trace: Array) -> ParametricDNOResult:
        """Direct boundary flux solve with logarithmic singularity subtraction."""
        trace = np.asarray(trace, dtype=float)
        if trace.shape != (self.n,):
            raise ValueError(f"trace must have shape ({self.n},)")
        trace_full = np.concatenate((trace, trace))
        rhs = (0.5 * np.eye(2 * self.n) + self.double_layer) @ trace_full
        normal_derivative = np.linalg.solve(self.single_layer, rhs)
        residual_vector = self.single_layer @ normal_derivative - rhs
        residual = float(
            np.linalg.norm(residual_vector, ord=np.inf)
            / max(float(np.linalg.norm(rhs, ord=np.inf)), 1.0)
        )
        # The direct representation should satisfy global flux compatibility.
        # Remove its roundoff-level constant component symmetrically.
        weights = np.concatenate((self.metric, self.metric)) * self.dalpha
        flux = float(np.dot(weights, normal_derivative))
        normal_derivative -= flux / float(np.sum(weights))
        return ParametricDNOResult(
            normal_derivative,
            normal_derivative[: self.n].copy(),
            residual,
            0.0,
        )

    def dirichlet_to_neumann(
        self, trace: Array, method: str = "singular"
    ) -> ParametricDNOResult:
        if method == "singular":
            return self.dirichlet_to_neumann_singular(trace)
        if method == "close":
            return self.dirichlet_to_neumann_close(trace)
        raise ValueError("method must be 'singular' or 'close'")

    def surface_velocity(
        self, trace: Array, method: str = "singular"
    ) -> tuple[Array, Array, ParametricDNOResult]:
        result = self.dirichlet_to_neumann(trace, method=method)
        tangential = self.surface_derivative(trace)
        velocity_x = tangential * self.tangent_x + result.unit_normal_derivative * self.normal_x
        velocity_z = tangential * self.tangent_z + result.unit_normal_derivative * self.normal_z
        return velocity_x, velocity_z, result


def manufactured_parametric_solution(
    solver: ParametricMirrorBIE, mode: int, phase: float = 0.0
) -> tuple[Array, Array]:
    k = 2.0 * math.pi * mode / solver.length
    angle = k * solver.x - phase
    height = solver.z + solver.depth
    normalization = math.cosh(k * solver.depth)
    trace = np.cos(angle) * np.cosh(k * height) / normalization
    phi_x = -k * np.sin(angle) * np.cosh(k * height) / normalization
    phi_z = k * np.cos(angle) * np.sinh(k * height) / normalization
    exact_q = phi_x * solver.normal_x + phi_z * solver.normal_z
    return trace, exact_q
