"""Mixed periodic BIE for a free surface over arbitrary smooth topography.

Known data are Dirichlet values on the free surface and zero Neumann flux on
the fixed bottom.  Green's boundary identity yields a square system for the
unknown surface normal derivative and bottom potential:

    [S_:f  -(1/2 I + K)_:b] [q_f, phi_b]^T
        = (1/2 I + K)_:f psi_f.

Logarithmic self interactions use Kress product quadrature.  The implementation
is dense and verification-oriented.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


Array = np.ndarray


# Fourier evaluation and trigonometric interpolation matrices depend only on
# the marker count and oversampling factor.  A shoaling RK stage constructs a
# fresh BIE several times, so keeping this immutable geometry-independent part
# at module scope avoids rebuilding the same dense phases in every stage.
_FINE_SPECTRAL_CACHE: dict[
    tuple[int, int], tuple[Array, Array, Array, Array]
] = {}


def _stable_periodic_kernel_terms(
    scaled_x: Array, scaled_z: Array
) -> tuple[Array, Array, Array]:
    """Return sin/D, sinh/D and log(2D) without hyperbolic overflow.

    Here ``D = cosh(scaled_z) - cos(scaled_x)``.  Factoring
    ``exp(abs(scaled_z))`` keeps all terms finite even after a rejected
    Runge--Kutta stage has produced a wildly distorted trial geometry.
    """
    absolute_z = np.abs(scaled_z)
    exponential = np.exp(-absolute_z)
    scaled_denominator = (
        1.0
        + exponential**2
        - 2.0 * np.cos(scaled_x) * exponential
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        sine_ratio = (
            2.0 * exponential * np.sin(scaled_x) / scaled_denominator
        )
        sinh_ratio = (
            np.sign(scaled_z)
            * (1.0 - exponential**2)
            / scaled_denominator
        )
        log_twice_denominator = absolute_z + np.log(scaled_denominator)
    return sine_ratio, sinh_ratio, log_twice_denominator


@dataclass(frozen=True)
class MixedBIEResult:
    surface_normal_derivative: Array
    bottom_potential: Array
    residual: float
    condition_number: float
    linear_iterations: int = 0
    linear_solver: str = "direct"


@dataclass(frozen=True)
class MixedNeumannResult:
    surface_potential: Array
    bottom_potential: Array
    residual: float


@dataclass(frozen=True)
class CurveGeometry:
    x: Array
    z: Array
    x_alpha: Array
    z_alpha: Array
    x_alpha_alpha: Array
    z_alpha_alpha: Array
    metric: Array


class TopographyBIE:
    def __init__(
        self,
        surface_x: Array,
        surface_z: Array,
        bottom_x: Array,
        bottom_z: Array,
        length: float,
        cross_oversampling: int | None = None,
        surface_self_oversampling: int | None = None,
    ) -> None:
        arrays = [np.asarray(values, dtype=float) for values in (
            surface_x, surface_z, bottom_x, bottom_z
        )]
        if any(values.ndim != 1 for values in arrays):
            raise ValueError("all curves must be one-dimensional arrays")
        sizes = {len(values) for values in arrays}
        if len(sizes) != 1:
            raise ValueError("surface and bottom must use the same marker count")
        self.n = len(arrays[0])
        if self.n < 24 or self.n % 2:
            raise ValueError("an even marker count N >= 24 is required")
        if length <= 0.0:
            raise ValueError("length must be positive")
        self.length = float(length)
        self.dalpha = 2.0 * math.pi / self.n
        self.alpha = np.arange(self.n) * self.dalpha
        self.alpha_modes = 2.0 * math.pi * np.fft.fftfreq(
            self.n, d=self.dalpha
        )
        self.surface = self._geometry(arrays[0], arrays[1])
        self.bottom = self._geometry(arrays[2], arrays[3])
        if cross_oversampling is None:
            separation = np.sqrt(
                (self.surface.x[:, None] - self.bottom.x[None, :]) ** 2
                + (self.surface.z[:, None] - self.bottom.z[None, :]) ** 2
            )
            minimum_separation = max(float(np.min(separation)), 1.0e-12)
            boundary_spacing = self.dalpha * max(
                float(np.mean(self.surface.metric)),
                float(np.mean(self.bottom.metric)),
            )
            cross_oversampling = int(
                np.clip(math.ceil(2.5 * boundary_spacing / minimum_separation), 1, 8)
            )
        if cross_oversampling < 1:
            raise ValueError("cross_oversampling must be a positive integer")
        self.cross_oversampling = int(cross_oversampling)
        (
            self.minimum_nonlocal_surface_separation,
            self.mean_surface_spacing,
        ) = self._nonlocal_surface_separation()
        if surface_self_oversampling is None:
            separation_ratio = self.mean_surface_spacing / max(
                self.minimum_nonlocal_surface_separation, 1.0e-12
            )
            # The periodic Kress rule resolves the diagonal singularity.  It
            # does not resolve a second, parameter-distant branch that enters
            # the near field of a target marker.  Activate a correction only
            # once that nonlocal gap is below 1.5 mean panel lengths.  A looser
            # trigger mistakes ordinary same-branch neighbours for an
            # approaching plunging jet.
            surface_self_oversampling = (
                int(np.clip(math.ceil(4.0 * separation_ratio), 2, 16))
                if separation_ratio > (2.0 / 3.0)
                else 1
            )
        if surface_self_oversampling < 1:
            raise ValueError(
                "surface_self_oversampling must be a positive integer"
            )
        self.surface_self_oversampling = int(surface_self_oversampling)
        self.source_x = np.concatenate((self.surface.x, self.bottom.x))
        self.source_z = np.concatenate((self.surface.z, self.bottom.z))
        # Top outward normal is left of its left-to-right tangent; bottom
        # outward normal is right of its left-to-right tangent.
        self.normal_ds_x = np.concatenate(
            (-self.surface.z_alpha, self.bottom.z_alpha)
        )
        self.normal_ds_z = np.concatenate(
            (self.surface.x_alpha, -self.bottom.x_alpha)
        )
        self.metrics = np.concatenate((self.surface.metric, self.bottom.metric))
        self._fine_geometry_cache: dict[
            tuple[int, int], tuple[Array, Array, Array, Array, Array, Array, Array]
        ] = {}
        self.double_layer = self._double_layer_matrix()
        self.single_layer = self._single_layer_matrix()
        self.half_plus_k = 0.5 * np.eye(2 * self.n) + self.double_layer
        self.mixed_matrix = np.hstack(
            (
                self.single_layer[:, : self.n],
                -self.half_plus_k[:, self.n :],
            )
        )

    def _nonlocal_surface_separation(self) -> tuple[float, float]:
        """Return the closest parameter-distant surface pair and mean ds.

        A local arclength neighbourhood of eight *mean panel lengths* is
        excluded because it belongs to the diagonal singularity already
        handled by Kress quadrature.  An index-only exclusion is incorrect
        after r-adaptation: packed markers can make an ordinary same-branch
        point look nonlocal and spuriously switch on close evaluation.  The x
        distance is periodic, so a jet approaching the next periodic image is
        still detected.
        """
        delta_x = self.surface.x[:, None] - self.surface.x[None, :]
        delta_x = (delta_x + 0.5 * self.length) % self.length - 0.5 * self.length
        delta_z = self.surface.z[:, None] - self.surface.z[None, :]
        distance = np.sqrt(delta_x**2 + delta_z**2)
        following_x = np.roll(self.surface.x, -1)
        following_x[-1] += self.length
        panel_length = np.hypot(
            following_x - self.surface.x,
            np.roll(self.surface.z, -1) - self.surface.z,
        )
        cumulative = np.r_[0.0, np.cumsum(panel_length)]
        node_arclength = cumulative[:-1]
        total_arclength = float(cumulative[-1])
        arc_distance = np.abs(
            node_arclength[:, None] - node_arclength[None, :]
        )
        arc_distance = np.minimum(
            arc_distance, total_arclength - arc_distance
        )
        mean_spacing = total_arclength / self.n
        distance[arc_distance <= 8.0 * mean_spacing] = np.inf
        return float(np.min(distance)), mean_spacing

    def _derivative(self, values: Array, order: int) -> Array:
        return np.fft.ifft(
            (1j * self.alpha_modes) ** order * np.fft.fft(values)
        ).real

    def _geometry(self, x: Array, z: Array) -> CurveGeometry:
        base_x = self.length * self.alpha / (2.0 * math.pi)
        displacement = x - base_x
        x_alpha = self.length / (2.0 * math.pi) + self._derivative(
            displacement, 1
        )
        z_alpha = self._derivative(z, 1)
        x_alpha_alpha = self._derivative(displacement, 2)
        z_alpha_alpha = self._derivative(z, 2)
        metric = np.sqrt(x_alpha**2 + z_alpha**2)
        if np.min(metric) <= 1.0e-10:
            raise ValueError("degenerate boundary parameterization")
        return CurveGeometry(
            x.copy(),
            z.copy(),
            x_alpha,
            z_alpha,
            x_alpha_alpha,
            z_alpha_alpha,
            metric,
        )

    def surface_derivative(self, values: Array) -> Array:
        return self._derivative(np.asarray(values, dtype=float), 1) / self.surface.metric

    def _double_layer_kernel(self, target_x: Array, target_z: Array) -> Array:
        target_x = np.asarray(target_x, dtype=float).reshape(-1, 1)
        target_z = np.asarray(target_z, dtype=float).reshape(-1, 1)
        delta_x = target_x - self.source_x.reshape(1, -1)
        delta_z = target_z - self.source_z.reshape(1, -1)
        scale = 2.0 * math.pi / self.length
        sine_ratio, sinh_ratio, _ = _stable_periodic_kernel_terms(
            scale * delta_x, scale * delta_z
        )
        kernel = scale / (4.0 * math.pi) * (
            sine_ratio * self.normal_ds_x.reshape(1, -1)
            + sinh_ratio * self.normal_ds_z.reshape(1, -1)
        )
        return kernel * self.dalpha

    def _double_layer_matrix(self) -> Array:
        matrix = self._double_layer_kernel(self.source_x, self.source_z)
        surface_cross = (
            self.surface.x_alpha * self.surface.z_alpha_alpha
            - self.surface.z_alpha * self.surface.x_alpha_alpha
        )
        bottom_cross = (
            self.bottom.x_alpha * self.bottom.z_alpha_alpha
            - self.bottom.z_alpha * self.bottom.x_alpha_alpha
        )
        surface_diagonal = (
            surface_cross / (4.0 * math.pi * self.surface.metric**2) * self.dalpha
        )
        bottom_diagonal = (
            -bottom_cross / (4.0 * math.pi * self.bottom.metric**2) * self.dalpha
        )
        indices = np.arange(2 * self.n)
        matrix[indices, indices] = np.concatenate(
            (surface_diagonal, bottom_diagonal)
        )
        if self.cross_oversampling > 1:
            matrix[: self.n, self.n :] = self._oversampled_cross_double_layer(
                self.surface, self.bottom, orientation=-1.0
            )
            matrix[self.n :, : self.n] = self._oversampled_cross_double_layer(
                self.bottom, self.surface, orientation=1.0
            )
        if self.surface_self_oversampling > 1:
            matrix[: self.n, : self.n] = self._oversampled_self_double_layer(
                self.surface
            )
        return matrix

    def _fine_geometry(
        self, geometry: CurveGeometry, oversampling: int
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        cache_key = (id(geometry), oversampling)
        cached = self._fine_geometry_cache.get(cache_key)
        if cached is not None:
            return cached
        count = oversampling * self.n
        spectral_key = (self.n, oversampling)
        spectral = _FINE_SPECTRAL_CACHE.get(spectral_key)
        if spectral is None:
            fine_alpha = 2.0 * math.pi * np.arange(count) / count
            modes = self.alpha_modes
            phase = np.exp(1j * fine_alpha[:, None] * modes[None, :])
            delta = fine_alpha[:, None] - self.alpha[None, :]
            interpolation = np.ones((count, self.n))
            for mode in range(1, self.n // 2):
                interpolation += 2.0 * np.cos(mode * delta)
            interpolation += np.cos((self.n // 2) * delta)
            interpolation /= self.n
            spectral = (fine_alpha, modes, phase, interpolation)
            _FINE_SPECTRAL_CACHE[spectral_key] = spectral
        else:
            fine_alpha, modes, phase, interpolation = spectral
        base = self.length * self.alpha / (2.0 * math.pi)
        displacement_coefficients = np.fft.fft(geometry.x - base) / self.n
        z_coefficients = np.fft.fft(geometry.z) / self.n
        fine_x = self.length * fine_alpha / (2.0 * math.pi) + (
            phase @ displacement_coefficients
        ).real
        fine_z = (phase @ z_coefficients).real
        fine_x_alpha = self.length / (2.0 * math.pi) + (
            phase @ (1j * modes * displacement_coefficients)
        ).real
        fine_z_alpha = (phase @ (1j * modes * z_coefficients)).real
        fine_metric = np.sqrt(fine_x_alpha**2 + fine_z_alpha**2)
        result = (
            fine_alpha,
            fine_x,
            fine_z,
            fine_x_alpha,
            fine_z_alpha,
            fine_metric,
            interpolation,
        )
        self._fine_geometry_cache[cache_key] = result
        return result

    def _oversampled_cross_double_layer(
        self,
        target: CurveGeometry,
        source: CurveGeometry,
        orientation: float,
    ) -> Array:
        _, source_x, source_z, x_alpha, z_alpha, _, interpolation = (
            self._fine_geometry(source, self.cross_oversampling)
        )
        delta_x = target.x[:, None] - source_x[None, :]
        delta_z = target.z[:, None] - source_z[None, :]
        scale = 2.0 * math.pi / self.length
        normal_ds_x = orientation * (-z_alpha)
        normal_ds_z = orientation * x_alpha
        sine_ratio, sinh_ratio, _ = _stable_periodic_kernel_terms(
            scale * delta_x, scale * delta_z
        )
        fine_weight = 2.0 * math.pi / len(source_x)
        kernel = scale / (4.0 * math.pi) * (
            sine_ratio * normal_ds_x[None, :]
            + sinh_ratio * normal_ds_z[None, :]
        ) * fine_weight
        return kernel @ interpolation

    def _oversampled_self_double_layer(
        self, geometry: CurveGeometry
    ) -> Array:
        """Nystr\N{LATIN SMALL LETTER O WITH DIAERESIS}m double layer on an oversampled source curve."""
        factor = self.surface_self_oversampling
        (
            fine_alpha,
            fine_x,
            fine_z,
            fine_x_alpha,
            fine_z_alpha,
            fine_metric,
            interpolation,
        ) = self._fine_geometry(geometry, factor)
        fine_weight = 2.0 * math.pi / len(fine_alpha)
        scale = 2.0 * math.pi / self.length
        delta_x = geometry.x[:, None] - fine_x[None, :]
        delta_z = geometry.z[:, None] - fine_z[None, :]
        sine_ratio, sinh_ratio, _ = _stable_periodic_kernel_terms(
            scale * delta_x, scale * delta_z
        )
        with np.errstate(invalid="ignore"):
            fine_matrix = scale / (4.0 * math.pi) * (
                sine_ratio * (-fine_z_alpha)[None, :]
                + sinh_ratio * fine_x_alpha[None, :]
            ) * fine_weight
        target_columns = factor * np.arange(self.n)
        cross = (
            geometry.x_alpha * geometry.z_alpha_alpha
            - geometry.z_alpha * geometry.x_alpha_alpha
        )
        diagonal = (
            cross
            / (4.0 * math.pi * geometry.metric**2)
            * fine_weight
        )
        fine_matrix[np.arange(self.n), target_columns] = diagonal
        return fine_matrix @ interpolation

    @staticmethod
    def _circulant_kress_weights(count: int) -> Array:
        """First row of Kress logarithmic product weights on ``count`` nodes."""
        coefficients = np.zeros(count, dtype=complex)
        modes = np.arange(1, count // 2, dtype=float)
        coefficients[1 : count // 2] = count / (2.0 * modes)
        coefficients[count // 2 + 1 :] = count / (2.0 * modes[::-1])
        series = np.fft.ifft(coefficients).real
        indices = np.arange(count)
        return (
            -4.0 * math.pi / count * series
            - 4.0 * math.pi / count**2 * np.cos(math.pi * indices)
        )

    def _oversampled_self_single_layer(
        self, geometry: CurveGeometry
    ) -> Array:
        """Kress single layer evaluated on an oversampled source curve."""
        factor = self.surface_self_oversampling
        (
            fine_alpha,
            fine_x,
            fine_z,
            _,
            _,
            fine_metric,
            interpolation,
        ) = self._fine_geometry(geometry, factor)
        count = len(fine_alpha)
        fine_weight = 2.0 * math.pi / count
        scale = 2.0 * math.pi / self.length
        delta_x = geometry.x[:, None] - fine_x[None, :]
        delta_z = geometry.z[:, None] - fine_z[None, :]
        _, _, logarithm = _stable_periodic_kernel_terms(
            scale * delta_x, scale * delta_z
        )
        green = -logarithm / (4.0 * math.pi)
        delta_alpha = self.alpha[:, None] - fine_alpha[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_sine = np.log(4.0 * np.sin(0.5 * delta_alpha) ** 2)
            smooth = green + log_sine / (4.0 * math.pi)
        target_columns = factor * np.arange(self.n)
        smooth[np.arange(self.n), target_columns] = -np.log(
            (scale * fine_metric[target_columns]) ** 2
        ) / (4.0 * math.pi)
        base_weights = self._circulant_kress_weights(count)
        source_indices = np.arange(count)[None, :]
        target_indices = target_columns[:, None]
        singular_weights = base_weights[
            (source_indices - target_indices) % count
        ]
        fine_matrix = (
            -singular_weights / (4.0 * math.pi)
            + fine_weight * smooth
        ) * fine_metric[None, :]
        return fine_matrix @ interpolation

    def _periodic_green(self, target_x: Array, target_z: Array) -> Array:
        target_x = np.asarray(target_x, dtype=float).reshape(-1, 1)
        target_z = np.asarray(target_z, dtype=float).reshape(-1, 1)
        delta_x = target_x - self.source_x.reshape(1, -1)
        delta_z = target_z - self.source_z.reshape(1, -1)
        scale = 2.0 * math.pi / self.length
        _, _, log_twice_denominator = _stable_periodic_kernel_terms(
            scale * delta_x, scale * delta_z
        )
        return -log_twice_denominator / (4.0 * math.pi)

    def _kress_log_weights(self) -> Array:
        delta = self.alpha.reshape(-1, 1) - self.alpha.reshape(1, -1)
        modes = np.arange(1, self.n // 2, dtype=float)
        series = np.sum(
            np.cos(delta[:, :, None] * modes.reshape(1, 1, -1))
            / modes.reshape(1, 1, -1),
            axis=2,
        )
        return (
            -4.0 * math.pi / self.n * series
            - 4.0 * math.pi / self.n**2 * np.cos((self.n / 2.0) * delta)
        )

    def _single_layer_matrix(self) -> Array:
        matrix = self._periodic_green(self.source_x, self.source_z)
        matrix = matrix * self.dalpha * self.metrics.reshape(1, -1)
        log_weights = self._kress_log_weights()
        scale = 2.0 * math.pi / self.length
        delta = self.alpha.reshape(-1, 1) - self.alpha.reshape(1, -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_sine = np.log(4.0 * np.sin(0.5 * delta) ** 2)

        for block, geometry in enumerate((self.surface, self.bottom)):
            rows = slice(block * self.n, (block + 1) * self.n)
            columns = slice(block * self.n, (block + 1) * self.n)
            green_block = self._periodic_green(
                self.source_x[rows], self.source_z[rows]
            )[:, columns]
            with np.errstate(divide="ignore", invalid="ignore"):
                smooth = green_block + log_sine / (4.0 * math.pi)
            diagonal = -np.log((scale * geometry.metric) ** 2) / (
                4.0 * math.pi
            )
            indices = np.arange(self.n)
            smooth[indices, indices] = diagonal
            singular = -log_weights / (4.0 * math.pi)
            matrix[rows, columns] = (
                singular + self.dalpha * smooth
            ) * geometry.metric.reshape(1, -1)
        if self.cross_oversampling > 1:
            matrix[: self.n, self.n :] = self._oversampled_cross_single_layer(
                self.surface, self.bottom
            )
            matrix[self.n :, : self.n] = self._oversampled_cross_single_layer(
                self.bottom, self.surface
            )
        if self.surface_self_oversampling > 1:
            matrix[: self.n, : self.n] = self._oversampled_self_single_layer(
                self.surface
            )
        return matrix

    def _oversampled_cross_single_layer(
        self, target: CurveGeometry, source: CurveGeometry
    ) -> Array:
        _, source_x, source_z, _, _, metric, interpolation = self._fine_geometry(
            source, self.cross_oversampling
        )
        delta_x = target.x[:, None] - source_x[None, :]
        delta_z = target.z[:, None] - source_z[None, :]
        scale = 2.0 * math.pi / self.length
        _, _, log_twice_denominator = _stable_periodic_kernel_terms(
            scale * delta_x, scale * delta_z
        )
        green = -log_twice_denominator / (4.0 * math.pi)
        fine_weight = 2.0 * math.pi / len(source_x)
        return (green * (fine_weight * metric)[None, :]) @ interpolation

    def solve(
        self,
        surface_trace: Array,
        bottom_normal_derivative: Array | None = None,
        compute_condition_number: bool = False,
    ) -> MixedBIEResult:
        surface_trace = np.asarray(surface_trace, dtype=float)
        if surface_trace.shape != (self.n,):
            raise ValueError(f"surface_trace must have shape ({self.n},)")
        if bottom_normal_derivative is None:
            bottom_normal_derivative = np.zeros(self.n)
        bottom_normal_derivative = np.asarray(bottom_normal_derivative, dtype=float)
        if bottom_normal_derivative.shape != (self.n,):
            raise ValueError(
                f"bottom_normal_derivative must have shape ({self.n},)"
            )
        rhs = (
            self.half_plus_k[:, : self.n] @ surface_trace
            - self.single_layer[:, self.n :] @ bottom_normal_derivative
        )
        unknown, linear_iterations, linear_solver = self._solve_mixed_system(rhs)
        residual_vector = self.mixed_matrix @ unknown - rhs
        residual = float(
            np.linalg.norm(residual_vector, ord=np.inf)
            / max(float(np.linalg.norm(rhs, ord=np.inf)), 1.0)
        )
        return MixedBIEResult(
            surface_normal_derivative=unknown[: self.n].copy(),
            bottom_potential=unknown[self.n :].copy(),
            residual=residual,
            condition_number=(
                float(np.linalg.cond(self.mixed_matrix))
                if compute_condition_number
                else float("nan")
            ),
            linear_iterations=linear_iterations,
            linear_solver=linear_solver,
        )

    def solve_neumann_to_dirichlet(
        self,
        surface_normal_derivative: Array,
        bottom_normal_derivative: Array | None = None,
    ) -> MixedNeumannResult:
        """Recover boundary potentials from compatible Neumann data.

        The potential is defined up to a constant.  We replace one redundant
        boundary-identity row by a zero-mean surface-potential constraint and
        report the residual of the unreplaced equations separately.
        """
        surface_normal_derivative = np.asarray(
            surface_normal_derivative, dtype=float
        )
        if surface_normal_derivative.shape != (self.n,):
            raise ValueError(
                f"surface_normal_derivative must have shape ({self.n},)"
            )
        if bottom_normal_derivative is None:
            bottom_normal_derivative = np.zeros(self.n)
        bottom_normal_derivative = np.asarray(
            bottom_normal_derivative, dtype=float
        )
        if bottom_normal_derivative.shape != (self.n,):
            raise ValueError(
                f"bottom_normal_derivative must have shape ({self.n},)"
            )
        flux = self.dalpha * (
            np.dot(self.surface.metric, surface_normal_derivative)
            + np.dot(self.bottom.metric, bottom_normal_derivative)
        )
        flux_scale = max(
            self.dalpha
            * (
                np.dot(self.surface.metric, np.abs(surface_normal_derivative))
                + np.dot(self.bottom.metric, np.abs(bottom_normal_derivative))
            ),
            1.0,
        )
        if abs(float(flux)) > 5.0e-9 * float(flux_scale):
            raise ValueError("Neumann data violate global flux compatibility")
        operator = np.hstack(
            (
                self.half_plus_k[:, : self.n],
                self.half_plus_k[:, self.n :],
            )
        )
        rhs = (
            self.single_layer[:, : self.n] @ surface_normal_derivative
            + self.single_layer[:, self.n :] @ bottom_normal_derivative
        )
        # The continuous operator has the constant-potential null vector.
        # A minimum-norm least-squares solve is more robust than replacing an
        # arbitrary collocation row, especially for a very shallow shelf.
        unknown = np.linalg.lstsq(operator, rhs, rcond=2.0e-13)[0]
        gauge = float(np.mean(unknown[: self.n]))
        unknown -= gauge
        residual_vector = operator @ unknown - rhs
        residual = float(
            np.linalg.norm(residual_vector, ord=np.inf)
            / max(float(np.linalg.norm(rhs, ord=np.inf)), 1.0)
        )
        return MixedNeumannResult(
            surface_potential=unknown[: self.n].copy(),
            bottom_potential=unknown[self.n :].copy(),
            residual=residual,
        )

    def _solve_mixed_system(self, rhs: Array) -> tuple[Array, int, str]:
        """Solve the dense mixed system by full GMRES with safe fallback.

        The verification prototype originally called ``numpy.linalg.solve``
        at every Runge--Kutta stage.  On the bundled NumPy runtime that cubic
        solve dominates the calculation even for 64--128 markers.  The mixed
        second-kind boundary system normally reaches a 2e-12 relative
        residual in far fewer than its full dimension using only dense
        matrix-vector products.  A direct solve remains the correctness
        fallback if the Krylov residual stalls.
        """
        matrix = self.mixed_matrix
        dimension = len(rhs)
        maximum_iterations = min(dimension, 96)
        tolerance = 2.0e-12
        scale = max(float(np.linalg.norm(rhs, ord=np.inf)), 1.0)
        residual = rhs.copy()
        beta = float(np.linalg.norm(residual))
        if beta <= tolerance * scale:
            return np.zeros_like(rhs), 0, "gmres"
        basis = np.zeros((dimension, maximum_iterations + 1))
        hessenberg = np.zeros((maximum_iterations + 1, maximum_iterations))
        projected_rhs = np.zeros(maximum_iterations + 1)
        basis[:, 0] = residual / beta
        projected_rhs[0] = beta
        candidate = np.zeros_like(rhs)
        relative_residual = float("inf")
        for iteration in range(maximum_iterations):
            vector = matrix @ basis[:, iteration]
            # Modified Gram--Schmidt with a second pass is robust enough for
            # the moderately conditioned BIE matrices encountered here.
            for _ in range(2):
                coefficients = basis[:, : iteration + 1].T @ vector
                hessenberg[: iteration + 1, iteration] += coefficients
                vector -= basis[:, : iteration + 1] @ coefficients
            next_norm = float(np.linalg.norm(vector))
            hessenberg[iteration + 1, iteration] = next_norm
            if next_norm > 10.0 * np.finfo(float).eps:
                basis[:, iteration + 1] = vector / next_norm
            should_check = (
                (iteration + 1) % 4 == 0
                or iteration + 1 == maximum_iterations
                or next_norm <= 10.0 * np.finfo(float).eps
            )
            if should_check:
                active = iteration + 1
                coefficients = np.linalg.lstsq(
                    hessenberg[: active + 1, :active],
                    projected_rhs[: active + 1],
                    rcond=None,
                )[0]
                candidate = basis[:, :active] @ coefficients
                relative_residual = float(
                    np.linalg.norm(matrix @ candidate - rhs, ord=np.inf)
                    / scale
                )
                if relative_residual <= tolerance:
                    return candidate, active, "gmres"
            if next_norm <= 10.0 * np.finfo(float).eps:
                break
        # A stalled Krylov solve must never silently contaminate conservation
        # diagnostics or breaking criteria.
        direct = np.linalg.solve(matrix, rhs)
        return direct, dimension, "direct_fallback"

    def surface_velocity(
        self,
        surface_trace: Array,
        bottom_normal_derivative: Array | None = None,
    ) -> tuple[Array, Array, MixedBIEResult]:
        result = self.solve(surface_trace, bottom_normal_derivative)
        tangential = self.surface_derivative(surface_trace)
        tangent_x = self.surface.x_alpha / self.surface.metric
        tangent_z = self.surface.z_alpha / self.surface.metric
        normal_x = -tangent_z
        normal_z = tangent_x
        return (
            tangential * tangent_x + result.surface_normal_derivative * normal_x,
            tangential * tangent_z + result.surface_normal_derivative * normal_z,
            result,
        )


def conformal_manufactured_geometry(
    n: int,
    depth: float,
    amplitude: float,
    mode: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Analytic conformal image with exact bed-Neumann harmonic data."""
    if amplitude * math.cosh(depth) >= 0.45 * depth:
        raise ValueError("mapping amplitude is too large for a safe strip")
    xi = np.arange(n) * 2.0 * math.pi / n
    surface_x = xi.copy()
    surface_z = amplitude * np.cos(xi)
    bottom_x = xi - amplitude * np.sin(xi) * math.sinh(depth)
    bottom_z = -depth + amplitude * np.cos(xi) * math.cosh(depth)
    trace = np.cos(mode * xi)
    map_scale_top = np.sqrt(1.0 + amplitude**2 * np.sin(xi) ** 2)
    exact_surface_q = (
        mode * math.tanh(mode * depth) * np.cos(mode * xi) / map_scale_top
    )
    return surface_x, surface_z, bottom_x, bottom_z, trace, exact_surface_q
