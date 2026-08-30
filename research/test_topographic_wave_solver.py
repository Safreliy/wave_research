import math
import unittest

import numpy as np

from solitary_wave import periodic_solitary_wave
from topographic_wave_solver import (
    geometric_volume,
    spectral_viscosity_filter,
    smooth_periodic_shoal,
    smooth_reef_bathymetry,
    step_implicit_midpoint,
    step_rk4,
    theta_s_rhs,
)
from topography_bie import TopographyBIE


class TestTopographicWaveSolver(unittest.TestCase):
    def test_spectral_viscosity_preserves_low_modes_and_damps_nyquist(self) -> None:
        n = 64
        length = 2.0 * math.pi
        alpha = 2.0 * math.pi * np.arange(n) / n
        base = length * np.arange(n) / n
        low = np.sin(2.0 * alpha)
        nyquist = (-1.0) ** np.arange(n)
        x, z, potential, correction = spectral_viscosity_filter(
            base + 0.02 * low,
            low + nyquist,
            0.4 * low - 0.5 * nyquist,
            length,
            strength=36.0,
            order=16,
        )
        self.assertLess(float(np.max(np.abs((x - base) - 0.02 * low))), 1.0e-10)
        self.assertLess(float(np.max(np.abs(z - low))), 1.0e-10)
        self.assertLess(float(np.max(np.abs(potential - 0.4 * low))), 1.0e-10)
        self.assertGreater(correction, 0.0)

    def test_uniform_surface_shift_corrects_volume_exactly(self) -> None:
        n = 64
        length = 2.0 * math.pi
        parameter = 2.0 * math.pi * np.arange(n) / n
        base = length * np.arange(n) / n
        x = base + 0.2 * np.sin(parameter)
        z = 0.1 * np.cos(parameter) + 0.03 * np.sin(2.0 * parameter)
        bottom_x = base.copy()
        bottom_z = -np.ones(n)
        initial = geometric_volume(x, z, bottom_x, bottom_z, length)
        shift = 0.037
        shifted = geometric_volume(x, z + shift, bottom_x, bottom_z, length)
        self.assertAlmostEqual(shifted - initial, length * shift, places=12)

    def test_implicit_midpoint_preserves_rest_state_without_fallback(self) -> None:
        n = 24
        length = 2.0 * math.pi
        x = np.arange(n) * length / n
        z = np.zeros(n)
        potential = np.zeros(n)
        bottom_x, bottom_z = smooth_periodic_shoal(
            n, length, depth=1.0, height=0.0, center=0.0
        )
        result = step_implicit_midpoint(
            x,
            z,
            potential,
            bottom_x,
            bottom_z,
            length,
            gravity=1.0,
            dt=0.05,
            allow_explicit_fallback=False,
        )
        self.assertFalse(result[4].used_explicit_fallback)
        self.assertLess(float(np.max(np.abs(result[0] - x))), 1.0e-13)
        self.assertLess(float(np.max(np.abs(result[1]))), 1.0e-13)
        self.assertLess(float(np.max(np.abs(result[2]))), 1.0e-13)

    def test_implicit_and_rk4_agree_for_small_step(self) -> None:
        n = 24
        length = 12.0
        x = np.arange(n) * length / n
        state = periodic_solitary_wave(x, length, 4.0, 0.08, 1.0)
        bottom_x, bottom_z = smooth_periodic_shoal(
            n, length, depth=1.0, height=0.0, center=0.0
        )
        arguments = (
            x,
            state.elevation,
            state.periodic_potential,
            bottom_x,
            bottom_z,
            length,
            1.0,
            5.0e-4,
            state.background_current,
        )
        explicit = step_rk4(*arguments)
        implicit = step_implicit_midpoint(
            *arguments, allow_explicit_fallback=False
        )
        self.assertTrue(implicit[4].nonlinear.converged)
        difference = np.linalg.norm(
            np.concatenate(implicit[:3]) - np.concatenate(explicit[:3])
        )
        self.assertLess(float(difference), 2.0e-8)

    def test_smooth_reef_has_target_plateaux_and_slope(self) -> None:
        n = 512
        length = 40.0
        x, bottom = smooth_reef_bathymetry(
            n,
            length,
            depth=1.0,
            shallow_depth=0.25,
            slope=0.1,
            toe=15.0,
            shelf_length=4.0,
            return_length=5.0,
            corner_width=0.08,
        )
        deep = bottom[(x > 2.0) & (x < 12.0)]
        shelf = bottom[(x > 23.0) & (x < 25.5)]
        self.assertLess(float(np.max(np.abs(deep + 1.0))), 1.0e-8)
        self.assertLess(float(np.max(np.abs(shelf + 0.25))), 1.0e-4)
        derivative = np.gradient(bottom, x)
        slope_region = derivative[(x > 16.0) & (x < 21.5)]
        self.assertAlmostEqual(float(np.median(slope_region)), 0.1, places=4)

    def test_reef_parameterization_clusters_fixed_boundary_nodes(self) -> None:
        uniform_x, _ = smooth_reef_bathymetry(
            96, 64.0, 1.0, 0.05, 1.0 / 30.0, 19.0, 10.0, 5.0, 0.1
        )
        clustered_x, clustered_z = smooth_reef_bathymetry(
            96,
            64.0,
            1.0,
            0.05,
            1.0 / 30.0,
            19.0,
            10.0,
            5.0,
            0.1,
            cluster_strength=20.0,
            cluster_width=0.4,
        )
        self.assertTrue(np.all(np.diff(clustered_x) > 0.0))
        transition = 19.0 + 0.95 / (1.0 / 30.0)
        local = np.abs(clustered_x - transition) < 0.7
        local_spacing = np.min(np.diff(clustered_x)[np.where(local[:-1])[0]])
        self.assertLess(float(local_spacing), 0.35 * float(np.diff(uniform_x)[0]))
        self.assertLess(float(np.max(clustered_z)), -0.049)

    def test_rest_state_over_shoal_is_stationary(self) -> None:
        n = 32
        length = 2.0 * math.pi
        x = np.arange(n) * length / n
        z = np.zeros(n)
        potential = np.zeros(n)
        bottom_x, bottom_z = smooth_periodic_shoal(
            n, length, depth=1.2, height=0.5, center=4.5
        )
        x_t, z_t, potential_t, residual = theta_s_rhs(
            x,
            z,
            potential,
            bottom_x,
            bottom_z,
            length,
            gravity=1.0,
        )
        self.assertLess(float(np.max(np.abs(x_t))), 1.0e-12)
        self.assertLess(float(np.max(np.abs(z_t))), 1.0e-12)
        self.assertLess(float(np.max(np.abs(potential_t))), 1.0e-12)
        self.assertLess(residual, 1.0e-12)

    def test_uniform_background_current_over_flat_bed_is_steady(self) -> None:
        n = 32
        length = 2.0 * math.pi
        x = np.arange(n) * length / n
        bottom_x, bottom_z = smooth_periodic_shoal(
            n, length, depth=1.0, height=0.0, center=0.0
        )
        x_t, z_t, potential_t, residual = theta_s_rhs(
            x,
            np.zeros(n),
            np.zeros(n),
            bottom_x,
            bottom_z,
            length,
            gravity=1.0,
            background_current=0.17,
        )
        self.assertLess(float(np.max(np.abs(x_t - 0.17))), 1.0e-12)
        self.assertLess(float(np.max(np.abs(z_t))), 1.0e-12)
        self.assertLess(float(np.max(np.abs(potential_t))), 1.0e-12)
        self.assertLess(residual, 1.0e-12)

    def test_solitary_initializer_approximately_satisfies_travelling_kinematics(self) -> None:
        n = 128
        length = 40.0
        x = np.arange(n) * length / n
        state = periodic_solitary_wave(
            x, length, center=15.0, amplitude=0.2, depth=1.0
        )
        bottom_x, bottom_z = smooth_periodic_shoal(
            n, length, depth=1.0, height=0.0, center=0.0
        )
        x_t, z_t, _, _ = theta_s_rhs(
            x,
            state.elevation,
            state.periodic_potential,
            bottom_x,
            bottom_z,
            length,
            gravity=1.0,
            background_current=state.background_current,
        )
        geometry = TopographyBIE(
            x, state.elevation, bottom_x, bottom_z, length
        ).surface
        normal_x = -geometry.z_alpha / geometry.metric
        normal_z = geometry.x_alpha / geometry.metric
        computed = x_t * normal_x + z_t * normal_z
        expected = -state.speed * geometry.z_alpha / geometry.metric
        scale = max(float(np.linalg.norm(expected)), 1.0e-14)
        relative_error = float(np.linalg.norm(computed - expected) / scale)
        self.assertLess(relative_error, 0.08)

    def test_close_boundary_oversampling_restores_flux_compatibility(self) -> None:
        n = 40
        length = 20.0
        x = np.arange(n) * length / n
        state = periodic_solitary_wave(
            x, length, center=5.0, amplitude=0.4, depth=1.0
        )
        bottom_x, bottom_z = smooth_periodic_shoal(
            n, length, depth=1.0, height=0.75, center=13.0
        )

        def total_flux(cross_oversampling: int | None) -> float:
            bie = TopographyBIE(
                x,
                state.elevation,
                bottom_x,
                bottom_z,
                length,
                cross_oversampling=cross_oversampling,
            )
            bottom_flux = (
                -state.background_current
                * bie.bottom.z_alpha
                / bie.bottom.metric
            )
            result = bie.solve(state.periodic_potential, bottom_flux)
            surface_normal_x = -bie.surface.z_alpha / bie.surface.metric
            normal_velocity = (
                result.surface_normal_derivative
                + state.background_current * surface_normal_x
            )
            return float(
                bie.dalpha * np.sum(normal_velocity * bie.surface.metric)
            )

        plain_error = abs(total_flux(1))
        corrected_error = abs(total_flux(None))
        self.assertGreater(plain_error, 1.0e-2)
        self.assertLess(corrected_error, 2.0e-6)


if __name__ == "__main__":
    unittest.main()
