import math
import unittest

import numpy as np

from lagrangian_bie_solver import (
    initial_localized_breaker,
    monitor_equidistribution_cv,
    rhs,
    spectral_arclength_resample,
    uniform_arclength_resample,
)
from parametric_bie import ParametricMirrorBIE


class TestThetaSGauge(unittest.TestCase):
    def test_arclength_gauge_makes_metric_rate_uniform(self) -> None:
        n = 48
        length = 2.0 * math.pi
        x, z, potential = initial_localized_breaker(
            n, length, amplitude=0.08, jet_speed=0.3, width=0.9
        )
        x, z, potential = uniform_arclength_resample(x, z, potential, length)
        velocity_x, velocity_z, _, _ = rhs(
            x,
            z,
            potential,
            length,
            depth=1.0,
            gravity=1.0,
            tangential_mode="theta_s",
        )
        geometry = ParametricMirrorBIE(x, z, length, 1.0)
        velocity_x_alpha = geometry._derivative(velocity_x, 1)
        velocity_z_alpha = geometry._derivative(velocity_z, 1)
        metric_rate = (
            geometry.x_alpha * velocity_x_alpha
            + geometry.z_alpha * velocity_z_alpha
        ) / geometry.metric
        self.assertLess(float(np.std(metric_rate)), 5.0e-6)

    def test_spectral_reparameterization_equalizes_markers(self) -> None:
        n = 48
        length = 2.0 * math.pi
        alpha = 2.0 * math.pi * np.arange(n) / n
        x = alpha + 0.28 * np.sin(alpha)
        z = 0.20 * np.cos(alpha)
        potential = 0.1 * np.sin(2.0 * alpha)
        before = ParametricMirrorBIE(x, z, length, 1.0)
        new_x, new_z, new_potential = spectral_arclength_resample(
            x, z, potential, length, refinement=12
        )
        after = ParametricMirrorBIE(new_x, new_z, length, 1.0)
        before_cv = float(np.std(before.metric) / np.mean(before.metric))
        after_cv = float(np.std(after.metric) / np.mean(after.metric))
        self.assertGreater(before_cv, 0.15)
        self.assertLess(after_cv, 2.0e-5)
        self.assertLess(abs(float(np.mean(new_potential))), 1.0e-14)

    def test_curvature_monitor_clusters_markers_and_is_equidistributed(self) -> None:
        n = 64
        length = 2.0 * math.pi
        alpha = 2.0 * math.pi * np.arange(n) / n
        x = alpha
        z = 0.15 * np.cos(alpha) + 0.08 * np.cos(3.0 * alpha)
        potential = np.sin(alpha)
        uniform_x, uniform_z, _ = spectral_arclength_resample(
            x, z, potential, length, refinement=16
        )
        adaptive_x, adaptive_z, _ = spectral_arclength_resample(
            x,
            z,
            potential,
            length,
            refinement=16,
            curvature_strength=8.0,
        )
        uniform_panel = np.sqrt(
            (np.roll(uniform_x, -1) - uniform_x) ** 2
            + (np.roll(uniform_z, -1) - uniform_z) ** 2
        )[:-1]
        adaptive_panel = np.sqrt(
            (np.roll(adaptive_x, -1) - adaptive_x) ** 2
            + (np.roll(adaptive_z, -1) - adaptive_z) ** 2
        )[:-1]
        self.assertLess(float(np.min(adaptive_panel)), 0.8 * float(np.min(uniform_panel)))
        self.assertLess(
            monitor_equidistribution_cv(
                adaptive_x, adaptive_z, length, curvature_strength=8.0
            ),
            0.25,
        )


if __name__ == "__main__":
    unittest.main()
