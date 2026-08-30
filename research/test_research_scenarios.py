import math
import unittest

import numpy as np

from hos_solver import HOSConfig, HOSSolver
from hybrid_vortex_operator import (
    VortexCloud,
    velocity_above_flat_bed,
    vortex_dipole,
)
from wave_scenarios import dispersive_focusing, head_on_solitary_pair


class TestWaveScenarios(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = HOSSolver(HOSConfig(n=128, depth=1.0, order=3))

    def test_linear_focusing_phases_meet_at_prescribed_crest(self) -> None:
        scenario = dispersive_focusing(
            self.solver, target_crest=0.07, focus_time=1.2
        )
        focus_x = 0.5 * self.solver.config.length
        modes = np.arange(3, 10)
        sigma = 1.65
        weights = np.exp(-0.5 * ((modes - 6) / sigma) ** 2)
        modal_amplitudes = 0.07 * weights / np.sum(weights)
        reconstructed_crest = 0.0
        for mode, amplitude in zip(modes, modal_amplitudes):
            k = 2.0 * math.pi * mode / self.solver.config.length
            reconstructed_crest += amplitude * math.cos(k * (focus_x - focus_x))
        self.assertAlmostEqual(reconstructed_crest, 0.07, places=14)
        self.assertAlmostEqual(float(np.mean(scenario.eta)), 0.0, places=14)

    def test_kdv_pair_has_zero_mean_surface_and_potential(self) -> None:
        long_solver = HOSSolver(HOSConfig(n=256, length=40.0, depth=1.0))
        scenario = head_on_solitary_pair(long_solver, amplitude=0.08)
        self.assertAlmostEqual(float(np.mean(scenario.eta)), 0.0, places=14)
        self.assertAlmostEqual(float(np.mean(scenario.psi)), 0.0, places=14)
        self.assertTrue(np.all(np.isfinite(scenario.eta)))
        self.assertTrue(np.all(np.isfinite(scenario.psi)))


class TestPeriodicVortexOperator(unittest.TestCase):
    def test_image_system_annihilates_bed_normal_velocity(self) -> None:
        length = 2.0 * math.pi
        x = np.linspace(0.0, length, 257, endpoint=False)
        z = np.full_like(x, -1.0)
        cloud = VortexCloud(
            x=np.asarray([1.1, 4.0]),
            z=np.asarray([-0.3, -0.6]),
            circulation=np.asarray([0.7, -0.2]),
            core_radius=0.015,
        )
        _, normal_velocity = velocity_above_flat_bed(x, z, cloud, length, -1.0)
        self.assertLess(float(np.max(np.abs(normal_velocity))), 2.0e-15)

    def test_velocity_is_periodic(self) -> None:
        length = 2.0 * math.pi
        cloud = vortex_dipole(2.0, -0.4, 0.25, 0.5, core_radius=0.01)
        x = np.asarray([0.37, 1.91])
        z = np.asarray([-0.2, -0.7])
        u0, w0 = velocity_above_flat_bed(x, z, cloud, length, -1.0)
        u1, w1 = velocity_above_flat_bed(x + length, z, cloud, length, -1.0)
        self.assertLess(float(np.max(np.abs(u1 - u0))), 1.0e-14)
        self.assertLess(float(np.max(np.abs(w1 - w0))), 1.0e-14)

    def test_dipole_has_zero_total_circulation(self) -> None:
        cloud = vortex_dipole(1.0, -0.4, 0.2, 0.6)
        self.assertEqual(cloud.total_circulation, 0.0)


if __name__ == "__main__":
    unittest.main()
