import math
import unittest

import numpy as np

from hos_solver import HOSConfig, HOSSolver, initial_condition, simulate


class TestHOSSolver(unittest.TestCase):
    def setUp(self) -> None:
        self.config = HOSConfig(n=64, order=3, depth=1.0)
        self.solver = HOSSolver(self.config)

    def test_flat_surface_dno_symbol(self) -> None:
        mode = 3
        psi = np.cos(mode * self.solver.x)
        computed = self.solver.dirichlet_to_neumann(np.zeros_like(psi), psi)
        expected = mode * math.tanh(mode * self.config.depth) * psi
        self.assertLess(np.max(np.abs(computed - expected)), 1.0e-11)

    def test_rest_state_is_exact(self) -> None:
        zero = np.zeros(self.config.n)
        eta, psi = self.solver.step_rk4(zero, zero, 0.01)
        self.assertTrue(np.array_equal(eta, zero))
        self.assertTrue(np.array_equal(psi, zero))

    def test_linear_dispersion_after_one_period(self) -> None:
        amplitude = 1.0e-5
        eta0, psi0, period = initial_condition("linear_mode", self.solver, amplitude)
        result = simulate(self.solver, eta0, psi0, period / 800.0, period, snapshots=3)
        relative_error = np.linalg.norm(result["eta"][-1] - eta0) / np.linalg.norm(eta0)
        self.assertLess(relative_error, 2.0e-5)

    def test_small_gaussian_conserves_volume_and_energy(self) -> None:
        eta0, psi0, _ = initial_condition("gaussian", self.solver, 0.01)
        result = simulate(self.solver, eta0, psi0, 0.002, 0.5, snapshots=6)
        volume_drift = np.max(np.abs(result["volume"] - result["volume"][0]))
        energy_drift = np.max(
            np.abs((result["energy"] - result["energy"][0]) / result["energy"][0])
        )
        self.assertLess(volume_drift, 1.0e-12)
        self.assertLess(energy_drift, 2.0e-5)


if __name__ == "__main__":
    unittest.main()
