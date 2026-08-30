import math
import unittest

import numpy as np

from hos_solver import HOSConfig, HOSSolver
from periodic_bie import PeriodicMirrorBIE, manufactured_harmonic_solution


class TestPeriodicMirrorBIE(unittest.TestCase):
    def test_constant_trace_has_zero_dno(self) -> None:
        solver = PeriodicMirrorBIE(np.zeros(64), depth=1.0)
        result = solver.dirichlet_to_neumann(np.ones(64))
        self.assertLess(result.boundary_residual, 1.0e-12)
        self.assertLess(float(np.max(np.abs(result.dno))), 2.0e-5)

    def test_flat_surface_symbol(self) -> None:
        n = 128
        mode = 2
        solver = PeriodicMirrorBIE(np.zeros(n), depth=1.0)
        trace = np.cos(mode * solver.x)
        result = solver.dirichlet_to_neumann(trace)
        exact = mode * math.tanh(mode) * trace
        relative_error = np.linalg.norm(result.dno - exact) / np.linalg.norm(exact)
        self.assertLess(relative_error, 2.5e-2)

    def test_wavy_manufactured_solution_converges(self) -> None:
        errors = []
        for n in (64, 128):
            x = np.arange(n) * 2.0 * math.pi / n
            eta = 0.08 * np.cos(x) + 0.025 * np.cos(2.0 * x + 0.3)
            solver = PeriodicMirrorBIE(eta, depth=1.0)
            trace, exact = manufactured_harmonic_solution(solver, mode=2)
            result = solver.dirichlet_to_neumann(trace)
            errors.append(np.linalg.norm(result.dno - exact) / np.linalg.norm(exact))
            self.assertLess(result.boundary_residual, 1.0e-12)
        self.assertLess(errors[1], 0.45 * errors[0])

    def test_small_amplitude_dno_agrees_with_hos_baseline(self) -> None:
        n = 128
        hos = HOSSolver(HOSConfig(n=n, depth=1.0, order=5))
        eta = 0.01 * np.cos(hos.x)
        trace = 0.02 * np.cos(2.0 * hos.x) + 0.01 * np.sin(3.0 * hos.x)
        bie = PeriodicMirrorBIE(eta, depth=1.0)
        bie_dno = bie.dirichlet_to_neumann(trace).dno
        hos_dno = hos.dirichlet_to_neumann(eta, trace)
        relative_difference = np.linalg.norm(bie_dno - hos_dno) / np.linalg.norm(
            hos_dno
        )
        self.assertLess(relative_difference, 2.6e-2)


if __name__ == "__main__":
    unittest.main()
