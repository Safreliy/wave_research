import math
import unittest

import numpy as np

from parametric_bie import ParametricMirrorBIE, manufactured_parametric_solution


class TestParametricMirrorBIE(unittest.TestCase):
    def test_flat_symbol(self) -> None:
        n = 96
        length = 2.0 * math.pi
        alpha = np.arange(n) * length / n
        solver = ParametricMirrorBIE(alpha, np.zeros(n), length, 1.0)
        trace = np.cos(2.0 * alpha)
        result = solver.dirichlet_to_neumann(trace)
        exact = 2.0 * math.tanh(2.0) * trace
        error = np.linalg.norm(result.unit_normal_derivative - exact) / np.linalg.norm(exact)
        self.assertLess(error, 1.0e-10)

    def test_wavy_manufactured_solution(self) -> None:
        n = 128
        length = 2.0 * math.pi
        alpha = np.arange(n) * length / n
        x = alpha + 0.08 * np.sin(alpha)
        z = 0.10 * np.cos(alpha) + 0.02 * np.cos(2.0 * alpha)
        solver = ParametricMirrorBIE(x, z, length, 1.0)
        trace, exact = manufactured_parametric_solution(solver, mode=2)
        result = solver.dirichlet_to_neumann(trace)
        error = np.linalg.norm(result.unit_normal_derivative - exact) / np.linalg.norm(exact)
        self.assertLess(error, 1.0e-10)
        self.assertLess(result.boundary_residual, 1.0e-12)

    def test_singular_quadrature_outperforms_close_evaluation(self) -> None:
        n = 64
        length = 2.0 * math.pi
        alpha = np.arange(n) * length / n
        x = alpha + 0.08 * np.sin(alpha)
        z = 0.10 * np.cos(alpha)
        solver = ParametricMirrorBIE(x, z, length, 1.0)
        trace, exact = manufactured_parametric_solution(solver, mode=2)
        singular = solver.dirichlet_to_neumann(trace, method="singular")
        close = solver.dirichlet_to_neumann(trace, method="close")
        singular_error = np.linalg.norm(
            singular.unit_normal_derivative - exact
        ) / np.linalg.norm(exact)
        close_error = np.linalg.norm(close.unit_normal_derivative - exact) / np.linalg.norm(exact)
        self.assertLess(singular_error, 1.0e-10)
        self.assertLess(singular_error, 1.0e-6 * close_error)


if __name__ == "__main__":
    unittest.main()
