import math
import unittest

import numpy as np

from parametric_bie import ParametricMirrorBIE
from topography_bie import TopographyBIE, conformal_manufactured_geometry


class TestTopographyBIE(unittest.TestCase):
    def test_conformal_wavy_bottom_manufactured_solution(self) -> None:
        n = 64
        geometry = conformal_manufactured_geometry(
            n=n, depth=1.0, amplitude=0.08, mode=2
        )
        solver = TopographyBIE(*geometry[:4], length=2.0 * math.pi)
        result = solver.solve(geometry[4])
        error = np.linalg.norm(
            result.surface_normal_derivative - geometry[5]
        ) / np.linalg.norm(geometry[5])
        self.assertLess(error, 2.0e-10)
        self.assertLess(result.residual, 1.0e-12)

    def test_flat_bottom_agrees_with_mirror_solver(self) -> None:
        n = 64
        length = 2.0 * math.pi
        alpha = np.arange(n) * length / n
        surface_z = 0.06 * np.cos(alpha) + 0.015 * np.cos(2.0 * alpha)
        trace = 0.03 * np.cos(2.0 * alpha) + 0.02 * np.sin(3.0 * alpha)
        topography = TopographyBIE(
            alpha,
            surface_z,
            alpha,
            -np.ones(n),
            length,
        )
        mirror = ParametricMirrorBIE(alpha, surface_z, length, 1.0)
        q_topography = topography.solve(trace).surface_normal_derivative
        q_mirror = mirror.dirichlet_to_neumann(trace).unit_normal_derivative
        difference = np.linalg.norm(q_topography - q_mirror) / np.linalg.norm(q_mirror)
        self.assertLess(difference, 2.0e-10)

    def test_prescribed_bottom_flux_recovers_vertical_linear_potential(self) -> None:
        n = 64
        length = 2.0 * math.pi
        x = np.arange(n) * length / n
        solver = TopographyBIE(x, np.zeros(n), x, -np.ones(n), length)
        result = solver.solve(
            np.zeros(n), bottom_normal_derivative=-np.ones(n)
        )
        self.assertLess(
            np.linalg.norm(result.surface_normal_derivative - 1.0), 2.0e-10
        )
        self.assertLess(np.linalg.norm(result.bottom_potential + 1.0), 2.0e-10)

    def test_neumann_to_dirichlet_inverts_forward_map_up_to_constant(self) -> None:
        n = 48
        length = 2.0 * math.pi
        x = np.arange(n) * length / n
        surface_z = 0.04 * np.cos(x)
        bottom_z = -1.0 + 0.05 * np.cos(2.0 * x)
        trace = 0.07 * np.cos(2.0 * x) - 0.03 * np.sin(3.0 * x)
        solver = TopographyBIE(x, surface_z, x, bottom_z, length)
        forward = solver.solve(trace)
        inverse = solver.solve_neumann_to_dirichlet(
            forward.surface_normal_derivative, np.zeros(n)
        )
        recovered = inverse.surface_potential
        recovered -= np.mean(recovered)
        expected = trace - np.mean(trace)
        relative = np.linalg.norm(recovered - expected) / np.linalg.norm(expected)
        self.assertLess(relative, 2.0e-10)
        self.assertLess(inverse.residual, 2.0e-10)

    def test_oversampled_self_rule_resolves_a_close_overhanging_fold(self) -> None:
        """A harmonic field remains a manufactured solution on a folded top."""
        n = 64
        length = 12.0
        x_base = length * np.arange(n) / n
        distance = (x_base - 6.0 + 0.5 * length) % length - 0.5 * length
        bump = np.exp(-0.5 * (distance / 0.5) ** 2)
        surface_x = x_base + 2.0 * bump
        surface_z = 0.1 * bump
        bottom_z = -2.0 * np.ones(n)
        wavenumber = 4.0 * math.pi / length
        trace = (
            np.sin(wavenumber * surface_x)
            * np.cosh(wavenumber * (surface_z + 2.0))
            / np.cosh(2.0 * wavenumber)
        )

        plain = TopographyBIE(
            surface_x,
            surface_z,
            x_base,
            bottom_z,
            length,
            cross_oversampling=1,
            surface_self_oversampling=1,
        )
        resolved = TopographyBIE(
            surface_x,
            surface_z,
            x_base,
            bottom_z,
            length,
            cross_oversampling=1,
            surface_self_oversampling=16,
        )
        gradient_x = (
            wavenumber
            * np.cos(wavenumber * surface_x)
            * np.cosh(wavenumber * (surface_z + 2.0))
            / np.cosh(2.0 * wavenumber)
        )
        gradient_z = (
            wavenumber
            * np.sin(wavenumber * surface_x)
            * np.sinh(wavenumber * (surface_z + 2.0))
            / np.cosh(2.0 * wavenumber)
        )
        exact = (
            -resolved.surface.z_alpha * gradient_x
            + resolved.surface.x_alpha * gradient_z
        ) / resolved.surface.metric
        plain_error = np.linalg.norm(
            plain.solve(trace).surface_normal_derivative - exact
        ) / np.linalg.norm(exact)
        resolved_error = np.linalg.norm(
            resolved.solve(trace).surface_normal_derivative - exact
        ) / np.linalg.norm(exact)

        self.assertLess(resolved.minimum_nonlocal_surface_separation, resolved.mean_surface_spacing)
        self.assertLess(resolved_error, 0.1)
        self.assertLess(resolved_error, plain_error / 100.0)


if __name__ == "__main__":
    unittest.main()
