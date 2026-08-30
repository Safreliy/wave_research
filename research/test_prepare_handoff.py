import unittest

import numpy as np
from scipy import sparse

from two_phase_basilisk.prepare_handoff import (
    centered_derivative_operators,
    correct_fraction_mass,
    potential_harmonic_extension,
    solve_least_squares,
    standard_laplacian_operator,
    streamfunction_harmonic_extension,
)


class PrepareHandoffTests(unittest.TestCase):
    def test_least_squares_wrapper_solves_rectangular_system(self):
        system = sparse.csr_matrix(
            np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        )
        expected = np.array([0.75, -0.25])
        result = solve_least_squares(system, system @ expected)
        np.testing.assert_allclose(result.solution, expected, rtol=1.0e-10, atol=1.0e-10)
        self.assertLess(result.residual_norm, 1.0e-10)

    def test_column_equilibration_preserves_rectangular_solution(self):
        system = sparse.csr_matrix(
            np.array([[1.0e-4, 0.0], [0.0, 2.0e4], [1.0e-4, 1.0e4]])
        )
        expected = np.array([0.75, -0.25])
        result = solve_least_squares(
            system,
            system @ expected,
            equilibrate_columns=True,
        )
        np.testing.assert_allclose(result.solution, expected, rtol=1.0e-8, atol=1.0e-8)
        self.assertLess(result.residual_norm, 1.0e-8)

    def test_least_squares_wrapper_rejects_unknown_backend(self):
        with self.assertRaisesRegex(ValueError, "linear solver backend"):
            solve_least_squares(
                sparse.eye(2, format="csr"),
                np.ones(2),
                backend="unknown",
            )

    def test_interface_fraction_is_corrected_to_target_mass(self):
        fraction = np.array([[0.0, 0.25, 1.0], [0.0, 0.75, 1.0]])
        corrected = correct_fraction_mass(fraction, 3.25)
        self.assertAlmostEqual(float(np.sum(corrected)), 3.25, places=12)
        self.assertTrue(np.all((corrected >= 0.0) & (corrected <= 1.0)))

    def test_mass_can_be_removed_without_touching_bounds(self):
        fraction = np.array([[0.0, 0.4, 1.0], [0.0, 0.6, 1.0]])
        corrected = correct_fraction_mass(fraction, 2.5)
        self.assertAlmostEqual(float(np.sum(corrected)), 2.5, places=12)

    def test_streamfunction_derivatives_commute(self):
        nx, nz = 12, 9
        derivative_x, derivative_z = centered_derivative_operators(
            nx, nz, 0.2, 0.15
        )
        state = np.random.default_rng(4).normal(size=nx * nz)
        commutator = derivative_x @ (derivative_z @ state) - derivative_z @ (
            derivative_x @ state
        )
        self.assertLess(float(np.max(np.abs(commutator))), 1.0e-12)

    def test_regularizing_laplacian_detects_grid_checkerboard(self):
        nx, nz = 12, 9
        laplacian = standard_laplacian_operator(nx, nz, 0.2, 0.15)
        checkerboard = np.fromfunction(
            lambda k, i: (-1.0) ** (i + k), (nz, nx)
        ).ravel()
        self.assertGreater(float(np.linalg.norm(laplacian @ checkerboard)), 1.0e3)

    def test_streamfunction_extension_is_solenoidal_and_fits_consistent_data(self):
        nx, nz = 16, 11
        grid_x = np.arange(nx) * 2.0 * np.pi / nx
        grid_z = np.linspace(-1.0, 1.0, nz)
        derivative_x, derivative_z = centered_derivative_operators(
            nx, nz, grid_x[1] - grid_x[0], grid_z[1] - grid_z[0]
        )
        xx, zz = np.meshgrid(grid_x, grid_z)
        streamfunction = (np.sin(xx) * (1.0 + 0.2 * zz)).ravel()
        velocity_x = (derivative_z @ streamfunction).reshape(nz, nx)
        velocity_z = (-derivative_x @ streamfunction).reshape(nz, nx)
        surface_k = nz // 2
        data = {
            "velocity_valid": np.ones((nz, nx), dtype=bool),
            "velocity_x": velocity_x,
            "velocity_z": velocity_z,
            "surface_x": grid_x,
            "surface_z": np.full(nx, grid_z[surface_k]),
            "surface_velocity_x": velocity_x[surface_k],
            "surface_velocity_z": velocity_z[surface_k],
        }
        reconstructed_x, reconstructed_z, metrics = streamfunction_harmonic_extension(
            data,
            grid_x,
            grid_z,
            harmonic_weight=0.0,
            surface_weight=1.0,
        )
        divergence = derivative_x @ reconstructed_x.ravel() + derivative_z @ (
            reconstructed_z.ravel()
        )
        self.assertLess(float(np.max(np.abs(divergence))), 1.0e-11)
        self.assertLess(metrics["relative_bulk_velocity_fit_error"], 1.0e-6)

    def test_potential_extension_is_irrotational_and_fits_consistent_data(self):
        nx, nz = 16, 11
        grid_x = np.arange(nx) * 2.0 * np.pi / nx
        grid_z = np.linspace(-1.0, 1.0, nz)
        derivative_x, derivative_z = centered_derivative_operators(
            nx, nz, grid_x[1] - grid_x[0], grid_z[1] - grid_z[0]
        )
        xx, zz = np.meshgrid(grid_x, grid_z)
        potential = (np.cos(xx) + 0.2 * zz).ravel()
        velocity_x = (derivative_x @ potential).reshape(nz, nx)
        velocity_z = (derivative_z @ potential).reshape(nz, nx)
        surface_k = nz // 2
        data = {
            "velocity_valid": np.ones((nz, nx), dtype=bool),
            "velocity_x": velocity_x,
            "velocity_z": velocity_z,
            "surface_x": grid_x,
            "surface_z": np.full(nx, grid_z[surface_k]),
            "surface_potential": potential.reshape(nz, nx)[surface_k],
            "surface_velocity_x": velocity_x[surface_k],
            "surface_velocity_z": velocity_z[surface_k],
        }
        reconstructed_x, reconstructed_z, metrics = potential_harmonic_extension(
            data,
            grid_x,
            grid_z,
            harmonic_weight=0.0,
            surface_weight=1.0,
            exterior_energy_weight=0.0,
        )
        curl = derivative_x @ reconstructed_z.ravel() - derivative_z @ (
            reconstructed_x.ravel()
        )
        self.assertLess(float(np.max(np.abs(curl))), 1.0e-11)
        self.assertLess(metrics["relative_bulk_velocity_fit_error"], 1.0e-6)


if __name__ == "__main__":
    unittest.main()
