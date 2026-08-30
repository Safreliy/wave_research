import math
import unittest

import numpy as np

from topography_bie import TopographyBIE, conformal_manufactured_geometry
from vof_handoff import (
    _interior_potential,
    bulk_potential,
    bulk_velocity,
    choose_snapshot,
    interface_self_intersects,
    rasterize_volume_fraction,
)


class TestVOFHandoff(unittest.TestCase):
    def test_last_pre_event_selects_steep_conservation_admissible_state(self) -> None:
        n = 64
        length = 2.0 * math.pi
        x = np.arange(n) * length / n
        z = 1.1 * np.sin(x)
        data = {
            "x": x[None, :],
            "z": z[None, :],
            "bottom_x": x,
            "bottom_z": -2.0 * np.ones(n),
            "potential": np.zeros((1, n)),
            "length": np.asarray(length),
            "gravity": np.asarray(1.0),
            "background_current": np.asarray(0.0),
            "time": np.asarray([2.0]),
            "energy": np.asarray([0.0]),
            "energy_reference": np.asarray(-1.0),
            "volume": np.asarray([2.0 * length]),
            "min_x_alpha": np.asarray([1.0]),
            "marker_cv": np.asarray([0.0]),
            "bie_residual": np.asarray([1.0e-13]),
            "surface_flux_defect": np.asarray([1.0e-6]),
            "spectral_filter_relative_corrections": np.asarray([0.0]),
        }
        index, checks = choose_snapshot(
            data,
            "last_pre_event",
            max_energy_drift=5.0e-3,
            max_volume_drift=1.0e-3,
        )
        self.assertEqual(index, 0)
        self.assertGreater(checks["maximum_surface_slope"], 1.0)
        self.assertEqual(checks["controller_action"], "handoff_two_phase")

    def test_variable_order_active_sheet_archive_is_unpadded_and_gated(self) -> None:
        n = 48
        padded = 64
        length = 2.0 * math.pi
        alpha = np.arange(n) * length / n
        surface_x = alpha + 1.1 * np.sin(alpha)
        surface_z = 0.35 * np.sin(alpha)

        def padded_row(values: np.ndarray) -> np.ndarray:
            row = np.full((1, padded), np.nan)
            row[0, :n] = values
            return row

        data = {
            "x": padded_row(surface_x),
            "z": padded_row(surface_z),
            "bottom_x": padded_row(alpha),
            "bottom_z": padded_row(-2.0 * np.ones(n)),
            "potential": padded_row(np.zeros(n)),
            "panel_circulation": padded_row(np.zeros(n)),
            "count": np.asarray([n]),
            "length": np.asarray(length),
            "gravity": np.asarray(1.0),
            "background_current": np.asarray(0.0),
            "time": np.asarray([2.0]),
            "energy": np.asarray([0.0]),
            "energy_reference": np.asarray(-1.0),
            "global_initial_energy": np.asarray(0.0),
            "target_volume": np.asarray(2.0 * length),
            "volume": np.asarray([2.0 * length]),
            "min_x_alpha": np.asarray([-0.1]),
            "monitor_mass_cv": np.asarray([0.2]),
            "maximum_bie_residual": np.asarray([1.0e-13]),
            "surface_flux_defect": np.asarray([1.0e-5]),
            "cumulative_projection_correction": np.asarray([5.0e-3]),
        }
        index, checks = choose_snapshot(
            data,
            "first_overhang",
            max_energy_drift=5.0e-3,
            max_volume_drift=1.0e-3,
            max_surface_flux_defect=5.0e-3,
            max_marker_spacing_cv=0.9,
        )
        self.assertEqual(index, 0)
        self.assertEqual(checks["marker_quality_field"], "monitor_mass_cv")
        self.assertAlmostEqual(checks["cumulative_numerical_correction"], 5.0e-3)
        self.assertEqual(checks["controller_action"], "handoff_two_phase")

    def test_interior_representation_on_conformal_solution(self) -> None:
        n = 64
        depth = 1.0
        amplitude = 0.08
        mode = 2
        geometry = conformal_manufactured_geometry(
            n=n, depth=depth, amplitude=amplitude, mode=mode
        )
        bie = TopographyBIE(*geometry[:4], length=2.0 * math.pi)
        solution = bie.solve(geometry[4])
        xi = np.linspace(0.2, 5.8, 19)
        eta = -0.43 * np.ones_like(xi)
        target_x = xi + amplitude * np.sin(xi) * np.sinh(eta)
        target_z = eta + amplitude * np.cos(xi) * np.cosh(eta)
        computed = _interior_potential(
            bie, geometry[4], solution, target_x, target_z
        )
        exact = (
            np.cos(mode * xi)
            * np.cosh(mode * (eta + depth))
            / np.cosh(mode * depth)
        )
        relative_error = np.linalg.norm(computed - exact) / np.linalg.norm(exact)
        self.assertLess(relative_error, 2.0e-9)

    def test_rasterization_and_intersection_guard(self) -> None:
        n = 48
        x = np.arange(n) * 2.0 * math.pi / n
        top = np.zeros(n)
        bottom = -np.ones(n)
        _, grid_z, fraction = rasterize_volume_fraction(
            x, top, x, bottom, 2.0 * math.pi, nx=96, nz=48
        )
        middle = int(np.argmin(np.abs(grid_z + 0.5)))
        self.assertGreater(np.mean(fraction[middle, 5:-5]), 0.99)
        self.assertFalse(interface_self_intersects(x, top))
        crossing_x = np.array([0.0, 1.0, 0.0, 1.0])
        crossing_z = np.array([0.0, 1.0, 1.0, 0.0])
        self.assertTrue(interface_self_intersects(crossing_x, crossing_z))

    def test_rasterization_closes_a_shifted_periodic_seam(self) -> None:
        n = 48
        length = 2.0 * math.pi
        base = np.arange(n) * length / n
        surface_x = base - 0.07
        grid_x, grid_z, fraction = rasterize_volume_fraction(
            surface_x,
            np.zeros(n),
            base,
            -np.ones(n),
            length,
            nx=256,
            nz=128,
            supersample=4,
        )
        area = float(
            np.sum(fraction)
            * (grid_x[1] - grid_x[0])
            * (grid_z[1] - grid_z[0])
        )
        self.assertLess(abs(area - length) / length, 1.0e-2)

    def test_bulk_export_includes_uniform_background_current(self) -> None:
        n = 24
        length = 2.0 * math.pi
        x = np.arange(n) * length / n
        bie = TopographyBIE(x, np.zeros(n), x, -np.ones(n), length)
        potential = np.zeros(n)
        solution = bie.solve(potential)
        grid_x = (np.arange(32) + 0.5) * length / 32
        grid_z = np.linspace(-0.8, -0.2, 16)
        fraction = np.ones((len(grid_z), len(grid_x)))
        velocity_x, velocity_z, valid = bulk_velocity(
            bie,
            potential,
            solution,
            grid_x,
            grid_z,
            fraction,
            background_current=0.17,
        )
        self.assertGreater(int(np.sum(valid)), 0)
        self.assertLess(float(np.nanmax(np.abs(velocity_x[valid] - 0.17))), 1.0e-12)
        self.assertLess(float(np.nanmax(np.abs(velocity_z[valid]))), 1.0e-12)
        scalar = bulk_potential(
            bie,
            potential,
            solution,
            grid_x,
            grid_z,
            valid,
            background_current=0.17,
        )
        xx, _ = np.meshgrid(grid_x, grid_z)
        recovered = scalar[valid] - np.mean(scalar[valid] - 0.17 * xx[valid])
        self.assertLess(
            float(np.max(np.abs(recovered - 0.17 * xx[valid]))),
            1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
