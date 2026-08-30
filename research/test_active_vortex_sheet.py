import math
import unittest

import numpy as np

from active_vortex_sheet import (
    ActiveVortexSheetState,
    active_vortex_sheet_rhs,
    panel_circulation_to_potential,
    potential_to_panel_circulation,
    step_active_vortex_sheet_rk4,
)
from adaptive_vortex_mesh import (
    VortexMeshMonitor,
    project_vortex_sheet_invariants,
    remesh_active_vortex_sheet,
    resample_periodic_bottom,
    suggest_vortex_count,
)
from topographic_wave_solver import smooth_periodic_shoal, step_rk4, theta_s_rhs
from topographic_wave_solver import diagnostics, geometric_volume


class ActiveVortexSheetTests(unittest.TestCase):
    def test_potential_circulation_roundtrip_with_background_period(self) -> None:
        n = 48
        length = 2.0 * math.pi
        alpha = 2.0 * math.pi * np.arange(n) / n
        x = length * np.arange(n) / n + 0.31 * np.sin(alpha)
        potential = 0.2 * np.sin(2.0 * alpha) - 0.07 * np.cos(3.0 * alpha)
        potential -= np.mean(potential)
        background = 0.17
        circulation = potential_to_panel_circulation(
            x, potential, length, background
        )
        reconstructed = panel_circulation_to_potential(
            x, circulation, length, background
        )
        self.assertAlmostEqual(float(np.sum(circulation)), background * length, places=13)
        self.assertLess(float(np.max(np.abs(reconstructed - potential))), 2.0e-14)

    def test_active_rhs_is_exact_difference_of_potential_rhs(self) -> None:
        n = 24
        length = 2.0 * math.pi
        alpha = 2.0 * math.pi * np.arange(n) / n
        x = length * np.arange(n) / n + 0.03 * np.sin(alpha)
        z = 0.04 * np.cos(alpha)
        potential = 0.05 * np.sin(alpha) + 0.02 * np.sin(2.0 * alpha)
        potential -= np.mean(potential)
        background = 0.09
        bottom_x, bottom_z = smooth_periodic_shoal(
            n, length, depth=1.0, height=0.0, center=0.0
        )
        expected = theta_s_rhs(
            x, z, potential, bottom_x, bottom_z, length, 1.0, background
        )
        state = ActiveVortexSheetState(
            x,
            z,
            potential_to_panel_circulation(x, potential, length, background),
        )
        actual = active_vortex_sheet_rhs(
            state, bottom_x, bottom_z, length, 1.0, background
        )
        expected_phi_t = expected[2] + background * expected[0]
        expected_gamma_t = np.roll(expected_phi_t, -1) - expected_phi_t
        self.assertLess(float(np.max(np.abs(actual.x_t - expected[0]))), 2.0e-13)
        self.assertLess(float(np.max(np.abs(actual.z_t - expected[1]))), 2.0e-13)
        self.assertLess(
            float(np.max(np.abs(actual.circulation_t - expected_gamma_t))), 2.0e-13
        )
        self.assertLess(abs(actual.circulation_derivative_defect), 2.0e-15)

    def test_active_rk4_matches_potential_rk4(self) -> None:
        n = 24
        length = 2.0 * math.pi
        alpha = 2.0 * math.pi * np.arange(n) / n
        x = length * np.arange(n) / n
        z = 0.03 * np.cos(alpha)
        potential = 0.04 * np.sin(alpha)
        background = 0.07
        bottom_x, bottom_z = smooth_periodic_shoal(
            n, length, depth=1.0, height=0.0, center=0.0
        )
        reference = step_rk4(
            x,
            z,
            potential,
            bottom_x,
            bottom_z,
            length,
            1.0,
            2.0e-4,
            background,
        )
        state = ActiveVortexSheetState(
            x,
            z,
            potential_to_panel_circulation(x, potential, length, background),
        )
        advanced, diagnostics = step_active_vortex_sheet_rk4(
            state,
            bottom_x,
            bottom_z,
            length,
            1.0,
            2.0e-4,
            background,
        )
        active_potential = panel_circulation_to_potential(
            advanced.x,
            advanced.panel_circulation,
            length,
            background,
        )
        self.assertLess(float(np.max(np.abs(advanced.x - reference[0]))), 2.0e-12)
        self.assertLess(float(np.max(np.abs(advanced.z - reference[1]))), 2.0e-12)
        self.assertLess(float(np.max(np.abs(active_potential - reference[2]))), 2.0e-12)
        self.assertLess(abs(advanced.total_circulation - state.total_circulation), 2.0e-14)
        self.assertLess(diagnostics.maximum_reconstruction_relative_error, 2.0e-13)

    def test_adaptive_remesh_changes_count_and_preserves_circulation(self) -> None:
        n = 64
        length = 2.0 * math.pi
        alpha = 2.0 * math.pi * np.arange(n) / n
        x = length * np.arange(n) / n + 0.12 * np.sin(alpha)
        z = 0.18 * np.exp(-((alpha - math.pi) / 0.38) ** 2)
        potential = 0.08 * np.sin(alpha) + 0.03 * np.sin(3.0 * alpha)
        background = 0.11
        state = ActiveVortexSheetState(
            x,
            z,
            potential_to_panel_circulation(x, potential, length, background),
        )
        remeshed, diagnostics = remesh_active_vortex_sheet(
            state,
            length,
            background,
            target_count=96,
            proximity_weight=0.0,
        )
        self.assertEqual(len(remeshed.x), 96)
        self.assertLess(abs(diagnostics.total_circulation_defect), 2.0e-14)
        self.assertLess(diagnostics.potential_roundtrip_relative_error, 2.0e-13)
        self.assertLess(
            diagnostics.monitor_mass_cv_after,
            diagnostics.monitor_mass_cv_before,
        )
        self.assertGreater(diagnostics.minimum_panel_length, 0.0)

    def test_count_controller_has_refinement_hysteresis(self) -> None:
        n = 64
        arrays = np.ones(n)
        monitor = VortexMeshMonitor(
            arrays,
            arrays,
            arrays,
            2.0 * arrays,
            0.1 * arrays,
            maximum_turning_angle=0.2,
            minimum_nonlocal_gap_ratio=20.0,
            maximum_circulation_fraction=0.01,
        )
        decision = suggest_vortex_count(monitor, 64, 48, 192)
        self.assertEqual(decision.action, "refine")
        self.assertEqual(decision.target_count, 96)

    def test_bottom_resampling_preserves_periodic_map(self) -> None:
        n = 48
        length = 2.0 * math.pi
        alpha = 2.0 * math.pi * np.arange(n) / n
        bottom_x = length * np.arange(n) / n + 0.07 * np.sin(alpha)
        bottom_z = -1.0 + 0.1 * np.cos(2.0 * alpha)
        target_x, target_z = resample_periodic_bottom(
            bottom_x, bottom_z, length, 80
        )
        target_alpha = 2.0 * math.pi * np.arange(80) / 80
        expected_x = length * np.arange(80) / 80 + 0.07 * np.sin(target_alpha)
        expected_z = -1.0 + 0.1 * np.cos(2.0 * target_alpha)
        self.assertLess(float(np.max(np.abs(target_x - expected_x))), 2.0e-13)
        self.assertLess(float(np.max(np.abs(target_z - expected_z))), 2.0e-13)

    def test_energy_volume_projection_preserves_circulation(self) -> None:
        n = 24
        length = 2.0 * math.pi
        alpha = 2.0 * math.pi * np.arange(n) / n
        x = length * np.arange(n) / n
        z = 0.04 * np.cos(alpha)
        potential = 0.05 * np.sin(alpha)
        bottom_x, bottom_z = smooth_periodic_shoal(
            n, length, depth=1.0, height=0.0, center=0.0
        )
        circulation = potential_to_panel_circulation(x, potential, length)
        state = ActiveVortexSheetState(x, z, circulation)
        target_volume = geometric_volume(x, z, bottom_x, bottom_z, length)
        target_energy = diagnostics(
            x, z, potential, bottom_x, bottom_z, length, 1.0
        )[1]
        perturbed = ActiveVortexSheetState(
            x,
            z + 2.0e-4,
            1.01 * circulation,
        )
        projected, projection = project_vortex_sheet_invariants(
            perturbed,
            bottom_x,
            bottom_z,
            length,
            gravity=1.0,
            background_current=0.0,
            target_volume=target_volume,
            target_energy=target_energy,
            energy_scale=max(abs(target_energy), 1.0),
        )
        self.assertLess(abs(projected.total_circulation - state.total_circulation), 2.0e-14)
        self.assertLess(projection.relative_energy_error, 2.0e-13)
        self.assertLess(projection.relative_volume_error, 2.0e-13)
        self.assertAlmostEqual(projection.volume_shift, -2.0e-4, places=12)


if __name__ == "__main__":
    unittest.main()
