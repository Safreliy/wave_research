import unittest

import numpy as np

from impact_detection import periodic_impact_diagnostics


class ImpactDetectionTests(unittest.TestCase):
    def test_flat_periodic_surface_has_no_contact(self) -> None:
        x = np.arange(16) / 16.0
        result = periodic_impact_diagnostics(x, np.zeros_like(x), 1.0)
        self.assertFalse(result.self_intersection)
        self.assertGreater(result.normalized_distance, 1.9)

    def test_crossing_nonlocal_panels_is_detected(self) -> None:
        x = np.asarray([0.0, 0.3, 0.7, 0.3, 0.7, 0.9])
        z = np.asarray([0.0, 0.4, 0.0, 0.0, 0.4, 0.0])
        result = periodic_impact_diagnostics(x, z, 1.0, adjacent_exclusion=1)
        self.assertTrue(result.self_intersection)
        self.assertEqual(result.minimum_distance, 0.0)

    def test_near_contact_reports_submarker_gap(self) -> None:
        x = np.asarray([0.0, 0.2, 0.5, 0.7, 0.5, 0.2, 0.05, 0.8])
        z = np.asarray([0.5, 0.0, 0.0, 0.1, 0.02, 0.02, 0.4, 0.5])
        result = periodic_impact_diagnostics(x, z, 1.0, adjacent_exclusion=1)
        self.assertFalse(result.self_intersection)
        self.assertLess(result.normalized_distance, 0.1)

    def test_arc_exclusion_rejects_local_numerical_fold(self) -> None:
        n = 64
        x = np.arange(n) / n
        z = np.zeros(n)
        z[30:35] = np.asarray([0.0, 0.02, -0.01, 0.02, 0.0])
        local = periodic_impact_diagnostics(x, z, 1.0, adjacent_exclusion=2)
        topological = periodic_impact_diagnostics(
            x,
            z,
            1.0,
            adjacent_exclusion=2,
            minimum_arc_separation_panels=12.0,
        )
        self.assertLess(local.normalized_arc_separation, 12.0)
        self.assertGreaterEqual(topological.normalized_arc_separation, 12.0)
        self.assertGreater(topological.normalized_distance, local.normalized_distance)


if __name__ == "__main__":
    unittest.main()
