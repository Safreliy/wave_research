import unittest

import numpy as np

from jfm2026_benchmark import infer_deep_depth
from jfm2026_data_audit import infer_bed_geometry, is_overturned, surface_full_width


class JFM2026AuditTests(unittest.TestCase):
    def setUp(self):
        # Closed tank: left wall, h0=1 flat bed, 1:2 upslope, hs=0.5 shelf,
        # right wall.  Repeated corner points mirror the archived representation.
        self.boundary = np.array(
            [
                [0.0, 0.0], [0.0, -0.5], [0.0, -1.0],
                [2.0, -1.0], [3.0, -0.5], [5.0, -0.5],
                [5.0, 0.0],
            ]
        )

    def test_depth_and_bed_geometry_are_inferred(self):
        self.assertAlmostEqual(infer_deep_depth(self.boundary), 1.0)
        geometry = infer_bed_geometry(self.boundary)
        self.assertAlmostEqual(geometry["shallow_depth_over_h0"], 0.5)
        self.assertAlmostEqual(geometry["toe_over_h0"], 2.0)
        self.assertAlmostEqual(geometry["shelf_start_over_h0"], 3.0)
        self.assertAlmostEqual(geometry["effective_slope"], 0.5)

    def test_surface_metrics_are_parameterization_aware(self):
        graph = np.array([[3.0, 0.0], [2.0, 0.2], [1.0, 0.0], [0.0, 0.0]])
        overhang = np.array([[3.0, 0.0], [2.0, 0.2], [2.2, 0.1], [1.0, 0.0]])
        self.assertFalse(is_overturned(graph))
        self.assertTrue(is_overturned(overhang))
        self.assertAlmostEqual(surface_full_width(graph, 1.0, 0.1), 0.0)
        self.assertAlmostEqual(surface_full_width(graph, 1.0, 0.05), 0.0)
        self.assertAlmostEqual(surface_full_width(graph, 1.0, 0.0), 3.0)


if __name__ == "__main__":
    unittest.main()
