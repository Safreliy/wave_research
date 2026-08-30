import math
import unittest

import numpy as np

from boundary_vortex_sheet import (
    closed_boundary_vortex_sheet,
    surface_vortex_points,
)


class BoundaryVortexSheetTests(unittest.TestCase):
    def test_surface_circulation_telescopes_to_background_period(self) -> None:
        n = 32
        length = 2.0 * math.pi
        x = np.arange(n) * length / n
        potential = 0.2 * np.sin(x)
        _, _, circulation = surface_vortex_points(
            x, np.zeros(n), potential, length, background_current=0.17
        )
        self.assertAlmostEqual(float(np.sum(circulation)), 0.17 * length, places=13)

    def test_closed_rest_boundary_has_zero_circulation_defect(self) -> None:
        n = 24
        length = 2.0 * math.pi
        x = np.arange(n) * length / n
        sheet = closed_boundary_vortex_sheet(
            x,
            np.zeros(n),
            np.zeros(n),
            x,
            -np.ones(n),
            length,
            background_current=0.2,
        )
        self.assertLess(abs(sheet.closed_circulation_defect), 1.0e-13)
        self.assertAlmostEqual(
            float(np.sum(sheet.surface_circulation)), 0.2 * length, places=13
        )


if __name__ == "__main__":
    unittest.main()
