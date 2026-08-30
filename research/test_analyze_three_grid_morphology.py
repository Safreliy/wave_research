from __future__ import annotations

import unittest

import numpy as np

from analyze_three_grid_morphology import richardson_order


class ThreeGridMorphologyTests(unittest.TestCase):
    def test_richardson_order_for_factor_two_error_reduction(self) -> None:
        result = richardson_order(np.array([0.4, 0.2]), np.array([0.2, 0.05]))
        np.testing.assert_allclose(result, np.array([1.0, 2.0]))

    def test_richardson_order_marks_zero_error_undefined(self) -> None:
        result = richardson_order(np.array([0.0]), np.array([0.1]))
        self.assertTrue(np.isnan(result[0]))

    def test_richardson_order_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            richardson_order(np.ones(2), np.ones(3))


if __name__ == "__main__":
    unittest.main()
