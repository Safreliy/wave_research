from __future__ import annotations

import unittest

from render_impact_grid_comparison import frame_index


class ImpactGridComparisonTests(unittest.TestCase):
    def test_frame_index_rounds_and_clamps(self) -> None:
        self.assertEqual(frame_index(0.11, 0.02, 301), 6)
        self.assertEqual(frame_index(-1.0, 0.02, 301), 0)
        self.assertEqual(frame_index(10.0, 0.02, 301), 300)

    def test_frame_index_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            frame_index(1.0, 0.0, 10)
        with self.assertRaises(ValueError):
            frame_index(1.0, 0.1, 0)


if __name__ == "__main__":
    unittest.main()
