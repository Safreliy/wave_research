from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_impact_claim import _downsample, _grid_audit, _read_diagnostics


class ImpactClaimAuditTests(unittest.TestCase):
    def test_downsample_uses_block_liquid_fraction_not_pixel_phase(self) -> None:
        mask = np.array(
            [
                [1, 0, 1, 1],
                [1, 0, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
            ],
            dtype=bool,
        )
        expected = np.array([[True, True], [False, True]])
        np.testing.assert_array_equal(
            _downsample(mask, 2, liquid_threshold=0.5), expected
        )

    def test_downsample_threshold_band_is_monotone(self) -> None:
        mask = np.array([[1, 0], [0, 0]], dtype=bool)
        self.assertTrue(_downsample(mask, 2, liquid_threshold=0.25)[0, 0])
        self.assertFalse(_downsample(mask, 2, liquid_threshold=0.75)[0, 0])

    def test_diagnostics_parser_separates_mass_and_q_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.dat"
            path.write_text(
                "INIT_RECEIVER_CORRECTION 0 0 1e-2 1e-8 0 0 2e-2 3e-3 0 0 0 0\n"
                "RUN 0 0 10 0.5\n"
                "Q_RUN 0 0 10 0 -0 2e-15 0 0 0\n"
                "SHORE 0 0 40 0.2 0.7\n"
                "POSTADAPT_REPROJECT 0 0 3 2e-9 4e-12 8e-12 2e-4 3e-5 4e-3 7e-3\n"
                "RUN 1 4 9.995 0.4\n",
                encoding="utf-8",
            )
            result = _read_diagnostics(path)
            self.assertAlmostEqual(result["final_relative_mass_drift"], -5e-4)
            self.assertEqual(result["maximum_q_upper_bound_violation"], 2e-15)
            self.assertEqual(result["maximum_bed_pressure_magnitude"], 0.2)
            self.assertEqual(result["initial_receiver_relative_momentum_error_after"], 1e-8)
            self.assertEqual(result["post_adaptation_reprojection_count"], 1)
            self.assertEqual(
                result["maximum_post_projection_face_divergence_linf"], 8e-12
            )
            self.assertEqual(
                result["maximum_post_projection_relative_momentum_change"], 2e-4
            )

    def test_three_level_grid_gate_is_explicit(self) -> None:
        cases = [
            {
                "level": level,
                "delta": delta,
                "primary_impact_continuation_time": time,
                "pocket_area_at_fixed_lag": area,
            }
            for level, delta, time, area in (
                (8, 0.25, 4.66, 0.020),
                (9, 0.125, 4.63, 0.022),
                (10, 0.0625, 4.62, 0.023),
            )
        ]
        common = {
            "grid_time_tolerance": 0.08,
            "pocket_area_relative_tolerance": 0.25,
        }
        result = _grid_audit(cases, common)
        self.assertTrue(result["accepted"])
        self.assertAlmostEqual(result["observed_event_time_order"], 1.5849625007)


if __name__ == "__main__":
    unittest.main()
