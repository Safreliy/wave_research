import json
from pathlib import Path
import tempfile
import unittest

from two_phase_basilisk.analyze_handoff_transfer import analyze


class HandoffTransferAuditTests(unittest.TestCase):
    def test_conservation_pass_and_extension_failure_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            handoff = root / "handoff.json"
            prepared = root / "prepared.json"
            diagnostics = root / "diagnostics.dat"
            handoff.write_text(
                json.dumps(
                    {
                        "time": 1.0,
                        "snapshot_index": 3,
                        "source_volume": 2.0,
                        "target_momentum_x": 0.3,
                        "target_momentum_z": 0.4,
                    }
                ),
                encoding="utf-8",
            )
            prepared.write_text(
                json.dumps({"relative_weighted_velocity_smoothing_change": 0.2}),
                encoding="utf-8",
            )
            diagnostics.write_text(
                "TRANSFER 0 0 2.0000001 0.3 0.4 2 3 2 3 100\n"
                "RUN 0 0 2.0000001 0.1\n"
                "TRANSFER 0.01 1 2.0 0.3 0.4 1 2 0.0002 0.0008 90\n"
                "RUN 0.1 10 1.99 0.09\n",
                encoding="utf-8",
            )
            result = analyze(handoff, prepared, diagnostics)
            self.assertTrue(result["conservative_initialization_passed"])
            self.assertFalse(result["coupled_continuation_accepted"])
            self.assertAlmostEqual(result["relative_mass_error"], 5.0e-8)


if __name__ == "__main__":
    unittest.main()
