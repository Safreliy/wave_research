import math
import tempfile
import unittest
from pathlib import Path

from audit_l11_pilot import maximum, records, resource_value


class L11PilotAuditTests(unittest.TestCase):
    def test_records_and_maximum_keep_scientific_notation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.dat"
            path.write_text(
                "POSTADAPT_REPROJECT 0.01 2 7 1e-9 2e-12 3e-11 4e-6 5e-7 6e-3 7e-2\n"
                "Q_CLEANUP 0.01 2 0 -8e-14\n",
                encoding="utf-8",
            )
            parsed = records(path)
        self.assertEqual(len(parsed["POSTADAPT_REPROJECT"]), 1)
        self.assertAlmostEqual(maximum(parsed["POSTADAPT_REPROJECT"], 5), 3e-11)
        self.assertAlmostEqual(maximum(parsed["Q_CLEANUP"], 3), 8e-14)

    def test_missing_metric_is_nan_and_resource_field_is_parsed(self) -> None:
        self.assertTrue(math.isnan(maximum([], 0)))
        text = "\tMaximum resident set size (kbytes): 4193288\n"
        self.assertEqual(
            resource_value(text, "Maximum resident set size (kbytes)"),
            "4193288",
        )


if __name__ == "__main__":
    unittest.main()
