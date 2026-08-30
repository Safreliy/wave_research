from pathlib import Path
import tempfile
import unittest

from gpu.analyze_final_q_stage import final_record, maximum_mass_drift, records


class FinalQStageParserTests(unittest.TestCase):
    def test_q_log_parsing_uses_reported_physical_mass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "diagnostics.dat"
            path.write_text(
                "Q_MANUFACTURED 0 0 0.075 0 0 0 0 0 0\n"
                "Q_MANUFACTURED 0.1 2 0.0750000000000001 0 0 0 0 0 0\n"
                "Q_FINAL 7 1 0.0750000000000001 0.075 1.333e-15 0.008\n",
                encoding="utf-8",
            )
            self.assertEqual(len(records(path, "Q_MANUFACTURED")), 2)
            self.assertEqual(final_record(path)["level"], 7)
            self.assertLess(maximum_mass_drift(path), 2.0e-15)

    def test_embedded_measure_does_not_apply_cs_twice(self) -> None:
        root = Path(__file__).resolve().parent / "two_phase_basilisk"
        receiver = (root / "bie_handoff_receiver.c").read_text(encoding="utf-8")
        q_operator = (root / "q_embed_vof.h").read_text(encoding="utf-8")
        self.assertNotIn("f[]*cs[]*dv()", receiver)
        self.assertNotIn("q_liquid[]*dv()", receiver)
        self.assertNotIn("old_q*dv()", q_operator)
        self.assertIn("remap_raw_mass += raw_q*sq(Delta)", receiver)

    def test_q_operator_has_pure_state_flux_and_interface_band_guard(self) -> None:
        root = Path(__file__).resolve().parent / "two_phase_basilisk"
        q_operator = (root / "q_embed_vof.h").read_text(encoding="utf-8")
        self.assertIn("(scalar fraction, face vector open_flux", q_operator)
        self.assertIn("active_cut_cells", q_operator)
        self.assertIn("f[donor] >= 1. - Q_EMBED_GEOMETRIC_TOLERANCE", q_operator)
        self.assertIn(
            "transfer.donor.open_area = transfer.donor.liquid_area",
            q_operator,
        )
        self.assertIn("# define Q_EMBED_CONSERVATIVE_REDISTRIBUTION 0", q_operator)

    def test_receiver_refinement_family_is_level_specific_and_cadence_explicit(self) -> None:
        root = Path(__file__).resolve().parent / "two_phase_basilisk"
        receiver = (root / "bie_handoff_receiver.c").read_text(encoding="utf-8")
        self.assertIn("X1024_L8", receiver)
        self.assertIn("X1024_L10", receiver)
        self.assertIn("# define RECEIVER_ADAPT_INTERVAL 1", receiver)
        self.assertIn(
            "(i - INITIAL_ADAPT_ITERATION) % RECEIVER_ADAPT_INTERVAL != 0",
            receiver,
        )


if __name__ == "__main__":
    unittest.main()
