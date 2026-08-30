import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from analyze_coupled_morphology import analyze, roughness, surface_profile


def write_liquid_frame(path: Path, top_rows: np.ndarray) -> None:
    height, width = 48, len(top_rows)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:] = (0, 0, 180)
    for column, top in enumerate(top_rows):
        rgb[int(top):, column] = (180, 0, 0)
    Image.fromarray(rgb).save(path)


class CoupledMorphologyAuditTests(unittest.TestCase):
    def test_surface_profile_and_roughness_are_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.ppm"
            rows = 18 + np.rint(3.0 * np.sin(np.linspace(0, 2 * np.pi, 64))).astype(int)
            write_liquid_frame(path, rows)
            profile = surface_profile(path, -1.0, 1.0)
            self.assertTrue(np.all(np.isfinite(profile)))
            self.assertGreater(float(np.ptp(profile)), 0.0)
            self.assertGreaterEqual(roughness(profile), 0.0)

    def test_crest_gate_rejects_locally_divergent_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coarse, fine = root / "coarse", root / "fine"
            coarse.mkdir()
            fine.mkdir()
            x = np.linspace(-1.0, 1.0, 80)
            initial = 24 - np.rint(8.0 * np.exp(-8.0 * x**2)).astype(int)
            coarse_final = initial.copy()
            fine_final = initial.copy()
            fine_final[38:43] -= 5
            for index, (coarse_rows, fine_rows) in enumerate(
                ((initial, initial), (coarse_final, fine_final))
            ):
                write_liquid_frame(coarse / f"vof-{index:05d}.ppm", coarse_rows)
                write_liquid_frame(fine / f"vof-{index:05d}.ppm", fine_rows)
            report, _ = analyze(
                sorted(coarse.glob("*.ppm")),
                sorted(fine.glob("*.ppm")),
                0.0,
                2.0,
                -1.0,
                1.0,
                0.02,
            )
            self.assertFalse(
                report["crest_convergence_gate_relative_difference_le_0p05"]
            )
            self.assertFalse(report["accepted_long_coupled_morphology"])


if __name__ == "__main__":
    unittest.main()
