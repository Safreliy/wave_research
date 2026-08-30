import unittest

import numpy as np

from refine_topographic_restart import refine_restart
from topographic_wave_solver import simulate


class RefineTopographicRestartTests(unittest.TestCase):
    def test_spectral_refinement_preserves_resolved_state_and_invariants(self):
        source = simulate(
            n=24,
            length=2.0 * np.pi,
            depth=1.0,
            gravity=1.0,
            amplitude=0.04,
            crest_center=2.0,
            initial_condition="localized",
            bathymetry="cosine",
            shoal_height=0.0,
            dt=0.01,
            final_time=0.01,
            snapshots=2,
            time_integrator="rk4",
            project_volume=True,
            initial_reparameterize=False,
        )
        refined, report = refine_restart(source, new_n=32, new_dt=0.005)

        self.assertEqual(refined["x"].shape, (1, 32))
        self.assertEqual(float(refined["dt"]), 0.005)
        self.assertLess(report["roundtrip_state_error"], 1.0e-12)
        self.assertLess(abs(report["relative_volume_error"]), 1.0e-12)
        self.assertLess(report["relative_energy_jump"], 1.0e-3)
        self.assertTrue(report["accepted"])


if __name__ == "__main__":
    unittest.main()
