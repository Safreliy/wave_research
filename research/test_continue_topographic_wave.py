import unittest

import numpy as np

from continue_topographic_wave import continue_trajectory
from topographic_wave_solver import simulate


class ContinueTopographicWaveTests(unittest.TestCase):
    def test_restart_matches_uninterrupted_rk4_steps(self):
        settings = dict(
            n=24,
            length=2.0 * np.pi,
            depth=1.0,
            gravity=1.0,
            amplitude=0.08,
            crest_center=2.0,
            initial_condition="localized",
            bathymetry="cosine",
            shoal_height=0.0,
            dt=0.01,
            snapshots=3,
            time_integrator="rk4",
            project_volume=True,
            initial_reparameterize=False,
        )
        prefix = simulate(final_time=0.02, **settings)
        restarted = continue_trajectory(prefix, final_time=0.04)
        uninterrupted = simulate(final_time=0.04, snapshots=5, **{
            key: value for key, value in settings.items() if key != "snapshots"
        })
        self.assertTrue(np.allclose(restarted["x"][-1], uninterrupted["x"][-1]))
        self.assertTrue(np.allclose(restarted["z"][-1], uninterrupted["z"][-1]))
        self.assertTrue(
            np.allclose(restarted["potential"][-1], uninterrupted["potential"][-1])
        )
        self.assertEqual(str(restarted["termination_reason"]), "completed")


if __name__ == "__main__":
    unittest.main()
