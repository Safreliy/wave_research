import math
import unittest

import numpy as np

from solitary_wave import (
    fully_nonlinear_periodic_solitary_wave,
    full_width_above_threshold,
    periodic_solitary_wave,
)


class TestSolitaryWaveInitializer(unittest.TestCase):
    def test_periodic_flux_split_is_continuous(self) -> None:
        length = 40.0
        x = np.arange(256) * length / 256
        state = periodic_solitary_wave(
            x, length, center=15.0, amplitude=0.2, depth=1.0
        )
        endpoint = periodic_solitary_wave(
            np.asarray([0.0, length]),
            length,
            center=15.0,
            amplitude=0.2,
            depth=1.0,
        )
        self.assertAlmostEqual(
            float(endpoint.periodic_potential[0]),
            float(endpoint.periodic_potential[1]),
            places=12,
        )
        self.assertGreater(state.background_current, 0.0)
        self.assertAlmostEqual(float(np.max(state.elevation)), 0.2, places=10)
        self.assertAlmostEqual(state.speed, math.sqrt(1.2), places=14)

    def test_width_increases_as_amplitude_decreases(self) -> None:
        broad = full_width_above_threshold(0.2, 1.0)
        narrow = full_width_above_threshold(0.6, 1.0)
        self.assertGreater(broad, narrow)

    def test_babenko_initializer_hits_amplitude_and_residual(self) -> None:
        length = 20.0
        x = np.arange(128) * length / 128
        state = fully_nonlinear_periodic_solitary_wave(
            x,
            length,
            center=6.0,
            amplitude=0.2,
            depth=1.0,
            internal_points=512,
            tolerance=2.0e-10,
        )
        self.assertEqual(state.method, "babenko_petviashvili")
        self.assertLess(abs(float(np.max(state.elevation)) - 0.2), 5.0e-4)
        self.assertLess(state.equation_residual, 1.0e-8)
        self.assertGreater(state.background_current, 0.0)
        self.assertGreater(state.speed, 1.0)
        self.assertLess(state.speed, math.sqrt(1.2))


if __name__ == "__main__":
    unittest.main()
