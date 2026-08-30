import math
import unittest

import numpy as np

from implicit_midpoint import implicit_midpoint_step


class TestImplicitMidpoint(unittest.TestCase):
    def test_stiff_decay_is_stable_beyond_explicit_rk4_region(self) -> None:
        rate = 100.0
        dt = 0.1
        computed, diagnostics = implicit_midpoint_step(
            np.asarray([1.0]), lambda value: -rate * value, dt
        )
        exact_step = (1.0 - 0.5 * rate * dt) / (1.0 + 0.5 * rate * dt)
        self.assertTrue(diagnostics.converged)
        self.assertAlmostEqual(float(computed[0]), exact_step, places=9)
        # Classical explicit RK4 has |R(-10)| >> 1 for the same step.
        rk4_amplification = 1.0 - 10.0 + 50.0 - 1000.0 / 6.0 + 10000.0 / 24.0
        self.assertGreater(abs(rk4_amplification), 1.0)

    def test_midpoint_has_second_order_on_rotation(self) -> None:
        def rhs(value: np.ndarray) -> np.ndarray:
            return np.asarray([-value[1], value[0]])

        state = np.asarray([1.0, 0.0])
        dt = 0.05
        for _ in range(20):
            state, diagnostics = implicit_midpoint_step(state, rhs, dt)
            self.assertTrue(diagnostics.converged)
        error = np.linalg.norm(state - np.asarray([math.cos(1.0), math.sin(1.0)]))
        self.assertLess(error, 3.0e-4)
        self.assertLess(abs(float(np.dot(state, state)) - 1.0), 1.0e-10)


if __name__ == "__main__":
    unittest.main()
