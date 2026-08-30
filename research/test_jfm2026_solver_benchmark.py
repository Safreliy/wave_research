import unittest

from jfm2026_solver_benchmark import JFMFinalCase


class JFMFinalCaseTests(unittest.TestCase):
    def test_article_geometry_is_reconstructed(self):
        case = JFMFinalCase()
        self.assertAlmostEqual(case.deep_length, case.full_width + 5.0)
        self.assertAlmostEqual(case.crest_center, 0.5 * case.full_width + 1.0)
        self.assertAlmostEqual(case.slope_length, 28.5)
        self.assertLess(
            case.deep_length + case.slope_length + case.shelf_length + case.return_length,
            case.length,
        )


if __name__ == "__main__":
    unittest.main()
