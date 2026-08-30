import unittest

from selective_physics_controller import Action, DiagnosticState, decide


def state(**changes) -> DiagnosticState:
    values = {
        "relative_energy_drift": 1.0e-7,
        "wave_volume_normalized_drift": 2.0e-7,
        "surface_flux_defect": 1.0e-8,
        "bie_residual": 1.0e-13,
        "marker_spacing_cv": 1.0e-3,
        "minimum_x_alpha": 0.8,
        "mean_x_alpha": 1.0,
        "minimum_boundary_separation": 1.0,
        "boundary_spacing": 0.2,
        "breaking_precursor_b": 0.5,
        "self_intersection": False,
    }
    values.update(changes)
    return DiagnosticState(**values)


class TestSelectivePhysicsController(unittest.TestCase):
    def test_physical_precursor_activates_patch_only_when_gates_pass(self) -> None:
        decision = decide(state(breaking_precursor_b=0.88))
        self.assertEqual(decision.action, Action.PREPARE_ROTATIONAL_PATCH)
        unresolved = decide(
            state(breaking_precursor_b=0.88, surface_flux_defect=2.0e-3)
        )
        self.assertEqual(unresolved.action, Action.REFINE_SURFACE_GRID)

    def test_first_overhang_hands_off_but_intersection_is_rejected(self) -> None:
        handoff = decide(state(minimum_x_alpha=-0.01))
        self.assertEqual(handoff.action, Action.HANDOFF_TWO_PHASE)
        missed = decide(state(minimum_x_alpha=-0.2, self_intersection=True))
        self.assertEqual(missed.action, Action.REJECT_STATE)

    def test_steep_but_single_valued_surface_can_handoff_early(self) -> None:
        handoff = decide(state(maximum_surface_slope=1.1))
        self.assertEqual(handoff.action, Action.HANDOFF_TWO_PHASE)
        self.assertEqual(handoff.evidence_label, "validated pre-overturn handoff")
        unresolved = decide(
            state(maximum_surface_slope=1.1, surface_flux_defect=2.0e-3)
        )
        self.assertEqual(unresolved.action, Action.REFINE_SURFACE_GRID)

    def test_close_separation_requests_bounded_oversampling(self) -> None:
        decision = decide(
            state(minimum_boundary_separation=0.05, boundary_spacing=0.4)
        )
        self.assertEqual(decision.action, Action.REFINE_CLOSE_QUADRATURE)
        self.assertEqual(decision.requested_cross_oversampling, 8)


if __name__ == "__main__":
    unittest.main()
