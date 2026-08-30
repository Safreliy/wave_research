"""A posteriori controller for selective free-surface physics.

The controller deliberately separates *physical complexity* from *numerical
invalidity*.  A large breaking precursor can request a rotational/two-phase
patch only while conservation and operator gates remain satisfied.  A failed
numerical gate requests refinement; it is never re-labelled as wave breaking.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class Action(str, Enum):
    POTENTIAL = "potential"
    REFINE_CLOSE_QUADRATURE = "refine_close_quadrature"
    REFINE_SURFACE_GRID = "refine_surface_grid"
    PREPARE_ROTATIONAL_PATCH = "prepare_rotational_patch"
    HANDOFF_TWO_PHASE = "handoff_two_phase"
    REJECT_STATE = "reject_state"


@dataclass(frozen=True)
class DiagnosticState:
    relative_energy_drift: float
    wave_volume_normalized_drift: float
    surface_flux_defect: float
    bie_residual: float
    marker_spacing_cv: float
    minimum_x_alpha: float
    mean_x_alpha: float
    minimum_boundary_separation: float
    boundary_spacing: float
    breaking_precursor_b: float = float("nan")
    maximum_surface_slope: float = float("nan")
    self_intersection: bool = False


@dataclass(frozen=True)
class ControllerThresholds:
    bie_residual: float = 1.0e-10
    energy_drift: float = 2.0e-3
    volume_drift: float = 2.0e-3
    surface_flux: float = 1.0e-5
    handoff_surface_flux: float = 1.0e-3
    marker_cv: float = 2.0e-2
    handoff_marker_cv: float = 7.5e-2
    close_ratio: float = 2.5
    precursor_b: float = 0.85
    handoff_surface_slope: float = 1.0
    normalized_mapping_warning: float = 0.15


@dataclass(frozen=True)
class ControllerDecision:
    action: Action
    evidence_label: str
    reasons: tuple[str, ...]
    requested_cross_oversampling: int


def _finite_state(state: DiagnosticState) -> bool:
    scalar_values = (
        state.relative_energy_drift,
        state.wave_volume_normalized_drift,
        state.surface_flux_defect,
        state.bie_residual,
        state.marker_spacing_cv,
        state.minimum_x_alpha,
        state.mean_x_alpha,
        state.minimum_boundary_separation,
        state.boundary_spacing,
    )
    return all(math.isfinite(value) for value in scalar_values)


def decide(
    state: DiagnosticState,
    thresholds: ControllerThresholds = ControllerThresholds(),
) -> ControllerDecision:
    if not _finite_state(state) or state.mean_x_alpha <= 0.0:
        return ControllerDecision(
            Action.REJECT_STATE,
            "invalid numerical state",
            ("non-finite diagnostic or non-positive mean mapping",),
            1,
        )
    separation = max(state.minimum_boundary_separation, 1.0e-15)
    close_ratio = state.boundary_spacing / separation
    requested_oversampling = max(
        1, min(8, math.ceil(thresholds.close_ratio * close_ratio))
    )
    if state.self_intersection:
        return ControllerDecision(
            Action.REJECT_STATE,
            "handoff window missed",
            ("interface already self-intersects",),
            requested_oversampling,
        )

    hard_failures: list[str] = []
    if abs(state.relative_energy_drift) > thresholds.energy_drift:
        hard_failures.append("energy gate")
    if abs(state.wave_volume_normalized_drift) > thresholds.volume_drift:
        hard_failures.append("volume gate")
    if state.bie_residual > thresholds.bie_residual:
        hard_failures.append("BIE residual gate")
    if state.marker_spacing_cv > thresholds.handoff_marker_cv:
        hard_failures.append("marker-spacing gate")
    if abs(state.surface_flux_defect) > thresholds.handoff_surface_flux:
        hard_failures.append("surface-flux gate")
    if hard_failures:
        return ControllerDecision(
            Action.REFINE_SURFACE_GRID,
            "numerically unresolved",
            tuple(hard_failures),
            requested_oversampling,
        )

    if state.minimum_x_alpha <= 0.0:
        return ControllerDecision(
            Action.HANDOFF_TWO_PHASE,
            "validated pre-impact handoff",
            ("first overhanging interface",),
            requested_oversampling,
        )

    if (
        math.isfinite(state.maximum_surface_slope)
        and state.maximum_surface_slope >= thresholds.handoff_surface_slope
    ):
        return ControllerDecision(
            Action.HANDOFF_TWO_PHASE,
            "validated pre-overturn handoff",
            ("maximum free-surface slope reached the early-handoff threshold",),
            requested_oversampling,
        )

    soft_failures: list[str] = []
    if state.marker_spacing_cv > thresholds.marker_cv:
        soft_failures.append("marker-spacing refinement")
    if abs(state.surface_flux_defect) > thresholds.surface_flux:
        soft_failures.append("surface-flux refinement")
    if soft_failures:
        return ControllerDecision(
            Action.REFINE_SURFACE_GRID,
            "refinement required before physical classification",
            tuple(soft_failures),
            requested_oversampling,
        )

    normalized_mapping = state.minimum_x_alpha / state.mean_x_alpha
    precursor = state.breaking_precursor_b
    if (
        (math.isfinite(precursor) and precursor >= thresholds.precursor_b)
        or normalized_mapping <= thresholds.normalized_mapping_warning
    ):
        reasons = []
        if math.isfinite(precursor) and precursor >= thresholds.precursor_b:
            reasons.append("kinematic breaking precursor")
        if normalized_mapping <= thresholds.normalized_mapping_warning:
            reasons.append("near-vertical mapping")
        return ControllerDecision(
            Action.PREPARE_ROTATIONAL_PATCH,
            "validated physical precursor",
            tuple(reasons),
            requested_oversampling,
        )

    if close_ratio > 1.0 or requested_oversampling > 1:
        return ControllerDecision(
            Action.REFINE_CLOSE_QUADRATURE,
            "potential flow with close-boundary refinement",
            ("surface--bottom separation requires oversampling",),
            requested_oversampling,
        )
    return ControllerDecision(
        Action.POTENTIAL,
        "validated potential regime",
        ("all a posteriori gates passed",),
        1,
    )
