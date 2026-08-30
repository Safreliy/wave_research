"""Active circulation-state formulation of the pre-contact Euler--BIE solver.

The potential trace and the integrated panel circulation are equivalent before
topology change.  This module makes the latter the evolved state:

    Gamma_j = Phi_{j+1} - Phi_j,   Phi = psi + U x.

The geometry is still advanced by the mixed boundary-integral velocity.  This
is therefore an active boundary-vortex-sheet formulation of the same
irrotational Euler problem, not a post-impact bulk-vorticity closure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from topographic_wave_solver import theta_s_rhs


Array = np.ndarray


@dataclass(frozen=True)
class ActiveVortexSheetState:
    """Free-surface nodes and integrated circulation on outgoing panels."""

    x: Array
    z: Array
    panel_circulation: Array
    potential_mean: float = 0.0

    def __post_init__(self) -> None:
        shapes = {
            np.asarray(self.x).shape,
            np.asarray(self.z).shape,
            np.asarray(self.panel_circulation).shape,
        }
        if len(shapes) != 1 or np.asarray(self.x).ndim != 1:
            raise ValueError("x, z and panel_circulation must be matching vectors")

    @property
    def total_circulation(self) -> float:
        return float(np.sum(self.panel_circulation))


@dataclass(frozen=True)
class ActiveVortexSheetRHS:
    x_t: Array
    z_t: Array
    circulation_t: Array
    maximum_bie_residual: float
    circulation_derivative_defect: float
    reconstruction_relative_error: float


@dataclass(frozen=True)
class ActiveVortexStepDiagnostics:
    maximum_bie_residual: float
    maximum_circulation_derivative_defect: float
    maximum_reconstruction_relative_error: float
    circulation_change: float


def potential_to_panel_circulation(
    x: Array,
    periodic_potential: Array,
    length: float,
    background_current: float = 0.0,
) -> Array:
    """Return integrated circulation on each periodic surface panel."""
    x = np.asarray(x, dtype=float)
    potential = np.asarray(periodic_potential, dtype=float)
    if x.ndim != 1 or x.shape != potential.shape:
        raise ValueError("x and periodic_potential must be matching vectors")
    if length <= 0.0:
        raise ValueError("length must be positive")
    total_potential = potential + background_current * x
    following = np.roll(total_potential, -1)
    following[-1] += background_current * length
    return following - total_potential


def panel_circulation_to_potential(
    x: Array,
    panel_circulation: Array,
    length: float,
    background_current: float = 0.0,
    potential_mean: float = 0.0,
    circulation_tolerance: float = 2.0e-10,
) -> Array:
    """Reconstruct the single-valued periodic potential trace from ``Gamma``.

    The only compatibility condition is Kelvin's period constraint
    ``sum(Gamma) = U L``.  The additive potential gauge is fixed by its mean.
    """
    x = np.asarray(x, dtype=float)
    circulation = np.asarray(panel_circulation, dtype=float)
    if x.ndim != 1 or x.shape != circulation.shape:
        raise ValueError("x and panel_circulation must be matching vectors")
    expected = background_current * length
    defect = float(np.sum(circulation) - expected)
    scale = max(abs(expected), float(np.sum(np.abs(circulation))), 1.0)
    if abs(defect) > circulation_tolerance * scale:
        raise ValueError(
            "panel circulation violates the periodic Kelvin constraint: "
            f"defect={defect:.3e}"
        )
    total_relative = np.zeros_like(circulation)
    total_relative[1:] = np.cumsum(circulation[:-1])
    periodic = total_relative - background_current * (x - x[0])
    periodic += potential_mean - float(np.mean(periodic))
    return periodic


def active_vortex_sheet_rhs(
    state: ActiveVortexSheetState,
    bottom_x: Array,
    bottom_z: Array,
    length: float,
    gravity: float,
    background_current: float = 0.0,
    tangential_gauge: str = "theta_s",
) -> ActiveVortexSheetRHS:
    """Evaluate the Euler--BIE RHS with circulation as the dynamic variable."""
    potential = panel_circulation_to_potential(
        state.x,
        state.panel_circulation,
        length,
        background_current,
        state.potential_mean,
    )
    reconstructed = potential_to_panel_circulation(
        state.x, potential, length, background_current
    )
    reconstruction_error = float(
        np.linalg.norm(reconstructed - state.panel_circulation)
        / max(float(np.linalg.norm(state.panel_circulation)), np.finfo(float).eps)
    )
    x_t, z_t, potential_t, residual = theta_s_rhs(
        state.x,
        state.z,
        potential,
        bottom_x,
        bottom_z,
        length,
        gravity,
        background_current,
        tangential_gauge,
    )
    # Phi = psi + U x is the multi-valued total trace.  Its time derivative is
    # periodic because the period U L is constant, so a periodic difference
    # gives the exact semi-discrete circulation balance.
    total_potential_t = potential_t + background_current * x_t
    circulation_t = np.roll(total_potential_t, -1) - total_potential_t
    derivative_defect = float(np.sum(circulation_t))
    return ActiveVortexSheetRHS(
        x_t,
        z_t,
        circulation_t,
        float(residual),
        derivative_defect,
        reconstruction_error,
    )


def step_active_vortex_sheet_rk4(
    state: ActiveVortexSheetState,
    bottom_x: Array,
    bottom_z: Array,
    length: float,
    gravity: float,
    dt: float,
    background_current: float = 0.0,
    tangential_gauge: str = "theta_s",
) -> tuple[ActiveVortexSheetState, ActiveVortexStepDiagnostics]:
    """Advance geometry and panel circulation with classical RK4."""
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be finite and positive")

    def evaluate(candidate: ActiveVortexSheetState) -> ActiveVortexSheetRHS:
        return active_vortex_sheet_rhs(
            candidate,
            bottom_x,
            bottom_z,
            length,
            gravity,
            background_current,
            tangential_gauge,
        )

    def stage(base: ActiveVortexSheetState, rhs: ActiveVortexSheetRHS, scale: float):
        return ActiveVortexSheetState(
            base.x + scale * rhs.x_t,
            base.z + scale * rhs.z_t,
            base.panel_circulation + scale * rhs.circulation_t,
            base.potential_mean,
        )

    k1 = evaluate(state)
    k2 = evaluate(stage(state, k1, 0.5 * dt))
    k3 = evaluate(stage(state, k2, 0.5 * dt))
    k4 = evaluate(stage(state, k3, dt))
    next_x = state.x + dt * (
        k1.x_t + 2.0 * k2.x_t + 2.0 * k3.x_t + k4.x_t
    ) / 6.0
    next_z = state.z + dt * (
        k1.z_t + 2.0 * k2.z_t + 2.0 * k3.z_t + k4.z_t
    ) / 6.0
    next_circulation = state.panel_circulation + dt * (
        k1.circulation_t
        + 2.0 * k2.circulation_t
        + 2.0 * k3.circulation_t
        + k4.circulation_t
    ) / 6.0
    # Correct only roundoff in the one analytically invariant scalar.  This is
    # normally O(eps) because every stage derivative telescopes to zero.
    target_total = state.total_circulation
    circulation_change = float(np.sum(next_circulation) - target_total)
    next_circulation -= circulation_change / len(next_circulation)
    next_state = ActiveVortexSheetState(
        next_x,
        next_z,
        next_circulation,
        state.potential_mean,
    )
    stages = (k1, k2, k3, k4)
    diagnostics = ActiveVortexStepDiagnostics(
        max(item.maximum_bie_residual for item in stages),
        max(abs(item.circulation_derivative_defect) for item in stages),
        max(item.reconstruction_relative_error for item in stages),
        circulation_change,
    )
    return next_state, diagnostics
