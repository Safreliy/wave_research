"""Jacobian-free implicit midpoint integrator with audited diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ImplicitDiagnostics:
    converged: bool
    newton_iterations: int
    krylov_iterations: int
    rhs_evaluations: int
    residual: float
    line_search_reductions: int


def _callable_gmres(
    matvec: Callable[[Array], Array],
    rhs: Array,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[Array, int, float]:
    dimension = len(rhs)
    maximum_iterations = min(maximum_iterations, dimension)
    beta = float(np.linalg.norm(rhs))
    if beta <= tolerance:
        return np.zeros_like(rhs), 0, beta
    basis = np.zeros((dimension, maximum_iterations + 1))
    hessenberg = np.zeros((maximum_iterations + 1, maximum_iterations))
    projected_rhs = np.zeros(maximum_iterations + 1)
    basis[:, 0] = rhs / beta
    projected_rhs[0] = beta
    candidate = np.zeros_like(rhs)
    residual = beta
    for iteration in range(maximum_iterations):
        vector = matvec(basis[:, iteration])
        for _ in range(2):
            coefficients = basis[:, : iteration + 1].T @ vector
            hessenberg[: iteration + 1, iteration] += coefficients
            vector -= basis[:, : iteration + 1] @ coefficients
        next_norm = float(np.linalg.norm(vector))
        hessenberg[iteration + 1, iteration] = next_norm
        if next_norm > 20.0 * np.finfo(float).eps:
            basis[:, iteration + 1] = vector / next_norm
        active = iteration + 1
        coefficients = np.linalg.lstsq(
            hessenberg[: active + 1, :active],
            projected_rhs[: active + 1],
            rcond=None,
        )[0]
        candidate = basis[:, :active] @ coefficients
        projected_residual = float(
            np.linalg.norm(
                hessenberg[: active + 1, :active] @ coefficients
                - projected_rhs[: active + 1]
            )
        )
        if projected_residual <= tolerance:
            # The Arnoldi residual is cheap and normally exact to rounding.
            # Evaluate the expensive nonlinear finite-difference matvec only
            # once before accepting convergence.
            residual = float(np.linalg.norm(matvec(candidate) - rhs))
            if residual <= tolerance:
                return candidate, active, residual
        if next_norm <= 20.0 * np.finfo(float).eps:
            break
    residual = float(np.linalg.norm(matvec(candidate) - rhs))
    return candidate, active, residual


def implicit_midpoint_step(
    state: Array,
    rhs: Callable[[Array], Array],
    dt: float,
    *,
    nonlinear_tolerance: float = 2.0e-10,
    maximum_newton_iterations: int = 8,
    maximum_krylov_iterations: int = 24,
) -> tuple[Array, ImplicitDiagnostics]:
    """Advance one step by solving y1-y0-dt*f((y0+y1)/2)=0.

    Finite-difference Jacobian products are evaluated only through ``rhs``.
    A backtracking line search prevents a Newton correction from silently
    accepting a larger nonlinear residual.
    """
    initial = np.asarray(state, dtype=float)
    if initial.ndim != 1 or not np.all(np.isfinite(initial)):
        raise ValueError("state must be a finite vector")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    rhs_evaluations = 0

    def evaluate(values: Array) -> Array:
        nonlocal rhs_evaluations
        result = np.asarray(rhs(values), dtype=float)
        rhs_evaluations += 1
        if result.shape != initial.shape or not np.all(np.isfinite(result)):
            raise FloatingPointError("implicit RHS returned an invalid vector")
        return result

    predictor = initial + dt * evaluate(initial)

    def nonlinear_residual(candidate: Array) -> Array:
        midpoint = 0.5 * (initial + candidate)
        return candidate - initial - dt * evaluate(midpoint)

    candidate = predictor
    total_krylov = 0
    total_reductions = 0
    scale = max(float(np.linalg.norm(initial, ord=np.inf)), 1.0)
    target = nonlinear_tolerance * scale
    residual_vector = nonlinear_residual(candidate)
    residual_norm = float(np.linalg.norm(residual_vector, ord=np.inf))
    for newton_iteration in range(1, maximum_newton_iterations + 1):
        if residual_norm <= target:
            return candidate, ImplicitDiagnostics(
                True,
                newton_iteration - 1,
                total_krylov,
                rhs_evaluations,
                residual_norm / scale,
                total_reductions,
            )
        base_candidate = candidate.copy()
        base_residual = residual_vector.copy()
        base_norm = residual_norm

        def jacobian_product(direction: Array) -> Array:
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm == 0.0:
                return np.zeros_like(direction)
            epsilon = math.sqrt(np.finfo(float).eps) * (
                1.0 + float(np.linalg.norm(base_candidate))
            ) / direction_norm
            return (
                nonlinear_residual(base_candidate + epsilon * direction)
                - base_residual
            ) / epsilon

        forcing = min(0.25, math.sqrt(max(base_norm / scale, 1.0e-16)))
        linear_tolerance = max(
            forcing * float(np.linalg.norm(base_residual)),
            0.1 * target,
        )
        correction, krylov_iterations, _ = _callable_gmres(
            jacobian_product,
            -base_residual,
            linear_tolerance,
            maximum_krylov_iterations,
        )
        total_krylov += krylov_iterations
        accepted = False
        step_length = 1.0
        for reduction in range(9):
            trial = base_candidate + step_length * correction
            trial_residual = nonlinear_residual(trial)
            trial_norm = float(np.linalg.norm(trial_residual, ord=np.inf))
            if trial_norm < base_norm:
                candidate = trial
                residual_vector = trial_residual
                residual_norm = trial_norm
                total_reductions += reduction
                accepted = True
                break
            step_length *= 0.5
        if not accepted:
            break
    return candidate, ImplicitDiagnostics(
        False,
        maximum_newton_iterations,
        total_krylov,
        rhs_evaluations,
        residual_norm / scale,
        total_reductions,
    )
