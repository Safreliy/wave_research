"""Prepare a conservative regular-grid BIE state for the Basilisk receiver."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy import sparse
from scipy.sparse.linalg import lsqr


class LeastSquaresResult(NamedTuple):
    solution: np.ndarray
    stop_code: int
    iterations: int
    residual_norm: float


def symmetric_orthogonalization(a: float, b: float) -> tuple[float, float, float]:
    """Return a numerically stable real Givens rotation."""
    if b == 0.0:
        return float(np.sign(a)), 0.0, abs(a)
    if a == 0.0:
        return 0.0, float(np.sign(b)), abs(b)
    if abs(b) > abs(a):
        tau = a / b
        sine = math.copysign(1.0, b) / math.sqrt(1.0 + tau * tau)
        cosine = sine * tau
        radius = b / sine
    else:
        tau = b / a
        cosine = math.copysign(1.0, a) / math.sqrt(1.0 + tau * tau)
        sine = cosine * tau
        radius = a / cosine
    return cosine, sine, radius


def cupy_iterative_lsqr(
    system: sparse.spmatrix,
    rhs: np.ndarray,
    *,
    initial_solution: np.ndarray | None,
    tolerance: float,
    maximum_iterations: int,
) -> LeastSquaresResult:
    """Run the Paige--Saunders LSQR recurrence with CuPy sparse products.

    The scalar recurrence and stopping tests follow the SciPy reference path;
    only vectors and sparse matrix products live on the GPU.  Damping and
    variance estimation are intentionally omitted because the handoff system
    uses neither.
    """
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cupy_sparse
    except ImportError as error:
        raise RuntimeError(
            "cupy_lsqr requested but CuPy with CUDA support is not installed"
        ) from error

    gpu_system = cupy_sparse.csr_matrix(system)
    b = cp.asarray(rhs)
    column_count = system.shape[1]
    b_norm = float(cp.linalg.norm(b))
    if initial_solution is None:
        x = cp.zeros(column_count, dtype=b.dtype)
        u = b.copy()
    else:
        x = cp.asarray(initial_solution).copy()
        u = b - gpu_system @ x
    beta = float(cp.linalg.norm(u))
    if beta > 0.0:
        u /= beta
        v = gpu_system.T @ u
        alpha = float(cp.linalg.norm(v))
    else:
        v = x.copy()
        alpha = 0.0
    if alpha > 0.0:
        v /= alpha
    w = v.copy()

    rho_bar = alpha
    phi_bar = beta
    residual_norm = beta
    normal_residual_norm = alpha * beta
    if normal_residual_norm == 0.0:
        solution = cp.asnumpy(x)
        return LeastSquaresResult(solution, 0, 0, residual_norm)

    machine_epsilon = np.finfo(float).eps
    condition_limit_inverse = 1.0e-8
    matrix_norm = 0.0
    condition_estimate = 0.0
    direction_norm_sum = 0.0
    x_norm = 0.0
    x_norm_sum = 0.0
    z = 0.0
    cosine_right = -1.0
    sine_right = 0.0
    stop_code = 0
    iteration = 0
    while iteration < maximum_iterations:
        iteration += 1
        u = gpu_system @ v - alpha * u
        beta = float(cp.linalg.norm(u))
        if beta > 0.0:
            u /= beta
            matrix_norm = math.sqrt(matrix_norm**2 + alpha**2 + beta**2)
            v = gpu_system.T @ u - beta * v
            alpha = float(cp.linalg.norm(v))
            if alpha > 0.0:
                v /= alpha

        cosine, sine, rho = symmetric_orthogonalization(rho_bar, beta)
        theta = sine * alpha
        rho_bar = -cosine * alpha
        phi = cosine * phi_bar
        phi_bar = sine * phi_bar
        tau = sine * phi

        step = phi / rho
        recurrence = -theta / rho
        direction = w / rho
        x += step * w
        w = v + recurrence * w
        direction_norm_sum += float(cp.linalg.norm(direction)) ** 2

        delta = sine_right * rho
        gamma_bar = -cosine_right * rho
        right_hand_side = phi - delta * z
        z_bar = right_hand_side / gamma_bar
        x_norm = math.sqrt(x_norm_sum + z_bar**2)
        gamma = math.sqrt(gamma_bar**2 + theta**2)
        cosine_right = gamma_bar / gamma
        sine_right = theta / gamma
        z = right_hand_side / gamma
        x_norm_sum += z**2

        condition_estimate = matrix_norm * math.sqrt(direction_norm_sum)
        residual_norm = abs(phi_bar)
        normal_residual_norm = alpha * abs(tau)
        test_compatible = residual_norm / max(b_norm, machine_epsilon)
        test_least_squares = normal_residual_norm / (
            matrix_norm * residual_norm + machine_epsilon
        )
        test_condition = 1.0 / (condition_estimate + machine_epsilon)
        scaled_compatible = test_compatible / (
            1.0 + matrix_norm * x_norm / max(b_norm, machine_epsilon)
        )
        residual_tolerance = tolerance + (
            tolerance * matrix_norm * x_norm / max(b_norm, machine_epsilon)
        )

        if iteration >= maximum_iterations:
            stop_code = 7
        if 1.0 + test_condition <= 1.0:
            stop_code = 6
        if 1.0 + test_least_squares <= 1.0:
            stop_code = 5
        if 1.0 + scaled_compatible <= 1.0:
            stop_code = 4
        if test_condition <= condition_limit_inverse:
            stop_code = 3
        if test_least_squares <= tolerance:
            stop_code = 2
        if test_compatible <= residual_tolerance:
            stop_code = 1
        if stop_code:
            break

    cp.cuda.Stream.null.synchronize()
    solution = cp.asnumpy(x)
    residual_norm = float(np.linalg.norm(system @ solution - rhs))
    return LeastSquaresResult(solution, stop_code, iteration, residual_norm)


def solve_least_squares(
    system: sparse.spmatrix,
    rhs: np.ndarray,
    *,
    backend: str = "scipy_lsqr",
    initial_solution: np.ndarray | None = None,
    tolerance: float = 2.0e-9,
    maximum_iterations: int = 4000,
    equilibrate_columns: bool = False,
) -> LeastSquaresResult:
    """Solve a rectangular sparse least-squares system on CPU or GPU.

    SciPy LSQR remains the reference implementation.  The custom GPU LSQR path
    applies the same Paige--Saunders recurrence with GPU sparse products; CuPy
    LSMR is retained as a separate ablation.  Optional column equilibration is
    a right diagonal preconditioner and is undone before returning the state.
    """
    if equilibrate_columns:
        squared_norms = np.asarray(system.power(2).sum(axis=0)).ravel()
        scales = np.ones_like(squared_norms)
        nonzero = squared_norms > np.finfo(float).tiny
        scales[nonzero] = 1.0 / np.sqrt(squared_norms[nonzero])
        scaled_system = system @ sparse.diags(scales, format="csr")
        scaled_initial = (
            np.asarray(initial_solution) / scales
            if initial_solution is not None
            else None
        )
        scaled_result = solve_least_squares(
            scaled_system,
            rhs,
            backend=backend,
            initial_solution=scaled_initial,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
            equilibrate_columns=False,
        )
        solution = scales * scaled_result.solution
        return LeastSquaresResult(
            solution,
            scaled_result.stop_code,
            scaled_result.iterations,
            float(np.linalg.norm(system @ solution - rhs)),
        )
    if backend == "scipy_lsqr":
        result = lsqr(
            system,
            rhs,
            atol=tolerance,
            btol=tolerance,
            iter_lim=maximum_iterations,
            show=False,
            x0=initial_solution,
        )
        return LeastSquaresResult(
            np.asarray(result[0]),
            int(result[1]),
            int(result[2]),
            float(result[3]),
        )
    if backend == "cupy_lsqr":
        return cupy_iterative_lsqr(
            system,
            rhs,
            initial_solution=initial_solution,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
        )
    if backend != "cupy_lsmr":
        raise ValueError(
            "linear solver backend must be 'scipy_lsqr', 'cupy_lsqr' or "
            "'cupy_lsmr'"
        )
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cupy_sparse
        from cupyx.scipy.sparse.linalg import lsmr as cupy_lsmr
    except ImportError as error:
        raise RuntimeError(
            "cupy_lsmr requested but CuPy with CUDA support is not installed"
        ) from error

    gpu_system = cupy_sparse.csr_matrix(system)
    gpu_rhs = cp.asarray(rhs)
    gpu_initial = cp.asarray(initial_solution) if initial_solution is not None else None
    result = cupy_lsmr(
        gpu_system,
        gpu_rhs,
        x0=gpu_initial,
        atol=tolerance,
        btol=tolerance,
        maxiter=maximum_iterations,
    )
    cp.cuda.Stream.null.synchronize()
    solution = cp.asnumpy(result[0])
    # Compute the diagnostic with the same CPU matrix for both backends.  This
    # avoids relying on backend-specific meanings of the returned norm fields.
    residual_norm = float(np.linalg.norm(system @ solution - rhs))
    return LeastSquaresResult(
        solution,
        int(result[1]),
        int(result[2]),
        residual_norm,
    )


def correct_fraction_mass(fraction: np.ndarray, target_mass_cells: float) -> np.ndarray:
    corrected = np.clip(np.asarray(fraction, dtype=float).copy(), 0.0, 1.0)
    remaining = target_mass_cells - float(np.sum(corrected))
    for _ in range(4):
        if abs(remaining) <= 5.0e-13 * max(target_mass_cells, 1.0):
            break
        capacity = (1.0 - corrected) if remaining > 0.0 else corrected
        candidates = (corrected > 0.0) & (corrected < 1.0) & (capacity > 0.0)
        available = float(np.sum(capacity[candidates]))
        if available <= 0.0:
            raise ValueError("not enough interface-cell capacity for mass correction")
        amount = min(abs(remaining), available)
        sign = 1.0 if remaining > 0.0 else -1.0
        corrected[candidates] += sign * amount * capacity[candidates] / available
        corrected = np.clip(corrected, 0.0, 1.0)
        remaining = target_mass_cells - float(np.sum(corrected))
    return corrected


def centered_derivative_operators(
    nx: int, nz: int, dx: float, dz: float
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Return commuting cell-centred derivatives, periodic in x.

    The one-sided vertical boundary rows and centred interior rows form a
    Kronecker product with the periodic horizontal derivative. Consequently
    ``Dx @ Dz == Dz @ Dx`` up to roundoff, which makes a velocity obtained from
    one discrete streamfunction divergence-free by construction.
    """
    derivative_x = sparse.lil_matrix((nx, nx), dtype=float)
    for i in range(nx):
        derivative_x[i, (i + 1) % nx] = 0.5 / dx
        derivative_x[i, (i - 1) % nx] = -0.5 / dx
    derivative_z = sparse.lil_matrix((nz, nz), dtype=float)
    derivative_z[0, 0] = -1.0 / dz
    derivative_z[0, 1] = 1.0 / dz
    derivative_z[-1, -2] = -1.0 / dz
    derivative_z[-1, -1] = 1.0 / dz
    for k in range(1, nz - 1):
        derivative_z[k, k + 1] = 0.5 / dz
        derivative_z[k, k - 1] = -0.5 / dz
    dx_operator = sparse.kron(
        sparse.eye(nz, format="csr"), derivative_x.tocsr(), format="csr"
    )
    dz_operator = sparse.kron(
        derivative_z.tocsr(), sparse.eye(nx, format="csr"), format="csr"
    )
    return dx_operator, dz_operator


def standard_laplacian_operator(
    nx: int, nz: int, dx: float, dz: float
) -> sparse.csr_matrix:
    """Return a periodic-x, nonperiodic-z nearest-neighbour Laplacian.

    This operator is deliberately not formed as ``Dx@Dx + Dz@Dz``. Squaring a
    centred first derivative leaves the grid-Nyquist checkerboard in its null
    space, precisely the mode that contaminated the first streamfunction
    handoff experiment.
    """
    second_x = sparse.lil_matrix((nx, nx), dtype=float)
    for i in range(nx):
        second_x[i, i] = -2.0 / dx**2
        second_x[i, (i + 1) % nx] = 1.0 / dx**2
        second_x[i, (i - 1) % nx] = 1.0 / dx**2
    second_z = sparse.lil_matrix((nz, nz), dtype=float)
    for k in range(1, nz - 1):
        second_z[k, k] = -2.0 / dz**2
        second_z[k, k + 1] = 1.0 / dz**2
        second_z[k, k - 1] = 1.0 / dz**2
    # Reuse the closest interior stencil at the two artificial outer rows.
    second_z[0, 0] = 1.0 / dz**2
    second_z[0, 1] = -2.0 / dz**2
    second_z[0, 2] = 1.0 / dz**2
    second_z[-1, -3] = 1.0 / dz**2
    second_z[-1, -2] = -2.0 / dz**2
    second_z[-1, -1] = 1.0 / dz**2
    return sparse.kron(
        sparse.eye(nz, format="csr"), second_x.tocsr(), format="csr"
    ) + sparse.kron(
        second_z.tocsr(), sparse.eye(nx, format="csr"), format="csr"
    )


def bilinear_interpolation_matrix(
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    target_x: np.ndarray,
    target_z: np.ndarray,
) -> sparse.csr_matrix:
    """Interpolate cell-centred fields to arbitrary points, periodically in x."""
    nx = len(grid_x)
    nz = len(grid_z)
    dx = float(grid_x[1] - grid_x[0])
    dz = float(grid_z[1] - grid_z[0])
    gx = (np.asarray(target_x, dtype=float) - grid_x[0]) / dx
    gz = (np.asarray(target_z, dtype=float) - grid_z[0]) / dz
    i0_raw = np.floor(gx).astype(int)
    i0 = i0_raw % nx
    i1 = (i0 + 1) % nx
    ax = gx - i0_raw
    k0 = np.clip(np.floor(gz).astype(int), 0, nz - 2)
    az = np.clip(gz - k0, 0.0, 1.0)
    rows = np.repeat(np.arange(len(gx)), 4)
    columns = np.column_stack(
        (k0 * nx + i0, k0 * nx + i1, (k0 + 1) * nx + i0, (k0 + 1) * nx + i1)
    ).ravel()
    weights = np.column_stack(
        ((1.0 - ax) * (1.0 - az), ax * (1.0 - az), (1.0 - ax) * az, ax * az)
    ).ravel()
    return sparse.csr_matrix(
        (weights, (rows, columns)), shape=(len(gx), nx * nz)
    )


def streamfunction_harmonic_extension(
    data: dict[str, np.ndarray],
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    harmonic_weight: float = 0.20,
    surface_weight: float = 4.0,
    exterior_energy_weight: float = 0.05,
    speed_cap_ratio: float | None = None,
    exterior_speed_cap_ratio: float | None = None,
    speed_constraint_weight: float = 25.0,
    maximum_active_set_iterations: int = 4,
    bottom_streamfunction_weight: float = 100.0,
    momentum_constraint_weight: float = 0.0,
    linear_solver_backend: str = "scipy_lsqr",
    equilibrate_columns: bool = False,
    return_streamfunction: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]] | tuple[
    np.ndarray,
    np.ndarray,
    dict[str, float | int | str],
    np.ndarray,
]:
    """Fit a globally smooth, discretely solenoidal velocity extension.

    A scalar streamfunction is fitted to trustworthy bulk and surface
    velocities while a scaled Laplace residual regularizes the unobserved air,
    near-interface and solid cells. The returned velocity uses the same
    commuting derivative pair as the fit, hence its centred-grid divergence is
    roundoff-level before any pressure projection.
    """
    if (
        harmonic_weight < 0.0
        or surface_weight < 0.0
        or exterior_energy_weight < 0.0
        or (speed_cap_ratio is not None and speed_cap_ratio <= 0.0)
        or (exterior_speed_cap_ratio is not None and exterior_speed_cap_ratio <= 0.0)
        or speed_constraint_weight <= 0.0
        or maximum_active_set_iterations < 0
        or bottom_streamfunction_weight < 0.0
        or momentum_constraint_weight < 0.0
    ):
        raise ValueError("streamfunction weights cannot be negative")
    nx = len(grid_x)
    nz = len(grid_z)
    dx = float(grid_x[1] - grid_x[0])
    dz = float(grid_z[1] - grid_z[0])
    derivative_x, derivative_z = centered_derivative_operators(nx, nz, dx, dz)
    valid = np.asarray(data["velocity_valid"], dtype=bool)
    valid_indices = np.flatnonzero(valid.ravel())
    if not len(valid_indices):
        raise ValueError("handoff contains no valid bulk velocity samples")
    bulk_u = np.asarray(data["velocity_x"], dtype=float).ravel()[valid_indices]
    bulk_w = np.asarray(data["velocity_z"], dtype=float).ravel()[valid_indices]
    matrices: list[sparse.spmatrix] = [
        derivative_z[valid_indices],
        -derivative_x[valid_indices],
    ]
    right_hand_sides: list[np.ndarray] = [bulk_u, bulk_w]

    surface_x = np.asarray(data["surface_x"], dtype=float)
    surface_z = np.asarray(data["surface_z"], dtype=float)
    surface_u = np.asarray(data["surface_velocity_x"], dtype=float)
    surface_w = np.asarray(data["surface_velocity_z"], dtype=float)
    surface_interpolation = bilinear_interpolation_matrix(
        grid_x, grid_z, surface_x, surface_z
    )
    if surface_weight:
        scale = np.sqrt(surface_weight)
        matrices.extend(
            [
                scale * surface_interpolation @ derivative_z,
                -scale * surface_interpolation @ derivative_x,
            ]
        )
        right_hand_sides.extend([scale * surface_u, scale * surface_w])

    exterior_indices = np.flatnonzero(~valid.ravel())
    if exterior_energy_weight and len(exterior_indices):
        scale = np.sqrt(exterior_energy_weight)
        matrices.extend(
            [
                scale * derivative_z[exterior_indices],
                -scale * derivative_x[exterior_indices],
            ]
        )
        right_hand_sides.extend(
            [np.zeros(len(exterior_indices)), np.zeros(len(exterior_indices))]
        )

    laplacian = standard_laplacian_operator(nx, nz, dx, dz)
    regularization_length = harmonic_weight * min(dx, dz)
    if regularization_length:
        matrices.append(regularization_length * laplacian)
        right_hand_sides.append(np.zeros(nx * nz))
    has_bottom = "bottom_x" in data and "bottom_z" in data
    bottom_interpolation = (
        bilinear_interpolation_matrix(
            grid_x,
            grid_z,
            np.asarray(data["bottom_x"], dtype=float),
            np.asarray(data["bottom_z"], dtype=float),
        )
        if has_bottom
        else sparse.csr_matrix((0, nx * nz))
    )
    if bottom_streamfunction_weight and has_bottom:
        scale = np.sqrt(bottom_streamfunction_weight)
        matrices.append(scale * bottom_interpolation)
        right_hand_sides.append(np.zeros(bottom_interpolation.shape[0]))
    has_momentum = all(
        key in data
        for key in ("volume_fraction_prepared", "target_momentum_x", "target_momentum_z")
    )
    momentum_rows = sparse.csr_matrix((0, nx * nz))
    momentum_targets = np.empty(0)
    if momentum_constraint_weight and has_momentum:
        fraction_weights = (
            np.asarray(data["volume_fraction_prepared"], dtype=float).ravel()
            * dx
            * dz
        )
        momentum_rows = sparse.vstack(
            (
                sparse.csr_matrix(fraction_weights[np.newaxis, :]) @ derivative_z,
                -sparse.csr_matrix(fraction_weights[np.newaxis, :]) @ derivative_x,
            ),
            format="csr",
        )
        momentum_targets = np.asarray(
            [float(data["target_momentum_x"]), float(data["target_momentum_z"])],
            dtype=float,
        )
        scale = np.sqrt(momentum_constraint_weight)
        matrices.append(scale * momentum_rows)
        right_hand_sides.append(scale * momentum_targets)
    gauge = sparse.csr_matrix(np.ones((1, nx * nz)) / (nx * nz))
    matrices.append(gauge)
    right_hand_sides.append(np.zeros(1))
    system = sparse.vstack(matrices, format="csr")
    rhs = np.concatenate(right_hand_sides)
    solution = solve_least_squares(
        system,
        rhs,
        backend=linear_solver_backend,
        equilibrate_columns=equilibrate_columns,
    )
    active_set_iterations = 0
    active_set_size = 0
    active_union = np.empty(0, dtype=int)
    speed_cap = float("inf")
    exterior_speed_cap = float("inf")
    cap_field = np.full(nx * nz, np.inf)
    if speed_cap_ratio is not None:
        surface_speed_reference = max(
            float(np.quantile(np.hypot(surface_u, surface_w), 0.99)),
            np.finfo(float).eps,
        )
        speed_cap = speed_cap_ratio * surface_speed_reference
        exterior_speed_cap = (
            exterior_speed_cap_ratio * surface_speed_reference
            if exterior_speed_cap_ratio is not None
            else speed_cap
        )
        if "volume_fraction_prepared" in data:
            liquid_mask = np.asarray(
                data["volume_fraction_prepared"], dtype=float
            ).ravel() > 0.0
            cap_field[:] = exterior_speed_cap
            cap_field[liquid_mask] = speed_cap
        else:
            cap_field[:] = speed_cap
        for active_set_iterations in range(1, maximum_active_set_iterations + 1):
            trial_u = derivative_z @ solution.solution
            trial_w = -derivative_x @ solution.solution
            trial_speed = np.hypot(trial_u, trial_w)
            violations = np.flatnonzero(trial_speed > cap_field)
            if not len(violations):
                break
            # Keep all previously constrained degrees of freedom.  Replacing
            # the active set by only the newest violations lets an older peak
            # reappear when its constraint row is removed at the next solve.
            active_union = np.union1d(active_union, violations)
            active_set_size = int(len(active_union))
            active = active_union
            scale = np.sqrt(speed_constraint_weight)
            target_u = cap_field[active] * trial_u[active] / trial_speed[active]
            target_w = cap_field[active] * trial_w[active] / trial_speed[active]
            bounded_system = sparse.vstack(
                (
                    system,
                    scale * derivative_z[active],
                    -scale * derivative_x[active],
                ),
                format="csr",
            )
            bounded_rhs = np.concatenate(
                (rhs, scale * target_u, scale * target_w)
            )
            solution = solve_least_squares(
                bounded_system,
                bounded_rhs,
                backend=linear_solver_backend,
                initial_solution=solution.solution,
                equilibrate_columns=equilibrate_columns,
            )
    streamfunction = solution.solution
    velocity_x = (derivative_z @ streamfunction).reshape(nz, nx)
    velocity_z = (-derivative_x @ streamfunction).reshape(nz, nx)
    final_speed = np.hypot(velocity_x, velocity_z)
    final_speed_cap_violations = int(
        np.count_nonzero(
            final_speed.ravel() > cap_field * (1.0 + 1.0e-6)
        )
    ) if np.isfinite(speed_cap) else 0
    reconstructed_u = velocity_x.ravel()[valid_indices]
    reconstructed_w = velocity_z.ravel()[valid_indices]
    bulk_scale = max(
        float(np.linalg.norm(np.concatenate((bulk_u, bulk_w)))),
        np.finfo(float).eps,
    )
    bulk_error = float(
        np.linalg.norm(
            np.concatenate((reconstructed_u - bulk_u, reconstructed_w - bulk_w))
        )
        / bulk_scale
    )
    reconstructed_surface = np.concatenate(
        (
            surface_interpolation @ velocity_x.ravel(),
            surface_interpolation @ velocity_z.ravel(),
        )
    )
    target_surface = np.concatenate((surface_u, surface_w))
    surface_error = float(
        np.linalg.norm(reconstructed_surface - target_surface)
        / max(float(np.linalg.norm(target_surface)), np.finfo(float).eps)
    )
    divergence = (
        derivative_x @ velocity_x.ravel()
        + derivative_z @ velocity_z.ravel()
    )
    harmonic_residual = laplacian @ streamfunction
    bottom_streamfunction = bottom_interpolation @ streamfunction
    reconstructed_momentum = momentum_rows @ streamfunction
    momentum_error = (
        float(
            np.linalg.norm(reconstructed_momentum - momentum_targets)
            / max(np.linalg.norm(momentum_targets), np.finfo(float).eps)
        )
        if len(momentum_targets)
        else 0.0
    )
    velocity_scale = max(
        float(np.linalg.norm(np.concatenate((velocity_x.ravel(), velocity_z.ravel())))),
        np.finfo(float).eps,
    )
    metrics = {
        "extension_method": (
            "streamfunction_bounded"
            if speed_cap_ratio is not None
            else "streamfunction_harmonic"
        ),
        "harmonic_weight": harmonic_weight,
        "surface_weight": surface_weight,
        "exterior_energy_weight": exterior_energy_weight,
        "speed_cap_ratio": speed_cap_ratio,
        "speed_cap": speed_cap,
        "exterior_speed_cap_ratio": exterior_speed_cap_ratio,
        "exterior_speed_cap": exterior_speed_cap,
        "speed_constraint_weight": speed_constraint_weight,
        "bottom_streamfunction_weight": bottom_streamfunction_weight,
        "bottom_streamfunction_standard_deviation": float(
            np.std(bottom_streamfunction) if len(bottom_streamfunction) else 0.0
        ),
        "bottom_streamfunction_range": float(
            np.ptp(bottom_streamfunction) if len(bottom_streamfunction) else 0.0
        ),
        "momentum_constraint_weight": momentum_constraint_weight,
        "linear_solver_backend": linear_solver_backend,
        "least_squares_column_equilibration": equilibrate_columns,
        "relative_momentum_constraint_error": momentum_error,
        "active_set_iterations": active_set_iterations,
        "final_active_set_size": active_set_size,
        "final_speed_cap_violation_count": final_speed_cap_violations,
        "maximum_speed_after_bounding": float(np.max(final_speed)),
        "least_squares_stop_code": solution.stop_code,
        "least_squares_iterations": solution.iterations,
        "least_squares_relative_residual": float(
            solution.residual_norm / max(np.linalg.norm(rhs), 1.0)
        ),
        "relative_bulk_velocity_fit_error": bulk_error,
        "relative_surface_velocity_fit_error": surface_error,
        "streamfunction_divergence_l2": float(np.sqrt(np.mean(divergence**2))),
        "streamfunction_divergence_linf": float(np.max(np.abs(divergence))),
        "scaled_harmonic_residual": float(
            min(dx, dz) * np.linalg.norm(harmonic_residual) / velocity_scale
        ),
    }
    if return_streamfunction:
        return velocity_x, velocity_z, metrics, streamfunction.reshape(nz, nx)
    return velocity_x, velocity_z, metrics


def potential_harmonic_extension(
    data: dict[str, np.ndarray],
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    harmonic_weight: float = 0.05,
    surface_weight: float = 4.0,
    exterior_energy_weight: float = 0.01,
    bulk_velocity_weight: float = 1.0,
    surface_velocity_weight: float | None = None,
    bottom_normal_weight: float = 0.0,
    linear_solver_backend: str = "scipy_lsqr",
    equilibrate_columns: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    """Fit a curl-free potential field with a harmonicity penalty.

    Unlike a streamfunction fit this reconstruction preserves the defining
    irrotational structure of the pre-impact Euler state.  Its small residual
    divergence is intentionally left for the receiving solver's pressure
    projection and is reported separately from curl and data-fit errors.
    """
    if (
        harmonic_weight < 0.0
        or surface_weight < 0.0
        or exterior_energy_weight < 0.0
        or bulk_velocity_weight < 0.0
        or (surface_velocity_weight is not None and surface_velocity_weight < 0.0)
        or bottom_normal_weight < 0.0
    ):
        raise ValueError("potential-extension weights cannot be negative")
    nx = len(grid_x)
    nz = len(grid_z)
    dx = float(grid_x[1] - grid_x[0])
    dz = float(grid_z[1] - grid_z[0])
    derivative_x, derivative_z = centered_derivative_operators(nx, nz, dx, dz)
    laplacian = standard_laplacian_operator(nx, nz, dx, dz)
    valid = np.asarray(data["velocity_valid"], dtype=bool)
    valid_indices = np.flatnonzero(valid.ravel())
    if not len(valid_indices):
        raise ValueError("handoff contains no valid bulk velocity samples")
    background_current = float(data.get("background_current", np.asarray(0.0)))
    target_bulk_u = np.asarray(data["velocity_x"], dtype=float).ravel()[valid_indices]
    bulk_u = target_bulk_u - background_current
    bulk_w = np.asarray(data["velocity_z"], dtype=float).ravel()[valid_indices]
    matrices: list[sparse.spmatrix] = []
    right_hand_sides: list[np.ndarray] = []
    if bulk_velocity_weight:
        bulk_velocity_scale = np.sqrt(bulk_velocity_weight)
        matrices.extend(
            [
                bulk_velocity_scale * derivative_x[valid_indices],
                bulk_velocity_scale * derivative_z[valid_indices],
            ]
        )
        right_hand_sides.extend(
            [bulk_velocity_scale * bulk_u, bulk_velocity_scale * bulk_w]
        )
    characteristic_length = max(float(np.ptp(grid_z)), 1.0)
    if "bulk_potential" in data:
        potential_valid = np.asarray(
            data.get("potential_valid", valid), dtype=bool
        )
        potential_indices = np.flatnonzero(potential_valid.ravel())
        xx, _ = np.meshgrid(grid_x, grid_z)
        bulk_potential_values = (
            np.asarray(data["bulk_potential"], dtype=float).ravel()[potential_indices]
            - background_current * xx.ravel()[potential_indices]
        )
        bulk_potential_gauge = float(np.mean(bulk_potential_values))
        value_scale = 1.0 / characteristic_length
        selector = sparse.csr_matrix(
            (
                np.full(len(potential_indices), value_scale),
                (np.arange(len(potential_indices)), potential_indices),
            ),
            shape=(len(potential_indices), nx * nz),
        )
        matrices.append(selector)
        right_hand_sides.append(
            value_scale * (bulk_potential_values - bulk_potential_gauge)
        )
    surface_interpolation = bilinear_interpolation_matrix(
        grid_x,
        grid_z,
        np.asarray(data["surface_x"], dtype=float),
        np.asarray(data["surface_z"], dtype=float),
    )
    surface_potential = np.asarray(data["surface_potential"], dtype=float)
    potential_gauge = float(np.mean(surface_potential))
    surface_u = np.asarray(data["surface_velocity_x"], dtype=float)
    surface_w = np.asarray(data["surface_velocity_z"], dtype=float)
    if surface_weight:
        scale = np.sqrt(surface_weight) / characteristic_length
        matrices.append(scale * surface_interpolation)
        right_hand_sides.append(scale * (surface_potential - potential_gauge))
        # Potential values alone do not control the velocity trace on a coarse
        # Cartesian grid.  Fit both components of grad(phi) at the interface;
        # this retains exact discrete curl-freeness while preventing the
        # surface-speed mismatch seen in the value-only ablation.
        effective_surface_velocity_weight = (
            surface_weight
            if surface_velocity_weight is None
            else surface_velocity_weight
        )
        if effective_surface_velocity_weight:
            velocity_scale = np.sqrt(effective_surface_velocity_weight)
            matrices.extend(
                [
                    velocity_scale * surface_interpolation @ derivative_x,
                    velocity_scale * surface_interpolation @ derivative_z,
                ]
            )
            right_hand_sides.extend(
                [
                    velocity_scale * (surface_u - background_current),
                    velocity_scale * surface_w,
                ]
            )
    else:
        effective_surface_velocity_weight = 0.0
    if bottom_normal_weight and "bottom_x" in data and "bottom_z" in data:
        bottom_x = np.asarray(data["bottom_x"], dtype=float)
        bottom_z = np.asarray(data["bottom_z"], dtype=float)
        tangent_x = np.roll(bottom_x, -1) - np.roll(bottom_x, 1)
        tangent_z = np.roll(bottom_z, -1) - np.roll(bottom_z, 1)
        metric = np.maximum(np.hypot(tangent_x, tangent_z), np.finfo(float).eps)
        normal_x = tangent_z / metric
        normal_z = -tangent_x / metric
        bottom_interpolation = bilinear_interpolation_matrix(
            grid_x, grid_z, bottom_x, bottom_z
        )
        bottom_operator = (
            sparse.diags(normal_x) @ bottom_interpolation @ derivative_x
            + sparse.diags(normal_z) @ bottom_interpolation @ derivative_z
        )
        bottom_target = np.asarray(
            data.get("bottom_normal_derivative", np.zeros(len(bottom_x))),
            dtype=float,
        )
        bottom_scale = np.sqrt(bottom_normal_weight)
        matrices.append(bottom_scale * bottom_operator)
        right_hand_sides.append(bottom_scale * bottom_target)
    exterior_indices = np.flatnonzero(~valid.ravel())
    if exterior_energy_weight and len(exterior_indices):
        scale = np.sqrt(exterior_energy_weight)
        matrices.extend(
            [
                scale * derivative_x[exterior_indices],
                scale * derivative_z[exterior_indices],
            ]
        )
        right_hand_sides.extend(
            [np.zeros(len(exterior_indices)), np.zeros(len(exterior_indices))]
        )
    regularization_length = harmonic_weight * min(dx, dz)
    if regularization_length:
        matrices.append(regularization_length * laplacian)
        right_hand_sides.append(np.zeros(nx * nz))
    gauge = sparse.csr_matrix(np.ones((1, nx * nz)) / (nx * nz))
    matrices.append(gauge)
    right_hand_sides.append(np.zeros(1))
    system = sparse.vstack(matrices, format="csr")
    rhs = np.concatenate(right_hand_sides)
    solution = solve_least_squares(
        system,
        rhs,
        backend=linear_solver_backend,
        equilibrate_columns=equilibrate_columns,
    )
    potential = solution.solution
    velocity_x = (derivative_x @ potential).reshape(nz, nx) + background_current
    velocity_z = (derivative_z @ potential).reshape(nz, nx)
    reconstructed_bulk = np.concatenate(
        (velocity_x.ravel()[valid_indices], velocity_z.ravel()[valid_indices])
    )
    target_bulk = np.concatenate((target_bulk_u, bulk_w))
    bulk_error = float(
        np.linalg.norm(reconstructed_bulk - target_bulk)
        / max(float(np.linalg.norm(target_bulk)), np.finfo(float).eps)
    )
    reconstructed_surface_velocity = np.concatenate(
        (
            surface_interpolation @ velocity_x.ravel(),
            surface_interpolation @ velocity_z.ravel(),
        )
    )
    target_surface_velocity = np.concatenate((surface_u, surface_w))
    reconstructed_surface_potential = surface_interpolation @ potential
    potential_scale = max(
        float(np.linalg.norm(surface_potential - potential_gauge)),
        np.finfo(float).eps,
    )
    divergence = derivative_x @ velocity_x.ravel() + derivative_z @ velocity_z.ravel()
    curl = derivative_x @ velocity_z.ravel() - derivative_z @ velocity_x.ravel()
    return velocity_x, velocity_z, {
        "extension_method": "potential_harmonic",
        "harmonic_weight": harmonic_weight,
        "surface_weight": surface_weight,
        "exterior_energy_weight": exterior_energy_weight,
        "bulk_velocity_weight": bulk_velocity_weight,
        "surface_velocity_weight": effective_surface_velocity_weight,
        "bottom_normal_weight": bottom_normal_weight,
        "linear_solver_backend": linear_solver_backend,
        "least_squares_column_equilibration": equilibrate_columns,
        "least_squares_stop_code": solution.stop_code,
        "least_squares_iterations": solution.iterations,
        "least_squares_relative_residual": float(
            solution.residual_norm / max(np.linalg.norm(rhs), 1.0)
        ),
        "relative_bulk_velocity_fit_error": bulk_error,
        "relative_surface_velocity_fit_error": float(
            np.linalg.norm(reconstructed_surface_velocity - target_surface_velocity)
            / max(float(np.linalg.norm(target_surface_velocity)), np.finfo(float).eps)
        ),
        "relative_surface_potential_fit_error": float(
            np.linalg.norm(
                reconstructed_surface_potential - (surface_potential - potential_gauge)
            )
            / potential_scale
        ),
        "potential_divergence_l2": float(np.sqrt(np.mean(divergence**2))),
        "potential_divergence_linf": float(np.max(np.abs(divergence))),
        "potential_curl_l2": float(np.sqrt(np.mean(curl**2))),
        "potential_curl_linf": float(np.max(np.abs(curl))),
    }


def prepare(
    input_path: Path,
    output_prefix: Path,
    velocity_smoothing_sigma: float = 2.0,
    extension_method: str = "streamfunction_harmonic",
    harmonic_weight: float = 0.20,
    surface_weight: float = 4.0,
    exterior_energy_weight: float = 0.05,
    potential_bulk_velocity_weight: float = 1.0,
    potential_surface_velocity_weight: float | None = None,
    potential_bottom_normal_weight: float = 0.0,
    speed_cap_ratio: float = 1.35,
    exterior_speed_cap_ratio: float | None = None,
    speed_constraint_weight: float = 25.0,
    maximum_active_set_iterations: int = 4,
    bottom_streamfunction_weight: float = 100.0,
    momentum_constraint_weight: float = 0.0,
    linear_solver_backend: str = "scipy_lsqr",
    equilibrate_columns: bool = False,
) -> dict[str, object]:
    if velocity_smoothing_sigma < 0.0:
        raise ValueError("velocity_smoothing_sigma cannot be negative")
    with np.load(input_path) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    grid_x = np.asarray(data["grid_x"], dtype=float)
    grid_z = np.asarray(data["grid_z"], dtype=float)
    dx = float(grid_x[1] - grid_x[0])
    dz = float(grid_z[1] - grid_z[0])
    cell_area = dx * dz
    target_volume = float(data["source_volume"])
    fraction_raw = np.asarray(data["volume_fraction"], dtype=float)
    fraction = correct_fraction_mass(fraction_raw, target_volume / cell_area)
    valid = np.asarray(data["velocity_valid"], dtype=bool)
    if not np.any(valid):
        raise ValueError("handoff contains no valid bulk velocity samples")
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    nearest_x = np.asarray(data["velocity_x"], dtype=float)[tuple(nearest)]
    nearest_z = np.asarray(data["velocity_z"], dtype=float)[tuple(nearest)]
    active = fraction > 0.0
    extension_metrics: dict[str, float | int | str]
    streamfunction_prepared: np.ndarray | None = None
    if extension_method == "nearest_gaussian":
        velocity_x = nearest_x.copy()
        velocity_z = nearest_z.copy()
        velocity_x[~active] = 0.0
        velocity_z[~active] = 0.0
        unsmoothed_x = velocity_x.copy()
        unsmoothed_z = velocity_z.copy()
        if velocity_smoothing_sigma:
            weights = gaussian_filter(
                fraction, velocity_smoothing_sigma, mode=("nearest", "wrap")
            )
            for velocity in (velocity_x, velocity_z):
                numerator = gaussian_filter(
                    velocity * fraction,
                    velocity_smoothing_sigma,
                    mode=("nearest", "wrap"),
                )
                velocity[:] = numerator / np.maximum(weights, 1.0e-12)
                velocity[~active] = 0.0
        weighted_change = float(
            np.sqrt(
                np.sum(
                    fraction
                    * (
                        (velocity_x - unsmoothed_x) ** 2
                        + (velocity_z - unsmoothed_z) ** 2
                    )
                )
                / max(
                    np.sum(fraction * (unsmoothed_x**2 + unsmoothed_z**2)),
                    np.finfo(float).eps,
                )
            )
        )
        extension_metrics = {
            "extension_method": extension_method,
            "velocity_smoothing_sigma_cells": velocity_smoothing_sigma,
            "relative_weighted_velocity_smoothing_change": weighted_change,
        }
    elif extension_method in ("streamfunction_harmonic", "streamfunction_bounded"):
        extension_data = {
            **data,
            "volume_fraction_prepared": fraction,
        }
        (
            velocity_x,
            velocity_z,
            extension_metrics,
            streamfunction_prepared,
        ) = streamfunction_harmonic_extension(
            extension_data,
            grid_x,
            grid_z,
            harmonic_weight=harmonic_weight,
            surface_weight=surface_weight,
            exterior_energy_weight=exterior_energy_weight,
            speed_cap_ratio=(
                speed_cap_ratio
                if extension_method == "streamfunction_bounded"
                else None
            ),
            exterior_speed_cap_ratio=(
                exterior_speed_cap_ratio
                if extension_method == "streamfunction_bounded"
                else None
            ),
            speed_constraint_weight=speed_constraint_weight,
            maximum_active_set_iterations=maximum_active_set_iterations,
            bottom_streamfunction_weight=bottom_streamfunction_weight,
            momentum_constraint_weight=momentum_constraint_weight,
            linear_solver_backend=linear_solver_backend,
            equilibrate_columns=equilibrate_columns,
            return_streamfunction=True,
        )
        nearest_scale = max(
            float(
                np.sqrt(np.sum(fraction * (nearest_x**2 + nearest_z**2)))
            ),
            np.finfo(float).eps,
        )
        extension_metrics["relative_weighted_change_from_nearest_extension"] = float(
            np.sqrt(
                np.sum(
                    fraction
                    * ((velocity_x - nearest_x) ** 2 + (velocity_z - nearest_z) ** 2)
                )
            )
            / nearest_scale
        )
    elif extension_method == "potential_harmonic":
        velocity_x, velocity_z, extension_metrics = potential_harmonic_extension(
            data,
            grid_x,
            grid_z,
            harmonic_weight=harmonic_weight,
            surface_weight=surface_weight,
            exterior_energy_weight=exterior_energy_weight,
            bulk_velocity_weight=potential_bulk_velocity_weight,
            surface_velocity_weight=potential_surface_velocity_weight,
            bottom_normal_weight=potential_bottom_normal_weight,
            linear_solver_backend=linear_solver_backend,
            equilibrate_columns=equilibrate_columns,
        )
        nearest_scale = max(
            float(np.sqrt(np.sum(fraction * (nearest_x**2 + nearest_z**2)))),
            np.finfo(float).eps,
        )
        extension_metrics["relative_weighted_change_from_nearest_extension"] = float(
            np.sqrt(
                np.sum(
                    fraction
                    * ((velocity_x - nearest_x) ** 2 + (velocity_z - nearest_z) ** 2)
                )
            )
            / nearest_scale
        )
    else:
        raise ValueError(
            "extension_method must be 'nearest_gaussian', "
            "'streamfunction_harmonic', 'streamfunction_bounded' or "
            "'potential_harmonic'"
        )
    mass = float(np.sum(fraction) * cell_area)
    momentum_before = np.array(
        [
            np.sum(fraction * velocity_x) * cell_area,
            np.sum(fraction * velocity_z) * cell_area,
        ]
    )
    target_momentum = np.array(
        [float(data["target_momentum_x"]), float(data["target_momentum_z"])]
    )
    velocity_shift = (target_momentum - momentum_before) / mass
    if extension_method in (
        "streamfunction_harmonic",
        "streamfunction_bounded",
        "potential_harmonic",
    ):
        # A global constant shift is itself divergence free and corrects the
        # liquid-weighted momentum without introducing an interface jump.
        velocity_x += velocity_shift[0]
        velocity_z += velocity_shift[1]
    else:
        velocity_x[active] += velocity_shift[0]
        velocity_z[active] += velocity_shift[1]
    momentum_after = np.array(
        [
            np.sum(fraction * velocity_x) * cell_area,
            np.sum(fraction * velocity_z) * cell_area,
        ]
    )
    source_velocity_x = np.asarray(data["velocity_x"], dtype=float)
    source_velocity_z = np.asarray(data["velocity_z"], dtype=float)
    post_bulk_scale = max(
        float(
            np.linalg.norm(
                np.concatenate((source_velocity_x[valid], source_velocity_z[valid]))
            )
        ),
        np.finfo(float).eps,
    )
    post_bulk_velocity_error = float(
        np.linalg.norm(
            np.concatenate(
                (
                    velocity_x[valid] - source_velocity_x[valid],
                    velocity_z[valid] - source_velocity_z[valid],
                )
            )
        )
        / post_bulk_scale
    )
    surface_interpolation = bilinear_interpolation_matrix(
        grid_x,
        grid_z,
        np.asarray(data["surface_x"], dtype=float),
        np.asarray(data["surface_z"], dtype=float),
    )
    target_surface_velocity = np.concatenate(
        (
            np.asarray(data["surface_velocity_x"], dtype=float),
            np.asarray(data["surface_velocity_z"], dtype=float),
        )
    )
    post_surface_velocity = np.concatenate(
        (
            surface_interpolation @ velocity_x.ravel(),
            surface_interpolation @ velocity_z.ravel(),
        )
    )
    post_surface_velocity_error = float(
        np.linalg.norm(post_surface_velocity - target_surface_velocity)
        / max(float(np.linalg.norm(target_surface_velocity)), np.finfo(float).eps)
    )
    core = fraction > 0.999
    derivative_x, derivative_z = centered_derivative_operators(
        len(grid_x), len(grid_z), dx, dz
    )
    divergence = (
        derivative_x @ velocity_x.ravel()
        + derivative_z @ velocity_z.ravel()
    ).reshape(fraction.shape)
    divergence_l2 = float(np.sqrt(np.mean(divergence[core] ** 2)))
    divergence_linf = float(np.max(np.abs(divergence[core])))
    speed = np.hypot(velocity_x, velocity_z)
    liquid_speed = speed[fraction > 0.01]
    surface_speed = np.hypot(
        np.asarray(data["surface_velocity_x"], dtype=float),
        np.asarray(data["surface_velocity_z"], dtype=float),
    )
    surface_speed_q99 = float(np.quantile(surface_speed, 0.99))

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    prepared_arrays: dict[str, np.ndarray] = {
        **data,
        "volume_fraction_prepared": fraction,
        "velocity_x_prepared": velocity_x,
        "velocity_z_prepared": velocity_z,
        "velocity_shift": velocity_shift,
    }
    if streamfunction_prepared is not None:
        prepared_arrays["streamfunction_prepared"] = streamfunction_prepared
    np.savez_compressed(output_prefix.with_suffix(".npz"), **prepared_arrays)
    header = output_prefix.with_suffix(".h")
    arrays = {
        "handoff_fraction": fraction,
        "handoff_u": velocity_x,
        "handoff_w": velocity_z,
        "handoff_bottom_x": np.asarray(data["bottom_x"], dtype=float),
        "handoff_bottom_z": np.asarray(data["bottom_z"], dtype=float),
        "handoff_surface_x": np.asarray(data["surface_x"], dtype=float),
        "handoff_surface_z": np.asarray(data["surface_z"], dtype=float),
    }
    if streamfunction_prepared is not None:
        arrays["handoff_streamfunction"] = streamfunction_prepared
    lines = [
        "/* Generated by prepare_handoff.py; do not edit manually. */",
        f"#define HANDOFF_NX {len(grid_x)}",
        f"#define HANDOFF_NZ {len(grid_z)}",
        f"#define HANDOFF_NB {len(data['bottom_x'])}",
        f"#define HANDOFF_NS {len(data['surface_x'])}",
        f"#define HANDOFF_HAS_STREAMFUNCTION {int(streamfunction_prepared is not None)}",
        f"#define HANDOFF_DOMAIN_LENGTH {dx * len(grid_x):.17g}",
        f"#define HANDOFF_VERTICAL_ORIGIN {grid_z[0] - 0.5 * dz:.17g}",
        f"static const double handoff_x0 = {grid_x[0]:.17g};",
        f"static const double handoff_z0 = {grid_z[0]:.17g};",
        f"static const double handoff_dx = {dx:.17g};",
        f"static const double handoff_dz = {dz:.17g};",
        f"static const double handoff_target_volume = {target_volume:.17g};",
        f"static const double handoff_target_momentum_x = {target_momentum[0]:.17g};",
        f"static const double handoff_target_momentum_z = {target_momentum[1]:.17g};",
        f"static const double handoff_velocity_shift_x = {velocity_shift[0]:.17g};",
        f"static const double handoff_velocity_shift_z = {velocity_shift[1]:.17g};",
    ]
    for name, values in arrays.items():
        flat = np.asarray(values).ravel()
        lines.append(f"static const double {name}[{len(flat)}] = {{")
        for start in range(0, len(flat), 8):
            lines.append("  " + ", ".join(f"{v:.10g}" for v in flat[start:start + 8]) + ",")
        lines.append("};")
    header.write_text("\n".join(lines) + "\n", encoding="ascii")
    report: dict[str, object] = {
        "schema": "bie-to-vof-preparation-v2",
        "source": str(input_path),
        "prepared_npz": str(output_prefix.with_suffix(".npz")),
        "generated_header": str(header),
        "source_volume": target_volume,
        "raw_raster_volume": float(np.sum(fraction_raw) * cell_area),
        "prepared_volume": mass,
        "relative_mass_error": (mass - target_volume) / target_volume,
        "target_momentum": target_momentum.tolist(),
        "momentum_before_uniform_correction": momentum_before.tolist(),
        "momentum_after_uniform_correction": momentum_after.tolist(),
        "relative_momentum_error": float(
            np.linalg.norm(momentum_after - target_momentum)
            / max(np.linalg.norm(target_momentum), np.finfo(float).eps)
        ),
        "uniform_velocity_shift": velocity_shift.tolist(),
        **extension_metrics,
        "post_correction_relative_bulk_velocity_fit_error": post_bulk_velocity_error,
        "post_correction_relative_surface_velocity_fit_error": post_surface_velocity_error,
        "speed_q99_all": float(np.quantile(speed, 0.99)),
        "maximum_speed_all": float(np.max(speed)),
        "speed_q99_liquid": float(np.quantile(liquid_speed, 0.99)),
        "maximum_speed_liquid": float(np.max(liquid_speed)),
        "surface_speed_q99": surface_speed_q99,
        "maximum_liquid_to_surface_q99_speed_ratio": float(
            np.max(liquid_speed) / max(surface_speed_q99, np.finfo(float).eps)
        ),
        "pre_projection_divergence_l2_full_liquid": divergence_l2,
        "pre_projection_divergence_linf_full_liquid": divergence_linf,
        "note": (
            "Harmonic streamfunction and potential reconstructions are "
            "publication ablations. Mass and liquid momentum are corrected "
            "before the receiving solver's pressure projection."
        ),
    }
    output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--velocity-sigma", type=float, default=2.0)
    parser.add_argument(
        "--extension",
        choices=(
            "streamfunction_harmonic",
            "streamfunction_bounded",
            "potential_harmonic",
            "nearest_gaussian",
        ),
        default="streamfunction_harmonic",
    )
    parser.add_argument("--harmonic-weight", type=float, default=0.20)
    parser.add_argument("--surface-weight", type=float, default=4.0)
    parser.add_argument("--exterior-energy-weight", type=float, default=0.05)
    parser.add_argument("--potential-bulk-velocity-weight", type=float, default=1.0)
    parser.add_argument("--potential-surface-velocity-weight", type=float)
    parser.add_argument("--potential-bottom-normal-weight", type=float, default=0.0)
    parser.add_argument("--speed-cap-ratio", type=float, default=1.35)
    parser.add_argument("--exterior-speed-cap-ratio", type=float)
    parser.add_argument("--speed-constraint-weight", type=float, default=25.0)
    parser.add_argument("--active-set-iterations", type=int, default=4)
    parser.add_argument("--bottom-streamfunction-weight", type=float, default=100.0)
    parser.add_argument("--momentum-constraint-weight", type=float, default=0.0)
    parser.add_argument(
        "--linear-backend",
        choices=("scipy_lsqr", "cupy_lsqr", "cupy_lsmr"),
        default="scipy_lsqr",
    )
    parser.add_argument(
        "--equilibrate-columns",
        action="store_true",
        help="right-scale least-squares columns to unit 2-norm before solving",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                input_path=args.input,
                output_prefix=args.output_prefix,
                velocity_smoothing_sigma=args.velocity_sigma,
                extension_method=args.extension,
                harmonic_weight=args.harmonic_weight,
                surface_weight=args.surface_weight,
                exterior_energy_weight=args.exterior_energy_weight,
                potential_bulk_velocity_weight=args.potential_bulk_velocity_weight,
                potential_surface_velocity_weight=args.potential_surface_velocity_weight,
                potential_bottom_normal_weight=args.potential_bottom_normal_weight,
                speed_cap_ratio=args.speed_cap_ratio,
                exterior_speed_cap_ratio=args.exterior_speed_cap_ratio,
                speed_constraint_weight=args.speed_constraint_weight,
                maximum_active_set_iterations=args.active_set_iterations,
                bottom_streamfunction_weight=args.bottom_streamfunction_weight,
                momentum_constraint_weight=args.momentum_constraint_weight,
                linear_solver_backend=args.linear_backend,
                equilibrate_columns=args.equilibrate_columns,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
