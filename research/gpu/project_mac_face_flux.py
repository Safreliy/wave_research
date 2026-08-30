"""Project BIE face-average fluxes into a divergence-compatible MAC space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse

_HANDOFF_MODULE = Path(__file__).resolve().parents[1] / "two_phase_basilisk"
if str(_HANDOFF_MODULE) not in sys.path:
    sys.path.insert(0, str(_HANDOFF_MODULE))

from prepare_handoff import (
    bilinear_interpolation_matrix,
    correct_fraction_mass,
    solve_least_squares,
    standard_laplacian_operator,
)


def load_receiver_quadrature(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load q quadrature without silently applying the embedded metric twice.

    ``q`` is a full-cell aperture, so its physical weight is
    ``q*cell_area``.  Early receiver exports called the last factor ``dv``;
    some of those files contain Basilisk's embedded measure ``cs*Delta**2``
    while newer files contain ``Delta**2``.  The former is incompatible and
    is rejected rather than guessed.
    """
    receiver = np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True)
    )
    names = set(receiver.dtype.names or ())
    required = {"x", "y", "delta", "f", "cs", "q", "u_x", "u_y"}
    missing = required.difference(names)
    if missing:
        raise ValueError(f"receiver quadrature is missing {sorted(missing)}")
    delta = np.asarray(receiver["delta"], dtype=float)
    expected_area = delta**2
    if "cell_area" in names:
        cell_area = np.asarray(receiver["cell_area"], dtype=float)
        measure_schema = "explicit-full-cell-area-v2"
    elif "dv" in names:
        cell_area = np.asarray(receiver["dv"], dtype=float)
        ratio = np.divide(
            cell_area,
            expected_area,
            out=np.full_like(cell_area, np.nan),
            where=expected_area > 0.0,
        )
        cs = np.asarray(receiver["cs"], dtype=float)
        if np.allclose(ratio, cs, rtol=2.0e-11, atol=2.0e-13) and not np.allclose(
            ratio, 1.0, rtol=2.0e-11, atol=2.0e-13
        ):
            raise ValueError(
                "legacy receiver 'dv' is cs*Delta^2; q is already a full-cell "
                "aperture, so q*dv would apply cs twice; regenerate the CSV"
            )
        measure_schema = "validated-legacy-full-cell-area-v1"
    else:
        raise ValueError("receiver quadrature is missing 'cell_area'")
    if not np.all(np.isfinite(cell_area)) or np.any(cell_area <= 0.0):
        raise ValueError("receiver cell areas must be finite and positive")
    if not np.allclose(
        cell_area, expected_area, rtol=2.0e-11, atol=2.0e-13
    ):
        raise ValueError("receiver cell_area must equal Delta^2 for q quadrature")
    q = np.asarray(receiver["q"], dtype=float)
    f = np.asarray(receiver["f"], dtype=float)
    cs = np.asarray(receiver["cs"], dtype=float)
    tolerance = 5.0e-10
    if np.any(~np.isfinite(q)) or np.any(q < -tolerance) or np.any(q > cs + tolerance):
        raise ValueError("receiver q violates 0 <= q <= cs")
    if not np.allclose(q, f * cs, rtol=2.0e-9, atol=5.0e-11):
        raise ValueError("receiver columns are inconsistent: q != f*cs")
    return receiver, cell_area, measure_schema


def mac_curl_operators(nx: int, nz: int, dx: float, dz: float) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Map a periodic-x vertex streamfunction to vertical/horizontal faces."""
    vertex_count = nx * (nz + 1)
    vertical_rows = np.arange(nx * nz)
    vertical_k, vertical_i = np.divmod(vertical_rows, nx)
    vertical = sparse.csr_matrix(
        (
            np.concatenate(
                (
                    -np.full(len(vertical_rows), 1.0 / dz),
                    np.full(len(vertical_rows), 1.0 / dz),
                )
            ),
            (
                np.concatenate((vertical_rows, vertical_rows)),
                np.concatenate(
                    (
                        vertical_k * nx + vertical_i,
                        (vertical_k + 1) * nx + vertical_i,
                    )
                ),
            ),
        ),
        shape=(nx * nz, vertex_count),
    )
    horizontal_rows = np.arange(nx * (nz + 1))
    horizontal_k, horizontal_i = np.divmod(horizontal_rows, nx)
    horizontal = sparse.csr_matrix(
        (
            np.concatenate(
                (
                    np.full(len(horizontal_rows), 1.0 / dx),
                    -np.full(len(horizontal_rows), 1.0 / dx),
                )
            ),
            (
                np.concatenate((horizontal_rows, horizontal_rows)),
                np.concatenate(
                    (
                        horizontal_k * nx + horizontal_i,
                        horizontal_k * nx + (horizontal_i + 1) % nx,
                    )
                ),
            ),
        ),
        shape=(nx * (nz + 1), vertex_count),
    )
    return vertical, horizontal


def append_c_array(lines: list[str], name: str, values: np.ndarray) -> None:
    flat = np.asarray(values).ravel()
    lines.append(f"static const double {name}[{len(flat)}] = {{")
    for start in range(0, len(flat), 8):
        lines.append(
            "  " + ", ".join(f"{value:.10g}" for value in flat[start : start + 8]) + ","
        )
    lines.append("};")


def mac_point_interpolation_matrix(
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    target_x: np.ndarray,
    target_z: np.ndarray,
) -> sparse.csr_matrix:
    """Match ``sample_mac_streamfunction()`` including zero exterior rows."""
    target_z = np.asarray(target_z, dtype=float)
    inside = (target_z >= z_edges[0]) & (target_z <= z_edges[-1])
    interpolation = bilinear_interpolation_matrix(
        x_edges, z_edges, np.asarray(target_x, dtype=float), target_z
    )
    return sparse.diags(inside.astype(float), format="csr") @ interpolation


def receiver_cell_velocity_operators(
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    receiver_x: np.ndarray,
    receiver_z: np.ndarray,
    receiver_delta: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Differentiate the MAC streamfunction exactly as the C receiver does."""
    inverse_span = sparse.diags(0.5 / receiver_delta, format="csr")
    up = mac_point_interpolation_matrix(
        x_edges, z_edges, receiver_x, receiver_z + receiver_delta
    )
    down = mac_point_interpolation_matrix(
        x_edges, z_edges, receiver_x, receiver_z - receiver_delta
    )
    right = mac_point_interpolation_matrix(
        x_edges, z_edges, receiver_x + receiver_delta, receiver_z
    )
    left = mac_point_interpolation_matrix(
        x_edges, z_edges, receiver_x - receiver_delta, receiver_z
    )
    return (
        (inverse_span @ (up - down)).tocsr(),
        (-inverse_span @ (right - left)).tocsr(),
    )


def project(
    input_path: Path,
    output_prefix: Path,
    backend: str,
    tolerance: float,
    maximum_iterations: int,
    bottom_streamfunction_weight: float,
    momentum_constraint_weight: float,
    surface_velocity_weight: float,
    exterior_energy_weight: float,
    harmonic_weight: float,
    receiver_quadrature_path: Path | None = None,
    receiver_velocity_preservation_weight: float = 0.0,
    receiver_localization_sigma: float | None = None,
    receiver_localization_floor: float = 1.0e-3,
) -> dict[str, object]:
    with np.load(input_path) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    required = {
        "face_velocity_x",
        "face_velocity_z",
        "face_velocity_x_valid",
        "face_velocity_z_valid",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"face-flux archive is missing {sorted(missing)}")
    x = np.asarray(data["grid_x"], dtype=float)
    z = np.asarray(data["grid_z"], dtype=float)
    nx = len(x)
    nz = len(z)
    dx = float(x[1] - x[0])
    dz = float(z[1] - z[0])
    vertical_target = np.asarray(data["face_velocity_x"], dtype=float)
    horizontal_target = np.asarray(data["face_velocity_z"], dtype=float)
    vertical_valid = np.asarray(data["face_velocity_x_valid"], dtype=bool)
    horizontal_valid = np.asarray(data["face_velocity_z_valid"], dtype=bool)
    vertical_operator, horizontal_operator = mac_curl_operators(nx, nz, dx, dz)
    vertical_indices = np.flatnonzero(vertical_valid.ravel())
    horizontal_indices = np.flatnonzero(horizontal_valid.ravel())
    matrices: list[sparse.spmatrix] = [
        vertical_operator[vertical_indices],
        horizontal_operator[horizontal_indices],
    ]
    right_hand_sides: list[np.ndarray] = [
        vertical_target.ravel()[vertical_indices],
        horizontal_target.ravel()[horizontal_indices],
    ]
    x_edges = x - 0.5 * dx
    z_edges = np.concatenate(([z[0] - 0.5 * dz], z + 0.5 * dz))
    bottom_interpolation = bilinear_interpolation_matrix(
        x_edges,
        z_edges,
        np.asarray(data["bottom_x"], dtype=float),
        np.asarray(data["bottom_z"], dtype=float),
    )
    if bottom_streamfunction_weight:
        bottom_scale = np.sqrt(bottom_streamfunction_weight)
        matrices.append(bottom_scale * bottom_interpolation)
        right_hand_sides.append(np.zeros(bottom_interpolation.shape[0]))
    right_face_indices = (
        np.arange(nx * nz).reshape(nz, nx) + 1
    )
    right_face_indices[:, -1] -= nx
    bottom_face_indices = np.arange(nx * nz).reshape(nz, nx)
    top_face_indices = bottom_face_indices + nx
    cell_u_operator = 0.5 * (
        vertical_operator + vertical_operator[right_face_indices.ravel()]
    )
    cell_w_operator = 0.5 * (
        horizontal_operator[bottom_face_indices.ravel()]
        + horizontal_operator[top_face_indices.ravel()]
    )
    surface_interpolation = bilinear_interpolation_matrix(
        x,
        z,
        np.asarray(data["surface_x"], dtype=float),
        np.asarray(data["surface_z"], dtype=float),
    )
    target_surface_u = np.asarray(data["surface_velocity_x"], dtype=float)
    target_surface_w = np.asarray(data["surface_velocity_z"], dtype=float)
    if surface_velocity_weight:
        surface_scale = np.sqrt(surface_velocity_weight)
        matrices.extend(
            (
                surface_scale * surface_interpolation @ cell_u_operator,
                surface_scale * surface_interpolation @ cell_w_operator,
            )
        )
        right_hand_sides.extend(
            (surface_scale * target_surface_u, surface_scale * target_surface_w)
        )
    fraction_raw = np.asarray(data["volume_fraction"], dtype=float)
    fraction = correct_fraction_mass(
        fraction_raw,
        float(data["source_volume"]) / (dx * dz),
    )
    cell_weights = fraction.ravel() * dx * dz
    if exterior_energy_weight:
        exterior_scale = np.sqrt(exterior_energy_weight)
        vertical_exterior = np.flatnonzero(
            (
                (fraction <= 0.01)
                & (np.roll(fraction, 1, axis=1) <= 0.01)
            ).ravel()
        )
        horizontal_exterior_mask = np.zeros((nz + 1, nx), dtype=bool)
        horizontal_exterior_mask[1:nz] = (
            (fraction[:-1] <= 0.01) & (fraction[1:] <= 0.01)
        )
        horizontal_exterior = np.flatnonzero(horizontal_exterior_mask.ravel())
        matrices.extend(
            (
                exterior_scale * vertical_operator[vertical_exterior],
                exterior_scale * horizontal_operator[horizontal_exterior],
            )
        )
        right_hand_sides.extend(
            (np.zeros(len(vertical_exterior)), np.zeros(len(horizontal_exterior)))
        )
    if harmonic_weight:
        harmonic_scale = harmonic_weight * min(dx, dz)
        vertex_laplacian = standard_laplacian_operator(nx, nz + 1, dx, dz)
        matrices.append(harmonic_scale * vertex_laplacian)
        right_hand_sides.append(np.zeros(nx * (nz + 1)))
    momentum_operator = sparse.vstack(
        (
            sparse.csr_matrix(cell_weights.reshape(1, -1)) @ cell_u_operator,
            sparse.csr_matrix(cell_weights.reshape(1, -1)) @ cell_w_operator,
        ),
        format="csr",
    )
    target_momentum = np.asarray(
        [float(data["target_momentum_x"]), float(data["target_momentum_z"])],
        dtype=float,
    )
    receiver_operator: sparse.csr_matrix | None = None
    receiver: np.ndarray | None = None
    receiver_weights: np.ndarray | None = None
    receiver_measure_schema: str | None = None
    receiver_velocity_operators: tuple[sparse.csr_matrix, sparse.csr_matrix] | None = None
    receiver_exported_velocity: np.ndarray | None = None
    receiver_localization_profile: np.ndarray | None = None
    receiver_grid_level: int | None = None
    if receiver_quadrature_path is not None:
        receiver, receiver_cell_area, receiver_measure_schema = (
            load_receiver_quadrature(receiver_quadrature_path)
        )
        receiver_weights = np.asarray(receiver["q"] * receiver_cell_area, dtype=float)
        receiver_finest_delta = float(np.min(receiver["delta"]))
        receiver_grid_level = int(round(np.log2((dx * nx) / receiver_finest_delta)))
        if not np.isclose(
            (dx * nx) / (2**receiver_grid_level),
            receiver_finest_delta,
            rtol=2.0e-11,
            atol=2.0e-13,
        ):
            raise ValueError("receiver Delta is not a dyadic level of the domain")
        receiver_exported_velocity = np.column_stack(
            (np.asarray(receiver["u_x"]), np.asarray(receiver["u_y"]))
        )
        receiver_velocity_operators = receiver_cell_velocity_operators(
            x_edges,
            z_edges,
            np.asarray(receiver["x"], dtype=float),
            np.asarray(receiver["y"], dtype=float),
            np.asarray(receiver["delta"], dtype=float),
        )
        receiver_u_operator, receiver_w_operator = receiver_velocity_operators
        receiver_operator = sparse.vstack(
            (
                sparse.csr_matrix(receiver_weights.reshape(1, -1))
                @ receiver_u_operator,
                sparse.csr_matrix(receiver_weights.reshape(1, -1))
                @ receiver_w_operator,
            ),
            format="csr",
        )
        if receiver_velocity_preservation_weight:
            if receiver_localization_sigma is None or receiver_localization_sigma <= 0.0:
                raise ValueError(
                    "receiver localization sigma must be positive when the preservation penalty is enabled"
                )
            if not 0.0 < receiver_localization_floor <= 1.0:
                raise ValueError("receiver localization floor must lie in (0, 1]")
            mixed = (receiver["f"] > 1.0e-6) & (receiver["f"] < 1.0 - 1.0e-6)
            candidates = receiver[mixed] if np.any(mixed) else receiver
            crest_z = float(np.max(candidates["y"]))
            top = candidates[
                np.abs(candidates["y"] - crest_z) <= 0.51 * candidates["delta"]
            ]
            crest_x = float(
                np.average(
                    top["x"],
                    weights=np.asarray(top["q"] * top["delta"] ** 2),
                )
            )
            domain_length = dx * nx
            raw_x_distance = np.abs(np.asarray(receiver["x"]) - crest_x)
            x_distance = np.minimum(raw_x_distance, domain_length - raw_x_distance)
            distance = np.hypot(x_distance, np.asarray(receiver["y"]) - crest_z)
            receiver_localization_profile = np.exp(
                -0.5 * (distance / receiver_localization_sigma) ** 2
            )
            penalty = np.sqrt(
                receiver_velocity_preservation_weight
                * receiver_weights
                / np.maximum(receiver_localization_profile, receiver_localization_floor)
            )
            penalty_matrix = sparse.diags(penalty, format="csr")
            matrices.extend(
                (
                    penalty_matrix @ receiver_u_operator,
                    penalty_matrix @ receiver_w_operator,
                )
            )
            right_hand_sides.extend(
                (
                    penalty * receiver_exported_velocity[:, 0],
                    penalty * receiver_exported_velocity[:, 1],
                )
            )
    if momentum_constraint_weight:
        momentum_scale = np.sqrt(momentum_constraint_weight)
        matrices.append(
            momentum_scale
            * (receiver_operator if receiver_operator is not None else momentum_operator)
        )
        right_hand_sides.append(momentum_scale * target_momentum)
    gauge = sparse.csr_matrix(np.ones((1, nx * (nz + 1))) / (nx * (nz + 1)))
    matrices.append(gauge)
    right_hand_sides.append(np.zeros(1))
    system = sparse.vstack(matrices, format="csr")
    rhs = np.concatenate(right_hand_sides)
    started = time.perf_counter()
    solution = solve_least_squares(
        system,
        rhs,
        backend=backend,
        equilibrate_columns=True,
        tolerance=tolerance,
        maximum_iterations=maximum_iterations,
    )
    elapsed = time.perf_counter() - started
    streamfunction = solution.solution.reshape(nz + 1, nx)
    vertical = (vertical_operator @ solution.solution).reshape(nz, nx)
    horizontal = (horizontal_operator @ solution.solution).reshape(nz + 1, nx)
    cell_valid = (
        vertical_valid
        & np.roll(vertical_valid, -1, axis=1)
        & horizontal_valid[:-1]
        & horizontal_valid[1:]
    )
    divergence = (
        (np.roll(vertical, -1, axis=1) - vertical) / dx
        + (horizontal[1:] - horizontal[:-1]) / dz
    )
    fitted = np.concatenate(
        (vertical.ravel()[vertical_indices], horizontal.ravel()[horizontal_indices])
    )
    target = np.concatenate(
        (
            vertical_target.ravel()[vertical_indices],
            horizontal_target.ravel()[horizontal_indices],
        )
    )
    face_fit_error = float(
        np.linalg.norm(fitted - target)
        / max(float(np.linalg.norm(target)), np.finfo(float).eps)
    )
    cell_u = 0.5 * (vertical + np.roll(vertical, -1, axis=1))
    cell_w = 0.5 * (horizontal[:-1] + horizontal[1:])
    cell_speed = np.hypot(cell_u, cell_w)
    surface_speed_q99 = float(
        np.quantile(np.hypot(target_surface_u, target_surface_w), 0.99)
    )
    reconstructed_surface_u = np.asarray(
        surface_interpolation @ cell_u.ravel()
    ).ravel()
    reconstructed_surface_w = np.asarray(
        surface_interpolation @ cell_w.ravel()
    ).ravel()
    relative_surface_velocity_error = float(
        np.linalg.norm(
            np.concatenate(
                (
                    reconstructed_surface_u - target_surface_u,
                    reconstructed_surface_w - target_surface_w,
                )
            )
        )
        / max(
            float(np.linalg.norm(np.concatenate((target_surface_u, target_surface_w)))),
            np.finfo(float).eps,
        )
    )
    reconstructed_momentum = np.asarray(momentum_operator @ solution.solution).ravel()
    relative_momentum_error = float(
        np.linalg.norm(reconstructed_momentum - target_momentum)
        / max(float(np.linalg.norm(target_momentum)), np.finfo(float).eps)
    )
    reconstructed_receiver_momentum: np.ndarray | None = None
    relative_receiver_momentum_error: float | None = None
    receiver_velocity_fit_error: float | None = None
    receiver_weighted_velocity_change: float | None = None
    receiver_correction_lower_bound_ratio: float | None = None
    receiver_relative_kinetic_energy_change: float | None = None
    receiver_local_correction_energy_fraction: float | None = None
    if receiver_operator is not None and receiver_velocity_operators is not None:
        reconstructed_receiver_momentum = np.asarray(
            receiver_operator @ solution.solution
        ).ravel()
        relative_receiver_momentum_error = float(
            np.linalg.norm(reconstructed_receiver_momentum - target_momentum)
            / max(float(np.linalg.norm(target_momentum)), np.finfo(float).eps)
        )
        assert receiver is not None
        receiver_u_operator, receiver_w_operator = receiver_velocity_operators
        fitted_receiver_velocity = np.concatenate(
            (
                np.asarray(receiver_u_operator @ solution.solution).ravel(),
                np.asarray(receiver_w_operator @ solution.solution).ravel(),
            )
        )
        assert receiver_exported_velocity is not None
        exported_receiver_velocity_flat = receiver_exported_velocity.T.ravel()
        receiver_velocity_fit_error = float(
            np.linalg.norm(fitted_receiver_velocity - exported_receiver_velocity_flat)
            / max(
                float(np.linalg.norm(exported_receiver_velocity_flat)),
                np.finfo(float).eps,
            )
        )
        fitted_receiver_components = fitted_receiver_velocity.reshape(2, -1).T
        receiver_correction = fitted_receiver_components - receiver_exported_velocity
        receiver_velocity_norm = np.sqrt(
            float(np.sum(receiver_weights * np.sum(receiver_exported_velocity**2, axis=1)))
        )
        receiver_correction_norm = np.sqrt(
            float(np.sum(receiver_weights * np.sum(receiver_correction**2, axis=1)))
        )
        receiver_weighted_velocity_change = float(
            receiver_correction_norm
            / max(receiver_velocity_norm, np.finfo(float).eps)
        )
        exported_momentum = np.sum(
            receiver_weights[:, None] * receiver_exported_velocity, axis=0
        )
        posterior_global_lower_bound = float(
            np.linalg.norm(target_momentum - exported_momentum)
            / np.sqrt(np.sum(receiver_weights))
        )
        receiver_correction_lower_bound_ratio = float(
            receiver_correction_norm
            / max(posterior_global_lower_bound, np.finfo(float).eps)
        )
        receiver_energy_before = 0.5 * float(
            np.sum(receiver_weights * np.sum(receiver_exported_velocity**2, axis=1))
        )
        receiver_energy_after = 0.5 * float(
            np.sum(receiver_weights * np.sum(fitted_receiver_components**2, axis=1))
        )
        receiver_relative_kinetic_energy_change = abs(
            receiver_energy_after - receiver_energy_before
        ) / max(receiver_energy_before, np.finfo(float).eps)
        if receiver_localization_profile is not None:
            local = receiver_localization_profile >= np.exp(-0.5)
            correction_energy = receiver_weights * np.sum(receiver_correction**2, axis=1)
            receiver_local_correction_energy_fraction = float(
                np.sum(correction_energy[local])
                / max(float(np.sum(correction_energy)), np.finfo(float).eps)
            )
    bottom_streamfunction = np.asarray(
        bottom_interpolation @ solution.solution
    ).ravel()
    source_valid = np.asarray(data["velocity_valid"], dtype=bool) & cell_valid
    source_u = np.asarray(data["velocity_x"], dtype=float)
    source_w = np.asarray(data["velocity_z"], dtype=float)
    cell_target_scale = max(
        float(np.linalg.norm(np.concatenate((source_u[source_valid], source_w[source_valid])))),
        np.finfo(float).eps,
    )
    cell_velocity_error = float(
        np.linalg.norm(
            np.concatenate(
                (
                    cell_u[source_valid] - source_u[source_valid],
                    cell_w[source_valid] - source_w[source_valid],
                )
            )
        )
        / cell_target_scale
    )
    report: dict[str, object] = {
        "schema": "bie-to-vof-mac-face-projection-v1",
        "source": str(input_path),
        "backend": backend,
        "column_equilibration": True,
        "solver_tolerance": tolerance,
        "maximum_solver_iterations": maximum_iterations,
        "bottom_streamfunction_weight": bottom_streamfunction_weight,
        "momentum_constraint_weight": momentum_constraint_weight,
        "surface_velocity_weight": surface_velocity_weight,
        "exterior_energy_weight": exterior_energy_weight,
        "harmonic_weight": harmonic_weight,
        "least_squares_stop_code": solution.stop_code,
        "least_squares_iterations": solution.iterations,
        "least_squares_relative_residual": float(
            solution.residual_norm / max(float(np.linalg.norm(rhs)), 1.0)
        ),
        "solve_seconds": elapsed,
        "valid_vertical_face_count": int(len(vertical_indices)),
        "valid_horizontal_face_count": int(len(horizontal_indices)),
        "valid_flux_cell_count": int(np.count_nonzero(cell_valid)),
        "relative_face_flux_fit_error": face_fit_error,
        "relative_cell_center_velocity_error": cell_velocity_error,
        "relative_surface_velocity_error": relative_surface_velocity_error,
        "maximum_liquid_to_surface_q99_speed_ratio": float(
            np.max(cell_speed[fraction > 0.01])
            / max(surface_speed_q99, np.finfo(float).eps)
        ),
        "relative_momentum_error": relative_momentum_error,
        "momentum_constraint_quadrature": (
            "receiver" if receiver_operator is not None else "source"
        ),
        "receiver_quadrature": (
            str(receiver_quadrature_path) if receiver_quadrature_path is not None else None
        ),
        "receiver_quadrature_measure_schema": receiver_measure_schema,
        "receiver_grid_level": receiver_grid_level,
        "receiver_quadrature_cell_count": (
            int(len(receiver_weights)) if receiver_weights is not None else None
        ),
        "reconstructed_receiver_momentum": (
            reconstructed_receiver_momentum.tolist()
            if reconstructed_receiver_momentum is not None
            else None
        ),
        "relative_receiver_momentum_error": relative_receiver_momentum_error,
        "relative_change_from_exported_receiver_velocity": receiver_velocity_fit_error,
        "receiver_weighted_relative_velocity_change": receiver_weighted_velocity_change,
        "receiver_correction_norm_ratio_to_global_lower_bound": receiver_correction_lower_bound_ratio,
        "receiver_relative_kinetic_energy_change": receiver_relative_kinetic_energy_change,
        "receiver_velocity_preservation_weight": receiver_velocity_preservation_weight,
        "receiver_localization_sigma": receiver_localization_sigma,
        "receiver_localization_floor": receiver_localization_floor,
        "receiver_local_correction_energy_fraction_within_one_sigma": (
            receiver_local_correction_energy_fraction
        ),
        "bottom_streamfunction_standard_deviation": float(
            np.std(bottom_streamfunction)
        ),
        "bottom_streamfunction_range": float(np.ptp(bottom_streamfunction)),
        "projected_face_divergence_l2": float(np.sqrt(np.mean(divergence[cell_valid] ** 2))),
        "projected_face_divergence_linf": float(np.max(np.abs(divergence[cell_valid]))),
    }
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    bottom_x = np.asarray(data["bottom_x"], dtype=float)
    bottom_z = np.asarray(data["bottom_z"], dtype=float)
    surface_x = np.asarray(data["surface_x"], dtype=float)
    surface_z = np.asarray(data["surface_z"], dtype=float)
    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        grid_x=x,
        grid_z=z,
        volume_fraction=fraction,
        mac_streamfunction=streamfunction,
        face_velocity_x_projected=vertical,
        face_velocity_z_projected=horizontal,
        face_velocity_x_valid=vertical_valid,
        face_velocity_z_valid=horizontal_valid,
        cell_velocity_x_projected=cell_u,
        cell_velocity_z_projected=cell_w,
        bottom_x=bottom_x,
        bottom_z=bottom_z,
        surface_x=surface_x,
        surface_z=surface_z,
    )
    header = output_prefix.with_suffix(".h")
    header_lines = [
        "/* Generated by project_mac_face_flux.py; do not edit manually. */",
        f"#define HANDOFF_NX {nx}",
        f"#define HANDOFF_NZ {nz}",
        f"#define HANDOFF_NB {len(bottom_x)}",
        f"#define HANDOFF_NS {len(surface_x)}",
        "#define HANDOFF_HAS_STREAMFUNCTION 0",
        "#define HANDOFF_HAS_MAC_STREAMFUNCTION 1",
        f"#define HANDOFF_MOMENTUM_EMBEDDED {1 if receiver_operator is not None else 0}",
        f"#define HANDOFF_RECEIVER_LEVEL {receiver_grid_level if receiver_grid_level is not None else 0}",
        f"#define HANDOFF_DOMAIN_LENGTH {dx * nx:.17g}",
        f"#define HANDOFF_VERTICAL_ORIGIN {z[0] - 0.5 * dz:.17g}",
        f"static const double handoff_x0 = {x[0]:.17g};",
        f"static const double handoff_z0 = {z[0]:.17g};",
        f"static const double handoff_dx = {dx:.17g};",
        f"static const double handoff_dz = {dz:.17g};",
        f"static const double handoff_target_volume = {float(data['source_volume']):.17g};",
        f"static const double handoff_target_momentum_x = {target_momentum[0]:.17g};",
        f"static const double handoff_target_momentum_z = {target_momentum[1]:.17g};",
        "static const double handoff_velocity_shift_x = 0.;",
        "static const double handoff_velocity_shift_z = 0.;",
    ]
    append_c_array(header_lines, "handoff_fraction", fraction)
    append_c_array(header_lines, "handoff_u", cell_u)
    append_c_array(header_lines, "handoff_w", cell_w)
    append_c_array(header_lines, "handoff_bottom_x", bottom_x)
    append_c_array(header_lines, "handoff_bottom_z", bottom_z)
    append_c_array(header_lines, "handoff_surface_x", surface_x)
    append_c_array(header_lines, "handoff_surface_z", surface_z)
    append_c_array(header_lines, "handoff_mac_streamfunction", streamfunction)
    header.write_text("\n".join(header_lines) + "\n", encoding="ascii")
    report["generated_header"] = str(header)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument(
        "--linear-backend",
        choices=("scipy_lsqr", "cupy_lsqr", "cupy_lsmr"),
        default="cupy_lsqr",
    )
    parser.add_argument("--tolerance", type=float, default=2.0e-11)
    parser.add_argument("--maximum-iterations", type=int, default=20000)
    parser.add_argument("--bottom-streamfunction-weight", type=float, default=1.0e6)
    parser.add_argument("--momentum-constraint-weight", type=float, default=1.0e6)
    parser.add_argument("--surface-velocity-weight", type=float, default=32.0)
    parser.add_argument("--exterior-energy-weight", type=float, default=0.005)
    parser.add_argument("--harmonic-weight", type=float, default=0.02)
    parser.add_argument(
        "--receiver-quadrature",
        type=Path,
        help="CSV exported by bie_handoff_receiver.c; use its q*cell_area weights",
    )
    parser.add_argument("--receiver-velocity-preservation-weight", type=float, default=0.0)
    parser.add_argument("--receiver-localization-sigma", type=float)
    parser.add_argument("--receiver-localization-floor", type=float, default=1.0e-3)
    args = parser.parse_args()
    print(
        json.dumps(
            project(
                args.input,
                args.output_prefix,
                args.linear_backend,
                args.tolerance,
                args.maximum_iterations,
                args.bottom_streamfunction_weight,
                args.momentum_constraint_weight,
                args.surface_velocity_weight,
                args.exterior_energy_weight,
                args.harmonic_weight,
                args.receiver_quadrature,
                args.receiver_velocity_preservation_weight,
                args.receiver_localization_sigma,
                args.receiver_localization_floor,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
