"""Reproduce and diagnose the unmodified legacy solver without its infinite loop.

The module deliberately loads definitions from ``main.py`` only up to the
top-level ``while True`` statement.  This keeps the historical implementation
intact while making a bounded, non-plotting experiment possible.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def load_legacy(path: Path) -> dict[str, Any]:
    """Execute imports, globals, classes and functions, but not the main loop."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.While):
            break
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(name.startswith(("matplotlib", "scipy")) for name in module_names):
                continue
        body.append(node)
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    def uniform_filter1d(values: np.ndarray, size: int) -> np.ndarray:
        """Small NumPy equivalent sufficient for the legacy smoothing experiment."""
        left = size // 2
        right = size - 1 - left
        padded = np.pad(np.asarray(values), (left, right), mode="reflect")
        return np.convolve(padded, np.ones(size) / size, mode="valid")

    namespace: dict[str, Any] = {
        "__name__": "legacy_bounded",
        "uniform_filter1d": uniform_filter1d,
        "matplotlib": SimpleNamespace(rc=lambda *args, **kwargs: None),
        "plt": SimpleNamespace(),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def kernel_derivative_check(ns: dict[str, Any]) -> dict[str, float]:
    """Compare the coded analytic gradient of the layer kernel with differences."""
    Vortex = ns["Vortex"]
    vortex = Vortex(0.17, -0.23, 1.0, perpend_x=0.6, perpend_y=0.8)
    vortexes = np.asarray([vortex], dtype=object)
    points = np.asarray(
        [[0.71, 0.31], [-0.42, 0.77], [1.23, -0.91], [-0.64, -0.55]],
        dtype=float,
    )
    coded_x, coded_y = ns["phi_"](vortexes, points)
    eps = 1.0e-6
    shift_x = np.asarray([eps, 0.0])
    shift_y = np.asarray([0.0, eps])
    fd_x = (
        ns["phi"](vortexes, points + shift_x)
        - ns["phi"](vortexes, points - shift_x)
    ) / (2.0 * eps)
    fd_y = (
        ns["phi"](vortexes, points + shift_y)
        - ns["phi"](vortexes, points - shift_y)
    ) / (2.0 * eps)

    def relative_error(value: np.ndarray, reference: np.ndarray) -> float:
        scale = np.maximum(np.abs(reference), 1.0e-14)
        return float(np.max(np.abs(value - reference) / scale))

    return {
        "max_relative_error_dx": relative_error(coded_x, fd_x),
        "max_relative_error_dy": relative_error(coded_y, fd_y),
        "max_absolute_error_dx": float(np.max(np.abs(coded_x - fd_x))),
        "max_absolute_error_dy": float(np.max(np.abs(coded_y - fd_y))),
    }


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def min_adjacent_spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        return float("nan")
    order = np.argsort(points[:, 0])
    return float(np.min(np.linalg.norm(np.diff(points[order], axis=0), axis=1)))


def matrix_metrics(matrix: np.ndarray) -> dict[str, float | int]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = singular_values[0] * max(matrix.shape) * np.finfo(float).eps
    return {
        "condition_number_2": float(singular_values[0] / singular_values[-1]),
        "rank": int(np.count_nonzero(singular_values > tolerance)),
        "sigma_max": float(singular_values[0]),
        "sigma_min": float(singular_values[-1]),
    }


def run_main_legacy(
    ns: dict[str, Any], steps: int, smooth: bool = True
) -> list[dict[str, float | int | bool]]:
    """Run the numerical loop from main.py with plotting removed."""
    vortexes = ns["init"]()
    history: list[dict[str, float | int | bool]] = []

    for step in range(steps):
        vortexes = ns["update_all_vortexes_merge_and_add"](vortexes)
        ns["changeVPerps"](vortexes)

        matrix = ns["Acoefs"](vortexes).T
        rhs = ns["Bcoefs"](vortexes)
        metrics = matrix_metrics(matrix)

        strengths = np.linalg.solve(matrix, rhs)
        masks = {
            name: np.asarray(ns["checkVTypes"](vortexes, type_id), dtype=bool)
            for name, type_id in (
                ("H", ns["TYPE_H"]),
                ("h1", ns["TYPE_h1"]),
                ("h2", ns["TYPE_h2"]),
                ("h3", ns["TYPE_h3"]),
            )
        }
        if smooth:
            uniform_filter1d = ns["uniform_filter1d"]
            for mask in masks.values():
                strengths[mask] = uniform_filter1d(strengths[mask], 5)
        ns["changeVGs"](vortexes, strengths)

        free = vortexes[masks["H"]]
        free_points = ns["mapVortexesPoints"](free)
        free_control_points = ns["mapControlPoints"](free)
        velocity_x, velocity_y = ns["Phi_mean_"](vortexes, free_control_points)
        # This matches main.py's sign convention exactly.
        velocity = -np.column_stack([velocity_x, velocity_y])
        moved_free_points = free_points + velocity * ns["dt"]
        ns["moveVortexes"](free, moved_free_points)

        # Match the legacy tangential redistribution of rigid-boundary markers.
        wall_1 = vortexes[masks["h1"]]
        cp_1 = ns["mapControlPoints"](wall_1)
        _, wall_1_vy = ns["Phi_mean_"](vortexes, cp_1)
        p_1 = ns["mapVortexesPoints"](wall_1)
        p_1[:, 1] -= wall_1_vy * ns["dt"]
        ns["moveVortexes"](wall_1, p_1)

        bottom = vortexes[masks["h2"]]
        cp_2 = ns["mapControlPoints"](bottom)
        bottom_vx, _ = ns["Phi_mean_"](vortexes, cp_2)
        p_2 = ns["mapVortexesPoints"](bottom)
        p_2[:, 0] -= bottom_vx * ns["dt"]
        ns["moveVortexes"](bottom, p_2)

        wall_3 = vortexes[masks["h3"]]
        cp_3 = ns["mapControlPoints"](wall_3)
        _, wall_3_vy = ns["Phi_mean_"](vortexes, cp_3)
        p_3 = ns["mapVortexesPoints"](wall_3)
        p_3[:, 1] -= wall_3_vy * ns["dt"]
        ns["moveVortexes"](wall_3, p_3)

        wall_1[0].y = moved_free_points[np.argmin(moved_free_points[:, 0]), 1]
        wall_3[-1].y = moved_free_points[np.argmax(moved_free_points[:, 0]), 1]

        all_points = ns["mapVortexesPoints"](vortexes)
        fluid_boundary = np.vstack(
            [
                moved_free_points[np.argsort(moved_free_points[:, 0])],
                ns["mapVortexesPoints"](wall_3)[::-1],
                ns["mapVortexesPoints"](bottom)[::-1],
                ns["mapVortexesPoints"](wall_1)[::-1],
            ]
        )
        record: dict[str, float | int | bool] = {
            "step": step,
            "time": float((step + 1) * ns["dt"]),
            "n_unknowns": int(len(vortexes)),
            **metrics,
            "max_abs_strength": float(np.max(np.abs(strengths))),
            "rms_strength": float(np.sqrt(np.mean(strengths**2))),
            "max_free_surface_speed": float(np.max(np.linalg.norm(velocity, axis=1))),
            "min_free_surface_spacing": min_adjacent_spacing(moved_free_points),
            "fluid_polygon_area": polygon_area(fluid_boundary),
            "bounding_box_area": polygon_area(
                np.asarray(
                    [
                        [np.min(all_points[:, 0]), np.min(all_points[:, 1])],
                        [np.max(all_points[:, 0]), np.min(all_points[:, 1])],
                        [np.max(all_points[:, 0]), np.max(all_points[:, 1])],
                        [np.min(all_points[:, 0]), np.max(all_points[:, 1])],
                    ]
                )
            ),
            "finite": bool(
                np.all(np.isfinite(strengths))
                and np.all(np.isfinite(moved_free_points))
            ),
        }
        history.append(record)
        if not record["finite"]:
            break

    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-file", type=Path, default=Path("main.py"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ns = load_legacy(args.legacy_file.resolve())
    result = {
        "legacy_file": str(args.legacy_file),
        "kernel_derivative_check": kernel_derivative_check(ns),
        "history": run_main_legacy(ns, args.steps, smooth=not args.no_smoothing),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
