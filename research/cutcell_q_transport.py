"""Reference conservative transport of liquid volume in embedded cells.

This is deliberately a small, geometry-first verifier.  It is not a drop-in
replacement for Basilisk's multidimensional VOF sweep.  A moving polygonal
liquid patch is intersected with a fixed half-plane solid, and a single shared
space-time face flux updates the physical full-cell fraction ``q``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


Point = tuple[float, float]


def clip_half_plane(polygon: list[Point], normal: Point, alpha: float) -> list[Point]:
    """Clip a convex polygon to ``normal dot x <= alpha``."""
    if not polygon:
        return []
    nx, ny = normal

    def level(point: Point) -> float:
        return nx * point[0] + ny * point[1] - alpha

    clipped: list[Point] = []
    previous = polygon[-1]
    previous_level = level(previous)
    for current in polygon:
        current_level = level(current)
        previous_inside = previous_level <= 1.0e-14
        current_inside = current_level <= 1.0e-14
        if previous_inside != current_inside:
            denominator = previous_level - current_level
            weight = previous_level / denominator
            clipped.append(
                (
                    previous[0] + weight * (current[0] - previous[0]),
                    previous[1] + weight * (current[1] - previous[1]),
                )
            )
        if current_inside:
            clipped.append(current)
        previous = current
        previous_level = current_level
    return clipped


def polygon_area(polygon: list[Point]) -> float:
    if len(polygon) < 3:
        return 0.0
    coordinates = np.asarray(polygon, dtype=float)
    return 0.5 * abs(
        float(
            np.dot(coordinates[:, 0], np.roll(coordinates[:, 1], -1))
            - np.dot(coordinates[:, 1], np.roll(coordinates[:, 0], -1))
        )
    )


def rectangle(x0: float, x1: float, y0: float, y1: float) -> list[Point]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def clipped_area(cell: list[Point], half_planes: list[tuple[Point, float]]) -> float:
    polygon = cell
    for normal, alpha in half_planes:
        polygon = clip_half_plane(polygon, normal, alpha)
    return polygon_area(polygon)


@dataclass(frozen=True)
class Geometry:
    wall_slope: float = 0.35
    wall_intercept: float = 0.18
    patch_x_min: float = 0.24
    patch_x_max: float = 0.58
    patch_wall_min: float = -0.035
    patch_wall_max: float = 0.24
    velocity_x: float = 0.25

    @property
    def velocity_y(self) -> float:
        return self.wall_slope * self.velocity_x

    def fluid_planes(self) -> list[tuple[Point, float]]:
        # y >= m*x+b <=> m*x-y <= -b.
        return [((self.wall_slope, -1.0), -self.wall_intercept)]

    def liquid_planes(self, time: float) -> list[tuple[Point, float]]:
        left = self.patch_x_min + self.velocity_x * time
        right = self.patch_x_max + self.velocity_x * time
        # r=y-m*x-b is invariant under the selected tangent translation.
        return [
            ((-1.0, 0.0), -left),
            ((1.0, 0.0), right),
            (
                (self.wall_slope, -1.0),
                -(self.wall_intercept + self.patch_wall_min),
            ),
            (
                (-self.wall_slope, 1.0),
                self.wall_intercept + self.patch_wall_max,
            ),
        ]


def cell_fractions(n: int, geometry: Geometry, time: float) -> tuple[np.ndarray, ...]:
    dx = 1.0 / n
    full_liquid = np.zeros((n, n))
    solid_open = np.zeros((n, n))
    physical_q = np.zeros((n, n))
    fluid = geometry.fluid_planes()
    liquid = geometry.liquid_planes(time)
    for j in range(n):
        for i in range(n):
            cell = rectangle(i * dx, (i + 1) * dx, j * dx, (j + 1) * dx)
            full_liquid[j, i] = clipped_area(cell, liquid) / dx**2
            solid_open[j, i] = clipped_area(cell, fluid) / dx**2
            physical_q[j, i] = clipped_area(cell, liquid + fluid) / dx**2
    conditional_f = np.divide(
        physical_q,
        solid_open,
        out=np.zeros_like(physical_q),
        where=solid_open > 1.0e-14,
    )
    return full_liquid, solid_open, physical_q, conditional_f


def _linear_value(line: tuple[float, float], time: float) -> float:
    return line[0] * time + line[1]


def integrate_interval_overlap(
    lower_lines: list[tuple[float, float]],
    upper_lines: list[tuple[float, float]],
    duration: float,
) -> float:
    """Integrate max(0, min(upper)-max(lower)) for affine endpoints."""
    lines = lower_lines + upper_lines
    breakpoints = {0.0, duration}
    for first in range(len(lines)):
        for second in range(first + 1, len(lines)):
            slope = lines[first][0] - lines[second][0]
            if abs(slope) <= 1.0e-15:
                continue
            crossing = (lines[second][1] - lines[first][1]) / slope
            if 0.0 < crossing < duration:
                breakpoints.add(float(crossing))
    ordered = sorted(breakpoints)
    integral = 0.0
    for start, end in zip(ordered[:-1], ordered[1:]):
        midpoint = 0.5 * (start + end)
        lower = max(lower_lines, key=lambda line: _linear_value(line, midpoint))
        upper = min(upper_lines, key=lambda line: _linear_value(line, midpoint))
        difference = (upper[0] - lower[0], upper[1] - lower[1])
        root = None
        if abs(difference[0]) > 1.0e-15:
            candidate = -difference[1] / difference[0]
            if start < candidate < end:
                root = float(candidate)
        subintervals = [start, end] if root is None else [start, root, end]
        for left, right in zip(subintervals[:-1], subintervals[1:]):
            middle = 0.5 * (left + right)
            if _linear_value(difference, middle) > 0.0:
                integral += (
                    0.5 * difference[0] * (right**2 - left**2)
                    + difference[1] * (right - left)
                )
    return integral


def _positive_x_face_fluxes(
    n: int, geometry: Geometry, duration: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact double-intersection and independent-product x fluxes."""
    dx = 1.0 / n
    exact = np.zeros((n, n + 1))
    product = np.zeros_like(exact)
    m = geometry.wall_slope
    b = geometry.wall_intercept
    u = geometry.velocity_x
    for j in range(n):
        y0, y1 = j * dx, (j + 1) * dx
        for face in range(1, n):
            x = face * dx
            time_low = max(0.0, (x - geometry.patch_x_max) / u)
            time_high = min(duration, (x - geometry.patch_x_min) / u)
            active_duration = max(0.0, time_high - time_low)
            if active_duration == 0.0:
                continue
            liquid_low = max(y0, m * x + b + geometry.patch_wall_min)
            liquid_high = min(y1, m * x + b + geometry.patch_wall_max)
            fluid_low = max(y0, m * x + b)
            liquid_length = max(0.0, liquid_high - liquid_low)
            fluid_length = max(0.0, y1 - fluid_low)
            double_length = max(0.0, liquid_high - max(liquid_low, fluid_low))
            exact[j, face] = u * active_duration * double_length
            product[j, face] = (
                u * active_duration * liquid_length * fluid_length / dx
            )
    return exact, product


def _positive_y_face_fluxes(
    n: int, geometry: Geometry, duration: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact double-intersection and independent-product y fluxes."""
    dx = 1.0 / n
    exact = np.zeros((n + 1, n))
    product = np.zeros_like(exact)
    m = geometry.wall_slope
    b = geometry.wall_intercept
    u = geometry.velocity_x
    v = geometry.velocity_y
    for face in range(1, n):
        y = face * dx
        fluid_upper = (y - b) / m
        liquid_wall_lower = (y - b - geometry.patch_wall_max) / m
        liquid_wall_upper = (y - b - geometry.patch_wall_min) / m
        for i in range(n):
            x0, x1 = i * dx, (i + 1) * dx
            liquid_lower = [(0.0, x0), (u, geometry.patch_x_min), (0.0, liquid_wall_lower)]
            liquid_upper = [(0.0, x1), (u, geometry.patch_x_max), (0.0, liquid_wall_upper)]
            liquid_integral = integrate_interval_overlap(
                liquid_lower, liquid_upper, duration
            )
            fluid_length = max(0.0, min(x1, fluid_upper) - x0)
            double_integral = integrate_interval_overlap(
                liquid_lower,
                liquid_upper + [(0.0, fluid_upper)],
                duration,
            )
            exact[face, i] = v * double_integral
            product[face, i] = v * liquid_integral * fluid_length / dx
    return exact, product


def divergence_update(values: np.ndarray, flux_x: np.ndarray, flux_y: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    cell_area = (1.0 / n) ** 2
    return values + (
        flux_x[:, :-1]
        - flux_x[:, 1:]
        + flux_y[:-1, :]
        - flux_y[1:, :]
    ) / cell_area


@dataclass
class ResolutionResult:
    resolution: int
    dt: float
    product_geometry_l1: float
    product_transport_l1: float
    conservative_transport_l1: float
    product_relative_volume_change: float
    conservative_relative_volume_change: float
    conservative_lower_bound_violation: float
    conservative_upper_bound_violation: float


def run_resolution(n: int, cfl: float, geometry: Geometry) -> tuple[ResolutionResult, dict[str, np.ndarray]]:
    dx = 1.0 / n
    dt = cfl * min(dx / geometry.velocity_x, dx / geometry.velocity_y)
    f_full, cs, q_initial, conditional_f = cell_fractions(n, geometry, 0.0)
    _, _, q_target, _ = cell_fractions(n, geometry, dt)
    exact_x, product_x = _positive_x_face_fluxes(n, geometry, dt)
    exact_y, product_y = _positive_y_face_fluxes(n, geometry, dt)
    q_conservative = divergence_update(q_initial, exact_x, exact_y)
    f_product_updated = divergence_update(f_full, product_x, product_y)
    q_product_updated = f_product_updated * cs
    area = dx**2
    mass_initial = float(np.sum(q_initial) * area)
    mass_product_initial = float(np.sum(f_full * cs) * area)
    mass_product_final = float(np.sum(q_product_updated) * area)
    mass_conservative_final = float(np.sum(q_conservative) * area)

    def relative_l1(field: np.ndarray) -> float:
        return float(np.sum(np.abs(field - q_target)) * area / mass_initial)

    result = ResolutionResult(
        resolution=n,
        dt=dt,
        product_geometry_l1=float(
            np.sum(np.abs(f_full * cs - q_initial)) * area / mass_initial
        ),
        product_transport_l1=relative_l1(q_product_updated),
        conservative_transport_l1=relative_l1(q_conservative),
        product_relative_volume_change=(mass_product_final - mass_product_initial)
        / mass_product_initial,
        conservative_relative_volume_change=(mass_conservative_final - mass_initial)
        / mass_initial,
        conservative_lower_bound_violation=max(0.0, -float(np.min(q_conservative))),
        conservative_upper_bound_violation=max(
            0.0, float(np.max(q_conservative - cs))
        ),
    )
    fields = {
        "f_full": f_full,
        "cs": cs,
        "q_initial": q_initial,
        "conditional_f": conditional_f,
        "q_target": q_target,
        "q_product_updated": q_product_updated,
        "q_conservative": q_conservative,
    }
    return result, fields


def create_figure(results: list[ResolutionResult], fields: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(12.0, 6.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 0.9))
    shown = [
        ("q exact at $t+\\Delta t$", fields["q_target"]),
        ("independent $f c_s$ update", fields["q_product_updated"]),
        ("shared double-PLIC $q$ flux", fields["q_conservative"]),
    ]
    for column, (title, field) in enumerate(shown):
        axis = figure.add_subplot(grid[0, column])
        image = axis.imshow(field, origin="lower", extent=(0, 1, 0, 1), vmin=0, vmax=1, cmap="Blues")
        axis.set_title(title)
        axis.set_xlabel("x")
        if column == 0:
            axis.set_ylabel("y")
        figure.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
    resolutions = np.asarray([item.resolution for item in results])
    error_axis = figure.add_subplot(grid[1, :2])
    error_axis.loglog(
        resolutions,
        [item.product_transport_l1 for item in results],
        "o-",
        label="independent product",
    )
    error_axis.loglog(
        resolutions,
        [item.conservative_transport_l1 for item in results],
        "s-",
        label="double-PLIC q",
    )
    error_axis.set_xlabel("cells per direction")
    error_axis.set_ylabel("relative $L_1$ geometry error")
    error_axis.grid(True, which="both", alpha=0.25)
    error_axis.legend()
    volume_axis = figure.add_subplot(grid[1, 2])
    volume_axis.semilogy(
        resolutions,
        np.maximum(
            np.abs([item.product_relative_volume_change for item in results]),
            1.0e-18,
        ),
        "o-",
        label="independent product",
    )
    volume_axis.semilogy(
        resolutions,
        np.maximum(
            np.abs([item.conservative_relative_volume_change for item in results]),
            1.0e-18,
        ),
        "s-",
        label="double-PLIC q",
    )
    volume_axis.axhline(1.0e-12, color="black", linewidth=0.9, linestyle="--", label="gate")
    volume_axis.set_xlabel("cells per direction")
    volume_axis.set_ylabel("relative volume change")
    volume_axis.grid(True, which="both", alpha=0.25)
    volume_axis.legend(fontsize=8)
    figure.suptitle("Kinematic embedded-boundary transport audit (one exact space-time step)")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("research/results/q_transport/q_transport_audit"),
    )
    args = parser.parse_args()
    geometry = Geometry()
    results: list[ResolutionResult] = []
    finest_fields: dict[str, np.ndarray] | None = None
    for resolution in args.resolutions:
        result, fields = run_resolution(resolution, args.cfl, geometry)
        results.append(result)
        finest_fields = fields
    assert finest_fields is not None
    payload = {
        "schema": "cutcell-double-plic-q-transport-v1",
        "status": "kinematic-reference-only",
        "geometry": asdict(geometry),
        "cfl": args.cfl,
        "results": [asdict(item) for item in results],
        "gates": {
            "relative_volume_change": 1.0e-12,
            "bound_violation": 1.0e-12,
            "piecewise_linear_manufactured_error": 1.0e-12,
        },
        "limitations": [
            "one fixed-grid kinematic space-time step; the planar case is reproduced exactly",
            "analytic planar embedded boundary and polygonal liquid patch",
            "no AMR, topology change, pressure projection, viscosity, or 3-D reconstruction",
        ],
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output_prefix.with_suffix(".npz"), **finest_fields)
    create_figure(results, finest_fields, args.output_prefix.with_suffix(".png"))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
