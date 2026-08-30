"""Periodic geometric event detection for an overturning free surface."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ImpactDiagnostics:
    minimum_distance: float
    normalized_distance: float
    segment_pair: tuple[int, int] | None
    self_intersection: bool
    arc_separation: float = float("inf")
    normalized_arc_separation: float = float("inf")


def _point_segment_distance(point: Array, first: Array, second: Array) -> float:
    direction = second - first
    denominator = float(direction @ direction)
    if denominator <= np.finfo(float).eps:
        return float(np.linalg.norm(point - first))
    fraction = float((point - first) @ direction / denominator)
    projection = first + np.clip(fraction, 0.0, 1.0) * direction
    return float(np.linalg.norm(point - projection))


def _orientation(first: Array, second: Array, third: Array) -> float:
    first_direction = second - first
    second_direction = third - first
    return float(
        first_direction[0] * second_direction[1]
        - first_direction[1] * second_direction[0]
    )


def _segments_intersect(a: Array, b: Array, c: Array, d: Array) -> bool:
    scale = max(float(np.linalg.norm(b - a)), float(np.linalg.norm(d - c)), 1.0)
    tolerance = 64.0 * np.finfo(float).eps * scale * scale
    values = (
        _orientation(a, b, c),
        _orientation(a, b, d),
        _orientation(c, d, a),
        _orientation(c, d, b),
    )
    if values[0] * values[1] < -tolerance and values[2] * values[3] < -tolerance:
        return True
    if abs(values[0]) <= tolerance and _point_segment_distance(c, a, b) <= math.sqrt(tolerance):
        return True
    if abs(values[1]) <= tolerance and _point_segment_distance(d, a, b) <= math.sqrt(tolerance):
        return True
    if abs(values[2]) <= tolerance and _point_segment_distance(a, c, d) <= math.sqrt(tolerance):
        return True
    if abs(values[3]) <= tolerance and _point_segment_distance(b, c, d) <= math.sqrt(tolerance):
        return True
    return False


def _segment_distance(a: Array, b: Array, c: Array, d: Array) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance(a, c, d),
        _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b),
        _point_segment_distance(d, a, b),
    )


def periodic_impact_diagnostics(
    x: Array,
    z: Array,
    length: float,
    *,
    adjacent_exclusion: int = 2,
    minimum_arc_separation_panels: float | None = None,
) -> ImpactDiagnostics:
    """Measure the closest non-neighbouring surface panels in a periodic cell.

    The closing panel from marker ``N-1`` to marker ``0`` in the next periodic
    copy is included.  The normalized distance uses the median panel length,
    so a threshold can be stated in resolution-independent marker units.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    if x.ndim != 1 or x.shape != z.shape or len(x) < 2:
        raise ValueError("x and z must be matching surface vectors")
    if (
        length <= 0.0
        or adjacent_exclusion < 1
        or (
            minimum_arc_separation_panels is not None
            and minimum_arc_separation_panels <= 0.0
        )
    ):
        raise ValueError("invalid periodic impact parameters")
    starts = np.column_stack((x, z))
    ends = np.roll(starts, -1, axis=0)
    ends[-1, 0] += length
    panel_lengths = np.linalg.norm(ends - starts, axis=1)
    spacing = max(float(np.median(panel_lengths)), np.finfo(float).eps)
    n = len(x)
    first_start = starts[:, None, :]
    first_end = ends[:, None, :]
    midpoint_x = 0.5 * (starts[:, 0] + ends[:, 0])
    shifts = length * np.round(
        (midpoint_x[:, None] - midpoint_x[None, :]) / length
    )
    translation = np.stack((shifts, np.zeros_like(shifts)), axis=2)
    second_start = starts[None, :, :] + translation
    second_end = ends[None, :, :] + translation

    def cross(first_vector: Array, second_vector: Array) -> Array:
        return (
            first_vector[..., 0] * second_vector[..., 1]
            - first_vector[..., 1] * second_vector[..., 0]
        )

    def point_to_segments(point: Array, segment_start: Array, segment_end: Array) -> Array:
        direction = segment_end - segment_start
        denominator = np.sum(direction * direction, axis=2)
        fraction = np.sum((point - segment_start) * direction, axis=2) / np.maximum(
            denominator, np.finfo(float).eps
        )
        projection = segment_start + np.clip(fraction, 0.0, 1.0)[..., None] * direction
        return np.linalg.norm(point - projection, axis=2)

    ab = first_end - first_start
    cd = second_end - second_start
    orientations = (
        cross(ab, second_start - first_start),
        cross(ab, second_end - first_start),
        cross(cd, first_start - second_start),
        cross(cd, first_end - second_start),
    )
    geometric_scale = np.maximum(
        np.maximum(np.linalg.norm(ab, axis=2), np.linalg.norm(cd, axis=2)),
        1.0,
    )
    tolerance = 64.0 * np.finfo(float).eps * geometric_scale**2
    strict_intersection = (
        orientations[0] * orientations[1] < -tolerance
    ) & (
        orientations[2] * orientations[3] < -tolerance
    )
    distances = np.minimum.reduce(
        (
            point_to_segments(first_start, second_start, second_end),
            point_to_segments(first_end, second_start, second_end),
            point_to_segments(second_start, first_start, first_end),
            point_to_segments(second_end, first_start, first_end),
        )
    )
    distances[strict_intersection] = 0.0
    indices = np.arange(n)
    separation = np.abs(indices[:, None] - indices[None, :])
    separation = np.minimum(separation, n - separation)
    excluded = (separation <= adjacent_exclusion) | (
        indices[None, :] <= indices[:, None]
    )
    cumulative = np.concatenate(([0.0], np.cumsum(panel_lengths)))
    midpoint_arclength = cumulative[:-1] + 0.5 * panel_lengths
    arc_separation = np.abs(
        midpoint_arclength[:, None] - midpoint_arclength[None, :]
    )
    total_arclength = float(cumulative[-1])
    arc_separation = np.minimum(
        arc_separation, total_arclength - arc_separation
    )
    if minimum_arc_separation_panels is not None:
        excluded |= (
            arc_separation < minimum_arc_separation_panels * spacing
        )
    distances[excluded] = np.inf
    flat_index = int(np.argmin(distances))
    minimum = float(distances.flat[flat_index])
    if math.isfinite(minimum):
        pair = tuple(int(value) for value in np.unravel_index(flat_index, distances.shape))
    else:
        pair = None
    selected_arc_separation = (
        float(arc_separation[pair]) if pair is not None else float("inf")
    )
    return ImpactDiagnostics(
        minimum_distance=float(minimum),
        normalized_distance=float(minimum / spacing),
        segment_pair=pair,
        self_intersection=bool(minimum == 0.0),
        arc_separation=selected_arc_separation,
        normalized_arc_separation=float(selected_arc_separation / spacing),
    )


def interface_self_intersects(x: Array, z: Array, length: float | None = None) -> bool:
    if length is None:
        x = np.asarray(x, dtype=float)
        length = float(np.max(x) - np.min(x))
        if length <= 0.0:
            return False
    exclusion = 1 if len(np.asarray(x)) <= 5 else 2
    return periodic_impact_diagnostics(
        x, z, length, adjacent_exclusion=exclusion
    ).self_intersection
