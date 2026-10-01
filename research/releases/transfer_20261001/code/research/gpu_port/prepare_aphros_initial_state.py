"""Prepare a fixed-grid Aphros state from an accepted BIE handoff.

The production path intersects the closed BIE liquid polygon directly with
every target cell.  It therefore does not inherit the resolution of an
intermediate VOF raster and never repairs mass by spreading a residual over
unrelated mixed cells.  The legacy piecewise-constant raster remap is retained
only as an explicit ablation.  Velocity is obtained by differentiating the
same periodic MAC streamfunction on the target grid.  Aphros' origin is zero,
so physical vertical coordinates are shifted by the Basilisk vertical origin.
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from shapely import area as geometry_area
from shapely import box as geometry_box
from shapely import contains_xy, intersection
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.ops import unary_union

DOMAIN_LENGTH = 64.0
VERTICAL_ORIGIN = -1.0400178928822124
TARGET_VOLUME = 40.526278599447593
TARGET_MOMENTUM = np.array([1.3582959672850798, -0.0072546312766226927])
EMBEDDED_BOUNDARY_OFFSET = 1e-8


def overlap_weights(
    target_edges: np.ndarray,
    source_edges: np.ndarray,
    *,
    periodic: bool = False,
) -> np.ndarray:
    """Return target-width-normalized overlaps with source cells."""

    nt = target_edges.size - 1
    ns = source_edges.size - 1
    result = np.zeros((nt, ns), dtype=np.float64)
    period = source_edges[-1] - source_edges[0]
    for ti, (left, right) in enumerate(pairwise(target_edges)):
        width = right - left
        shifts = (-period, 0.0, period) if periodic else (0.0,)
        for shift in shifts:
            lo = np.maximum(left, source_edges[:-1] + shift)
            hi = np.minimum(right, source_edges[1:] + shift)
            result[ti] += np.maximum(0.0, hi - lo) / width
    return result


def periodic_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    xp0 = xp[0]
    wrapped = (x - xp0) % DOMAIN_LENGTH + xp0
    return np.interp(
        wrapped,
        np.r_[xp, xp[0] + DOMAIN_LENGTH],
        np.r_[fp, fp[0]],
    )


def periodic_liquid_geometry(
    surface_x: np.ndarray,
    surface_z: np.ndarray,
    bottom_x: np.ndarray,
    bottom_z: np.ndarray,
    *,
    y_min: float,
    y_max: float,
):
    """Return one clipped period of the exact piecewise-linear liquid polygon."""

    start_x = float(surface_x[0])
    end_x = start_x + DOMAIN_LENGTH
    top_x = np.r_[surface_x, end_x]
    top_z = np.r_[surface_z, surface_z[0]]
    # Include bed markers from neighbouring periods when the surface seam is
    # shifted; otherwise a seam-spanning segment becomes a longer chord.
    first_period = int(np.floor((start_x - bottom_x[0]) / DOMAIN_LENGTH)) - 1
    last_period = int(np.ceil((end_x - bottom_x[0]) / DOMAIN_LENGTH)) + 1
    periodic_bed_x = np.sort(
        np.concatenate(
            [
                bottom_x + period * DOMAIN_LENGTH
                for period in range(first_period, last_period + 1)
            ]
        )
    )
    interior = (periodic_bed_x > start_x) & (periodic_bed_x < end_x)
    bed_x = np.r_[start_x, periodic_bed_x[interior], end_x]
    bed_z = periodic_interp(bed_x, bottom_x, bottom_z)
    polygon = Polygon(
        np.column_stack(
            (
                np.r_[top_x, bed_x[::-1]],
                np.r_[top_z, bed_z[::-1]],
            )
        )
    )
    if not polygon.is_valid:
        raise ValueError("BIE liquid polygon is self-intersecting or otherwise invalid")
    period_copies = unary_union(
        [
            translate(polygon, xoff=shift)
            for shift in (-DOMAIN_LENGTH, 0.0, DOMAIN_LENGTH)
        ]
    )
    domain = geometry_box(0.0, y_min, DOMAIN_LENGTH, y_max)
    result = period_copies.intersection(domain)
    if result.is_empty or not result.is_valid:
        raise ValueError("clipped periodic liquid geometry is empty or invalid")
    return result


def _mark_geometry_boundary_cells(
    geometry,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> np.ndarray:
    """Return cells which may be cut by any exact polygon boundary segment."""

    nx = x_edges.size - 1
    ny = y_edges.size - 1
    dx = float(x_edges[1] - x_edges[0])
    dy = float(y_edges[1] - y_edges[0])
    marked = np.zeros((ny, nx), dtype=bool)
    polygons = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
    for polygon in polygons:
        rings = [polygon.exterior, *polygon.interiors]
        for ring in rings:
            coordinates = np.asarray(ring.coords, dtype=np.float64)
            for first, second in pairwise(coordinates):
                ix0 = max(0, int(np.floor(min(first[0], second[0]) / dx)) - 1)
                ix1 = min(
                    nx - 1,
                    int(np.floor(max(first[0], second[0]) / dx)) + 1,
                )
                iy0 = max(
                    0,
                    int(np.floor((min(first[1], second[1]) - y_edges[0]) / dy)) - 1,
                )
                iy1 = min(
                    ny - 1,
                    int(np.floor((max(first[1], second[1]) - y_edges[0]) / dy)) + 1,
                )
                if ix1 >= ix0 and iy1 >= iy0:
                    marked[iy0 : iy1 + 1, ix0 : ix1 + 1] = True
    return marked


def direct_liquid_aperture(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    surface_x: np.ndarray,
    surface_z: np.ndarray,
    bottom_x: np.ndarray,
    bottom_z: np.ndarray,
    *,
    target_volume: float,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Intersect the physical BIE polygon directly with target-grid cells."""

    uncorrected_geometry = periodic_liquid_geometry(
        surface_x,
        surface_z,
        bottom_x,
        bottom_z,
        y_min=float(y_edges[0]),
        y_max=float(y_edges[-1]),
    )
    uncorrected_area = float(uncorrected_geometry.area)
    # The BIE conserves its spectral/trapezoidal volume, while the receiver
    # consumes a piecewise-linear polygon.  Reconcile these two continuum
    # measures once by a resolution-independent uniform free-surface shift.
    # This is a geometric gauge correction, not a cellwise mass repair.
    surface_shift = (target_volume - uncorrected_area) / DOMAIN_LENGTH
    geometry = periodic_liquid_geometry(
        surface_x,
        surface_z + surface_shift,
        bottom_x,
        bottom_z,
        y_min=float(y_edges[0]),
        y_max=float(y_edges[-1]),
    )
    q, quadrature_report = geometry_aperture(geometry, x_edges, y_edges)
    return q, {
        "geometry_initialization": "direct_piecewise_linear_polygon_intersection",
        "uncorrected_piecewise_linear_polygon_area": uncorrected_area,
        "volume_gauge_surface_shift": surface_shift,
        **quadrature_report,
    }


def geometry_aperture(
    geometry,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Return exact cell apertures for an already-clipped polygonal geometry."""

    dx = float(x_edges[1] - x_edges[0])
    dy = float(y_edges[1] - y_edges[0])
    cell_area = dx * dy
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    q = np.zeros((yc.size, xc.size), dtype=np.float64)
    # Centre classification fills cells that cannot meet the boundary.  Exact
    # vector intersections below replace every potentially cut cell, including
    # thin structures which do not contain a cell centre.
    rows_per_chunk = max(1, min(128, yc.size))
    for start in range(0, yc.size, rows_per_chunk):
        stop = min(yc.size, start + rows_per_chunk)
        xx, yy = np.meshgrid(xc, yc[start:stop])
        q[start:stop] = contains_xy(geometry, xx, yy)
    boundary = _mark_geometry_boundary_cells(geometry, x_edges, y_edges)
    iy, ix = np.nonzero(boundary)
    cells = geometry_box(
        x_edges[ix],
        y_edges[iy],
        x_edges[ix + 1],
        y_edges[iy + 1],
    )
    q[iy, ix] = np.asarray(geometry_area(intersection(cells, geometry))) / cell_area
    q = np.clip(q, 0.0, 1.0)
    represented_area = float(q.sum() * cell_area)
    exact_area = float(geometry.area)
    return q, {
        "exact_clipped_polygon_area": exact_area,
        "represented_polygon_area": represented_area,
        "relative_polygon_quadrature_error": abs(represented_area - exact_area)
        / max(exact_area, np.finfo(float).tiny),
        "exact_boundary_cell_count": int(np.count_nonzero(boundary)),
    }


def sample_mac_streamfunction(
    psi: np.ndarray,
    source_dx: float,
    source_dz: float,
    x: np.ndarray,
    z: np.ndarray,
    *,
    x_origin: float = 0.0,
    z_origin: float = VERTICAL_ORIGIN,
) -> np.ndarray:
    """Sample a periodic-x MAC streamfunction stored at grid vertices."""

    gx = ((x - x_origin) % DOMAIN_LENGTH) / source_dx
    gz = (z - z_origin) / source_dz
    i0raw = np.floor(gx).astype(np.int64)
    i0 = i0raw % psi.shape[1]
    i1 = (i0 + 1) % psi.shape[1]
    ax = gx - i0raw
    k0 = np.floor(gz).astype(np.int64)
    k0 = np.clip(k0, 0, psi.shape[0] - 2)
    k1 = k0 + 1
    az = np.clip(gz - k0, 0.0, 1.0)
    lower = psi[k0, i0] * (1.0 - ax) + psi[k0, i1] * ax
    upper = psi[k1, i0] * (1.0 - ax) + psi[k1, i1] * ax
    # Clamp vertically here. target_mac_velocity_state replaces the clamped
    # exterior rows by a periodic harmonic continuation. Returning zero would
    # create a gauge-dependent jump in psi and an O(1/dx) velocity sheet.
    return lower * (1.0 - az) + upper * az


def target_mac_velocity_state(
    psi: np.ndarray,
    source_dx: float,
    source_dz: float,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    source_x_origin: float = 0.0,
    source_z_origin: float = VERTICAL_ORIGIN,
) -> dict[str, np.ndarray | float]:
    """Restrict one streamfunction to a target MAC grid without breaking curl.

    ``fluxx``/``fluxxp`` and ``fluxy``/``fluxyp`` are normal face velocities,
    not integrated volume fluxes.  The patched Aphros receiver multiplies them
    by its own embedded-face areas.  Constructing all four arrays from one
    target-grid streamfunction makes the full-cell discrete divergence vanish
    algebraically and keeps opposite representations of shared faces identical.
    """

    dx = float(x_edges[1] - x_edges[0])
    dy = float(y_edges[1] - y_edges[0])
    if not np.isclose(dx, dy, rtol=1.0e-13, atol=1.0e-15):
        raise ValueError("Aphros target cells must be square")
    xx, yy = np.meshgrid(x_edges[:-1], y_edges)
    target_psi = sample_mac_streamfunction(
        psi,
        source_dx,
        source_dz,
        xx,
        yy,
        x_origin=source_x_origin,
        z_origin=source_z_origin,
    )
    source_z_max = source_z_origin + source_dz * (psi.shape[0] - 1)
    wave_number = 2.0 * np.pi * np.fft.rfftfreq(x_edges.size - 1, d=dx)
    # Outside the rectangular BIE projection, continue each boundary Fourier
    # mode as exp(-|k| distance). This is the unique bounded periodic harmonic
    # half-space extension. It preserves the gauge mode, incompressibility,
    # and bounded velocity without inventing an O(1/dx) cutoff layer.
    for row, y in enumerate(y_edges):
        distance = (
            source_z_origin - y
            if y < source_z_origin
            else y - source_z_max
            if y > source_z_max
            else 0.0
        )
        if distance > 0.0:
            spectrum = np.fft.rfft(target_psi[row])
            target_psi[row] = np.fft.irfft(
                spectrum * np.exp(-wave_number * distance),
                n=target_psi.shape[1],
            )
    fluxx = (target_psi[1:] - target_psi[:-1]) / dy
    fluxxp = np.roll(fluxx, -1, axis=1)
    horizontal = -(np.roll(target_psi, -1, axis=1) - target_psi) / dx
    fluxy = horizontal[:-1]
    fluxyp = horizontal[1:]
    vx = 0.5 * (fluxx + fluxxp)
    vy = 0.5 * (fluxy + fluxyp)
    divergence = (fluxxp - fluxx) / dx + (fluxyp - fluxy) / dy
    return {
        "vx": vx,
        "vy": vy,
        "fluxx": fluxx,
        "fluxxp": fluxxp,
        "fluxy": fluxy,
        "fluxyp": fluxyp,
        "fluxeb": np.zeros_like(vx),
        "target_streamfunction": target_psi,
        "vertical_extension": "periodic_harmonic_half_space",
        "rectangular_face_divergence_l2": float(
            np.sqrt(np.mean(divergence * divergence))
        ),
        "rectangular_face_divergence_linf": float(np.max(np.abs(divergence))),
        "shared_x_face_mismatch_linf": float(
            np.max(np.abs(fluxxp - np.roll(fluxx, -1, axis=1)))
        ),
        "shared_y_face_mismatch_linf": float(
            np.max(np.abs(fluxyp[:-1] - fluxy[1:])) if fluxyp.shape[0] > 1 else 0.0
        ),
    }


def fluid_aperture(
    x_edges: np.ndarray,
    y_edges_physical: np.ndarray,
    bottom_x: np.ndarray,
    bottom_z: np.ndarray,
) -> np.ndarray:
    """Integrate the fraction above a periodic piecewise-linear beach."""

    flat_surface = np.full_like(bottom_x, float(y_edges_physical[-1]))
    geometry = periodic_liquid_geometry(
        bottom_x,
        flat_surface,
        bottom_x,
        bottom_z,
        y_min=float(y_edges_physical[0]),
        y_max=float(y_edges_physical[-1]),
    )
    cs, _ = geometry_aperture(geometry, x_edges, y_edges_physical)
    return cs


def write_body_polygon(path: Path, bottom_x: np.ndarray, bottom_z: np.ndarray) -> None:
    xnodes_center = np.unique(np.r_[0.0, bottom_x, DOMAIN_LENGTH])
    # The x direction is periodic.  Put the polygon's vertical closing sides
    # one full period outside the simulated interval, otherwise a closing side
    # at x=0/64 creates degenerate cut faces exactly on the periodic seam.
    xnodes = np.unique(
        np.r_[
            xnodes_center - DOMAIN_LENGTH,
            xnodes_center,
            xnodes_center + DOMAIN_LENGTH,
        ]
    )
    # Bias the solid boundary upward by a negligible amount.  Several L11
    # vertices otherwise land on grid nodes to roundoff, producing zero-area
    # intersections and non-finite pressure coefficients in the cut-cell
    # operator.  The offset is 3.2e-7 of one L11 cell width.
    znodes = (
        periodic_interp(xnodes, bottom_x, bottom_z)
        - VERTICAL_ORIGIN
        + EMBEDDED_BOUNDARY_OFFSET
    )
    # Close the solid polygon strictly below the computational domain.  If its
    # base coincides with y=0, Aphros classifies zero level-set nodes as fluid
    # and creates a spurious disconnected one-cell channel below the beach.
    base_y = -DOMAIN_LENGTH
    points = [(-DOMAIN_LENGTH, base_y), (2 * DOMAIN_LENGTH, base_y)]
    points.extend(zip(xnodes[::-1], znodes[::-1]))
    points.append(points[0])
    coords = " ".join(f"{x:.17g} {y:.17g}" for x, y in points)
    path.write_text(f"polygon2 0 0 1 0 1 {coords}\n", encoding="utf-8")


def validate_projection_export(data) -> None:
    """Diagnostic iterates cannot silently become a physical receiver input.

    Legacy archives predate these markers and retain their separate campaign
    qualification path. New markers are mandatory once either is present.
    """
    keys = ("projection_diagnostic_only", "projection_solver_converged")
    if any(key in data for key in keys):
        if not all(key in data for key in keys):
            raise ValueError("incomplete projection eligibility markers")
        if bool(data[keys[0]]) or not bool(data[keys[1]]):
            raise ValueError(
                "diagnostic or unconverged projection is not a receiver input"
            )
    if str(data.get("projection_schema", "")) == "bie-boundary-fitted-mac-v1":
        required = (
            "target_momentum_x",
            "target_momentum_z",
            "source_volume",
            "surface_streamfunction_trace",
            "bottom_streamfunction_trace",
            "surface_streamfunction_gradient_x",
            "surface_streamfunction_gradient_z",
            "bottom_streamfunction_gradient_x",
            "bottom_streamfunction_gradient_z",
            "projection_qualification_scope",
        )
        if any(key not in data for key in required):
            raise ValueError("incomplete boundary-fitted provenance")
        if str(data["projection_qualification_scope"]) != "receiver-startup-only":
            raise ValueError("unrecognized boundary-fitted qualification scope")


def boundary_fitted_receiver_state(
    data, state, q, x_edges, y_edges, surface_shift, receiver_bottom=None
):
    """Re-integrate phase moments on the exact receiver polygon, not old q."""
    from cutcell_transfer import fitted_moments

    h = float(x_edges[1] - x_edges[0])
    if (
        len(data["grid_x"]) != len(x_edges) - 1
        or len(data["grid_z"]) != len(y_edges) - 1
        or not np.allclose(data["grid_x"], x_edges[:-1] + h / 2, atol=1e-13, rtol=0)
        or not np.allclose(data["grid_z"], y_edges[:-1] + h / 2, atol=1e-13, rtol=0)
    ):
        raise ValueError("boundary-fitted moments require the same receiver grid")
    curves = {
        name: {
            "x": data[name + "_x"],
            "z": data[name + "_z"]
            + (surface_shift if name == "surface" else EMBEDDED_BOUNDARY_OFFSET),
            "psi": data[name + "_streamfunction_trace"],
            "psi_x": data[name + "_streamfunction_gradient_x"],
            "psi_z": data[name + "_streamfunction_gradient_z"],
        }
        for name in ("surface", "bottom")
    }
    if receiver_bottom is not None:
        bx, bz = receiver_bottom
        old = curves["bottom"]
        if np.ptp(old["psi"]) > 1e-12:
            raise ValueError(
                "receiver bed remap requires an impermeable constant-psi bed"
            )
        curves["bottom"] = {
            "x": bx,
            "z": bz,
            "psi": np.full_like(bx, old["psi"][0]),
            # Gradients are physical BIE trace data, not MAC face averages.
            "psi_x": periodic_interp(bx, old["x"], old["psi_x"]),
            "psi_z": periodic_interp(bx, old["x"], old["psi_z"]),
        }
    geometry = periodic_liquid_geometry(
        curves["surface"]["x"],
        curves["surface"]["z"],
        curves["bottom"]["x"],
        curves["bottom"]["z"],
        y_min=y_edges[0],
        y_max=y_edges[-1],
    )
    vx, vy, report = fitted_moments(
        state["target_streamfunction"],
        q,
        x_edges[:-1] + h / 2,
        y_edges[:-1] + h / 2,
        geometry,
        curves,
    )
    return {**state, "vx": vx, "vy": vy}, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument(
        "--vertical-cells",
        type=int,
        required=True,
        help="Number of cells above the shifted zero boundary; cell width is 64/2^level.",
    )
    parser.add_argument(
        "--geometry-mode",
        choices=("direct", "raster"),
        default="direct",
        help="Direct polygon intersection is the production path; raster is an ablation.",
    )
    parser.add_argument(
        "--allow-coarse-velocity-source",
        action="store_true",
        help="Ablation only: allow a streamfunction coarser than the Aphros grid.",
    )
    parser.add_argument(
        "--allow-legacy-projection",
        action="store_true",
        help="Ablation only: accept an archive without v2 projection provenance.",
    )
    parser.add_argument(
        "--boundary-fitted-startup",
        action="store_true",
        help="Explicit startup-only use of a qualified boundary-fitted candidate.",
    )
    args = parser.parse_args()

    data = np.load(args.source)
    validate_projection_export(data)
    nx = 1 << args.level
    ny = args.vertical_cells
    dx = DOMAIN_LENGTH / nx
    x_edges = np.linspace(0.0, DOMAIN_LENGTH, nx + 1)
    y_edges_shifted = np.arange(ny + 1, dtype=np.float64) * dx
    y_edges = y_edges_shifted + VERTICAL_ORIGIN

    source_x = data["grid_x"]
    source_z = data["grid_z"]
    source_dx = float(source_x[1] - source_x[0])
    source_dz = float(source_z[1] - source_z[0])
    if not np.allclose(np.diff(source_x), source_dx, rtol=1e-12, atol=1e-14):
        raise ValueError("projected source x grid is not uniform")
    if not np.allclose(np.diff(source_z), source_dz, rtol=1e-12, atol=1e-14):
        raise ValueError("projected source z grid is not uniform")
    if not np.isclose(source_dx * source_x.size, DOMAIN_LENGTH, rtol=1e-12):
        raise ValueError("projected source does not span one periodic x domain")
    psi = np.asarray(data["mac_streamfunction"], dtype=np.float64)
    if psi.shape != (source_z.size + 1, source_x.size):
        raise ValueError(
            "mac_streamfunction shape must be (grid_z.size + 1, grid_x.size)"
        )
    if not np.isfinite(psi).all():
        raise ValueError("mac_streamfunction contains NaN or infinity")
    projection_schema = str(data.get("projection_schema", ""))
    fitted = projection_schema == "bie-boundary-fitted-mac-v1"
    if fitted and (not args.boundary_fitted_startup or args.geometry_mode != "direct"):
        raise ValueError(
            "boundary-fitted inputs require explicit direct-geometry startup"
        )
    if (
        projection_schema != "bie-to-vof-mac-face-projection-v2"
        and not fitted
        and not args.allow_legacy_projection
    ):
        raise ValueError(
            "projected source lacks v2 conservation provenance; regenerate it "
            "or use --allow-legacy-projection only as an ablation"
        )
    source_x_origin = float(source_x[0] - 0.5 * source_dx)
    source_z_origin = float(source_z[0] - 0.5 * source_dz)
    if source_dx > dx * (1.0 + 1.0e-12) and not args.allow_coarse_velocity_source:
        raise ValueError(
            "projected velocity source is coarser than the Aphros grid: "
            f"source_dx={source_dx:.9g}, target_dx={dx:.9g}; "
            "build an independent projection for this level"
        )
    target_volume = float(data.get("source_volume", TARGET_VOLUME))
    receiver_bottom = None
    bottom_x, bottom_z = data["bottom_x"], data["bottom_z"] + EMBEDDED_BOUNDARY_OFFSET
    if fitted:
        from receiver_embedded_geometry import receiver_bed

        receiver_bottom = receiver_bed(x_edges, y_edges, bottom_x, bottom_z)
        bottom_x, bottom_z = receiver_bottom
    cs = fluid_aperture(
        x_edges,
        y_edges,
        bottom_x,
        bottom_z,
    )
    geometry_report: dict[str, float | int | str]
    mass_repair_l1 = 0.0
    if args.geometry_mode == "direct":
        q, geometry_report = direct_liquid_aperture(
            x_edges,
            y_edges,
            data["surface_x"],
            data["surface_z"],
            bottom_x,
            bottom_z,
            target_volume=target_volume,
        )
        # Both geometries use the same offset bed.  A violation here signals a
        # real inconsistency and must not be hidden by a global mass repair.
        maximum_violation = float(np.max(q - cs))
        local_bed_clip_l1 = float(np.maximum(q - cs, 0.0).sum() * dx * dx)
        if local_bed_clip_l1 / target_volume > 1.0e-8:
            raise ValueError(
                "direct liquid/fluid bed intersection mismatch has relative L1 "
                f"{local_bed_clip_l1 / target_volume:.9g}"
            )
        q = np.minimum(q, cs)
        geometry_report.update(
            {
                "maximum_direct_q_minus_cs_before_local_clip": maximum_violation,
                "direct_bed_consistency_clip_l1": local_bed_clip_l1,
            }
        )
    else:
        source_x_edges = np.arange(source_x.size + 1) * source_dx
        source_z_edges = (
            source_z[0] - 0.5 * source_dz + np.arange(source_z.size + 1) * source_dz
        )
        wx = overlap_weights(x_edges, source_x_edges, periodic=True)
        wz = overlap_weights(y_edges, source_z_edges)
        q = np.clip(wz @ data["volume_fraction"] @ wx.T, 0.0, cs)
        geometry_report = {
            "geometry_initialization": "legacy_piecewise_constant_source_raster",
            "source_raster_dx": source_dx,
            "source_raster_dz": source_dz,
        }
        for _ in range(5):
            difference = target_volume - float(q.sum() * dx * dx)
            room = np.where(difference > 0.0, cs - q, q)
            mixed = (q > 0.0) & (q < cs) & (room > 0.0)
            capacity = float(room[mixed].sum() * dx * dx)
            if capacity == 0.0:
                break
            old = q[mixed].copy()
            q[mixed] = np.clip(
                q[mixed] + difference * room[mixed] / capacity,
                0.0,
                cs[mixed],
            )
            mass_repair_l1 += float(np.abs(q[mixed] - old).sum() * dx * dx)
    vf = np.divide(q, cs, out=np.zeros_like(q), where=cs > 0.0)

    velocity_state = target_mac_velocity_state(
        psi,
        source_dx,
        source_dz,
        x_edges,
        y_edges,
        source_x_origin=source_x_origin,
        source_z_origin=source_z_origin,
    )
    fitted_report = None
    if fitted:
        velocity_state, fitted_report = boundary_fitted_receiver_state(
            data,
            velocity_state,
            q,
            x_edges,
            y_edges,
            geometry_report["volume_gauge_surface_shift"],
            receiver_bottom=receiver_bottom,
        )
    vx = np.asarray(velocity_state["vx"])
    vy = np.asarray(velocity_state["vy"])
    vx[cs <= 0.0] = 0.0
    vy[cs <= 0.0] = 0.0

    args.output.mkdir(parents=True, exist_ok=True)
    if receiver_bottom is not None:
        np.savez_compressed(args.output / "receiver_bed.npz", x=bottom_x, z=bottom_z)
        geometry_report["embedded_geometry"] = (
            "aphros_signed_distance_edge_crossings_2d"
        )
        geometry_report["receiver_bed_nodes"] = len(bottom_x)
    for name, field in (
        ("q", q),
        ("vf", vf),
        ("vx", vx),
        ("vy", vy),
        ("cs", cs),
        ("fluxx", velocity_state["fluxx"]),
        ("fluxxp", velocity_state["fluxxp"]),
        ("fluxy", velocity_state["fluxy"]),
        ("fluxyp", velocity_state["fluxyp"]),
        ("fluxeb", velocity_state["fluxeb"]),
    ):
        np.asarray(field, dtype=np.float64).tofile(args.output / f"{name}.raw")
    write_body_polygon(args.output / "body.dat", data["bottom_x"], data["bottom_z"])

    momentum = np.array(
        [float((q * vx).sum() * dx * dx), float((q * vy).sum() * dx * dx)]
    )
    liquid_volume = float(q.sum() * dx * dx)
    physical_liquid_volume_from_vf_cs = float((vf * cs).sum() * dx * dx)
    speed = np.hypot(vx, vy)
    target_momentum = np.asarray(
        [
            float(data.get("target_momentum_x", TARGET_MOMENTUM[0])),
            float(data.get("target_momentum_z", TARGET_MOMENTUM[1])),
        ]
    )
    report = {
        "schema": "aphros-initial-state-v3",
        "boundary_fitted_moments": fitted_report,
        "qualification_scope": "receiver-startup-only" if fitted else "legacy-campaign",
        "source": str(args.source.resolve()),
        "level": args.level,
        "nx": nx,
        "ny": ny,
        "cell_size": dx,
        "physical_y_min": VERTICAL_ORIGIN,
        "physical_y_max": float(y_edges[-1]),
        "source_grid_nx": int(source_x.size),
        "source_grid_nz": int(source_z.size),
        "source_cell_size_x": source_dx,
        "source_cell_size_z": source_dz,
        "source_to_target_dx_ratio": source_dx / dx,
        "source_projection_schema": projection_schema,
        "raw_dtype": "float64-native-little-endian",
        "raw_field_bytes": int(q.nbytes),
        "liquid_volume": liquid_volume,
        "physical_liquid_volume_from_vf_cs": physical_liquid_volume_from_vf_cs,
        "target_liquid_volume": target_volume,
        "relative_volume_error": abs(liquid_volume - target_volume) / target_volume,
        "relative_vf_cs_volume_error": abs(
            physical_liquid_volume_from_vf_cs - target_volume
        )
        / target_volume,
        "fluid_volume": float(cs.sum() * dx * dx),
        "mass_repair_l1": mass_repair_l1,
        "liquid_momentum": momentum.tolist(),
        "target_liquid_momentum": target_momentum.tolist(),
        "relative_momentum_error": float(
            np.linalg.norm(momentum - target_momentum) / np.linalg.norm(target_momentum)
        ),
        "maximum_q_minus_cs": float(np.max(q - cs)),
        "minimum_q": float(np.min(q)),
        "maximum_q": float(np.max(q)),
        "minimum_vf": float(np.min(vf)),
        "maximum_vf": float(np.max(vf)),
        "maximum_abs_q_minus_vf_cs": float(np.max(np.abs(q - vf * cs))),
        "maximum_speed_all_fluid_cells": float(np.max(speed[cs > 0.0])),
        "maximum_speed_liquid_regular_cells": float(
            np.max(speed[(cs >= 0.5) & (vf > 0.01)])
        ),
        "all_raw_fields_finite": bool(
            all(
                np.isfinite(field).all()
                for field in (
                    q,
                    vf,
                    vx,
                    vy,
                    cs,
                    velocity_state["fluxx"],
                    velocity_state["fluxxp"],
                    velocity_state["fluxy"],
                    velocity_state["fluxyp"],
                    velocity_state["fluxeb"],
                )
            )
        ),
        "initial_flux_semantics": (
            "fluxx/fluxxp/fluxy/fluxyp are target-MAC normal velocities; "
            "the receiver must set flux_init_raw_is_velocity=1"
        ),
        "velocity_vertical_extension": velocity_state["vertical_extension"],
        "rectangular_face_divergence_l2": velocity_state[
            "rectangular_face_divergence_l2"
        ],
        "rectangular_face_divergence_linf": velocity_state[
            "rectangular_face_divergence_linf"
        ],
        "shared_x_face_mismatch_linf": velocity_state["shared_x_face_mismatch_linf"],
        "shared_y_face_mismatch_linf": velocity_state["shared_y_face_mismatch_linf"],
        "initial_vf_semantics": {
            "q.raw": "liquid aperture q=f*cs; requires init_vf_is_aperture=1",
            "vf.raw": "normalized liquid fraction f; requires init_vf_is_aperture=0",
            "cs.raw": "offline embedded fluid-volume fraction",
        },
        **geometry_report,
    }
    (args.output / "initial_state.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
