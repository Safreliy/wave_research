"""Experimental boundary-driven liquid streamfunction on the receiver MAC grid.

Periodic x, impermeable bed, zero background current. Shortley--Weller
cut-distance Laplacian with Dirichlet values on the actual polygon segments.
No gas velocity/energy rows enter this solve. This nonsymmetric operator must
not be passed to CG. Exterior corners of cut cells use a separately reported
first- or second-order normal continuation from the BIE trace, not a hidden
fluid solve. These diagnostic states are not qualified wave initial conditions.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import bicgstab
from shapely import STRtree, contains_xy, linestrings, points
from shapely.affinity import translate
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu"))
from bie_interior_evaluation import periodic_resample
from cutcell_transfer import fitted_moments
from physical_trace import ParentChordTrace, gauge_to_source_horizontal_momentum
from extend_gas_streamfunction import extend_gas, liquid_support_vertices
from prepare_aphros_initial_state import geometry_aperture, periodic_liquid_geometry
from rebuild_bie_velocity_source import boundary_flux_preflight


def boundary_traces(data, *, length=64.0, factor=4):
    """Construct psi_D from normal flux, preserving its integral diagnostics.

    Only a flux imbalance below the existing 5e-9 tolerance may be removed
    when integrating the periodic trace. The removed mode is recorded; this
    is not a projection of incompatible N=576 data into admissibility.
    """
    if float(data.get("background_current", 0.0)) != 0.0:
        raise ValueError("zero background current required")
    if np.max(np.abs(data["bottom_normal_derivative"])) > 1e-14:
        raise ValueError("impermeable bed required")
    preflight = boundary_flux_preflight(data, length)
    if not preflight["pass"]:
        raise ValueError("incompatible BIE boundary flux; refine/re-solve the source")
    curves = {}
    for name in ("surface", "bottom"):
        n = len(data[name + "_x"])
        m = n * factor
        base = length * np.arange(n) / n
        fine_base = length * np.arange(m) / m
        x = periodic_resample(data[name + "_x"] - base, factor) + fine_base
        z = periodic_resample(data[name + "_z"], factor)
        modes = np.fft.fftfreq(m) * m

        def derivative(f, modes=modes):
            return np.fft.ifft(1j * modes * np.fft.fft(f)).real

        xa = length / (2 * np.pi) + derivative(x - fine_base)
        za = derivative(z)
        metric = np.hypot(xa, za)
        phi = periodic_resample(data[name + "_potential"], factor)
        q = periodic_resample(data[name + "_normal_derivative"], factor)
        curves[name] = {
            "x": x,
            "z": z,
            "xa": xa,
            "za": za,
            "metric": metric,
            "phi": phi,
            "q": q,
            # psi_n = phi_s on surface; opposite on bottom.
            "psi_n": (1 if name == "surface" else -1) * derivative(phi) / metric,
        }
    surface, bottom = curves["surface"], curves["bottom"]
    derivative = -surface["q"] * surface["metric"]
    modes = np.fft.fftfreq(len(derivative)) * len(derivative)
    spectrum = np.fft.fft(derivative)
    removed_mean = float(spectrum[0].real / len(derivative))
    spectrum[0] = 0
    spectrum[1:] /= 1j * modes[1:]
    psi = np.fft.ifft(spectrum).real
    mx = (
        2
        * np.pi
        * (
            np.mean(bottom["phi"] * bottom["za"])
            - np.mean(surface["phi"] * surface["za"])
        )
    )
    mz = (
        2
        * np.pi
        * (
            np.mean(surface["phi"] * surface["xa"])
            - np.mean(bottom["phi"] * bottom["xa"])
        )
    )
    psi += (mx - 2 * np.pi * np.mean(psi * surface["xa"])) / length
    surface["psi"] = psi
    bottom["psi"] = np.zeros_like(bottom["x"])
    for name, curve in curves.items():
        modes = np.fft.fftfreq(len(curve["x"])) * len(curve["x"])

        def dalpha(f, modes=modes):
            return np.fft.ifft(1j * modes * np.fft.fft(f)).real

        curvature = (
            (1 if name == "surface" else -1)
            * (curve["xa"] * dalpha(curve["za"]) - curve["za"] * dalpha(curve["xa"]))
            / curve["metric"] ** 3
        )
        # t_s = curvature*n; harmonicity implies psi_nn = kappa*psi_n-psi_ss.
        curve["psi_nn"] = (
            curvature * curve["psi_n"]
            - dalpha(dalpha(curve["psi"]) / curve["metric"]) / curve["metric"]
        )
        tangential = -curve["q"] if name == "surface" else np.zeros_like(curve["q"])
        orientation = 1 if name == "surface" else -1
        curve["psi_x"] = (
            tangential * curve["xa"] - orientation * curve["psi_n"] * curve["za"]
        ) / curve["metric"]
        curve["psi_z"] = (
            tangential * curve["za"] + orientation * curve["psi_n"] * curve["xa"]
        ) / curve["metric"]
    mz_psi = 2 * np.pi * np.mean(psi * surface["za"])
    report = {
        "flux_preflight": preflight,
        "removed_psi_derivative_mean": removed_mean,
        "momentum_target": [float(mx), float(mz)],
        "momentum_trace_defect": float(abs(mz_psi - mz)),
        "source_nodes": len(data["surface_x"]),
        "geometry_factor": factor,
    }
    if abs(mz_psi - mz) > 1e-8 * max(1.0, np.hypot(mx, mz)):
        raise ValueError("potential/streamfunction momentum identities disagree")
    surface["source_momentum_target"] = np.array([mx, mz], dtype=float)
    return curves, report


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class BoundarySegments:
    def __init__(self, curves, length, trace=None):
        self.trace = trace
        (
            starts,
            ends,
            values,
            endvalues,
            normals,
            derivatives,
            endderivatives,
            second,
            endsecond,
        ) = ([] for _ in range(9))
        trace_names, trace_indices = [], []
        for name, curve in curves.items():
            a = np.column_stack((curve["x"], curve["z"]))
            b = np.roll(a, -1, axis=0)
            b[-1, 0] += length
            delta = b - a
            normals0 = np.column_stack((-delta[:, 1], delta[:, 0]))
            normals0 *= (1 if name == "surface" else -1) / np.linalg.norm(
                delta, axis=1
            )[:, None]
            for shift in (-length, 0.0, length):
                trace_names.extend([name] * len(a))
                trace_indices.extend(range(len(a)))
                starts.append(a + [shift, 0.0])
                ends.append(b + [shift, 0.0])
                values.append(curve["psi"])
                endvalues.append(np.roll(curve["psi"], -1))
                normals.append(normals0)
                derivatives.append(curve["psi_n"])
                endderivatives.append(np.roll(curve["psi_n"], -1))
                second.append(curve["psi_nn"])
                endsecond.append(np.roll(curve["psi_nn"], -1))
        (
            self.a,
            self.b,
            self.va,
            self.vb,
            self.normals,
            self.da,
            self.db,
            self.d2a,
            self.d2b,
        ) = [
            np.concatenate(v)
            for v in (
                starts,
                ends,
                values,
                endvalues,
                normals,
                derivatives,
                endderivatives,
                second,
                endsecond,
            )
        ]
        self.tree = STRtree(linestrings(np.stack((self.a, self.b), axis=1)))
        self.trace_names = np.asarray(trace_names)
        self.trace_indices = np.asarray(trace_indices)

    def trace_value(self, segment, t):
        if self.trace is None:
            return (1 - t) * self.va[segment] + t * self.vb[segment]
        result = np.empty_like(t, dtype=float)
        for name in ("surface", "bottom"):
            mask = self.trace_names[segment] == name
            if np.any(mask):
                result[mask] = self.trace.value(
                    name, self.trace_indices[segment[mask]], t[mask]
                )
        return result

    def trace_normal(self, segment, t):
        if self.trace is None:
            return (1 - t) * self.da[segment] + t * self.db[segment]
        result = np.empty_like(t, dtype=float)
        for name in ("surface", "bottom"):
            mask = self.trace_names[segment] == name
            if np.any(mask):
                # Existing outward normal convention is opposite on the bed.
                sign = 1 if name == "surface" else -1
                result[mask] = sign * self.trace.normal_derivative(
                    name, self.trace_indices[segment[mask]], t[mask]
                )
        return result

    def crossings(self, start, end):
        """First boundary crossing, including thin obstacles with both ends liquid."""
        pairs = self.tree.query(
            linestrings(np.stack((start, end), axis=1)), predicate="intersects"
        )
        theta = np.ones(len(start))
        value = np.zeros(len(start))
        hit = np.zeros(len(start), dtype=bool)
        if pairs.size:
            target, segment = pairs
            d, s = end[target] - start[target], self.b[segment] - self.a[segment]
            denominator = cross(d, s)
            valid = np.abs(denominator) > 1e-25
            target, segment, d, s, denominator = [
                v[valid] for v in (target, segment, d, s, denominator)
            ]
            a = self.a[segment] - start[target]
            t, u = cross(a, s) / denominator, cross(a, d) / denominator
            valid = (t > 0) & (t <= 1 + 1e-10) & (u >= -1e-10) & (u <= 1 + 1e-10)
            target, segment, t, u = [v[valid] for v in (target, segment, t, u)]
            order = np.lexsort((t, target))
            target, segment, t, u = [v[order] for v in (target, segment, t, u)]
            first = (
                np.r_[True, np.diff(target) != 0] if len(target) else np.zeros(0, bool)
            )
            target, segment, t, u = [v[first] for v in (target, segment, t, u)]
            if np.any(t < 1e-11):
                raise ValueError(
                    "near-zero cut distance; change/refine grid, no clipping allowed"
                )
            theta[target] = np.minimum(t, 1.0)
            value[target] = self.trace_value(segment, u)
            hit[target] = True
        return theta, value, hit

    def continue_normal(self, locations, order=1):
        pair = self.tree.query_nearest(points(locations), all_matches=False)
        nearest_order = np.argsort(pair[0])
        segment = pair[1, nearest_order]
        if not np.array_equal(pair[0, nearest_order], np.arange(len(locations))):
            raise ValueError("missing nearest-boundary continuation")
        s = self.b[segment] - self.a[segment]
        t = np.clip(
            np.sum((locations - self.a[segment]) * s, axis=1) / np.sum(s * s, axis=1),
            0,
            1,
        )
        delta = locations - (self.a[segment] + t[:, None] * s)
        distance = np.sum(delta * self.normals[segment], axis=1)
        psi = self.trace_value(segment, t)
        normal = self.trace_normal(segment, t)
        if order not in (1, 2):
            raise ValueError("normal continuation order must be 1 or 2")
        psi += distance * normal
        if order == 2:
            psi += (
                0.5
                * distance**2
                * ((1 - t) * self.d2a[segment] + t * self.d2b[segment])
            )
        return psi, float(np.max(np.linalg.norm(delta, axis=1)))


def assemble(curves, x, z, *, length=64.0, trace=None):
    if length != 64.0:
        raise ValueError("this receiver currently requires a domain length of 64")
    for grid in (x, z):
        if (
            grid.ndim != 1
            or len(grid) < 3
            or not np.isfinite(grid).all()
            or np.any(np.diff(grid) <= 0)
            or not np.allclose(np.diff(grid), grid[1] - grid[0], rtol=1e-11, atol=1e-13)
        ):
            raise ValueError("receiver grids must be finite, increasing and uniform")
    if not np.isclose(x[0], 0.0, atol=1e-13):
        raise ValueError("receiver x origin must be zero")
    if (
        min(np.min(c["z"]) for c in curves.values()) < z[0] - 1e-13
        or max(np.max(c["z"]) for c in curves.values()) > z[-1] + 1e-13
    ):
        raise ValueError("receiver vertical domain clips the physical boundary")
    dx, dz = x[1] - x[0], z[1] - z[0]
    if not np.isclose(dx, dz) or not np.isclose(dx * len(x), length):
        raise ValueError("square periodic receiver grid required")
    geometry = periodic_liquid_geometry(
        curves["surface"]["x"],
        curves["surface"]["z"],
        curves["bottom"]["x"],
        curves["bottom"]["z"],
        y_min=z[0],
        y_max=z[-1],
    )
    xx, zz = np.meshgrid(x, z)
    # The central clipped polygon has artificial vertical edges at x=0,L.
    # Union periodic copies before classification, or seam vertices would be
    # falsely treated as outside liquid.
    periodic = unary_union(
        [translate(geometry, xoff=-length), geometry, translate(geometry, xoff=length)]
    )
    inside = contains_xy(periodic, xx, zz)
    if not inside.any():
        raise ValueError("no resolved liquid vertices")
    if inside[[0, -1]].any():
        raise ValueError("artificial vertical boundary intersects liquid")
    index = np.full(inside.shape, -1, dtype=int)
    index[inside] = np.arange(inside.sum())
    k, i = np.nonzero(inside)
    locations = np.column_stack((xx[inside], zz[inside]))
    boundary = BoundarySegments(curves, length, trace=trace)
    links = []
    for di, dk in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        end = locations + [di * dx, dk * dz]
        theta, values, hit = boundary.crossings(locations, end)
        neighbour = index[k + dk, (i + di) % len(x)]
        if np.any((neighbour < 0) & ~hit):
            raise ValueError("outside neighbour without a Dirichlet crossing")
        links.append((theta * dx, values, hit, neighbour))
    weights = []
    for d in range(4):
        distance, _, _, _ = links[d]
        span = links[(d // 2) * 2][0] + links[(d // 2) * 2 + 1][0]
        weights.append(2 / (distance * span))
    diagonal = np.sum(weights, axis=0)
    rows, cols, values = [np.arange(len(k))], [np.arange(len(k))], [np.ones(len(k))]
    rhs = np.zeros(len(k))
    for weight, (_, trace, hit, neighbour) in zip(weights, links):
        coefficient = weight / diagonal
        rhs[hit] += coefficient[hit] * trace[hit]
        rows.append(np.flatnonzero(~hit))
        cols.append(neighbour[~hit])
        values.append(-coefficient[~hit])
    matrix = sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(k), len(k)),
    )
    return matrix, rhs, inside, geometry, boundary


def solve(matrix, rhs, *, backend="numpy", rtol=1e-11, maxiter=20000):
    started = time.perf_counter()
    count = 0

    def callback(_):
        nonlocal count
        count += 1

    if backend == "numpy":
        values, info = bicgstab(
            matrix, rhs, rtol=rtol, atol=1e-14, maxiter=maxiter, callback=callback
        )
    elif backend == "cupy":
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix
        from cupyx.scipy.sparse.linalg import gmres

        # GMRES is appropriate for the nonsymmetric cut-distance stencil.
        values, info = gmres(
            csr_matrix(matrix),
            cp.asarray(rhs),
            rtol=rtol,
            atol=1e-14,
            restart=50,
            maxiter=maxiter,
            callback=callback,
            callback_type="pr_norm",
        )
        values = cp.asnumpy(values)
    else:
        raise ValueError("unknown backend")
    residual = np.linalg.norm(matrix @ values - rhs) / max(np.linalg.norm(rhs), 1e-30)
    if info != 0 or not np.isfinite(values).all() or residual > max(10 * rtol, 1e-10):
        raise ValueError(
            f"Dirichlet solve failed: info={info}, true relative residual={residual}"
        )
    return values, {
        "backend": backend,
        "iterations_or_restart_callbacks": count,
        "info": int(info),
        "relative_residual": float(residual),
        "solve_seconds": time.perf_counter() - started,
        "unknowns": matrix.shape[0],
        "nnz": matrix.nnz,
    }


def transfer(
    curves,
    x,
    z,
    *,
    length=64.0,
    backend="numpy",
    gas=True,
    continuation_order=1,
    cutcell_moments=False,
    physical_trace="linear",
):
    started = time.perf_counter()
    # Reconcile spectral area with polygon area once, as in the existing
    # receiver's direct-geometry path. Never spread mass over arbitrary cells.
    target_volume = (
        2
        * np.pi
        * (
            np.mean(curves["surface"]["z"] * curves["surface"]["xa"])
            - np.mean(curves["bottom"]["z"] * curves["bottom"]["xa"])
        )
    )
    polygon_volume = 0.0
    for name, curve in curves.items():
        next_x = np.r_[curve["x"][1:], curve["x"][0] + length]
        polygon_volume += (1 if name == "surface" else -1) * np.sum(
            0.5 * (curve["z"] + np.roll(curve["z"], -1)) * (next_x - curve["x"])
        )
    shift = (target_volume - polygon_volume) / length
    curves = {
        name: {**curve, "z": curve["z"] + (shift if name == "surface" else 0.0)}
        for name, curve in curves.items()
    }
    if physical_trace not in ("linear", "parent_hermite", "parent_hermite_matched"):
        raise ValueError("unknown physical_trace mode")
    if physical_trace != "linear" and not cutcell_moments:
        raise ValueError("parent Hermite physical trace requires cutcell_moments")
    trace = None
    trace_report = None
    if physical_trace != "linear":
        curves, trace, trace_report = gauge_to_source_horizontal_momentum(
            curves, length
        )
        vertical_defect = abs(trace_report["vertical_source_momentum_defect"])
        source_target = np.asarray(trace_report["source_momentum_target"])
        if vertical_defect > 1e-8 * max(1.0, np.linalg.norm(source_target)):
            raise ValueError("Hermite polygon trace incompatible with source vertical momentum")
    matrix, rhs, inside, geometry, boundary = assemble(
        curves, x, z, length=length, trace=trace
    )
    assembly_seconds = time.perf_counter() - started
    values, report = solve(matrix, rhs, backend=backend)
    psi = np.zeros_like(inside, dtype=float)
    psi[inside] = values
    dx = x[1] - x[0]
    q, qreport = geometry_aperture(geometry, np.r_[x, x[0] + length], z)
    support = liquid_support_vertices(q)
    outside_support = support & ~inside
    xx, zz = np.meshgrid(x, z)
    max_distance = 0.0
    if outside_support.any():
        psi[outside_support], max_distance = boundary.continue_normal(
            np.column_stack((xx[outside_support], zz[outside_support])),
            order=continuation_order,
        )
    before = psi.copy()
    gas_report = None
    if gas:
        # Do not extend through the bed: below it the exterior velocity is zero
        # except for the explicit one-cell cut-support continuation above.
        bed = curves["bottom"]
        bed_z = np.interp(
            x,
            np.r_[bed["x"] - length, bed["x"], bed["x"] + length],
            np.tile(bed["z"], 3),
        )
        solid = zz < bed_z[None, :]
        psi, gas_report = extend_gas(
            psi, q, dx, additional_fixed=solid, backend=backend
        )
        if not np.array_equal(psi[support], before[support]):
            raise ValueError("gas continuation changed liquid support")
    fx = np.diff(psi, axis=0) / dx
    fz = -(np.roll(psi, -1, axis=1) - psi) / dx
    u, w = 0.5 * (fx + np.roll(fx, -1, axis=1)), 0.5 * (fz[:-1] + fz[1:])
    cut_report = None
    if cutcell_moments:
        u, w, cut_report = fitted_moments(
            psi, q, x + 0.5 * dx, 0.5 * (z[:-1] + z[1:]), geometry, curves,
            trace=trace,
            matched_flat_surface_edges=(physical_trace == "parent_hermite_matched"),
        )
    div = (np.roll(fx, -1, axis=1) - fx + np.diff(fz, axis=0)) / dx
    report.update(
        assembly_seconds=assembly_seconds,
        geometry=qreport,
        continuation="normal Taylor only at exterior liquid-cell corners",
        continuation_order=continuation_order,
        continuation_vertices=int(outside_support.sum()),
        continuation_max_distance=max_distance,
        gas=gas_report,
        liquid_support_bitwise_preserved=bool(
            np.array_equal(psi[support], before[support])
        ),
        liquid_divergence_linf=float(np.max(np.abs(div[q > 0]))),
        maximum_liquid_speed=float(np.max(np.hypot(u, w)[q > 0])),
        volume=float(q.sum() * dx * dx),
        target_volume=float(target_volume),
        surface_geometry_shift=float(shift),
        relative_volume_error=float(
            abs(q.sum() * dx * dx - target_volume) / target_volume
        ),
        momentum=[float(np.sum(q * u) * dx * dx), float(np.sum(q * w) * dx * dx)],
        full_l12_qualified=False,
        cutcell_moments=cut_report,
        physical_trace=physical_trace,
        physical_trace_report=trace_report,
    )
    return {
        "grid_x": x + 0.5 * dx,
        "grid_z": 0.5 * (z[1:] + z[:-1]),
        "volume_fraction": q,
        "mac_streamfunction": psi,
        "liquid_vertex_mask": inside,
        "face_velocity_x_projected": fx,
        "face_velocity_z_projected": fz,
        "cell_velocity_x_projected": u,
        "cell_velocity_z_projected": w,
        "projection_schema": np.asarray(
            "bie-boundary-fitted-mac-v1"
            if cutcell_moments
            else "bie-liquid-dirichlet-mac-v1"
        ),
        "projection_diagnostic_only": np.asarray(True),
        "projection_solver_converged": np.asarray(True),
        "source_volume": np.asarray(target_volume),
        "surface_x": curves["surface"]["x"],
        "surface_z": curves["surface"]["z"],
        "bottom_x": curves["bottom"]["x"],
        "bottom_z": curves["bottom"]["z"],
        "surface_streamfunction_trace": curves["surface"]["psi"],
        "bottom_streamfunction_trace": curves["bottom"]["psi"],
        "surface_streamfunction_gradient_x": curves["surface"]["psi_x"],
        "surface_streamfunction_gradient_z": curves["surface"]["psi_z"],
        "bottom_streamfunction_gradient_x": curves["bottom"]["psi_x"],
        "bottom_streamfunction_gradient_z": curves["bottom"]["psi_z"],
        "target_momentum_x": np.asarray(
            2
            * np.pi
            * (
                np.mean(curves["bottom"]["phi"] * curves["bottom"]["za"])
                - np.mean(curves["surface"]["phi"] * curves["surface"]["za"])
            )
        ),
        "target_momentum_z": np.asarray(
            2
            * np.pi
            * (
                np.mean(curves["surface"]["phi"] * curves["surface"]["xa"])
                - np.mean(curves["bottom"]["phi"] * curves["bottom"]["xa"])
            )
        ),
    }, report
