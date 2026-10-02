"""Conservative liquid moments from a boundary-fitted streamfunction trace.

Full MAC face velocities and phase-cell means are different quantities. Do not
silently replace either by the other. No global momentum repair is performed.
"""

import numpy as np
from shapely import STRtree, box, intersection, linestrings, points
from shapely.geometry.polygon import orient
from physical_trace import cartesian_blend_weight


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def fitted_moments(psi, q, x, z, geometry, curves, *, trace=None,
                   vertex_gradient_override=None,
                   matched_flat_surface_edges=False):
    if matched_flat_surface_edges and trace is None:
        raise ValueError("matched surface edges require the parent Hermite trace")
    h = float(x[1] - x[0])
    k, i = np.nonzero((q > 0) & (q < 1))
    clipped = intersection(
        geometry, box(x[i] - h / 2, z[k] - h / 2, x[i] + h / 2, z[k] + h / 2)
    )
    starts, ends, va, vb, ga, gb = [], [], [], [], [], []
    trace_names, trace_indices = [], []
    target = np.zeros(2)
    for name, sign in (("surface", 1), ("bottom", -1)):
        c = curves[name]
        a = np.column_stack((c["x"], c["z"]))
        b = np.roll(a, -1, axis=0)
        b[-1, 0] += 64
        if trace is None:
            target += sign * np.sum(
                0.5 * (c["psi"] + np.roll(c["psi"], -1))[:, None] * (b - a), axis=0
            )
        for shift in (-64, 0, 64):
            trace_names.extend([name] * len(a))
            trace_indices.extend(range(len(a)))
            starts.append(a + [shift, 0])
            ends.append(b + [shift, 0])
            va.append(c["psi"])
            vb.append(np.roll(c["psi"], -1))
            gradients = np.column_stack((c["psi_x"], c["psi_z"]))
            ga.append(gradients)
            gb.append(np.roll(gradients, -1, axis=0))
    a, b, va, vb, ga, gb = map(np.concatenate, (starts, ends, va, vb, ga, gb))
    trace_names = np.asarray(trace_names)
    trace_indices = np.asarray(trace_indices)
    if trace is not None:
        target = trace.momentum()
    tree = STRtree(linestrings(np.stack((a, b), axis=1)))
    vertices, following, owners = [], [], []
    offset = 0
    for owner, piece in enumerate(clipped):
        polygons = list(piece.geoms) if hasattr(piece, "geoms") else [piece]
        for polygon in polygons:
            if polygon.area <= 0:
                continue
            polygon = orient(polygon, sign=1)
            for ring in (polygon.exterior, *polygon.interiors):
                coords = np.asarray(ring.coords)[:-1]
                vertices.extend(coords)
                following.extend(offset + np.roll(np.arange(len(coords)), -1))
                owners.extend([owner] * len(coords))
                offset += len(coords)
    v, nxt, owner = np.asarray(vertices), np.asarray(following), np.asarray(owners)
    if not len(v):
        raise ValueError("boundary-fitted transfer requires resolved mixed cells")
    pair, distance = tree.query_nearest(
        points(v), all_matches=False, return_distance=True
    )
    ordering = np.argsort(pair[0])
    segment, distance = pair[1, ordering], distance[ordering]
    on_boundary = distance <= 1e-9 * h
    delta = b[segment] - a[segment]
    t = np.clip(
        np.sum((v - a[segment]) * delta, axis=1) / np.sum(delta * delta, axis=1), 0, 1
    )
    if trace is None:
        trace_value = (1 - t) * va[segment] + t * vb[segment]
    else:
        trace_value = np.empty(len(v))
        for name in ("surface", "bottom"):
            mask = trace_names[segment] == name
            if np.any(mask):
                trace_value[mask] = trace.value(
                    name, trace_indices[segment[mask]], t[mask]
                )
    gx, gz = (v[:, 0] - (x[0] - h / 2)) / h, (v[:, 1] - (z[0] - h / 2)) / h
    ix, iz = np.floor(gx).astype(int), np.floor(gz).astype(int)
    if np.any(iz < 0) or np.any(iz >= psi.shape[0] - 1):
        raise ValueError("liquid polygon intersects an artificial vertical boundary")
    ax, az = gx - ix, gz - iz
    ix %= len(x)
    jx = (ix + 1) % len(x)
    value = (
        (1 - ax) * (1 - az) * psi[iz, ix]
        + ax * (1 - az) * psi[iz, jx]
        + (1 - ax) * az * psi[iz + 1, ix]
        + ax * az * psi[iz + 1, jx]
    )
    value = np.where(on_boundary, trace_value, value)
    dxpsi = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2 * h)
    dzpsi = np.gradient(psi, h, axis=0, edge_order=2)
    if vertex_gradient_override is not None:
        oracle = np.asarray(vertex_gradient_override)
        if oracle.shape != (2, *psi.shape) or not np.isfinite(oracle).all():
            raise ValueError("oracle vertex gradients must be finite with shape (2, *psi.shape)")
        dxpsi, dzpsi = oracle
    gradient = np.column_stack((dxpsi[iz, ix], dzpsi[iz, ix]))
    # Nonboundary polygon vertices are grid corners. Round rather than use a
    # floor perturbed by clipping roundoff when selecting their derivatives.
    nearest_i, nearest_k = np.rint(gx).astype(int) % len(x), np.rint(gz).astype(int)
    gradient[~on_boundary] = np.column_stack(
        (
            dxpsi[nearest_k[~on_boundary], nearest_i[~on_boundary]],
            dzpsi[nearest_k[~on_boundary], nearest_i[~on_boundary]],
        )
    )
    if trace is None:
        gradient[on_boundary] = (
            (1 - t)[:, None] * ga[segment] + t[:, None] * gb[segment]
        )[on_boundary]
    else:
        for name in ("surface", "bottom"):
            mask = on_boundary & (trace_names[segment] == name)
            if np.any(mask):
                gradient[mask] = trace.gradient(
                    name, trace_indices[segment[mask]], t[mask]
                )
    horizontal_mean = (
        0.5 * (psi + np.roll(psi, -1, axis=1))
        + h * (dxpsi - np.roll(dxpsi, -1, axis=1)) / 12
    )
    matched_rows = np.zeros(len(horizontal_mean), dtype=bool)
    max_matched_change = 0.0
    if matched_flat_surface_edges:
        surface_z = np.asarray(curves["surface"]["z"])
        if np.ptp(surface_z) > 1e-10:
            raise ValueError("matched surface edges require a flat surface")
        xedges = np.r_[x - 0.5 * h, x[-1] + 0.5 * h]
        zedges = np.r_[z - 0.5 * h, z[-1] + 0.5 * h]
        jet_h, jet_n, jet_nn = trace.flat_surface_jet_means(xedges)
        distance = float(surface_z[0]) - zedges
        matched_rows = (distance >= -1e-12 * h) & (distance < h)
        for row in np.flatnonzero(matched_rows):
            d = max(0.0, float(distance[row]))
            s = d / h
            # Cartesian weight beta=3s²-2s³: beta(0)=beta'(0)=0,
            # beta(1)=1 and beta'(1)=0. Thus any fixed mismatch in the
            # Cartesian integral is suppressed as O(d²) near the interface.
            beta = float(cartesian_blend_weight(s))
            jet = jet_h - d * jet_n + 0.5 * d * d * jet_nn
            old = horizontal_mean[row].copy()
            horizontal_mean[row] = (1 - beta) * jet + beta * old
            max_matched_change = max(
                max_matched_change, float(np.max(np.abs(horizontal_mean[row] - old)))
            )
    midpoint = 0.5 * (v + v[nxt])
    middle_pair, middle_distance = tree.query_nearest(
        points(midpoint), all_matches=False, return_distance=True
    )
    middle_segment = middle_pair[1, np.argsort(middle_pair[0])]
    physical_edge = middle_distance[np.argsort(middle_pair[0])] <= 1e-9 * h
    edge_delta = v[nxt] - v
    hermite = np.sum((gradient - gradient[nxt]) * edge_delta, axis=1) / 12
    # The physical edge must restrict one parent chord.  In the opt-in mode,
    # use the exact cubic restriction rather than endpoint interpolation.
    if trace is None:
        hermite[physical_edge] = 0
    # Subtract a constant per cell before Green integration to avoid cancellation
    # of large gauge values on extremely small clipped polygons.
    gauge = psi[k, i]
    edge_mean = 0.5 * (value + value[nxt]) + hermite
    if trace is not None:
        boundary_starts = a[middle_segment]
        boundary_delta = b[middle_segment] - boundary_starts
        denominator = np.sum(boundary_delta**2, axis=1)
        t0 = np.sum((v - boundary_starts) * boundary_delta, axis=1) / denominator
        t1 = np.sum((v[nxt] - boundary_starts) * boundary_delta, axis=1) / denominator
        distance0 = np.abs(cross(v - boundary_starts, boundary_delta)) / np.sqrt(denominator)
        distance1 = np.abs(cross(v[nxt] - boundary_starts, boundary_delta)) / np.sqrt(denominator)
        invalid = physical_edge & (
            (t0 < -1e-9) | (t0 > 1 + 1e-9)
            | (t1 < -1e-9) | (t1 > 1 + 1e-9)
            | (distance0 > 1e-9 * h) | (distance1 > 1e-9 * h)
        )
        if np.any(invalid):
            raise ValueError("physical cut edge does not restrict one parent chord")
        for name in ("surface", "bottom"):
            mask = physical_edge & (trace_names[middle_segment] == name)
            if np.any(mask):
                edge_mean[mask] = trace.mean(
                    name, trace_indices[middle_segment[mask]], t0[mask], t1[mask]
                )
    if matched_flat_surface_edges:
        zedge0 = z[0] - 0.5 * h
        xedge0 = x[0] - 0.5 * h
        row = np.rint((v[:, 1] - zedge0) / h).astype(int)
        horizontal = (~physical_edge) & (np.abs(edge_delta[:, 1]) <= 1e-9 * h)
        in_range = (row >= 0) & (row < len(matched_rows))
        near = horizontal & in_range & matched_rows[np.clip(row, 0, len(matched_rows)-1)]
        left = np.minimum(v[:, 0], v[nxt, 0])
        col = np.rint((left - xedge0) / h).astype(int)
        bad = near & (
            (np.abs(np.abs(edge_delta[:, 0]) - h) > 1e-9 * h)
            | (np.abs(v[:, 1] - (zedge0 + row * h)) > 1e-9 * h)
            | (np.abs(left - (xedge0 + col * h)) > 1e-9 * h)
        )
        if np.any(bad):
            raise ValueError("near-surface Cartesian cut edge is not a full shared grid edge")
        edge_mean[near] = horizontal_mean[row[near], col[near] % len(x)]
    contribution = -((edge_mean - gauge[owner])[:, None] * edge_delta)
    moments = np.zeros((len(k), 2))
    np.add.at(moments, owner, contribution)
    fx = np.diff(psi, axis=0) / h
    fz = -(np.roll(psi, -1, axis=1) - psi) / h
    u, w = 0.5 * (fx + np.roll(fx, -1, axis=1)), 0.5 * (fz[:-1] + fz[1:])
    original = np.array([u[k, i], w[k, i]])
    vertical_mean = 0.5 * (psi[:-1] + psi[1:]) + h * (dzpsi[:-1] - dzpsi[1:]) / 12
    # Use identical shared-edge integrals in full and cut cells.
    liquid = q > 0
    u[liquid] = (np.diff(horizontal_mean, axis=0) / h)[liquid]
    w[liquid] = (-(np.roll(vertical_mean, -1, axis=1) - vertical_mean) / h)[liquid]
    u[k, i], w[k, i] = (moments / (q[k, i] * h * h)[:, None]).T
    total = np.array([np.sum(q * u), np.sum(q * w)]) * h * h
    defect = np.linalg.norm(total - target)
    if (
        not np.isfinite(u).all()
        or not np.isfinite(w).all()
        or defect > 1e-10 * max(1, np.linalg.norm(target))
    ):
        raise ValueError(f"boundary-fitted moment closure failed: {defect}")
    return (
        u,
        w,
        {
            "schema": (
                "boundary-fitted-matched-flat-surface-liquid-moments-v1"
                if matched_flat_surface_edges else
                "boundary-fitted-parent-chord-hermite-liquid-moments-v1"
                if trace is not None else "boundary-fitted-hermite-liquid-moments-v2"
            ),
            "matched_flat_surface_edge_rows": int(matched_rows.sum()),
            "maximum_matched_edge_mean_change": max_matched_change,
            "mixed_cells": len(k),
            "polygon_trace_momentum": target.tolist(),
            "momentum": total.tolist(),
            "trace_closure_absolute": float(defect),
            "max_mixed_cell_velocity_change": float(
                np.max(np.abs(np.array([u[k, i], w[k, i]]) - original))
            ),
            "global_rescale_applied": False,
        },
    )
