"""Oracle decomposition of flat top-strip moment error (diagnostic only)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cutcell_transfer import fitted_moments
from liquid_dirichlet_transfer import boundary_traces, transfer
from physical_trace import ParentChordTrace
from prepare_aphros_initial_state import (EMBEDDED_BOUNDARY_OFFSET,
                                          VERTICAL_ORIGIN,
                                          periodic_liquid_geometry)
from verify_parent_trace_harmonic import LENGTH, exact_phase_means, source


CASES = (("manufactured", 128, 0.00001),
         ("manufactured", 128, 0.001),
         ("manufactured", 128, 0.01),
         ("manufactured", 256, 0.001),
         ("native", 512, 0.0005),
         ("native", 1024, 0.001))


def configuration(family, nx, fraction):
    h = LENGTH / nx
    if family == "manufactured":
        bed = -4.0
        zedge = np.arange(bed - 0.37 * h, 4 + h, h)
        surface = (-0.37 + fraction) * h
        mode, ntrace, factor = 4, 256, nx // 16
        k = 2 * np.pi * mode / LENGTH
        amplitude = np.cosh(k * (surface - bed)) / np.cosh(k * 4.0)
    else:
        bed = VERTICAL_ORIGIN + 0.53 + EMBEDDED_BOUNDARY_OFFSET
        zedge = VERTICAL_ORIGIN + np.arange(nx // 16 + 1) * h
        surface = VERTICAL_ORIGIN + 1.5 + fraction * h
        mode, ntrace, factor, amplitude = 16, 512, 8, 0.02
    return bed, surface, zedge, mode, ntrace, factor, amplitude


def norm(value):
    return float(np.linalg.norm(value))


def edge_error(curves, trace, psi, gradx, *, h, nx, zedge, bed,
               surface, mode, amplitude, top_row):
    k = 2 * np.pi * mode / LENGTH
    x = np.arange(nx) * h
    xc = x + 0.5 * h
    q = (surface - zedge[top_row]) / h
    curve = curves["surface"]
    m = len(curve["x"])
    if m % nx:
        raise ValueError("top-strip diagnostic needs commensurate trace and grid")
    stride = m // nx
    ds = np.diff(np.r_[curve["x"], curve["x"][0] + LENGTH])
    if np.max(np.abs(ds - LENGTH / m)) > 1e-10:
        raise ValueError("top-strip diagnostic needs flat uniform trace")
    if trace is None:
        top_piece = 0.5 * (curve["psi"] + np.roll(curve["psi"], -1))
    else:
        j = np.arange(m)
        top_piece = trace.integral("surface", j, 0.0, 1.0)
    top_mean = (top_piece * ds).reshape(nx, stride).sum(axis=1) / h
    bottom_mean = 0.5 * (psi[top_row] + np.roll(psi[top_row], -1)) + (
        h * (gradx[top_row] - np.roll(gradx[top_row], -1)) / 12
    )
    normfactor = amplitude / np.cosh(k * (surface - bed))
    top_exact = -normfactor * np.sin(k * xc) * np.sinc(k * h / (2 * np.pi)) * np.sinh(k * (surface - bed))
    bottom_exact = -normfactor * np.sin(k * xc) * np.sinc(k * h / (2 * np.pi)) * np.sinh(k * (zedge[top_row] - bed))
    # Means of psi on the two horizontal edges; their difference is q*h*u.
    exact_gap = top_exact - bottom_exact
    top_error = top_mean - top_exact
    bottom_error = bottom_mean - bottom_exact
    return {
        "top_interpolation_l2": norm(top_error),
        "cartesian_bottom_interpolation_l2": norm(bottom_error),
        "edge_gap_error_l2": norm(top_error - bottom_error),
        "physical_gap_l2": norm(exact_gap),
        "edge_gap_relative_error": norm(top_error - bottom_error) / norm(exact_gap),
        "top_bottom_error_inner_product": float(np.dot(top_error, bottom_error)),
        "top_bottom_error_cosine": float(np.dot(top_error, bottom_error)
                                         / (norm(top_error) * norm(bottom_error))),
        "top_row_q": q,
    }


def run_case(family, nx, fraction):
    h = LENGTH / nx
    xedge = np.arange(nx) * h
    bed, surface, zedge, mode, ntrace, factor, amplitude = configuration(family, nx, fraction)
    curves, _ = boundary_traces(source(bed, surface, mode=mode,
                                       amplitude=amplitude, ntrace=ntrace), factor=factor)
    k = 2 * np.pi * mode / LENGTH
    xx, zz = np.meshgrid(xedge, zedge)
    variants = {}
    for label in ("linear", "parent_hermite"):
        data, report = transfer(curves, xedge, zedge, gas=False,
                                continuation_order=2, cutcell_moments=True,
                                physical_trace=label)
        effective = {name: {**c, "z": data[name + "_z"],
                            "psi": data[name + "_streamfunction_trace"]}
                     for name, c in curves.items()}
        trace = ParentChordTrace(effective) if label == "parent_hermite" else None
        geometry = periodic_liquid_geometry(
            effective["surface"]["x"], effective["surface"]["z"],
            effective["bottom"]["x"], effective["bottom"]["z"],
            y_min=zedge[0], y_max=zedge[-1]
        )
        actual_surface = float(data["surface_z"][0])
        actual_bed = float(data["bottom_z"][0])
        q = data["volume_fraction"]
        psi_exact = (-amplitude * np.sin(k * xx)
                     * np.sinh(k * (zz - actual_bed))
                     / np.cosh(k * (actual_surface - actual_bed)))
        grad_exact = np.array([
            -amplitude * k * np.cos(k * xx) * np.sinh(k * (zz - actual_bed))
            / np.cosh(k * (actual_surface - actual_bed)),
            -amplitude * k * np.sin(k * xx) * np.cosh(k * (zz - actual_bed))
            / np.cosh(k * (actual_surface - actual_bed)),
        ])
        lo = np.maximum(zedge[:-1], actual_bed)
        hi = np.minimum(zedge[1:], actual_surface)
        wet = hi > lo
        ref = np.zeros((2, len(lo), nx))
        ref[:, wet] = exact_phase_means(data["grid_x"], lo[wet], hi[wet],
                                        actual_bed, actual_surface,
                                        mode, amplitude, h)
        top_row = np.flatnonzero(wet)[-1]
        top_mask = np.zeros(q.shape, dtype=bool)
        top_mask[top_row] = q[top_row] > 0
        outcomes = {}
        for psi_label, values in (("solved", data["mac_streamfunction"]),
                                  ("analytic", psi_exact)):
            for grad_label, override in (("finite_difference", None),
                                         ("analytic_oracle", grad_exact)):
                u, w, closure = fitted_moments(
                    values, q, data["grid_x"], data["grid_z"],
                    geometry, effective, trace=trace,
                    vertex_gradient_override=override,
                )
                found = np.array([u, w])
                key = psi_label + "_" + grad_label
                outcomes[key] = {
                    "top_relative_l2": norm((found-ref)[:, top_mask]) / norm(ref[:, top_mask]),
                    "top_max_absolute": float(np.max(np.abs((found-ref)[:, top_mask]))),
                    "closure": closure["trace_closure_absolute"],
                }
                dxpsi = ((np.roll(values, -1, axis=1)-np.roll(values, 1, axis=1))/(2*h)
                         if override is None else grad_exact[0])
                outcomes[key]["edge_decomposition"] = edge_error(
                    effective, trace, values, dxpsi, h=h, nx=nx,
                    zedge=zedge, bed=actual_bed, surface=actual_surface,
                    mode=mode, amplitude=amplitude, top_row=top_row,
                )
        variants[label] = outcomes
    return {"case": f"{family}_nx{nx}_q{fraction:g}", "variants": variants}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for spec in CASES:
        row = run_case(*spec)
        rows.append(row)
        print(row["case"], row["variants"]["parent_hermite"]["solved_finite_difference"]["top_relative_l2"],
              row["variants"]["parent_hermite"]["analytic_analytic_oracle"]["top_relative_l2"],
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = ("diagnose_parent_trace_error.py", "cutcell_transfer.py",
             "physical_trace.py", "liquid_dirichlet_transfer.py",
             "verify_parent_trace_harmonic.py")
    report = {"schema": "parent-chord-hermite-oracle-decomposition-v1",
              "oracle_is_diagnostic_not_a_method": True, "rows": rows,
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in files}}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
