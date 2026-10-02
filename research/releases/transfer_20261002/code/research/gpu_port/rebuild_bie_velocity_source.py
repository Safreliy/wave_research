"""Rebuild a handoff's velocities using independently refined GPU quadrature.

Preserves the BIE state and geometry; changes only interior/face evaluation.
Legacy scalar bulk potentials are removed rather than silently mixed with the
new velocities. No interpolation or speed clipping repairs rejected samples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bie_interior_evaluation import boundary_quadrature, interior_velocity
from campaign_integrity import sha256


def boundary_flux_preflight(data, length=64.0):
    """Check a necessary divergence-theorem identity before costly sampling.

    Reuse the mixed Neumann solver's 5e-9 relative compatibility tolerance.
    Oversampling evaluates the saved traces; it does not repair their flux.
    """
    rows = []
    for factor in (4, 8, 16):
        qds = boundary_quadrature(data, length, factor)[4]
        net = float(np.sum(qds))
        scale = max(1.0, float(np.sum(np.abs(qds))))
        rows.append(
            {
                "factor": factor,
                "net_flux": net,
                "relative_flux_imbalance": abs(net) / scale,
                "pass": bool(np.isfinite(net) and abs(net) <= 5e-9 * scale),
            }
        )
    return {
        "checks": rows,
        "pass": all(row["pass"] for row in rows[-2:]),
        "meaning": "Necessary boundary compatibility, not full source qualification",
    }


def converged_velocity(data, x, z, xp, label, *, atol=1e-7, rtol=1e-5):
    """Require two consecutive source-factor agreement checks at each point."""
    pending = np.arange(len(x))
    accepted = np.zeros(len(x), dtype=bool)
    values = np.full((2, len(x)), np.nan)
    previous = None
    streak = np.zeros(len(x), dtype=int)
    history = []
    for factor in (4, 8, 16, 32, 64, 128, 256):
        current = interior_velocity(
            data,
            x[pending],
            z[pending],
            64,
            factor,
            xp=xp,
            chunk=min(128, 16384 // factor),
        )
        if previous is not None:
            good = np.max(np.abs(current - previous), axis=0) <= atol + rtol * np.max(
                np.abs(current), axis=0
            )
            streak[pending] = np.where(good, streak[pending] + 1, 0)
            done = streak[pending] >= 2
            values[:, pending[done]] = current[:, done]
            accepted[pending[done]] = True
            pending = pending[~done]
            previous = current[:, ~done]
        else:
            previous = current
        history.append({"factor": factor, "remaining": len(pending)})
        print(json.dumps({"label": label, **history[-1]}), flush=True)
        if not len(pending):
            break
    return (
        values,
        accepted,
        {
            "atol": atol,
            "rtol": rtol,
            "history": history,
            "accepted": int(accepted.sum()),
            "rejected": int((~accepted).sum()),
        },
    )


def rebuild(source, output):
    import bie_interior_evaluation as evaluator

    data = dict(np.load(source))
    if float(data["background_current"]) != 0:
        raise ValueError("audit rebuild currently requires zero current")
    flux_check = boundary_flux_preflight(data)
    if not flux_check["pass"]:
        raise ValueError(
            "BIE boundary violates global flux compatibility; re-solve/refine the boundary before rebuilding velocities. "
            + json.dumps(flux_check)
        )
    import cupy as cp

    x, z, q = data["grid_x"], data["grid_z"], data["volume_fraction"]
    dx, dz = float(x[1] - x[0]), float(z[1] - z[0])
    xx, zz = np.meshgrid(x, z)
    bx, bz, *_ = boundary_quadrature(data, 64, 64)
    half = len(bx) // 2
    max_segment = 0.0
    for start in (0, half):
        sx, sz = bx[start : start + half], bz[start : start + half]
        ddx = (np.roll(sx, -1) - sx + 32) % 64 - 32
        max_segment = max(max_segment, float(np.hypot(ddx, np.roll(sz, -1) - sz).max()))
    points = np.c_[bx, bz]
    tree = cKDTree(np.vstack([points + [shift, 0] for shift in (-64, 0, 64)]))
    selected = np.flatnonzero(q.ravel() > 0.999)
    distance, _ = tree.query(
        np.c_[xx.ravel()[selected], zz.ravel()[selected]], workers=-1
    )
    # Conservative distance guard relative to sampled polygon; its spectral
    # approximation remains separately testable, not an exact-curve proof.
    selected = selected[distance - max_segment / 2 > 1.75 * max(dx, dz)]
    uv, good, bulk_audit = converged_velocity(
        data, xx.ravel()[selected], zz.ravel()[selected], cp, "bulk"
    )
    bulk_fields = np.full((2, q.size), np.nan)
    bulk_fields[:, selected[good]] = uv[:, good]
    valid = np.zeros(q.size, dtype=bool)
    valid[selected[good]] = True
    valid = valid.reshape(q.shape)
    data["velocity_x"], data["velocity_z"] = bulk_fields.reshape((2,) + q.shape)
    data["velocity_valid"] = valid
    data.pop("bulk_potential", None)
    data.pop("potential_valid", None)
    face_reports = {}
    for direction, name in ((0, "x"), (1, "z")):
        if direction == 0:
            mask = valid & np.roll(valid, 1, axis=1)
        else:
            mask = np.zeros((q.shape[0] + 1, q.shape[1]), dtype=bool)
            mask[1:-1] = valid[:-1] & valid[1:]
        rows, cols = np.nonzero(mask)
        face_valid = np.ones(len(rows), dtype=bool)
        estimates = {}
        node_audits = []
        for order in (2, 4):
            estimate = np.zeros(len(rows))
            nodes, weights = np.polynomial.legendre.leggauss(order)
            for node, weight in zip(nodes, weights):
                tx = x[cols] - dx / 2 if direction == 0 else x[cols] + dx / 2 * node
                tz = (
                    z[rows] + dz / 2 * node
                    if direction == 0
                    else z[0] - dz / 2 + rows * dz
                )
                value, good, audit = converged_velocity(
                    data, tx, tz, cp, f"face_{name}_{order}_{node:.3f}"
                )
                face_valid &= good
                estimate += weight / 2 * np.where(good, value[direction], 0)
                node_audits.append(audit)
            estimates[order] = estimate
        relative = float(
            np.linalg.norm((estimates[2] - estimates[4])[face_valid])
            / max(np.linalg.norm(estimates[4][face_valid]), 1e-30)
        )
        field = np.full(mask.shape, np.nan)
        field[rows[face_valid], cols[face_valid]] = estimates[4][face_valid]
        final_mask = np.zeros(mask.shape, dtype=bool)
        final_mask[rows[face_valid], cols[face_valid]] = True
        data[f"face_velocity_{name}"] = field
        data[f"face_velocity_{name}_valid"] = final_mask
        face_reports[name] = {
            "gauss2_vs4_relative_l2": relative,
            "passes_1e3_gate": relative <= 1e-3,
            "accepted_faces": int(face_valid.sum()),
            "node_audits": node_audits,
        }
    report = {
        "schema": "bie-refined-velocity-source-v1",
        "boundary_flux_preflight": flux_check,
        "source_sha256": sha256(source),
        "evaluator_sha256": sha256(Path(evaluator.__file__)),
        "generator_sha256": sha256(Path(__file__)),
        "bulk": bulk_audit,
        "faces": face_reports,
        "coverage_liquid_volume_fraction": float(q[valid].sum() / q.sum()),
        "valid_only_kinetic_energy": float(
            0.5
            * dx
            * dz
            * np.sum(
                q[valid]
                * np.sum(bulk_fields.reshape((2,) + q.shape)[:, valid] ** 2, axis=0)
            )
        ),
        "maximum_valid_bulk_speed": float(
            np.max(np.hypot(data["velocity_x"][valid], data["velocity_z"][valid]))
        ),
        "fine_boundary_maximum_segment": max_segment,
        "physical_accuracy_qualified": False,
        "limits": [
            "Refined-quadrature candidate; not a qualified continuum reference.",
            "Missing near-boundary data still require a tested reconstruction.",
            "Two successive source-quadrature agreements are numerical evidence, not a rigorous error bound.",
        ],
    }
    data["velocity_evaluation_schema"] = np.asarray(report["schema"])
    data["velocity_note"] = np.asarray(
        "Analytic kernel gradient; two successive source-quadrature checks; 4-point face Gauss rule; invalid values remain NaN"
    )
    data["velocity_evaluation_report"] = np.asarray(json.dumps(report))
    np.savez_compressed(output.with_suffix(".npz"), **data)
    report["output_sha256"] = sha256(output.with_suffix(".npz"))
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.with_suffix(".npz").exists()
        or args.output.with_suffix(".json").exists()
    ):
        raise FileExistsError("Refusing to overwrite a source rebuild")
    rebuild(args.source, args.output)
