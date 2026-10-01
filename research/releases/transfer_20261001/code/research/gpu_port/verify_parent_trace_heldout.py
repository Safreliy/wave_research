"""Held-out flat-strip transition suite, mode 7 and q=0.3/0.7/0.99."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import shapely

from liquid_dirichlet_transfer import boundary_traces, transfer
from prepare_aphros_initial_state import VERTICAL_ORIGIN, EMBEDDED_BOUNDARY_OFFSET
from verify_parent_trace_harmonic import (LENGTH, centroid_curl,
                                          exact_phase_means, relative_l2, source)


MODE, AMPLITUDE, NTRACE, TRACE_FACTOR = 7, 0.02, 512, 8
FRACTIONS = (0.3, 0.7, 0.99)
LEVELS = (512, 1024)
METHODS = ("linear", "parent_hermite", "parent_hermite_matched")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_case(nx, fraction, root):
    h = LENGTH / nx
    xedge = np.arange(nx) * h
    zedge = VERTICAL_ORIGIN + np.arange(nx // 16 + 1) * h
    bed = VERTICAL_ORIGIN + 0.53 + EMBEDDED_BOUNDARY_OFFSET
    top_row = int(np.floor(1.5000625 / h))
    surface = zedge[top_row] + fraction * h
    src = source(bed, surface, mode=MODE, amplitude=AMPLITUDE, ntrace=NTRACE)
    curves, _ = boundary_traces(src, factor=TRACE_FACTOR)
    data, reports, timings = {}, {}, {}
    for method in METHODS:
        start = time.perf_counter()
        data[method], reports[method] = transfer(
            curves, xedge, zedge, gas=False, continuation_order=2,
            cutcell_moments=True, physical_trace=method,
        )
        timings[method] = time.perf_counter() - start
    q = data["linear"]["volume_fraction"]
    if any(not np.array_equal(q, data[m]["volume_fraction"]) for m in METHODS):
        raise ValueError("held-out methods changed geometry")
    lo = np.maximum(zedge[:-1], bed)
    hi = np.minimum(zedge[1:], surface)
    wet = hi > lo
    expected_q = np.broadcast_to(np.maximum(hi-lo, 0)[:, None] / h, q.shape)
    if np.max(np.abs(q-expected_q)) > 1e-10:
        raise ValueError("held-out native geometry mismatch")
    exact = np.zeros((2, len(lo), nx))
    exact[:, wet] = exact_phase_means(data["linear"]["grid_x"],
                                      lo[wet], hi[wet], bed, surface,
                                      MODE, AMPLITUDE, h)
    top = np.zeros_like(q, dtype=bool)
    top[top_row] = q[top_row] > 0
    arrays = {"q": q, "exact_phase_mean": exact}
    metrics = {}
    for method in METHODS:
        current = data[method]
        fitted = np.array([current["cell_velocity_x_projected"],
                           current["cell_velocity_z_projected"]])
        centroid = centroid_curl(current["mac_streamfunction"], zedge, lo, hi, h)
        arrays[method + "_fitted"] = fitted
        arrays[method + "_same_psi_centroid"] = centroid
        metrics[method] = {}
        for label, field in (("fitted", fitted), ("same_psi_centroid", centroid)):
            metrics[method][label] = {
                "top_relative_l2": relative_l2(field, exact, q, top, False),
                "top_q_weighted_relative_l2": relative_l2(field, exact, q, top, True),
                "all_liquid_q_weighted_relative_l2": relative_l2(field, exact, q, q>0, True),
            }
    path = root / f"mode7_nx{nx}_q{fraction:g}.npz"
    np.savez_compressed(path, **arrays)
    return {"nx": nx, "fraction": fraction, "mode": MODE,
            "source_nodes": NTRACE, "trace_factor": TRACE_FACTOR,
            "surface": surface, "bed": bed, "q_top_actual": float(q[top].mean()),
            "metrics": metrics, "timing_seconds": timings,
            "closure": {m: reports[m]["cutcell_moments"]["trace_closure_absolute"]
                        for m in METHODS},
            "arrays": path.name, "arrays_sha256": digest(path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    rows = []
    for nx in LEVELS:
        for q in FRACTIONS:
            row = run_case(nx, q, args.root)
            rows.append(row)
            print(json.dumps({"case": row["arrays"],
                              "top": {m: row["metrics"][m]["fitted"]["top_relative_l2"]
                                      for m in METHODS}}), flush=True)
    code = Path(__file__)
    files = ("verify_parent_trace_heldout.py", "verify_parent_trace_harmonic.py",
             "physical_trace.py", "cutcell_transfer.py", "liquid_dirichlet_transfer.py")
    report = {"schema": "parent-chord-hermite-heldout-mode7-v1",
              "rows": rows,
              "protocol_sha256": digest(code.parents[1] / "results/method_trace_hermite_20261001/HELDOUT_PROTOCOL.md"),
              "source_sha256": {name: digest(code.with_name(name)) for name in files},
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "scipy": scipy.__version__, "shapely": shapely.__version__}}
    (args.root / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
