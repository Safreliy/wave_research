"""Separate in-memory transfer stages from serialization and native CFD cost."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import scipy
import shapely

import liquid_dirichlet_transfer as ldt
from prepare_aphros_initial_state import DOMAIN_LENGTH, EMBEDDED_BOUNDARY_OFFSET, VERTICAL_ORIGIN
from prepare_flat_dynamic_benchmark import (BED_SOLVER_Y, SURFACE_SOLVER_Y,
                                            harmonic_source)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values: list[float]) -> dict:
    return {"samples_seconds": values, "median_seconds": statistics.median(values),
            "minimum_seconds": min(values), "maximum_seconds": max(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nx", type=int, choices=(512, 1024, 2048), required=True)
    parser.add_argument("--trace-factor", type=int, choices=(8, 16, 32), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        raise ValueError("at least three measured repetitions required")
    nx = args.nx
    h = DOMAIN_LENGTH / nx
    ny = nx // 16
    x = np.arange(nx) * h
    zedge = VERTICAL_ORIGIN + np.arange(ny + 1) * h
    bed = VERTICAL_ORIGIN + BED_SOLVER_Y + EMBEDDED_BOUNDARY_OFFSET
    surface = VERTICAL_ORIGIN + SURFACE_SOLVER_Y
    source = harmonic_source(bed, surface, 512)
    original = {"solve": ldt.solve,
                "gas": ldt.extend_gas,
                "moments": ldt.fitted_moments}
    samples = {name: [] for name in ("trace", "total", "assembly", "solve", "gas", "moments", "other")}
    active = {}

    def timed(name, function):
        def wrapper(*func_args, **func_kwargs):
            start = time.perf_counter()
            output = function(*func_args, **func_kwargs)
            active[name] += time.perf_counter() - start
            return output
        return wrapper

    ldt.solve = timed("solve", original["solve"])
    ldt.extend_gas = timed("gas", original["gas"])
    ldt.fitted_moments = timed("moments", original["moments"])
    checks = []
    try:
        for iteration in range(args.repeats + 1):
            active = {"solve": 0.0, "gas": 0.0, "moments": 0.0}
            start = time.perf_counter()
            curves, _ = ldt.boundary_traces(source, factor=args.trace_factor)
            trace_seconds = time.perf_counter() - start
            start = time.perf_counter()
            data, report = ldt.transfer(
                curves, x, zedge, backend="numpy", gas=True,
                continuation_order=2, cutcell_moments=True,
                physical_trace="parent_hermite_matched")
            transfer_seconds = time.perf_counter() - start
            q = data["volume_fraction"]
            checks.append({"volume": float(q.sum() * h*h),
                           "velocity_sha256": hashlib.sha256(
                               np.stack((data["cell_velocity_x_projected"],
                                         data["cell_velocity_z_projected"])).tobytes()).hexdigest()})
            if iteration:
                samples["trace"].append(trace_seconds)
                samples["total"].append(transfer_seconds)
                samples["assembly"].append(float(report["assembly_seconds"]))
                for name in ("solve", "gas", "moments"):
                    samples[name].append(active[name])
                samples["other"].append(transfer_seconds - report["assembly_seconds"]
                                        - sum(active.values()))
    finally:
        ldt.solve, ldt.extend_gas, ldt.fitted_moments = (
            original["solve"], original["gas"], original["moments"])
    if len({row["velocity_sha256"] for row in checks}) != 1:
        raise ValueError("timed transfer changed across repeats")
    report = {"schema": "auditor-transfer-cost-v1", "nx": nx, "ny": ny,
              "trace_refinement_factor": args.trace_factor,
              "warmup_runs": 1, "measured_runs": args.repeats,
              "hardware": {"machine": platform.node(), "processor": platform.processor(),
                           "operating_system": platform.platform()},
              "libraries": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "shapely": shapely.__version__},
              "stage_definition": {
                  "trace": "boundary data resampling and streamfunction trace, in memory",
                  "assembly": "geometry reconciliation and sparse harmonic-system assembly",
                  "solve": "harmonic liquid streamfunction solve only",
                  "gas": "gas continuation solve",
                  "moments": "fitted liquid-cell moment construction",
                  "other": "remaining in-memory steps including aperture and flux assembly",
                  "total": "transfer() wall time, excludes trace, serialization, checkpoint and CFD"},
              "stages": {name: summary(values) for name, values in samples.items()},
              "repeat_check": checks,
              "source_sha256": {name: sha(Path(__file__).with_name(name))
                                for name in ("liquid_dirichlet_transfer.py",
                                             "cutcell_transfer.py", "physical_trace.py")},
              "benchmark_script_sha256": sha(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"nx": nx, "median_seconds":
                      {name: data["median_seconds"] for name, data in report["stages"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
