"""Benchmark the CPU/GPU sparse least-squares kernels used by BIE-to-VOF transfer."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as cpsparse
from cupyx.scipy.sparse.linalg import lsmr as gpu_lsmr
import numpy as np
from scipy.sparse import vstack
from scipy.sparse.linalg import lsqr as cpu_lsqr


def synchronize() -> None:
    cp.cuda.Stream.null.synchronize()


def timed(callable_, *, synchronize_gpu: bool = False):
    if synchronize_gpu:
        synchronize()
    start = time.perf_counter()
    result = callable_()
    if synchronize_gpu:
        synchronize()
    return result, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--nz", type=int, default=96)
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # Import after argument parsing so this script can live outside src/ on the host.
    from prepare_handoff import (  # pylint: disable=import-outside-toplevel
        centered_derivative_operators,
        standard_laplacian_operator,
    )

    dx = 20.0 / args.nx
    dz = 3.0 / (args.nz - 1)
    derivative_x, derivative_z = centered_derivative_operators(
        args.nx, args.nz, dx, dz
    )
    laplacian = standard_laplacian_operator(args.nx, args.nz, dx, dz)
    system = vstack(
        (derivative_z, -derivative_x, 0.02 * laplacian), format="csr"
    )

    grid_x = np.arange(args.nx) * dx
    grid_z = np.arange(args.nz) * dz
    xx, zz = np.meshgrid(grid_x, grid_z)
    expected = (
        np.sin(2.0 * np.pi * xx / 20.0) * np.cos(np.pi * zz / 3.0)
        + 0.15 * np.sin(6.0 * np.pi * xx / 20.0)
    ).ravel()
    rhs = system @ expected
    rhs_norm = max(float(np.linalg.norm(rhs)), np.finfo(float).eps)

    cpu_result, cpu_seconds = timed(
        lambda: cpu_lsqr(
            system,
            rhs,
            atol=args.tolerance,
            btol=args.tolerance,
            iter_lim=args.maxiter,
            show=False,
        )
    )
    cpu_residual = float(np.linalg.norm(system @ cpu_result[0] - rhs) / rhs_norm)

    transfer_start = time.perf_counter()
    gpu_system = cpsparse.csr_matrix(system)
    gpu_rhs = cp.asarray(rhs)
    synchronize()
    transfer_seconds = time.perf_counter() - transfer_start

    gpu_result, gpu_seconds = timed(
        lambda: gpu_lsmr(
            gpu_system,
            gpu_rhs,
            atol=args.tolerance,
            btol=args.tolerance,
            maxiter=args.maxiter,
        ),
        synchronize_gpu=True,
    )
    gpu_solution = cp.asnumpy(gpu_result[0])
    gpu_residual = float(np.linalg.norm(system @ gpu_solution - rhs) / rhs_norm)

    properties = cp.cuda.runtime.getDeviceProperties(0)
    report = {
        "schema": "wave-simulation-gpu-smoke-v1",
        "host": platform.node(),
        "gpu": properties["name"].decode(),
        "cupy": cp.__version__,
        "numpy": np.__version__,
        "grid": {"nx": args.nx, "nz": args.nz, "unknowns": system.shape[1]},
        "system": {"rows": system.shape[0], "columns": system.shape[1], "nnz": system.nnz},
        "limits": {"maximum_iterations": args.maxiter, "tolerance": args.tolerance},
        "cpu_lsqr": {
            "seconds": cpu_seconds,
            "iterations": int(cpu_result[2]),
            "stop_code": int(cpu_result[1]),
            "relative_residual": cpu_residual,
        },
        "gpu_lsmr": {
            "transfer_seconds": transfer_seconds,
            "solve_seconds": gpu_seconds,
            "iterations": int(gpu_result[2]),
            "stop_code": int(gpu_result[1]),
            "relative_residual": gpu_residual,
        },
        "solve_speedup": cpu_seconds / gpu_seconds,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
