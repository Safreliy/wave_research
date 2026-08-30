"""Decompose the constrained streamfunction handoff objective on the GPU.

These runs are diagnostic only.  They remove the nonlinear speed cap and add
one linear objective block at a time so that a large bulk-fit floor can be
attributed to a specific constraint rather than hidden by a parameter sweep.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


BASE = {
    "harmonic": 0.0,
    "surface": 0.0,
    "exterior": 0.0,
    "bottom": 0.0,
    "momentum": 0.0,
}

CASES = [
    {"name": "d0_bulk_only", **BASE},
    {"name": "d1_harmonic_only", **BASE, "harmonic": 0.005},
    {"name": "d2_surface_only", **BASE, "surface": 64.0},
    {"name": "d3_exterior_only", **BASE, "exterior": 0.005},
    {"name": "d4_bottom_only", **BASE, "bottom": 1.0e6},
    {"name": "d5_momentum_only", **BASE, "momentum": 1.0e6},
    {
        "name": "d6_all_linear",
        "harmonic": 0.005,
        "surface": 64.0,
        "exterior": 0.005,
        "bottom": 1.0e6,
        "momentum": 1.0e6,
    },
]


METRIC_KEYS = (
    "least_squares_stop_code",
    "least_squares_iterations",
    "least_squares_relative_residual",
    "relative_bulk_velocity_fit_error",
    "relative_surface_velocity_fit_error",
    "relative_mass_error",
    "relative_momentum_error",
    "relative_momentum_constraint_error",
    "bottom_streamfunction_standard_deviation",
    "bottom_streamfunction_range",
    "maximum_liquid_to_surface_q99_speed_ratio",
    "pre_projection_divergence_linf_full_liquid",
)


def write_summary(path: Path, results: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "bie-to-vof-constraint-diagnostics-v1",
                "purpose": (
                    "GPU screening ablation; publication candidates require a "
                    "SciPy LSQR rerun and field-level backend agreement."
                ),
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    root = Path.cwd()
    source = root / "data" / "jfm_n576_active_preimpact_handoff_x1024.npz"
    run_directory = root / "runs" / "constraint_diagnostics"
    log_directory = root / "logs" / "constraint_diagnostics"
    run_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    summary_path = run_directory / "summary.json"
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    results: list[dict[str, object]] = []

    for case in CASES:
        name = str(case["name"])
        output_prefix = run_directory / name
        output_json = output_prefix.with_suffix(".json")
        log_path = log_directory / f"{name}.out"
        started = time.perf_counter()
        if output_json.exists():
            return_code = 0
            status = "reused"
        else:
            command = [
                sys.executable,
                str(root / "src" / "prepare_handoff.py"),
                str(source),
                str(output_prefix),
                "--extension", "streamfunction_harmonic",
                "--harmonic-weight", str(case["harmonic"]),
                "--surface-weight", str(case["surface"]),
                "--exterior-energy-weight", str(case["exterior"]),
                "--bottom-streamfunction-weight", str(case["bottom"]),
                "--momentum-constraint-weight", str(case["momentum"]),
                "--linear-backend", "cupy_lsqr",
            ]
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            return_code = completed.returncode
            status = "completed" if return_code == 0 else "failed"
        elapsed = time.perf_counter() - started
        metrics = (
            json.loads(output_json.read_text(encoding="utf-8"))
            if output_json.exists()
            else {}
        )
        compact = {key: metrics.get(key) for key in METRIC_KEYS}
        results.append(
            {
                "case": case,
                "status": status,
                "return_code": return_code,
                "elapsed_seconds": elapsed,
                "metrics": compact,
            }
        )
        write_summary(summary_path, results)
        if metrics:
            print(
                f"{name}: bulk={metrics['relative_bulk_velocity_fit_error']:.6g} "
                f"surface={metrics['relative_surface_velocity_fit_error']:.6g} "
                f"momentum={metrics['relative_momentum_error']:.6g} "
                f"stop={metrics['least_squares_stop_code']} "
                f"iterations={metrics['least_squares_iterations']} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        else:
            print(f"{name}: failed; see {log_path}", flush=True)


if __name__ == "__main__":
    main()
