"""Run a restartable GPU screening sweep for the constrained BIE-to-VOF fit."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


CASES = [
    {"name": "hw020_sw64_e010_c150", "harmonic": 0.0200, "surface": 64, "exterior": 0.010, "cap": 1.5},
    {"name": "hw010_sw64_e010_c150", "harmonic": 0.0100, "surface": 64, "exterior": 0.010, "cap": 1.5},
    {"name": "hw005_sw64_e010_c150", "harmonic": 0.0050, "surface": 64, "exterior": 0.010, "cap": 1.5},
    {"name": "hw0025_sw64_e010_c150", "harmonic": 0.0025, "surface": 64, "exterior": 0.010, "cap": 1.5},
    {"name": "hw005_sw32_e010_c150", "harmonic": 0.0050, "surface": 32, "exterior": 0.010, "cap": 1.5},
    {"name": "hw005_sw16_e010_c150", "harmonic": 0.0050, "surface": 16, "exterior": 0.010, "cap": 1.5},
    {"name": "hw005_sw64_e005_c150", "harmonic": 0.0050, "surface": 64, "exterior": 0.005, "cap": 1.5},
    {"name": "hw005_sw64_e001_c150", "harmonic": 0.0050, "surface": 64, "exterior": 0.001, "cap": 1.5},
    {"name": "hw0025_sw32_e005_c150", "harmonic": 0.0025, "surface": 32, "exterior": 0.005, "cap": 1.5},
    {"name": "hw005_sw64_e005_c160", "harmonic": 0.0050, "surface": 64, "exterior": 0.005, "cap": 1.6},
]


def screening_gates(metrics: dict[str, object]) -> dict[str, bool]:
    bulk_fit = metrics.get(
        "post_correction_relative_bulk_velocity_fit_error",
        metrics["relative_bulk_velocity_fit_error"],
    )
    surface_fit = metrics.get(
        "post_correction_relative_surface_velocity_fit_error",
        metrics["relative_surface_velocity_fit_error"],
    )
    return {
        "mass": abs(float(metrics["relative_mass_error"])) <= 1.0e-10,
        "momentum": abs(float(metrics["relative_momentum_error"])) <= 1.0e-8,
        "divergence": abs(float(metrics["pre_projection_divergence_linf_full_liquid"])) <= 1.0e-10,
        "bulk_fit": float(bulk_fit) <= 0.10,
        "surface_fit": float(surface_fit) <= 0.05,
        "speed": float(metrics["maximum_liquid_to_surface_q99_speed_ratio"]) <= 1.5 * (1.0 + 1.0e-6),
        "active_cap": int(metrics["final_speed_cap_violation_count"]) == 0,
    }


def compact_metrics(metrics: dict[str, object]) -> dict[str, object]:
    keys = (
        "linear_solver_backend",
        "least_squares_stop_code",
        "least_squares_iterations",
        "least_squares_relative_residual",
        "relative_mass_error",
        "relative_momentum_error",
        "relative_bulk_velocity_fit_error",
        "relative_surface_velocity_fit_error",
        "post_correction_relative_bulk_velocity_fit_error",
        "post_correction_relative_surface_velocity_fit_error",
        "maximum_liquid_to_surface_q99_speed_ratio",
        "pre_projection_divergence_linf_full_liquid",
        "final_speed_cap_violation_count",
        "final_active_set_size",
    )
    return {key: metrics.get(key) for key in keys}


def write_summary(path: Path, results: list[dict[str, object]]) -> None:
    payload = {
        "schema": "bie-to-vof-gpu-screening-v1",
        "warning": (
            "GPU LSQR is a screening backend. Every shortlisted state must be "
            "rerun with SciPy LSQR and pass a field-level comparison."
        ),
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    root = Path.cwd()
    source = root / "data" / "jfm_n576_active_preimpact_handoff_x1024.npz"
    run_directory = root / "runs" / "handoff_sweep"
    log_directory = root / "logs" / "handoff_sweep"
    run_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    summary_path = run_directory / "summary.json"
    results: list[dict[str, object]] = []
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}

    for case in CASES:
        name = str(case["name"])
        output_prefix = run_directory / name
        output_json = output_prefix.with_suffix(".json")
        log_path = log_directory / f"{name}.out"
        started = time.perf_counter()
        if output_json.exists():
            metrics = json.loads(output_json.read_text(encoding="utf-8"))
            status = "reused"
            return_code = 0
        else:
            command = [
                sys.executable,
                str(root / "src" / "prepare_handoff.py"),
                str(source),
                str(output_prefix),
                "--extension", "streamfunction_bounded",
                "--harmonic-weight", str(case["harmonic"]),
                "--surface-weight", str(case["surface"]),
                "--exterior-energy-weight", str(case["exterior"]),
                "--speed-cap-ratio", str(case["cap"]),
                "--exterior-speed-cap-ratio", "2.0",
                "--speed-constraint-weight", "1000000",
                "--active-set-iterations", "8",
                "--bottom-streamfunction-weight", "1000000",
                "--momentum-constraint-weight", "1000000",
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
            metrics = (
                json.loads(output_json.read_text(encoding="utf-8"))
                if output_json.exists()
                else {}
            )
        elapsed = time.perf_counter() - started
        gates = screening_gates(metrics) if metrics else {}
        results.append(
            {
                "case": case,
                "status": status,
                "return_code": return_code,
                "elapsed_seconds": elapsed,
                "metrics": compact_metrics(metrics),
                "screening_gates": gates,
                "all_screening_gates_passed": bool(gates) and all(gates.values()),
            }
        )
        write_summary(summary_path, results)
        if return_code != 0:
            print(f"{name}: failed; see {log_path}", flush=True)
        else:
            print(
                f"{name}: bulk={metrics['relative_bulk_velocity_fit_error']:.6g} "
                f"surface={metrics['relative_surface_velocity_fit_error']:.6g} "
                f"speed={metrics['maximum_liquid_to_surface_q99_speed_ratio']:.6g} "
                f"passes={all(gates.values())} elapsed={elapsed:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
