"""Screen circulation/potential-first handoff reconstructions on the GPU."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


CASES = [
    {"name": "ps_h005_sv0_b0", "harmonic": 0.005, "surface_velocity": 0.0, "bottom": 0.0},
    {"name": "ps_h005_sv0_b1e4", "harmonic": 0.005, "surface_velocity": 0.0, "bottom": 1.0e4},
    {"name": "ps_h005_sv0_b1e6", "harmonic": 0.005, "surface_velocity": 0.0, "bottom": 1.0e6},
    {"name": "ps_h005_sv1_b1e4", "harmonic": 0.005, "surface_velocity": 1.0, "bottom": 1.0e4},
    {"name": "ps_h005_sv8_b1e4", "harmonic": 0.005, "surface_velocity": 8.0, "bottom": 1.0e4},
    {"name": "ps_h020_sv1_b1e4", "harmonic": 0.020, "surface_velocity": 1.0, "bottom": 1.0e4},
]


def main() -> None:
    root = Path.cwd()
    source = root / "data" / "jfm_n576_active_preimpact_handoff_x1024.npz"
    run_directory = root / "runs" / "scalar_potential_screen"
    log_directory = root / "logs" / "scalar_potential_screen"
    run_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    results: list[dict[str, object]] = []

    for case in CASES:
        name = str(case["name"])
        output_prefix = run_directory / name
        command = [
            sys.executable,
            str(root / "src" / "prepare_handoff.py"),
            str(source),
            str(output_prefix),
            "--extension", "potential_harmonic",
            "--harmonic-weight", str(case["harmonic"]),
            "--surface-weight", "64",
            "--exterior-energy-weight", "0.005",
            "--potential-bulk-velocity-weight", "0",
            "--potential-surface-velocity-weight", str(case["surface_velocity"]),
            "--potential-bottom-normal-weight", str(case["bottom"]),
            "--linear-backend", "cupy_lsqr",
            "--equilibrate-columns",
        ]
        started = time.perf_counter()
        log_path = log_directory / f"{name}.out"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.perf_counter() - started
        output_json = output_prefix.with_suffix(".json")
        metrics = (
            json.loads(output_json.read_text(encoding="utf-8"))
            if output_json.exists()
            else {}
        )
        results.append(
            {
                "case": case,
                "return_code": completed.returncode,
                "elapsed_seconds": elapsed,
                "metrics": metrics,
            }
        )
        (run_directory / "summary.json").write_text(
            json.dumps(
                {
                    "schema": "bie-to-vof-scalar-potential-screen-v1",
                    "warning": "GPU screening only; CPU and receiver validation are mandatory.",
                    "results": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if metrics:
            print(
                f"{name}: bulk={metrics['relative_bulk_velocity_fit_error']:.6g} "
                f"surface={metrics['relative_surface_velocity_fit_error']:.6g} "
                f"div={metrics['potential_divergence_linf']:.6g} "
                f"speed={metrics['maximum_liquid_to_surface_q99_speed_ratio']:.6g} "
                f"stop={metrics['least_squares_stop_code']} "
                f"iterations={metrics['least_squares_iterations']} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        else:
            print(f"{name}: failed; see {log_path}", flush=True)


if __name__ == "__main__":
    main()
