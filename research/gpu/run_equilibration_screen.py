"""Screen right-diagonal column equilibration on representative handoff fits."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


CASES = [
    {
        "name": "eq_bulk_only",
        "extension": "streamfunction_harmonic",
        "harmonic": 0.0,
        "surface": 0.0,
        "exterior": 0.0,
        "bottom": 0.0,
        "momentum": 0.0,
    },
    {
        "name": "eq_all_linear",
        "extension": "streamfunction_harmonic",
        "harmonic": 0.005,
        "surface": 64.0,
        "exterior": 0.005,
        "bottom": 1.0e6,
        "momentum": 1.0e6,
    },
    {
        "name": "eq_bounded_best_screen",
        "extension": "streamfunction_bounded",
        "harmonic": 0.0025,
        "surface": 32.0,
        "exterior": 0.005,
        "bottom": 1.0e6,
        "momentum": 1.0e6,
        "cap": 1.5,
    },
]


def main() -> None:
    root = Path.cwd()
    source = root / "data" / "jfm_n576_active_preimpact_handoff_x1024.npz"
    run_directory = root / "runs" / "equilibration_screen"
    log_directory = root / "logs" / "equilibration_screen"
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
            "--extension", str(case["extension"]),
            "--harmonic-weight", str(case["harmonic"]),
            "--surface-weight", str(case["surface"]),
            "--exterior-energy-weight", str(case["exterior"]),
            "--bottom-streamfunction-weight", str(case["bottom"]),
            "--momentum-constraint-weight", str(case["momentum"]),
            "--linear-backend", "cupy_lsqr",
            "--equilibrate-columns",
        ]
        if case["extension"] == "streamfunction_bounded":
            command.extend(
                [
                    "--speed-cap-ratio", str(case["cap"]),
                    "--exterior-speed-cap-ratio", "2.0",
                    "--speed-constraint-weight", "1000000",
                    "--active-set-iterations", "8",
                ]
            )
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
                    "schema": "bie-to-vof-column-equilibration-screen-v1",
                    "warning": "GPU screening only; CPU reference rerun is mandatory.",
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
                f"stop={metrics['least_squares_stop_code']} "
                f"iterations={metrics['least_squares_iterations']} "
                f"residual={metrics['least_squares_relative_residual']:.6g} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        else:
            print(f"{name}: failed; see {log_path}", flush=True)


if __name__ == "__main__":
    main()
