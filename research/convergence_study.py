"""Generate temporal and spatial convergence evidence for the HOS baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hos_solver import HOSConfig, HOSSolver, initial_condition, simulate


def temporal_study() -> list[dict[str, float | int]]:
    solver = HOSSolver(HOSConfig(n=64, order=3, depth=1.0))
    eta0, psi0, period = initial_condition("linear_mode", solver, 1.0e-10)
    final_time = period / 4.0
    records: list[dict[str, float | int]] = []
    previous_error: float | None = None
    for steps_per_quarter_period in (4, 8, 16, 32):
        result = simulate(
            solver,
            eta0,
            psi0,
            final_time / steps_per_quarter_period,
            final_time,
            snapshots=2,
        )
        # The exact elevation is zero at a quarter period.  Measuring there
        # exposes RK4's phase error directly instead of squaring it at a crest.
        error = float(np.linalg.norm(result["eta"][-1]) / np.linalg.norm(eta0))
        records.append(
            {
                "steps_per_quarter_period": steps_per_quarter_period,
                "dt": float(result["dt"]),
                "relative_l2_error": error,
                "error_ratio_from_previous": (
                    float(previous_error / error) if previous_error is not None else float("nan")
                ),
            }
        )
        previous_error = error
    return records


def spatial_study() -> list[dict[str, float | int]]:
    final_time = 0.5
    dt = 0.001
    amplitude = 0.02
    reference_n = 256
    reference_solver = HOSSolver(HOSConfig(n=reference_n, order=3, depth=1.0))
    eta0, psi0, _ = initial_condition("gaussian", reference_solver, amplitude)
    reference = simulate(
        reference_solver, eta0, psi0, dt, final_time, snapshots=2
    )["eta"][-1]

    records: list[dict[str, float | int]] = []
    previous_error: float | None = None
    for n in (32, 64, 128):
        solver = HOSSolver(HOSConfig(n=n, order=3, depth=1.0))
        eta0, psi0, _ = initial_condition("gaussian", solver, amplitude)
        result = simulate(solver, eta0, psi0, dt, final_time, snapshots=2)
        aligned_reference = reference[:: reference_n // n]
        error = float(
            np.linalg.norm(result["eta"][-1] - aligned_reference)
            / np.linalg.norm(aligned_reference)
        )
        records.append(
            {
                "n": n,
                "relative_l2_error_vs_n256": error,
                "error_ratio_from_previous": (
                    float(previous_error / error) if previous_error is not None else float("nan")
                ),
            }
        )
        previous_error = error
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/hos_convergence.json")
    )
    args = parser.parse_args()
    result = {"temporal": temporal_study(), "spatial": spatial_study()}
    rendered = json.dumps(result, indent=2, allow_nan=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
