"""Publication ablation: RK4 versus Jacobian-free implicit midpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from topographic_wave_solver import (
    diagnostics,
    reference_energy,
    smooth_periodic_shoal,
    step_implicit_midpoint,
    step_rk4,
)


def run_case(
    integrator: str,
    dt: float,
    *,
    n: int,
    final_time: float,
    mode: int,
    amplitude: float,
    depth: float,
    gravity: float,
) -> dict[str, float | int | bool | str]:
    length = 2.0 * math.pi
    x = np.arange(n) * length / n
    initial_x = x.copy()
    z = amplitude * np.cos(mode * x)
    potential = np.zeros(n)
    bottom_x, bottom_z = smooth_periodic_shoal(
        n, length, depth, 0.0, 0.0
    )
    initial_energy = diagnostics(
        x, z, potential, bottom_x, bottom_z, length, gravity, 0.0
    )[1]
    energy_reference = reference_energy(
        bottom_x, bottom_z, length, gravity, 0.0
    )
    steps = int(math.ceil(final_time / dt))
    actual_dt = final_time / steps
    rhs_evaluations = 0
    newton_iterations = 0
    fallback_count = 0
    completed = True
    start = time.perf_counter()
    try:
        for _ in range(steps):
            if integrator == "rk4":
                x, z, potential, _ = step_rk4(
                    x,
                    z,
                    potential,
                    bottom_x,
                    bottom_z,
                    length,
                    gravity,
                    actual_dt,
                    0.0,
                )
                rhs_evaluations += 4
            else:
                result = step_implicit_midpoint(
                    x,
                    z,
                    potential,
                    bottom_x,
                    bottom_z,
                    length,
                    gravity,
                    actual_dt,
                    0.0,
                    nonlinear_tolerance=2.0e-10,
                    allow_explicit_fallback=False,
                )
                x, z, potential = result[:3]
                nonlinear = result[4].nonlinear
                rhs_evaluations += nonlinear.rhs_evaluations
                newton_iterations += nonlinear.newton_iterations
                fallback_count += int(result[4].used_explicit_fallback)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError):
        completed = False
    wall_time = time.perf_counter() - start
    if completed and np.all(np.isfinite(z)):
        omega = math.sqrt(gravity * mode * math.tanh(mode * depth))
        exact = amplitude * np.cos(mode * initial_x) * math.cos(omega * final_time)
        relative_shape_error = float(
            np.linalg.norm(z - exact)
            / max(np.linalg.norm(exact), amplitude * math.sqrt(n) * 1.0e-4)
        )
        final_energy = diagnostics(
            x, z, potential, bottom_x, bottom_z, length, gravity, 0.0
        )[1]
        energy_drift = abs(final_energy - initial_energy) / max(
            abs(initial_energy - energy_reference), np.finfo(float).eps
        )
        maximum_elevation = float(np.max(np.abs(z)))
    else:
        relative_shape_error = float("inf")
        energy_drift = float("inf")
        maximum_elevation = float("inf")
    stable = bool(
        completed
        and math.isfinite(maximum_elevation)
        and maximum_elevation < 10.0 * amplitude
    )
    return {
        "integrator": integrator,
        "dt": actual_dt,
        "steps": steps,
        "completed": completed,
        "stable": stable,
        "relative_shape_error": relative_shape_error,
        "relative_energy_drift": energy_drift,
        "maximum_elevation": maximum_elevation,
        "wall_time_seconds": wall_time,
        "rhs_evaluations": rhs_evaluations,
        "newton_iterations": newton_iterations,
        "fallback_count": fallback_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--mode", type=int, default=12)
    parser.add_argument("--amplitude", type=float, default=1.0e-5)
    parser.add_argument("--dt", type=float, nargs="+", default=[0.2, 0.5, 1.0])
    args = parser.parse_args()
    cases = [
        run_case(
            integrator,
            dt,
            n=args.n,
            final_time=args.final_time,
            mode=args.mode,
            amplitude=args.amplitude,
            depth=1.0,
            gravity=1.0,
        )
        for dt in args.dt
        for integrator in ("rk4", "implicit_midpoint")
    ]
    payload = {
        "problem": "small-amplitude finite-depth Fourier mode",
        "n": args.n,
        "mode": args.mode,
        "amplitude": args.amplitude,
        "final_time": args.final_time,
        "linear_frequency": math.sqrt(args.mode * math.tanh(args.mode)),
        "cases": cases,
        "interpretation": (
            "Stability is distinct from accuracy: midpoint can remain bounded "
            "outside the explicit RK4 stability region while accumulating phase error."
        ),
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    colors = {"rk4": "#cc4b37", "implicit_midpoint": "#176b87"}
    labels = {"rk4": "explicit RK4", "implicit_midpoint": "implicit midpoint"}
    canvas = Image.new("RGB", (1440, 480), "#f7f5ef")
    draw = ImageDraw.Draw(canvas)
    draw.text((38, 20), "Time-integration ablation · high Fourier mode", fill="#17202a")
    panels = (
        ("relative shape error", "relative_shape_error"),
        ("relative energy drift", "relative_energy_drift"),
        ("Euler–BIE RHS evaluations", "rhs_evaluations"),
    )
    for panel, (title, key) in enumerate(panels):
        left = 48 + panel * 472
        top, right, bottom = 75, 430 + panel * 472, 410
        draw.rectangle((left, top, right, bottom), outline="#a9adb3", width=1)
        draw.text((left, 50), title, fill="#17202a")
        finite_values = [
            float(case[key]) for case in cases
            if math.isfinite(float(case[key])) and float(case[key]) > 0.0
        ]
        log_min = math.log10(min(finite_values))
        log_max = math.log10(max(finite_values))
        if log_max - log_min < 1.0e-12:
            log_max = log_min + 1.0
        dt_min, dt_max = min(args.dt), max(args.dt)
        for integrator in colors:
            subset = [case for case in cases if case["integrator"] == integrator]
            points = []
            for case in subset:
                value = float(case[key])
                if not math.isfinite(value) or value <= 0.0:
                    continue
                px = left + 18 + (float(case["dt"]) - dt_min) / (dt_max - dt_min) * (right - left - 36)
                py = bottom - 18 - (math.log10(value) - log_min) / (log_max - log_min) * (bottom - top - 36)
                points.append((px, py))
            if len(points) > 1:
                draw.line(points, fill=colors[integrator], width=3)
            for point in points:
                draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=colors[integrator])
        draw.text((left + 8, bottom + 10), f"dt: {dt_min:g} … {dt_max:g}", fill="#59636e")
        draw.text((left + 8, top + 7), f"10^{log_max:.1f}", fill="#59636e")
        draw.text((left + 8, bottom - 22), f"10^{log_min:.1f}", fill="#59636e")
    draw.text((1040, 25), "red: explicit RK4", fill=colors["rk4"])
    draw.text((1190, 25), "blue: implicit midpoint", fill=colors["implicit_midpoint"])
    canvas.save(args.output_prefix.with_suffix(".png"))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
