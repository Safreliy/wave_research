"""Publication-oriented wave scenarios for the verified HOS baseline.

The initial data in this module are deliberately separated from ``hos_solver``:
the solver is a numerical baseline, while a scenario defines a physical test and
its observables.  None of the approximate initial conditions below is presented
as a new water-wave solution.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hos_solver import HOSConfig, HOSSolver, relative_drift, simulate, write_svg


Array = np.ndarray


@dataclass(frozen=True)
class Scenario:
    name: str
    eta: Array
    psi: Array
    final_time: float
    event_time: float | None
    event_x: float | None
    description: str
    approximation: str


def _dispersion(solver: HOSSolver, mode: int) -> tuple[float, float]:
    k = 2.0 * math.pi * mode / solver.config.length
    omega = math.sqrt(
        solver.config.gravity * k * math.tanh(k * solver.config.depth)
    )
    return k, omega


def standing_wave(
    solver: HOSSolver, amplitude: float, mode: int = 1
) -> Scenario:
    """A sinusoidal standing wave released from rest at maximum elevation."""
    k, omega = _dispersion(solver, mode)
    eta = amplitude * np.cos(k * solver.x)
    period = 2.0 * math.pi / omega
    return Scenario(
        name="standing_wave",
        eta=eta,
        psi=np.zeros_like(eta),
        final_time=period,
        event_time=0.5 * period,
        event_x=None,
        description="Standing gravity wave over a flat bed",
        approximation="Airy eigenmode; nonlinear evolution is computed by HOS",
    )


def dispersive_focusing(
    solver: HOSSolver,
    target_crest: float,
    center_mode: int = 6,
    half_width: int = 3,
    focus_time: float = 1.5,
    focus_x: float | None = None,
) -> Scenario:
    """Create a right-going packet whose *linear* phases meet at one crest.

    ``target_crest`` is the sum of modal amplitudes at the prescribed linear
    focus.  The nonlinear focus shift is therefore an output, not fitted data.
    """
    if center_mode <= half_width or half_width < 1:
        raise ValueError("require center_mode > half_width >= 1")
    if target_crest <= 0.0 or focus_time <= 0.0:
        raise ValueError("target_crest and focus_time must be positive")
    if focus_x is None:
        focus_x = 0.5 * solver.config.length

    modes = np.arange(center_mode - half_width, center_mode + half_width + 1)
    sigma = max(0.55 * half_width, 1.0)
    weights = np.exp(-0.5 * ((modes - center_mode) / sigma) ** 2)
    modal_amplitudes = target_crest * weights / np.sum(weights)

    eta = np.zeros_like(solver.x)
    psi = np.zeros_like(solver.x)
    for mode, modal_amplitude in zip(modes, modal_amplitudes):
        k, omega = _dispersion(solver, int(mode))
        phase = k * (solver.x - focus_x) + omega * focus_time
        eta += modal_amplitude * np.cos(phase)
        psi += (
            modal_amplitude
            * omega
            / (k * math.tanh(k * solver.config.depth))
            * np.sin(phase)
        )

    return Scenario(
        name="dispersive_focusing",
        eta=eta,
        psi=psi,
        final_time=2.0 * focus_time,
        event_time=focus_time,
        event_x=focus_x,
        description=(
            f"Focused right-going packet, modes {modes[0]}--{modes[-1]}"
        ),
        approximation="Linear phase design; HOS measures nonlinear focus shift",
    )


def _periodic_distance(x: Array, center: float, length: float) -> Array:
    return (x - center + 0.5 * length) % length - 0.5 * length


def _spectral_antiderivative(solver: HOSSolver, derivative: Array) -> Array:
    coefficients = np.fft.fft(derivative - np.mean(derivative))
    multiplier = np.zeros_like(solver.k, dtype=complex)
    nonzero = solver.k != 0.0
    multiplier[nonzero] = 1.0 / (1j * solver.k[nonzero])
    primitive = np.fft.ifft(multiplier * coefficients).real
    return primitive - np.mean(primitive)


def head_on_solitary_pair(
    solver: HOSSolver,
    amplitude: float,
    separation_fraction: float = 0.40,
) -> Scenario:
    """Two counter-propagating KdV solitary-wave approximations.

    This is useful as a convergence and collision benchmark at small ``a/h``.
    It is not an exact Euler solitary wave; publication runs should replace it
    by a numerically continued fully nonlinear solitary-wave profile.
    """
    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive")
    depth = solver.config.depth
    length = solver.config.length
    if amplitude / depth > 0.2:
        raise ValueError("KdV initializer requires amplitude/depth <= 0.2")
    if not 0.2 <= separation_fraction <= 0.7:
        raise ValueError("separation_fraction must lie in [0.2, 0.7]")

    separation = separation_fraction * length
    left_center = 0.5 * (length - separation)
    right_center = 0.5 * (length + separation)
    kappa = math.sqrt(3.0 * amplitude / (4.0 * depth**3))
    celerity = math.sqrt(solver.config.gravity * (depth + amplitude))

    rightgoing = amplitude / np.cosh(
        kappa * _periodic_distance(solver.x, left_center, length)
    ) ** 2
    leftgoing = amplitude / np.cosh(
        kappa * _periodic_distance(solver.x, right_center, length)
    ) ** 2
    eta = rightgoing + leftgoing
    eta -= np.mean(eta)

    # Leading shallow-water relation u = +/- c eta / h, followed by psi_x=u.
    surface_u = celerity * (rightgoing - leftgoing) / depth
    psi = _spectral_antiderivative(solver, surface_u)
    collision_time = 0.5 * separation / celerity
    return Scenario(
        name="head_on_solitary_pair",
        eta=eta,
        psi=psi,
        final_time=2.0 * collision_time,
        event_time=collision_time,
        event_x=0.5 * length,
        description="Head-on collision of two equal solitary-wave approximations",
        approximation="Leading-order KdV profile and shallow-water surface velocity",
    )


def make_scenario(name: str, solver: HOSSolver, amplitude: float) -> Scenario:
    if name == "standing_wave":
        return standing_wave(solver, amplitude)
    if name == "dispersive_focusing":
        return dispersive_focusing(solver, amplitude)
    if name == "head_on_solitary_pair":
        return head_on_solitary_pair(solver, amplitude)
    raise ValueError(f"unknown scenario: {name}")


def scenario_metrics(
    scenario: Scenario, result: dict[str, Array]
) -> dict[str, float | str | None]:
    crest = np.max(result["eta"], axis=1)
    trough = np.min(result["eta"], axis=1)
    crest_index = int(np.argmax(crest))
    event_time_error = None
    if scenario.event_time is not None:
        event_time_error = float(result["time"][crest_index] - scenario.event_time)
    metrics: dict[str, float | str | None] = {
        "description": scenario.description,
        "initialization": scenario.approximation,
        "expected_event_time": scenario.event_time,
        "observed_max_crest_time": float(result["time"][crest_index]),
        "event_time_shift": event_time_error,
        "max_crest": float(crest[crest_index]),
        "min_trough": float(np.min(trough)),
        "crest_amplification_from_initial": float(crest[crest_index] / crest[0]),
        "max_relative_energy_drift": relative_drift(result["energy"]),
        "max_absolute_volume_drift": float(
            np.max(np.abs(result["volume"] - result["volume"][0]))
        ),
    }
    if scenario.event_x is not None:
        event_x_index = int(np.argmin(np.abs(result["x"] - scenario.event_x)))
        local_elevation = result["eta"][:, event_x_index]
        local_index = int(np.argmax(local_elevation))
        prescribed_index = int(
            np.argmin(np.abs(result["time"] - float(scenario.event_time)))
        )
        metrics.update(
            {
                "event_x": float(result["x"][event_x_index]),
                "crest_at_prescribed_event": float(local_elevation[prescribed_index]),
                "local_max_crest": float(local_elevation[local_index]),
                "observed_local_focus_time": float(result["time"][local_index]),
                "local_focus_time_shift": float(
                    result["time"][local_index] - float(scenario.event_time)
                ),
            }
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=("standing_wave", "dispersive_focusing", "head_on_solitary_pair"),
        default="dispersive_focusing",
    )
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--length", type=float, default=2.0 * math.pi)
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--final-time", type=float)
    parser.add_argument("--snapshots", type=int, default=301)
    parser.add_argument(
        "--output-prefix", type=Path, default=Path("research/results/scenario")
    )
    args = parser.parse_args()

    solver = HOSSolver(
        HOSConfig(
            n=args.n,
            order=args.order,
            depth=args.depth,
            length=args.length,
        )
    )
    scenario = make_scenario(args.scenario, solver, args.amplitude)
    final_time = args.final_time or scenario.final_time
    result = simulate(
        solver,
        scenario.eta,
        scenario.psi,
        args.dt,
        final_time,
        snapshots=args.snapshots,
    )
    metrics = {
        "scenario": scenario.name,
        "n": args.n,
        "order": args.order,
        "dt": float(result["dt"]),
        **scenario_metrics(scenario, result),
    }

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prefix.with_suffix(".npz"), **result)
    write_svg(result, prefix.with_suffix(".svg"), scenario.description)
    prefix.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
