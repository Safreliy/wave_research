"""Boundary-only reference solver for two-dimensional irrotational water waves.

This is a compact high-order spectral (HOS) implementation of the standard
Zakharov/Craig--Sulem surface formulation.  It is intentionally a *reference
baseline*, not a claim of novelty: the legacy vortex/double-layer idea must be
validated against a solver with known dispersion, conserved volume and a
Hamiltonian energy diagnostic before new variants are proposed.

Only NumPy is required.  The fluid is periodic in x, has mean depth ``depth``,
flat impermeable bottom, no surface tension, and unit density.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class HOSConfig:
    length: float = 2.0 * math.pi
    depth: float = 1.0
    gravity: float = 9.80665
    n: int = 128
    order: int = 3
    dealias_fraction: float = 2.0 / 3.0


class HOSSolver:
    """Truncated HOS Dirichlet-to-Neumann map with RK4 time stepping."""

    def __init__(self, config: HOSConfig):
        if config.n < 8 or config.n % 2:
            raise ValueError("n must be an even integer >= 8")
        if config.order < 1:
            raise ValueError("order must be >= 1")
        if config.depth <= 0 or config.length <= 0:
            raise ValueError("depth and length must be positive")
        self.config = config
        self.dx = config.length / config.n
        self.x = np.arange(config.n, dtype=float) * self.dx
        self.k = 2.0 * math.pi * np.fft.fftfreq(config.n, d=self.dx)
        self.abs_k = np.abs(self.k)
        mode = np.abs(np.fft.fftfreq(config.n) * config.n)
        self.keep_modes = mode <= (config.n / 2.0) * config.dealias_fraction

    def filter(self, values: Array) -> Array:
        coeffs = np.fft.fft(values)
        coeffs[~self.keep_modes] = 0.0
        return np.fft.ifft(coeffs).real

    def derivative_x(self, values: Array) -> Array:
        return np.fft.ifft(1j * self.k * np.fft.fft(values)).real

    def vertical_derivative_at_rest(self, trace: Array, order: int) -> Array:
        """Return d_z**order of the harmonic extension at z=0.

        The extension satisfies a Neumann condition at z=-depth.  Odd vertical
        derivatives therefore carry tanh(|k| h), while even derivatives do not.
        """
        if order < 0:
            raise ValueError("derivative order must be non-negative")
        if order == 0:
            return np.asarray(trace, dtype=float).copy()
        multiplier = self.abs_k**order
        if order % 2:
            multiplier = multiplier * np.tanh(self.abs_k * self.config.depth)
        multiplier[0] = 0.0
        return np.fft.ifft(multiplier * np.fft.fft(trace)).real

    def vertical_velocity(self, eta: Array, psi: Array) -> Array:
        """HOS approximation of phi_z evaluated at z=eta(x)."""
        eta = self.filter(np.asarray(eta, dtype=float))
        psi = self.filter(np.asarray(psi, dtype=float))
        m_max = self.config.order
        traces: list[Array | None] = [None] * (m_max + 1)
        traces[1] = psi

        for m in range(2, m_max + 1):
            correction = np.zeros_like(eta)
            for power in range(1, m):
                base = traces[m - power]
                assert base is not None
                correction += (
                    eta**power
                    / math.factorial(power)
                    * self.vertical_derivative_at_rest(base, power)
                )
            traces[m] = -self.filter(correction)

        vertical = np.zeros_like(eta)
        for m in range(1, m_max + 1):
            base = traces[m]
            assert base is not None
            for power in range(0, m_max - m + 1):
                vertical += (
                    eta**power
                    / math.factorial(power)
                    * self.vertical_derivative_at_rest(base, power + 1)
                )
        return self.filter(vertical)

    def dirichlet_to_neumann(self, eta: Array, psi: Array) -> Array:
        eta_x = self.derivative_x(eta)
        psi_x = self.derivative_x(psi)
        w = self.vertical_velocity(eta, psi)
        return self.filter((1.0 + eta_x**2) * w - eta_x * psi_x)

    def rhs(self, eta: Array, psi: Array) -> tuple[Array, Array]:
        eta_x = self.derivative_x(eta)
        psi_x = self.derivative_x(psi)
        normal_velocity = self.dirichlet_to_neumann(eta, psi)
        # The exact DNO annihilates the zero Fourier mode.  Projecting the tiny
        # truncation residue restores the discrete volume invariant.
        normal_velocity -= np.mean(normal_velocity)
        denominator = 1.0 + eta_x**2
        bernoulli = (
            -self.config.gravity * eta
            - 0.5 * psi_x**2
            + 0.5 * (normal_velocity + eta_x * psi_x) ** 2 / denominator
        )
        return self.filter(normal_velocity), self.filter(bernoulli)

    def step_rk4(self, eta: Array, psi: Array, dt: float) -> tuple[Array, Array]:
        if dt <= 0:
            raise ValueError("dt must be positive")
        k1_eta, k1_psi = self.rhs(eta, psi)
        k2_eta, k2_psi = self.rhs(
            eta + 0.5 * dt * k1_eta, psi + 0.5 * dt * k1_psi
        )
        k3_eta, k3_psi = self.rhs(
            eta + 0.5 * dt * k2_eta, psi + 0.5 * dt * k2_psi
        )
        k4_eta, k4_psi = self.rhs(eta + dt * k3_eta, psi + dt * k3_psi)
        next_eta = eta + dt * (k1_eta + 2 * k2_eta + 2 * k3_eta + k4_eta) / 6.0
        next_psi = psi + dt * (k1_psi + 2 * k2_psi + 2 * k3_psi + k4_psi) / 6.0
        next_eta = self.filter(next_eta)
        next_psi = self.filter(next_psi)
        # Fix the physically irrelevant constant-potential gauge only.
        next_psi -= np.mean(next_psi)
        return next_eta, next_psi

    def volume(self, eta: Array) -> float:
        return float(self.dx * np.sum(eta))

    def energy(self, eta: Array, psi: Array) -> float:
        dno = self.dirichlet_to_neumann(eta, psi)
        density = 0.5 * (self.config.gravity * eta**2 + psi * dno)
        return float(self.dx * np.sum(density))

    def max_surface_speed(self, eta: Array, psi: Array) -> float:
        eta_x = self.derivative_x(eta)
        psi_x = self.derivative_x(psi)
        w = self.vertical_velocity(eta, psi)
        u = psi_x - eta_x * w
        return float(np.max(np.sqrt(u**2 + w**2)))


def simulate(
    solver: HOSSolver,
    eta0: Array,
    psi0: Array,
    dt: float,
    final_time: float,
    snapshots: int = 9,
) -> dict[str, Array]:
    steps = int(math.ceil(final_time / dt))
    actual_dt = final_time / steps
    save_indices = set(np.linspace(0, steps, snapshots, dtype=int).tolist())
    eta = np.asarray(eta0, dtype=float).copy()
    psi = np.asarray(psi0, dtype=float).copy()
    eta -= np.mean(eta)
    psi -= np.mean(psi)

    times: list[float] = []
    eta_history: list[Array] = []
    energy_history: list[float] = []
    volume_history: list[float] = []
    speed_history: list[float] = []

    def save(step: int) -> None:
        times.append(step * actual_dt)
        eta_history.append(eta.copy())
        energy_history.append(solver.energy(eta, psi))
        volume_history.append(solver.volume(eta))
        speed_history.append(solver.max_surface_speed(eta, psi))

    save(0)
    for step in range(1, steps + 1):
        eta, psi = solver.step_rk4(eta, psi, actual_dt)
        if step in save_indices:
            save(step)
        if not (np.all(np.isfinite(eta)) and np.all(np.isfinite(psi))):
            raise FloatingPointError(f"non-finite state at step {step}")

    return {
        "x": solver.x.copy(),
        "time": np.asarray(times),
        "eta": np.asarray(eta_history),
        "energy": np.asarray(energy_history),
        "volume": np.asarray(volume_history),
        "max_surface_speed": np.asarray(speed_history),
        "final_psi": psi.copy(),
        "dt": np.asarray(actual_dt),
    }


def relative_drift(values: Array) -> float:
    scale = max(abs(float(values[0])), np.finfo(float).eps)
    return float(np.max(np.abs(values - values[0])) / scale)


def _polyline(points: list[tuple[float, float]], color: str, width: float = 1.5) -> str:
    encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{encoded}" fill="none" stroke="{escape(color)}" '
        f'stroke-width="{width}" vector-effect="non-scaling-stroke"/>'
    )


def write_svg(result: dict[str, Array], path: Path, title: str) -> None:
    """Write a dependency-free two-panel SVG of snapshots and energy drift."""
    width, height = 1000, 650
    left, right = 80.0, 965.0
    top_1, bottom_1 = 70.0, 365.0
    top_2, bottom_2 = 440.0, 600.0
    x = result["x"]
    eta = result["eta"]
    times = result["time"]
    y_extent = max(float(np.max(np.abs(eta))) * 1.15, 1.0e-12)

    def map_x(value: float) -> float:
        return left + (value - x[0]) / (x[-1] + (x[1] - x[0]) - x[0]) * (right - left)

    def map_eta(value: float) -> float:
        return (top_1 + bottom_1) / 2.0 - value / y_extent * (bottom_1 - top_1) / 2.0

    palette = ["#17324d", "#1f5a7a", "#258ea6", "#52b788", "#f4a261", "#e76f51", "#9b5de5", "#5f0f40", "#111111"]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        f'<text x="{left}" y="32" font-family="Segoe UI,Arial" font-size="22" fill="#17202a">{escape(title)}</text>',
        f'<line x1="{left}" y1="{map_eta(0):.2f}" x2="{right}" y2="{map_eta(0):.2f}" stroke="#aab7c4" stroke-width="1"/>',
        f'<text x="{left}" y="55" font-family="Segoe UI,Arial" font-size="14" fill="#52616b">Free-surface snapshots (vertical scale shared)</text>',
    ]
    display_indices = np.linspace(
        0, len(eta) - 1, min(9, len(eta)), dtype=int
    )
    for display_i, history_i in enumerate(display_indices):
        profile = eta[history_i]
        points = [(map_x(float(x_j)), map_eta(float(y_j))) for x_j, y_j in zip(x, profile)]
        color = palette[display_i % len(palette)]
        elements.append(_polyline(points, color))
        legend_x = left + (display_i % 5) * 170
        legend_y = bottom_1 + 24 + (display_i // 5) * 20
        elements.append(f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 24}" y2="{legend_y - 4}" stroke="{color}" stroke-width="2"/>')
        elements.append(f'<text x="{legend_x + 30}" y="{legend_y}" font-family="Segoe UI,Arial" font-size="12" fill="#34495e">t={times[history_i]:.3f}</text>')

    energy = result["energy"]
    rel_energy = (energy - energy[0]) / max(abs(float(energy[0])), np.finfo(float).eps)
    e_extent = max(float(np.max(np.abs(rel_energy))) * 1.15, 1.0e-15)

    def map_t(value: float) -> float:
        return left + value / max(float(times[-1]), np.finfo(float).eps) * (right - left)

    def map_e(value: float) -> float:
        return (top_2 + bottom_2) / 2.0 - value / e_extent * (bottom_2 - top_2) / 2.0

    elements.extend(
        [
            f'<text x="{left}" y="420" font-family="Segoe UI,Arial" font-size="14" fill="#52616b">Relative Hamiltonian drift</text>',
            f'<rect x="{left}" y="{top_2}" width="{right-left}" height="{bottom_2-top_2}" fill="white" stroke="#d5dde5"/>',
            f'<line x1="{left}" y1="{map_e(0):.2f}" x2="{right}" y2="{map_e(0):.2f}" stroke="#aab7c4" stroke-width="1"/>',
            _polyline([(map_t(float(t)), map_e(float(e))) for t, e in zip(times, rel_energy)], "#c0392b", 1.8),
            f'<text x="{left}" y="630" font-family="Segoe UI,Arial" font-size="12" fill="#52616b">max |ΔE/E₀| = {np.max(np.abs(rel_energy)):.3e}; max |ΔV| = {np.max(np.abs(result["volume"] - result["volume"][0])):.3e}</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def initial_condition(name: str, solver: HOSSolver, amplitude: float) -> tuple[Array, Array, float]:
    x = solver.x
    if name == "linear_mode":
        eta = amplitude * np.cos(2.0 * math.pi * x / solver.config.length)
        wavenumber = 2.0 * math.pi / solver.config.length
        omega = math.sqrt(
            solver.config.gravity
            * wavenumber
            * math.tanh(wavenumber * solver.config.depth)
        )
        return eta, np.zeros_like(eta), 2.0 * math.pi / omega
    if name == "gaussian":
        distance = np.minimum(x, solver.config.length - x)
        eta = amplitude * np.exp(-(distance / (0.12 * solver.config.length)) ** 2)
        eta -= np.mean(eta)
        return eta, np.zeros_like(eta), 2.5
    raise ValueError(f"unknown scenario: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("linear_mode", "gaussian"), default="linear_mode")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--length", type=float, default=2.0 * math.pi)
    parser.add_argument("--amplitude", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--final-time", type=float)
    parser.add_argument("--output-prefix", type=Path, default=Path("research/results/hos"))
    args = parser.parse_args()

    config = HOSConfig(
        length=args.length, depth=args.depth, n=args.n, order=args.order
    )
    solver = HOSSolver(config)
    eta0, psi0, suggested_time = initial_condition(args.scenario, solver, args.amplitude)
    final_time = args.final_time if args.final_time is not None else suggested_time
    result = simulate(solver, eta0, psi0, args.dt, final_time)

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prefix.with_suffix(".npz"), **result)
    write_svg(result, prefix.with_suffix(".svg"), f"HOS reference: {args.scenario}")

    metrics = {
        "scenario": args.scenario,
        "config": {
            "length": config.length,
            "depth": config.depth,
            "gravity": config.gravity,
            "n": config.n,
            "order": config.order,
        },
        "dt": float(result["dt"]),
        "final_time": float(result["time"][-1]),
        "initial_energy": float(result["energy"][0]),
        "max_relative_energy_drift": relative_drift(result["energy"]),
        "max_absolute_volume_drift": float(
            np.max(np.abs(result["volume"] - result["volume"][0]))
        ),
        "max_surface_speed": float(np.max(result["max_surface_speed"])),
        "finite": bool(np.all(np.isfinite(result["eta"]))),
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
