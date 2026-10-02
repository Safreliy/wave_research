"""Independent linear two-fluid wave reference, with a pressure-release top.

This is an inviscid small-amplitude reference, NOT an exact solution of the
nonlinear Navier--Stokes problem. The liquid bottom is impermeable, the upper
gas boundary has zero pressure perturbation, and x is periodic. Nonlinearity
must be checked by an amplitude-halving control in any native comparison.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Wave:
    length: float = 64.0
    bottom: float = 0.53000001
    surface: float = 1.5000625
    top: float = 4.0
    mode: int = 16
    potential_amplitude: float = 2e-5
    rho_liquid: float = 1.0
    rho_gas: float = 0.001176470588235294
    gravity: float = 1.0
    sigma: float = 0.001

    @property
    def k(self) -> float:
        return 2 * np.pi * self.mode / self.length

    @property
    def depth(self) -> float:
        return self.surface - self.bottom

    @property
    def gas_depth(self) -> float:
        return self.top - self.surface

    @property
    def omega(self) -> float:
        k = self.k
        inertia = self.rho_liquid / np.tanh(k * self.depth) + self.rho_gas * np.tanh(k * self.gas_depth)
        return float(np.sqrt(((self.rho_liquid - self.rho_gas) * self.gravity * k + self.sigma * k**3) / inertia))

    @property
    def period(self) -> float:
        return float(2 * np.pi / self.omega)

    @property
    def surface_amplitude(self) -> float:
        return float(self.potential_amplitude * self.k * np.tanh(self.k * self.depth) / self.omega)

    def eta(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.surface_amplitude * np.sin(self.omega * t) * np.cos(self.k * x)

    def velocity(self, x: np.ndarray, y: np.ndarray, t: float, phase: str) -> np.ndarray:
        k = self.k
        scale = self.potential_amplitude * k * np.cos(self.omega * t)
        if phase == "liquid":
            u = -scale * np.sin(k * x) * np.cosh(k * (y - self.bottom)) / np.cosh(k * self.depth)
            v = scale * np.cos(k * x) * np.sinh(k * (y - self.bottom)) / np.cosh(k * self.depth)
        elif phase == "gas":
            scale *= np.tanh(k * self.depth) / np.cosh(k * self.gas_depth)
            u = scale * np.sin(k * x) * np.sinh(k * (self.top - y))
            v = scale * np.cos(k * x) * np.cosh(k * (self.top - y))
        else:
            raise ValueError(phase)
        return np.stack(np.broadcast_arrays(u, v))

    def cells(self, nx: int, t: float, order: int = 16) -> dict[str, np.ndarray]:
        """Integrate phase volumes/moments on the linearly displaced surface.

        Closed vertical antiderivatives and Gaussian horizontal quadrature
        are used. Orders must be compared if the surface crosses grid lines.
        """
        h = self.length / nx
        ny = round(self.top / h)
        if abs(ny * h - self.top) > 1e-12:
            raise ValueError("Expected square cells and an integer vertical grid")
        nodes, weights = np.polynomial.legendre.leggauss(order)
        x = (np.arange(nx)[None, :, None] + (nodes[None, None, :] + 1) / 2) * h
        yl = np.arange(ny)[:, None, None] * h
        yh = yl + h
        surface = self.surface + self.eta(x, t)
        lo_l = np.broadcast_to(np.maximum(yl, self.bottom), (ny, nx, order))
        hi_l = np.maximum(lo_l, np.minimum(yh, surface))
        lo_g = np.minimum(self.top, np.maximum(yl, surface))
        hi_g = np.maximum(lo_g, np.broadcast_to(np.minimum(yh, self.top), (ny, nx, order)))
        k = self.k
        scale = self.potential_amplitude * np.cos(self.omega * t)
        ul = -scale * np.sin(k * x) * (np.sinh(k * (hi_l - self.bottom)) - np.sinh(k * (lo_l - self.bottom))) / np.cosh(k * self.depth)
        vl = scale * np.cos(k * x) * (np.cosh(k * (hi_l - self.bottom)) - np.cosh(k * (lo_l - self.bottom))) / np.cosh(k * self.depth)
        gas_scale = scale * np.tanh(k * self.depth) / np.cosh(k * self.gas_depth)
        ug = gas_scale * np.sin(k * x) * (np.cosh(k * (self.top - lo_g)) - np.cosh(k * (self.top - hi_g)))
        vg = gas_scale * np.cos(k * x) * (np.sinh(k * (self.top - lo_g)) - np.sinh(k * (self.top - hi_g)))

        def integrate_x(values: np.ndarray) -> np.ndarray:
            return (values @ weights) * h / 2

        area_l, area_g = integrate_x(hi_l - lo_l), integrate_x(hi_g - lo_g)
        moment_l, moment_g = integrate_x(np.stack((ul, vl))), integrate_x(np.stack((ug, vg)))
        mean_l = np.divide(moment_l, area_l, out=np.zeros_like(moment_l), where=area_l > 0)
        mean_g = np.divide(moment_g, area_g, out=np.zeros_like(moment_g), where=area_g > 0)
        total_area = area_l + area_g
        mass = self.rho_liquid * area_l + self.rho_gas * area_g
        mixture_momentum = self.rho_liquid * moment_l + self.rho_gas * moment_g
        mass_mean = np.divide(mixture_momentum, mass, out=np.zeros_like(moment_l), where=mass > 0)
        volume_mean = np.divide(moment_l + moment_g, total_area, out=np.zeros_like(moment_l), where=total_area > 0)
        phase_injection = np.where((area_l > 0)[None], mean_l, mean_g)
        return {"q": area_l / h**2, "cs": total_area / h**2,
                "liquid_mean": mean_l, "gas_mean": mean_g,
                "mass_mean": mass_mean, "volume_mean": volume_mean,
                "phase_injection": phase_injection,
                "mixture_momentum": mixture_momentum, "cell_mass": mass,
                "column_depth": area_l.sum(axis=0) / h}


def validate(wave: Wave) -> dict:
    """Boundary-equation and independent point-quadrature checks."""
    x = np.linspace(0, wave.length, 103, endpoint=False)
    t = 0.137 * wave.period
    k, a, w = wave.k, wave.potential_amplitude, wave.omega
    eta_t = wave.surface_amplitude * w * np.cos(w * t) * np.cos(k * x)
    liquid_v = wave.velocity(x, np.full_like(x, wave.surface), t, "liquid")[1]
    gas_v = wave.velocity(x, np.full_like(x, wave.surface), t, "gas")[1]
    dphi_l_dt = -a * w * np.sin(w * t) * np.cos(k * x)
    dphi_g_dt = a * np.tanh(k * wave.depth) * np.tanh(k * wave.gas_depth) * w * np.sin(w * t) * np.cos(k * x)
    pressure_jump = -wave.rho_liquid * dphi_l_dt + wave.rho_gas * dphi_g_dt - (wave.rho_liquid - wave.rho_gas) * wave.gravity * wave.eta(x, t)
    capillary_jump = wave.sigma * k**2 * wave.eta(x, t)
    metrics = {"kinematic_liquid_max_abs": float(np.max(np.abs(liquid_v - eta_t))),
               "kinematic_gas_max_abs": float(np.max(np.abs(gas_v - eta_t))),
               "dynamic_jump_max_abs": float(np.max(np.abs(pressure_jump - capillary_jump))),
               "bottom_normal_velocity_max_abs": float(np.max(np.abs(wave.velocity(x, np.full_like(x, wave.bottom), t, "liquid")[1])))}
    # Direct 12x12 point quadrature does not use the vertical antiderivatives.
    cells = wave.cells(512, 0)
    nodes, weights = np.polynomial.legendre.leggauss(12)
    h = wave.length / 512
    largest = 0.0
    for phase, row in (("liquid", 4), ("liquid", 8), ("liquid", 12), ("gas", 12), ("gas", 22), ("gas", 31)):
        lo = max(row * h, wave.bottom if phase == "liquid" else wave.surface)
        hi = min((row + 1) * h, wave.surface if phase == "liquid" else wave.top)
        for col in (0, 5, 19):
            xs = (col + (nodes + 1) / 2) * h
            ys = lo + (nodes + 1) * (hi - lo) / 2
            point = wave.velocity(xs[:, None], ys[None, :], 0, phase)
            direct = np.einsum("ixy,x,y->i", point, weights, weights) / 4
            expected = cells[phase + "_mean"][:, row, col]
            largest = max(largest, float(np.max(np.abs(direct - expected))))
    metrics["independent_point_quadrature_max_abs"] = largest
    metrics["mixture_momentum_closure_max_abs"] = float(np.max(np.abs(cells["mass_mean"] * cells["cell_mass"] - cells["mixture_momentum"])))
    metrics["total_fluid_volume_error"] = float(abs(cells["cs"].sum() * h**2 - wave.length * (wave.top - wave.bottom)))
    if max(metrics.values()) > 1e-10:
        raise ValueError(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wave = Wave()
    report = {"schema": "linear-twofluid-wave-reference-v1", "parameters": asdict(wave),
              "period": wave.period, "surface_amplitude": wave.surface_amplitude,
              "ka": wave.k * wave.surface_amplitude,
              "scope": "linear inviscid two-fluid wave with pressure-release top, not exact nonlinear Navier-Stokes",
              "checks": validate(wave),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
