"""Diagnose phase-to-one-velocity injection; this is not a native accuracy test.

The mass mean minimizes density-weighted squared discrepancy to the two phase
means in a cell. This algebraic fact alone does not identify the receiver's
discrete velocity contract or establish improved wave dynamics.
"""
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from linear_twofluid_wave_reference import Wave


def audit(nx: int, q_top: float) -> dict:
    h = 64.0 / nx
    wave = replace(Wave(), surface=1.5 + q_top * h)
    c = wave.cells(nx, 0)
    ml = wave.rho_liquid * c["q"] * h**2
    mg = wave.rho_gas * (c["cs"] - c["q"]) * h**2
    mixed = (c["q"] > 1e-12) & (c["cs"] - c["q"] > 1e-12)
    baseline_energy = float(np.sum(ml * np.sum(c["liquid_mean"]**2, axis=0)
                                   + mg * np.sum(c["gas_mean"]**2, axis=0)))
    rows = {}
    for method in ("phase_injection", "mass_mean", "volume_mean"):
        v = c[method]
        delta = v - c["mass_mean"]
        excess = float(np.sum(c["cell_mass"] * np.sum(delta**2, axis=0)))
        loss = float(np.sum(ml * np.sum((v-c["liquid_mean"])**2, axis=0)
                            + mg * np.sum((v-c["gas_mean"])**2, axis=0)))
        optimum_loss = float(np.sum(ml * np.sum((c["mass_mean"]-c["liquid_mean"])**2, axis=0)
                                    + mg * np.sum((c["mass_mean"]-c["gas_mean"])**2, axis=0)))
        identity_residual = loss - optimum_loss - excess
        local_momentum_error = c["cell_mass"] * v - c["mixture_momentum"]
        rows[method] = {
            "phase_mean_projection_loss_over_phase_mean_energy": loss / baseline_energy,
            "excess_projection_loss_over_phase_mean_energy": excess / baseline_energy,
            "orthogonal_projection_identity_residual": identity_residual,
            "max_cell_mixture_momentum_error": float(np.max(np.abs(local_momentum_error))),
            "global_mixture_momentum_error": local_momentum_error.sum(axis=(1, 2)).tolist(),
        }
        if abs(identity_residual) > baseline_energy * 1e-12:
            raise AssertionError(rows)
    return {"nx": nx, "liquid_fraction_in_top_cell": q_top,
            "mixed_cells": int(mixed.sum()),
            "gas_mass_fraction_in_mixed_cell": float((mg/(ml+mg+1e-300))[mixed].mean()),
            "methods": rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = {"scope": "offline phase-mean projection diagnostic; native velocity contract and physical accuracy unproven",
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "reference_sha256": hashlib.sha256(Path(__file__).with_name("linear_twofluid_wave_reference.py").read_bytes()).hexdigest(),
              "rows": [audit(n, q) for n in (512, 1024) for q in (0.0005, 0.001, 0.01, 0.1)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    for row in report["rows"]:
        print(row["nx"], row["liquid_fraction_in_top_cell"],
              row["gas_mass_fraction_in_mixed_cell"],
              row["methods"]["phase_injection"]["excess_projection_loss_over_phase_mean_energy"])


if __name__ == "__main__":
    main()
