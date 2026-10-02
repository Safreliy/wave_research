"""Explain legacy receiver q-weighted momentum versus true phase momentum."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from linear_twofluid_wave_reference import Wave


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def velocity(folder: Path, shape: tuple[int, int]) -> np.ndarray:
    return np.stack([np.fromfile(folder / f"{kind}.raw", dtype="<f8").reshape(shape)
                     for kind in ("vx", "vy")])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root / "physical/campaign.json").read_text())
    old = Path(__file__).resolve().parents[1] / "results/publication_package"
    wave = Wave()
    output = {"schema": "auditor-physical-initial-momentum-erratum-v1",
              "executed_campaign_sha256": sha(root / "physical/campaign.json"),
              "executed_generator_sha256": sha(root / "physical/prepare_auditor_twofluid_grid_executed.py"),
              "future_generator_sha256": sha(Path(__file__).with_name("prepare_auditor_twofluid_grid.py")),
              "erratum": "executed initial_state.liquid_momentum is sum(q*u_receiver)*h^2, a receiver-velocity loading diagnostic; it is not the liquid-phase momentum sum(q*u_liquid)*h^2",
              "preflight_limit": "the native preflight gates initial receiver velocity and near-zero target, but does not independently verify phase momentum or mixture formula",
              "grids": {}}
    for nx in (512, 1024):
        tag = f"L{nx.bit_length()-1}"
        shape = (nx//16, nx)
        h = wave.length/nx
        exact_cells = wave.cells(nx, 0)
        folder = root / "physical" / tag
        q = np.fromfile(folder / "exact/q.raw", dtype="<f8").reshape(shape)
        cs = np.fromfile(folder / "exact/cs.raw", dtype="<f8").reshape(shape)
        gas = exact_cells["gas_mean"]
        weight_l = wave.rho_liquid*q
        weight_g = wave.rho_gas*(cs-q)
        total_weight = weight_l+weight_g
        rows = {}
        for label in ("exact", "matched", "bilinear"):
            if label == "exact":
                phase = exact_cells["liquid_mean"]
            elif nx == 512:
                name = "A_matched" if label == "matched" else "A_practical"
                phase = velocity(old / "transfer_physical_pilot_20261001" /
                                 "flat_native_L9_guarded_transfer" / name, shape)
            else:
                source = (old / "transfer_matched_native_20261001/flat_native_L10_matched/fitted"
                          if label == "matched" else
                          old / "transfer_dynamic_followup_20261001/flat_native_L10/bilinear_centroid")
                phase = velocity(source, shape)*1e-3
            receiver = velocity(folder / label, shape)
            expected = np.divide(weight_l[None]*phase+weight_g[None]*gas,
                                 total_weight[None], out=np.zeros_like(receiver),
                                 where=total_weight[None] > 0)
            expected[:, q <= 0] = exact_cells["mass_mean"][:, q <= 0]
            formula_defect = float(np.max(abs(receiver-expected)))
            phase_momentum = [float(np.sum(q*phase[d])*h*h) for d in range(2)]
            gas_momentum = [float(np.sum((cs-q)*gas[d])*h*h) for d in range(2)]
            q_receiver = [float(np.sum(q*receiver[d])*h*h) for d in range(2)]
            mixture_momentum = [float(np.sum(total_weight*receiver[d])*h*h)
                                for d in range(2)]
            physical_sum = [wave.rho_liquid*phase_momentum[d] +
                            wave.rho_gas*gas_momentum[d] for d in range(2)]
            mixture_closure = float(np.max(abs(np.array(mixture_momentum)-physical_sum)))
            state = json.loads((folder / label / "initial_state.json").read_text())
            metadata_defect = float(np.max(abs(np.array(q_receiver)-state["liquid_momentum"])))
            if formula_defect > 2e-14 or mixture_closure > 2e-12 or metadata_defect > 2e-14:
                raise ValueError((tag, label, formula_defect, mixture_closure, metadata_defect))
            rows[label] = {"source_input_sha256": plan["grids"][tag]["branches"][label]["input_sha256"],
                           "receiver_q_weighted_momentum": q_receiver,
                           "true_liquid_phase_momentum": phase_momentum,
                           "true_gas_phase_momentum": gas_momentum,
                           "density_weighted_mixture_momentum": mixture_momentum,
                           "density_weighted_phase_sum": physical_sum,
                           "mixture_formula_max_absolute_defect": formula_defect,
                           "mixture_momentum_closure_linf": mixture_closure,
                           "executed_metadata_defect_linf": metadata_defect}
        output["grids"][tag] = rows
    path = root / "physical_initial_metadata_erratum.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({tag: {label: {"formula_defect": v["mixture_formula_max_absolute_defect"],
                                    "phase_momentum": v["true_liquid_phase_momentum"]}
                            for label, v in rows.items()}
                      for tag, rows in output["grids"].items()}, indent=2))


if __name__ == "__main__":
    main()
