"""Assess physical H and same-solver transfer errors for guarded two-fluid runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tarfile
from pathlib import Path

import numpy as np

from linear_twofluid_wave_reference import Wave


PATHS = {"phase_exact_input": "guarded_full/A_phase",
         "matched": "flat_native_L9_guarded_transfer/runs/full/A_matched",
         "practical": "flat_native_L9_guarded_transfer/runs/full/A_practical"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wave = Wave()
    nx, ny = 512, 32
    h = wave.length / nx
    x = (np.arange(nx) + 0.5) * h
    envelope = np.exp(-1j * wave.k * x)
    fixed_denominator = wave.surface_amplitude * np.sinc(wave.k*h/(2*np.pi)) / math.sqrt(2)
    with tarfile.open(args.archive, "r:gz") as archive:
        def read(path: str) -> bytes:
            member = archive.extractfile(path)
            if member is None:
                raise FileNotFoundError(path)
            return member.read()

        def field(branch: str, kind: str, index: int) -> np.ndarray:
            data = np.frombuffer(read(f"{PATHS[branch]}/segments/segment_0000/{kind}_{index:04d}.raw"),
                                 dtype="<f8")
            if data.size != nx*ny or not np.isfinite(data).all():
                raise ValueError((branch, kind, index))
            return data.reshape(ny, nx)

        branch_meta = {}
        times = {}
        for name, root in PATHS.items():
            status = json.loads(read(root + "/status.json"))
            audits = [json.loads(line) for line in read(root + "/semantic_audit.jsonl").decode().splitlines()]
            if status["state"] != "complete" or not audits or not all(row["valid"] for row in audits):
                raise ValueError(f"invalid {name}")
            tmap = {}
            for row in audits:
                match = re.search(r"dump_(\d+)", row["source"])
                if match:
                    tmap[int(match.group(1))] = row["simulation_time"]
            times[name] = tmap
            branch_meta[name] = {"audits": len(audits),
                                 "initial_flux_divergence_linf": next(row["volume_flux_divergence_linf"] for row in audits if row["simulation_time"] == 0),
                                 "maximum_post_start_flux_divergence_linf": max(row["volume_flux_divergence_linf"] for row in audits if row["simulation_time"] > 0),
                                 "maximum_runtime_mass_relative_error": max(abs(row["runtime_mass_relative_error"]) for row in audits),
                                 "maximum_speed": max(row["maximum_speed_all_cells"] for row in audits)}
        if any(set(tmap) != set(times["phase_exact_input"]) for tmap in times.values()):
            raise ValueError("dump indices differ")
        series = []
        for index, t in sorted(times["phase_exact_input"].items()):
            if any(abs(times[branch][index]-t) > 1e-10 for branch in PATHS):
                raise ValueError("time mismatch")
            analytic = wave.cells(nx, t)["column_depth"]
            q_ref = field("phase_exact_input", "vf", index) * field("phase_exact_input", "ebvf", index)
            vx_ref, vy_ref = field("phase_exact_input", "vx", index), field("phase_exact_input", "vy", index)
            denom_velocity = float(np.sum(q_ref*(vx_ref**2+vy_ref**2)))
            if denom_velocity <= 0:
                raise ValueError("zero velocity reference norm")
            results = {}
            for name in PATHS:
                vf, eb = field(name, "vf", index), field(name, "ebvf", index)
                depth = h * np.sum(vf*eb, axis=0)
                exact_depth = h*np.sum(q_ref, axis=0)
                ephys = depth-analytic
                etransfer = depth-exact_depth
                mode = 2*np.mean((depth-depth.mean())*envelope)
                target_mode = 2*np.mean((analytic-analytic.mean())*envelope)
                emode = 2*np.mean(ephys*envelope)
                reconstructed_error = ephys.mean() + np.real(emode*np.conj(envelope))
                vx, vy = field(name, "vx", index), field(name, "vy", index)
                results[name] = {
                    "physical_H_error_rms": float(np.sqrt(np.mean(ephys**2))),
                    "physical_H_error_normalized_fixed_amplitude": float(np.sqrt(np.mean(ephys**2))/fixed_denominator),
                    "H_difference_from_exact_input_branch_normalized": float(np.sqrt(np.mean(etransfer**2))/fixed_denominator),
                    "first_mode_complex": [float(mode.real), float(mode.imag)],
                    "analytic_first_mode_complex": [float(target_mode.real), float(target_mode.imag)],
                    "first_mode_physical_error_normalized_amplitude": float(abs(emode)/(math.sqrt(2)*fixed_denominator)),
                    "orthogonal_physical_error_normalized_fixed_amplitude": float(np.sqrt(np.mean((ephys-reconstructed_error)**2))/fixed_denominator),
                    "global_liquid_velocity_difference_from_exact_input_branch_relative_l2": float(math.sqrt(np.sum(q_ref*((vx-vx_ref)**2+(vy-vy_ref)**2))/denom_velocity)),
                }
            series.append({"dump_index": index, "time": t, "results": results})
    endpoint = series[-1]["results"]
    report = {"schema": "guarded-twofluid-transfer-physical-height-v1",
              "archive_sha256": sha(args.archive), "analysis_code_sha256": sha(Path(__file__)),
              "analytic_reference_code_sha256": sha(Path(__file__).with_name("linear_twofluid_wave_reference.py")),
              "reference_scope": "linear two-fluid inviscid wave; amplitude-halving phase input and zero-floor controls are separate",
              "fixed_H_normalization": fixed_denominator,
              "runtime_flux_note": "same stored face arrays; native flux_init zero reconstructs actual t0 flux from each cell velocity independently",
              "branch_meta": branch_meta,
              "series": series,
              "endpoint_practical_over_matched_physical_H_error_ratio": endpoint["practical"]["physical_H_error_normalized_fixed_amplitude"] / endpoint["matched"]["physical_H_error_normalized_fixed_amplitude"],
              "endpoint_practical_over_matched_transfer_H_error_ratio": endpoint["practical"]["H_difference_from_exact_input_branch_normalized"] / endpoint["matched"]["H_difference_from_exact_input_branch_normalized"],
              "endpoint_practical_over_matched_velocity_difference_ratio": endpoint["practical"]["global_liquid_velocity_difference_from_exact_input_branch_relative_l2"] / endpoint["matched"]["global_liquid_velocity_difference_from_exact_input_branch_relative_l2"],
              }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"time": series[-1]["time"], "endpoint": endpoint,
                      "physical_H_gain_practical_over_matched": report["endpoint_practical_over_matched_physical_H_error_ratio"]}, indent=2))


if __name__ == "__main__":
    main()
