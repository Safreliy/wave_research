"""Independent two-grid linear-wave audit from full guarded native raw fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from audit_auditor_dynamic_outputs import Run, compare, sha
from linear_twofluid_wave_reference import Wave


def initial_field(folder: Path, kind: str, shape: tuple[int, int]) -> np.ndarray:
    data = np.fromfile(folder / f"{kind}.raw", dtype="<f8")
    if data.size != math.prod(shape) or not np.isfinite(data).all():
        raise ValueError((folder, kind))
    return data.reshape(shape)


def linear_depth_error(run: Run, index: int, time: float, wave: Wave, nx: int) -> dict:
    h = wave.length / nx
    x = (np.arange(nx)+0.5)*h
    sinc = float(np.sinc(wave.k*h/(2*np.pi)))
    amplitude = wave.surface_amplitude*sinc
    expected = wave.depth + amplitude*np.sin(wave.omega*time)*np.cos(wave.k*x)
    q = run.field("vf", index)*run.field("ebvf", index)
    depth = h*q.sum(axis=0)
    error = depth-expected
    normalizer = abs(amplitude)/math.sqrt(2)
    return {"normalised_depth_rms": float(np.sqrt(np.mean(error**2))/normalizer),
            "absolute_depth_rms": float(np.sqrt(np.mean(error**2))),
            "max_absolute_depth_error": float(np.max(abs(error))),
            "total_liquid_volume": float(q.sum()*h*h)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root / "physical/campaign.json").read_text())
    wave = Wave()
    result = {"schema": "auditor-single-runtime-physical-raw-audit-v1",
              "campaign_sha256": sha(root / "physical/campaign.json"),
              "input_archive_sha256": sha(root / "physical_inputs.tar.gz"),
              "runtime_sha256": plan["expected_runtime_sha256"],
              "source_build_match_sha256": sha(root / "runtime_provenance/source_build_match_report.json"),
              "reference": "linear inviscid two-fluid capillary-gravity wave with pressure-release top, fixed cell-averaged amplitude normalizer; not exact nonlinear Navier-Stokes",
              "normalization": {
                  "linear_wave.normalised_depth_rms":
                  "RMS(column depth minus analytic cell-averaged linear-wave depth) / (absolute cell-averaged wave amplitude / sqrt(2)); fixed across times and branches on each grid",
                  "numerical_exact_input_discrepancy.column_depth_wave_relative_l2":
                  "L2(candidate column depth minus exact-input numerical column depth) / L2(exact-input numerical column depth minus its spatial mean); denominator varies with time and grid",
                  "numerical_exact_input_discrepancy.global_liquid_velocity_relative_l2":
                  "sqrt(sum(q_exact * squared velocity difference) / sum(q_exact * squared exact-input numerical velocity)), with q_exact = vf * ebvf"
              },
              "receiver_assignment": plan["receiver_assignment"],
              "grids": {}}
    for tag, details in plan["grids"].items():
        nx = details["nx"]
        shape = (nx//16, nx)
        h = wave.length/nx
        runs = {label: Run(root / f"physical_{tag}_{label}_outputs.tar.gz", tag, label, nx)
                for label in ("exact", "matched", "bilinear")}
        expected_runtime = plan["expected_runtime_sha256"]
        for label, run in runs.items():
            fingerprint = run.status["runtime_fingerprint"]
            if (fingerprint["solver_sha256"] != expected_runtime[fingerprint["solver_path"]]
                    or fingerprint["solver_library_sha256"] != expected_runtime[fingerprint["solver_library_path"]]):
                raise ValueError(f"{tag}/{label}: executed runtime fingerprint mismatch")
        indices = sorted(runs["exact"].times)
        if any(sorted(run.times) != indices for run in runs.values()):
            raise ValueError(f"{tag}: dump indices differ")
        q0 = initial_field(root / f"physical/{tag}/exact", "q", shape)
        exact_input = np.stack([initial_field(root / f"physical/{tag}/exact", component, shape)
                                for component in ("vx", "vy")])
        initial_norms = {}
        for label in ("matched", "bilinear"):
            values = np.stack([initial_field(root / f"physical/{tag}/{label}", component, shape)
                               for component in ("vx", "vy")])
            initial_norms[label] = float(np.sqrt(
                np.sum(q0*np.sum((values-exact_input)**2, axis=0)) /
                np.sum(q0*np.sum(exact_input**2, axis=0))))
        series = []
        for index in indices:
            times = [runs[label].times[index] for label in ("exact", "matched", "bilinear")]
            if max(times)-min(times) > 1e-10:
                raise ValueError(f"{tag}: time mismatch {index}: {times}")
            row = {"dump_index": index, "time": times[0],
                   "linear_wave": {label: linear_depth_error(run, index, times[0], wave, nx)
                                   for label, run in runs.items()}}
            if index > 0:
                row["numerical_exact_input_discrepancy"] = {
                    label: compare(runs[label], runs["exact"], index, h)
                    for label in ("matched", "bilinear")}
            series.append(row)
        target = plan["target_time"]
        if abs(series[-1]["time"]-target) > 1e-10:
            raise ValueError(f"{tag}: endpoint mismatch")
        if any(abs(run.status["checkpoint"]["time"]-target) > 1e-10
               for run in runs.values()):
            raise ValueError(f"{tag}: incomplete checkpoint")
        end = series[-1]["linear_wave"]
        result["grids"][tag] = {
            "nx": nx, "ny": shape[0], "config_sha256": details["config_sha256"],
            "supervisor_sha256": details["supervisor_sha256"],
            "input_hashes": {label: details["branches"][label]["input_sha256"]
                             for label in runs},
            "initial_liquid_weighted_receiver_velocity_relative_l2": initial_norms,
            "runs": {label: run.report() for label, run in runs.items()},
            "series": series,
            "endpoint_matched_reduction_against_bilinear_percent": 100*(
                1-end["matched"]["normalised_depth_rms"] /
                end["bilinear"]["normalised_depth_rms"]),
            "endpoint_exact_input_physical_error": end["exact"]["normalised_depth_rms"]}
        for run in runs.values():
            run.archive.close()
    output = root / "physical_two_grid_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({tag: {"endpoint": data["series"][-1]["linear_wave"],
                           "matched_reduction_percent": data["endpoint_matched_reduction_against_bilinear_percent"]}
                      for tag, data in result["grids"].items()}, indent=2))


if __name__ == "__main__":
    main()
