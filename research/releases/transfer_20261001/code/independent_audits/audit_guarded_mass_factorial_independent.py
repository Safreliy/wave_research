"""Independent frozen-field audit of the guarded phase/mass factorial runs.

Reads raw native fields and three frozen input archives; it does not import
the campaign's analytic-wave model, transfer builder, or norm implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tarfile
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "flat_native_L9_twofluid" / "protocol.json"
BASE_INPUTS = HERE / "flat_native_L9_twofluid_inputs.tar.gz"
PHASE_INPUTS = HERE / "flat_native_L9_guarded_transfer_inputs.tar.gz"
MASS_INPUTS = HERE / "flat_native_L9_guarded_mass_factorial_inputs.tar.gz"
PREFIXES = {
    "phase_exact_input": "guarded_full/A_phase",
    "mass_exact_input": "guarded_full/A_mass",
    "phase_matched": "flat_native_L9_guarded_transfer/runs/full/A_matched",
    "phase_practical": "flat_native_L9_guarded_transfer/runs/full/A_practical",
    "mass_matched": "flat_native_L9_guarded_mass_factorial/runs/full/A_matched_mass",
    "mass_practical": "flat_native_L9_guarded_mass_factorial/runs/full/A_practical_mass",
    "zero": "guarded_full/zero",
}
TIME_TOL = 1e-12
NORM_TOL = 2e-10
INPUT_TOL = 2e-13


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def member_bytes(tar: tarfile.TarFile, name: str) -> bytes:
    member = tar.getmember(name)
    if not member.isfile():
        raise ValueError(f"not a regular archive member: {name}")
    return tar.extractfile(member).read()


def raw(tar: tarfile.TarFile, name: str, shape: tuple[int, int]) -> np.ndarray:
    data = member_bytes(tar, name)
    if len(data) != 8 * math.prod(shape):
        raise ValueError(f"wrong raw-field length: {name}")
    result = np.frombuffer(data, dtype="<f8").reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError(f"nonfinite raw field: {name}")
    return result


def audit_times(tar: tarfile.TarFile, prefix: str) -> dict[int, float]:
    lines = member_bytes(tar, prefix + "/semantic_audit.jsonl").decode().splitlines()
    times: dict[int, float] = {}
    for line in lines:
        item = json.loads(line)
        if not item.get("valid", False):
            raise ValueError(f"invalid semantic audit: {prefix}")
        match = re.search(r"dump_(\d+)(?::exit)?$", item["source"])
        if match:
            index, time = int(match.group(1)), float(item["simulation_time"])
            if index in times and abs(time - times[index]) > TIME_TOL:
                raise ValueError(f"duplicate time mismatch: {prefix}/{index}")
            times[index] = time
    if set(times) != set(range(5)):
        raise ValueError(f"unexpected dump indices: {prefix}: {sorted(times)}")
    return times


def final_stat_time(tar: tarfile.TarFile, prefix: str) -> float:
    lines = member_bytes(tar, prefix + "/segments/segment_0000/stat.dat").decode().splitlines()
    return float(lines[-1].split()[lines[0].split().index("t")])


def flat_gas_means(wave: dict, nx: int, ny: int) -> np.ndarray:
    """Closed t=0 gas-potential integrals for each receiver rectangle."""
    length = float(wave["length"])
    h = length / nx
    k = 2 * math.pi * int(wave["mode"]) / length
    top, surface, bed = float(wave["top"]), float(wave["surface"]), float(wave["bottom"])
    A = float(wave["potential_amplitude"])
    depth, gas_depth = surface - bed, top - surface
    coefficient = A * math.tanh(k * depth) / math.cosh(k * gas_depth)
    xc = (np.arange(nx) + 0.5) * h
    sinc = math.sin(k * h / 2) / (k * h / 2)
    ys = (np.arange(ny) * h)[:, None]
    lo = np.minimum(top, np.maximum(ys, surface))
    hi = np.maximum(lo, np.minimum(ys + h, top))
    thickness = hi - lo
    horizontal_sin = (sinc * np.sin(k * xc))[None, :]
    horizontal_cos = (sinc * np.cos(k * xc))[None, :]
    u_vertical = np.divide(
        np.cosh(k * (top - lo)) - np.cosh(k * (top - hi)),
        thickness, out=np.zeros_like(thickness), where=thickness > 0,
    )
    v_vertical = np.divide(
        np.sinh(k * (top - lo)) - np.sinh(k * (top - hi)),
        thickness, out=np.zeros_like(thickness), where=thickness > 0,
    )
    return np.stack((coefficient * u_vertical * horizontal_sin,
                     coefficient * v_vertical * horizontal_cos))


def audit_inputs(report: dict, protocol: dict) -> None:
    shape = 32, 512
    rho_l = float(protocol["wave"]["rho_liquid"])
    rho_g = float(protocol["wave"]["rho_gas"])
    gas_mean = flat_gas_means(protocol["wave"], 512, 32)
    shared = ("q.raw", "cs.raw", "vf.raw", "body.dat", "fluxx.raw",
              "fluxxp.raw", "fluxy.raw", "fluxyp.raw", "fluxeb.raw")
    with (tarfile.open(BASE_INPUTS, "r:gz") as base,
          tarfile.open(PHASE_INPUTS, "r:gz") as phase,
          tarfile.open(MASS_INPUTS, "r:gz") as mass):
        exact_prefix = "flat_native_L9_twofluid/A_mass/"
        q = raw(base, exact_prefix + "q.raw", shape)
        cs = raw(base, exact_prefix + "cs.raw", shape)
        gas_only = (cs > 0) & (q == 0)
        liquid_weight = rho_l * q
        gas_weight = rho_g * (cs - q)
        mass_weight = liquid_weight + gas_weight
        for target, source in (("A_matched_mass", "A_matched"),
                               ("A_practical_mass", "A_practical")):
            new_prefix = f"flat_native_L9_guarded_mass_factorial/{target}/"
            source_prefix = f"flat_native_L9_guarded_transfer/{source}/"
            for name in shared:
                b = member_bytes(base, exact_prefix + name)
                if b != member_bytes(mass, new_prefix + name):
                    raise ValueError(f"new mass common input differs: {target}/{name}")
                if b != member_bytes(phase, source_prefix + name):
                    raise ValueError(f"source phase common input differs: {source}/{name}")
            max_difference = 0.0
            gas_byte_equal = True
            for component, component_index in (("vx", 0), ("vy", 1)):
                liquid = raw(phase, source_prefix + component + ".raw", shape)
                actual = raw(mass, new_prefix + component + ".raw", shape)
                reference_gas = raw(base, exact_prefix + component + ".raw", shape)
                expected = np.divide(
                    liquid_weight * liquid + gas_weight * gas_mean[component_index],
                    mass_weight, out=np.zeros(shape), where=mass_weight > 0,
                )
                max_difference = max(max_difference, float(np.max(np.abs(actual - expected))))
                gas_byte_equal &= np.array_equal(actual[gas_only], reference_gas[gas_only])
            if max_difference > INPUT_TOL or not gas_byte_equal:
                raise ValueError(f"candidate is not density-weighted mixture: {target}")
            report["input_checks"][target] = {
                "source_phase_branch": source,
                "common_geometry_and_serialized_faces_byte_identical": True,
                "gas_only_input_velocity_bit_identical_to_mass_exact": gas_byte_equal,
                "max_absolute_difference_from_independently_integrated_density_mixture": max_difference,
                "mixed_phase_cells": int(np.count_nonzero((q > 0) & (cs > q))),
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=HERE / "twofluid_guarded_mass_factorial_outputs.tar.gz")
    parser.add_argument("--analysis", type=Path, default=HERE / "guarded_mass_factorial_height_analysis.json")
    parser.add_argument("--report", type=Path, default=HERE / "guarded_mass_factorial_independent_audit.json")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    primary = json.loads(args.analysis.read_text(encoding="utf-8"))
    wave = protocol["wave"]
    shape = 32, 512
    h = float(wave["length"]) / shape[1]
    depth = float(wave["surface"] - wave["bottom"])
    gas_depth = float(wave["top"] - wave["surface"])
    k = 2 * math.pi * int(wave["mode"]) / float(wave["length"])
    rho_l, rho_g = float(wave["rho_liquid"]), float(wave["rho_gas"])
    omega = math.sqrt(
        ((rho_l-rho_g)*float(wave["gravity"])*k + float(wave["sigma"])*k**3)
        / (rho_l / math.tanh(k*depth) + rho_g * math.tanh(k*gas_depth))
    )
    B = float(wave["potential_amplitude"]) * k * math.tanh(k*depth) / omega
    sinc = math.sin(k*h/2) / (k*h/2)
    fixed_scale = B * sinc / math.sqrt(2)
    xc = (np.arange(shape[1]) + .5) * h
    present = tuple(primary["series"][0]["results"])
    required = {"phase_exact_input", "mass_exact_input", "phase_practical",
                "mass_matched", "mass_practical"}
    if not required.issubset(present):
        raise ValueError(f"missing required factorial branches: {required-set(present)}")
    report = {
        "schema": "guarded-mass-factorial-independent-raw-audit-v1",
        "scope": "Linear-wave physical height and mass-exact same-grid native discrepancies, not exact nonlinear truth",
        "checker_sha256": sha256(Path(__file__)),
        "output_archive_sha256": sha256(args.outputs),
        "primary_analysis_sha256": sha256(args.analysis),
        "protocol_sha256": sha256(PROTOCOL),
        "base_input_archive_sha256": sha256(BASE_INPUTS),
        "phase_input_archive_sha256": sha256(PHASE_INPUTS),
        "mass_input_archive_sha256": sha256(MASS_INPUTS),
        "analytic_height": "depth+B*sin(omega*t)*sinc(k*h/2)*cos(k*xcenter)",
        "native_height": "h*sum_y(VF*EBVF)",
        "fixed_normalizer": fixed_scale,
        "native_velocity_weight": "q_ref=VF_mass_exact*EBVF_mass_exact",
        "branches_in_primary": list(present),
        "input_checks": {}, "series": [],
    }
    if report["output_archive_sha256"] != primary["archive_sha256"]:
        raise ValueError("raw archive hash differs from primary analysis")
    audit_inputs(report, protocol)
    with tarfile.open(args.outputs, "r:gz") as tar:
        times = {}
        statuses = {}
        for name in (*present, "zero"):
            prefix = PREFIXES[name]
            times[name] = audit_times(tar, prefix)
            status = json.loads(member_bytes(tar, prefix + "/status.json"))
            if status["state"] != "complete" or status["return_code"] != 0:
                raise ValueError(f"native branch incomplete: {name}")
            if not status.get("initial_preflight", {}).get("valid", False):
                raise ValueError(f"initial preflight failed: {name}")
            if (abs(status["progress_time"] - protocol["target_time"]) > TIME_TOL
                    or abs(final_stat_time(tar, prefix) - protocol["target_time"]) > TIME_TOL):
                raise ValueError(f"endpoint time mismatch: {name}")
            statuses[name] = status
        common_times = times["mass_exact_input"]
        common_runtime = statuses["mass_exact_input"]["runtime_fingerprint"]
        for name in (*present, "zero"):
            if (any(abs(times[name][i]-common_times[i]) > TIME_TOL for i in range(5))
                    or statuses[name]["runtime_fingerprint"] != common_runtime):
                raise ValueError(f"time/runtime mismatch: {name}")
        report["common_observed_times"] = [common_times[i] for i in range(5)]
        report["common_runtime_fingerprint"] = common_runtime
        report["all_initial_preflights_and_semantic_audits_valid"] = True
        for index in range(5):
            t = common_times[index]
            analytic = depth + B * math.sin(omega*t) * sinc * np.cos(k*xc)
            values = {}
            heights = {}
            for name in (*present, "zero"):
                prefix = PREFIXES[name] + "/segments/segment_0000/"
                values[name] = {
                    field: raw(tar, prefix + f"{field}_{index:04d}.raw", shape)
                    for field in ("vf", "ebvf", "vx", "vy")
                }
                heights[name] = h * np.sum(values[name]["vf"] * values[name]["ebvf"], axis=0)
            mass_ref = values["mass_exact_input"]
            q_ref = mass_ref["vf"] * mass_ref["ebvf"]
            speed_ref = float(np.sum(q_ref * (mass_ref["vx"]**2 + mass_ref["vy"]**2)))
            if speed_ref <= 0:
                raise ValueError("zero native reference speed norm")
            row = {"dump_index": index, "time": t, "results": {}}
            original_row = primary["series"][index]
            if original_row["dump_index"] != index or abs(original_row["time"]-t) > TIME_TOL:
                raise ValueError(f"primary dump-index/time mismatch: {index}")
            for name in present:
                arr = values[name]
                physical = float(np.sqrt(np.mean((heights[name]-analytic)**2))) / fixed_scale
                transfer = float(np.sqrt(np.mean((heights[name]-heights["mass_exact_input"])**2))) / fixed_scale
                velocity = math.sqrt(float(np.sum(q_ref * (
                    (arr["vx"]-mass_ref["vx"])**2 + (arr["vy"]-mass_ref["vy"])**2
                ))) / speed_ref)
                prior = original_row["results"][name]
                for computed, key in (
                    (physical, "physical_H_error_normalized_fixed_amplitude"),
                    (transfer, "H_difference_from_exact_input_branch_normalized"),
                    (velocity, "global_liquid_velocity_difference_from_exact_input_branch_relative_l2"),
                ):
                    if abs(computed - float(prior[key])) > NORM_TOL:
                        raise ValueError(f"norm mismatch: {name}/{index}/{key}")
                row["results"][name] = {
                    "physical_H_normalized": physical,
                    "same_mass_exact_native_H_normalized": transfer,
                    "same_mass_exact_native_velocity_Eu": velocity,
                }
                if np.max(np.abs(arr["ebvf"]-mass_ref["ebvf"])) > 1e-14:
                    raise ValueError(f"native EBVF differs: {name}/{index}")
            if np.max(np.abs(values["zero"]["ebvf"]-mass_ref["ebvf"])) > 1e-14:
                raise ValueError(f"zero native EBVF differs: {index}")
            row["zero_height_floor_rms"] = float(np.sqrt(np.mean((heights["zero"]-depth)**2)))
            report["series"].append(row)
    endpoint = report["series"][-1]["results"]
    m = endpoint["mass_matched"]["physical_H_normalized"]
    pm = endpoint["mass_practical"]["physical_H_normalized"]
    pp = endpoint["phase_practical"]["physical_H_normalized"]
    report["endpoint"] = endpoint
    report["endpoint_phase_practical_over_mass_matched_ratio"] = pp/m
    report["endpoint_phase_practical_to_mass_matched_error_reduction_percent"] = 100*(1-m/pp)
    report["endpoint_mass_practical_over_mass_matched_ratio"] = pm/m
    report["endpoint_mass_practical_to_mass_matched_error_reduction_percent"] = 100*(1-m/pm)
    report["nonzero_physical_H_rankings"] = [
        {"time": row["time"], "best_to_worst": sorted(
            row["results"], key=lambda name: row["results"][name]["physical_H_normalized"]
        )} for row in report["series"][1:]
    ]
    report["all_nonzero_mass_matched_better_than_phase_practical"] = all(
        row["results"]["mass_matched"]["physical_H_normalized"]
        < row["results"]["phase_practical"]["physical_H_normalized"]
        for row in report["series"][1:]
    )
    report["mass_matched_better_than_mass_practical_by_nonzero_time"] = [
        row["results"]["mass_matched"]["physical_H_normalized"]
        < row["results"]["mass_practical"]["physical_H_normalized"]
        for row in report["series"][1:]
    ]
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_archive_sha256": report["output_archive_sha256"],
        "branches": report["branches_in_primary"],
        "endpoint": endpoint,
        "combined_gain": report["endpoint_phase_practical_over_mass_matched_ratio"],
        "within_mass_gain": report["endpoint_mass_practical_over_mass_matched_ratio"],
        "mass_matched_better_than_mass_practical_by_nonzero_time": report["mass_matched_better_than_mass_practical_by_nonzero_time"],
    }, indent=2))


if __name__ == "__main__":
    main()
