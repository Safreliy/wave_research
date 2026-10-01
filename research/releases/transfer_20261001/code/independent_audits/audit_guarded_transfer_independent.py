"""Independent raw-field audit of the guarded physical transfer pair.

No campaign analysis/reference module is imported. The exact-phase branch is
a same-grid numerical reference; the analytic height is a linear-wave model.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tarfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "flat_native_L9_twofluid" / "protocol.json"
FROZEN_OUTPUTS = ROOT / "twofluid_guarded_transfer_comparison_outputs.tar.gz"
EXACT_INPUTS = ROOT / "flat_native_L9_twofluid_inputs.tar.gz"
TRANSFER_INPUTS = ROOT / "flat_native_L9_guarded_transfer_inputs.tar.gz"
PRIMARY = ROOT / "guarded_transfer_height_analysis.json"
OUTPUT = ROOT / "guarded_transfer_independent_audit.json"
PREFIXES = {
    "phase_exact_input": "guarded_full/A_phase",
    "matched": "flat_native_L9_guarded_transfer/runs/full/A_matched",
    "practical": "flat_native_L9_guarded_transfer/runs/full/A_practical",
    "Ahalf_phase": "guarded_full/Ahalf_phase",
    "zero": "guarded_full/zero",
}
TIME_TOLERANCE = 1e-12
METRIC_TOLERANCE = 2e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contents(tar: tarfile.TarFile, name: str) -> bytes:
    member = tar.getmember(name)
    if not member.isfile():
        raise ValueError(f"missing regular archive member: {name}")
    return tar.extractfile(member).read()


def field(tar: tarfile.TarFile, name: str, shape: tuple[int, int]) -> np.ndarray:
    data = contents(tar, name)
    if len(data) != 8 * math.prod(shape):
        raise ValueError(f"wrong size for field {name}: {len(data)}")
    result = np.frombuffer(data, dtype="<f8").reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError(f"nonfinite field: {name}")
    return result


def observed_times(tar: tarfile.TarFile, prefix: str) -> dict[int, float]:
    lines = contents(tar, prefix + "/semantic_audit.jsonl").decode("utf-8").splitlines()
    result: dict[int, float] = {}
    for line in lines:
        record = json.loads(line)
        match = re.search(r"dump_(\d+)(?::exit)?$", record["source"])
        if not match:
            continue
        index = int(match.group(1))
        time = float(record["simulation_time"])
        if index in result and abs(result[index] - time) > TIME_TOLERANCE:
            raise ValueError(f"duplicate dump with conflicting time: {prefix}/{index}")
        result[index] = time
    if set(result) != set(range(5)):
        raise ValueError(f"unexpected dump times: {prefix}: {sorted(result)}")
    return result


def stat_final_time(tar: tarfile.TarFile, prefix: str) -> float:
    lines = contents(tar, prefix + "/segments/segment_0000/stat.dat").decode("utf-8").splitlines()
    return float(lines[-1].split()[lines[0].split().index("t")])


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    primary = json.loads(PRIMARY.read_text(encoding="utf-8"))
    wave = protocol["wave"]
    nx, ny = 512, 32
    shape = (ny, nx)
    length = float(wave["length"])
    h = length / nx
    k = 2 * math.pi * float(wave["mode"]) / length
    depth = float(wave["surface"] - wave["bottom"])
    gas_depth = float(wave["top"] - wave["surface"])
    rho_l, rho_g = float(wave["rho_liquid"]), float(wave["rho_gas"])
    omega = math.sqrt(
        ((rho_l - rho_g) * float(wave["gravity"]) * k + float(wave["sigma"]) * k**3)
        / (rho_l / math.tanh(k * depth) + rho_g * math.tanh(k * gas_depth))
    )
    B = float(wave["potential_amplitude"]) * k * math.tanh(k * depth) / omega
    sinc = math.sin(k * h / 2) / (k * h / 2)
    fixed_scale = abs(B) * sinc / math.sqrt(2)
    xc = (np.arange(nx, dtype=float) + 0.5) * h
    report = {
        "schema": "guarded-transfer-independent-raw-field-audit-v1",
        "scope": "Linear-wave physical height and same-grid exact-phase-input numerical discrepancies; not nonlinear exact truth",
        "checker_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(PROTOCOL),
        "output_archive_sha256": sha256(FROZEN_OUTPUTS),
        "exact_input_archive_sha256": sha256(EXACT_INPUTS),
        "transfer_input_archive_sha256": sha256(TRANSFER_INPUTS),
        "primary_analysis_sha256": sha256(PRIMARY),
        "grid_shape_yx": [ny, nx],
        "height_definition": "H_i=h*sum_j(VF_ji*EBVF_ji)",
        "analytic_height_definition": "depth+B*sin(omega*t)*sinc(k*h/2)*cos(k*xcenter)",
        "fixed_height_scale": fixed_scale,
        "velocity_reference_weight": "q_ref=VF_phase_exact_input*EBVF_phase_exact_input",
        "series": [],
    }
    if report["output_archive_sha256"] != primary["archive_sha256"]:
        raise ValueError("raw archive hash does not match primary analysis")
    with (tarfile.open(FROZEN_OUTPUTS, "r:gz") as raw,
          tarfile.open(EXACT_INPUTS, "r:gz") as exact_input,
          tarfile.open(TRANSFER_INPUTS, "r:gz") as transfer_input):
        exact_base = "flat_native_L9_twofluid/A_phase/"
        transfer_base = "flat_native_L9_guarded_transfer/"
        common_files = (
            "q.raw", "cs.raw", "vf.raw", "body.dat", "fluxx.raw",
            "fluxxp.raw", "fluxy.raw", "fluxyp.raw", "fluxeb.raw",
        )
        for branch in ("A_matched", "A_practical"):
            for name in common_files:
                if contents(exact_input, exact_base + name) != contents(
                    transfer_input, transfer_base + branch + "/" + name
                ):
                    raise ValueError(f"unequal frozen common input: {branch}/{name}")
        q0 = field(exact_input, exact_base + "q.raw", shape)
        cs0 = field(exact_input, exact_base + "cs.raw", shape)
        vf0 = field(exact_input, exact_base + "vf.raw", shape)
        gas = (cs0 > 0) & (vf0 == 0)
        if not np.array_equal(q0, cs0 * vf0) or not np.any(gas):
            raise ValueError("initial geometry contract or gas mask invalid")
        for branch in ("A_matched", "A_practical"):
            for component in ("vx.raw", "vy.raw"):
                original = field(exact_input, exact_base + component, shape)
                candidate = field(
                    transfer_input, transfer_base + branch + "/" + component, shape
                )
                if not np.array_equal(original[gas], candidate[gas]):
                    raise ValueError(f"gas input differs: {branch}/{component}")
        report["common_initial_inputs"] = {
            "byte_identical_files_for_both_transfer_branches": list(common_files),
            "bit_identical_gas_vx_vy_for_both_transfer_branches": True,
            "gas_cell_count": int(np.count_nonzero(gas)),
        }
        times = {}
        statuses = {}
        for name, prefix in PREFIXES.items():
            times[name] = observed_times(raw, prefix)
            status = json.loads(contents(raw, prefix + "/status.json"))
            if status["state"] != "complete" or status["return_code"] != 0:
                raise ValueError(f"incomplete run: {name}")
            if abs(status["progress_time"] - protocol["target_time"]) > TIME_TOLERANCE:
                raise ValueError(f"status target time mismatch: {name}")
            if abs(stat_final_time(raw, prefix) - protocol["target_time"]) > TIME_TOLERANCE:
                raise ValueError(f"stat target time mismatch: {name}")
            statuses[name] = status
        reference_times = times["phase_exact_input"]
        reference_runtime = statuses["phase_exact_input"]["runtime_fingerprint"]
        for name in PREFIXES:
            if any(abs(times[name][i] - reference_times[i]) > TIME_TOLERANCE for i in range(5)):
                raise ValueError(f"unequal native dump times: {name}")
            if statuses[name]["runtime_fingerprint"] != reference_runtime:
                raise ValueError(f"unequal native solver/library: {name}")
        report["common_observed_times"] = [reference_times[i] for i in range(5)]
        report["common_runtime_fingerprint"] = reference_runtime
        report["all_five_runs_complete_at_target"] = True
        for index in range(5):
            t = reference_times[index]
            analytic = depth + B * math.sin(omega * t) * sinc * np.cos(k * xc)
            fields = {}
            heights = {}
            for name in ("phase_exact_input", "matched", "practical"):
                prefix = PREFIXES[name] + "/segments/segment_0000/"
                fields[name] = {
                    component: field(raw, prefix + f"{component}_{index:04d}.raw", shape)
                    for component in ("vf", "ebvf", "vx", "vy")
                }
                heights[name] = h * np.sum(
                    fields[name]["vf"] * fields[name]["ebvf"], axis=0
                )
            q_ref = fields["phase_exact_input"]["vf"] * fields["phase_exact_input"]["ebvf"]
            velocity_denominator = float(np.sum(q_ref * (
                fields["phase_exact_input"]["vx"]**2
                + fields["phase_exact_input"]["vy"]**2
            )))
            if velocity_denominator <= 0:
                raise ValueError(f"zero reference velocity norm at dump {index}")
            entry = {"dump_index": index, "time": t, "branches": {}}
            primary_entry = primary["series"][index]
            if index != primary_entry["dump_index"] or abs(t - primary_entry["time"]) > TIME_TOLERANCE:
                raise ValueError(f"primary dump misalignment at {index}")
            for name in ("phase_exact_input", "matched", "practical"):
                f = fields[name]
                height_rms = float(np.sqrt(np.mean((heights[name] - analytic)**2)))
                transfer_rms = float(np.sqrt(np.mean(
                    (heights[name] - heights["phase_exact_input"])**2
                )))
                velocity_numerator = float(np.sum(q_ref * (
                    (f["vx"] - fields["phase_exact_input"]["vx"])**2
                    + (f["vy"] - fields["phase_exact_input"]["vy"])**2
                )))
                physical_H = height_rms / fixed_scale
                transfer_H = transfer_rms / fixed_scale
                velocity_E = math.sqrt(velocity_numerator / velocity_denominator)
                prior = primary_entry["results"][name]
                for computed, key in (
                    (physical_H, "physical_H_error_normalized_fixed_amplitude"),
                    (transfer_H, "H_difference_from_exact_input_branch_normalized"),
                    (velocity_E, "global_liquid_velocity_difference_from_exact_input_branch_relative_l2"),
                ):
                    if abs(computed - float(prior[key])) > METRIC_TOLERANCE:
                        raise ValueError(f"primary metric mismatch: {name}/{index}/{key}")
                entry["branches"][name] = {
                    "physical_H_error_rms": height_rms,
                    "physical_H_normalized": physical_H,
                    "same_phase_native_H_normalized": transfer_H,
                    "same_phase_native_velocity_Eu": velocity_E,
                    "mean_height": float(np.mean(heights[name])),
                }
            eb_ref = fields["phase_exact_input"]["ebvf"]
            entry["max_embedded_volume_fraction_difference"] = max(
                float(np.max(np.abs(fields[name]["ebvf"] - eb_ref)))
                for name in ("matched", "practical")
            )
            if entry["max_embedded_volume_fraction_difference"] > 1e-14:
                raise ValueError(f"native embedded geometry differed at dump {index}")
            report["series"].append(entry)
    nonzero = report["series"][1:]
    physical_ranking = [
        row["branches"]["phase_exact_input"]["physical_H_normalized"]
        < row["branches"]["matched"]["physical_H_normalized"]
        < row["branches"]["practical"]["physical_H_normalized"]
        for row in nonzero
    ]
    final = report["series"][-1]["branches"]
    matched = final["matched"]["physical_H_normalized"]
    practical = final["practical"]["physical_H_normalized"]
    report["all_recorded_nonzero_times_exact_matched_practical_physical_H_ranking"] = all(physical_ranking)
    report["physical_H_ranking_per_nonzero_dump"] = physical_ranking
    report["endpoint_physical_H_reduction_fraction_matched_vs_practical"] = 1 - matched / practical
    report["endpoint_practical_over_matched_physical_H_ratio"] = practical / matched
    report["endpoint_practical_over_matched_same_phase_H_ratio"] = (
        final["practical"]["same_phase_native_H_normalized"]
        / final["matched"]["same_phase_native_H_normalized"]
    )
    report["endpoint_practical_over_matched_same_phase_velocity_ratio"] = (
        final["practical"]["same_phase_native_velocity_Eu"]
        / final["matched"]["same_phase_native_velocity_Eu"]
    )
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_archive_sha256": report["output_archive_sha256"],
        "times": report["common_observed_times"],
        "endpoint": final,
        "physical_H_reduction_percent": 100 * report["endpoint_physical_H_reduction_fraction_matched_vs_practical"],
        "all_nonzero_physical_H_ranking": report["all_recorded_nonzero_times_exact_matched_practical_physical_H_ranking"],
    }, indent=2))


if __name__ == "__main__":
    main()
