"""Independent raw-field height audit for the guarded two-fluid phase pilot.

Reads only the frozen protocol, the selected native tar, and the published
height-analysis JSON. It does not import the campaign's analysis/reference code.
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
ARCHIVE = ROOT / "twofluid_guarded_phase_outputs.tar.gz"
COMPARISON = ROOT / "guarded_phase_height_analysis.json"
CONFIG = ROOT / "flat_native_L9_twofluid" / "twofluid_slip_inviscid_dt00125.conf"
OUTPUT = ROOT / "guarded_phase_height_independent_audit.json"
BRANCH_AMPLITUDES = {"zero": 0.0, "A_phase": 2e-5, "Ahalf_phase": 1e-5}
TIME_TOLERANCE = 1e-12
METRIC_TOLERANCE = 2e-10


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_field(tar: tarfile.TarFile, name: str, shape: tuple[int, int]) -> np.ndarray:
    member = tar.getmember(name)
    if not member.isfile() or member.size != 8 * math.prod(shape):
        raise ValueError(f"wrong raw-field size: {name}: {member.size}")
    data = tar.extractfile(member).read()
    return np.frombuffer(data, dtype="<f8").reshape(shape)


def observed_times(tar: tarfile.TarFile, branch: str) -> dict[int, float]:
    path = f"guarded_full/{branch}/semantic_audit.jsonl"
    lines = tar.extractfile(path).read().decode("utf-8").splitlines()
    result: dict[int, float] = {}
    for line in lines:
        record = json.loads(line)
        hit = re.search(r"dump_(\d+)(?::exit)?$", record["source"])
        if hit:
            index = int(hit.group(1))
            value = float(record["simulation_time"])
            if index in result and abs(result[index] - value) > TIME_TOLERANCE:
                raise ValueError(f"duplicate dump with conflicting time: {branch}/{index}")
            result[index] = value
    if set(result) != set(range(5)):
        raise ValueError(f"unexpected audit dump indices: {branch}: {sorted(result)}")
    return result


def stat_final_time(tar: tarfile.TarFile, branch: str) -> float:
    name = f"guarded_full/{branch}/segments/segment_0000/stat.dat"
    lines = tar.extractfile(name).read().decode("utf-8").splitlines()
    columns = lines[0].split()
    return float(lines[-1].split()[columns.index("t")])


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    wave = protocol["wave"]
    length = float(wave["length"])
    nx, ny = 512, 32
    h = length / nx
    depth = float(wave["surface"] - wave["bottom"])
    gas_depth = float(wave["top"] - wave["surface"])
    k = 2 * math.pi * float(wave["mode"]) / length
    rho_l, rho_g = float(wave["rho_liquid"]), float(wave["rho_gas"])
    gravity, sigma = float(wave["gravity"]), float(wave["sigma"])
    omega2 = ((rho_l - rho_g) * gravity * k + sigma * k**3) / (
        rho_l / math.tanh(k * depth) + rho_g * math.tanh(k * gas_depth)
    )
    omega = math.sqrt(omega2)
    sinc = math.sin(k * h / 2) / (k * h / 2)
    xc = (np.arange(nx, dtype=float) + 0.5) * h
    config_text = CONFIG.read_text(encoding="utf-8")
    if file_sha256(CONFIG) != protocol["config_sha256"]:
        raise ValueError("receiver config no longer matches frozen protocol")
    for key, expected in (
        ("rho1", rho_g), ("rho2", rho_l), ("mu1", 0.0), ("mu2", 0.0),
        ("sigma", sigma),
    ):
        match = re.search(rf"^set double {key} ([^\s]+)$", config_text, re.MULTILINE)
        if match is None or not math.isclose(float(match.group(1)), expected, abs_tol=1e-15):
            raise ValueError(f"config/wave mismatch: {key}")
    if not all(item in config_text for item in (
        "slipwall {", "outletpressure 0 {", "set vect gravity 0 -1 0",
    )):
        raise ValueError("declared bottom/top/gravity setting missing")
    B_full = float(wave["potential_amplitude"]) * k * math.tanh(k * depth) / omega
    vertical_interface = float(wave["potential_amplitude"]) * k * math.tanh(k * depth)
    dynamic_left = float(wave["potential_amplitude"]) * omega * (
        rho_l + rho_g * math.tanh(k * depth) * math.tanh(k * gas_depth)
    )
    dynamic_right = ((rho_l - rho_g) * gravity + sigma * k * k) * B_full
    if abs(dynamic_left - dynamic_right) > 1e-14 * max(abs(dynamic_left), abs(dynamic_right)):
        raise ValueError("linear dynamic-interface relation does not close")
    report = {
        "schema": "guarded-twofluid-independent-raw-height-audit-v1",
        "scope": "linear two-fluid analytic column height, not exact nonlinear flow truth",
        "protocol_sha256": file_sha256(PROTOCOL),
        "archive_sha256": file_sha256(ARCHIVE),
        "comparison_sha256": file_sha256(COMPARISON),
        "checker_sha256": file_sha256(Path(__file__)),
        "config_sha256": file_sha256(CONFIG),
        "source_formula": {
            "H": "depth + B*sin(omega*t)*sinc(k*h/2)*cos(k*xcenter)",
            "B": "A*k*tanh(k*depth)/omega",
            "omega_squared": "((rho_l-rho_g)*g*k+sigma*k^3)/(rho_l*coth(k*depth)+rho_g*tanh(k*gas_depth))",
            "fixed_normalizer": "abs(B)*sinc(k*h/2)/sqrt(2)",
            "native_H": "h*sum_y(VF*EBVF)",
            "wave_parameters": wave,
            "depth": depth,
            "gas_depth": gas_depth,
            "k": k,
            "h": h,
            "omega": omega,
            "sinc": sinc,
        },
        "grid_shape_yx": [ny, nx],
        "linear_boundary_algebra": {
            "gas_potential": "-A*tanh(k*depth)*sinh(k*(top-y))/cosh(k*gas_depth)*cos(k*x)*cos(omega*t)",
            "liquid_and_gas_interface_vertical_velocity_amplitude": vertical_interface,
            "top_gas_potential": 0.0,
            "dynamic_interface_balance_absolute_residual": abs(dynamic_left - dynamic_right),
            "receiver_config_declares": "rho2=liquid, rho1=gas, mu1=mu2=0, sigma=0.001, g_y=-1, slipwall bottom, outletpressure 0 top",
        },
        "branches": {},
    }
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        expected_times = None
        expected_runtime = None
        for branch, amplitude in BRANCH_AMPLITUDES.items():
            times = observed_times(tar, branch)
            status_path = f"guarded_full/{branch}/status.json"
            status = json.load(tar.extractfile(status_path))
            if status["state"] != "complete" or status["return_code"] != 0:
                raise ValueError(f"incomplete native run: {branch}")
            runtime = status["runtime_fingerprint"]
            if expected_runtime is None:
                expected_runtime = runtime
            elif runtime != expected_runtime:
                raise ValueError(f"unequal native runtime fingerprint: {branch}")
            if expected_times is None:
                expected_times = times
            elif any(abs(times[i] - expected_times[i]) > TIME_TOLERANCE for i in range(5)):
                raise ValueError(f"unequal observed dump times: {branch}")
            final_stat = stat_final_time(tar, branch)
            if (abs(final_stat - times[4]) > TIME_TOLERANCE
                    or abs(status["progress_time"] - times[4]) > TIME_TOLERANCE
                    or abs(times[4] - protocol["target_time"]) > TIME_TOLERANCE):
                raise ValueError(f"terminal time mismatch: {branch}")
            B = amplitude * k * math.tanh(k * depth) / omega
            denominator = abs(B) * sinc / math.sqrt(2) if amplitude else None
            entries = []
            for index in range(5):
                prefix = f"guarded_full/{branch}/segments/segment_0000/"
                vf = raw_field(tar, prefix + f"vf_{index:04d}.raw", (ny, nx))
                ebvf = raw_field(tar, prefix + f"ebvf_{index:04d}.raw", (ny, nx))
                if not np.isfinite(vf).all() or not np.isfinite(ebvf).all():
                    raise ValueError(f"nonfinite raw field: {branch}/{index}")
                H = h * np.sum(vf * ebvf, axis=0)
                t = times[index]
                analytic_H = depth + B * math.sin(omega * t) * sinc * np.cos(k * xc)
                error = H - analytic_H
                rms = float(np.sqrt(np.mean(error * error)))
                entry = {
                    "dump_index": index,
                    "observed_time": t,
                    "mean_height": float(np.mean(H)),
                    "rms_absolute": rms,
                    "max_absolute": float(np.max(np.abs(error))),
                }
                primary = comparison["branches"][branch]["series"][index]
                if abs(t - primary["time"]) > TIME_TOLERANCE:
                    raise ValueError(f"primary time mismatch: {branch}/{index}")
                if amplitude:
                    normalized = rms / denominator
                    prior = float(primary["height_error_normalized_fixed_amplitude"])
                    if abs(normalized - prior) > METRIC_TOLERANCE:
                        raise ValueError(f"primary metric mismatch: {branch}/{index}")
                    entry.update({
                        "fixed_normalizer": denominator,
                        "normalized_fixed_amplitude": normalized,
                        "difference_from_primary_normalized": normalized - prior,
                    })
                else:
                    prior = float(primary["height_floor_rms_absolute"])
                    if abs(rms - prior) > METRIC_TOLERANCE:
                        raise ValueError(f"zero floor mismatch: {branch}/{index}")
                    entry["difference_from_primary_absolute_floor"] = rms - prior
                entries.append(entry)
            report["branches"][branch] = {
                "amplitude": amplitude,
                "B": B,
                "status_complete": True,
                "status_final_time": status["progress_time"],
                "stat_final_time": final_stat,
                "series": entries,
            }
    report["runtime_fingerprint_common"] = expected_runtime
    report["common_observed_times"] = [expected_times[i] for i in range(5)]
    report["algebra_and_config_check"] = (
        "The frozen config hash and declared gas/liquid densities, zero "
        "viscosities, gravity, surface tension, slipwall bottom, and pressure "
        "outlet top match the protocol. The gas potential vanishes at the top, "
        "its interface vertical derivative equals the liquid derivative, and "
        "the linear pressure-jump balance closes. This is not nonlinear truth."
    )
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "archive_sha256": report["archive_sha256"],
        "times": report["common_observed_times"],
        "endpoints": {name: record["series"][-1] for name, record in report["branches"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
