"""Independent raw-height and frozen mixture-input audit for the guarded pilot.

The phase means below come from closed integrals of the declared t=0 liquid
and gas potentials. No campaign analysis or wave-reference module is imported.
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
INPUTS = ROOT / "flat_native_L9_twofluid_inputs.tar.gz"
OUTPUTS = ROOT / "twofluid_guarded_mass_outputs.tar.gz"
PRIMARY = ROOT / "guarded_mass_height_analysis.json"
REPORT = ROOT / "guarded_mass_independent_audit.json"
BRANCHES = ("zero", "A_phase", "Ahalf_phase", "A_mass", "Ahalf_mass")
TIME_TOL = 1e-12
NORM_TOL = 2e-10
INPUT_TOL = 2e-13


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def member_data(tar: tarfile.TarFile, name: str) -> bytes:
    member = tar.getmember(name)
    if not member.isfile():
        raise ValueError(f"non-file archive member: {name}")
    return tar.extractfile(member).read()


def array(tar: tarfile.TarFile, name: str, shape: tuple[int, int]) -> np.ndarray:
    data = member_data(tar, name)
    if len(data) != 8 * math.prod(shape):
        raise ValueError(f"wrong raw size: {name}")
    result = np.frombuffer(data, dtype="<f8").reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError(f"nonfinite raw array: {name}")
    return result


def audited_times(tar: tarfile.TarFile, branch: str) -> dict[int, float]:
    lines = member_data(tar, f"guarded_full/{branch}/semantic_audit.jsonl").decode().splitlines()
    observed: dict[int, float] = {}
    for line in lines:
        record = json.loads(line)
        match = re.search(r"dump_(\d+)(?::exit)?$", record["source"])
        if not match:
            continue
        index, time = int(match.group(1)), float(record["simulation_time"])
        if index in observed and abs(observed[index] - time) > TIME_TOL:
            raise ValueError(f"duplicate audited time differs: {branch}/{index}")
        observed[index] = time
    if set(observed) != set(range(5)):
        raise ValueError(f"unexpected dump-index set for {branch}: {sorted(observed)}")
    return observed


def closed_phase_means(wave: dict, amplitude: float, nx: int, ny: int):
    """Exact t=0 rectangular phase means for a flat interface and bed."""
    length = float(wave["length"])
    h = length / nx
    k = 2 * math.pi * int(wave["mode"]) / length
    bed, surface, top = (float(wave[key]) for key in ("bottom", "surface", "top"))
    liquid_depth, gas_depth = surface - bed, top - surface
    xc = (np.arange(nx) + 0.5) * h
    sinc = math.sin(k * h / 2) / (k * h / 2)
    mean_sin = (sinc * np.sin(k * xc))[None, :]
    mean_cos = (sinc * np.cos(k * xc))[None, :]
    ylo = (np.arange(ny) * h)[:, None]
    yhi = ylo + h
    lo_l = np.maximum(ylo, bed)
    hi_l = np.maximum(lo_l, np.minimum(yhi, surface))
    lo_g = np.minimum(top, np.maximum(ylo, surface))
    hi_g = np.maximum(lo_g, np.minimum(yhi, top))
    dl = hi_l - lo_l
    dg = hi_g - lo_g
    coefficient_l = amplitude / math.cosh(k * liquid_depth)
    coefficient_g = amplitude * math.tanh(k * liquid_depth) / math.cosh(k * gas_depth)
    ul = np.zeros((ny, nx), dtype=float)
    vl = np.zeros((ny, nx), dtype=float)
    ug = np.zeros((ny, nx), dtype=float)
    vg = np.zeros((ny, nx), dtype=float)
    mask_l = np.broadcast_to(dl > 0, (ny, nx))
    mask_g = np.broadcast_to(dg > 0, (ny, nx))
    ul_base = np.divide(
        np.sinh(k * (hi_l - bed)) - np.sinh(k * (lo_l - bed)),
        dl, out=np.zeros_like(dl), where=dl > 0,
    )
    vl_base = np.divide(
        np.cosh(k * (hi_l - bed)) - np.cosh(k * (lo_l - bed)),
        dl, out=np.zeros_like(dl), where=dl > 0,
    )
    ug_base = np.divide(
        np.cosh(k * (top - lo_g)) - np.cosh(k * (top - hi_g)),
        dg, out=np.zeros_like(dg), where=dg > 0,
    )
    vg_base = np.divide(
        np.sinh(k * (top - lo_g)) - np.sinh(k * (top - hi_g)),
        dg, out=np.zeros_like(dg), where=dg > 0,
    )
    ul[:] = -coefficient_l * ul_base * mean_sin
    vl[:] = coefficient_l * vl_base * mean_cos
    ug[:] = coefficient_g * ug_base * mean_sin
    vg[:] = coefficient_g * vg_base * mean_cos
    return (ul, vl), (ug, vg), (np.broadcast_to(dl, (ny, nx)) / h,
                                    np.broadcast_to(dg, (ny, nx)) / h), mask_l, mask_g


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    primary = json.loads(PRIMARY.read_text(encoding="utf-8"))
    wave = protocol["wave"]
    nx, ny = 512, 32
    shape = ny, nx
    h = float(wave["length"]) / nx
    k = 2 * math.pi * int(wave["mode"]) / float(wave["length"])
    depth = float(wave["surface"] - wave["bottom"])
    gas_depth = float(wave["top"] - wave["surface"])
    rho_l, rho_g = float(wave["rho_liquid"]), float(wave["rho_gas"])
    omega = math.sqrt(
        ((rho_l - rho_g) * float(wave["gravity"]) * k + float(wave["sigma"]) * k**3)
        / (rho_l / math.tanh(k * depth) + rho_g * math.tanh(k * gas_depth))
    )
    sinc = math.sin(k * h / 2) / (k * h / 2)
    xc = (np.arange(nx) + 0.5) * h
    report = {
        "schema": "guarded-mass-mixture-independent-audit-v1",
        "scope": "Frozen mixture input and linear two-fluid analytic height; not an exact nonlinear trajectory",
        "checker_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(PROTOCOL),
        "input_archive_sha256": sha256(INPUTS),
        "output_archive_sha256": sha256(OUTPUTS),
        "primary_analysis_sha256": sha256(PRIMARY),
        "grid_shape_yx": [ny, nx],
        "mixture_formula": "(rho_l*q*u_l + rho_g*(cs-q)*u_g)/(rho_l*q+rho_g*(cs-q))",
        "native_height_formula": "h*sum_y(VF*EBVF)",
        "analytic_height_formula": "depth+B*sin(omega*t)*sinc(k*h/2)*cos(k*xcenter)",
        "input_checks": {}, "branches": {},
    }
    if report["output_archive_sha256"] != primary["archive_sha256"]:
        raise ValueError("frozen output archive differs from primary record")
    with tarfile.open(INPUTS, "r:gz") as frozen_input:
        for full, half in (("A_phase", "A_mass"), ("Ahalf_phase", "Ahalf_mass")):
            amplitude = 2e-5 if full == "A_phase" else 1e-5
            base_phase = "flat_native_L9_twofluid/" + full + "/"
            base_mass = "flat_native_L9_twofluid/" + half + "/"
            for name in ("q.raw", "cs.raw", "vf.raw", "body.dat"):
                if member_data(frozen_input, base_phase + name) != member_data(frozen_input, base_mass + name):
                    raise ValueError(f"non-common frozen geometry: {full}/{half}/{name}")
            q = array(frozen_input, base_mass + "q.raw", shape)
            cs = array(frozen_input, base_mass + "cs.raw", shape)
            (ul, vl), (ug, vg), (q_closed, gas_closed), liquid, gas = closed_phase_means(
                wave, amplitude, nx, ny
            )
            geometry_error = max(float(np.max(np.abs(q - q_closed))),
                                 float(np.max(np.abs((cs - q) - gas_closed))))
            if geometry_error > INPUT_TOL:
                raise ValueError(f"closed geometry differs from frozen input: {full}")
            phase_expected = (
                np.where(liquid, ul, ug), np.where(liquid, vl, vg)
            )
            cell_mass = rho_l * q + rho_g * (cs - q)
            mass_expected = tuple(np.divide(
                rho_l * q * liquid_component + rho_g * (cs - q) * gas_component,
                cell_mass, out=np.zeros(shape), where=cell_mass > 0,
            ) for liquid_component, gas_component in zip((ul, vl), (ug, vg)))
            volume_expected = tuple(np.divide(
                q * liquid_component + (cs - q) * gas_component,
                cs, out=np.zeros(shape), where=cs > 0,
            ) for liquid_component, gas_component in zip((ul, vl), (ug, vg)))
            mixed = (q > 0) & ((cs - q) > 0)
            if not np.any(mixed):
                raise ValueError("no mixed phase cells")
            phase_error = 0.0
            mass_error = 0.0
            wrong_volume_gap = 0.0
            for component, expected_phase, expected_mass, wrong_volume in zip(
                ("vx.raw", "vy.raw"), phase_expected, mass_expected, volume_expected
            ):
                phase_data = array(frozen_input, base_phase + component, shape)
                mass_data = array(frozen_input, base_mass + component, shape)
                phase_error = max(phase_error, float(np.max(np.abs(phase_data - expected_phase))))
                mass_error = max(mass_error, float(np.max(np.abs(mass_data - expected_mass))))
                wrong_volume_gap = max(
                    wrong_volume_gap, float(np.max(np.abs(mass_data[mixed] - wrong_volume[mixed])))
                )
            if max(phase_error, mass_error) > INPUT_TOL or wrong_volume_gap < 1e-9:
                raise ValueError(f"frozen velocity is not the density-weighted mixture: {half}")
            report["input_checks"][half] = {
                "amplitude": amplitude,
                "geometry_max_abs_difference": geometry_error,
                "phase_input_max_abs_difference_from_closed_means": phase_error,
                "mass_input_max_abs_difference_from_density_weighted_formula": mass_error,
                "mass_input_max_abs_difference_from_wrong_volume_mean_on_mixed_cells": wrong_volume_gap,
                "mixed_cell_count": int(np.count_nonzero(mixed)),
            }
    with tarfile.open(OUTPUTS, "r:gz") as frozen_output:
        reference_times = None
        reference_runtime = None
        for branch in BRANCHES:
            times = audited_times(frozen_output, branch)
            status = json.loads(member_data(frozen_output, f"guarded_full/{branch}/status.json"))
            if status["state"] != "complete" or status["return_code"] != 0:
                raise ValueError(f"incomplete native branch: {branch}")
            if abs(status["progress_time"] - protocol["target_time"]) > TIME_TOL:
                raise ValueError(f"endpoint-time mismatch: {branch}")
            if reference_times is None:
                reference_times = times
                reference_runtime = status["runtime_fingerprint"]
            elif (any(abs(times[i] - reference_times[i]) > TIME_TOL for i in range(5))
                  or status["runtime_fingerprint"] != reference_runtime):
                raise ValueError(f"run times/runtime differ for {branch}")
            amplitude = 2e-5 if branch.startswith("A_") else (1e-5 if branch.startswith("Ahalf_") else 0.0)
            B = amplitude * k * math.tanh(k * depth) / omega
            denominator = abs(B) * sinc / math.sqrt(2) if amplitude else None
            rows = []
            for index in range(5):
                prefix = f"guarded_full/{branch}/segments/segment_0000/"
                vf = array(frozen_output, prefix + f"vf_{index:04d}.raw", shape)
                ebvf = array(frozen_output, prefix + f"ebvf_{index:04d}.raw", shape)
                H = h * np.sum(vf * ebvf, axis=0)
                t = times[index]
                H_analytic = depth + B * math.sin(omega * t) * sinc * np.cos(k * xc)
                rms = float(np.sqrt(np.mean((H - H_analytic)**2)))
                original = primary["branches"][branch]["series"][index]
                if original["dump_index"] != index or abs(original["time"] - t) > TIME_TOL:
                    raise ValueError(f"primary dump alignment differs: {branch}/{index}")
                row = {"dump_index": index, "time": t, "mean_height": float(np.mean(H)),
                       "physical_H_error_rms": rms}
                if amplitude:
                    normalized = rms / denominator
                    if abs(normalized - original["height_error_normalized_fixed_amplitude"]) > NORM_TOL:
                        raise ValueError(f"physical H norm mismatch: {branch}/{index}")
                    row["physical_H_normalized_fixed_amplitude"] = normalized
                    row["fixed_normalizer"] = denominator
                elif abs(rms - original["height_floor_rms_absolute"]) > NORM_TOL:
                    raise ValueError(f"zero-floor mismatch: {index}")
                rows.append(row)
            report["branches"][branch] = {"amplitude": amplitude, "series": rows,
                                           "status_complete": True}
    report["common_observed_times"] = [reference_times[i] for i in range(5)]
    report["common_runtime_fingerprint"] = reference_runtime
    report["endpoint"] = {branch: report["branches"][branch]["series"][-1] for branch in BRANCHES}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_archive_sha256": report["output_archive_sha256"],
        "input_checks": report["input_checks"],
        "endpoint": report["endpoint"],
    }, indent=2))


if __name__ == "__main__":
    main()
