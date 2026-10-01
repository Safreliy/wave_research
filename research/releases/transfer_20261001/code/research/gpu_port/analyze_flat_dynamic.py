"""Compare native flat-wave transfer branches to exact-input receiver branch.

The exact-input branch is an independent *initialisation* reference. Its
subsequent Aphros solution contains the same spatial/time discretisation and
is not an analytic solution of the two-phase Navier--Stokes problem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


BRANCHES = ("fitted", "bilinear_centroid", "exact_initial_means")
FIELDS = ("vf", "ebvf", "vx", "vy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_times(path: Path) -> dict[int, float]:
    out = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        matched = re.search(r"dump_(\d+)", row.get("source", ""))
        if matched and row.get("valid") and "simulation_time" in row:
            out[int(matched.group(1))] = float(row["simulation_time"])
    return out


def read_field(root: Path, name: str, index: int, shape: tuple[int, int]) -> np.ndarray:
    path = root / "segments" / "segment_0000" / f"{name}_{index:04d}.raw"
    result = np.fromfile(path, dtype="<f8")
    if result.size != shape[0] * shape[1] or not np.isfinite(result).all():
        raise ValueError(f"invalid {path}")
    return result.reshape(shape)


def relative_norm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def compare(field: dict[str, np.ndarray], reference: dict[str, np.ndarray], h: float) -> dict:
    q = field["vf"] * field["ebvf"]
    qref = reference["vf"] * reference["ebvf"]
    d2 = (field["vx"] - reference["vx"]) ** 2 + (field["vy"] - reference["vy"]) ** 2
    ref2 = reference["vx"] ** 2 + reference["vy"] ** 2
    depth = q.sum(axis=0) * h
    reference_depth = qref.sum(axis=0) * h
    wave = reference_depth - reference_depth.mean()
    wave_norm = float(np.linalg.norm(wave))
    wave_relative = (float(np.linalg.norm(depth-reference_depth)/wave_norm)
                     if wave_norm > 1e-12 * np.linalg.norm(reference_depth) else None)
    mode16 = 2.0*np.fft.rfft(wave)[16]/wave.size
    return {
        "global_liquid_velocity_relative_l2": float(np.sqrt(np.sum(qref * d2) / np.sum(qref * ref2))),
        "vof_relative_l2": relative_norm(field["vf"], reference["vf"]),
        "column_depth_relative_l2": relative_norm(depth, reference_depth),
        "column_depth_wave_relative_l2": wave_relative,
        "reference_mean_column_depth": float(reference_depth.mean()),
        "reference_mode16_column_depth_real": float(mode16.real),
        "reference_mode16_column_depth_imag": float(mode16.imag),
        "reference_mode16_column_depth_amplitude": float(abs(mode16)),
        "column_depth_max_absolute": float(np.max(np.abs(depth-reference_depth))),
        "mass_difference_relative": float(abs(q.sum()-qref.sum())/qref.sum()),
    }


def load(root: Path, shape: tuple[int, int], target: float) -> tuple[dict, dict[int, dict[str, np.ndarray]]]:
    status_path = root / "status.json"
    audit_path = root / "semantic_audit.jsonl"
    status = json.loads(status_path.read_text())
    if status["state"] != "complete" or abs(status["checkpoint"]["time"]-target)>1e-10:
        raise ValueError(f"run did not complete exact target: {root}")
    times = dump_times(audit_path)
    audit_rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
    if not audit_rows or not all(row.get("valid") for row in audit_rows):
        raise ValueError(f"failed semantic checkpoint in {root}")
    progressed = [row for row in audit_rows if row.get("simulation_time", 0.0) > 0]
    stat_path = root / "segments" / "segment_0000" / "stat.dat"
    stat_lines = stat_path.read_text().splitlines()
    stat_header = stat_lines[0].split()
    step_column, dt_column = stat_header.index("step"), stat_header.index("dt")
    stat_rows = [line.split() for line in stat_lines[1:] if line.strip()]
    actual_steps = round(float(stat_rows[-1][step_column]))
    actual_dtmax = max(float(row[dt_column]) for row in stat_rows)
    selected = {}
    for index, value in times.items():
        if 0 <= value <= target + 1e-10:
            selected[index] = {name: read_field(root, name, index, shape) for name in FIELDS}
    for expected in (0.0, target):
        hits = [i for i, t in times.items() if abs(t-expected)<=1e-10]
        if len(hits) != 1:
            raise ValueError(f"expected one dump at {expected} in {root}, got {hits}")
    metadata = {
        "status_sha256": sha256(status_path), "semantic_audit_sha256": sha256(audit_path),
        "time": status["checkpoint"]["time"],
        "initial_dump_index": min(selected), "final_dump_index": max(selected),
        "recorded_dump_times": times,
        "status_state": status["state"],
        "solver_steps": actual_steps,
        "observed_maximum_dt": actual_dtmax,
        "stat_sha256": sha256(stat_path),
        "semantic_gate_summary": {
            "audited_dumps": len(audit_rows),
            "maximum_initial_mass_relative_error": max(abs(row["initial_mass_relative_error"]) for row in audit_rows),
            "maximum_runtime_mass_relative_error_after_start": max(abs(row["runtime_mass_relative_error"]) for row in progressed),
            "maximum_runtime_vof_mass_relative_error_after_start": max(abs(row["runtime_vof_mass_relative_error"]) for row in progressed),
            "maximum_embedded_volume_flux_divergence_linf_after_start": max(row["volume_flux_divergence_linf"] for row in progressed),
            "initial_loaded_momentum_absolute_error": audit_rows[0]["momentum_absolute_difference_from_initial_target"],
            "maximum_liquid_regular_speed": max(row["maximum_speed_liquid_regular_cells"] for row in audit_rows),
            "maximum_all_cell_speed": max(row["maximum_speed_all_cells"] for row in audit_rows),
        },
    }
    return metadata, selected


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-root", type=Path,
                   help="override frozen absolute reference path after archive extraction")
    a = p.parse_args()
    protocol = json.loads((a.root / "protocol.json").read_text())
    target = protocol["target_time_one_tenth_period"]
    nx = protocol["geometry"]["nx"]
    ny = protocol["geometry"]["ny"]
    shape = (ny, nx)
    h = protocol["geometry"]["period"] / nx
    result = {"schema": "flat-harmonic-native-dynamic-analysis-v1",
              "input_protocol_sha256": sha256(a.root / "protocol.json"),
              "launch_protocol_sha256": sha256(a.root / "native_launch_protocol.json"),
              "target_time": target, "fraction_linear_wave_period": 0.1,
              "norm": "reference-branch liquid q weights for both numerator and denominator",
              "reference_limit": "exact phase-mean initialisation, shared receiver dynamics; not exact physical truth",
              "phases": {}, "code_sha256": sha256(Path(__file__))}
    loaded = {}
    for phase in ("full", "half"):
        phase_roots = {name: a.root / "runs" / phase / name for name in BRANCHES}
        reused_reference = a.reference_root or protocol.get("reused_reference_native_root")
        if reused_reference:
            phase_roots["exact_initial_means"] = (
                Path(reused_reference) / "runs" / phase / "exact_initial_means"
            )
        if not all((root / "status.json").is_file() for root in phase_roots.values()):
            continue
        metadata, snapshots = {}, {}
        for name, root in phase_roots.items():
            metadata[name], snapshots[name] = load(root, shape, target)
        labels = {name: {"t0": snapshots[name][metadata[name]["initial_dump_index"]],
                         "final": snapshots[name][metadata[name]["final_dump_index"]]}
                  for name in BRANCHES}
        comparisons = {}
        for time_label in ("t0", "final"):
            comparisons[time_label] = {
                name: compare(labels[name][time_label], labels["exact_initial_means"][time_label], h)
                for name in ("fitted", "bilinear_centroid")
            }
        common_indices = set(snapshots["exact_initial_means"])
        for name in BRANCHES:
            common_indices &= set(snapshots[name])
        series = []
        for index in sorted(common_indices):
            ref_time = metadata["exact_initial_means"]["recorded_dump_times"][index]
            if any(abs(metadata[name]["recorded_dump_times"][index]-ref_time)>1e-10 for name in BRANCHES):
                raise ValueError(f"unmatched dump time at index {index}")
            series.append({"time": ref_time, "dump_index": index,
                           "fitted": compare(snapshots["fitted"][index], snapshots["exact_initial_means"][index], h),
                           "bilinear_centroid": compare(snapshots["bilinear_centroid"][index], snapshots["exact_initial_means"][index], h)})
        result["phases"][phase] = {"runs": metadata, "comparisons": comparisons,
                                    "time_series": series}
        loaded[phase] = labels
    if "full" in loaded and "half" in loaded:
        result["time_step_sensitivity"] = {
            name: compare(loaded["full"][name]["final"], loaded["half"][name]["final"], h)
            for name in BRANCHES
        }
    half_reference = None
    if "half" in loaded:
        half_reference = loaded["half"]["exact_initial_means"]["final"]
    elif "full" in loaded and reused_reference:
        half_reference_root = Path(reused_reference) / "runs" / "half" / "exact_initial_means"
        if (half_reference_root / "status.json").is_file():
            half_reference_metadata, half_reference_snapshots = load(half_reference_root, shape, target)
            half_reference = half_reference_snapshots[half_reference_metadata["final_dump_index"]]
            result["cross_time_reference_half_run"] = half_reference_metadata
    if "full" in loaded and half_reference is not None:
        result["cross_time_reference"] = {
            "description": "full-step candidate endpoint versus half-step exact-initial-input numerical reference; shared time-discretisation error is retained",
            "fitted": compare(loaded["full"]["fitted"]["final"],
                              half_reference, h),
            "bilinear_centroid": compare(loaded["full"]["bilinear_centroid"]["final"],
                                         half_reference, h),
        }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({phase: value["comparisons"] for phase, value in result["phases"].items()}, indent=2))


if __name__ == "__main__":
    main()
