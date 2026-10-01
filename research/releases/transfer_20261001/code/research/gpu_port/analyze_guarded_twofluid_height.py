"""Compare guarded native free-surface trajectories with the linear reference."""

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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(archive: tarfile.TarFile, path: str) -> bytes:
    member = archive.extractfile(path)
    if member is None:
        raise FileNotFoundError(path)
    return member.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branches", nargs="+", default=["zero", "A_phase", "Ahalf_phase"])
    args = parser.parse_args()
    nx, ny = 512, 32
    h = 64 / nx
    x = (np.arange(nx) + 0.5) * h
    envelope = np.exp(-1j * 2 * np.pi * 16 * x / 64)
    records = {}
    with tarfile.open(args.archive, "r:gz") as archive:
        for branch in args.branches:
            root = f"guarded_full/{branch}"
            status = json.loads(read(archive, root + "/status.json"))
            audits = [json.loads(line) for line in read(archive, root + "/semantic_audit.jsonl").decode().splitlines()]
            if status["state"] != "complete" or not audits or not all(row["valid"] for row in audits):
                raise ValueError(f"branch {branch} did not pass semantic gates")
            times = {}
            for row in audits:
                match = re.search(r"dump_(\d+)", row["source"])
                if match:
                    times[int(match.group(1))] = row["simulation_time"]
            amp = 0.0 if branch == "zero" else (1e-5 if "half" in branch.lower() else 2e-5)
            wave = Wave(potential_amplitude=amp)
            denom = wave.surface_amplitude * np.sinc(wave.k * h / (2 * np.pi)) / math.sqrt(2)
            rows = []
            for index, t in sorted(times.items()):
                segment = root + "/segments/segment_0000/"
                vf = np.frombuffer(read(archive, segment + f"vf_{index:04d}.raw"), dtype="<f8").reshape(ny, nx)
                eb = np.frombuffer(read(archive, segment + f"ebvf_{index:04d}.raw"), dtype="<f8").reshape(ny, nx)
                H = (vf * eb).sum(axis=0) * h
                if amp == 0:
                    rows.append({"dump_index": index, "time": t,
                                 "mean_height": float(np.mean(H)),
                                 "height_floor_rms_absolute": float(np.sqrt(np.mean((H - wave.depth)**2))),
                                 "height_floor_max_absolute": float(np.max(np.abs(H - wave.depth))),
                                 "first_mode_complex": [float(z) for z in (2*np.mean((H-H.mean())*envelope).real,
                                                                       2*np.mean((H-H.mean())*envelope).imag)]})
                    continue
                target = wave.cells(nx, t)["column_depth"]
                error = H - target
                first = 2 * np.mean((H - H.mean()) * envelope)
                desired = 2 * np.mean((target - target.mean()) * envelope)
                error_mode = 2 * np.mean(error * envelope)
                recon = error.mean() + np.real(error_mode * np.conj(envelope))
                rows.append({"dump_index": index, "time": t,
                             "mean_height": float(np.mean(H)),
                             "analytic_mean_height": float(np.mean(target)),
                             "height_error_rms_absolute": float(np.sqrt(np.mean(error**2))),
                             "height_error_normalized_fixed_amplitude": float(np.sqrt(np.mean(error**2))/denom),
                             "height_error_max_absolute": float(np.max(np.abs(error))),
                             "first_mode_complex": [float(first.real), float(first.imag)],
                             "analytic_first_mode_complex": [float(desired.real), float(desired.imag)],
                             "first_mode_error_normalized_amplitude": float(abs(error_mode)/(math.sqrt(2)*denom)),
                             "orthogonal_error_normalized_fixed_amplitude": float(np.sqrt(np.mean((error-recon)**2))/denom),
                             "analytic_wave_rms": float(np.sqrt(np.mean((target-target.mean())**2))),
                             "fixed_denominator": denom})
            records[branch] = {"status": status["state"], "audits": len(audits),
                               "max_mass_relative_error": max(abs(row["runtime_mass_relative_error"]) for row in audits),
                               "max_flux_divergence_linf": max(row["volume_flux_divergence_linf"] for row in audits),
                               "max_speed": max(row["maximum_speed_all_cells"] for row in audits),
                               "series": rows}
    report = {"schema": "guarded-twofluid-physical-height-analysis-v1",
              "archive_sha256": sha(args.archive), "code_sha256": sha(Path(__file__)),
              "reference_code_sha256": sha(Path(__file__).with_name("linear_twofluid_wave_reference.py")),
              "reference_scope": "linear inviscid two-fluid capillary-gravity wave, not exact nonlinear solver truth",
              "metric": "native column depth H=h*sum_y(vf*ebvf), L2 error against analytic cell-average H, normalized by B*sinc(kh/2)/sqrt(2) fixed at all times",
              "zero_is_separate_floor_not_subtracted": True,
              "branches": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({branch: {"endpoint": data["series"][-1],
                               "max_mass_relative_error": data["max_mass_relative_error"]}
                      for branch, data in records.items()}, indent=2))


if __name__ == "__main__":
    main()
