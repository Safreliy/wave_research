"""Independent same-runtime raw-field audit of all three flat-wave grids."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tarfile
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Run:
    def __init__(self, archive: Path, tag: str, label: str, nx: int):
        self.archive_path = archive
        self.archive = tarfile.open(archive, "r:gz")
        self.base = f"{tag}/runs/{label}"
        self.segment = self.base + "/segments/segment_0000"
        self.shape = (nx//16, nx)
        self.label = label
        self.status = json.loads(self.read(self.base + "/status.json"))
        if self.status["state"] != "complete":
            raise ValueError((archive, self.status["state"]))
        self.audit = [json.loads(line) for line in self.read(
            self.base + "/semantic_audit.jsonl").decode().splitlines() if line.strip()]
        if not self.audit or not all(row.get("valid") for row in self.audit):
            raise ValueError(f"invalid semantic audit: {archive}")
        self.times = {}
        for row in self.audit:
            match = re.search(r"dump_(\d+)", row.get("source", ""))
            if match:
                self.times[int(match.group(1))] = float(row["simulation_time"])
        if len(self.times) < 3:
            raise ValueError(f"insufficient dumps: {archive}")

    def read(self, name: str) -> bytes:
        return self.archive.extractfile(name).read()

    def field(self, kind: str, index: int) -> np.ndarray:
        name = f"{self.segment}/{kind}_{index:04d}.raw"
        data = np.frombuffer(self.read(name), dtype="<f8")
        if data.size != math.prod(self.shape) or not np.isfinite(data).all():
            raise ValueError(name)
        return data.reshape(self.shape)

    def report(self) -> dict:
        rows = [r for r in self.audit if r["simulation_time"] > 0]
        return {"archive_sha256": sha(self.archive_path),
                "status": self.status["state"],
                "runtime_fingerprint": self.status["runtime_fingerprint"],
                "effective_config_sha256": hashlib.sha256(
                    self.read(self.segment + "/config.conf")).hexdigest(),
                "endpoint_time": self.status["checkpoint"]["time"],
                "audit_records": len(self.audit),
                "maximum_post_start_mass_relative_error": max(
                    abs(row["runtime_mass_relative_error"]) for row in rows),
                "maximum_post_start_flux_divergence_linf": max(
                    row["volume_flux_divergence_linf"] for row in rows),
                "raw_field_sha256": {
                    f"{kind}_{index:04d}.raw": hashlib.sha256(
                        self.read(f"{self.segment}/{kind}_{index:04d}.raw")).hexdigest()
                    for kind in ("vx", "vy", "vf", "ebvf") for index in sorted(self.times)}}


def compare(candidate: Run, reference: Run, index: int, h: float) -> dict:
    q = reference.field("vf", index) * reference.field("ebvf", index)
    ref = np.stack((reference.field("vx", index), reference.field("vy", index)))
    vel = np.stack((candidate.field("vx", index), candidate.field("vy", index)))
    if not np.isfinite(q).all() or np.min(q) < -1e-9:
        raise ValueError("invalid reference aperture")
    velocity = float(np.sqrt(np.sum(q*np.sum((vel-ref)**2, axis=0)) /
                             np.sum(q*np.sum(ref**2, axis=0))))
    vf = candidate.field("vf", index)
    vf_ref = reference.field("vf", index)
    vof = float(np.linalg.norm(vf-vf_ref)/np.linalg.norm(vf_ref))
    depth = (vf*candidate.field("ebvf", index)).sum(axis=0)*h
    depth_ref = q.sum(axis=0)*h
    wave = depth_ref-depth_ref.mean()
    wave_norm = float(np.linalg.norm(wave))
    depth_error = (float(np.linalg.norm(depth-depth_ref)/wave_norm)
                   if wave_norm > 1e-12*np.linalg.norm(depth_ref) else None)
    return {"global_liquid_velocity_relative_l2": velocity,
            "vof_relative_l2": vof,
            "column_depth_wave_relative_l2": depth_error,
            "column_depth_max_absolute": float(np.max(abs(depth-depth_ref)))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root / "dynamic/campaign.json").read_text())
    result = {"schema": "auditor-single-runtime-dynamic-raw-audit-v1",
              "campaign_sha256": sha(root / "dynamic/campaign.json"),
              "input_archive_sha256": sha(root / "dynamic_inputs.tar.gz"),
              "runtime_sha256": plan["expected_runtime_sha256"],
              "source_build_match_sha256": sha(root / "runtime_provenance/source_build_match_report.json"),
              "claim_scope": "same-grid numerical exact-input trajectory; no exact nonlinear flow claim",
              "grids": {}}
    for tag, details in plan["grids"].items():
        nx = details["nx"]
        h = 64/nx
        runs = {label: Run(root / f"{tag}_{label}_outputs.tar.gz", tag, label, nx)
                for label in ("exact", "matched", "bilinear")}
        expected = plan["expected_runtime_sha256"]
        for label, run in runs.items():
            fingerprint = run.status["runtime_fingerprint"]
            if (fingerprint["solver_sha256"] != expected[fingerprint["solver_path"]]
                    or fingerprint["solver_library_sha256"] != expected[fingerprint["solver_library_path"]]):
                raise ValueError(f"{tag}/{label}: executed runtime fingerprint mismatch")
        indices = sorted(runs["exact"].times)
        if any(sorted(r.times) != indices for r in runs.values()):
            raise ValueError(f"{tag}: dump indices differ")
        series = []
        for index in indices:
            times = [runs[label].times[index] for label in ("exact", "matched", "bilinear")]
            if max(times)-min(times) > 1e-10:
                raise ValueError(f"{tag}: dump time mismatch {index}: {times}")
            series.append({"dump_index": index, "time": times[0],
                           "matched": compare(runs["matched"], runs["exact"], index, h),
                           "bilinear": compare(runs["bilinear"], runs["exact"], index, h)})
        target = float(plan["commands"][f"{tag}_exact"][plan["commands"][f"{tag}_exact"].index("--target-time")+1])
        if abs(series[-1]["time"]-target) > 1e-10:
            raise ValueError(f"{tag}: wrong endpoint")
        if any(abs(r.status["checkpoint"]["time"]-target) > 1e-10 for r in runs.values()):
            raise ValueError(f"{tag}: incomplete checkpoint")
        end = series[-1]
        result["grids"][tag] = {
            "nx": nx, "ny": nx//16, "config_sha256": details["config_sha256"],
            "supervisor_sha256": details["supervisor_sha256"],
            "input_hashes": {label: details["branches"][label]["input_sha256"]
                             for label in runs},
            "runs": {label: run.report() for label, run in runs.items()},
            "series": series,
            "endpoint_gain_bilinear_over_matched": {
                metric: (end["bilinear"][metric]/end["matched"][metric]
                         if end["matched"][metric] else None)
                for metric in ("global_liquid_velocity_relative_l2",
                               "vof_relative_l2", "column_depth_wave_relative_l2")}}
        for run in runs.values():
            run.archive.close()
    output = root / "dynamic_three_grid_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({tag: {"endpoint": data["series"][-1],
                           "gain": data["endpoint_gain_bilinear_over_matched"]}
                      for tag, data in result["grids"].items()}, indent=2))


if __name__ == "__main__":
    main()
