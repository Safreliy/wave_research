"""Recompute all declared full-step velocity errors from frozen native raw fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np


LABELS = (
    "flat_native_L9", "flat_native_L10", "flat_native_L9_trace4",
    "flat_native_L10_trace4", "flat_native_L11_resolved",
)
REFERENCE_LABEL = {
    "flat_native_L9_trace4": "flat_native_L9",
    "flat_native_L10_trace4": "flat_native_L10",
}
BRANCHES = ("fitted", "bilinear_centroid")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with tarfile.open(args.archive, "r:gz") as archive:
        def read(name: str) -> bytes:
            return archive.extractfile(name).read()

        for label in LABELS:
            analysis = json.loads(read(f"{label}/analysis_full_series.json"))
            protocol = json.loads(read(f"{label}/protocol.json"))
            shape = (protocol["geometry"]["ny"], protocol["geometry"]["nx"])
            full = analysis["phases"]["full"]
            reference_label = REFERENCE_LABEL.get(label, label)

            def field(branch: str, kind: str, *, reference: bool = False) -> np.ndarray:
                index = full["runs"][branch]["final_dump_index"]
                owner = reference_label if reference else label
                member = (f"{owner}/runs/full/{branch}/segments/segment_0000/"
                          f"{kind}_{index:04d}.raw")
                values = np.frombuffer(read(member), dtype="<f8")
                if values.size != shape[0] * shape[1] or not np.isfinite(values).all():
                    raise ValueError(f"invalid field {member}")
                return values.reshape(shape)

            ref_vx = field("exact_initial_means", "vx", reference=True)
            ref_vy = field("exact_initial_means", "vy", reference=True)
            q = (field("exact_initial_means", "vf", reference=True)
                 * field("exact_initial_means", "ebvf", reference=True))
            denominator = np.sum(q * (ref_vx**2 + ref_vy**2))
            for branch in BRANCHES:
                error = ((field(branch, "vx") - ref_vx)**2
                         + (field(branch, "vy") - ref_vy)**2)
                actual = float(np.sqrt(np.sum(q * error) / denominator))
                recorded = full["comparisons"]["final"][branch][
                    "global_liquid_velocity_relative_l2"]
                relative_difference = abs(actual - recorded) / recorded
                if relative_difference > 2e-12:
                    raise ValueError(f"archive/raw analysis mismatch: {label}/{branch}")
                rows.append({"case": label, "branch": branch,
                             "computed_from_raw": actual, "recorded_in_analysis": recorded,
                             "relative_difference": relative_difference})
    report = {
        "schema": "flat-native-frozen-raw-endpoint-audit-v1",
        "archive_sha256": sha256(args.archive),
        "code_sha256": sha256(Path(__file__)),
        "relative_tolerance": 2e-12,
        "all_passed": True,
        "comparisons": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive_sha256": report["archive_sha256"],
                      "verified_comparisons": len(rows),
                      "max_relative_difference": max(row["relative_difference"] for row in rows),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
