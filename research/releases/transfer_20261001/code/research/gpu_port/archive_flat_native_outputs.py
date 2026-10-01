"""Freeze selected native field dumps and audit logs for flat-wave benchmark.

The archive uses an explicit per-run file list and hashes every payload. It
omits full rotating checkpoints, machine credentials, and transient processes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


LABEL_PHASES = {
    "flat_native_L9": ("full", "half"),
    "flat_native_L10": ("full", "half"),
    "flat_native_L9_trace4": ("full",),
    "flat_native_L10_trace4": ("full", "half"),
    "flat_native_L11_resolved": ("full",),
}
FIELDS = ("vf", "ebvf", "vx", "vy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected: list[Path] = []
    runs = {}
    for label, phases in LABEL_PHASES.items():
        root = args.campaign_root / label
        protocol = json.loads((root / "protocol.json").read_text())
        launch = json.loads((root / "native_launch_protocol.json").read_text())
        reused = bool(protocol.get("reused_reference_native_root"))
        config_names = [Path(launch["commands"][f"{phase}_fitted"][
            launch["commands"][f"{phase}_fitted"].index("--base-config") + 1]).name
            for phase in ("full", "half")]
        for name in ("protocol.json", "native_launch_protocol.json", "preparation_report.json",
                     "harmonic_source.npz", "benchmark_supervisor.py", "analyze_flat_dynamic.py",
                     *config_names, "analysis_full_series.json"):
            selected.append(root / name)
        if reused:
            selected.append(root / "trace_diagnosis.json")
        if label != "flat_native_L11_resolved":
            selected.append(root / "input_reconstruction_audit.json")
        for phase in phases:
            for branch in ("fitted", "bilinear_centroid", "exact_initial_means"):
                if reused and branch == "exact_initial_means":
                    continue
                run = root / "runs" / phase / branch
                status = json.loads((run / "status.json").read_text())
                if status["state"] != "complete":
                    raise ValueError(f"incomplete run: {run}")
                runs[f"{label}/{phase}/{branch}"] = status["checkpoint"]["time"]
                for name in ("status.json", "initial_preflight.json", "semantic_audit.jsonl"):
                    selected.append(run / name)
                segment = run / "segments" / "segment_0000"
                for name in ("run.log", "stat.dat", "amgx.log", "config.conf"):
                    if (segment / name).is_file():
                        selected.append(segment / name)
                for name in FIELDS:
                    selected.extend(sorted(segment.glob(f"{name}_*.raw")))
    names = [p.relative_to(args.campaign_root).as_posix() for p in selected]
    if len(set(names)) != len(names):
        raise ValueError("duplicate archive member")
    if not all(p.is_file() for p in selected):
        raise FileNotFoundError([str(p) for p in selected if not p.is_file()])
    manifest = {
        "schema": "flat-native-selected-output-archive-v1",
        "scope": "five frozen input conditions; declared full/half runs and reused exact-input native references in trace4 pairs",
        "declared_phases": LABEL_PHASES,
        "runs": runs,
        "files": [{"path": name, "sha256": sha256(path), "size": path.stat().st_size}
                  for path, name in zip(selected, names)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for path, name in zip(selected, names):
            info = archive.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as stream:
                archive.addfile(info, stream)
        payload = (json.dumps(manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("allowlist.json")
        info.size = len(payload)
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        archive.addfile(info, io.BytesIO(payload))
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output),
                      "members": len(selected), "runs": len(runs),
                      "bytes": args.output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
