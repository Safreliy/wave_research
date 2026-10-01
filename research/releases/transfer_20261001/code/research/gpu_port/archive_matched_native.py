"""Freeze selected native fields and audit logs for one matched-trace run."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("full", "half"), default="full")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    launch_name = ("native_launch_protocol.json" if args.phase == "full"
                   else "native_half_launch_protocol.json")
    launch = json.loads((root / launch_name).read_text())
    config = Path(launch["command"][launch["command"].index("--base-config")+1]).name
    run = root / "runs" / args.phase / "fitted"
    status = json.loads((run / "status.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    if status["state"] != "complete" or abs(status["checkpoint"]["time"]-protocol["target_time_one_tenth_period"]) > 1e-10:
        raise ValueError("candidate run incomplete")
    audit = [json.loads(line) for line in (run / "semantic_audit.jsonl").read_text().splitlines()]
    if not audit or not all(row.get("valid") for row in audit):
        raise ValueError("native semantic audit failed")
    selected = [root / name for name in ("protocol.json", launch_name,
                                               "preparation_report.json", "harmonic_source.npz",
                                               "benchmark_supervisor.py", config)]
    if args.phase == "half":
        selected.append(root / "native_launch_protocol.json")
    selected += [run / name for name in ("status.json", "initial_preflight.json",
                                             "semantic_audit.jsonl")]
    segment = run / "segments" / "segment_0000"
    selected += [segment / name for name in ("run.log", "stat.dat", "amgx.log", "config.conf")]
    for field in ("vf", "ebvf", "vx", "vy"):
        selected.extend(sorted(segment.glob(f"{field}_*.raw")))
    if not all(path.is_file() for path in selected):
        raise FileNotFoundError([str(p) for p in selected if not p.is_file()])
    names = [path.relative_to(root.parent).as_posix() for path in selected]
    manifest = {"schema": "flat-matched-native-selected-output-v1",
                "phase": args.phase,
                "semantic_audits": len(audit),
                "files": [{"path": name, "sha256": sha256(path), "size": path.stat().st_size}
                          for path, name in zip(selected, names)]}
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
    print(json.dumps({"output_sha256": sha256(args.output), "files": len(selected),
                      "bytes": args.output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
