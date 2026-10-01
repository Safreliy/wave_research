"""Freeze selected raw native fields, configs and audits from two-fluid runs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


FIELDS = {"vf", "ebvf", "vx", "vy", "p", "fluxx", "fluxxp", "fluxy", "fluxyp", "fluxeb"}
ROOT_FILES = {"status.json", "initial_preflight.json", "semantic_audit.jsonl"}
SEGMENT_FILES = {"config.conf", "out.conf", "run.log", "stat.dat", "amgx.log"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True,
                        help="Path relative to campaign root, e.g. guarded_full/zero")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selected = []
    runs = []
    for run_name in args.run:
        path = (args.campaign_root / run_name).resolve()
        if not path.is_relative_to(args.campaign_root.resolve()) or not path.is_dir():
            raise ValueError(run_name)
        state = json.loads((path / "status.json").read_text())["state"]
        runs.append({"path": run_name, "state": state})
        for name in ROOT_FILES:
            file = path / name
            if file.is_file():
                selected.append(file)
        for segment in sorted((path / "segments").glob("segment_*")):
            for file in segment.iterdir():
                if file.name in SEGMENT_FILES or (file.suffix == ".raw" and file.stem.rsplit("_", 1)[0] in FIELDS):
                    selected.append(file)
    selected.sort()
    manifest = {"schema": "twofluid-selected-native-outputs-v1", "runs": runs,
                "files": [{"path": p.relative_to(args.campaign_root).as_posix(),
                           "sha256": sha(p), "bytes": p.stat().st_size} for p in selected]}
    payload = (json.dumps(manifest, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for path in selected:
            archive.add(path, arcname=path.relative_to(args.campaign_root).as_posix(), recursive=False)
        info = tarfile.TarInfo("allowlist.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output),
                      "runs": runs, "files": len(selected), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
