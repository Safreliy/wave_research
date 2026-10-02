"""Archive a complete isolated Aphros source tree with per-file hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    members = sorted(p for p in args.source.rglob("*") if p.is_file() or p.is_symlink())
    manifest = {
        "schema": "isolated-aphros-source-tree-v1",
        "source_root": str(args.source),
        "files": [{"path": p.relative_to(args.source).as_posix(),
                   "sha256": sha(p) if p.is_file() else None,
                   "symlink": p.readlink().as_posix() if p.is_symlink() else None}
                  for p in members],
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for path in members:
            archive.add(path, arcname=path.relative_to(args.source).as_posix(), recursive=False)
        import io
        info = tarfile.TarInfo("SOURCE_TREE_MANIFEST.json")
        info.size = len(manifest_bytes)
        archive.addfile(info, io.BytesIO(manifest_bytes))
    print(json.dumps({"archive": str(args.output), "sha256": sha(args.output),
                      "files": len(members), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
