"""Archive isolated Aphros binaries and CMake state with runtime hashes."""

from __future__ import annotations

import argparse
import hashlib
import io
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
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("baseline", "guarded"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = [args.prefix / "bin/ap.mfer", args.prefix / "lib/libaphros.so",
             args.prefix / "lib/libaphros_c.so"]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError([str(p) for p in paths if not p.is_file()])
    manifest = {"schema": "aphros-zero-flux-guard-runtime-v1", "variant": args.variant,
                "prefix": str(args.prefix), "files": [{"path": p.relative_to(args.prefix).as_posix(),
                                                         "sha256": sha(p), "bytes": p.stat().st_size}
                                                        for p in paths],
                "external_dependency": "libamgxsh.so is a symlink to existing host dependency and is not included"}
    payload = (json.dumps(manifest, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=path.relative_to(args.prefix).as_posix(), recursive=False)
        info = tarfile.TarInfo("RUNTIME_MANIFEST.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output),
                      "bytes": args.output.stat().st_size, "variant": args.variant}))


if __name__ == "__main__":
    main()
