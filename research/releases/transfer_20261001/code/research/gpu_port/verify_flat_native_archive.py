"""Verify each frozen native-output archive member against its SHA allowlist."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import PurePosixPath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    args = parser.parse_args()
    with tarfile.open(args.archive, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or names.count("allowlist.json") != 1:
            raise ValueError("duplicate or missing allowlist member")
        if any(not member.isfile() or PurePosixPath(member.name).is_absolute()
               or ".." in PurePosixPath(member.name).parts for member in members):
            raise ValueError("unsafe or non-file archive member")
        allowlist = json.load(archive.extractfile("allowlist.json"))
        if allowlist["schema"] != "flat-native-selected-output-archive-v1":
            raise ValueError("unexpected allowlist schema")
        expected = {item["path"]: item for item in allowlist["files"]}
        if set(names) != set(expected) | {"allowlist.json"}:
            raise ValueError("archive membership differs from allowlist")
        for name, item in expected.items():
            member = archive.getmember(name)
            source = archive.extractfile(member)
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
            if size != item["size"] or digest.hexdigest() != item["sha256"]:
                raise ValueError(f"archive payload hash/size mismatch: {name}")
    print(json.dumps({"schema": allowlist["schema"], "verified_files": len(expected),
                      "runs": len(allowlist["runs"]), "declared_phases": allowlist["declared_phases"]},
                     indent=2))


if __name__ == "__main__":
    main()
