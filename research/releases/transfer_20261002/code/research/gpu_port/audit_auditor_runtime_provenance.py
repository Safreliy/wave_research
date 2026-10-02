"""Verify current guarded Aphros bytes against the archived build receipt."""

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


def members(path: Path, strip: str = "") -> dict[str, str]:
    result = {}
    with tarfile.open(path, "r:gz") as archive:
        for entry in archive:
            if not entry.isfile():
                continue
            if strip and not entry.name.startswith(strip):
                raise ValueError(entry.name)
            name = entry.name[len(strip):]
            result[name] = hashlib.sha256(archive.extractfile(entry).read()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    guard = root / "runtime_provenance"
    build = json.loads((guard / "build_manifest.json").read_text())
    for kind, record in build["source_archives"].items():
        if sha(guard / record["archive"]) != record["sha256"]:
            raise ValueError(f"frozen {kind} source tar mismatch")
    for kind, record in build["runtime_archives"].items():
        if sha(guard / record["archive"]) != record["sha256"]:
            raise ValueError(f"frozen {kind} runtime tar mismatch")
    if sha(guard / "zero_flux_guard.patch") != build["guard_patch_sha256"]:
        raise ValueError("guard patch mismatch")
    for name, digest in build["build_state_files"].items():
        if sha(guard / name) != digest:
            raise ValueError(f"build receipt mismatch: {name}")
    fresh = members(root / "aphros_guard_sources_20261002.tar.gz",
                    "aphros_zero_flux_guard_20261001/")
    frozen = members(guard / "aphros_guard_patched_source.tar.gz")
    missing = sorted(frozen.keys() - fresh.keys())
    if missing != ["SOURCE_TREE_MANIFEST.json"]:
        raise ValueError(f"unexpected missing source entries: {missing}")
    changed = sorted(name for name in fresh.keys() & frozen.keys()
                     if fresh[name] != frozen[name])
    if changed or fresh.keys() - frozen.keys():
        raise ValueError({"changed": changed, "extra": sorted(fresh.keys()-frozen.keys())})
    runtime = members(guard / "aphros_guard_guarded_runtime.tar.gz")
    for item in build["runtime_archives"]["guarded"]["files"]:
        if runtime[item["path"]] != item["sha256"]:
            raise ValueError(f"runtime member mismatch: {item['path']}")
    report = {"schema": "auditor-guarded-source-build-match-v1",
              "upstream_commit": build["upstream_head_commit"],
              "fresh_remote_source_archive_sha256": sha(root / "aphros_guard_sources_20261002.tar.gz"),
              "frozen_patched_source_archive_sha256": sha(guard / "aphros_guard_patched_source.tar.gz"),
              "matched_source_files": len(fresh),
              "missing_only_generated_manifest": missing,
              "changed_source_files": changed,
              "zero_flux_patch_sha256": build["guard_patch_sha256"],
              "build_manifest_sha256": sha(guard / "build_manifest.json"),
              "guarded_runtime_sha256": {item["path"]: item["sha256"]
                                         for item in build["runtime_archives"]["guarded"]["files"]},
              "meaning": "fresh guarded source bytes match the source tree tied to the saved CMake/Ninja build receipt; execute only when live binary hashes match the receipt"}
    (guard / "source_build_match_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
