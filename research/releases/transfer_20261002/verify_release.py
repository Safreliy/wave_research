"""Verify the frozen auditor follow-up; optionally recompute the three raw audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
BRANCHES = ("exact", "matched", "bilinear")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        relative = PurePosixPath(member.name)
        if (relative.is_absolute() or ".." in relative.parts or not (member.isfile() or member.isdir())
                or member.issym() or member.islnk()):
            raise ValueError(f"unsafe tar member: {member.name}")
    return members


def verify_files() -> dict:
    manifest = read_json(ROOT / "SHA256SUMS.json")
    if manifest.get("schema") != "auditor-revision-release-sha256-v1" or manifest.get("doi") is not None:
        raise ValueError("wrong release manifest schema or unexpected DOI")
    expected = set()
    for item in manifest["files"]:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in expected:
            raise ValueError(f"unsafe/duplicate path: {relative}")
        path = ROOT.joinpath(*relative.parts)
        if not path.is_file() or path.stat().st_size != item["size"] or sha(path) != item["sha256"]:
            raise AssertionError(f"missing or changed: {relative}")
        expected.add(relative.as_posix())
    actual = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file()} - {"SHA256SUMS.json"}
    if expected != actual:
        raise AssertionError({"missing": sorted(expected - actual), "unlisted": sorted(actual - expected)})
    return {"verified_files": len(expected), "manifest_sha256": sha(ROOT / "SHA256SUMS.json")}


def verify_audit_links() -> dict:
    results = {}
    for name, grids, prefix, schema in (
        ("dynamic_three_grid_audit.json", ("L9", "L10", "L11"), "", "auditor-single-runtime-dynamic-raw-audit-v1"),
        ("physical_two_grid_audit.json", ("L9", "L10"), "physical_", "auditor-single-runtime-physical-raw-audit-v1"),
    ):
        audit = read_json(ROOT / name)
        if audit.get("schema") != schema or set(audit.get("grids", {})) != set(grids):
            raise ValueError(f"incomplete audit: {name}")
        campaign = "physical" if prefix else "dynamic"
        if audit["campaign_sha256"] != sha(ROOT / campaign / "campaign.json"):
            raise AssertionError(f"campaign hash differs: {name}")
        if audit["input_archive_sha256"] != sha(ROOT / f"{campaign}_inputs.tar.gz"):
            raise AssertionError(f"input archive hash differs: {name}")
        if audit["source_build_match_sha256"] != sha(ROOT / "runtime_provenance/source_build_match_report.json"):
            raise AssertionError(f"source/build evidence hash differs: {name}")
        for grid in grids:
            row = audit["grids"][grid]
            if set(row.get("runs", {})) != set(BRANCHES) or len(row.get("series", [])) < 3:
                raise ValueError(f"missing runs/series in {name}:{grid}")
            for branch in BRANCHES:
                archive_path = ROOT / f"{prefix}{grid}_{branch}_outputs.tar.gz"
                run = row["runs"][branch]
                if run["status"] != "complete" or run["archive_sha256"] != sha(archive_path):
                    raise AssertionError(f"run hash/status differs: {archive_path.name}")
                with tarfile.open(archive_path, "r:gz") as archive:
                    members = {item.name: item for item in safe_members(archive)}
                    base = f"{grid}/runs/{branch}"
                    for suffix in ("status.json", "semantic_audit.jsonl"):
                        if base + "/" + suffix not in members:
                            raise AssertionError(f"missing {suffix}: {archive_path.name}")
                    status = json.loads(archive.extractfile(members[base + "/status.json"]).read())
                    if status.get("state") != "complete":
                        raise AssertionError(f"native run not complete: {archive_path.name}")
                    segment = base + "/segments/segment_0000/"
                    for raw_name, raw_sha in run["raw_field_sha256"].items():
                        raw = archive.extractfile(members[segment + raw_name])
                        if hashlib.sha256(raw.read()).hexdigest() != raw_sha:
                            raise AssertionError(f"raw field changed: {archive_path.name}!{raw_name}")
        results[campaign + "_runs"] = len(grids) * len(BRANCHES)
    with tarfile.open(ROOT / "joint_refinement_inputs.tar.gz", "r:gz") as archive:
        members = {item.name: item for item in safe_members(archive)}
        for grid in ("L9", "L10", "L11"):
            base = f"joint_{grid}"
            for suffix in ("protocol.json", "preparation_report.json", "harmonic_source.npz",
                           "exact_initial_means/q.raw", "exact_initial_means/vx.raw", "exact_initial_means/vy.raw",
                           "fitted/vx.raw", "fitted/vy.raw", "bilinear_centroid/vx.raw", "bilinear_centroid/vy.raw"):
                if base + "/" + suffix not in members:
                    raise AssertionError(f"missing joint member: {base}/{suffix}")
    return results


def verify_input_archives() -> dict:
    checked = 0
    common = ("body.dat", "cs.raw", "fluxeb.raw", "fluxx.raw", "fluxxp.raw",
              "fluxy.raw", "fluxyp.raw", "q.raw", "vf.raw")
    for campaign, grids in (("dynamic", ("L9", "L10", "L11")),
                            ("physical", ("L9", "L10"))):
        plan = read_json(ROOT / campaign / "campaign.json")
        if set(plan["grids"]) != set(grids):
            raise ValueError(f"wrong {campaign} grid set")
        with tarfile.open(ROOT / f"{campaign}_inputs.tar.gz", "r:gz") as archive:
            members = {member.name: member for member in safe_members(archive)}
            for grid in grids:
                branches = plan["grids"][grid]["branches"]
                if set(branches) != set(BRANCHES):
                    raise ValueError(f"wrong {campaign}/{grid} branch set")
                for branch in BRANCHES:
                    hashes = branches[branch]["input_sha256"]
                    for filename, expected in hashes.items():
                        name = f"{campaign}/{grid}/{branch}/{filename}"
                        if name not in members:
                            raise AssertionError(f"missing archived input: {name}")
                        actual = hashlib.sha256(archive.extractfile(members[name]).read()).hexdigest()
                        if actual != expected:
                            raise AssertionError(f"input hash changed: {name}")
                        checked += 1
                for filename in common:
                    hashes = [branches[branch]["input_sha256"][filename]
                              for branch in BRANCHES]
                    if len(set(hashes)) != 1:
                        raise AssertionError(f"branchwise common field differs: {campaign}/{grid}/{filename}")
    return {"verified_input_members": checked}


def extract_safe(path: Path, target: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        members = safe_members(archive)
        for member in members:
            output = target.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, output.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def same_numbers(actual: object, expected: object, location: str = "root") -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"replay keys differ at {location}")
        for key in actual:
            same_numbers(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"replay length differs at {location}")
        for index, (a, b) in enumerate(zip(actual, expected)):
            same_numbers(a, b, f"{location}[{index}]")
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and not isinstance(actual, bool):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
            raise AssertionError(f"replay value differs at {location}: {actual} != {expected}")
    elif actual != expected:
        raise AssertionError(f"replay value differs at {location}: {actual!r} != {expected!r}")


def replay() -> None:
    with tempfile.TemporaryDirectory(prefix="auditor_revision_replay_") as temp:
        root = Path(temp)
        for name in ("dynamic_inputs.tar.gz", "physical_inputs.tar.gz", "joint_refinement_inputs.tar.gz"):
            extract_safe(ROOT / name, root)
            link_or_copy(ROOT / name, root / name)
        link_or_copy(ROOT / "runtime_provenance/source_build_match_report.json",
                     root / "runtime_provenance/source_build_match_report.json")
        for grid in ("L9", "L10", "L11"):
            for branch in BRANCHES:
                name = f"{grid}_{branch}_outputs.tar.gz"
                link_or_copy(ROOT / name, root / name)
        for grid in ("L9", "L10"):
            for branch in BRANCHES:
                name = f"physical_{grid}_{branch}_outputs.tar.gz"
                link_or_copy(ROOT / name, root / name)
        code = ROOT / "code/research/gpu_port"
        child_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for script, result in (
            ("audit_auditor_joint_refinement.py", "joint_refinement_initial_audit.json"),
            ("audit_auditor_dynamic_outputs.py", "dynamic_three_grid_audit.json"),
            ("audit_auditor_physical_outputs.py", "physical_two_grid_audit.json"),
        ):
            subprocess.run([sys.executable, str(code / script), "--root", str(root)],
                           check=True, stdout=subprocess.DEVNULL, env=child_env)
            same_numbers(read_json(root / result), read_json(ROOT / result), result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="extract archived inputs and recompute all three audits")
    args = parser.parse_args()
    result = verify_files()
    result.update(verify_audit_links())
    result.update(verify_input_archives())
    if args.replay:
        replay()
        result["replayed_audits"] = 3
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
