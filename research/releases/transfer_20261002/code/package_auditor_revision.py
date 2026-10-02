"""Freeze the complete 2026-10-02 auditor follow-up as a self-checking release.

The campaign writes its results asynchronously. This program deliberately refuses to
package missing or incomplete audits, or to replace an existing release directory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
RESULTS = REPOSITORY / "research/results/transfer_auditor_revision_20261002"
RELEASE = REPOSITORY / "tmp/wave_research_transfer_20261001/research/releases/transfer_20261002"
BRANCHES = ("exact", "matched", "bilinear")
DYNAMIC_GRIDS = ("L9", "L10", "L11")
PHYSICAL_GRIDS = ("L9", "L10")

ROOT_FILES = (
    "PREDECLARED_PROTOCOL.md",
    "PROTOCOL_AT_DYNAMIC_FREEZE.md",
    "PROTOCOL_AT_PHYSICAL_FREEZE.md",
    "dynamic_inputs.tar.gz",
    "physical_inputs.tar.gz",
    "joint_refinement_initial_audit.json",
    "dynamic_three_grid_audit.json",
    "physical_two_grid_audit.json",
    "physical_initial_metadata_erratum.json",
    "cost_L9.json",
    "cost_L10.json",
    "cost_hardware.json",
    "dynamic/campaign.json",
    "physical/campaign.json",
    "freeze_protocol_snapshots.py",
)

RUNTIME_FILES = (
    "AMGX_CMakeCache_20261002.txt",
    "AMGX_DEPENDENCY.json",
    "AMGX_PCG_AGGREGATION_JACOBI.json",
    "AMGX_source_cc1cebdb_20261002.tar.gz",
    "APHROS_LICENSE",
    "aphros_guard_baseline_runtime.tar.gz",
    "aphros_guard_baseline_source.tar.gz",
    "aphros_guard_guarded_runtime.tar.gz",
    "aphros_guard_patched_source.tar.gz",
    "aphros_sim_base.conf",
    "build_manifest.json",
    "freeze_aphros_guard_runtime.py",
    "freeze_aphros_guard_source.py",
    "guard_build__ninja_log",
    "guard_build_build_ninja",
    "guard_build_CMakeCache_txt",
    "guard_build_install_manifest_txt",
    "independent_kernel_regression_audit.json",
    "independent_source_diff_audit.json",
    "independent_source_manifest_audit.json",
    "patch_aphros_zero_plic_flux.py",
    "runtime_config_dependency_audit.json",
    "runtime_config_dependency_hashes.json",
    "source_build_match_report.json",
    "summarize_guard_build_evidence.py",
    "zero_flux_guard.patch",
    "zero_flux_guard_regression.cpp",
    "zero_flux_reconst_repro.cpp",
    "zero_flux_regression_guarded.txt",
    "zero_flux_regression_old.txt",
)

# The frozen Python source manifest must have exactly these paths. A change in the
# executed dependency set calls for an explicit packager revision, not a wildcard.
PYTHON_FILES = (
    "research/bie_interior_evaluation.py",
    "research/gpu/extend_gas_streamfunction.py",
    "research/gpu_port/campaign_integrity.py",
    "research/gpu_port/rebuild_bie_velocity_source.py",
    "research/gpu_port/prepare_aphros_initial_state.py",
    "research/gpu_port/cutcell_transfer.py",
    "research/gpu_port/physical_trace.py",
    "research/gpu_port/liquid_dirichlet_transfer.py",
    "research/gpu_port/prepare_flat_dynamic_benchmark.py",
    "research/gpu_port/prepare_twofluid_physical_pilot.py",
    "research/gpu_port/linear_twofluid_wave_reference.py",
    "research/gpu_port/aphros_checkpoint_supervisor.py",
    "research/gpu_port/prepare_auditor_runtime_campaign.py",
    "research/gpu_port/prepare_auditor_twofluid_grid.py",
    "research/gpu_port/remote_auditor_runtime_campaign.py",
    "research/gpu_port/remote_auditor_physical.py",
    "research/gpu_port/remote_flat_dynamic.py",
    "research/gpu_port/benchmark_auditor_transfer_cost.py",
    "research/gpu_port/audit_auditor_joint_refinement.py",
    "research/gpu_port/audit_auditor_runtime_provenance.py",
    "research/gpu_port/audit_auditor_dynamic_outputs.py",
    "research/gpu_port/audit_auditor_physical_outputs.py",
    "research/gpu_port/audit_auditor_physical_metadata.py",
    "research/results/transfer_auditor_revision_20261002/dynamic/source/prepare_flat_dynamic_benchmark.py",
    "research/results/transfer_auditor_revision_20261002/physical/prepare_auditor_twofluid_grid_executed.py",
    "research/gpu_port/audit_auditor_runtime_config.py",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty required file: {path}")


def output_archives() -> tuple[str, ...]:
    dynamic = tuple(f"{tag}_{branch}_outputs.tar.gz"
                    for tag in DYNAMIC_GRIDS for branch in BRANCHES)
    physical = tuple(f"physical_{tag}_{branch}_outputs.tar.gz"
                     for tag in PHYSICAL_GRIDS for branch in BRANCHES)
    return dynamic + physical


def verify_frozen_sources(root: Path) -> None:
    bundle = root / "python_source_bundle"
    manifest = read_json(bundle / "MANIFEST.json")
    if manifest.get("schema") != "auditor-python-source-bundle-v1":
        raise ValueError("unknown Python source bundle schema")
    entries = manifest["files"]
    if set(item["path"] for item in entries) != set(PYTHON_FILES) or len(entries) != len(PYTHON_FILES):
        raise ValueError("frozen Python source allowlist changed")
    for item in entries:
        path = bundle / item["path"]
        require_file(path)
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError(f"frozen source hash mismatch: {item['path']}")


def verify_final_audit(root: Path, name: str, schema: str,
                       grids: tuple[str, ...], archive_prefix: str) -> None:
    audit = read_json(root / name)
    if audit.get("schema") != schema or set(audit.get("grids", {})) != set(grids):
        raise ValueError(f"incomplete/unknown audit {name}")
    for tag in grids:
        row = audit["grids"][tag]
        if set(row.get("runs", {})) != set(BRANCHES) or len(row.get("series", [])) < 3:
            raise ValueError(f"missing run or time series in {name}:{tag}")
        for branch in BRANCHES:
            run = row["runs"][branch]
            archive = root / f"{archive_prefix}{tag}_{branch}_outputs.tar.gz"
            require_file(archive)
            if run.get("status") != "complete" or run.get("archive_sha256") != sha(archive):
                raise ValueError(f"incomplete run/hash mismatch: {archive.name}")


def preflight(root: Path, input_review: Path) -> None:
    for relative in ROOT_FILES + output_archives():
        require_file(root / relative)
    for relative in RUNTIME_FILES:
        require_file(root / "runtime_provenance" / relative)
    for relative in ("check_inputs.py", "input_review.json"):
        require_file(input_review / relative)
    verify_frozen_sources(root)
    verify_final_audit(root, "dynamic_three_grid_audit.json",
                       "auditor-single-runtime-dynamic-raw-audit-v1", DYNAMIC_GRIDS, "")
    verify_final_audit(root, "physical_two_grid_audit.json",
                       "auditor-single-runtime-physical-raw-audit-v1", PHYSICAL_GRIDS, "physical_")
    joint = read_json(root / "joint_refinement_initial_audit.json")
    if joint.get("schema") != "auditor-fixed-ratio-three-level-initial-v1" or len(joint.get("rows", [])) != 3:
        raise ValueError("incomplete joint initial refinement audit")
    for tag in DYNAMIC_GRIDS:
        folder = root / f"joint_{tag}"
        for relative in joint_members(tag):
            require_file(root / relative)
    if read_json(root / "physical_initial_metadata_erratum.json").get("schema") != "auditor-physical-initial-momentum-erratum-v1":
        raise ValueError("metadata erratum missing or unknown")


def joint_members(tag: str) -> tuple[str, ...]:
    prefix = f"joint_{tag}/"
    files = [prefix + name for name in ("protocol.json", "preparation_report.json", "harmonic_source.npz")]
    for branch in ("exact_initial_means", "fitted", "bilinear_centroid"):
        for component in (("q", "vx", "vy") if branch == "exact_initial_means" else ("vx", "vy")):
            files.append(prefix + f"{branch}/{component}.raw")
    return tuple(files)


def make_joint_archive(root: Path, output: Path) -> None:
    with output.open("wb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for tag in DYNAMIC_GRIDS:
                for relative in joint_members(tag):
                    source = root / relative
                    info = tar.gettarinfo(str(source), arcname=relative)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with source.open("rb") as field:
                        tar.addfile(info, field)


def copy_one(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def write_manifest(stage: Path) -> None:
    records = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.json":
            records.append({"path": path.relative_to(stage).as_posix(),
                            "size": path.stat().st_size, "sha256": sha(path)})
    (stage / "SHA256SUMS.json").write_text(json.dumps({
        "schema": "auditor-revision-release-sha256-v1",
        "release": "transfer_20261002",
        "doi": None,
        "files": records,
    }, indent=2) + "\n", encoding="utf-8")


def package(root: Path, destination: Path, input_review: Path) -> None:
    preflight(root, input_review)
    if destination.exists():
        raise FileExistsError(f"release already exists; refuse to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".transfer_20261002_staging_", dir=destination.parent) as temp:
        stage = Path(temp)
        for relative in ROOT_FILES + output_archives():
            copy_one(root / relative, stage / relative)
        for relative in RUNTIME_FILES:
            copy_one(root / "runtime_provenance" / relative,
                     stage / "runtime_provenance" / relative)
        bundle = root / "python_source_bundle"
        copy_one(bundle / "MANIFEST.json", stage / "code/MANIFEST.json")
        for relative in PYTHON_FILES:
            copy_one(bundle / relative, stage / "code" / relative)
        copy_one(Path(__file__), stage / "code/package_auditor_revision.py")
        for relative in ("check_inputs.py", "input_review.json"):
            copy_one(input_review / relative, stage / "review" / relative)
        make_joint_archive(root, stage / "joint_refinement_inputs.tar.gz")
        shutil.copy2(Path(__file__).with_name("auditor_revision_release_README.md"), stage / "README.md")
        shutil.copy2(Path(__file__).with_name("verify_auditor_revision_release.py"), stage / "verify_release.py")
        write_manifest(stage)
        subprocess.run([sys.executable, str(stage / "verify_release.py")], check=True)
        stage.rename(destination)
    print(json.dumps({"release": str(destination), "manifest_sha256": sha(destination / "SHA256SUMS.json")}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=RESULTS)
    parser.add_argument("--release-dir", type=Path, default=RELEASE)
    parser.add_argument("--input-review-dir", type=Path,
                        default=REPOSITORY / "tmp/auditor_revision/input_review")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    preflight(args.source_root, args.input_review_dir)
    if args.check_only:
        print("all final experiment files and audits are complete")
    else:
        package(args.source_root, args.release_dir, args.input_review_dir)


if __name__ == "__main__":
    main()
