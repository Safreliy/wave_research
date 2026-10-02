"""Freeze a same-runtime flat-wave transfer comparison before native runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np


COMMON = ("q.raw", "vf.raw", "cs.raw", "body.dat", "fluxx.raw", "fluxxp.raw",
          "fluxy.raw", "fluxyp.raw", "fluxeb.raw")
RUNTIME = {
    "/opt/gpu-cfd/prefix_zero_flux_guard_20261001/bin/ap.mfer":
        "8d89145282ab58164d8f4acfcb761b126fc48607100797ab8b980e6152578154",
    "/opt/gpu-cfd/prefix_zero_flux_guard_20261001/lib/libaphros.so":
        "039091fe16b554635eea7507eda87de4216289a9ec20c9abb8ba06a415dd374f",
}
REMOTE_ROOT = "/opt/gpu-cfd/transfer_auditor_revision_20261002/dynamic"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_arg(command: list[str], key: str, value: str) -> None:
    command[command.index(key) + 1] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    if root.exists():
        raise FileExistsError(root)
    research = Path(__file__).resolve().parents[1]
    old = research / "results/publication_package"
    dynamic = old / "transfer_dynamic_followup_20261001"
    matched = old / "transfer_matched_native_20261001"
    new_matched = root.parent / "flat_native_L11_matched"
    source_files = (
        "prepare_flat_dynamic_benchmark.py", "liquid_dirichlet_transfer.py",
        "cutcell_transfer.py", "physical_trace.py", "prepare_aphros_initial_state.py",
        "aphros_checkpoint_supervisor.py", "prepare_auditor_runtime_campaign.py",
    )
    # Validate scientific source before copying any input: a changed operator
    # requires a fresh, explicit protocol rather than silently mixing versions.
    manifest = json.loads((research / "results/method_trace_hermite_20261001/"
                           "matched_source_snapshot/SNAPSHOT_MANIFEST.json").read_text())
    for entry in manifest["files"]:
        if entry["path"] in {"research/gpu_port/liquid_dirichlet_transfer.py",
                             "research/gpu_port/cutcell_transfer.py",
                             "research/gpu_port/physical_trace.py"}:
            if sha(research.parent / entry["path"]) != entry["sha256"]:
                raise ValueError(f"scientific source changed: {entry['path']}")
    root.mkdir(parents=True)
    (root / "source").mkdir()
    for name in source_files:
        shutil.copy2(research / "gpu_port" / name, root / "source" / name)
    result = {"schema": "auditor-single-runtime-dynamic-v1",
              "predeclared_protocol_sha256": sha(root.parent / "PREDECLARED_PROTOCOL.md"),
              "remote_root": REMOTE_ROOT, "expected_runtime_sha256": RUNTIME,
              "source_sha256": {n: sha(root / "source" / n) for n in source_files},
              "grids": {}, "run_order": [], "commands": {}}
    for nx, old_name in ((512, "flat_native_L9"), (1024, "flat_native_L10"),
                         (2048, "flat_native_L11_resolved")):
        tag = f"L{nx.bit_length()-1}"
        old_root = dynamic / old_name
        old_launch = json.loads((old_root / "native_launch_protocol.json").read_text())
        source_map = {
            "exact": old_root / "exact_initial_means",
            "bilinear": old_root / "bilinear_centroid",
            "matched": (matched / f"flat_native_{tag}_matched" / "fitted"
                        if nx < 2048 else new_matched / "fitted"),
        }
        grid = root / tag
        grid.mkdir()
        config_name = Path(old_launch["commands"]["full_fitted"][
            old_launch["commands"]["full_fitted"].index("--base-config") + 1]).name
        for name in (config_name, "benchmark_supervisor.py"):
            shutil.copy2(old_root / name, grid / name)
        grid_report = {"nx": nx, "ny": nx // 16,
                       "source_protocol_sha256": sha(old_root / "native_launch_protocol.json"),
                       "config_sha256": sha(grid / config_name),
                       "supervisor_sha256": sha(grid / "benchmark_supervisor.py"),
                       "branches": {}}
        for label, source in source_map.items():
            branch = f"{tag}_{label}"
            dest = grid / label
            shutil.copytree(source, dest)
            grid_report["branches"][label] = {
                "source": str(source),
                "input_sha256": {p.name: sha(p) for p in sorted(dest.iterdir()) if p.is_file()},
            }
            cmd = list(old_launch["commands"]["full_fitted"])
            cmd[1] = f"{REMOTE_ROOT}/{tag}/benchmark_supervisor.py"
            set_arg(cmd, "--run-root", f"{REMOTE_ROOT}/{tag}/runs/{label}")
            set_arg(cmd, "--base-config", f"{REMOTE_ROOT}/{tag}/{config_name}")
            set_arg(cmd, "--input-dir", f"{REMOTE_ROOT}/{tag}/{label}")
            set_arg(cmd, "--solver", next(p for p in RUNTIME if p.endswith("/ap.mfer")))
            set_arg(cmd, "--solver-library", next(p for p in RUNTIME if p.endswith("/libaphros.so")))
            result["run_order"].append(branch)
            result["commands"][branch] = cmd
        # Every comparison holds the receiver geometry and face input fixed.
        common = grid_report["branches"]["exact"]["input_sha256"]
        for label in ("bilinear", "matched"):
            if any(grid_report["branches"][label]["input_sha256"][n] != common[n]
                   for n in COMMON):
                raise ValueError(f"{tag}: common inputs differ for {label}")
            q = np.fromfile(grid / "exact/q.raw", dtype="<f8").reshape(nx // 16, nx)
            dry = q <= 0
            for component in ("vx", "vy"):
                reference = np.fromfile(grid / "exact" / f"{component}.raw", dtype="<f8")
                candidate = np.fromfile(grid / label / f"{component}.raw", dtype="<f8")
                if not np.array_equal(reference.reshape(q.shape)[dry],
                                      candidate.reshape(q.shape)[dry]):
                    raise ValueError(f"{tag}: gas changed in {label}")
        result["grids"][tag] = grid_report
    (root / "campaign.json").write_text(json.dumps(result, indent=2) + "\n")
    archive = root.parent / "dynamic_inputs.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(root.parent).as_posix())
    print(json.dumps({"root": str(root), "campaign_sha256": sha(root / "campaign.json"),
                      "archive_sha256": sha(archive), "archive_bytes": archive.stat().st_size},
                     indent=2))


if __name__ == "__main__":
    main()
