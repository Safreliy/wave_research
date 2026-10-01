"""Freeze native command matrix for the five-run physical wave pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


BRANCHES = ("zero", "A_phase", "A_mass", "Ahalf_phase", "Ahalf_mass")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_argument(command: list[str], name: str, value: str) -> None:
    command[command.index(name)+1] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    args = parser.parse_args()
    launch_path = args.root / "native_launch_protocol.json"
    if launch_path.exists():
        raise FileExistsError(launch_path)
    protocol = json.loads((args.root / "protocol.json").read_text())
    prior = json.loads((args.prior_root / "native_launch_protocol.json").read_text())
    remote_root = protocol["remote_root"] + f"/{args.root.name}"
    commands = {}
    for branch in BRANCHES:
        command = prior["commands"]["half_exact_initial_means"].copy()
        command[1] = f"{remote_root}/benchmark_supervisor.py"
        replace_argument(command, "--run-root", f"{remote_root}/runs/full/{branch}")
        replace_argument(command, "--base-config", f"{remote_root}/twofluid_slip_inviscid_dt00125.conf")
        replace_argument(command, "--input-dir", f"{remote_root}/{branch}")
        replace_argument(command, "--target-time", str(protocol["target_time"]))
        replace_argument(command, "--dump-field-dt", str(protocol["target_time"]/4))
        for flag in ("--cfl", "--cfla"):
            replace_argument(command, flag, "0.05")
        replace_argument(command, "--dtmax", "0.00125")
        commands[branch] = command
    launch = {
        "schema": "twofluid-small-amplitude-native-launch-v1",
        "input_protocol_sha256": sha256(args.root / "protocol.json"),
        "preparation_report_sha256": sha256(args.root / "preparation_report.json"),
        "config_sha256": sha256(args.root / "twofluid_slip_inviscid_dt00125.conf"),
        "supervisor_sha256": sha256(args.root / "benchmark_supervisor.py"),
        "prior_native_launch_sha256": sha256(args.prior_root / "native_launch_protocol.json"),
        "expected_runtime_sha256": protocol["expected_runtime_sha256"],
        "native_physics": {"rho_gas": 0.001176470588235294, "rho_liquid": 1.0,
                           "mu_gas": 0.0, "mu_liquid": 0.0,
                           "sigma": 0.001, "gravity": [0.0, -1.0],
                           "bottom_and_embed_boundary": "slipwall",
                           "top_boundary": "outletpressure 0",
                           "x_boundary": "periodic"},
        "commands": commands,
        "run_order": list(BRANCHES),
        "zero_floor_rule": "run zero first; report absolute H floor without subtraction. If floor approaches wave signal/error, do not interpret method differences as physical accuracy; no amplitude retuning.",
        "remote_root": remote_root,
        "code_sha256": sha256(Path(__file__)),
    }
    launch_path.write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
    archive_path = args.root.parent / f"{args.root.name}_inputs.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(args.root.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(args.root.parent).as_posix())
    print(json.dumps({"launch_protocol_sha256": sha256(launch_path),
                      "input_archive_sha256": sha256(archive_path),
                      "bytes": archive_path.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
