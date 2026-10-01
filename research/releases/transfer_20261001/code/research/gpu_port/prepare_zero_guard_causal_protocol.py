"""Freeze a one-step zero-state run for an isolated Aphros build."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "guarded"), required=True)
    parser.add_argument("--runtime-prefix", required=True)
    parser.add_argument("--solver-sha256", required=True)
    parser.add_argument("--library-sha256", required=True)
    args = parser.parse_args()
    home = Path(__file__).resolve().parents[1] / "results/publication_package/transfer_physical_pilot_20261001"
    physical = home / "flat_native_L9_twofluid"
    old = json.loads((physical / "native_launch_protocol.json").read_text())
    command = list(old["commands"]["zero"])
    prefix = args.runtime_prefix.rstrip("/")
    remote_run = f"/opt/gpu-cfd/twofluid_physical_pilot_20261001/guard_causal_runs/{args.variant}_zero"

    def setopt(name: str, value: str) -> None:
        command[command.index(name) + 1] = value

    setopt("--run-root", remote_run)
    setopt("--solver", f"{prefix}/bin/ap.mfer")
    setopt("--solver-library", f"{prefix}/lib/libaphros.so")
    setopt("--target-time", "0.00125")
    setopt("--dump-field-dt", "0.00125")
    report = {
        "schema": "aphros-zero-plic-guard-causal-run-v1",
        "variant": args.variant,
        "claim_scope": "one-step same-source baseline/guarded ablation on byte-identical physical zero input and config; not a validation of analytic wave dynamics",
        "original_launch_protocol_sha256": sha(physical / "native_launch_protocol.json"),
        "input_archive_sha256": sha(home / "flat_native_L9_twofluid_inputs.tar.gz"),
        "input_file_sha256": {p.name: sha(p) for p in (physical / "zero").iterdir()},
        "base_config_sha256": sha(physical / "twofluid_slip_inviscid_dt00125.conf"),
        "supervisor_sha256": sha(physical / "benchmark_supervisor.py"),
        "runtime_sha256": {"solver": args.solver_sha256, "library": args.library_sha256},
        "runtime_prefix": prefix,
        "command": command,
        "declared_gate": "unchanged physical pilot semantic tolerances; first dump at one step",
    }
    output = home / f"zero_guard_{args.variant}_protocol.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"path": str(output), "sha256": sha(output), "run_root": remote_run}, indent=2))


if __name__ == "__main__":
    main()
