"""Freeze the original physical pilot under the separately built zero-flux guard."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    home = Path(__file__).resolve().parents[1] / "results/publication_package/transfer_physical_pilot_20261001"
    source = home / "flat_native_L9_twofluid"
    original_path = source / "native_launch_protocol.json"
    original = json.loads(original_path.read_text())
    prefix = "/opt/gpu-cfd/prefix_zero_flux_guard_20261001"
    runtime = {f"{prefix}/bin/ap.mfer": "8d89145282ab58164d8f4acfcb761b126fc48607100797ab8b980e6152578154",
               f"{prefix}/lib/libaphros.so": "039091fe16b554635eea7507eda87de4216289a9ec20c9abb8ba06a415dd374f"}
    branches = ("zero", "A_phase", "Ahalf_phase", "A_mass", "Ahalf_mass")
    commands = {}
    for branch in branches:
        cmd = list(original["commands"][branch])
        cmd[cmd.index("--run-root") + 1] = (
            f"/opt/gpu-cfd/twofluid_physical_pilot_20261001/guarded_full/{branch}")
        cmd[cmd.index("--solver") + 1] = f"{prefix}/bin/ap.mfer"
        cmd[cmd.index("--solver-library") + 1] = f"{prefix}/lib/libaphros.so"
        commands[branch] = cmd
    report = {
        "schema": "twofluid-guarded-physical-pilot-v1",
        "reason": "original pilot zero branch failed due exact-zero 2D PLIC NaN; isolated same-source guard removes only this undefined zero displacement",
        "original_protocol_sha256": sha(original_path),
        "input_archive_sha256": sha(home / "flat_native_L9_twofluid_inputs.tar.gz"),
        "base_config_sha256": sha(source / "twofluid_slip_inviscid_dt00125.conf"),
        "analytic_reference_source_sha256": sha(Path(__file__).resolve().parent / "linear_twofluid_wave_reference.py"),
        "supervisor_sha256": sha(source / "benchmark_supervisor.py"),
        "expected_runtime_sha256": runtime,
        "run_order": branches,
        "commands": commands,
        "primary_observable": "cell-averaged column depth vs fixed linear two-fluid analytic H, normalized by B*sinc(kh/2)/sqrt(2); phase and half-amplitude controls first",
        "zero_floor_rule": "zero must pass unchanged mass/flux/finite/speed gates to full horizon; do not subtract zero field from primary wave error",
        "interpretation": "analytical reference is linear, not an exact nonlinear receiver trajectory; amplitude halving assesses nonlinearity",
    }
    output = home / "guarded_full_launch_protocol.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"path": str(output), "sha256": sha(output), "branches": branches}, indent=2))


if __name__ == "__main__":
    main()
