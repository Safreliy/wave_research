"""Run the frozen independent seven-way physical audit from curated archives."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True, help="Output JSON path outside this immutable release")
    args = parser.parse_args()
    report = args.report.resolve()
    if report.is_relative_to(ROOT.resolve()):
        parser.error("report must be outside the release directory")
    with tempfile.TemporaryDirectory(prefix="transfer_physical_audit_") as scratch_name:
        scratch = Path(scratch_name)
        script = scratch / "audit_guarded_mass_factorial_independent.py"
        shutil.copyfile(ROOT / "code/independent_audits/audit_guarded_mass_factorial_independent.py", script)
        for name in (
            "flat_native_L9_twofluid_inputs.tar.gz",
            "flat_native_L9_guarded_transfer_inputs.tar.gz",
            "flat_native_L9_guarded_mass_factorial_inputs.tar.gz",
        ):
            shutil.copyfile(ROOT / "archives" / name, scratch / name)
        with tarfile.open(scratch / "flat_native_L9_twofluid_inputs.tar.gz", "r:gz") as archive:
            data = archive.extractfile("flat_native_L9_twofluid/protocol.json").read()
        protocol = scratch / "flat_native_L9_twofluid/protocol.json"
        protocol.parent.mkdir(parents=True)
        protocol.write_bytes(data)
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--outputs", str(ROOT / "archives/twofluid_guarded_mass_factorial_complete_outputs.tar.gz"),
                "--analysis", str(ROOT / "results/physical/guarded_mass_factorial_complete_height_analysis.json"),
                "--report", str(report),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
