"""Verify included Aphros and AMGX config bytes for every frozen branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    provenance = root / "runtime_provenance"
    live = json.loads((provenance / "runtime_config_dependency_hashes.json").read_text())
    base_path = "/opt/gpu-cfd/aphros/deploy/scripts/sim_base.conf"
    amgx_path = "/opt/gpu-cfd/prefix/lib/configs/PCG_AGGREGATION_JACOBI.json"
    if sha(provenance / "aphros_sim_base.conf") != live[base_path]:
        raise ValueError("base config live copy hash mismatch")
    if sha(provenance / "AMGX_PCG_AGGREGATION_JACOBI.json") != live[amgx_path]:
        raise ValueError("AMGX config live copy hash mismatch")
    with tarfile.open(provenance / "aphros_guard_patched_source.tar.gz", "r:gz") as archive:
        source_base = archive.extractfile("deploy/scripts/sim_base.conf").read()
    if hashlib.sha256(source_base).hexdigest() != live[base_path]:
        raise ValueError("live base config differs from guarded source archive")
    if re.search(r"^\s*include\s", source_base.decode(), re.MULTILINE):
        raise ValueError("transitive base include not captured")
    report = {"schema": "auditor-runtime-config-dependency-v1",
              "aphros_base_include": base_path,
              "aphros_base_sha256": live[base_path],
              "aphros_base_matches_guarded_source_archive": True,
              "aphros_base_has_nested_include": False,
              "amgx_config": amgx_path,
              "amgx_config_sha256": live[amgx_path],
              "campaigns": {}}
    for category in ("dynamic", "physical"):
        plan = json.loads((root / category / "campaign.json").read_text())
        entries = {}
        for tag, details in plan["grids"].items():
            command = plan["commands"][f"{tag}_exact"]
            config_path = root / category / tag / Path(command[command.index("--base-config")+1]).name
            config = config_path.read_text()
            if re.findall(r"^\s*include\s+([^\s]+)", config, re.MULTILINE) != [base_path]:
                raise ValueError(f"unexpected includes: {category}/{tag}")
            if amgx_path not in config:
                raise ValueError(f"wrong AMGX config: {category}/{tag}")
            if sha(config_path) != details["config_sha256"]:
                raise ValueError(f"frozen config changed: {category}/{tag}")
            for label in ("exact", "matched", "bilinear"):
                branch = plan["commands"][f"{tag}_{label}"]
                if branch[branch.index("--base-config")+1] != command[command.index("--base-config")+1]:
                    raise ValueError(f"branch config differs: {category}/{tag}/{label}")
                for flag, suffix in (("--solver", "/ap.mfer"),
                                     ("--solver-library", "/libaphros.so")):
                    runtime = branch[branch.index(flag)+1]
                    if not runtime.endswith(suffix) or runtime not in plan["expected_runtime_sha256"]:
                        raise ValueError(f"runtime differs: {category}/{tag}/{label}")
            entries[tag] = {"config_sha256": details["config_sha256"],
                            "base_include_sha256": live[base_path],
                            "amgx_config_sha256": live[amgx_path]}
        report["campaigns"][category] = entries
    target = provenance / "runtime_config_dependency_audit.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
