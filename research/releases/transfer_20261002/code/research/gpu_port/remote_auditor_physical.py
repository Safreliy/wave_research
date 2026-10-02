"""Stage, run, and retrieve frozen guarded two-fluid grid comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from remote_flat_dynamic import command, connect


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stage", "run", "fetch"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--branch")
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root / "physical/campaign.json").read_text())
    if args.branch is not None and args.branch not in plan["run_order"]:
        raise ValueError("branch not predeclared")
    client = connect("192.168.1.28", "Administrator")
    try:
        for path, digest in plan["expected_runtime_sha256"].items():
            if command(client, ["sha256sum", path]).split()[0] != digest:
                raise ValueError(f"runtime hash mismatch: {path}")
        remote_base = "/opt/gpu-cfd/transfer_auditor_revision_20261002"
        if args.action == "stage":
            name = "transfer_auditor_physical_inputs_20261002.tar.gz"
            local_archive = root / "physical_inputs.tar.gz"
            with client.open_sftp() as sftp:
                sftp.put(str(local_archive), "C:/Users/Administrator/" + name)
            mounted = "/mnt/c/Users/Administrator/" + name
            if command(client, ["sha256sum", mounted]).split()[0] != sha(local_archive):
                raise ValueError("archive transfer mismatch")
            command(client, ["mkdir", "-p", remote_base])
            command(client, ["tar", "-xzf", mounted, "-C", remote_base])
            if command(client, ["sha256sum", plan["remote_root"] + "/campaign.json"]).split()[0] != sha(root / "physical/campaign.json"):
                raise ValueError("remote protocol mismatch")
            print(json.dumps({"remote_root": plan["remote_root"],
                              "archive_sha256": sha(local_archive)}, indent=2))
        elif args.action == "run":
            if args.branch is None:
                raise ValueError("--branch required")
            command(client, plan["commands"][args.branch], stream=True)
        else:
            if args.branch is None:
                raise ValueError("--branch required")
            grid, label = args.branch.split("_", 1)
            source = f"{plan['remote_root']}/{grid}/runs/{label}"
            remote_archive = f"{remote_base}/physical_{args.branch}_outputs.tar.gz"
            command(client, ["tar", "-czf", remote_archive, "-C", plan["remote_root"],
                             f"{grid}/runs/{label}"])
            windows_name = f"physical_{args.branch}_auditor_outputs.tar.gz"
            command(client, ["cp", remote_archive,
                             f"/mnt/c/Users/Administrator/{windows_name}"])
            local_archive = root / f"physical_{args.branch}_outputs.tar.gz"
            with client.open_sftp() as sftp:
                sftp.get("C:/Users/Administrator/" + windows_name, str(local_archive))
            if sha(local_archive) != command(client, ["sha256sum", remote_archive]).split()[0]:
                raise ValueError("retrieved archive mismatch")
            print(json.dumps({"branch": args.branch,
                              "archive_sha256": sha(local_archive),
                              "archive_bytes": local_archive.stat().st_size}, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
