"""Stage and run the flat harmonic benchmark on an SSH-accessible WSL host.

The password is read from FLAT_BENCHMARK_SSH_PASSWORD and is never saved in an
artifact. Each solver call stays in the foreground over SSH so WSL lifetime is
not tied to a detached Windows launcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import time
from pathlib import Path

import paramiko


REMOTE_CAMPAIGN = "/opt/gpu-cfd/flat_harmonic_dynamic_20261001"
BASE_CONFIG_L10 = "/opt/gpu-cfd/refined_full_v45/configs/full_l10_cfl010.conf"
BASE_CONFIG_L11 = "/opt/gpu-cfd/refined_full_v45/configs/full_l11_cfl010.conf"
PYTHON = "/mnt/d/wave_simulation_gpu/env/cupy/bin/python"
SUPERVISOR_NAME = "benchmark_supervisor.py"
SOLVER = "/opt/gpu-cfd/prefix/bin/ap.mfer"
LIBRARY = "/opt/gpu-cfd/prefix/lib/libaphros.so"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def connect(host: str, user: str) -> paramiko.SSHClient:
    password = os.environ.get("FLAT_BENCHMARK_SSH_PASSWORD")
    if not password:
        raise ValueError("FLAT_BENCHMARK_SSH_PASSWORD is required")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=15)
    return client


def command(client: paramiko.SSHClient, arguments: list[str], *, stream: bool = False) -> str:
    # Paths and numeric parameters are predetermined or validated below; do
    # not pass arbitrary shell fragments through Windows SSH command parsing.
    for arg in arguments:
        if not re.fullmatch(r"[A-Za-z0-9_./=+:-]+", arg):
            raise ValueError(f"unsafe remote argument: {arg!r}")
    remote = "wsl.exe -- " + " ".join(arguments)
    _, stdout, stderr = client.exec_command(remote, get_pty=stream)
    output = []
    for line in stdout:
        if stream:
            print(line.rstrip(), flush=True)
        output.append(line)
    status = stdout.channel.recv_exit_status()
    error = stderr.read().decode(errors="replace")
    if status:
        raise RuntimeError(f"remote exit {status}: {remote}\n{''.join(output)[-3000:]}\n{error[-1000:]}")
    return "".join(output)


def stage(client: paramiko.SSHClient, root: Path) -> None:
    if root.name not in ("flat_native_L9", "flat_native_L10",
                         "flat_native_L9_trace4", "flat_native_L10_trace4",
                         "flat_native_L11_resolved"):
        raise ValueError("stage currently supports frozen L9/L10/L11 benchmark inputs")
    if (root / "native_launch_protocol.json").exists():
        raise FileExistsError("launch protocol already frozen")
    base_config = BASE_CONFIG_L11 if root.name == "flat_native_L11_resolved" else BASE_CONFIG_L10
    base_text = command(client, ["cat", base_config])
    if base_text.count("set double dt0 0.00125") != 1:
        raise ValueError("unexpected source config dt0")
    config = base_text.replace("set double dt0 0.00125", "set double dt0 0.0025")
    config = config.replace("set double dtmax 0.00125", "set double dtmax 0.0025")
    if root.name.startswith("flat_native_L9"):
        for old, new in (("set int bsx 16", "set int bsx 8"),
                         ("set int bsy 8", "set int bsy 4")):
            if config.count(old) != 1:
                raise ValueError(f"unexpected L10 block layout: {old}")
            config = config.replace(old, new)
    config_path = root / "flat_dt0025.conf"
    config_path.write_text(config, encoding="utf-8", newline="\n")
    supervisor_path = root / SUPERVISOR_NAME
    shutil.copy2(Path(__file__).with_name("aphros_checkpoint_supervisor.py"), supervisor_path)
    prep = json.loads((root / "protocol.json").read_text())
    expected_nx = (512 if root.name.startswith("flat_native_L9") else
                   2048 if root.name == "flat_native_L11_resolved" else 1024)
    if prep["geometry"]["nx"] != expected_nx:
        raise ValueError("folder name and grid disagree")
    tmax = prep["target_time_one_tenth_period"]
    remote_root = f"{REMOTE_CAMPAIGN}/{root.name}"
    supervisor = f"{remote_root}/{SUPERVISOR_NAME}"
    run_commands = {}
    for step_tag, dt in (("full", 0.0025), ("half", 0.00125)):
        for branch in prep["methods"]:
            run_commands[f"{step_tag}_{branch}"] = [
                PYTHON, supervisor,
                "--run-root", f"{remote_root}/runs/{step_tag}/{branch}",
                "--base-config", f"{remote_root}/flat_dt0025.conf",
                "--input-dir", f"{remote_root}/{branch}",
                "--solver", SOLVER, "--solver-library", LIBRARY,
                "--target-time", str(tmax),
                "--dump-field-dt", str(tmax/4),
                "--checkpoint-seconds", "300", "--poll-seconds", "1",
                "--threads", "12", "--max-stalled-restarts", "2",
                "--initial-mass-relative-tolerance", "1e-8",
                "--initial-momentum-absolute-tolerance", "1e-6",
                "--mass-relative-tolerance", "1e-3",
                "--vof-mass-relative-tolerance", "5e-4",
                "--flux-divergence-tolerance", "1e-9",
                "--shared-face-tolerance", "1e-10",
                "--max-liquid-speed", "10", "--max-all-speed", "50",
                "--initial-velocity-init", "raw",
                "--initial-flux-init", "reconstruct",
                "--restart-pressure-init", "raw",
                "--restart-flux-init", "raw",
                "--cfl", str(0.1 if step_tag == "full" else 0.05),
                "--cfla", str(0.1 if step_tag == "full" else 0.05),
                "--dtmax", str(dt), "--minimum-dt", "1e-12",
                "--abort-speed", "50",
            ]
    # A separate half-step base config changes dt0, matching the prior
    # campaign's four-parameter halving rather than only changing dtmax.
    half_config = config.replace("set double dt0 0.0025", "set double dt0 0.00125")
    half_path = root / "flat_dt00125.conf"
    half_path.write_text(half_config, encoding="utf-8", newline="\n")
    for key, values in run_commands.items():
        if key.startswith("half_"):
            values[values.index("--base-config") + 1] = f"{remote_root}/flat_dt00125.conf"
    runtime_hashes = {
        SOLVER: "9cd89c6fb6554db133aedc02d66ff5f6654ae91a727c24677e6edbaadd16c4d5",
        LIBRARY: "2a2edcdb1e6e897aff006577c5a8cf951ff21cb5f7f6c1ccb6050c241e337a09",
    }
    for path, expected in runtime_hashes.items():
        actual = command(client, ["sha256sum", path]).split()[0]
        if actual != expected:
            raise ValueError(f"remote solver dependency hash differs: {path}")
    protocol = {
        "schema": "flat-harmonic-native-launch-v1",
        "input_protocol_sha256": sha256(root / "protocol.json"),
        "preparation_report_sha256": sha256(root / "preparation_report.json"),
        "benchmark_supervisor_sha256": sha256(supervisor_path),
        "full_config_sha256": sha256(config_path),
        "half_config_sha256": sha256(half_path),
        "source_base_config": base_config,
        "expected_runtime_sha256": runtime_hashes,
        "native_physics": {
            "gas_density": 0.001176470588235294, "liquid_density": 1.0,
            "gas_dynamic_viscosity": 4.887640449438202e-7,
            "liquid_dynamic_viscosity": 2.5e-5,
            "surface_tension": 0.001, "gravity": [0.0, -1.0],
            "projection": "native Aphros/AMGX", "x_boundary": "periodic",
            "bottom_boundary": "embedded no-penetration wall",
            "top_boundary": "pressure outlet at solver y=4",
        },
        "commands": run_commands,
        "comparison": "full/half each compared to exact_initial_means at identical t; same dt within each trio",
        "reused_reference_native_root": prep.get("reused_reference_native_root"),
        "decision_rule": "fitted dynamic gain requires lower error than same-psi bilinear-centroid in primary global q-weighted velocity norm and check of VOF/depth; no selective region or parameter tuning after seeing outcomes",
        "execution_note": "foreground SSH-Windows-WSL call keeps WSL process alive",
        "code_sha256": sha256(Path(__file__)),
    }
    launch_path = root / "native_launch_protocol.json"
    launch_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    tar_path = root.parent / f"{root.name}_inputs.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and "runs" not in path.parts:
                archive.add(path, arcname=path.relative_to(root.parent).as_posix())
    sftp = client.open_sftp()
    windows_archive = f"C:/Users/Administrator/{root.name}_inputs.tar.gz"
    sftp.put(str(tar_path), windows_archive)
    sftp.close()
    command(client, ["mkdir", "-p", REMOTE_CAMPAIGN])
    command(client, ["tar", "-xzf", f"/mnt/c/Users/Administrator/{root.name}_inputs.tar.gz", "-C", REMOTE_CAMPAIGN])
    print(json.dumps({"launch_protocol_sha256": sha256(launch_path),
                      "archive_sha256": sha256(tar_path),
                      "remote_root": remote_root}, indent=2), flush=True)


def run(client: paramiko.SSHClient, root: Path, mode: str, selected_branch: str | None = None) -> None:
    protocol = json.loads((root / "native_launch_protocol.json").read_text())
    branches = ["fitted", "bilinear_centroid"]
    if not protocol.get("reused_reference_native_root"):
        branches.append("exact_initial_means")
    if selected_branch is not None:
        if selected_branch not in branches:
            raise ValueError(f"branch {selected_branch!r} unavailable for this staged run")
        branches = [selected_branch]
    for branch in branches:
        key = f"{mode}_{branch}"
        print(f"BEGIN {key} {time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}", flush=True)
        command(client, protocol["commands"][key], stream=True)
        print(f"END {key} {time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}", flush=True)


def analyze(client: paramiko.SSHClient, root: Path) -> None:
    remote_root = f"{REMOTE_CAMPAIGN}/{root.name}"
    local_source = Path(__file__).with_name("analyze_flat_dynamic.py")
    windows_source = f"C:/Users/Administrator/{root.name}_analyze_flat_dynamic.py"
    windows_output = f"C:/Users/Administrator/{root.name}_analysis_full_series.json"
    sftp = client.open_sftp()
    try:
        sftp.put(str(local_source), windows_source)
    finally:
        sftp.close()
    command(client, ["cp", f"/mnt/c/Users/Administrator/{root.name}_analyze_flat_dynamic.py",
                     f"{remote_root}/analyze_flat_dynamic.py"])
    command(client, ["python3", f"{remote_root}/analyze_flat_dynamic.py",
                     "--root", remote_root, "--output", f"{remote_root}/analysis_full_series.json"])
    command(client, ["cp", f"{remote_root}/analysis_full_series.json",
                     f"/mnt/c/Users/Administrator/{root.name}_analysis_full_series.json"])
    sftp = client.open_sftp()
    try:
        sftp.get(windows_output, str(root / "analysis_full_series.json"))
    finally:
        sftp.close()
    print(json.dumps({"analysis_sha256": sha256(root / "analysis_full_series.json"),
                      "path": str(root / "analysis_full_series.json")}, indent=2), flush=True)


def archive(client: paramiko.SSHClient, output: Path) -> None:
    local_source = Path(__file__).with_name("archive_flat_native_outputs.py")
    windows_source = "C:/Users/Administrator/archive_flat_native_outputs.py"
    windows_output = "C:/Users/Administrator/flat_native_selected_outputs.tar.gz"
    remote_source = f"{REMOTE_CAMPAIGN}/archive_flat_native_outputs.py"
    remote_output = f"{REMOTE_CAMPAIGN}/flat_native_selected_outputs.tar.gz"
    sftp = client.open_sftp()
    try:
        sftp.put(str(local_source), windows_source)
        for label in ("flat_native_L9", "flat_native_L10",
                      "flat_native_L9_trace4", "flat_native_L10_trace4",
                      "flat_native_L11_resolved"):
            local_root = output.parent / label
            launch = json.loads((local_root / "native_launch_protocol.json").read_text())
            names = [Path(launch["commands"][f"{phase}_fitted"][
                launch["commands"][f"{phase}_fitted"].index("--base-config") + 1]).name
                for phase in ("full", "half")]
            if label != "flat_native_L11_resolved":
                names.append("input_reconstruction_audit.json")
            if label.endswith("trace4"):
                names.append("trace_diagnosis.json")
            for name in names:
                local = local_root / name
                if not local.is_file():
                    raise FileNotFoundError(local)
                if name == names[0] and sha256(local) != launch["full_config_sha256"]:
                    raise ValueError(f"frozen full config hash mismatch: {label}")
                if name == names[1] and sha256(local) != launch["half_config_sha256"]:
                    raise ValueError(f"frozen half config hash mismatch: {label}")
                sftp.put(str(local), f"C:/Users/Administrator/{label}_{name}")
    finally:
        sftp.close()
    command(client, ["cp", "/mnt/c/Users/Administrator/archive_flat_native_outputs.py", remote_source])
    for label in ("flat_native_L9", "flat_native_L10",
                  "flat_native_L9_trace4", "flat_native_L10_trace4",
                  "flat_native_L11_resolved"):
        launch = json.loads((output.parent / label / "native_launch_protocol.json").read_text())
        names = [Path(launch["commands"][f"{phase}_fitted"][
            launch["commands"][f"{phase}_fitted"].index("--base-config") + 1]).name
            for phase in ("full", "half")]
        if label != "flat_native_L11_resolved":
            names.append("input_reconstruction_audit.json")
        if label.endswith("trace4"):
            names.append("trace_diagnosis.json")
        for name in names:
            command(client, ["cp", f"/mnt/c/Users/Administrator/{label}_{name}",
                             f"{REMOTE_CAMPAIGN}/{label}/{name}"])
    command(client, ["python3", remote_source, "--campaign-root", REMOTE_CAMPAIGN,
                     "--output", remote_output])
    command(client, ["cp", remote_output, "/mnt/c/Users/Administrator/flat_native_selected_outputs.tar.gz"])
    output.parent.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    try:
        sftp.get(windows_output, str(output))
    finally:
        sftp.close()
    print(json.dumps({"archive_sha256": sha256(output), "path": str(output),
                      "bytes": output.stat().st_size}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stage", "run-full", "run-half", "analyze", "archive"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", default="192.168.1.28")
    parser.add_argument("--user", default="Administrator")
    parser.add_argument("--branch", choices=("fitted", "bilinear_centroid", "exact_initial_means"),
                        help="Run one staged branch (useful for a measured pilot).")
    args = parser.parse_args()
    client = connect(args.host, args.user)
    try:
        if args.action == "stage":
            stage(client, args.root)
        elif args.action == "analyze":
            analyze(client, args.root)
        elif args.action == "archive":
            archive(client, args.root)
        else:
            run(client, args.root, args.action.split("-")[1], args.branch)
    finally:
        client.close()


if __name__ == "__main__":
    main()
