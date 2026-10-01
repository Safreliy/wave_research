"""Freeze three one-step controls for the failed flat two-fluid zero state."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
SRC = HERE / "results/publication_package/transfer_physical_pilot_20261001/flat_native_L9_twofluid"
ROOT = HERE / "results/publication_package/transfer_physical_pilot_20261001/flat_native_L9_zero_diagnostics"
REMOTE = "/opt/gpu-cfd/twofluid_physical_pilot_20261001/flat_native_L9_zero_diagnostics"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError((old, text.count(old)))
    return text.replace(old, new)


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(ROOT)
    ROOT.mkdir(parents=True)
    base = (SRC / "twofluid_slip_inviscid_dt00125.conf").read_text()
    frozen = json.loads((SRC / "native_launch_protocol.json").read_text())
    branches = {
        "zero_sigma0": {"changes": "slipwall, zero viscosities, zero surface tension, dt=.00125"},
        "zero_oldphysics": {"changes": "wall and original positive viscosities, original sigma, dt=.00125"},
        "zero_halfrow": {"changes": "slipwall, zero viscosities, original sigma, flat q=.5 upper cut row, dt=.00125"},
    }
    configs = {}
    for branch in branches:
        target = ROOT / branch
        shutil.copytree(SRC / "zero", target)
        conf = base
        if branch == "zero_oldphysics":
            conf = replace(conf, "slipwall {", "wall 0 0 0 {")
            conf = replace(conf, "set double mu1 0\n", "set double mu1 4.887640449438202e-7\n")
            conf = replace(conf, "set double mu2 0\n", "set double mu2 2.5e-5\n")
        elif branch == "zero_sigma0":
            conf = replace(conf, "set double sigma 0.001\n", "set double sigma 0\n")
        else:
            shape = (32, 512)
            q = np.fromfile(target / "q.raw", dtype="<f8").reshape(shape)
            vf = np.fromfile(target / "vf.raw", dtype="<f8").reshape(shape)
            cs = np.fromfile(target / "cs.raw", dtype="<f8").reshape(shape)
            if not (np.all(q[12] == q[12, 0]) and np.all(q[12] == 0.0005000000000006111)
                    and np.all(vf[12] == q[12]) and np.all(cs[12] == 1)):
                raise ValueError("expected exact thin top row")
            q[12] = vf[12] = 0.5
            q.astype("<f8").tofile(target / "q.raw")
            vf.astype("<f8").tofile(target / "vf.raw")
            manifest = json.loads((target / "initial_state.json").read_text())
            manifest["target_liquid_volume"] = float(np.sum(q) * 0.125**2)
            manifest["branch"] = branch
            manifest["diagnostic_geometry"] = "flat interface at y=1.5625; top row j=12 liquid fraction 0.5"
            (target / "initial_state.json").write_text(json.dumps(manifest, indent=2) + "\n")
        config = ROOT / f"{branch}.conf"
        config.write_text(conf)
        configs[branch] = config
    shutil.copy2(SRC / "benchmark_supervisor.py", ROOT / "benchmark_supervisor.py")
    commands = {}
    for branch in branches:
        cmd = list(frozen["commands"]["zero"])
        def setopt(name: str, value: str) -> None:
            cmd[cmd.index(name) + 1] = value
        setopt("--run-root", f"{REMOTE}/runs/full/{branch}")
        setopt("--base-config", f"{REMOTE}/{configs[branch].name}")
        setopt("--input-dir", f"{REMOTE}/{branch}")
        setopt("--target-time", "0.00125")
        setopt("--dump-field-dt", "0.00125")
        commands[branch] = cmd
    source_hashes = {name: sha(SRC / name) for name in
                     ("native_launch_protocol.json", "twofluid_slip_inviscid_dt00125.conf", "benchmark_supervisor.py")}
    report = {"schema": "twofluid-flat-zero-one-step-diagnostics-v1",
              "purpose": "distinguish BC/viscosity, capillarity, and nearly-empty top-row effects in the failed zero floor; no candidate transfer outcomes",
              "predeclared_branches": branches,
              "source_hashes": source_hashes,
              "source_zero_input_sha256": {p.name: sha(p) for p in (SRC / "zero").iterdir()},
              "input_sha256": {branch: {p.name: sha(p) for p in (ROOT / branch).iterdir()} for branch in branches},
              "config_sha256": {branch: sha(configs[branch]) for branch in branches},
              "native_solver_sha256": frozen["expected_runtime_sha256"],
              "remote_root": REMOTE,
              "run_order": list(branches),
              "expected_runtime_sha256": frozen["expected_runtime_sha256"],
              "commands": commands,
              "gates": "identical supervisor mass, flux and speed tolerances as failed zero pilot; loss remains a failed result",
              "geometry_note": "halfrow branch changes the physical datum, not an operator parameter; it is only a diagnostic and will require a separate analytic reference if used subsequently"}
    (ROOT / "native_launch_protocol.json").write_text(json.dumps(report, indent=2) + "\n")
    archive = ROOT.parent / f"{ROOT.name}_inputs.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for path in sorted(ROOT.rglob("*")):
            if path.is_file():
                stream.add(path, arcname=path.relative_to(ROOT.parent).as_posix())
    print(json.dumps({"root": str(ROOT), "protocol_sha256": sha(ROOT / "native_launch_protocol.json"),
                      "archive_sha256": sha(archive), "branches": list(branches)}, indent=2))


if __name__ == "__main__":
    main()
