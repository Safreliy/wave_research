"""Freeze L9/L10 three-way linear two-fluid tests on one guarded runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np

from aphros_checkpoint_supervisor import audit_initial_state
from linear_twofluid_wave_reference import Wave, validate
from prepare_twofluid_physical_pilot import face_velocity
from prepare_auditor_runtime_campaign import COMMON, RUNTIME


REMOTE_ROOT = "/opt/gpu-cfd/transfer_auditor_revision_20261002/physical"
BRANCHES = ("exact", "matched", "bilinear")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_arg(command: list[str], key: str, value: str) -> None:
    command[command.index(key) + 1] = value


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError((old, text.count(old)))
    return text.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    if root.exists():
        raise FileExistsError(root)
    research = Path(__file__).resolve().parents[1]
    old = research / "results/publication_package"
    physical = old / "transfer_physical_pilot_20261001"
    dynamic = old / "transfer_dynamic_followup_20261001"
    matched = old / "transfer_matched_native_20261001"
    wave = Wave()
    validation = validate(wave)
    root.mkdir(parents=True)
    report = {"schema": "auditor-single-runtime-twofluid-v1",
              "predeclared_protocol_sha256": sha(root.parent / "PREDECLARED_PROTOCOL.md"),
              "remote_root": REMOTE_ROOT, "expected_runtime_sha256": RUNTIME,
              "wave": vars(wave), "reference_validation": validation,
              "receiver_assignment": "density-weighted liquid/gas phase means in mixed cells",
              "amplitude": wave.potential_amplitude, "target_time": 0.1 * wave.period,
              "grids": {}, "run_order": [], "commands": {}}
    for nx in (512, 1024):
        tag = f"L{nx.bit_length()-1}"
        shape = (nx // 16, nx)
        h = wave.length / nx
        grid = root / tag
        grid.mkdir()
        cells = wave.cells(nx, 0)
        if nx == 512:
            source = {
                "exact": physical / "flat_native_L9_twofluid/A_mass",
                "matched": physical / "flat_native_L9_guarded_mass_factorial/A_matched_mass",
                "bilinear": physical / "flat_native_L9_guarded_mass_factorial/A_practical_mass",
            }
            for label, path in source.items():
                shutil.copytree(path, grid / label)
            config_source = physical / "flat_native_L9_twofluid/twofluid_slip_inviscid_dt00125.conf"
            command_source = json.loads((physical / "flat_native_L9_guarded_mass_factorial/"
                                         "native_launch_protocol.json").read_text())["commands"]["A_matched_mass"]
            supervisor_source = physical / "flat_native_L9_twofluid/benchmark_supervisor.py"
            grid_source = {key: str(path) for key, path in source.items()}
        else:
            initial = dynamic / "flat_native_L10/exact_initial_means"
            matched_input = matched / "flat_native_L10_matched/fitted"
            practical_input = dynamic / "flat_native_L10/bilinear_centroid"
            source = {"exact": initial, "matched": matched_input,
                      "bilinear": practical_input}
            q = np.fromfile(initial / "q.raw", dtype="<f8").reshape(shape)
            cs = np.fromfile(initial / "cs.raw", dtype="<f8").reshape(shape)
            if max(np.max(abs(q - cells["q"])), np.max(abs(cs - cells["cs"]))) > 2e-14:
                raise ValueError("L10 native and analytic geometry differ")
            weights_l = wave.rho_liquid * q
            weights_g = wave.rho_gas * (cs - q)
            total = weights_l + weights_g
            flux = face_velocity(wave, nx)
            for label, path in source.items():
                folder = grid / label
                folder.mkdir()
                for name in ("q.raw", "vf.raw", "cs.raw", "body.dat"):
                    shutil.copy2(initial / name, folder / name)
                for name in ("fluxx", "fluxxp", "fluxy", "fluxyp", "fluxeb"):
                    np.asarray(flux[name], dtype="<f8").tofile(folder / f"{name}.raw")
                if label == "exact":
                    velocity = cells["mass_mean"]
                else:
                    liquid = np.stack([
                        np.fromfile(path / f"{component}.raw", dtype="<f8").reshape(shape)
                        for component in ("vx", "vy")]) * 1e-3
                    velocity = np.divide(weights_l[None] * liquid +
                                         weights_g[None] * cells["gas_mean"],
                                         total[None], out=np.zeros_like(liquid),
                                         where=total[None] > 0)
                    # Gas-only cells are exact and identical across branches.
                    velocity[:, q <= 0] = cells["mass_mean"][:, q <= 0]
                for component, values in zip(("vx", "vy"), velocity):
                    np.asarray(values, dtype="<f8").tofile(folder / f"{component}.raw")
                initial_state = {"schema": "auditor-twofluid-L10-initial-v1",
                                 "nx": nx, "ny": shape[0], "cell_size": h,
                                 "target_liquid_volume": wave.length * wave.depth,
                                 "target_liquid_momentum": [0.0, 0.0],
                                 "liquid_momentum": [float(np.sum(q * velocity[d]) * h*h)
                                                     for d in range(2)],
                                 "branch": label}
                (folder / "initial_state.json").write_text(json.dumps(initial_state, indent=2) + "\n")
            source_config = matched / "flat_native_L10_matched/flat_l10_dt0025.conf"
            text = source_config.read_text()
            for old_text, new_text in (
                ("wall 0 0 0 {", "slipwall {"),
                ("set double mu1 4.887640449438202e-7", "set double mu1 0"),
                ("set double mu2 2.5e-5", "set double mu2 0"),
                ("set double dt0 0.0025", "set double dt0 0.00125"),
                ("set double dtmax 0.0025", "set double dtmax 0.00125"),
                ("set double cfl 0.1", "set double cfl 0.05"),
                ("set double cfla 0.1", "set double cfla 0.05"),
            ):
                text = replace_once(text, old_text, new_text)
            config_source = grid / "twofluid_slip_inviscid_dt00125.conf"
            config_source.write_text(text, encoding="utf-8", newline="\n")
            command_source = json.loads((matched / "flat_native_L10_matched/"
                                         "native_launch_protocol.json").read_text())["command"]
            supervisor_source = matched / "flat_native_L10_matched/benchmark_supervisor.py"
            grid_source = {key: str(path) for key, path in source.items()}
        config = grid / "twofluid_slip_inviscid_dt00125.conf"
        if nx == 512:
            shutil.copy2(config_source, config)
        supervisor = grid / "benchmark_supervisor.py"
        shutil.copy2(supervisor_source, supervisor)
        grid_report = {"nx": nx, "ny": shape[0], "source": grid_source,
                       "config_sha256": sha(config), "supervisor_sha256": sha(supervisor),
                       "branches": {}}
        for label in BRANCHES:
            folder = grid / label
            state = json.loads((folder / "initial_state.json").read_text())
            check = audit_initial_state(folder, initial_mass_relative_tolerance=1e-8,
                                        face_divergence_tolerance=1e-10,
                                        shared_face_tolerance=1e-10,
                                        momentum_absolute_tolerance=1e-6)
            if not check["valid"]:
                raise ValueError((tag, label, check["violations"]))
            grid_report["branches"][label] = {
                "input_sha256": {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
                "preflight": check, "initial_state": state,
            }
            branch = f"{tag}_{label}"
            command = list(command_source)
            command[1] = f"{REMOTE_ROOT}/{tag}/benchmark_supervisor.py"
            set_arg(command, "--run-root", f"{REMOTE_ROOT}/{tag}/runs/{label}")
            set_arg(command, "--base-config", f"{REMOTE_ROOT}/{tag}/{config.name}")
            set_arg(command, "--input-dir", f"{REMOTE_ROOT}/{tag}/{label}")
            set_arg(command, "--solver", next(p for p in RUNTIME if p.endswith("/ap.mfer")))
            set_arg(command, "--solver-library", next(p for p in RUNTIME if p.endswith("/libaphros.so")))
            set_arg(command, "--target-time", str(0.1 * wave.period))
            set_arg(command, "--dump-field-dt", str(0.025 * wave.period))
            set_arg(command, "--cfl", "0.05")
            set_arg(command, "--cfla", "0.05")
            set_arg(command, "--dtmax", "0.00125")
            report["run_order"].append(branch)
            report["commands"][branch] = command
        common = grid_report["branches"]["exact"]["input_sha256"]
        for label in ("matched", "bilinear"):
            if any(grid_report["branches"][label]["input_sha256"][n] != common[n]
                   for n in COMMON):
                raise ValueError(f"{tag}: common physical fields differ")
        report["grids"][tag] = grid_report
    (root / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    archive = root.parent / "physical_inputs.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(root.parent).as_posix())
    print(json.dumps({"campaign_sha256": sha(root / "campaign.json"),
                      "archive_sha256": sha(archive), "archive_bytes": archive.stat().st_size},
                     indent=2))


if __name__ == "__main__":
    main()
