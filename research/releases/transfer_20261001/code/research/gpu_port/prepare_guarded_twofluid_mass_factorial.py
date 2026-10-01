"""Freeze the predeclared density-weighted matched/practical physical pair."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np

from aphros_checkpoint_supervisor import audit_initial_state
from linear_twofluid_wave_reference import Wave


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    research = Path(__file__).resolve().parents[1]
    home = research / "results/publication_package/transfer_physical_pilot_20261001"
    base = home / "flat_native_L9_twofluid"
    phase_candidates = home / "flat_native_L9_guarded_transfer"
    root = home / "flat_native_L9_guarded_mass_factorial"
    if root.exists():
        raise FileExistsError(root)
    wave = Wave()
    shape = (32, 512)
    common = ("q.raw", "vf.raw", "cs.raw", "body.dat", "fluxx.raw", "fluxxp.raw",
              "fluxy.raw", "fluxyp.raw", "fluxeb.raw")
    q = np.fromfile(base / "A_mass/q.raw", dtype="<f8").reshape(shape)
    cs = np.fromfile(base / "A_mass/cs.raw", dtype="<f8").reshape(shape)
    cells = wave.cells(512, 0)
    if max(np.max(abs(q - cells["q"])), np.max(abs(cs - cells["cs"]))) > 2e-14:
        raise ValueError("frozen and analytic cell geometry differ")
    liquid_weight = wave.rho_liquid * q
    gas_weight = wave.rho_gas * (cs - q)
    total_weight = liquid_weight + gas_weight
    wet = q > 0
    exact_phase = [np.fromfile(base / "A_phase" / f"{name}.raw", dtype="<f8").reshape(shape)
                   for name in ("vx", "vy")]
    exact_mass = [np.fromfile(base / "A_mass" / f"{name}.raw", dtype="<f8").reshape(shape)
                  for name in ("vx", "vy")]
    reconstructed_exact_mass = np.divide(
        liquid_weight[None] * np.stack(exact_phase) + gas_weight[None] * cells["gas_mean"],
        total_weight[None], out=np.zeros((2, *shape)), where=total_weight[None] > 0)
    exact_equivalence = float(np.max(abs(reconstructed_exact_mass - np.stack(exact_mass))))
    if exact_equivalence > 2e-15:
        raise ValueError(f"mixing formula does not reproduce frozen exact mass input: {exact_equivalence}")
    root.mkdir()
    outputs = {}
    commands = {}
    source_launch = json.loads((phase_candidates / "native_launch_protocol.json").read_text())
    remote_root = "/opt/gpu-cfd/twofluid_physical_pilot_20261001/flat_native_L9_guarded_mass_factorial"
    for name, source in (("A_matched_mass", "A_matched"),
                         ("A_practical_mass", "A_practical")):
        folder = root / name
        shutil.copytree(base / "A_mass", folder)
        if any(sha(phase_candidates / source / f) != sha(base / "A_mass" / f)
               for f in common):
            raise ValueError(f"common physical input differs: {source}")
        velocities = []
        phase_fields = []
        for d, component in enumerate(("vx", "vy")):
            liquid = np.fromfile(phase_candidates / source / f"{component}.raw", dtype="<f8").reshape(shape)
            phase_fields.append(liquid)
            field = np.divide(liquid_weight * liquid + gas_weight * cells["gas_mean"][d],
                              total_weight, out=np.zeros(shape), where=total_weight > 0)
            # All gas-only cells retain the exact baseline bytes.
            field[~wet] = exact_mass[d][~wet]
            if not np.array_equal(field[~wet], exact_mass[d][~wet]):
                raise ValueError("gas field changed")
            field.astype("<f8").tofile(folder / f"{component}.raw")
            velocities.append(field)
        initial = json.loads((folder / "initial_state.json").read_text())
        initial.update(schema="twofluid-guarded-mass-factorial-input-v1", branch=name,
                       liquid_momentum=[float(np.sum(q * field) * (wave.length / 512)**2)
                                        for field in velocities],
                       source_phase_branch=source,
                       mixture_formula="(rho_l*q*u_l + rho_g*(cs-q)*u_g)/(rho_l*q+rho_g*(cs-q))")
        (folder / "initial_state.json").write_text(json.dumps(initial, indent=2) + "\n")
        if any(sha(folder / f) != sha(base / "A_mass" / f) for f in common):
            raise ValueError("common fields changed")
        preflight = audit_initial_state(folder, initial_mass_relative_tolerance=1e-8,
                                        face_divergence_tolerance=1e-10,
                                        shared_face_tolerance=1e-10,
                                        momentum_absolute_tolerance=1e-6)
        if not preflight["valid"]:
            raise ValueError((name, preflight["violations"]))
        eu = np.sqrt(np.sum(q * sum((phase_fields[d] - exact_phase[d])**2 for d in range(2))) /
                     np.sum(q * sum(exact_phase[d]**2 for d in range(2))))
        outputs[name] = {"source_phase_branch": source,
                         "source_phase_velocity_sha256": {
                             component: sha(phase_candidates / source / f"{component}.raw")
                             for component in ("vx", "vy")},
                         "initial_liquid_phase_relative_l2": float(eu),
                         "preflight": preflight,
                         "input_sha256": {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}}
        cmd = list(source_launch["commands"][source])
        cmd[cmd.index("--run-root") + 1] = f"{remote_root}/runs/full/{name}"
        cmd[cmd.index("--input-dir") + 1] = f"{remote_root}/{name}"
        commands[name] = cmd
    report = {
        "schema": "twofluid-guarded-mass-factorial-preparation-v1",
        "analytic_reference_sha256": sha(Path(__file__).with_name("linear_twofluid_wave_reference.py")),
        "source_phase_candidate_protocol_sha256": sha(phase_candidates / "native_launch_protocol.json"),
        "source_exact_mass_input_archive_sha256": sha(home / "flat_native_L9_twofluid_inputs.tar.gz"),
        "source_phase_candidate_input_archive_sha256": sha(home / "flat_native_L9_guarded_transfer_inputs.tar.gz"),
        "exact_mass_reconstruction_max_abs": exact_equivalence,
        "geometry_and_serialized_faces": "byte-identical to A_mass and source phase candidates; receiver reconstructs runtime flux per branch",
        "gas_source": "analytic gas phase mean at t=0; gas-only cells copied bytewise from A_mass",
        "density_weighted_formula": "(rho_l*q*u_l+rho_g*(cs-q)*u_g)/(rho_l*q+rho_g*(cs-q))",
        "code_sha256": sha(Path(__file__)),
        "branches": outputs,
    }
    (root / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(Path(__file__), root / Path(__file__).name)
    launch = {
        "schema": "twofluid-guarded-mass-factorial-native-v1",
        "preparation_report_sha256": sha(root / "preparation_report.json"),
        "remote_root": remote_root,
        "run_order": list(commands),
        "expected_runtime_sha256": source_launch["expected_runtime_sha256"],
        "commands": commands,
        "reference_runs": ["guarded_full/zero", "guarded_full/A_phase", "guarded_full/A_mass"],
        "primary_metric": "analytic cell-average H RMS normalized by B*sinc(kh/2)/sqrt(2), all nonzero dumps and 0.1T endpoint",
        "fixed_amplitude": wave.potential_amplitude,
        "no_further_tuning": True,
    }
    (root / "native_launch_protocol.json").write_text(json.dumps(launch, indent=2) + "\n")
    archive = home / f"{root.name}_inputs.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for file in sorted(root.rglob("*")):
            if file.is_file():
                stream.add(file, arcname=file.relative_to(home).as_posix())
    print(json.dumps({"root": str(root), "exact_mass_max_abs": exact_equivalence,
                      "phase_initial_errors": {k: v["initial_liquid_phase_relative_l2"]
                                               for k, v in outputs.items()},
                      "launch_protocol_sha256": sha(root / "native_launch_protocol.json"),
                      "input_archive_sha256": sha(archive)}, indent=2))


if __name__ == "__main__":
    main()
