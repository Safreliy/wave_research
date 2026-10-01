"""Scale frozen L9 liquid transfer arrays into the guarded physical-wave pilot."""

from __future__ import annotations

import hashlib
import json
import math
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
    physical_home = research / "results/publication_package/transfer_physical_pilot_20261001"
    base = physical_home / "flat_native_L9_twofluid"
    old_home = research / "results/publication_package/transfer_matched_native_20261001"
    old = old_home / "flat_native_L9_matched"
    root = physical_home / "flat_native_L9_guarded_transfer"
    if root.exists():
        raise FileExistsError(root)
    root.mkdir()
    old_protocol = json.loads((old / "protocol.json").read_text())
    if old_protocol["physical_trace"] != "parent_hermite_matched":
        raise ValueError("frozen old candidate is not matched Hermite")
    old_amplitude = old_protocol["initial_field"]["potential_amplitude"]
    wave = Wave()
    scale = wave.potential_amplitude / old_amplitude
    if not math.isclose(scale, 0.001, rel_tol=0, abs_tol=1e-15):
        raise ValueError(scale)
    shape = (32, 512)
    q = np.fromfile(base / "A_phase/q.raw", dtype="<f8").reshape(shape)
    cs = np.fromfile(base / "A_phase/cs.raw", dtype="<f8").reshape(shape)
    wet = q > 0
    common = ("q.raw", "vf.raw", "cs.raw", "body.dat", "fluxx.raw", "fluxxp.raw",
              "fluxy.raw", "fluxyp.raw", "fluxeb.raw")
    for branch in ("fitted", "bilinear_centroid", "exact_initial_means"):
        for name in ("q.raw", "vf.raw", "cs.raw", "body.dat"):
            if sha(old / branch / name) != sha(base / "A_phase" / name):
                raise ValueError(f"geometry differs: {branch}/{name}")
    exact = [np.fromfile(base / "A_phase" / f"{component}.raw", dtype="<f8").reshape(shape)
             for component in ("vx", "vy")]
    old_exact = [np.fromfile(old / "exact_initial_means" / f"{component}.raw", dtype="<f8").reshape(shape)
                 for component in ("vx", "vy")]
    exact_scaling_error = max(float(np.max(np.abs(scale * old_exact[d][wet] - exact[d][wet])))
                              for d in range(2))
    if exact_scaling_error > 1e-12:
        raise ValueError(f"liquid exact field not linearly equivalent: {exact_scaling_error}")
    physical = json.loads((physical_home / "guarded_full_launch_protocol.json").read_text())
    remote_root = "/opt/gpu-cfd/twofluid_physical_pilot_20261001/flat_native_L9_guarded_transfer"
    outputs = {}
    commands = {}
    for name, source_branch in (("A_matched", "fitted"), ("A_practical", "bilinear_centroid")):
        folder = root / name
        shutil.copytree(base / "A_phase", folder)
        velocities = []
        for d, component in enumerate(("vx", "vy")):
            arr = exact[d].copy()
            source = np.fromfile(old / source_branch / f"{component}.raw", dtype="<f8").reshape(shape)
            arr[wet] = scale * source[wet]
            arr.astype("<f8").tofile(folder / f"{component}.raw")
            velocities.append(arr)
        initial = json.loads((folder / "initial_state.json").read_text())
        initial["schema"] = "twofluid-guarded-scaled-transfer-input-v1"
        initial["branch"] = name
        initial["liquid_momentum"] = [float(np.sum(q * a) * wave.length**2 / 512**2) for a in velocities]
        initial["source_liquid_velocity_branch"] = source_branch
        initial["liquid_velocity_scale"] = scale
        (folder / "initial_state.json").write_text(json.dumps(initial, indent=2) + "\n")
        if any(sha(folder / file) != sha(base / "A_phase" / file) for file in common):
            raise ValueError("common input changed")
        if any(not np.array_equal(velocities[d][~wet], exact[d][~wet]) for d in range(2)):
            raise ValueError("analytic gas changed")
        preflight = audit_initial_state(
            folder, initial_mass_relative_tolerance=1e-8,
            face_divergence_tolerance=1e-10, shared_face_tolerance=1e-10,
            momentum_absolute_tolerance=1e-6,
        )
        if not preflight["valid"]:
            raise ValueError((name, preflight["violations"]))
        eu = math.sqrt(float(np.sum(q * sum((velocities[d]-exact[d])**2 for d in range(2))) /
                             np.sum(q * sum(exact[d]**2 for d in range(2)))))
        outputs[name] = {"source_branch": source_branch, "source_velocity_hashes": {
            component: sha(old / source_branch / f"{component}.raw") for component in ("vx", "vy")},
            "initial_liquid_velocity_relative_l2": eu,
            "preflight": preflight,
            "input_hashes": {p.name: sha(p) for p in folder.iterdir() if p.is_file()}}
        cmd = list(physical["commands"]["A_phase"])
        cmd[cmd.index("--run-root") + 1] = f"{remote_root}/runs/full/{name}"
        cmd[cmd.index("--input-dir") + 1] = f"{remote_root}/{name}"
        commands[name] = cmd
    report = {
        "schema": "twofluid-guarded-scaled-transfer-preparation-v1",
        "old_matched_inputs_archive_sha256": sha(old_home / "flat_native_L9_matched_inputs.tar.gz"),
        "old_matched_protocol_sha256": sha(old / "protocol.json"),
        "physical_inputs_archive_sha256": sha(physical_home / "flat_native_L9_twofluid_inputs.tar.gz"),
        "physical_launch_sha256": sha(physical_home / "guarded_full_launch_protocol.json"),
        "scale": scale,
        "exact_liquid_scaling_max_abs_error": exact_scaling_error,
        "common_input_fields": common,
        "common_input_hashes": {file: sha(base / "A_phase" / file) for file in common},
        "dry_and_gas_velocity_source": "A_phase analytic input, byte-identical across candidates",
        "runtime_flux_note": "serialized face arrays are common inputs but receiver starts with flux_init zero and reconstructs actual face flux independently per cell field",
        "code_sha256": sha(Path(__file__)),
        "branches": outputs,
    }
    (root / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(Path(__file__), root / "prepare_guarded_twofluid_transfer_inputs.py")
    launch = {"schema": "twofluid-guarded-scaled-transfer-native-v1",
              "preparation_report_sha256": sha(root / "preparation_report.json"),
              "remote_root": remote_root,
              "run_order": tuple(commands),
              "expected_runtime_sha256": physical["expected_runtime_sha256"],
              "commands": commands,
              "zero_and_phase_reference": "guarded_full/zero and guarded_full/A_phase from existing frozen protocol",
              "primary_metric": physical["primary_observable"]}
    (root / "native_launch_protocol.json").write_text(json.dumps(launch, indent=2) + "\n")
    archive = root.parent / f"{root.name}_inputs.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for file in sorted(root.rglob("*")):
            if file.is_file():
                stream.add(file, arcname=file.relative_to(root.parent).as_posix())
    print(json.dumps({"root": str(root), "scale": scale, "exact_scaling_max_abs": exact_scaling_error,
                      "initial_errors": {k: v["initial_liquid_velocity_relative_l2"] for k, v in outputs.items()},
                      "launch_protocol_sha256": sha(root / "native_launch_protocol.json"),
                      "input_archive_sha256": sha(archive)}, indent=2))


if __name__ == "__main__":
    main()
