"""Freeze a five-run small-amplitude two-fluid wave pilot before native runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from aphros_checkpoint_supervisor import audit_initial_state
from linear_twofluid_wave_reference import Wave, validate


REMOTE_ROOT = "/opt/gpu-cfd/twofluid_physical_pilot_20261001"
BRANCHES = (
    ("A_phase", 2e-5, "phase_injection"),
    ("A_mass", 2e-5, "mass_mean"),
    ("Ahalf_phase", 1e-5, "phase_injection"),
    ("Ahalf_mass", 1e-5, "mass_mean"),
    ("zero", 0.0, "phase_injection"),
)
COMMON_RAW = ("q", "vf", "cs")
FLUX_RAW = ("fluxx", "fluxxp", "fluxy", "fluxyp", "fluxeb")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def streamfunction(wave: Wave, nx: int) -> np.ndarray:
    h = wave.length / nx
    x = np.arange(nx)[None, :] * h
    y = np.arange(round(wave.top/h) + 1)[:, None] * h
    k = wave.k
    liquid = (-wave.potential_amplitude * np.sin(k*x)
              * np.sinh(k*(y-wave.bottom)) / np.cosh(k*wave.depth))
    gas = (-wave.potential_amplitude * np.tanh(k*wave.depth) * np.sin(k*x)
           * np.cosh(k*(wave.top-y)) / np.cosh(k*wave.gas_depth))
    return np.where(y < wave.bottom, 0, np.where(y <= wave.surface, liquid, gas))


def face_velocity(wave: Wave, nx: int) -> dict[str, np.ndarray]:
    h = wave.length / nx
    psi = streamfunction(wave, nx)
    fluxx = np.diff(psi, axis=0) / h
    fluxxp = np.roll(fluxx, -1, axis=1)
    horizontal = -(np.roll(psi, -1, axis=1) - psi) / h
    fluxy, fluxyp = horizontal[:-1], horizontal[1:]
    divergence = (fluxxp-fluxx + fluxyp-fluxy)/h
    if np.max(np.abs(divergence)) > 1e-12:
        raise ValueError("discrete face velocities are not divergence free")
    return {"fluxx": fluxx, "fluxxp": fluxxp, "fluxy": fluxy,
            "fluxyp": fluxyp, "fluxeb": np.zeros_like(fluxx)}


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one config token: {old!r}")
    return text.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True,
                        help="frozen L9 prior benchmark root for identical geometry/solver")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    source_protocol = json.loads((args.source_root / "protocol.json").read_text())
    source_launch = json.loads((args.source_root / "native_launch_protocol.json").read_text())
    nx = source_protocol["geometry"]["nx"]
    if nx != 512:
        raise ValueError("physical pilot is frozen for L9, 512x32")
    wave = Wave()
    checks = validate(wave)
    target = 0.1 * wave.period
    args.output_root.mkdir(parents=True)
    supervisor = args.output_root / "benchmark_supervisor.py"
    shutil.copy2(args.source_root / "benchmark_supervisor.py", supervisor)
    source_config = args.source_root / "flat_dt00125.conf"
    config_text = source_config.read_text()
    for old, new in (
        ("wall 0 0 0 {", "slipwall {"),
        ("set double mu1 4.887640449438202e-7", "set double mu1 0"),
        ("set double mu2 2.5e-5", "set double mu2 0"),
        ("set double dtmax 0.0025", "set double dtmax 0.00125"),
        ("set double cfl 0.1", "set double cfl 0.05"),
        ("set double cfla 0.1", "set double cfla 0.05"),
    ):
        config_text = replace_once(config_text, old, new)
    config = args.output_root / "twofluid_slip_inviscid_dt00125.conf"
    config.write_text(config_text, encoding="utf-8", newline="\n")
    original_input = args.source_root / "exact_initial_means"
    original_q = np.fromfile(original_input / "q.raw", dtype="<f8").reshape(32, 512)
    original_vf = np.fromfile(original_input / "vf.raw", dtype="<f8").reshape(32, 512)
    original_cs = np.fromfile(original_input / "cs.raw", dtype="<f8").reshape(32, 512)
    cells = wave.cells(nx, 0)
    if max(float(np.max(np.abs(cells["q"]-original_q))),
           float(np.max(np.abs(cells["cs"]-original_cs)))) > 2e-14:
        raise ValueError("analytic cell geometry differs from frozen native geometry")
    protocol = {
        "schema": "twofluid-small-amplitude-native-pilot-v1",
        "claim_scope": "linear two-fluid analytic column-depth reference, with amplitude-halving and zero-floor controls; not exact nonlinear Navier-Stokes truth",
        "source_prior_protocol_sha256": sha256(args.source_root / "protocol.json"),
        "source_prior_launch_protocol_sha256": sha256(args.source_root / "native_launch_protocol.json"),
        "source_exact_geometry_sha256": {name: sha256(original_input / f"{name}.raw")
                                         for name in COMMON_RAW},
        "body_sha256": sha256(original_input / "body.dat"),
        "linear_reference_code_sha256": sha256(Path(__file__).with_name("linear_twofluid_wave_reference.py")),
        "linear_reference_validation": checks,
        "wave": asdict(wave),
        "linear_period": wave.period,
        "target_time": target,
        "amplitudes": [2e-5, 1e-5, 0.0],
        "branches": [{"name": name, "potential_amplitude": amp, "input_operator": operator}
                     for name, amp, operator in BRANCHES],
        "initial_condition": "analytic two-fluid phase-cell moments at t=0; phase_injection or density-weighted mass_mean in mixed cells, same analytic piecewise streamfunction face velocities",
        "native_physics_changes_from_prior": "mu1=mu2=0; embedded/domain bottom slipwall instead of no-slip wall; pressure outlet top unchanged",
        "primary_metric": "RMS column-depth error against cell-averaged linear analytic H, normalized by fixed B*sinc(kh/2)/sqrt(2); zero branch reports absolute floor separately and is not subtracted",
        "secondary_metrics": ["complex Fourier mode-16 depth amplitude/phase error", "orthogonal-mode RMS error", "mass and semantic gates"],
        "frozen_step_setting": {"dt0": 0.00125, "dtmax": 0.00125,
                                "cfl": 0.05, "cfla": 0.05},
        "reference_limit": "A/2 control checks amplitude dependence; systematic spatial/boundary errors remain; mass_mean is a candidate, not an asserted native contract",
        "expected_runtime_sha256": source_launch["expected_runtime_sha256"],
        "config_sha256": sha256(config),
        "supervisor_sha256": sha256(supervisor),
        "remote_root": REMOTE_ROOT,
        "code_sha256": sha256(Path(__file__)),
    }
    (args.output_root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    branch_reports = {}
    for name, amplitude, operator in BRANCHES:
        branch_wave = Wave(potential_amplitude=amplitude)
        branch_cells = branch_wave.cells(nx, 0)
        branch_flux = face_velocity(branch_wave, nx)
        folder = args.output_root / name
        folder.mkdir()
        for field in COMMON_RAW:
            shutil.copy2(original_input / f"{field}.raw", folder / f"{field}.raw")
        shutil.copy2(original_input / "body.dat", folder / "body.dat")
        for field in FLUX_RAW:
            np.asarray(branch_flux[field], dtype="<f8").tofile(folder / f"{field}.raw")
        velocity = branch_cells[operator]
        np.asarray(velocity[0], dtype="<f8").tofile(folder / "vx.raw")
        np.asarray(velocity[1], dtype="<f8").tofile(folder / "vy.raw")
        h = wave.length/nx
        momentum = [float(np.sum(original_q*velocity[j])*h*h) for j in (0, 1)]
        initial = {"schema": "twofluid-small-amplitude-initial-v1", "nx": nx, "ny": 32,
                   "cell_size": h, "target_liquid_volume": wave.length*wave.depth,
                   "target_liquid_momentum": [0.0, 0.0], "liquid_momentum": momentum,
                   "branch": name, "protocol_sha256": sha256(args.output_root / "protocol.json")}
        (folder / "initial_state.json").write_text(json.dumps(initial, indent=2) + "\n")
        preflight = audit_initial_state(folder, initial_mass_relative_tolerance=1e-8,
                                        face_divergence_tolerance=1e-10,
                                        shared_face_tolerance=1e-10,
                                        momentum_absolute_tolerance=1e-6)
        if not preflight["valid"]:
            raise ValueError(f"{name} preflight failed: {preflight['violations']}")
        branch_reports[name] = {"preflight": preflight,
                                "initial_max_velocity": float(np.max(np.hypot(*velocity))),
                                "input_sha256": {p.name: sha256(p) for p in sorted(folder.iterdir())
                                                 if p.is_file()}}
    (args.output_root / "preparation_report.json").write_text(
        json.dumps({"schema": "twofluid-pilot-preparation-v1", "branches": branch_reports}, indent=2) + "\n")
    input_archive = args.output_root.parent / f"{args.output_root.name}_inputs.tar.gz"
    with tarfile.open(input_archive, "w:gz") as archive:
        for path in sorted(args.output_root.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(args.output_root.parent).as_posix())
    print(json.dumps({"protocol_sha256": sha256(args.output_root / "protocol.json"),
                      "input_archive_sha256": sha256(input_archive),
                      "preflight_valid": {name: data["preflight"]["valid"]
                                          for name, data in branch_reports.items()},
                      "target_time": target}, indent=2))


if __name__ == "__main__":
    main()
