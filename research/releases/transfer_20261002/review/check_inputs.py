"""Independent, read-only audit of frozen auditor-revision native inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/gpu_port"))
from linear_twofluid_wave_reference import Wave, validate  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def field(folder: Path, name: str, shape: tuple[int, int]) -> np.ndarray:
    a = np.fromfile(folder / f"{name}.raw", dtype="<f8")
    if a.size != shape[0] * shape[1] or not np.isfinite(a).all():
        raise ValueError((folder, name, a.size))
    return a.reshape(shape)


def command_except_branch(command: list[str]) -> list[str]:
    result = list(command)
    for key in ("--run-root", "--input-dir"):
        result[result.index(key) + 1] = "<branch>"
    return result


def main() -> None:
    frozen = ROOT / "research/results/transfer_auditor_revision_20261002"
    physical = json.loads((frozen / "physical/campaign.json").read_text())
    dynamic = json.loads((frozen / "dynamic/campaign.json").read_text())
    wave = Wave()
    common = ("q.raw", "vf.raw", "cs.raw", "body.dat", "fluxx.raw", "fluxxp.raw",
              "fluxy.raw", "fluxyp.raw", "fluxeb.raw")
    source_build = json.loads((frozen / "runtime_provenance/source_build_match_report.json").read_text())
    source_runtime = source_build["guarded_runtime_sha256"]
    assert source_build["changed_source_files"] == []
    assert source_build["matched_source_files"] == 1605
    for plan in (physical, dynamic):
        for path, digest in plan["expected_runtime_sha256"].items():
            assert digest == source_runtime["bin/ap.mfer" if path.endswith("ap.mfer") else "lib/libaphros.so"]
    snapshots = (("physical", "PROTOCOL_AT_PHYSICAL_FREEZE.md"),
                 ("dynamic", "PROTOCOL_AT_DYNAMIC_FREEZE.md"))
    for kind, name in snapshots:
        assert json.loads((frozen / kind / "campaign.json").read_text())["predeclared_protocol_sha256"] == sha(frozen / name)

    physical_report = {}
    for nx in (512, 1024):
        tag = f"L{nx.bit_length() - 1}"
        shape = (nx // 16, nx)
        grid = frozen / "physical" / tag
        reference = wave.cells(nx, 0)
        q, cs = field(grid / "exact", "q", shape), field(grid / "exact", "cs", shape)
        exact = np.stack([field(grid / "exact", component, shape) for component in ("vx", "vy")])
        q_error = float(np.max(abs(q - reference["q"])))
        cs_error = float(np.max(abs(cs - reference["cs"])))
        exact_error = float(np.max(abs(exact - reference["mass_mean"])))
        assert max(q_error, cs_error) < 2e-14 and exact_error < 1e-14
        wl, wg = wave.rho_liquid * q, wave.rho_gas * (cs - q)
        total = wl + wg
        branches = {}
        for branch in ("matched", "bilinear"):
            folder = grid / branch
            common_identical = all(sha(folder / name) == sha(grid / "exact" / name)
                                   for name in common)
            assert common_identical
            velocity = np.stack([field(folder, component, shape) for component in ("vx", "vy")])
            old_root = ROOT / "research/results/publication_package"
            if branch == "matched":
                old = old_root / "transfer_matched_native_20261001" / f"flat_native_{tag}_matched/fitted"
            else:
                old = old_root / "transfer_dynamic_followup_20261001" / f"flat_native_{tag}/bilinear_centroid"
            liquid = np.stack([field(old, component, shape) for component in ("vx", "vy")]) * 1e-3
            predicted = np.divide(wl[None] * liquid + wg[None] * reference["gas_mean"],
                                  total[None], out=np.zeros_like(liquid), where=total[None] > 0)
            predicted[:, q <= 0] = reference["mass_mean"][:, q <= 0]
            mixture_error = float(np.max(abs(velocity - predicted)))
            dry_error = float(np.max(abs(velocity[:, q <= 0] - exact[:, q <= 0])))
            phase_moment_difference = q[None] * (velocity - liquid) * (wave.length / nx) ** 2
            local_moment_mislabel = float(np.max(abs(phase_moment_difference)))
            assert mixture_error < 2e-14 and dry_error == 0
            branches[branch] = {"common_inputs_byte_identical": common_identical,
                                "mixture_formula_max_abs_error": mixture_error,
                                "gas_only_max_abs_error": dry_error,
                                "max_abs_cellwise_q_weighted_mixture_vs_liquid_moment": local_moment_mislabel}
        config = (grid / "twofluid_slip_inviscid_dt00125.conf").read_text()
        for token in ("slipwall {", "outletpressure 0 {", "set double mu1 0",
                      "set double mu2 0", "set double rho1 0.001176470588235294",
                      "set double rho2 1", "set double sigma 0.001",
                      "set vect gravity 0 -1 0", "set double dtmax 0.00125"):
            assert token in config, (tag, token)
        cmds = [physical["commands"][f"{tag}_{b}"] for b in ("exact", "matched", "bilinear")]
        assert command_except_branch(cmds[0]) == command_except_branch(cmds[1]) == command_except_branch(cmds[2])
        physical_report[tag] = {"grid": list(shape), "q_analytic_max_abs_error": q_error,
                                "cs_analytic_max_abs_error": cs_error,
                                "exact_analytic_mixture_max_abs_error": exact_error,
                                "same_configuration_except_branch_paths": True,
                                "config_sha256": sha(grid / "twofluid_slip_inviscid_dt00125.conf"),
                                "branches": branches}

    dynamic_report = {}
    for nx in (512, 1024, 2048):
        tag = f"L{nx.bit_length() - 1}"
        shape = (nx // 16, nx)
        grid = frozen / "dynamic" / tag
        q = field(grid / "exact", "q", shape)
        branch_report = {}
        for branch in ("matched", "bilinear"):
            folder = grid / branch
            common_identical = all(sha(folder / name) == sha(grid / "exact" / name)
                                   for name in common)
            assert common_identical
            gas_error = max(float(np.max(abs(field(folder, name, shape)[q <= 0] -
                                             field(grid / "exact", name, shape)[q <= 0])))
                            for name in ("vx", "vy"))
            assert gas_error == 0
            branch_report[branch] = {"common_inputs_byte_identical": True,
                                     "gas_only_max_abs_error": gas_error}
        cmds = [dynamic["commands"][f"{tag}_{b}"] for b in ("exact", "matched", "bilinear")]
        assert command_except_branch(cmds[0]) == command_except_branch(cmds[1]) == command_except_branch(cmds[2])
        dynamic_report[tag] = {"grid": list(shape), "same_configuration_except_branch_paths": True,
                               "branches": branch_report}

    norm = {}
    for nx in (512, 1024):
        h = wave.length / nx
        t = 0.1 * wave.period
        x = (np.arange(nx) + 0.5) * h
        analytic = wave.depth + wave.surface_amplitude * np.sin(wave.omega * t) * \
            np.sinc(wave.k * h / (2 * np.pi)) * np.cos(wave.k * x)
        integrated = wave.cells(nx, t)["column_depth"]
        reference_difference = float(np.max(abs(integrated - analytic)))
        denominator = abs(wave.surface_amplitude * np.sinc(wave.k * h / (2 * np.pi))) / np.sqrt(2)
        direct_rms = float(np.sqrt(np.mean((analytic - wave.depth) ** 2)))
        assert reference_difference < 1e-12
        assert abs(direct_rms - denominator * abs(np.sin(wave.omega * t))) < 1e-12
        norm[f"L{nx.bit_length()-1}"] = {"cell_averaged_reference_max_abs_error": reference_difference,
                                       "fixed_amplitude_rms_denominator": float(denominator),
                                       "instantaneous_rms": direct_rms}

    report = {"schema": "auditor-independent-input-review-v1",
              "source_scripts_sha256": {name: sha(ROOT / "research/gpu_port" / name) for name in
                                        ("prepare_auditor_twofluid_grid.py", "prepare_auditor_runtime_campaign.py",
                                         "linear_twofluid_wave_reference.py", "prepare_twofluid_physical_pilot.py",
                                         "analyze_flat_dynamic.py", "analyze_guarded_twofluid_mass_factorial.py")},
              "campaign_sha256": {name: sha(frozen / name / "campaign.json") for name in ("physical", "dynamic")},
              "wave_boundary_validation": validate(wave),
              "physical": physical_report, "dynamic": dynamic_report,
              "physical_reference_normalization": norm,
              "runtime_source_build_match": {"matched_source_files": source_build["matched_source_files"],
                                             "changed_source_files": source_build["changed_source_files"],
                                             "guarded_runtime_sha256": source_runtime},
              "metric_scope": {"dynamic": "q-weighted velocity difference from same-grid exact-initial-input native trajectory; not physical truth",
                               "physical": "RMS column-depth error against cell-averaged inviscid linear two-fluid wave, divided by fixed analytic amplitude RMS",
                               "face_arrays": "Serialized face arrays are common, but the receiver reconstructs native initial face fluxes from each branch's cell velocities."},
              "runtime_include_caveat": "Grid configs include /opt/gpu-cfd/aphros/deploy/scripts/sim_base.conf; this check did not read or hash that live include file.",
              "metadata_issue": "Physical mixed-cell initial_state.json and supervisor call sum(q*assigned mixture velocity) liquid_momentum; this is not the liquid-phase moment sum(q*liquid mean). The test is an assigned-velocity loading diagnostic only."}
    out = Path(__file__).with_name("input_review.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "physical_grids": list(physical_report),
                      "dynamic_grids": list(dynamic_report), "normalization_grids": list(norm)}, indent=2))


if __name__ == "__main__":
    main()
