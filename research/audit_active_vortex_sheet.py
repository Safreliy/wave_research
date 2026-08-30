"""Audit active circulation evolution and conservative adaptive remeshing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from active_vortex_sheet import (
    ActiveVortexSheetState,
    active_vortex_sheet_rhs,
    panel_circulation_to_potential,
    potential_to_panel_circulation,
    step_active_vortex_sheet_rk4,
)
from adaptive_vortex_mesh import (
    project_vortex_sheet_invariants,
    remesh_active_vortex_sheet,
    resample_periodic_bottom,
    suggest_vortex_count,
    vortex_mesh_monitor,
)
from topographic_wave_solver import diagnostics, geometric_volume, step_rk4, theta_s_rhs


Array = np.ndarray


def _relative(first: Array, second: Array) -> float:
    return float(
        np.linalg.norm(np.asarray(first) - np.asarray(second))
        / max(float(np.linalg.norm(second)), np.finfo(float).eps)
    )


def _panel_arclength(x: Array, z: Array, length: float) -> tuple[Array, Array]:
    next_x = np.roll(x, -1)
    next_x[-1] += length
    ds = np.hypot(next_x - x, np.roll(z, -1) - z)
    s = np.r_[0.0, np.cumsum(ds)]
    return ds, s


def _plot(
    source: ActiveVortexSheetState,
    adapted: ActiveVortexSheetState,
    length: float,
    output: Path,
    time: float,
) -> None:
    source_monitor = vortex_mesh_monitor(source, length)
    adapted_monitor = vortex_mesh_monitor(adapted, length)
    source_ds, source_s = _panel_arclength(source.x, source.z, length)
    adapted_ds, adapted_s = _panel_arclength(adapted.x, adapted.z, length)
    source_coordinate = source_s[:-1] / source_s[-1]
    adapted_coordinate = adapted_s[:-1] / adapted_s[-1]
    source_mass = 0.5 * (
        source_monitor.value + np.roll(source_monitor.value, -1)
    ) * source_ds
    adapted_mass = 0.5 * (
        adapted_monitor.value + np.roll(adapted_monitor.value, -1)
    ) * adapted_ds

    figure, axes = plt.subplots(2, 2, figsize=(14.4, 9.0), constrained_layout=True)
    figure.suptitle(
        f"Active boundary-vortex-sheet state and conservative adaptation at t={time:.2f}",
        fontsize=16,
        fontweight="bold",
    )
    ax = axes[0, 0]
    ax.plot(source.x, source.z, color="#9aa6b2", linewidth=1.5, label=f"source N={len(source.x)}")
    scale = max(float(np.quantile(np.abs(adapted.panel_circulation), 0.98)), 1.0e-14)
    colors = np.where(adapted.panel_circulation >= 0.0, "#d94841", "#176b87")
    sizes = 8.0 + 54.0 * np.sqrt(np.minimum(np.abs(adapted.panel_circulation) / scale, 1.0))
    ax.scatter(adapted.x, adapted.z, c=colors, s=sizes, alpha=0.82, label=f"adapted N={len(adapted.x)}")
    crest = int(np.argmax(source.z))
    ax.set_xlim(source.x[crest] - 5.5, source.x[crest] + 5.5)
    ax.set_ylim(min(-0.05, float(np.min(source.z)) - 0.05), float(np.max(source.z)) + 0.12)
    ax.set_xlabel(r"$x/h_0$")
    ax.set_ylabel(r"$z/h_0$")
    ax.set_title("Geometric model: evolved circulation carriers")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.plot(source_coordinate, source_monitor.value, color="#171c2b", label="combined monitor")
    curvature_component = 4.0 * np.minimum(
        np.abs(source_monitor.curvature * np.median(source_monitor.panel_length))
        / max(float(np.quantile(np.abs(source_monitor.curvature * np.median(source_monitor.panel_length)), 0.90)), 1.0e-14),
        8.0,
    )
    strength_component = 1.5 * np.minimum(
        np.abs(source_monitor.strength_density)
        / max(float(np.quantile(np.abs(source_monitor.strength_density), 0.90)), 1.0e-14),
        8.0,
    )
    ax.plot(source_coordinate, curvature_component, color="#e09f3e", label="curvature contribution")
    ax.plot(source_coordinate, strength_component, color="#5b4b8a", label="sheet-strength contribution")
    ax.set_xlabel("normalized periodic arclength")
    ax.set_ylabel("monitor weight")
    ax.set_title("Discrete model: where vortices are requested")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    ax.plot(source_s / source_s[-1], np.r_[0.0, np.cumsum(source.panel_circulation)], label=f"source N={len(source.x)}")
    ax.plot(adapted_s / adapted_s[-1], np.r_[0.0, np.cumsum(adapted.panel_circulation)], "--", label=f"adapted N={len(adapted.x)}")
    ax.set_xlabel("normalized periodic arclength")
    ax.set_ylabel(r"cumulative circulation $C(s)$")
    ax.set_title("Combinatorial model: conservative cumulative remap")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    ax.plot(source_coordinate, source_mass / np.mean(source_mass), color="#cc4b37", alpha=0.8, label="before")
    ax.plot(adapted_coordinate, adapted_mass / np.mean(adapted_mass), color="#087e8b", alpha=0.9, label="after")
    ax.axhline(1.0, color="#334155", linestyle="--", linewidth=1.0)
    ax.set_xlabel("normalized periodic arclength")
    ax.set_ylabel(r"panel monitor mass / mean")
    ax.set_title("Equidistribution observation")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def audit(source_path: Path, output_prefix: Path, requested_time: float) -> dict[str, object]:
    with np.load(source_path) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    index = int(np.argmin(np.abs(np.asarray(data["time"]) - requested_time)))
    x = np.asarray(data["x"][index], dtype=float)
    z = np.asarray(data["z"][index], dtype=float)
    potential = np.asarray(data["potential"][index], dtype=float)
    bottom_x = np.asarray(data["bottom_x"], dtype=float)
    bottom_z = np.asarray(data["bottom_z"], dtype=float)
    length = float(data["length"])
    gravity = float(data["gravity"])
    background = float(data.get("background_current", np.asarray(0.0)))
    dt = float(data["dt"])
    gauge = str(data.get("tangential_gauge", np.asarray("theta_s")))
    circulation = potential_to_panel_circulation(x, potential, length, background)
    state = ActiveVortexSheetState(x, z, circulation, float(np.mean(potential)))
    reconstructed = panel_circulation_to_potential(
        x, circulation, length, background, state.potential_mean
    )

    potential_rhs = theta_s_rhs(
        x, z, potential, bottom_x, bottom_z, length, gravity, background, gauge
    )
    active_rhs = active_vortex_sheet_rhs(
        state, bottom_x, bottom_z, length, gravity, background, gauge
    )
    expected_phi_t = potential_rhs[2] + background * potential_rhs[0]
    expected_gamma_t = np.roll(expected_phi_t, -1) - expected_phi_t

    potential_step = step_rk4(
        x,
        z,
        potential,
        bottom_x,
        bottom_z,
        length,
        gravity,
        dt,
        background,
        gauge,
    )
    active_step, step_diagnostics = step_active_vortex_sheet_rk4(
        state,
        bottom_x,
        bottom_z,
        length,
        gravity,
        dt,
        background,
        gauge,
    )
    active_step_potential = panel_circulation_to_potential(
        active_step.x,
        active_step.panel_circulation,
        length,
        background,
        state.potential_mean,
    )
    monitor = vortex_mesh_monitor(state, length)
    decision = suggest_vortex_count(monitor, len(x), max(64, len(x) // 2), 2 * len(x))
    adapted, remesh_diagnostics = remesh_active_vortex_sheet(
        state,
        length,
        background,
        target_count=decision.target_count,
    )
    adapted_bottom_x, adapted_bottom_z = resample_periodic_bottom(
        bottom_x, bottom_z, length, decision.target_count
    )
    target_volume = float(
        data.get("global_target_volume", data["volume"][index])
    )
    raw_adapted_volume = geometric_volume(
        adapted.x,
        adapted.z,
        adapted_bottom_x,
        adapted_bottom_z,
        length,
    )
    volume_shift = (target_volume - raw_adapted_volume) / length
    volume_shifted = ActiveVortexSheetState(
        adapted.x,
        adapted.z + volume_shift,
        adapted.panel_circulation,
        adapted.potential_mean,
    )
    raw_potential = panel_circulation_to_potential(
        volume_shifted.x,
        volume_shifted.panel_circulation,
        length,
        background,
        volume_shifted.potential_mean,
    )
    raw_volume, raw_energy, _, _, raw_flux = diagnostics(
        volume_shifted.x,
        volume_shifted.z,
        raw_potential,
        adapted_bottom_x,
        adapted_bottom_z,
        length,
        gravity,
        background,
    )
    energy_scale = max(
        abs(float(data["global_initial_energy"]) - float(data["energy_reference"])),
        np.finfo(float).eps,
    )
    adapted, invariant_projection = project_vortex_sheet_invariants(
        adapted,
        adapted_bottom_x,
        adapted_bottom_z,
        length,
        gravity,
        background,
        target_volume,
        float(data["energy"][index]),
        energy_scale,
    )
    adapted_potential = panel_circulation_to_potential(
        adapted.x,
        adapted.panel_circulation,
        length,
        background,
        adapted.potential_mean,
    )
    adapted_volume, adapted_energy, _, _, adapted_flux = diagnostics(
        adapted.x,
        adapted.z,
        adapted_potential,
        adapted_bottom_x,
        adapted_bottom_z,
        length,
        gravity,
        background,
    )
    global_initial_energy = float(data["global_initial_energy"])
    remesh_physics = {
        "raw_relative_volume_jump": (raw_adapted_volume - target_volume) / target_volume,
        "volume_projection_shift": volume_shift,
        "volume_only_relative_volume_error": (raw_volume - target_volume)
        / target_volume,
        "volume_only_relative_wave_energy_jump": abs(
            raw_energy - float(data["energy"][index])
        )
        / energy_scale,
        "volume_only_global_energy_drift": abs(
            raw_energy - global_initial_energy
        )
        / energy_scale,
        "volume_only_raw_dno_flux_defect": abs(raw_flux),
        "invariant_projection": asdict(invariant_projection),
        "projected_relative_volume_error": (adapted_volume - target_volume)
        / target_volume,
        "projected_relative_wave_energy_error": abs(
            adapted_energy - float(data["energy"][index])
        )
        / energy_scale,
        "projected_global_energy_drift": abs(
            adapted_energy - global_initial_energy
        )
        / energy_scale,
        "projected_raw_dno_flux_defect": abs(adapted_flux),
    }
    _plot(
        state,
        adapted,
        length,
        output_prefix.with_suffix(".png"),
        float(data["time"][index]),
    )

    metrics = {
        "potential_roundtrip_relative_error": _relative(reconstructed, potential),
        "rhs_x_relative_error": _relative(active_rhs.x_t, potential_rhs[0]),
        "rhs_z_relative_error": _relative(active_rhs.z_t, potential_rhs[1]),
        "rhs_circulation_relative_error": _relative(active_rhs.circulation_t, expected_gamma_t),
        "one_step_x_relative_error": _relative(active_step.x, potential_step[0]),
        "one_step_z_relative_error": _relative(active_step.z, potential_step[1]),
        "one_step_potential_relative_error": _relative(active_step_potential, potential_step[2]),
        "one_step_circulation_change": float(active_step.total_circulation - state.total_circulation),
    }
    gates = {
        "roundtrip_le_1e-12": metrics["potential_roundtrip_relative_error"] <= 1.0e-12,
        "rhs_equivalence_le_1e-11": max(
            metrics["rhs_x_relative_error"],
            metrics["rhs_z_relative_error"],
            metrics["rhs_circulation_relative_error"],
        ) <= 1.0e-11,
        "one_step_equivalence_le_1e-10": max(
            metrics["one_step_x_relative_error"],
            metrics["one_step_z_relative_error"],
            metrics["one_step_potential_relative_error"],
        ) <= 1.0e-10,
        "circulation_change_le_1e-12": abs(metrics["one_step_circulation_change"]) <= 1.0e-12,
        "remesh_circulation_defect_le_1e-12": abs(remesh_diagnostics.total_circulation_defect) <= 1.0e-12,
        "remesh_monitor_cv_reduced": remesh_diagnostics.monitor_mass_cv_after < remesh_diagnostics.monitor_mass_cv_before,
        "raw_remesh_global_energy_gate_rejects": remesh_physics[
            "volume_only_global_energy_drift"
        ]
        > 5.0e-3,
        "invariant_projection_correction_le_1e-3": invariant_projection.relative_circulation_correction
        <= 1.0e-3,
        "projected_local_energy_error_le_1e-10": remesh_physics[
            "projected_relative_wave_energy_error"
        ]
        <= 1.0e-10,
        "projected_global_energy_drift_le_5e-3": remesh_physics[
            "projected_global_energy_drift"
        ]
        <= 5.0e-3,
        "remesh_projected_volume_error_le_5e-5": abs(remesh_physics["projected_relative_volume_error"]) <= 5.0e-5,
        "remesh_raw_flux_defect_le_2e-2": remesh_physics[
            "projected_raw_dno_flux_defect"
        ]
        <= 2.0e-2,
    }
    report = {
        "schema": "active-vortex-sheet-audit-v1",
        "evidence_class": "verified pre-contact state equivalence and conservative adaptive remesh" if all(gates.values()) else "rejected active-sheet prototype",
        "source": str(source_path),
        "snapshot_index": index,
        "time": float(data["time"][index]),
        "dt": dt,
        "source_count": len(x),
        "total_circulation": state.total_circulation,
        "kelvin_period_target": background * length,
        "metrics": metrics,
        "active_step_diagnostics": asdict(step_diagnostics),
        "monitor": {
            "maximum_turning_angle": monitor.maximum_turning_angle,
            "minimum_nonlocal_gap_ratio": monitor.minimum_nonlocal_gap_ratio,
            "maximum_circulation_fraction": monitor.maximum_circulation_fraction,
        },
        "adaptation_decision": asdict(decision),
        "remesh": asdict(remesh_diagnostics),
        "remesh_physics": remesh_physics,
        "acceptance_gates": gates,
        "accepted": all(gates.values()),
        "claim_boundary": {
            "picture": "node density and signed marker size visualize the active panel-circulation state",
            "computation": "state equivalence, one-step equivalence, Kelvin identities and the volume/energy-projected remesh pass the declared gates; the volume-only remesh is rejected by the global energy budget",
            "not_proved": "accuracy after self-contact, reconnection, entrained-air dynamics and viscous vorticity generation",
        },
    }
    output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--time", type=float, default=19.95)
    args = parser.parse_args()
    report = audit(args.source, args.output_prefix, args.time)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
