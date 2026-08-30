"""Reproducible benchmark driver for the final JFM 1:30, H0/h0=0.6 case.

The article uses walls.  The present spectral BIE uses a periodic cell, so a
smooth return slope is placed downstream of the 10 h0 shelf.  The wave reaches
the physical upslope and breaks before that closure can influence the event;
this must nevertheless be treated as a declared model difference.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np

from solitary_wave import full_width_above_threshold, fully_nonlinear_periodic_solitary_wave
from topography_bie import TopographyBIE
from topographic_wave_solver import diagnostics, simulate, smooth_reef_bathymetry


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "publication_package"


@dataclass(frozen=True)
class JFMFinalCase:
    h0: float = 1.0
    gravity: float = 1.0
    amplitude: float = 0.6
    slope: float = 1.0 / 30.0
    shallow_depth: float = 0.05
    shelf_length: float = 10.0
    return_length: float = 5.0
    corner_width: float = 0.1
    cluster_strength: float = 3.0
    cluster_width: float = 0.4
    length: float = 64.0

    @property
    def full_width(self) -> float:
        return full_width_above_threshold(self.amplitude, self.h0, 1.0e-3)

    @property
    def deep_length(self) -> float:
        return self.full_width + 5.0 * self.h0

    @property
    def crest_center(self) -> float:
        # Article coordinates: x_toe=0 and x0=-Wf/2-4h0.  Shift the left
        # boundary x=-Ld to zero for the periodic implementation.
        return self.deep_length - 0.5 * self.full_width - 4.0 * self.h0

    @property
    def slope_length(self) -> float:
        return (self.h0 - self.shallow_depth) / self.slope

    def declared_geometry(self) -> dict[str, object]:
        return {
            **asdict(self),
            "full_width_over_h0": self.full_width / self.h0,
            "deep_length_over_h0": self.deep_length / self.h0,
            "crest_center_over_h0": self.crest_center / self.h0,
            "slope_length_over_h0": self.slope_length / self.h0,
            "article_boundary_model": "closed tank with vertical walls",
            "present_boundary_model": (
                "periodic smooth return slope downstream of the physical shelf"
            ),
            "comparison_event": "first vertical front: min(x_alpha)=0",
        }


def _bed(case: JFMFinalCase, n: int) -> tuple[np.ndarray, np.ndarray]:
    return smooth_reef_bathymetry(
        n,
        case.length,
        case.h0,
        case.shallow_depth,
        case.slope,
        case.deep_length,
        case.shelf_length,
        case.return_length,
        case.corner_width,
        case.cluster_strength,
        case.cluster_width,
    )


def initial_operator_audit(
    case: JFMFinalCase, resolutions: tuple[int, ...]
) -> dict[str, object]:
    rows: list[dict[str, float | int]] = []
    for n in resolutions:
        x = np.arange(n) * case.length / n
        wave = fully_nonlinear_periodic_solitary_wave(
            x,
            case.length,
            case.crest_center,
            case.amplitude,
            case.h0,
            case.gravity,
            internal_points=max(2048, 16 * n),
        )
        bottom_x, bottom_z = _bed(case, n)
        started = time.perf_counter()
        volume, energy, _, marker_cv, flux = diagnostics(
            x,
            wave.elevation,
            wave.periodic_potential,
            bottom_x,
            bottom_z,
            case.length,
            case.gravity,
            wave.background_current,
        )
        flat_bie = TopographyBIE(
            x, wave.elevation, x, -case.h0 * np.ones(n), case.length
        )
        flat_solution = flat_bie.solve(wave.periodic_potential)
        flat_normal_x = -flat_bie.surface.z_alpha / flat_bie.surface.metric
        raw_target = (
            flat_solution.surface_normal_derivative
            + wave.background_current * flat_normal_x
        )
        target = raw_target.copy()
        target_flux = flat_bie.dalpha * np.dot(flat_bie.surface.metric, target)
        target -= target_flux / (
            flat_bie.dalpha * np.sum(flat_bie.surface.metric)
        )
        flux_correction = float(
            np.linalg.norm(target - raw_target)
            / max(float(np.linalg.norm(raw_target)), np.finfo(float).eps)
        )
        topographic_bie = TopographyBIE(
            x, wave.elevation, bottom_x, bottom_z, case.length
        )
        projection = topographic_bie.solve_neumann_to_dirichlet(target)
        projected_check = topographic_bie.solve(projection.surface_potential)
        projection_error = float(
            np.linalg.norm(projected_check.surface_normal_derivative - target)
            / max(float(np.linalg.norm(target)), np.finfo(float).eps)
        )
        rows.append(
            {
                "n": n,
                "nominal_surface_dx_over_h0": case.length / n / case.h0,
                "minimum_bottom_dx_over_h0": float(np.min(np.diff(bottom_x)) / case.h0),
                "maximum_bottom_dx_over_h0": float(np.max(np.diff(bottom_x)) / case.h0),
                "absolute_surface_flux_defect": abs(flux),
                "initial_volume": volume,
                "initial_energy": energy,
                "marker_spacing_cv": marker_cv,
                "relative_neumann_projection_error": projection_error,
                "neumann_projection_residual": projection.residual,
                "relative_flux_compatibility_correction": flux_correction,
                "wall_seconds_one_diagnostic": time.perf_counter() - started,
            }
        )
    return {
        "schema": "jfm-final-case-initial-operator-audit-v1",
        "evidence_class": "new solver, initial-state spatial audit",
        "case": case.declared_geometry(),
        "rows": rows,
        "acceptance": {
            "quantities": {
                "absolute_surface_flux_defect": 1.0e-3,
                "relative_neumann_projection_error": 1.0e-2,
            },
            "interpretation": (
                "initial compatibility screen only; not event convergence"
            ),
        },
    }


def _breaking_metrics(data: dict[str, np.ndarray]) -> dict[str, float | bool | str]:
    initial_energy = float(
        data.get("global_initial_energy", np.asarray(data["energy"][0]))
    )
    initial_volume = float(
        data.get("global_target_volume", np.asarray(data["volume"][0]))
    )
    energy_scale = max(
        abs(float(initial_energy - data["energy_reference"])),
        np.finfo(float).eps,
    )
    relative_energy = np.abs(data["energy"] - initial_energy) / energy_scale
    relative_volume = np.abs(data["volume"] - initial_volume) / max(
        abs(initial_volume), np.finfo(float).eps
    )
    uses_curvature_monitor = float(data["reparameterization_strength"]) > 0.0
    spacing_defect = (
        data["monitor_equidistribution_cv"]
        if uses_curvature_monitor
        else data["marker_cv"]
    )
    cumulative_filter_correction = float(
        data.get("cumulative_filter_correction_offset", np.asarray(0.0))
    ) + np.cumsum(data["spectral_filter_relative_corrections"])
    admissible = (
        (relative_energy <= 5.0e-3)
        & (relative_volume <= 5.0e-5)
        & (np.abs(data["surface_flux_defect"]) <= 2.0e-2)
        & (spacing_defect <= 1.0e-1)
        & (cumulative_filter_correction <= 5.0e-2)
    )
    failures = np.flatnonzero(~admissible)
    first_failure = int(failures[0]) if len(failures) else None
    last_admissible = (
        max(0, first_failure - 1) if first_failure is not None else len(admissible) - 1
    )
    crossings = np.where(data["min_x_alpha"] <= 0.0)[0]
    event = int(crossings[0]) if len(crossings) else len(data["time"]) - 1
    accepted_event = bool(len(crossings) and admissible[event])
    surface_x = data["x"][event]
    surface_z = data["z"][event]
    crest = int(np.argmax(surface_z))
    crest_x = float(surface_x[crest] % float(data["length"]))
    bottom_x = data["bottom_x"]
    bottom_z = data["bottom_z"]
    local_bed = float(np.interp(crest_x, bottom_x, bottom_z, period=float(data["length"])))
    h0 = float(data["depth"])
    wave_height = float(surface_z[crest])
    return {
        "reached_first_vertical_front": bool(len(crossings)),
        "accepted_first_vertical_front": accepted_event,
        "event_status": (
            "accepted_by_conservation_gates"
            if accepted_event
            else "rejected_by_conservation_gates"
            if len(crossings)
            else "not_reached"
        ),
        "termination_reason": str(data["termination_reason"]),
        "event_time_sqrt_g_over_h0": float(data["time"][event]),
        "crest_x_over_h0": crest_x / h0,
        "H_b_over_h0": wave_height / h0,
        "h_b_over_h0": -local_bed / h0,
        "gamma_b": wave_height / (-local_bed),
        "minimum_x_alpha": float(data["min_x_alpha"][event]),
        "max_relative_wave_energy_drift": float(
            np.max(relative_energy)
        ),
        "max_relative_volume_drift": float(
            np.max(relative_volume)
        ),
        "max_absolute_surface_flux_defect": float(
            np.max(np.abs(data["surface_flux_defect"]))
        ),
        "max_bie_residual": float(np.max(data["bie_residual"])),
        "max_marker_spacing_cv": float(np.max(data["marker_cv"])),
        "spacing_gate_quantity": (
            "curvature_monitor_equidistribution_cv"
            if uses_curvature_monitor
            else "marker_spacing_cv"
        ),
        "max_spacing_gate_defect": float(np.max(spacing_defect)),
        "maximum_cumulative_filter_correction": float(
            np.max(cumulative_filter_correction)
        ),
        "first_conservation_gate_failure_time": (
            float(data["time"][first_failure]) if first_failure is not None else None
        ),
        "last_admissible_time": float(data["time"][last_admissible]),
        "first_conservation_gate_failure_reasons": (
            [
                name
                for name, failed in (
                    ("relative_wave_energy_drift_gt_5e-3", relative_energy[first_failure] > 5.0e-3),
                    ("relative_volume_drift_gt_5e-5", relative_volume[first_failure] > 5.0e-5),
                    ("absolute_surface_flux_defect_gt_2e-2", abs(data["surface_flux_defect"][first_failure]) > 2.0e-2),
                    ("spacing_defect_gt_1e-1", spacing_defect[first_failure] > 1.0e-1),
                    ("cumulative_filter_correction_gt_5e-2", cumulative_filter_correction[first_failure] > 5.0e-2),
                )
                if failed
            ]
            if first_failure is not None
            else []
        ),
    }


def run_grid(
    case: JFMFinalCase,
    n: int,
    dt: float,
    final_time: float,
    output_prefix: Path,
    time_integrator: str = "rk4",
    reparameterization_strength: float = 0.0,
    reparameterization_power: float = 0.5,
    reparameterize_every: int = 0,
    spectral_filter_strength: float = 0.0,
    spectral_filter_order: int = 16,
) -> dict[str, object]:
    started = time.perf_counter()
    data = simulate(
        n=n,
        length=case.length,
        depth=case.h0,
        gravity=case.gravity,
        amplitude=case.amplitude,
        crest_center=case.crest_center,
        initial_condition="projected_exact_solitary",
        exact_internal_points=max(2048, 16 * n),
        bathymetry="reef",
        shallow_depth=case.shallow_depth,
        reef_slope=case.slope,
        reef_toe=case.deep_length,
        shelf_length=case.shelf_length,
        return_length=case.return_length,
        corner_width=case.corner_width,
        reef_cluster_strength=case.cluster_strength,
        reef_cluster_width=case.cluster_width,
        dt=dt,
        final_time=final_time,
        snapshots=max(121, int(math.ceil(final_time / dt)) + 1),
        stop_at_overturning=True,
        time_integrator=time_integrator,
        tangential_gauge="theta_s",
        reparameterization_strength=reparameterization_strength,
        reparameterization_power=reparameterization_power,
        reparameterize_every=reparameterize_every,
        spectral_filter_strength=spectral_filter_strength,
        spectral_filter_order=spectral_filter_order,
        initial_reparameterize=(reparameterization_strength > 0.0),
        project_volume=True,
        implicit_fallback=False,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_prefix.with_suffix(".npz"), **data)
    initial_x = data["x"][0]
    initial_z = data["z"][0]
    initial_panel_length = np.hypot(
        np.diff(np.r_[initial_x, initial_x[0] + case.length]),
        np.diff(np.r_[initial_z, initial_z[0]]),
    )
    report: dict[str, object] = {
        "schema": "jfm-final-case-grid-run-v1",
        "evidence_class": "new solver, event run",
        "case": case.declared_geometry(),
        "n": n,
        "nominal_surface_dx_over_h0": case.length / n / case.h0,
        "minimum_initial_surface_panel_over_h0": float(
            np.min(initial_panel_length) / case.h0
        ),
        "maximum_initial_surface_panel_over_h0": float(
            np.max(initial_panel_length) / case.h0
        ),
        "reparameterization_strength": reparameterization_strength,
        "reparameterization_power": reparameterization_power,
        "reparameterize_every": reparameterize_every,
        "initial_reparameterize": bool(data["initial_reparameterize"]),
        "spectral_filter_strength": spectral_filter_strength,
        "spectral_filter_order": spectral_filter_order,
        "maximum_spectral_filter_relative_correction": float(
            np.max(data["spectral_filter_relative_corrections"])
        ),
        "cumulative_spectral_filter_relative_correction": float(
            np.sum(data["spectral_filter_relative_corrections"])
        ),
        "dt_sqrt_g_over_h0": float(data["dt"]),
        "time_integrator": str(data["time_integrator"]),
        "implicit_fallback_count": int(np.sum(data["implicit_fallback"])),
        "maximum_nonlinear_residual": float(
            np.max(data["nonlinear_residual"])
        ),
        "initial_projection_error": float(data["initial_projection_error"]),
        "initial_projection_residual": float(data["initial_projection_residual"]),
        "initial_projection_flux_correction": float(
            data["initial_projection_flux_correction"]
        ),
        "maximum_absolute_volume_projection_shift": float(
            np.max(np.abs(data["volume_projection_shifts"]))
        ),
        "cumulative_absolute_volume_projection_shift": float(
            np.sum(np.abs(data["volume_projection_shifts"]))
        ),
        "wall_seconds": time.perf_counter() - started,
        "published_reference": {
            "H_b_over_h0": 0.749,
            "H_I_over_h0": 0.707,
            "note": "JFM 2026 headline 1:30, H0/h0=0.6 case",
        },
        **_breaking_metrics(data),
    }
    output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("initial-audit", "run-grid"), default="initial-audit")
    parser.add_argument("--resolutions", type=int, nargs="+", default=(64, 96, 128, 192))
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--final-time", type=float, default=12.0)
    parser.add_argument(
        "--time-integrator",
        choices=("rk4", "implicit_midpoint"),
        default="rk4",
    )
    parser.add_argument("--reparameterization-strength", type=float, default=0.0)
    parser.add_argument("--reparameterization-power", type=float, default=0.5)
    parser.add_argument("--reparameterize-every", type=int, default=0)
    parser.add_argument("--spectral-filter-strength", type=float, default=0.0)
    parser.add_argument("--spectral-filter-order", type=int, default=16)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    case = JFMFinalCase()
    if args.mode == "initial-audit":
        report = initial_operator_audit(case, tuple(args.resolutions))
        output = args.output or RESULTS / "jfm_final_case_initial_operator_audit.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    else:
        output = args.output or RESULTS / f"jfm_final_case_n{args.n}"
        report = run_grid(
            case,
            args.n,
            args.dt,
            args.final_time,
            output,
            args.time_integrator,
            args.reparameterization_strength,
            args.reparameterization_power,
            args.reparameterize_every,
            args.spectral_filter_strength,
            args.spectral_filter_order,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
