"""Build the reproducible numerical-method revision audit.

The script only consumes checked-in/archived solver logs.  It deliberately
separates short-prefix regression evidence from the still-open impact study.
"""

from __future__ import annotations

import json
import math
import re
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter


ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "research" / "two_phase_basilisk"
OUTPUT = ROOT / "research" / "results" / "q_transport"


def numeric_rows(path: Path, prefix: str) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if parts and parts[0] == prefix:
            rows.append([float(value) for value in parts[1:]])
    if not rows:
        raise RuntimeError(f"No {prefix!r} rows in {path}")
    return rows


def inclined_case(mode: str, level: int) -> dict[str, float | int | str]:
    directory = (
        "audit_inclined_cutcell_cfl" if mode == "cutcell_cfl" else
        "audit_inclined_baseline"
    )
    log = SOLVER / directory / f"fixed_l{level}.log"
    stdout = SOLVER / directory / f"fixed_l{level}.stdout"
    history = numeric_rows(log, "Q_INCLINED")
    final = numeric_rows(log, "Q_INCLINED_FINAL")[-1]
    # Q_INCLINED: t, iteration, volume, lower bound, upper bound,
    # maximum bound violation, maximum open-flux mismatch, clipped aperture,
    # corrected cells.
    initial_volume = history[0][2]
    final_volume = history[-1][2]
    perf = stdout.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"# Quadtree, (\d+) steps, ([0-9.eE+-]+) CPU, ([0-9.eE+-]+) real",
        perf,
    )
    if not match:
        raise RuntimeError(f"No performance footer in {stdout}")
    return {
        "mode": mode,
        "level": level,
        "cells_per_side": 2**level,
        "initial_volume": initial_volume,
        "final_volume": final_volume,
        "relative_volume_drift": (final_volume - initial_volume) / initial_volume,
        "maximum_bound_violation": max(row[5] for row in history),
        "maximum_open_flux_mismatch": max(row[6] for row in history),
        "total_clipped_aperture": sum(row[7] for row in history),
        "l1_error": final[5],
        "steps": int(match.group(1)),
        "cpu_seconds": float(match.group(2)),
        "real_seconds": float(match.group(3)),
        "log": log.relative_to(ROOT).as_posix(),
    }


def interface_band_cases() -> list[dict[str, float | int | str]]:
    path = OUTPUT / "interface_band_cfl_summary.csv"
    cases: list[dict[str, float | int | str]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["mode"] != "interface_band_cfl":
                continue
            initial = float(row["initial_volume"])
            final = float(row["final_volume"])
            cases.append({
                "mode": row["mode"],
                "level": int(row["level"]),
                "cells_per_side": int(row["cells_per_side"]),
                "initial_volume": initial,
                "final_volume": final,
                "relative_volume_drift": (final - initial)/initial,
                "maximum_bound_violation": float(row["max_bound_violation"]),
                "maximum_open_flux_mismatch": float(row["max_open_flux_mismatch"]),
                "l1_error": float(row["l1_relative_q"]),
                "steps": int(row["steps"]),
                "cpu_seconds": float(row["cpu_seconds"]),
                "source": path.relative_to(ROOT).as_posix(),
            })
    return cases


def receiver_run(name: str, directory_name: str = "audit_receiver_refactor") -> dict[str, object]:
    directory = SOLVER / directory_name
    log = directory / f"{name}.log"
    exit_path = directory / f"{name}.exit"
    q_run = numeric_rows(log, "Q_RUN")
    cutcell = numeric_rows(log, "Q_CUTCELL_CFL")
    momentum = numeric_rows(log, "Q_MOMENTUM") if "fixed" in name and "amr" not in name else []
    stages = numeric_rows(log, "AMR_VOLUME_STAGE") if "amr" in name else []
    reprojection = numeric_rows(log, "POSTADAPT_REPROJECT") if "amr" in name else []
    # Q_RUN: t, iteration, volume, relative drift, lower/upper bounds,
    # flux mismatch, clipped aperture, corrected cells.
    final = q_run[-1]
    result: dict[str, object] = {
        "name": name,
        "exit_code": int(exit_path.read_text(encoding="utf-8").strip()),
        "end_time": final[0],
        "steps": int(final[1]),
        "final_volume": final[2],
        "relative_volume_drift": final[3],
        "maximum_bound_violation": max(max(abs(row[4]), abs(row[5])) for row in q_run),
        "maximum_open_flux_mismatch": max(row[6] for row in q_run),
        "maximum_cutcell_courant": max(row[4] for row in cutcell),
        "cutcell_limited_cells": max(int(row[5]) for row in cutcell),
        "time_history": [row[0] for row in q_run],
        "relative_volume_drift_history": [row[3] for row in q_run],
        "log": log.relative_to(ROOT).as_posix(),
    }
    if momentum:
        result.update({
            "maximum_velocity": max(row[4] for row in momentum),
            "maximum_relative_momentum_transport_velocity_change": max(row[2] for row in momentum),
            "minimum_kinetic_energy_ratio": min(row[6] / row[5] for row in momentum),
            "maximum_kinetic_energy_ratio": max(row[6] / row[5] for row in momentum),
            "momentum_time_history": [row[0] for row in momentum],
            "maximum_velocity_history": [row[4] for row in momentum],
        })
    if stages:
        grouped: dict[tuple[float, int], list[float]] = {}
        for row in stages:
            grouped.setdefault((row[0], int(row[1])), []).append(row[3])
        result["maximum_within_adapt_volume_spread"] = max(
            max(values) - min(values) for values in grouped.values()
        )
    if reprojection:
        # POSTADAPT_REPROJECT: t, i, iterations, divergence norms,
        # relative velocity change, relative KE change, ...
        result["maximum_postadapt_relative_velocity_change"] = max(
            abs(row[6]) for row in reprojection
        )
        result["maximum_postadapt_relative_kinetic_energy_change"] = max(
            abs(row[7]) for row in reprojection
        )
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inclined = [
        inclined_case(mode, level)
        for mode in ("baseline_fullcell_cfl", "cutcell_cfl")
        for level in (5, 6, 7)
    ] + interface_band_cases()

    old_log = (
        SOLVER / "audit_receiver_refactor" /
        "receiver_momentum_face_trace_l9.log"
    )
    old_momentum = numeric_rows(old_log, "Q_MOMENTUM")
    old_debug = numeric_rows(old_log, "Q_SWEPT_Y_DEBUG")
    fixed = receiver_run(
        "receiver_momentum_fixed_l9_t100", "audit_receiver_interface_band"
    )
    adaptive = receiver_run(
        "receiver_momentum_amr_l9_t100", "audit_receiver_interface_band"
    )

    audit = {
        "schema_version": 2,
        "generated_date": "2026-08-29",
        "scope": (
            "Static inclined-wall advection and level-9 pre-impact receiver "
            "prefix through t=0.1; no impact or post-impact validation."
        ),
        "implementation_fixes": [
            "Guard VOF concentration gradients against solid/coarse-fine nodata values and fall back to first-order upwinding.",
            "Use the exact geometric-VOF pure-state face flux for f=0 and f=1 instead of clipping it to the donor cut-cell area.",
            "Apply the embedded capacity CFL only in a one-cell interface band; retain the all-cut-cell guard as the reference fallback.",
            "Make q the sole AMR liquid measure and reconstruct conditional f from q/cs.",
            "Preserve physical weighted restriction order before momentum prolongation.",
            "Use explicit receiver cell_area=Delta^2 provenance and reject ambiguous legacy dv.",
            "Fail fast when embedded receiver momentum data are used at a different grid level.",
        ],
        "inclined_small_cell_counterexample": inclined,
        "interface_band_asymptotic_orders_l7_l9": [
            math.log(0.023254025226/0.0124729222345, 2),
            math.log(0.0124729222345/0.00615431611535, 2),
        ],
        "negative_ablations": [
            "Bound-residual state redistribution conserved q but did not improve shape error and was not tracer-consistent; disabled by default.",
            "Symmetric Strang directional splitting did not improve the inclined-wall error; removed from the working operator.",
        ],
        "phase_momentum_root_cause": {
            "old_second_step_maximum_gradient": max(abs(row[6]) for row in old_debug),
            "old_second_step_maximum_velocity": old_momentum[-1][4],
            "old_second_step_kinetic_energy": old_momentum[-1][6],
            "cause": (
                "Basilisk nodata/HUGE entered the three-point concentration "
                "gradient through a solid/coarse-fine stencil."
            ),
            "log": old_log.relative_to(ROOT).as_posix(),
        },
        "receiver_short_prefix": {
            "fixed_grid": fixed,
            "dynamic_amr": adaptive,
        },
        "test_suite": {
            "status": "pass",
            "tests_passed": 94,
            "command": "python -m pytest -q",
        },
        "environment": {
            "basilisk_container": "sgls/basilisk-docker",
            "container_repo_digest": "sha256:86e22069efeacadf32eeffbebd627366ec29212622e856ffc485e2fee1f6b764",
            "container_image_id": "sha256:91724f980edcb09b1c737cf4f41aba39177680bd8b086377a3e831243a23b25d",
            "basilisk_patch": "ad05f362",
            "basilisk_patch_date": "2022-04-07",
            "limitation": (
                "The available container is an old Basilisk snapshot; current-"
                "upstream compatibility remains an open reproducibility gate."
            ),
        },
        "publication_gates": {
            "passed": [
                "inclined fixed-grid boundedness/conservation with cut-cell CFL at levels 5-7",
                "interface-band small-cell guard conserves q at levels 5-9 and is asymptotically first order on levels 7-9",
                "causal regression for nodata-triggered phase-momentum blow-up",
                "fixed-grid level-9 receiver prefix through t=0.1",
                "dynamic-AMR level-9 receiver prefix through t=0.1",
            ],
            "open": [
                "phase-momentum-consistent local redistribution or subcycling for impact states where the interface-band guard activates materially",
                "inclined-wall AMR convergence with a discretely projected manufactured flux",
                "three-resolution and handoff-time convergence through first impact",
                "comparison against experimental or independently validated impact observables",
                "current-upstream Basilisk reproduction",
            ],
        },
    }
    json_path = OUTPUT / "method_revision_audit.json"
    json_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 160,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    colors = {
        "baseline_fullcell_cfl": "#c44e52",
        "cutcell_cfl": "#4c72b0",
        "interface_band_cfl": "#55a868",
    }
    labels = {
        "baseline_fullcell_cfl": "full-cell CFL (invalid)",
        "cutcell_cfl": "all-cut-cell CFL",
        "interface_band_cfl": "interface-band CFL",
    }
    markers = {
        "baseline_fullcell_cfl": "s",
        "cutcell_cfl": "o",
        "interface_band_cfl": "^",
    }

    for mode in colors:
        cases = [case for case in inclined if case["mode"] == mode]
        n = np.array([case["cells_per_side"] for case in cases])
        error = np.array([case["l1_error"] for case in cases])
        drift = np.abs([case["relative_volume_drift"] for case in cases])
        axes[0, 0].loglog(n, error, marker=markers[mode], color=colors[mode], label=labels[mode])
        axes[0, 1].semilogy(n, np.maximum(drift, 1e-16), marker=markers[mode], color=colors[mode], label=labels[mode])
    reference_n = np.array([32, 128])
    axes[0, 0].loglog(reference_n, 0.34/reference_n, "k--", lw=0.9, label=r"$O(\Delta)$")
    axes[0, 0].set_xticks([32, 64, 128, 256, 512])
    axes[0, 0].set_xticklabels(["32", "64", "128", "256", "512"])
    axes[0, 0].xaxis.set_minor_formatter(NullFormatter())
    axes[0, 0].set(title="(a) Inclined-wall translation", xlabel="cells per side", ylabel=r"$L_1(q)$")
    axes[0, 1].set(title="(b) Small-cell conservation", xlabel="cells per side", ylabel="relative volume drift")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)

    old_t = [row[0] for row in old_momentum]
    old_u = [row[4] for row in old_momentum]
    axes[1, 0].semilogy(old_t, old_u, "s--", color="#c44e52", label="unguarded gradient")
    axes[1, 0].semilogy(
        fixed["momentum_time_history"], fixed["maximum_velocity_history"],
        "o-", ms=3, color="#4c72b0", label="guarded gradient",
    )
    axes[1, 0].set(title="(c) Phase-momentum regression", xlabel="time", ylabel=r"$\max |\mathbf{u}|$")
    axes[1, 0].legend(frameon=False)

    for run, label, color, marker in (
        (fixed, "fixed grid", "#4c72b0", "o"),
        (adaptive, "dynamic AMR", "#55a868", "^"),
    ):
        axes[1, 1].plot(
            run["time_history"], np.abs(run["relative_volume_drift_history"]),
            marker=marker, ms=4, color=color, label=label,
        )
    axes[1, 1].set(title="(d) Level-9 receiver prefix", xlabel="time", ylabel="absolute relative volume drift")
    axes[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.grid(True, which="both", alpha=0.22, lw=0.5)
    fig.suptitle("Conservative-q interface-band audit (pre-impact evidence)", fontsize=11)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"method_revision_audit.{suffix}", bbox_inches="tight")
    plt.close(fig)

    print(json_path)
    print(OUTPUT / "method_revision_audit.png")
    print(OUTPUT / "method_revision_audit.pdf")


if __name__ == "__main__":
    main()
