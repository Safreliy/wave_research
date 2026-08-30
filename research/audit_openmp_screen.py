#!/usr/bin/env python3
"""Audit OpenMP scaling without treating parallel reduction drift as proof."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_THREADS = (1, 2, 4, 8)


def elapsed_seconds(text: str) -> float:
    match = re.search(r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)", text)
    if not match:
        raise ValueError("elapsed time is missing")
    parts = [float(part) for part in match.group(1).split(":")]
    if len(parts) == 2:
        return 60.0 * parts[0] + parts[1]
    if len(parts) == 3:
        return 3600.0 * parts[0] + 60.0 * parts[1] + parts[2]
    return parts[0]


def records(text: str, tag: str) -> list[list[float]]:
    found: list[list[float]] = []
    prefix = tag + " "
    for line in text.splitlines():
        if line.startswith(prefix):
            found.append([float(token) for token in line.split()[1:]])
    return found


def invert_jet(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    valid = np.any(rgb != 0.0, axis=2)
    lut = plt.colormaps["jet"](np.linspace(0.0, 1.0, 256))[:, :3] * 255.0
    distance = np.sum((rgb[:, :, None, :] - lut[None, None, :, :]) ** 2, axis=3)
    scalar = np.argmin(distance, axis=2).astype(np.float64) / 255.0
    scalar[~valid] = np.nan
    return rgb, scalar, valid


def component_count(mask: np.ndarray) -> int:
    visited = np.zeros(mask.shape, dtype=bool)
    count = 0
    height, width = mask.shape
    for row in range(height):
        for column in range(width):
            if not mask[row, column] or visited[row, column]:
                continue
            count += 1
            queue = deque([(row, column)])
            visited[row, column] = True
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and mask[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
    return count


def surface_rows(liquid: np.ndarray) -> np.ndarray:
    result = np.full(liquid.shape[1], np.nan)
    for column in range(liquid.shape[1]):
        rows = np.flatnonzero(liquid[:, column])
        if rows.size:
            result[column] = float(rows.min())
    return result


def image_metrics(reference_path: Path, candidate_path: Path) -> dict[str, float | int]:
    ref_rgb, ref_scalar, ref_valid = invert_jet(reference_path)
    rgb, scalar, valid = invert_jet(candidate_path)
    common = ref_valid & valid
    liquid_ref = ref_valid & (ref_scalar >= 0.5)
    liquid = valid & (scalar >= 0.5)
    mismatch = liquid_ref ^ liquid
    ref_surface = surface_rows(liquid_ref)
    surface = surface_rows(liquid)
    columns = np.isfinite(ref_surface) & np.isfinite(surface)
    row_delta = surface[columns] - ref_surface[columns]
    return {
        "rgb_rmse_0_1": float(np.sqrt(np.mean(((rgb - ref_rgb) / 255.0) ** 2))),
        "scalar_rmse": float(np.sqrt(np.nanmean((scalar[common] - ref_scalar[common]) ** 2))),
        "liquid_mask_mismatch_fraction": float(np.mean(mismatch)),
        "liquid_components": component_count(liquid),
        "surface_row_rmse": float(np.sqrt(np.mean(row_delta**2))) if row_delta.size else math.nan,
        "surface_row_max_abs": float(np.max(np.abs(row_delta))) if row_delta.size else math.nan,
    }


def audit_run(root: Path, threads: int) -> dict[str, object]:
    run_dir = root / f"omp_{threads}"
    text = (run_dir / "diagnostics_and_time.log").read_text(encoding="utf-8")
    q_run = max(records(text, "Q_RUN"), key=lambda values: values[0])
    run = max(records(text, "RUN"), key=lambda values: values[0])
    reprojection = records(text, "POSTADAPT_REPROJECT")
    q_momentum = records(text, "Q_MOMENTUM")
    elapsed = elapsed_seconds(text)
    cpu_match = re.search(r"Percent of CPU this job got:\s*([0-9.]+)%", text)
    rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    unsafe = re.search(r"(^|[^A-Za-z])(nan|[-+]?inf|fpe)([^A-Za-z]|$)", text, re.I)
    return {
        "threads": threads,
        "exit_status": int((run_dir / "exit_status.txt").read_text().strip()),
        "elapsed_seconds": elapsed,
        "cpu_percent": float(cpu_match.group(1)) if cpu_match else math.nan,
        "max_rss_kib": int(rss_match.group(1)) if rss_match else -1,
        "final_time": run[0],
        "final_iteration": int(run[1]),
        "final_mass": q_run[2],
        "final_relative_mass_drift": q_run[3],
        "max_q_lower_violation": max(row[4] for row in records(text, "Q_RUN")),
        "max_q_upper_violation": max(row[5] for row in records(text, "Q_RUN")),
        "final_kinetic": run[3],
        "max_post_divergence_rms": max(row[4] for row in reprojection),
        "max_post_divergence_linf": max(row[5] for row in reprojection),
        "max_relative_momentum_correction": max(row[6] for row in reprojection),
        "max_relative_energy_correction": max(row[7] for row in reprojection),
        "max_q_momentum_velocity_change": max(row[2] for row in q_momentum),
        "finite_log": unsafe is None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--threads", type=int, nargs="+")
    args = parser.parse_args()
    root = args.root.resolve()
    threads_used = tuple(args.threads or DEFAULT_THREADS)
    if 1 not in threads_used:
        raise ValueError("the one-thread reference is required")
    runs = [audit_run(root, threads) for threads in threads_used]
    reference = runs[0]
    reference_frame = root / "omp_1" / "handoff_frames" / "vof-00002.ppm"
    reference_vorticity = root / "omp_1" / "handoff_vorticity" / "vorticity-00002.ppm"
    for run in runs:
        threads = int(run["threads"])
        run["speedup_vs_1"] = float(reference["elapsed_seconds"]) / float(run["elapsed_seconds"])
        run["parallel_efficiency"] = float(run["speedup_vs_1"]) / threads
        run["relative_mass_difference_vs_1"] = abs(
            float(run["final_mass"]) - float(reference["final_mass"])
        ) / abs(float(reference["final_mass"]))
        run["relative_kinetic_difference_vs_1"] = abs(
            float(run["final_kinetic"]) - float(reference["final_kinetic"])
        ) / abs(float(reference["final_kinetic"]))
        run["vof_geometry"] = image_metrics(
            reference_frame,
            root / f"omp_{threads}" / "handoff_frames" / "vof-00002.ppm",
        )
        vort_ref = np.asarray(Image.open(reference_vorticity), dtype=np.float64)
        vort = np.asarray(
            Image.open(root / f"omp_{threads}" / "handoff_vorticity" / "vorticity-00002.ppm"),
            dtype=np.float64,
        )
        run["vorticity_rgb_rmse_0_1"] = float(
            np.sqrt(np.mean(((vort - vort_ref) / 255.0) ** 2))
        )

    (root / "openmp_screen_summary.json").write_text(
        json.dumps({"reference_threads": 1, "runs": runs}, indent=2) + "\n",
        encoding="utf-8",
    )
    flat_rows = []
    for run in runs:
        row = {key: value for key, value in run.items() if key != "vof_geometry"}
        row.update({f"vof_{key}": value for key, value in run["vof_geometry"].items()})
        flat_rows.append(row)
    with (root / "openmp_screen_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    threads = np.array(threads_used)
    elapsed = np.array([float(run["elapsed_seconds"]) for run in runs])
    speedup = np.array([float(run["speedup_vs_1"]) for run in runs])
    axes[0].plot(threads, elapsed, "o-", color="#2457a6")
    axes[0].set(xlabel="OpenMP threads", ylabel="Wall time [s]", xticks=threads)
    axes[0].grid(alpha=0.25)
    axes[1].plot(threads, speedup, "o-", label="measured", color="#b23a48")
    axes[1].plot(threads, threads, "--", label="ideal", color="0.45")
    axes[1].set(xlabel="OpenMP threads", ylabel="Speedup", xticks=threads)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(root / "openmp_scaling.png", dpi=220)
    plt.close(fig)

    best = min(runs, key=lambda run: float(run["elapsed_seconds"]))
    lines = [
        "# OpenMP screening",
        "",
        "This is a short performance/equivalence screen, not an L11 physics or convergence result.",
        "",
        "| threads | wall [s] | speedup | mass drift | rel. kinetic vs 1 | post-div RMS | post-div Linf | VOF mask mismatch | components |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        geometry = run["vof_geometry"]
        lines.append(
            f"| {run['threads']} | {run['elapsed_seconds']:.3f} | {run['speedup_vs_1']:.3f} "
            f"| {run['final_relative_mass_drift']:.3e} | {run['relative_kinetic_difference_vs_1']:.3e} "
            f"| {run['max_post_divergence_rms']:.3e} | {run['max_post_divergence_linf']:.3e} "
            f"| {geometry['liquid_mask_mismatch_fraction']:.3e} | {geometry['liquid_components']} |"
        )
    lines += [
        "",
        f"Fastest observed configuration: {best['threads']} threads ({best['speedup_vs_1']:.2f}x).",
        "The trajectories are not bitwise identical: OpenMP reduction order produces a measurable kinetic-energy spread. A longer fixed-threshold L10/L11 equivalence run is required before using the fastest setting for publication data.",
    ]
    (root / "openmp_screen_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
