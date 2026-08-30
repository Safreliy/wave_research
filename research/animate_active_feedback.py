"""Render accepted active-vortex-sheet continuation states as publication GIFs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


Array = np.ndarray


def _load(paths: list[Path]) -> list[dict[str, Array]]:
    frames: list[dict[str, Array]] = []
    last_time = -np.inf
    for path in paths:
        with np.load(path) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        for index, time in enumerate(np.asarray(data["time"], dtype=float)):
            if float(time) <= last_time + 1.0e-12:
                continue
            count = int(data["count"][index])
            frame = {
                "time": np.asarray(float(time)),
                "x": np.asarray(data["x"][index, :count], dtype=float),
                "z": np.asarray(data["z"][index, :count], dtype=float),
                "gamma": np.asarray(
                    data["panel_circulation"][index, :count], dtype=float
                ),
                "count": np.asarray(count),
                "min_x_alpha": np.asarray(float(data["min_x_alpha"][index])),
                "gap": np.asarray(float(data["normalized_impact_distance"][index])),
                "flux": np.asarray(float(data["surface_flux_defect"][index])),
                "projection": np.asarray(
                    float(data["projection_relative_correction"][index])
                ),
                "circulation_defect": np.asarray(
                    float(data["circulation_total_defect"][index])
                ),
                "accepted_dt": np.asarray(
                    float(data["accepted_dt"][index])
                    if "accepted_dt" in data
                    else 0.0
                ),
            }
            frames.append(frame)
            last_time = float(time)
    if not frames:
        raise ValueError("no non-duplicate frames were loaded")
    return frames


def _window(frames: list[dict[str, Array]]) -> tuple[float, float, float, float]:
    crest_positions = [
        float(frame["x"][int(np.argmax(frame["z"]))]) for frame in frames
    ]
    center = 0.5 * (min(crest_positions) + max(crest_positions))
    maximum = max(float(np.max(frame["z"])) for frame in frames)
    minimum = min(float(np.min(frame["z"])) for frame in frames)
    return center - 3.3, center + 4.6, min(-0.10, minimum - 0.05), maximum + 0.16


def render(
    frames: list[dict[str, Array]],
    output: Path,
    representation: str,
    fps: int,
) -> None:
    if representation not in {"smooth", "vortices"}:
        raise ValueError("representation must be smooth or vortices")
    x_min, x_max, z_min, z_max = _window(frames)
    gamma_scale = max(
        float(
            np.quantile(
                np.concatenate([np.abs(frame["gamma"]) for frame in frames]),
                0.985,
            )
        ),
        1.0e-14,
    )
    figure, axis = plt.subplots(figsize=(12.0, 7.2), constrained_layout=True)
    figure.patch.set_facecolor("#f5f7fb")
    axis.set_facecolor("#fbfcfe")

    def draw(frame_index: int) -> None:
        axis.clear()
        frame = frames[frame_index]
        x = frame["x"]
        z = frame["z"]
        gamma = frame["gamma"]
        inside = (x >= x_min - 0.4) & (x <= x_max + 0.4)
        selected = np.flatnonzero(inside)
        if len(selected) < 3:
            selected = np.arange(len(x))
        local_x = x[selected]
        local_z = z[selected]
        local_gamma = gamma[selected]
        polygon_x = np.r_[local_x, local_x[-1], local_x[0]]
        polygon_z = np.r_[local_z, z_min, z_min]
        axis.fill(polygon_x, polygon_z, color="#55acd1", alpha=0.72, zorder=1)
        if representation == "smooth":
            axis.plot(
                local_x,
                local_z,
                color="#123a57",
                linewidth=3.0,
                solid_capstyle="round",
                zorder=3,
            )
            subtitle = "Smooth rendering of the computed free-surface geometry"
        else:
            axis.plot(
                local_x,
                local_z,
                color="#8c98a8",
                linewidth=0.75,
                alpha=0.55,
                zorder=2,
            )
            colors = np.where(local_gamma >= 0.0, "#e34a3b", "#136f8a")
            sizes = 8.0 + 74.0 * np.sqrt(
                np.clip(np.abs(local_gamma) / gamma_scale, 0.0, 1.0)
            )
            axis.scatter(
                local_x,
                local_z,
                c=colors,
                s=sizes,
                edgecolors="none",
                alpha=0.88,
                zorder=4,
            )
            subtitle = "Discrete rendering: signed active panel circulations"
        vertical = float(frame["min_x_alpha"]) <= 0.0
        regime = "OVERTURNING" if vertical else "single-valued front"
        axis.set_title(
            "Active adaptive boundary-vortex-sheet evolution\n"
            + subtitle,
            loc="left",
            fontsize=15,
            fontweight="bold",
            color="#172235",
        )
        axis.text(
            0.01,
            0.97,
            r"$\Gamma_j$ is advanced at every step; accepted r/p-remeshes feed the next BIE solve",
            transform=axis.transAxes,
            va="top",
            fontsize=10.5,
            color="#b33a1b",
        )
        axis.text(
            0.01,
            0.035,
            (
                f"t={float(frame['time']):.5f}   N={int(frame['count'])}   "
                f"{regime}   min x_alpha={float(frame['min_x_alpha']):+.3e}\n"
                f"gap/ds={float(frame['gap']):.3f}   "
                f"|flux defect|={abs(float(frame['flux'])):.2e}   "
                f"projection={float(frame['projection']):.2e}   "
                f"sum Gamma defect={float(frame['circulation_defect']):+.1e}"
            ),
            transform=axis.transAxes,
            va="bottom",
            fontsize=9.5,
            color="#29445a",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#ccd6e2",
                "alpha": 0.92,
            },
        )
        if representation == "vortices":
            axis.text(
                0.99,
                0.97,
                "red: Gamma > 0    blue: Gamma < 0\nmarker area scales with |Gamma|",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=9.5,
                color="#4c596b",
            )
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(z_min, z_max)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(r"$x/h_0$")
        axis.set_ylabel(r"$z/h_0$")
        axis.grid(color="#b8c6d4", alpha=0.22, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    animation = FuncAnimation(
        figure,
        draw,
        frames=len(frames),
        interval=1000 / fps,
        repeat=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--smooth-output", required=True, type=Path)
    parser.add_argument("--vortex-output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=7)
    args = parser.parse_args()
    frames = _load(args.inputs)
    render(frames, args.smooth_output, "smooth", args.fps)
    render(frames, args.vortex_output, "vortices", args.fps)
    print(f"rendered {len(frames)} accepted states")


if __name__ == "__main__":
    main()
