"""Animate the verified active-Gamma representation with adaptive carriers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from active_vortex_sheet import ActiveVortexSheetState, potential_to_panel_circulation
from adaptive_vortex_mesh import (
    remesh_active_vortex_sheet,
    suggest_vortex_count,
    vortex_mesh_monitor,
)


INK = "#192338"
MUTED = "#4b607c"
WATER = "#52abd0"
BED = "#c8b07e"
POSITIVE = "#ef4b45"
NEGATIVE = "#176b87"
BACKGROUND = "#f3f7fb"


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _midpoints(state: ActiveVortexSheetState, length: float):
    next_x = np.roll(state.x, -1)
    next_x[-1] += length
    return 0.5 * (state.x + next_x), 0.5 * (state.z + np.roll(state.z, -1))


def _frame(
    state: ActiveVortexSheetState,
    bottom_x: np.ndarray,
    bottom_z: np.ndarray,
    length: float,
    time: float,
    source_count: int,
    scale: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Image.Image:
    width, height = 1200, 720
    left, right, top, bottom = 50, width - 50, 125, height - 95
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title = _font(29, True)
    label = _font(17, False)
    small = _font(14, False)

    def map_x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    def map_z(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    draw.text((left, 24), "Active adaptive boundary-vortex-sheet evolution", fill=INK, font=title)
    draw.text(
        (left, 68),
        "Gamma is the evolved panel state; carrier insertion/removal preserves cumulative circulation",
        fill="#b54708",
        font=label,
    )
    draw.text(
        (left, 94),
        "verified state-equivalent pre-contact replay; no post-contact reconnection law",
        fill=MUTED,
        font=small,
    )

    surface = [(map_x(float(x)), map_z(float(z))) for x, z in zip(state.x, state.z)]
    bed_points = [
        (map_x(float(x)), map_z(float(z)))
        for x, z in zip(bottom_x, bottom_z)
        if x_min <= x <= x_max
    ]
    visible_surface = [point for point, x in zip(surface, state.x) if x_min <= x <= x_max]
    if visible_surface and bed_points:
        polygon = visible_surface + list(reversed(bed_points))
        draw.polygon(polygon, fill=WATER)
        draw.line(bed_points, fill="#75664d", width=5, joint="curve")
    if len(visible_surface) > 1:
        draw.line(visible_surface, fill="#14556d", width=3, joint="curve")

    vortex_x, vortex_z = _midpoints(state, length)
    for px, pz, gamma in zip(vortex_x, vortex_z, state.panel_circulation):
        if not (x_min <= px <= x_max and y_min <= pz <= y_max):
            continue
        radius = 2.0 + 6.5 * math.sqrt(min(abs(float(gamma)) / scale, 1.0))
        cx, cy = map_x(float(px)), map_z(float(pz))
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=POSITIVE if gamma >= 0.0 else NEGATIVE,
        )
    draw.text(
        (left, height - 65),
        f"t={time:.3f}   source N={source_count}   active adaptive N={len(state.x)}   "
        f"sum Gamma={state.total_circulation:+.2e}",
        fill=MUTED,
        font=label,
    )
    draw.text((width - 305, 32), "red/blue: signed panel circulation", fill=POSITIVE, font=small)
    return image


def animate(
    source: Path,
    output: Path,
    frame_count: int,
    fps: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> dict[str, object]:
    with np.load(source) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    length = float(data["length"])
    background = float(data.get("background_current", np.asarray(0.0)))
    all_indices = np.linspace(0, len(data["time"]) - 1, frame_count).round().astype(int)
    all_indices = np.unique(all_indices)
    records: list[dict[str, object]] = []
    states: list[ActiveVortexSheetState] = []
    gamma_values: list[np.ndarray] = []
    for index in all_indices:
        x = np.asarray(data["x"][index], dtype=float)
        z = np.asarray(data["z"][index], dtype=float)
        potential = np.asarray(data["potential"][index], dtype=float)
        state = ActiveVortexSheetState(
            x,
            z,
            potential_to_panel_circulation(x, potential, length, background),
            float(np.mean(potential)),
        )
        monitor = vortex_mesh_monitor(state, length)
        decision = suggest_vortex_count(
            monitor, len(x), max(64, len(x) // 2), 2 * len(x)
        )
        adapted, diagnostics = remesh_active_vortex_sheet(
            state,
            length,
            background,
            target_count=decision.target_count,
        )
        states.append(adapted)
        gamma_values.append(adapted.panel_circulation)
        records.append(
            {
                "source_index": int(index),
                "time": float(data["time"][index]),
                "source_count": len(x),
                "adaptive_count": len(adapted.x),
                "action": decision.action,
                "turning": monitor.maximum_turning_angle,
                "minimum_nonlocal_gap_ratio": monitor.minimum_nonlocal_gap_ratio,
                "maximum_circulation_fraction": monitor.maximum_circulation_fraction,
                "circulation_defect": diagnostics.total_circulation_defect,
                "monitor_mass_cv_before": diagnostics.monitor_mass_cv_before,
                "monitor_mass_cv_after": diagnostics.monitor_mass_cv_after,
            }
        )
    scale = max(
        float(np.quantile(np.abs(np.concatenate(gamma_values)), 0.98)),
        1.0e-14,
    )
    frames = [
        _frame(
            state,
            np.asarray(data["bottom_x"]),
            np.asarray(data["bottom_z"]),
            length,
            record["time"],
            record["source_count"],
            scale,
            x_min,
            x_max,
            y_min,
            y_max,
        )
        for state, record in zip(states, records)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, int(round(1000 / fps))),
        loop=0,
        disposal=2,
    )
    report = {
        "schema": "active-vortex-sheet-animation-v1",
        "evidence_class": "verified state-equivalent pre-contact replay with conservative adaptive carrier visualization",
        "source": str(source),
        "frame_count": len(frames),
        "time_interval": [records[0]["time"], records[-1]["time"]],
        "records": records,
        "claim_boundary": "the active Gamma formulation is RHS/step-equivalent and each remesh preserves circulation; the stored geometry was not recomputed with remesh feedback",
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--x-min", type=float, default=28.0)
    parser.add_argument("--x-max", type=float, default=39.0)
    parser.add_argument("--y-min", type=float, default=-0.05)
    parser.add_argument("--y-max", type=float, default=0.9)
    args = parser.parse_args()
    report = animate(
        args.source,
        args.output,
        args.frames,
        args.fps,
        args.x_min,
        args.x_max,
        args.y_min,
        args.y_max,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
