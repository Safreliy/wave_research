"""Geometric target animation for a shoaling plunging breaker.

This animation demonstrates the geometry that a future parametric BIE or
Navier--Stokes solver must reproduce.  It is not obtained by integrating the
Euler/Navier--Stokes equations and is labelled accordingly in every frame.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from animate_results import (
    BACKGROUND,
    CREST,
    EVENT,
    GRID,
    INK,
    MUTED,
    SURFACE,
    WATER,
    _font,
)


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _bezier(
    start: tuple[float, float],
    control_1: tuple[float, float],
    control_2: tuple[float, float],
    end: tuple[float, float],
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, samples, endpoint=False)
    one_minus_t = 1.0 - t
    points = (
        one_minus_t[:, None] ** 3 * np.asarray(start)
        + 3.0 * one_minus_t[:, None] ** 2 * t[:, None] * np.asarray(control_1)
        + 3.0 * one_minus_t[:, None] * t[:, None] ** 2 * np.asarray(control_2)
        + t[:, None] ** 3 * np.asarray(end)
    )
    return points[:, 0], points[:, 1]


def breaker_curve(progress: float, samples: int = 700) -> dict[str, np.ndarray | float | str]:
    """Return a graph that continuously deforms into a C-shaped overhang."""
    shoaling = _smoothstep((progress - 0.08) / 0.72)
    plunging = _smoothstep((progress - 0.68) / 0.30)
    crest_center = 3.4 + 4.0 * progress
    amplitude = 0.17 + 0.39 * shoaling
    local_count = max(320, samples // 2)
    per_segment = local_count // 4

    # Final breaker geometry: rear face -> crest -> forward tip -> underside ->
    # front trough.  The third segment travels back in x, creating the overhang.
    segments = [
        _bezier((-1.50, -0.20), (-0.95, -0.14), (-0.42, 0.54), (-0.05, 0.60), per_segment),
        _bezier((-0.05, 0.60), (0.20, 0.72), (0.84, 0.58), (0.86, 0.25), per_segment),
        _bezier((0.86, 0.25), (0.88, 0.09), (0.61, -0.08), (0.28, -0.03), per_segment),
        _bezier((0.28, -0.03), (0.56, -0.15), (1.05, -0.25), (1.50, -0.20), per_segment + 1),
    ]
    target_x = np.concatenate([segment[0] for segment in segments])
    target_z = np.concatenate([segment[1] for segment in segments])
    vertical_scale = amplitude / 0.56
    target_z *= vertical_scale

    reference_local = np.linspace(-1.50, 1.50, len(target_x))
    graph_z = amplitude * np.cos(math.pi * reference_local / 1.50)
    local_x = (1.0 - plunging) * reference_local + plunging * target_x
    local_z = (1.0 - plunging) * graph_z + plunging * target_z

    left_x = np.linspace(0.0, crest_center - 1.50, max(120, samples // 4), endpoint=False)
    right_x = np.linspace(crest_center + 1.50, 11.0, max(120, samples // 4))
    left_z = local_z[0] * np.exp(-((left_x - (crest_center - 1.50)) / 0.95) ** 2)
    right_z = local_z[-1] * np.exp(-((right_x - (crest_center + 1.50)) / 1.05) ** 2)
    # A small secondary crest shoreward makes the shoaling context visible.
    right_z += 0.06 * shoaling * np.exp(
        -((right_x - (crest_center + 2.15)) / 0.58) ** 2
    )

    horizontal = np.concatenate((left_x, crest_center + local_x, right_x))
    vertical = np.concatenate((left_z, local_z, right_z))
    reference_x = np.concatenate(
        (left_x, crest_center + reference_local, right_x)
    )
    dx_ds = np.gradient(horizontal) / np.gradient(reference_x)
    minimum_mapping_jacobian = float(np.min(dx_ds))
    if minimum_mapping_jacobian > 0.20:
        stage = "shoaling"
    elif minimum_mapping_jacobian > 0.0:
        stage = "crest steepening"
    elif minimum_mapping_jacobian > -0.22:
        stage = "vertical tangent / onset of overhang"
    else:
        stage = "overturning and plunging jet"
    return {
        "x": horizontal,
        "z": vertical,
        "bottom_x": np.linspace(0.0, 11.0, 700),
        "bottom": -1.58 + 0.138 * np.linspace(0.0, 11.0, 700),
        "minimum_mapping_jacobian": minimum_mapping_jacobian,
        "ka": float((1.72 + 0.34 * shoaling) * amplitude),
        "stage": stage,
        "progress": progress,
    }


def render_frame(state: dict[str, np.ndarray | float | str]) -> Image.Image:
    width, height = 1200, 760
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    left, right = 90.0, 1150.0
    top, bottom = 118.0, 620.0
    title_font = _font(30, bold=True)
    label_font = _font(17)
    small_font = _font(15)
    metric_font = _font(18, bold=True)
    draw.text((left, 28), "Shoaling wave → plunging breaker", fill=INK, font=title_font)
    draw.text(
        (left, 72),
        "GEOMETRIC TARGET — not an Euler/Navier–Stokes solution",
        fill=CREST,
        font=small_font,
    )

    x_min, x_max = 0.0, 11.0
    z_min, z_max = -1.70, 0.82
    map_x = lambda value: left + (value - x_min) / (x_max - x_min) * (right - left)
    map_z = lambda value: bottom - (value - z_min) / (z_max - z_min) * (bottom - top)
    draw.rectangle((left, top, right, bottom), fill="white", outline=GRID, width=2)

    for x_tick in np.arange(0.0, 12.0, 2.0):
        px = map_x(float(x_tick))
        draw.line((px, top, px, bottom), fill="#edf1f5", width=1)
        draw.text((px - 8, bottom + 10), f"{x_tick:.0f}", fill=MUTED, font=small_font)
    for z_tick in (-1.5, -1.0, -0.5, 0.0, 0.5):
        pz = map_z(z_tick)
        draw.line((left, pz, right, pz), fill="#edf1f5", width=1)
        draw.text((24, pz - 9), f"{z_tick:.1f}", fill=MUTED, font=small_font)
    draw.line((left, map_z(0.0), right, map_z(0.0)), fill="#9dacba", width=2)

    surface_x = np.asarray(state["x"])
    surface_z = np.asarray(state["z"])
    bottom_x = np.asarray(state["bottom_x"])
    bed = np.asarray(state["bottom"])
    surface_points = [
        (map_x(float(x)), map_z(float(z))) for x, z in zip(surface_x, surface_z)
    ]
    bed_points = [
        (map_x(float(x)), map_z(float(z))) for x, z in zip(bottom_x[::-1], bed[::-1])
    ]
    water_polygon = surface_points + bed_points
    draw.polygon(water_polygon, fill=WATER)

    beach_points = [
        (map_x(float(x)), map_z(float(z))) for x, z in zip(bottom_x, bed)
    ] + [(right, bottom), (left, bottom)]
    draw.polygon(beach_points, fill="#d9c39a")
    draw.line(
        [(map_x(float(x)), map_z(float(z))) for x, z in zip(bottom_x, bed)],
        fill="#8a7656",
        width=4,
    )
    draw.line(surface_points, fill=SURFACE, width=5, joint="curve")

    crest_index = int(np.argmax(surface_z))
    crest_px = map_x(float(surface_x[crest_index]))
    crest_pz = map_z(float(surface_z[crest_index]))
    draw.ellipse(
        (crest_px - 7, crest_pz - 7, crest_px + 7, crest_pz + 7),
        fill=EVENT,
        outline="white",
        width=2,
    )
    draw.text((right - 180, bottom - 45), "sloping beach", fill="#725f40", font=label_font)

    stage = str(state["stage"])
    draw.text((left, 660), stage, fill=CREST, font=metric_font)
    draw.text(
        (left + 390, 660),
        f"kA = {float(state['ka']):.3f}",
        fill=INK,
        font=label_font,
    )
    jacobian = float(state["minimum_mapping_jacobian"])
    draw.text(
        (left + 590, 660),
        f"min(dx/ds) = {jacobian:+.3f}",
        fill=CREST if jacobian <= 0.0 else INK,
        font=label_font,
    )
    draw.text(
        (left, 699),
        "dx/ds < 0 means the interface has overhung and eta(x) is no longer valid",
        fill=MUTED,
        font=small_font,
    )
    progress = float(state["progress"])
    draw.line((left, 737, right, 737), fill=GRID, width=8)
    draw.line((left, 737, left + progress * (right - left), 737), fill=SURFACE, width=8)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output", type=Path, nargs="?", default=Path("results/plunging_breaker_target.gif")
    )
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=18)
    args = parser.parse_args()
    progress_values = np.linspace(0.0, 1.0, args.frames)
    states = [breaker_curve(float(progress)) for progress in progress_values]
    rgb_frames = [render_frame(state) for state in states]
    palette = rgb_frames[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    gif_frames = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in rgb_frames
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(
        args.output,
        save_all=True,
        append_images=gif_frames[1:],
        duration=round(1000 / args.fps),
        loop=0,
        disposal=2,
        optimize=False,
    )

    # Save stages useful for discussion and future validation figures.
    stage_indices = {
        "shoaling": 15,
        "steep": 50,
        "overhang": 80,
        "plunging": args.frames - 1,
    }
    frame_dir = args.output.with_suffix("").with_name(args.output.stem + "_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    for name, index in stage_indices.items():
        index = min(index, args.frames - 1)
        rgb_frames[index].save(frame_dir / f"{name}.png", optimize=True)


if __name__ == "__main__":
    main()
