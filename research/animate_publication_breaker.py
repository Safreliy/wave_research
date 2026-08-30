"""Create a close-up, evidence-labelled GIF of shoaling and overturning."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from animate_lagrangian_bie import spectral_curve_resample
from animate_results import BACKGROUND, CREST, GRID, INK, MUTED, SURFACE, WATER, _font


def render_closeup(
    data: dict[str, np.ndarray],
    index: int,
    x_limits: tuple[float, float],
    z_limits: tuple[float, float],
) -> Image.Image:
    width, height = 1200, 720
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 92.0, 1150.0, 118.0, 590.0
    title_font = _font(29, bold=True)
    metric_font = _font(18, bold=True)
    label_font = _font(16)
    small_font = _font(14)
    draw.text(
        (left, 25),
        "Shoaling solitary wave · crest close-up",
        fill=INK,
        font=title_font,
    )
    flux = abs(float(data["surface_flux_defect"][index]))
    validated = flux <= 1.0e-3
    evidence = (
        "VALIDATED TRAJECTORY · flux gate passed"
        if validated
        else "EXPLORATORY OVERHANG · flux gate exceeded"
    )
    evidence_color = "#197a61" if validated else CREST
    draw.text((left, 72), evidence, fill=evidence_color, font=metric_font)

    x_min, x_max = x_limits
    z_min, z_max = z_limits
    map_x = lambda value: left + (value - x_min) / (x_max - x_min) * (right - left)
    map_z = lambda value: bottom - (value - z_min) / (z_max - z_min) * (bottom - top)
    draw.rectangle((left, top, right, bottom), fill="white", outline=GRID, width=2)
    for tick in np.linspace(x_min, x_max, 6):
        px = map_x(float(tick))
        draw.line((px, top, px, bottom), fill="#edf1f5", width=1)
        draw.text((px - 16, bottom + 8), f"{tick:.1f}", fill=MUTED, font=small_font)
    for tick in np.linspace(z_min, z_max, 5):
        pz = map_z(float(tick))
        draw.line((left, pz, right, pz), fill="#edf1f5", width=1)
        draw.text((25, pz - 8), f"{tick:.2f}", fill=MUTED, font=small_font)

    length = float(data["length"])
    surface_x, surface_z = spectral_curve_resample(
        data["x"][index], data["z"][index], length, refinement=12
    )
    bottom_x, bottom_z = spectral_curve_resample(
        data["bottom_x"], data["bottom_z"], length, refinement=12
    )
    surface_points = [
        (map_x(float(x)), map_z(float(z)))
        for x, z in zip(surface_x, surface_z)
    ]
    bottom_points = [
        (map_x(float(x)), map_z(float(z)))
        for x, z in zip(bottom_x, bottom_z)
    ]
    draw.polygon(surface_points + list(reversed(bottom_points)), fill=WATER)
    draw.line(bottom_points, fill="#7a6a50", width=5, joint="curve")
    draw.line(surface_points, fill=SURFACE, width=5, joint="curve")
    crest = int(np.argmax(surface_z))
    crest_point = surface_points[crest]
    draw.ellipse(
        (
            crest_point[0] - 6,
            crest_point[1] - 6,
            crest_point[0] + 6,
            crest_point[1] + 6,
        ),
        fill="#f59e0b",
        outline="white",
        width=2,
    )

    time = float(data["time"][index])
    mapping = float(data["min_x_alpha"][index])
    if mapping <= 0.0:
        stage = "overhanging crest"
    elif mapping < 0.5:
        stage = "near-vertical face"
    else:
        stage = "shoaling / steepening"
    energy_reference = float(data.get("energy_reference", np.asarray(0.0)))
    energy_drift = float(
        (data["energy"][index] - data["energy"][0])
        / max(abs(float(data["energy"][0]) - energy_reference), np.finfo(float).eps)
    )
    draw.text(
        (left, 632),
        f"t = {time:.3f} · {stage}",
        fill=evidence_color,
        font=metric_font,
    )
    draw.text(
        (left + 350, 632),
        f"min x_alpha = {mapping:+.3f}",
        fill=INK,
        font=label_font,
    )
    draw.text(
        (left + 645, 632),
        f"|flux defect| = {flux:.2e}",
        fill=evidence_color,
        font=label_font,
    )
    draw.text(
        (left, 675),
        f"Delta E/Ewave = {energy_drift:+.2e} · Euler–BIE, N={data['x'].shape[1]} · no post-impact claim",
        fill=MUTED,
        font=small_font,
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-time", type=float, default=9.0)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--x-min", type=float, default=17.5)
    parser.add_argument("--x-max", type=float, default=26.0)
    parser.add_argument("--z-min", type=float, default=-0.35)
    parser.add_argument("--z-max", type=float, default=0.82)
    args = parser.parse_args()
    with np.load(args.input) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    candidates = np.where(data["time"] >= args.start_time)[0]
    if len(candidates) < 2:
        raise ValueError("start-time leaves fewer than two frames")
    indices = np.linspace(candidates[0], len(data["time"]) - 1, args.frames, dtype=int)
    frames = [
        render_closeup(
            data,
            int(index),
            (args.x_min, args.x_max),
            (args.z_min, args.z_max),
        )
        for index in indices
    ]
    palette = frames[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    gif_frames = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in frames
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
    frame_directory = args.output.with_suffix("").with_name(
        args.output.stem + "_frames"
    )
    frame_directory.mkdir(parents=True, exist_ok=True)
    labels = ("shoaling", "steep", "vertical", "overhang")
    positions = np.linspace(0, len(frames) - 1, len(labels), dtype=int)
    for label, position in zip(labels, positions):
        frames[int(position)].save(frame_directory / f"{label}.png", optimize=True)


if __name__ == "__main__":
    main()
