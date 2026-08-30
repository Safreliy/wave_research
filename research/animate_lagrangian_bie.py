"""Render the pre-impact portion of a parametric Euler--BIE simulation."""

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


def spectral_curve_resample(
    x: np.ndarray, z: np.ndarray, length: float, refinement: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the periodic spectral curve without drawing marker-to-marker corners."""
    n = len(x)
    count = refinement * n
    alpha = 2.0 * math.pi * np.arange(count) / count
    modes = np.fft.fftfreq(n, d=1.0 / n)
    phase = np.exp(1j * alpha[:, None] * modes[None, :])
    base = length * np.arange(n) / n
    displacement_coefficients = np.fft.fft(x - base) / n
    z_coefficients = np.fft.fft(z) / n
    dense_x = length * alpha / (2.0 * math.pi) + (
        phase @ displacement_coefficients
    ).real
    dense_z = (phase @ z_coefficients).real
    return dense_x, dense_z


def render(data: dict[str, np.ndarray], index: int, y_min: float, y_max: float) -> Image.Image:
    width, height = 1200, 760
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    left, right = 95.0, 1150.0
    top, bottom = 120.0, 610.0
    title_font = _font(29, bold=True)
    label_font = _font(17)
    small_font = _font(15)
    metric_font = _font(18, bold=True)
    draw.text((left, 28), "Pre-impact overturning · parametric Euler–BIE", fill=INK, font=title_font)
    has_topography = "bottom_x" in data and "bottom_z" in data
    bed_label = "smooth periodic shoal" if has_topography else "flat bed"
    draw.text(
        (left, 72),
        f"PRE-IMPACT EULER: {bed_label} · singularity-subtracted DNO · theta-s ALE gauge",
        fill=CREST,
        font=small_font,
    )

    length = float(data["length"])
    depth = float(data["depth"])
    map_x = lambda value: left + value / length * (right - left)
    map_z = lambda value: bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
    draw.rectangle((left, top, right, bottom), fill="white", outline=GRID, width=2)
    for tick in np.linspace(0.0, length, 7):
        px = map_x(float(tick))
        draw.line((px, top, px, bottom), fill="#edf1f5", width=1)
        draw.text((px - 12, bottom + 9), f"{tick:.1f}", fill=MUTED, font=small_font)
    for tick in np.linspace(-depth, y_max, 5):
        pz = map_z(float(tick))
        draw.line((left, pz, right, pz), fill="#edf1f5", width=1)
        draw.text((23, pz - 9), f"{tick:.2f}", fill=MUTED, font=small_font)

    surface_x, surface_z = spectral_curve_resample(
        data["x"][index], data["z"][index], length
    )
    points = [(map_x(float(x)), map_z(float(z))) for x, z in zip(surface_x, surface_z)]
    if has_topography:
        dense_bottom_x, dense_bottom_z = spectral_curve_resample(
            data["bottom_x"], data["bottom_z"], length
        )
        bed_points = [
            (map_x(float(x)), map_z(float(z)))
            for x, z in zip(dense_bottom_x, dense_bottom_z)
        ]
    else:
        bed_points = [(left, map_z(-depth)), (right, map_z(-depth))]
    water_polygon = points + list(reversed(bed_points))
    draw.polygon(water_polygon, fill=WATER)
    draw.line(bed_points, fill="#7a6a50", width=5, joint="curve")
    draw.line(points, fill=SURFACE, width=5, joint="curve")
    marker_index = int(np.argmax(surface_z))
    marker_x, marker_z = points[marker_index]
    draw.ellipse(
        (marker_x - 7, marker_z - 7, marker_x + 7, marker_z + 7),
        fill=EVENT,
        outline="white",
        width=2,
    )

    time = float(data["time"][index])
    energy_drift = (
        float(data["energy"][index]) - float(data["energy"][0])
    ) / max(
        abs(
            float(data["energy"][0])
            - float(data.get("energy_reference", np.asarray(0.0)))
        ),
        np.finfo(float).eps,
    )
    volume_drift = float(data["volume"][index]) - float(data["volume"][0])
    mapping = float(data["min_x_alpha"][index])
    if mapping > 0.15:
        stage = "steepening"
    elif mapping > 0.0:
        stage = "near-vertical front"
    else:
        stage = "overhanging interface"
    draw.text((left, 655), f"t = {time:.3f} · {stage}", fill=CREST, font=metric_font)
    draw.text(
        (left + 360, 655),
        f"min x_alpha = {mapping:+.3f}",
        fill=CREST if mapping <= 0.0 else INK,
        font=label_font,
    )
    draw.text(
        (left + 650, 655),
        f"Delta E/Ewave = {energy_drift:+.2e}",
        fill=CREST if abs(energy_drift) > 0.02 else INK,
        font=label_font,
    )
    draw.text(
        (left, 697),
        f"Delta V = {volume_drift:+.2e} · BIE residual = {float(data['bie_residual'][index]):.2e}",
        fill=MUTED,
        font=small_font,
    )
    progress = index / max(len(data["time"]) - 1, 1)
    draw.line((left, 735, right, 735), fill=GRID, width=8)
    draw.line((left, 735, left + progress * (right - left), 735), fill=SURFACE, width=8)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--max-energy-drift", type=float, default=0.06)
    args = parser.parse_args()
    with np.load(args.input) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    relative_energy = np.abs(
        (data["energy"] - data["energy"][0])
        / max(abs(float(data["energy"][0])), np.finfo(float).eps)
    )
    violations = np.where(relative_energy > args.max_energy_drift)[0]
    last = int(violations[0] - 1) if len(violations) else len(relative_energy) - 1
    if last < 1:
        raise ValueError("no admissible animation interval under the energy threshold")
    # Never jump across a bad interval even if a non-conservative error happens
    # to cross back below the scalar threshold later.
    overhang = np.where(data["min_x_alpha"] <= 0.0)[0]
    indices = np.linspace(0, last, args.frames, dtype=int)
    bottom_minimum = (
        float(np.min(data["bottom_z"]))
        if "bottom_z" in data
        else -float(data["depth"])
    )
    y_min = min(bottom_minimum, float(np.min(data["z"][indices]))) - 0.08
    y_max = float(np.max(data["z"][indices])) * 1.25 + 0.05
    rgb_frames = [render(data, int(index), y_min, y_max) for index in indices]
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
    frame_dir = args.output.with_suffix("").with_name(args.output.stem + "_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    rgb_frames[-1].save(frame_dir / "last_admissible.png", optimize=True)
    if len(overhang) and overhang[0] <= last:
        position = int(np.argmin(np.abs(indices - overhang[0])))
        rgb_frames[position].save(frame_dir / "first_overhang.png", optimize=True)


if __name__ == "__main__":
    main()
