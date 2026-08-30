"""Animate a manufactured moving-geometry verification of periodic_bie.

This is intentionally labelled as an operator test, not a time integration of
the water-wave equations.  Every frame rebuilds and solves the second-kind BIE
and compares its DNO with an exact harmonic field.
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
    _font,
)
from periodic_bie import PeriodicMirrorBIE


def _map(value: float, source_min: float, source_max: float, target_min: float, target_max: float) -> float:
    return target_min + (value - source_min) / (source_max - source_min) * (
        target_max - target_min
    )


def compute_frames(n: int, frame_count: int) -> list[dict[str, np.ndarray | float]]:
    length = 2.0 * math.pi
    depth = 1.0
    x = np.arange(n) * length / n
    frames = []
    for phase in np.linspace(0.0, 2.0 * math.pi, frame_count, endpoint=False):
        shift = phase / 2.0
        eta = 0.08 * np.cos(x - shift) + 0.025 * np.cos(2.0 * (x - shift) + 0.3)
        solver = PeriodicMirrorBIE(eta, length=length, depth=depth)
        k = 2.0
        angle = k * x - phase
        height = eta + depth
        normalization = math.cosh(k * depth)
        trace = np.cos(angle) * np.cosh(k * height) / normalization
        phi_x = -k * np.sin(angle) * np.cosh(k * height) / normalization
        phi_z = k * np.cos(angle) * np.sinh(k * height) / normalization
        exact = phi_z - solver.eta_x * phi_x
        result = solver.dirichlet_to_neumann(trace)
        error = float(np.linalg.norm(result.dno - exact) / np.linalg.norm(exact))
        frames.append(
            {
                "x": x,
                "eta": eta,
                "mirror": -2.0 * depth - eta,
                "computed": result.dno,
                "exact": exact,
                "phase": float(phase),
                "error": error,
                "residual": result.boundary_residual,
            }
        )
    return frames


def render(frame: dict[str, np.ndarray | float], n: int) -> Image.Image:
    width, height = 1200, 760
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    left, right = 105.0, 1145.0
    title_font = _font(29, bold=True)
    label_font = _font(17)
    small_font = _font(15)
    metric_font = _font(18, bold=True)
    draw.text((left, 30), "Second-kind BIE · moving manufactured geometry", fill=INK, font=title_font)
    draw.text(
        (left, 70),
        "Operator verification only — not a water-wave time integration",
        fill=CREST,
        font=small_font,
    )

    x = np.asarray(frame["x"])
    eta = np.asarray(frame["eta"])
    mirror = np.asarray(frame["mirror"])
    exact = np.asarray(frame["exact"])
    computed = np.asarray(frame["computed"])
    map_x = lambda value: _map(value, 0.0, 2.0 * math.pi, left, right)

    geo_top, geo_bottom = 112.0, 338.0
    map_geo = lambda value: _map(value, -2.15, 0.15, geo_bottom, geo_top)
    draw.rectangle((left, geo_top, right, geo_bottom), fill="white", outline=GRID, width=2)
    surface_points = [(map_x(float(xi)), map_geo(float(zi))) for xi, zi in zip(x, eta)]
    mirror_points = [(map_x(float(xi)), map_geo(float(zi))) for xi, zi in zip(x, mirror)]
    draw.line(surface_points, fill=SURFACE, width=4, joint="curve")
    draw.line(mirror_points, fill="#7a4fa3", width=3, joint="curve")
    bed_y = map_geo(-1.0)
    draw.line((left, bed_y, right, bed_y), fill=EVENT, width=2)
    draw.text((right - 150, bed_y - 25), "Neumann bed", fill=EVENT, font=small_font)
    draw.text((left + 12, geo_top + 10), "surface", fill=SURFACE, font=small_font)
    draw.text((left + 12, geo_bottom - 30), "mirror", fill="#7a4fa3", font=small_font)

    dno_top, dno_bottom = 415.0, 625.0
    d_extent = 1.12 * max(float(np.max(np.abs(exact))), float(np.max(np.abs(computed))))
    map_dno = lambda value: _map(value, -d_extent, d_extent, dno_bottom, dno_top)
    draw.rectangle((left, dno_top, right, dno_bottom), fill="white", outline=GRID, width=2)
    draw.line((left, map_dno(0.0), right, map_dno(0.0)), fill="#9dacba", width=1)
    exact_points = [(map_x(float(xi)), map_dno(float(yi))) for xi, yi in zip(x, exact)]
    computed_points = [(map_x(float(xi)), map_dno(float(yi))) for xi, yi in zip(x, computed)]
    draw.line(exact_points, fill=INK, width=4, joint="curve")
    draw.line(computed_points, fill=CREST, width=2, joint="curve")
    draw.text((left, dno_top - 32), "Dirichlet-to-Neumann map", fill=MUTED, font=label_font)
    draw.line((right - 230, dno_top - 22, right - 195, dno_top - 22), fill=INK, width=4)
    draw.text((right - 185, dno_top - 32), "exact", fill=INK, font=small_font)
    draw.line((right - 105, dno_top - 22, right - 70, dno_top - 22), fill=CREST, width=2)
    draw.text((right - 60, dno_top - 32), "BIE", fill=CREST, font=small_font)

    draw.text((left, 672), f"N = {n}", fill=INK, font=metric_font)
    draw.text(
        (left + 180, 672),
        f"relative DNO error = {float(frame['error']):.3e}",
        fill=CREST,
        font=metric_font,
    )
    draw.text(
        (left + 550, 672),
        f"BIE residual = {float(frame['residual']):.2e}",
        fill=INK,
        font=label_font,
    )
    phase_fraction = float(frame["phase"]) / (2.0 * math.pi)
    draw.line((left, 725, right, 725), fill=GRID, width=8)
    draw.line((left, 725, left + phase_fraction * (right - left), 725), fill=SURFACE, width=8)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, nargs="?", default=Path("results/bie_manufactured.gif"))
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    if args.n < 32 or args.n % 2 or args.frames < 4:
        raise ValueError("use an even N >= 32 and at least four frames")
    data_frames = compute_frames(args.n, args.frames)
    rgb_frames = [render(frame, args.n) for frame in data_frames]
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
    rgb_frames[0].save(args.output.with_name(args.output.stem + "_start.png"), optimize=True)
    rgb_frames[len(rgb_frames) // 2].save(
        args.output.with_name(args.output.stem + "_mid.png"), optimize=True
    )


if __name__ == "__main__":
    main()
