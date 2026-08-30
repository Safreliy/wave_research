"""Manufactured close-self-interaction ablation for the periodic Euler BIE."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from animate_results import BACKGROUND, GRID, INK, MUTED, SURFACE, WATER, _font
from topography_bie import TopographyBIE


ROOT = Path(__file__).resolve().parent


def manufactured_fold(n: int) -> tuple[np.ndarray, ...]:
    length = 12.0
    base = length * np.arange(n) / n
    distance = (base - 6.0 + 0.5 * length) % length - 0.5 * length
    bump = np.exp(-0.5 * (distance / 0.5) ** 2)
    x = base + 2.0 * bump
    z = 0.15 * bump
    bottom_z = -2.0 * np.ones(n)
    wavenumber = 4.0 * math.pi / length
    trace = (
        np.sin(wavenumber * x)
        * np.cosh(wavenumber * (z + 2.0))
        / np.cosh(2.0 * wavenumber)
    )
    return x, z, base, bottom_z, trace, np.asarray(length), np.asarray(wavenumber)


def run_case(n: int, factor: int) -> dict[str, float | int]:
    x, z, bottom_x, bottom_z, trace, length_array, k_array = manufactured_fold(n)
    length, wavenumber = float(length_array), float(k_array)
    start = time.perf_counter()
    solver = TopographyBIE(
        x,
        z,
        bottom_x,
        bottom_z,
        length,
        cross_oversampling=1,
        surface_self_oversampling=factor,
    )
    solution = solver.solve(trace)
    elapsed = time.perf_counter() - start
    gradient_x = (
        wavenumber
        * np.cos(wavenumber * x)
        * np.cosh(wavenumber * (z + 2.0))
        / np.cosh(2.0 * wavenumber)
    )
    gradient_z = (
        wavenumber
        * np.sin(wavenumber * x)
        * np.sinh(wavenumber * (z + 2.0))
        / np.cosh(2.0 * wavenumber)
    )
    exact = (
        -solver.surface.z_alpha * gradient_x
        + solver.surface.x_alpha * gradient_z
    ) / solver.surface.metric
    error = float(
        np.linalg.norm(solution.surface_normal_derivative - exact)
        / np.linalg.norm(exact)
    )
    return {
        "n": n,
        "factor": factor,
        "relative_normal_derivative_error": error,
        "wall_seconds": elapsed,
        "bie_residual": solution.residual,
        "gap_over_mean_spacing": (
            solver.minimum_nonlocal_surface_separation
            / solver.mean_surface_spacing
        ),
    }


def draw_report(report: dict[str, object], output: Path) -> None:
    image = Image.new("RGB", (1500, 760), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text(
        (55, 25),
        "Close self-interaction: oversampled Kress--Nystrom ablation",
        fill=INK,
        font=_font(29, bold=True),
    )
    draw.text(
        (55, 70),
        "manufactured harmonic field on a non-self-intersecting overhanging fold",
        fill=MUTED,
        font=_font(16),
    )
    left_box = (45, 115, 720, 700)
    right_box = (750, 115, 1455, 700)
    for box in (left_box, right_box):
        draw.rounded_rectangle(box, radius=10, fill="white", outline=GRID, width=2)

    x, z, _, bottom_z, _, _, _ = manufactured_fold(512)
    left, right, top, bottom = 90.0, 680.0, 180.0, 600.0
    x_min, x_max, z_min, z_max = 4.8, 8.7, -0.05, 0.28
    mx = lambda value: left + (value - x_min) / (x_max - x_min) * (right - left)
    mz = lambda value: bottom - (value - z_min) / (z_max - z_min) * (bottom - top)
    local = (x >= x_min) & (x <= x_max)
    points = [(mx(float(a)), mz(float(b))) for a, b in zip(x[local], z[local])]
    water = points + [(points[-1][0], bottom), (points[0][0], bottom)]
    draw.polygon(water, fill=WATER)
    draw.line(points, fill=SURFACE, width=5, joint="curve")
    draw.text((78, 135), "A  Manufactured folded boundary", fill=INK, font=_font(20, bold=True))
    draw.text((78, 632), "minimum nonlocal gap / mean ds = 0.66 (N=64)", fill=MUTED, font=_font(14))

    plot = (830.0, 190.0, 1405.0, 590.0)
    draw.text((785, 135), "B  DNO error versus source oversampling", fill=INK, font=_font(20, bold=True))
    factors = (1, 2, 4, 8, 16)
    xmap = lambda factor: plot[0] + math.log2(factor) / 4.0 * (plot[2] - plot[0])
    ymin, ymax = 1.0e-2, 10.0
    ymap = lambda value: plot[3] - (
        (math.log10(value) - math.log10(ymin))
        / (math.log10(ymax) - math.log10(ymin))
        * (plot[3] - plot[1])
    )
    for exponent in (-2, -1, 0, 1):
        py = ymap(10.0**exponent)
        draw.line((plot[0], py, plot[2], py), fill="#edf1f5", width=1)
        draw.text((plot[0] - 53, py - 7), f"1e{exponent}", fill=MUTED, font=_font(12))
    for factor in factors:
        px = xmap(factor)
        draw.line((px, plot[1], px, plot[3]), fill="#edf1f5", width=1)
        draw.text((px - 7, plot[3] + 10), str(factor), fill=MUTED, font=_font(12))
    colors = {64: "#ef4444", 128: "#087ea4"}
    for n in (64, 128):
        rows = [row for row in report["rows"] if row["n"] == n]
        curve = [
            (xmap(int(row["factor"])), ymap(float(row["relative_normal_derivative_error"])))
            for row in rows
        ]
        draw.line(curve, fill=colors[n], width=4, joint="curve")
        for px, py in curve:
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=colors[n], outline="white", width=1)
        draw.text((plot[2] - 95, plot[1] + (22 if n == 128 else 49)), f"N={n}", fill=colors[n], font=_font(14, bold=True))
    draw.text((1040, 645), "oversampling factor", fill=MUTED, font=_font(14))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "results" / "publication_package",
    )
    args = parser.parse_args()
    factors = (1, 2, 4, 8, 16)
    rows = [run_case(n, factor) for n in (64, 128) for factor in factors]
    report: dict[str, object] = {
        "evidence_class": "manufactured harmonic solution",
        "geometry": "periodic non-self-intersecting Gaussian fold",
        "rows": rows,
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    json_path = args.output_directory / "near_self_quadrature_ablation.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    draw_report(report, args.output_directory / "near_self_quadrature_ablation.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
