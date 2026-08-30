"""Detect post-impact trapped-air components in Basilisk VOF frames.

The first non-boundary-connected air component is a discrete, auditable marker
of free-surface topology change. It is deliberately separate from geometric
"minimum gap" diagnostics used by the pre-impact boundary-integral solver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


X_MIN, X_MAX = 0.0, 32.0
Y_MIN, Y_MAX = -1.35, 1.15
BEACH_TOE = 10.0
BEACH_SLOPE = np.deg2rad(3.0)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf")
        if bold
        else Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def masks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    liquid = encoded[..., 0].astype(int) > encoded[..., 2].astype(int)
    height, width = liquid.shape
    x = X_MIN + (np.arange(width) + 0.5) * (X_MAX - X_MIN) / width
    y = Y_MAX - (np.arange(height) + 0.5) * (Y_MAX - Y_MIN) / height
    beach_height = np.where(
        x < BEACH_TOE,
        -1.0,
        -1.0 + BEACH_SLOPE * (x - BEACH_TOE),
    )
    solid = y[:, None] < beach_height[None, :]
    return liquid, solid


def trapped_air(path: Path, minimum_pixels: int) -> dict[str, float | int]:
    liquid, solid = masks(path)
    air = ~liquid & ~solid
    labels, count = ndimage.label(
        air, structure=ndimage.generate_binary_structure(2, 1)
    )
    boundary_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    candidate = np.ones(count + 1, dtype=bool)
    candidate[0] = False
    candidate[boundary_labels] = False
    candidate &= sizes >= minimum_pixels
    pocket_sizes = sizes[candidate]
    pixel_area = (
        (X_MAX - X_MIN) / liquid.shape[1]
        * (Y_MAX - Y_MIN)
        / liquid.shape[0]
    )
    return {
        "pocket_count": int(pocket_sizes.size),
        "largest_pocket_pixels": int(pocket_sizes.max(initial=0)),
        "largest_pocket_area": float(pocket_sizes.max(initial=0) * pixel_area),
        "total_pocket_area": float(pocket_sizes.sum() * pixel_area),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_directory", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_plot", type=Path)
    parser.add_argument("--simulation-dt", type=float, default=0.08)
    parser.add_argument("--minimum-pixels", type=int, default=4)
    args = parser.parse_args()

    paths = sorted(args.frame_directory.glob("vof-*.ppm"))
    if not paths:
        raise ValueError("no Basilisk VOF frames found")
    records = []
    for index, path in enumerate(paths):
        record = trapped_air(path, args.minimum_pixels)
        record.update({"frame": index, "time": index * args.simulation_dt})
        records.append(record)

    impacted = [record for record in records if record["pocket_count"] > 0]
    summary = {
        "criterion": "first air component disconnected from all image boundaries",
        "minimum_component_pixels": args.minimum_pixels,
        "frame_count": len(records),
        "impact_detected": bool(impacted),
        "impact_time": impacted[0]["time"] if impacted else None,
        "impact_frame": impacted[0]["frame"] if impacted else None,
        "coordinate_box": [X_MIN, X_MAX, Y_MIN, Y_MAX],
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    times = np.asarray([record["time"] for record in records], dtype=float)
    areas = np.asarray([record["total_pocket_area"] for record in records], dtype=float)
    image = Image.new("RGB", (1500, 760), "#f8fafc")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 150, 100, 1435, 635
    draw.text((50, 25), "Resolved trapped-air area after jet closure", fill="#172235", font=font(34, True))
    draw.line((left, top, left, bottom, right, bottom), fill="#475569", width=3)
    time_span = max(float(times[-1] - times[0]), 1e-12)
    area_max = max(float(areas.max(initial=0.0)), 1e-12)

    def plot_x(value: float) -> float:
        return left + (value - times[0]) / time_span * (right - left)

    def plot_y(value: float) -> float:
        return bottom - value / area_max * (bottom - top)

    points = [(plot_x(float(t)), plot_y(float(a))) for t, a in zip(times, areas)]
    if len(points) > 1:
        draw.line(points, fill="#126e96", width=5, joint="curve")
    for fraction in np.linspace(0.0, 1.0, 6):
        x_value = float(times[0] + fraction * time_span)
        x_coordinate = plot_x(x_value)
        draw.line((x_coordinate, bottom, x_coordinate, bottom + 9), fill="#475569", width=2)
        draw.text((x_coordinate - 22, bottom + 15), f"{x_value:.1f}", fill="#475569", font=font(17))
        area_value = fraction * area_max
        y_coordinate = plot_y(area_value)
        draw.line((left - 9, y_coordinate, left, y_coordinate), fill="#475569", width=2)
        draw.text((20, y_coordinate - 10), f"{area_value:.3g}", fill="#475569", font=font(17))
    if impacted:
        impact_x = plot_x(float(impacted[0]["time"]))
        for y_coordinate in range(top, bottom, 18):
            draw.line((impact_x, y_coordinate, impact_x, min(y_coordinate + 9, bottom)), fill="#c2410c", width=3)
        draw.text(
            (min(impact_x + 12, right - 300), top + 18),
            f"first resolved closure: t={impacted[0]['time']:.2f}",
            fill="#9a3412",
            font=font(20, True),
        )
    draw.text((610, 700), "nondimensional time", fill="#334155", font=font(20))
    draw.text((18, 70), "area", fill="#334155", font=font(20))
    image.save(args.output_plot, optimize=True)


if __name__ == "__main__":
    main()
