"""Render and topology-audit the Basilisk continuation of a BIE handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def liquid_mask(path: Path) -> np.ndarray:
    encoded = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return encoded[..., 0].astype(int) > encoded[..., 2].astype(int)


def solid_mask(
    shape: tuple[int, int],
    bottom_x: np.ndarray,
    bottom_z: np.ndarray,
    length: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    height, width = shape
    x = x_min + (np.arange(width) + 0.5) * (x_max - x_min) / width
    y = y_max - (np.arange(height) + 0.5) * (y_max - y_min) / height
    bed = np.interp(x, bottom_x, bottom_z, period=length)
    return y[:, None] < bed[None, :]


def exterior_connected_air(air: np.ndarray, periodic_x: bool) -> np.ndarray:
    structure = ndimage.generate_binary_structure(2, 1)
    if periodic_x:
        # Three tiled copies encode the horizontal wrap without a Python
        # fixed-point loop.  Propagation in the middle copy can cross either
        # seam, exactly matching periodic four-neighbour connectivity.
        tiled = np.concatenate((air, air, air), axis=1)
        seed = np.zeros_like(tiled)
        seed[0] = tiled[0]
        reachable = ndimage.binary_propagation(
            seed, structure=structure, mask=tiled
        )
        width = air.shape[1]
        return reachable[:, width : 2 * width]
    seed = np.zeros_like(air)
    seed[0] = air[0]
    seed[:, 0] |= air[:, 0]
    seed[:, -1] |= air[:, -1]
    return ndimage.binary_propagation(seed, structure=structure, mask=air)


def connected_component_labels(
    mask: np.ndarray, *, periodic_x: bool
) -> tuple[np.ndarray, int]:
    """Label four-neighbour components, merging the periodic x seam.

    ``scipy.ndimage.label`` has no periodic boundary mode.  Labelling the raw
    raster without the merge splits a physical component crossing the seam,
    underestimates its area, and can change an area-gated topology decision.
    """
    structure = ndimage.generate_binary_structure(2, 1)
    labels, count = ndimage.label(mask, structure=structure)
    if not periodic_x or count == 0:
        return labels, count

    parent = np.arange(count + 1, dtype=int)

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = int(parent[label])
        return label

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    seam_rows = np.flatnonzero(mask[:, 0] & mask[:, -1])
    for row in seam_rows:
        union(int(labels[row, 0]), int(labels[row, -1]))

    roots = np.array([find(label) for label in range(count + 1)], dtype=int)
    unique_roots = np.unique(roots[1:])
    compact = np.zeros(count + 1, dtype=int)
    compact[unique_roots] = np.arange(1, len(unique_roots) + 1)
    return compact[roots[labels]], len(unique_roots)


def topology_record(
    path: Path,
    bottom_x: np.ndarray,
    bottom_z: np.ndarray,
    length: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    minimum_pixels: int,
    minimum_area: float = 0.0,
    bed_guard_distance: float | None = None,
) -> dict[str, float | int]:
    liquid = liquid_mask(path)
    return topology_record_from_liquid(
        liquid, bottom_x, bottom_z, length, x_min, x_max, y_min, y_max,
        minimum_pixels, minimum_area, bed_guard_distance,
    )


def topology_record_from_liquid(
    liquid: np.ndarray,
    bottom_x: np.ndarray,
    bottom_z: np.ndarray,
    length: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    minimum_pixels: int,
    minimum_area: float = 0.0,
    bed_guard_distance: float | None = None,
) -> dict[str, float | int]:
    """Measure enclosed gas on a specified raster.

    ``minimum_pixels`` protects against connectivity noise on the diagnostic
    raster, while ``minimum_area`` is a physical, raster-independent gate.
    Keeping both gates explicit is important: a fixed four-pixel rule becomes
    less stringent when the same state is rendered more finely.
    """
    solid = solid_mask(
        liquid.shape, bottom_x, bottom_z, length, x_min, x_max, y_min, y_max
    )
    air = ~liquid & ~solid
    periodic_x = abs((x_max - x_min) - length) <= 1.0e-10 * length
    raw_pocket = air & ~exterior_connected_air(air, periodic_x)
    structure = ndimage.generate_binary_structure(2, 1)
    raw_labels, _ = connected_component_labels(
        raw_pocket, periodic_x=periodic_x
    )
    # Reject entire components that touch a two-pixel bed guard.  Merely
    # cutting the guarded pixels out would leave the core of a bed-attached
    # cavity and falsely relabel it as jet-impact entrainment.
    dx = (x_max - x_min) / liquid.shape[1]
    dy = (y_max - y_min) / liquid.shape[0]
    guard_iterations = (
        2 if bed_guard_distance is None
        else max(1, int(np.ceil(bed_guard_distance / min(dx, dy))))
    )
    solid_guard = ndimage.binary_dilation(
        solid, structure=structure, iterations=guard_iterations
    )
    bed_labels = np.unique(raw_labels[solid_guard & raw_pocket])
    bed_labels = bed_labels[bed_labels != 0]
    pocket = raw_pocket & ~np.isin(raw_labels, bed_labels)
    labels, count = connected_component_labels(pocket, periodic_x=periodic_x)
    pixel_area = (x_max - x_min) / liquid.shape[1] * (y_max - y_min) / liquid.shape[0]
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    raw_sizes = sizes[sizes > 0]
    minimum_size = max(minimum_pixels, int(np.ceil(minimum_area / pixel_area)))
    sizes = sizes[sizes >= minimum_size]
    return {
        "unfiltered_pocket_count": int(len(raw_sizes)),
        "unfiltered_largest_pocket_area": float(
            raw_sizes.max(initial=0) * pixel_area
        ),
        "pocket_count": int(len(sizes)),
        "largest_pocket_pixels": int(sizes.max(initial=0)),
        "largest_pocket_area": float(sizes.max(initial=0) * pixel_area),
        "total_pocket_area": float(np.sum(sizes) * pixel_area),
        "pixel_area": float(pixel_area),
        "effective_minimum_pixels": int(minimum_size),
        "bed_guard_distance": float(guard_iterations * min(dx, dy)),
    }


def persistent_onset(
    records: list[dict[str, float | int]], persistence_frames: int
) -> int | None:
    """Return the first index starting a persistent topological closure."""
    if persistence_frames < 1:
        raise ValueError("persistence_frames must be positive")
    occupied = [int(record["pocket_count"]) > 0 for record in records]
    for start in range(0, len(occupied) - persistence_frames + 1):
        if all(occupied[start : start + persistence_frames]):
            return start
    return None


def decorate(
    path: Path,
    continuation_time: float,
    source_time: float,
    bottom_x: np.ndarray,
    bottom_z: np.ndarray,
    length: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    points: bool,
    level_label: str,
) -> Image.Image:
    liquid = liquid_mask(path)
    interface = np.zeros_like(liquid)
    interface[1:] |= liquid[1:] != liquid[:-1]
    interface[:, 1:] |= liquid[:, 1:] != liquid[:, :-1]
    clean = np.empty((*liquid.shape, 3), dtype=np.uint8)
    clean[:] = (245, 249, 252)
    clean[liquid] = (76, 174, 215)
    clean[interface] = (14, 88, 128)
    plot_width, plot_height = 1100, 500
    plot = Image.fromarray(clean).resize(
        (plot_width, plot_height),
        Image.Resampling.NEAREST if points else Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (1200, 710), "#f4f7fb")
    canvas.paste(plot, (50, 120))
    draw = ImageDraw.Draw(canvas)

    px = lambda value: 50 + (value - x_min) / (x_max - x_min) * plot_width
    py = lambda value: 120 + (y_max - value) / (y_max - y_min) * plot_height
    bed_x = np.linspace(x_min, x_max, 1000)
    bed_z = np.interp(bed_x, bottom_x, bottom_z, period=length)
    bed = [(px(float(x)), py(float(z))) for x, z in zip(bed_x, bed_z)]
    polygon = [(50, 120 + plot_height), *bed, (50 + plot_width, 120 + plot_height)]
    draw.polygon(polygon, fill="#c9b285")
    draw.line(bed, fill="#6f5a36", width=3)
    draw.text((50, 22), "Conservative Euler--BIE to two-phase VOF continuation", fill="#172235", font=font(27, True))
    draw.text((50, 68), "same transferred state; pressure-projected adaptive interface evolution", fill="#b45309", font=font(17, True))
    draw.text(
        (50, 660),
        f"BIE t={source_time:.4f} + VOF Δt={continuation_time:.2f}  |  x=[{x_min:g},{x_max:g}]  |  {level_label}",
        fill="#4b6179",
        font=font(15),
    )
    if points:
        iy, ix = np.nonzero(interface)
        stride = max(2, len(ix) // 260)
        for sy, sx in zip(iy[::stride], ix[::stride]):
            marker_x = 50 + sx * plot_width / max(1, liquid.shape[1] - 1)
            marker_y = 120 + sy * plot_height / max(1, liquid.shape[0] - 1)
            draw.ellipse((marker_x - 2, marker_y - 2, marker_x + 2, marker_y + 2), fill="#ef4444")
        draw.text((815, 69), "interface samples, not vortices", fill="#b91c1c", font=font(14, True))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_directory", type=Path)
    parser.add_argument("handoff_npz", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--simulation-dt", type=float, default=0.02)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--minimum-pixels", type=int, default=4)
    parser.add_argument("--minimum-area", type=float, default=0.0)
    parser.add_argument("--persistence-frames", type=int, default=3)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float)
    parser.add_argument("--y-min", type=float, default=-1.28)
    parser.add_argument("--y-max", type=float, default=1.25)
    parser.add_argument("--level-label", default="adaptive VOF")
    args = parser.parse_args()
    with np.load(args.handoff_npz) as loaded:
        bottom_x = loaded["bottom_x"]
        bottom_z = loaded["bottom_z"]
        grid_x = loaded["grid_x"]
        length = float(grid_x[-1] - grid_x[0] + (grid_x[1] - grid_x[0]))
        source_time = float(loaded["time"])
    x_min = args.x_min
    x_max = length if args.x_max is None else args.x_max
    if not 0.0 <= x_min < x_max <= length:
        raise ValueError("require 0 <= x_min < x_max <= periodic length")
    paths = sorted(args.frame_directory.glob("vof-*.ppm"))
    if len(paths) < 2:
        raise ValueError("fewer than two continuation frames")
    records = []
    for index, path in enumerate(paths):
        record = topology_record(
            path, bottom_x, bottom_z, length, x_min, x_max, args.y_min, args.y_max,
            args.minimum_pixels,
            args.minimum_area,
        )
        record.update({"frame": index, "continuation_time": index * args.simulation_dt})
        records.append(record)
    impact_index = persistent_onset(records, args.persistence_frames)
    impact_time = (
        float(records[impact_index]["continuation_time"])
        if impact_index is not None else None
    )
    summary = {
        "schema": "bie-to-vof-topology-v1",
        "criterion": (
            "air disconnected from the top boundary with periodic horizontal connectivity; bed-attached components rejected"
            if abs((x_max - x_min) - length) <= 1.0e-10 * length
            else "air disconnected from top and lateral boundaries in the zoom window; bed-attached components rejected"
        ),
        "minimum_component_pixels": args.minimum_pixels,
        "minimum_component_area": args.minimum_area,
        "persistence_frames": args.persistence_frames,
        "source_bie_time": source_time,
        "frame_count": len(paths),
        "impact_detected": impact_index is not None,
        "last_open_continuation_time": (
            float(records[impact_index - 1]["continuation_time"])
            if impact_index is not None and impact_index > 0 else None
        ),
        "impact_continuation_time": impact_time,
        "impact_total_time": source_time + impact_time if impact_time is not None else None,
        "records": records,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for points, suffix in ((False, "smooth"), (True, "points")):
        frames = [
            decorate(
                path,
                index * args.simulation_dt,
                source_time,
                bottom_x,
                bottom_z,
                length,
                x_min,
                x_max,
                args.y_min,
                args.y_max,
                points,
                args.level_label,
            )
            for index, path in enumerate(paths)
        ]
        palette = frames[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        gif = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        output = args.output_prefix.with_name(args.output_prefix.name + f"_{suffix}.gif")
        gif[0].save(
            output,
            save_all=True,
            append_images=gif[1:],
            duration=round(1000 / args.fps),
            loop=0,
            disposal=2,
            optimize=False,
        )
        keyframes = output.with_suffix("").with_name(output.stem + "_keyframes")
        keyframes.mkdir(parents=True, exist_ok=True)
        keyframe_impact_index = (
            impact_index if impact_index is not None else len(frames) - 1
        )
        positions = {
            "start": 0,
            "middle": len(frames) // 2,
            "preimpact": max(0, keyframe_impact_index - 3),
            "impact": keyframe_impact_index,
            "end": len(frames) - 1,
        }
        for label, position in positions.items():
            frames[position].save(keyframes / f"{label}.png", optimize=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
