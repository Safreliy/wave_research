"""Convert Basilisk PPM frames into labelled smooth and cell-point GIFs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


FULL_X_MIN, FULL_X_MAX = 0.0, 32.0
Y_MIN, Y_MAX = -1.35, 1.15
PLOT_WIDTH, PLOT_HEIGHT = 1100, 550


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def decorate(
    source: Image.Image,
    index: int,
    dt: float,
    points: bool,
    x_min: float = FULL_X_MIN,
    x_max: float = FULL_X_MAX,
    source_x_min: float = FULL_X_MIN,
    source_x_max: float = FULL_X_MAX,
) -> Image.Image:
    source = source.convert("RGB")
    full_width = source.width
    left = round((x_min - source_x_min) / (source_x_max - source_x_min) * full_width)
    right = round((x_max - source_x_min) / (source_x_max - source_x_min) * full_width)
    source = source.crop((left, 0, right, source.height))
    canvas = Image.new("RGB", (1200, 720), "#f4f7fb")
    encoded = np.asarray(source, dtype=np.uint8)
    # output_ppm uses a blue-to-red map: water (f=1) is red and air (f=0)
    # is blue.  Convert this presentation encoding back into a clean binary
    # liquid/air view.  The interface is retained as a dark one-pixel band.
    liquid = encoded[..., 0].astype(int) > encoded[..., 2].astype(int)
    interface = np.zeros_like(liquid)
    interface[1:, :] |= liquid[1:, :] != liquid[:-1, :]
    interface[:, 1:] |= liquid[:, 1:] != liquid[:, :-1]
    if points:
        clean = np.empty_like(encoded)
        clean[:] = (245, 249, 252)
        clean[liquid] = (95, 186, 223)
        clean[interface] = (18, 96, 135)
        plot = Image.fromarray(clean).resize(
            (PLOT_WIDTH, PLOT_HEIGHT), Image.Resampling.NEAREST
        )
    else:
        # This is a presentation-only reconstruction: supersample the binary
        # VOF mask and blur by about one display pixel before taking the 1/2
        # contour.  The topology and contact time are unchanged, while blocky
        # AMR-cell staircases are not mistaken for physical capillary waves.
        mask = Image.fromarray((255 * liquid).astype(np.uint8)).resize(
            (PLOT_WIDTH, PLOT_HEIGHT), Image.Resampling.LANCZOS
        )
        smooth_fraction = np.asarray(
            mask.filter(ImageFilter.GaussianBlur(radius=1.1)), dtype=np.uint8
        )
        liquid_plot = smooth_fraction >= 128
        interface_plot = np.zeros_like(liquid_plot)
        interface_plot[1:, :] |= liquid_plot[1:, :] != liquid_plot[:-1, :]
        interface_plot[:, 1:] |= liquid_plot[:, 1:] != liquid_plot[:, :-1]
        clean_plot = np.empty((PLOT_HEIGHT, PLOT_WIDTH, 3), dtype=np.uint8)
        clean_plot[:] = (245, 249, 252)
        clean_plot[liquid_plot] = (95, 186, 223)
        clean_plot[interface_plot] = (18, 96, 135)
        plot = Image.fromarray(clean_plot)
    canvas.paste(plot, (50, 112))
    draw = ImageDraw.Draw(canvas)

    # The solid beach is reconstructed from the exact analytic geometry used
    # in plunging_beach.c, avoiding colormap-dependent masking artefacts.
    def px(horizontal: float) -> float:
        return 50 + (horizontal - x_min) * PLOT_WIDTH / (x_max - x_min)

    def py(vertical: float) -> float:
        return 112 + (Y_MAX - vertical) * PLOT_HEIGHT / (Y_MAX - Y_MIN)

    beach_toe = 10.0
    slope = np.deg2rad(3.0)
    beach_polygon = [
        (px(x_min), py(Y_MIN)),
        (px(x_min), py(-1.0 if x_min < beach_toe else -1.0 + slope * (x_min - beach_toe))),
        (px(max(x_min, beach_toe)), py(-1.0 if x_min < beach_toe else -1.0 + slope * (x_min - beach_toe))),
        (px(x_max), py(-1.0 + slope * (x_max - beach_toe))),
        (px(x_max), py(Y_MIN)),
    ]
    draw.polygon(beach_polygon, fill="#c9b285")
    draw.line(beach_polygon[1:4], fill="#6f5a36", width=3)

    draw.text((50, 22), "Two-phase VOF: solitary wave breaking on a 3-degree beach", fill="#172235", font=font(28, True))
    subtitle = (
        "Basilisk test -- interface-smoothed rendering; numerical topology unchanged"
        if not points
        else "Basilisk test -- raw interface samples; real topology change, not BIE output"
    )
    draw.text((50, 68), subtitle, fill="#b45309", font=font(17, True))
    vertical_exaggeration = ((x_max - x_min) / (Y_MAX - Y_MIN)) / (
        PLOT_WIDTH / PLOT_HEIGHT
    )
    draw.text(
        (50, 676),
        f"t={index*dt:.2f}  |  x=[{x_min:g},{x_max:g}]  |  adaptive VOF  |  vertical x{vertical_exaggeration:.1f}",
        fill="#4b6179",
        font=font(15),
    )
    if points:
        # These are occupied interface samples, explicitly not point vortices.
        iy, ix = np.nonzero(interface)
        stride = max(3, len(ix) // 220)
        for sy, sx in zip(iy[::stride], ix[::stride]):
            marker_x = 50 + sx * PLOT_WIDTH / max(1, source.width - 1)
            marker_y = 112 + sy * PLOT_HEIGHT / max(1, source.height - 1)
            draw.ellipse(
                (marker_x - 2, marker_y - 2, marker_x + 2, marker_y + 2),
                fill="#ef4444",
            )
        draw.text(
            (790, 68),
            "red = interface samples (not vortices)",
            fill="#b91c1c",
            font=font(14, True),
        )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--simulation-dt", type=float, default=0.08)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--points", action="store_true")
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--x-min", type=float, default=FULL_X_MIN)
    parser.add_argument("--x-max", type=float, default=FULL_X_MAX)
    parser.add_argument("--source-x-min", type=float, default=FULL_X_MIN)
    parser.add_argument("--source-x-max", type=float, default=FULL_X_MAX)
    parser.add_argument("--impact-time", type=float)
    args = parser.parse_args()
    if not args.source_x_min <= args.x_min < args.x_max <= args.source_x_max:
        raise ValueError("invalid x crop")
    all_paths = sorted(args.frame_directory.glob("vof-*.ppm"))
    start_frame = max(0, round(args.start_time / args.simulation_dt))
    paths = all_paths[start_frame:]
    if len(paths) < 2:
        raise ValueError("fewer than two Basilisk frames")
    frames = [
        decorate(
            Image.open(path),
            start_frame + index,
            args.simulation_dt,
            args.points,
            args.x_min,
            args.x_max,
            args.source_x_min,
            args.source_x_max,
        )
        for index, path in enumerate(paths)
    ]
    palette = frames[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    gifs = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gifs[0].save(
        args.output,
        save_all=True,
        append_images=gifs[1:],
        duration=round(1000/args.fps),
        loop=0,
        disposal=2,
        optimize=False,
    )
    keyframes = args.output.with_suffix("").with_name(args.output.stem + "_keyframes")
    keyframes.mkdir(parents=True, exist_ok=True)
    impact_position = (
        min(len(frames) - 1, max(0, round(args.impact_time / args.simulation_dt) - start_frame))
        if args.impact_time is not None
        else len(frames) - 1
    )
    positions = {
        "start": 0,
        "shoaling": min(len(frames) - 1, len(frames) // 2),
        "curling": max(0, impact_position - 5),
        "impact": impact_position,
        "postimpact": len(frames) - 1,
    }
    for label, position in positions.items():
        frames[position].save(keyframes / f"{label}.png", optimize=True)


if __name__ == "__main__":
    main()
