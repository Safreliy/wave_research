"""Convert Basilisk PPM sequences to reviewable PNG keyframes and GIFs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def labelled_frame(path: Path, label: str) -> Image.Image:
    with Image.open(path) as source:
        frame = source.convert("RGB")
    banner_height = 44
    canvas = Image.new("RGB", (frame.width, frame.height + banner_height), "white")
    canvas.paste(frame, (0, banner_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    draw.text((14, 10), label, fill=(20, 35, 50), font=font)
    return canvas


def render_sequence(
    source_directory: Path,
    output_directory: Path,
    stem: str,
    label: str,
    time_step: float,
) -> None:
    paths = sorted(source_directory.glob("*.ppm"))
    if not paths:
        raise ValueError(f"no PPM frames found in {source_directory}")
    frames = [
        labelled_frame(path, f"{label} | dt = {index * time_step:.3f}")
        for index, path in enumerate(paths)
    ]
    output_directory.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_directory / f"{stem}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=220,
        loop=0,
        optimize=False,
        disposal=2,
    )
    for index in sorted({0, len(frames) // 2, len(frames) - 1}):
        frames[index].save(output_directory / f"{stem}_{index:03d}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--time-step", type=float, default=0.02)
    parser.add_argument("--level", type=int, default=9)
    args = parser.parse_args()
    cadence_tag = f"dt{int(round(1000 * args.time_step)):03d}"
    render_sequence(
        args.run_directory / "handoff_frames",
        args.output_directory,
        f"mac_vof_level{args.level}_{cadence_tag}",
        f"MAC-flux BIE-to-VOF, level {args.level}",
        args.time_step,
    )
    render_sequence(
        args.run_directory / "handoff_vorticity",
        args.output_directory,
        f"mac_vorticity_level{args.level}_{cadence_tag}",
        f"Resolved vorticity, level {args.level}",
        args.time_step,
    )


if __name__ == "__main__":
    main()
