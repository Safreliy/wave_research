"""Render VOF evolution beside signed samples of the resolved vorticity.

The right panel is a diagnostic quadrature of the Eulerian vorticity field,
not a claim that the post-handoff solver advances Lagrangian point vortices.
Each marker represents the sign and clipped magnitude of one sampled output
pixel, i.e. qualitatively ``Gamma_ij ~ omega_ij Delta A``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _liquid(encoded: np.ndarray) -> np.ndarray:
    return encoded[..., 0].astype(int) > encoded[..., 2].astype(int)


def render_frame(
    vof_path: Path,
    vorticity_path: Path,
    continuation_time: float,
    stride: int,
    threshold: float,
    level_label: str = "receiver grid",
) -> Image.Image:
    vof = np.asarray(Image.open(vof_path).convert("RGB"), dtype=np.uint8)
    omega_rgb = np.asarray(Image.open(vorticity_path).convert("RGB"), dtype=np.uint8)
    if vof.shape != omega_rgb.shape:
        raise ValueError("VOF and vorticity rasters have different shapes")
    liquid = _liquid(vof)
    smooth = np.empty_like(vof)
    smooth[:] = (247, 250, 252)
    smooth[liquid] = (54, 162, 205)
    interface = np.zeros_like(liquid)
    interface[1:] |= liquid[1:] != liquid[:-1]
    interface[:, 1:] |= liquid[:, 1:] != liquid[:, :-1]
    smooth[interface] = (8, 75, 112)

    height, width = liquid.shape
    panel_width, panel_height = 690, 260
    canvas = Image.new("RGB", (1500, 440), "#f7fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (45, 20),
        f"Conservative-q breaker continuation — {level_label}",
        fill="#172235", font=_font(27, True),
    )
    draw.text(
        (45, 59),
        f"VOF time after BIE handoff: {continuation_time:.2f}",
        fill="#475569", font=_font(17),
    )
    left = Image.fromarray(smooth).resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    canvas.paste(left, (45, 110))
    draw.rectangle((45, 110, 45 + panel_width, 110 + panel_height), outline="#94a3b8", width=2)
    draw.text((45, 382), "two-phase VOF interface", fill="#334155", font=_font(16, True))

    right_x = 765
    draw.rectangle((right_x, 110, right_x + panel_width, 110 + panel_height), fill="white", outline="#94a3b8", width=2)
    silhouette = np.empty_like(vof)
    silhouette[:] = (255, 255, 255)
    silhouette[liquid] = (226, 232, 240)
    base = Image.fromarray(silhouette).resize((panel_width, panel_height), Image.Resampling.NEAREST)
    canvas.paste(base, (right_x, 110))
    draw = ImageDraw.Draw(canvas)

    # The default Basilisk colour map used in these files is green near zero,
    # red for positive and blue for negative values.  R-B is therefore a
    # monotone signed proxy; the field was clipped to |omega| <= 8 at output.
    signed = (omega_rgb[..., 0].astype(float) - omega_rgb[..., 2].astype(float)) / 255.0
    for row in range(stride // 2, height, stride):
        for col in range(stride // 2, width, stride):
            value = float(signed[row, col])
            magnitude = abs(value)
            if magnitude < threshold:
                continue
            x = right_x + (col + 0.5) / width * panel_width
            y = 110 + (row + 0.5) / height * panel_height
            radius = 1.2 + 3.0 * min(1.0, magnitude)
            color = "#dc2626" if value > 0.0 else "#2563eb"
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    draw.text(
        (right_x, 382),
        "signed vorticity quadrature: red +, blue -",
        fill="#334155", font=_font(16, True),
    )
    draw.text(
        (835, 413),
        "diagnostic samples of Eulerian omega; not solver particles",
        fill="#9a3412", font=_font(14),
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--level-label", default="receiver grid")
    args = parser.parse_args()
    vof_paths = sorted((args.run_directory / "handoff_frames").glob("vof-*.ppm"))
    omega_paths = sorted((args.run_directory / "handoff_vorticity").glob("vorticity-*.ppm"))
    if len(vof_paths) < 2 or len(vof_paths) != len(omega_paths):
        raise ValueError("matching VOF and vorticity sequences are required")
    frames = [
        render_frame(
            vof, omega, index * args.dt, args.stride, args.threshold,
            args.level_label,
        )
        for index, (vof, omega) in enumerate(zip(vof_paths, omega_paths))
    ]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    palette = frames[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    gif = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    gif[0].save(
        args.output_prefix.with_suffix(".gif"), save_all=True, append_images=gif[1:],
        duration=round(1000 / args.fps), loop=0, disposal=2, optimize=False,
    )
    keyframes = args.output_prefix.with_name(args.output_prefix.name + "_keyframes")
    keyframes.mkdir(parents=True, exist_ok=True)
    for label, index in {
        "handoff": 0,
        "rollup": len(frames) // 2,
        "near_closure": min(len(frames) - 1, round(4.1 / args.dt)),
        "late_pocket": min(len(frames) - 1, round(5.1 / args.dt)),
        "end": len(frames) - 1,
    }.items():
        frames[index].save(keyframes / f"{label}.png", optimize=True)


if __name__ == "__main__":
    main()
