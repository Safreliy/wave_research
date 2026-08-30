"""Create scientific GIFs and lossless key frames from simulation NPZ files.

The renderer uses Pillow only.  Axes remain fixed over the animation, numerical
diagnostics are drawn into every frame, and a fixed palette avoids GIF flicker.
PNG key frames are kept for figures or supplementary-video assembly.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from topographic_wave_solver import reference_energy


Array = np.ndarray


BACKGROUND = "#f7f9fc"
INK = "#172433"
MUTED = "#5b6b7b"
GRID = "#d7e0e8"
WATER = "#d9eff7"
SURFACE = "#087eaa"
CREST = "#d94841"
EVENT = "#f09a35"


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    filename = "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"
    try:
        return ImageFont.truetype(filename, size=size)
    except OSError:
        return ImageFont.load_default()


def _nice_ticks(lower: float, upper: float, count: int = 5) -> Array:
    span = max(upper - lower, np.finfo(float).eps)
    raw = span / count
    magnitude = 10.0 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    step = (1.0 if normalized < 1.5 else 2.0 if normalized < 3.5 else 5.0) * magnitude
    first = math.ceil(lower / step) * step
    return np.arange(first, upper + 0.5 * step, step)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float, float, float],
    fill: str,
    width: int = 2,
    dash: int = 8,
) -> None:
    x0, y0, x1, y1 = xy
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0.0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    position = 0.0
    while position < length:
        end = min(position + dash, length)
        draw.line(
            (
                x0 + ux * position,
                y0 + uy * position,
                x0 + ux * end,
                y0 + uy * end,
            ),
            fill=fill,
            width=width,
        )
        position += 2.0 * dash


class ScientificAnimation:
    def __init__(
        self,
        data: dict[str, Array],
        title: str,
        event_x: float | None,
        event_time: float | None,
        width: int = 1200,
        height: int = 720,
    ) -> None:
        self.data = data
        self.title = title
        self.event_x = event_x
        self.event_time = event_time
        self.width = width
        self.height = height
        self.left, self.right = 105, width - 55
        self.top, self.bottom = 115, height - 175
        self.x = data["x"]
        self.time = data["time"]
        self.eta = data["eta"]
        extent = float(np.max(np.abs(self.eta)))
        self.y_lower = -1.18 * extent
        self.y_upper = 1.18 * extent
        if "energy_reference" in data:
            energy_reference = float(data["energy_reference"])
        elif "bottom_x" in data and "bottom_z" in data:
            energy_reference = reference_energy(
                data["bottom_x"],
                data["bottom_z"],
                float(data["length"]),
                float(data["gravity"]),
                float(data.get("background_current", np.asarray(0.0))),
            )
        else:
            energy_reference = 0.0
        self.energy_scale = max(
            abs(float(data["energy"][0]) - energy_reference),
            np.finfo(float).eps,
        )
        self.volume_0 = float(data["volume"][0])

    def map_x(self, value: float) -> float:
        x_end = float(self.x[-1] + (self.x[1] - self.x[0]))
        return self.left + (value - float(self.x[0])) / (x_end - float(self.x[0])) * (
            self.right - self.left
        )

    def map_y(self, value: float) -> float:
        return self.bottom - (value - self.y_lower) / (
            self.y_upper - self.y_lower
        ) * (self.bottom - self.top)

    def draw_frame(self, history_index: int) -> Image.Image:
        image = Image.new("RGB", (self.width, self.height), BACKGROUND)
        draw = ImageDraw.Draw(image)
        title_font = _font(29, bold=True)
        label_font = _font(17)
        small_font = _font(15)
        metric_font = _font(18, bold=True)

        draw.text((self.left, 35), self.title, fill=INK, font=title_font)
        draw.text(
            (self.left, 76),
            "Fully nonlinear HOS baseline · fixed axes · eta(x,t)",
            fill=MUTED,
            font=small_font,
        )
        draw.rectangle(
            (self.left, self.top, self.right, self.bottom),
            fill="white",
            outline=GRID,
            width=2,
        )

        x_ticks = _nice_ticks(float(self.x[0]), float(self.x[-1] + self.x[1] - self.x[0]))
        for tick in x_ticks:
            px = self.map_x(float(tick))
            draw.line((px, self.top, px, self.bottom), fill="#edf1f5", width=1)
            label = f"{tick:.0f}" if abs(tick) >= 1 else f"{tick:.1f}"
            draw.text((px - 12, self.bottom + 10), label, fill=MUTED, font=small_font)

        for tick in _nice_ticks(self.y_lower, self.y_upper):
            py = self.map_y(float(tick))
            draw.line((self.left, py, self.right, py), fill="#edf1f5", width=1)
            draw.text((18, py - 9), f"{tick:.3f}", fill=MUTED, font=small_font)

        zero_y = self.map_y(0.0)
        draw.line((self.left, zero_y, self.right, zero_y), fill="#9dacba", width=2)
        if self.event_x is not None:
            event_px = self.map_x(self.event_x)
            _draw_dashed_line(
                draw, (event_px, self.top, event_px, self.bottom), EVENT, width=2
            )
            draw.text(
                (event_px + 7, self.top + 8), "target", fill=EVENT, font=small_font
            )

        profile = self.eta[history_index]
        surface_points = [
            (self.map_x(float(x)), self.map_y(float(y)))
            for x, y in zip(self.x, profile)
        ]
        water_polygon = surface_points + [
            (self.right, self.bottom),
            (self.left, self.bottom),
        ]
        draw.polygon(water_polygon, fill=WATER)
        draw.line(surface_points, fill=SURFACE, width=4, joint="curve")

        crest_index = int(np.argmax(profile))
        crest_x = self.map_x(float(self.x[crest_index]))
        crest_y = self.map_y(float(profile[crest_index]))
        draw.ellipse(
            (crest_x - 6, crest_y - 6, crest_x + 6, crest_y + 6),
            fill=CREST,
            outline="white",
            width=2,
        )

        time = float(self.time[history_index])
        energy_drift = (
            float(self.data["energy"][history_index]) - float(self.data["energy"][0])
        ) / self.energy_scale
        volume_drift = float(self.data["volume"][history_index]) - self.volume_0
        metrics_y = self.bottom + 52
        draw.text((self.left, metrics_y), f"t = {time:.3f}", fill=INK, font=metric_font)
        draw.text(
            (self.left + 220, metrics_y),
            f"crest = {float(profile[crest_index]):.5f}",
            fill=CREST,
            font=metric_font,
        )
        draw.text(
            (self.left + 485, metrics_y),
            f"Delta E/E0 = {energy_drift:+.2e}",
            fill=INK,
            font=label_font,
        )
        draw.text(
            (self.left + 775, metrics_y),
            f"Delta V = {volume_drift:+.2e}",
            fill=INK,
            font=label_font,
        )

        timeline_y = self.height - 48
        draw.line((self.left, timeline_y, self.right, timeline_y), fill=GRID, width=8)
        progress = time / max(float(self.time[-1]), np.finfo(float).eps)
        draw.line(
            (self.left, timeline_y, self.left + progress * (self.right - self.left), timeline_y),
            fill=SURFACE,
            width=8,
        )
        if self.event_time is not None:
            event_progress = self.event_time / float(self.time[-1])
            event_timeline_x = self.left + event_progress * (self.right - self.left)
            draw.line(
                (event_timeline_x, timeline_y - 11, event_timeline_x, timeline_y + 11),
                fill=EVENT,
                width=3,
            )
        return image


def create_animation(
    input_path: Path,
    output_path: Path,
    title: str,
    event_x: float | None,
    event_time: float | None,
    frame_count: int,
    fps: int,
) -> None:
    with np.load(input_path) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    renderer = ScientificAnimation(data, title, event_x, event_time)
    indices = np.linspace(0, len(data["time"]) - 1, frame_count, dtype=int)
    rgb_frames = [renderer.draw_frame(int(index)) for index in indices]

    keyframe_dir = output_path.with_suffix("").with_name(
        output_path.stem + "_frames"
    )
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    key_positions = {
        "start": 0,
        "quarter": frame_count // 4,
        "event": int(
            np.argmin(np.abs(data["time"][indices] - event_time))
        )
        if event_time is not None
        else frame_count // 2,
        "three_quarters": 3 * frame_count // 4,
        "end": frame_count - 1,
    }
    for label, position in key_positions.items():
        rgb_frames[position].save(keyframe_dir / f"{label}.png", optimize=True)

    # A single fixed adaptive palette prevents frame-to-frame color changes.
    palette = rgb_frames[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in rgb_frames]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
        disposal=2,
        optimize=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--event-x", type=float)
    parser.add_argument("--event-time", type=float)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=18)
    args = parser.parse_args()
    if args.frames < 2 or args.fps < 1:
        raise ValueError("frames must be >= 2 and fps must be >= 1")
    create_animation(
        args.input,
        args.output,
        args.title,
        args.event_x,
        args.event_time,
        args.frames,
        args.fps,
    )


if __name__ == "__main__":
    main()
