"""Fetch and render the Pick--Feddersen (JFM 2026) breaker benchmark.

The generated animation is a *reference result*, not output of this project.
It supplies the missing visual and quantitative target for the new solver:
shoaling, overturning, inward roll-up, and the last pre-impact profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat

from animate_results import BACKGROUND, GRID, INK, MUTED, SURFACE, WATER, _font


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache" / "jfm2026"
RESULTS = ROOT / "results" / "publication_package"
ARCHIVE_URL = (
    "https://zenodo.org/api/records/17716653/files/mat.zip/content"
)
ARCHIVE_MD5 = "71dbbef0460cc8698ea6b1e2c462b800"
ZENODO_DOI = "10.5281/zenodo.17716653"
ARTICLE_DOI = "10.1017/jfm.2026.11869"
GRAVITY = 9.81
CASES = {
    "gentle_1_50_H0_0p6": "mat_planar_slope_0.02_a0_0.6",
    "benchmark_1_30_H0_0p5": "mat_planar_slope_0.033333_a0_0.5",
    "steep_1_15_H0_0p3": "mat_planar_slope_0.066667_a0_0.3",
    "steep_1_10_H0_0p5": "mat_planar_slope_0.1_a0_0.5",
}


def _checksum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_archive(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _checksum(destination) == ARCHIVE_MD5:
        return destination
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={"User-Agent": "wave-simulation-reproducibility/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    if hashlib.md5(payload).hexdigest() != ARCHIVE_MD5:
        raise RuntimeError("Zenodo archive checksum mismatch")
    destination.write_bytes(payload)
    return destination


def extract_archive(archive: Path, directory: Path) -> Path:
    root = directory.resolve()
    marker = directory / ".complete"
    if marker.exists():
        return directory / "mat_ex_zenodo"
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (directory / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError("unsafe path in benchmark archive")
        bundle.extractall(directory)
    marker.write_text(ARCHIVE_MD5 + "\n", encoding="ascii")
    return directory / "mat_ex_zenodo"


def load_case(directory: Path, case: str) -> list[dict[str, np.ndarray | float]]:
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}")
    files = sorted((directory / CASES[case]).glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"no MAT frames for {case}")
    frames: list[dict[str, np.ndarray | float]] = []
    for path in files:
        raw = loadmat(path, squeeze_me=True)
        boundary = np.asarray(raw["boundaries"], dtype=float)
        deep_depth = infer_deep_depth(boundary)
        dimensional_time = float(raw["time"])
        frames.append(
            {
                "surface": np.asarray(raw["XZ_FS"], dtype=float),
                "boundary": boundary,
                # The Zenodo MAT files store seconds.  ``time`` is retained as
                # the paper's nondimensional t*sqrt(g/h0), while the raw value
                # remains available for a fully auditable conversion.
                "time_dimensional": dimensional_time,
                "time": dimensional_time * np.sqrt(GRAVITY / deep_depth),
                "deep_depth": deep_depth,
                "energy": float(raw["E"]),
                "flux": float(raw["flux"]),
            }
        )
    return frames


def infer_deep_depth(boundary: np.ndarray) -> float:
    """Infer h0 from the lowest horizontal bed segment in a closed boundary."""
    boundary = np.asarray(boundary, dtype=float)
    if boundary.ndim != 2 or boundary.shape[1] != 2 or len(boundary) < 4:
        raise ValueError("boundary must be an N-by-2 polyline")
    delta = np.diff(boundary, axis=0)
    domain_length = float(np.ptp(boundary[:, 0]))
    horizontal = np.abs(delta[:, 0]) > max(1.0e-12, 1.0e-8 * domain_length)
    if not np.any(horizontal):
        raise ValueError("boundary contains no resolvable bed segment")
    bed_z = 0.5 * (boundary[:-1, 1] + boundary[1:, 1])[horizontal]
    depth = -float(np.min(bed_z))
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError("could not infer a positive deep-water depth")
    return depth


def _polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    fill: str,
    width: int,
) -> None:
    if len(points) > 1:
        draw.line(points, fill=fill, width=width, joint="curve")


def render_frame(
    frame: dict[str, np.ndarray | float],
    final_crest_x: float,
    limits: tuple[float, float, float, float],
    case_label: str,
    show_markers: bool,
) -> Image.Image:
    width, height = 1200, 760
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 92.0, 1150.0, 130.0, 610.0
    x_min, x_max, z_min, z_max = limits
    map_x = lambda value: left + (value - x_min) / (x_max - x_min) * (right - left)
    map_z = lambda value: bottom - (value - z_min) / (z_max - z_min) * (bottom - top)
    draw.text(
        (left, 24),
        "Reference plunging breaker: inward roll-up to impact",
        fill=INK,
        font=_font(28, bold=True),
    )
    draw.text(
        (left, 69),
        "Pick--Feddersen JFM 2026 / surftank -- reference data, not the new solver",
        fill="#b45309",
        font=_font(17, bold=True),
    )
    draw.text((left, 101), case_label, fill=MUTED, font=_font(15))
    draw.rectangle((left, top, right, bottom), fill="white", outline=GRID, width=2)
    for tick in np.linspace(x_min, x_max, 6):
        px = map_x(float(tick))
        draw.line((px, top, px, bottom), fill="#edf1f5", width=1)
        draw.text((px - 19, bottom + 8), f"{tick - final_crest_x:+.1f}", fill=MUTED, font=_font(12))
    for tick in np.linspace(z_min, z_max, 5):
        py = map_z(float(tick))
        draw.line((left, py, right, py), fill="#edf1f5", width=1)
        draw.text((30, py - 8), f"{tick:.2f}", fill=MUTED, font=_font(12))

    surface = np.asarray(frame["surface"])
    # Data are stored from the right wall to the left wall.  A broad polygon
    # clipped by the plotting rectangle supplies the water fill.
    surface_points = [
        (map_x(float(x)), map_z(float(z))) for x, z in surface
    ]
    water_bottom = map_z(z_min)
    polygon = surface_points + [
        (surface_points[-1][0], water_bottom),
        (surface_points[0][0], water_bottom),
    ]
    draw.polygon(polygon, fill=WATER)
    _polyline(draw, surface_points, fill=SURFACE, width=5)
    if show_markers:
        local = (
            (surface[:, 0] >= x_min)
            & (surface[:, 0] <= x_max)
            & (surface[:, 1] >= z_min)
            & (surface[:, 1] <= z_max)
        )
        for x, z in surface[local][::2]:
            px, py = map_x(float(x)), map_z(float(z))
            draw.ellipse((px - 2.5, py - 2.5, px + 2.5, py + 2.5), fill="#f59e0b")

    crest = int(np.argmax(surface[:, 1]))
    crest_x, crest_z = surface[crest]
    draw.ellipse(
        (
            map_x(float(crest_x)) - 6,
            map_z(float(crest_z)) - 6,
            map_x(float(crest_x)) + 6,
            map_z(float(crest_z)) + 6,
        ),
        fill="#f59e0b",
        outline="white",
        width=2,
    )
    direction = np.diff(surface[:, 0])
    overturned = bool(np.max(direction) > 0.0 and np.min(direction) < 0.0)
    stage = "overturning / plunging" if overturned else "shoaling / steepening"
    draw.text(
        (left, 654),
        (
            f"t sqrt(g/h0)={float(frame['time']):.3f}  |  {stage}  |  "
            f"N_FS={len(surface)}"
        ),
        fill=INK,
        font=_font(17, bold=True),
    )
    draw.text(
        (left, 690),
        f"reported flux={float(frame['flux']):.2e}  |  x-axis is relative to final crest",
        fill=MUTED,
        font=_font(14),
    )
    return image


def build_animation(
    frames: list[dict[str, np.ndarray | float]],
    output: Path,
    case: str,
    frame_count: int,
    fps: int,
    show_markers: bool,
) -> dict[str, object]:
    final_surface = np.asarray(frames[-1]["surface"])
    final_crest_x = float(final_surface[np.argmax(final_surface[:, 1]), 0])
    z_max = max(float(np.max(np.asarray(frame["surface"])[:, 1])) for frame in frames)
    limits = (
        final_crest_x - 2.2,
        final_crest_x + 1.7,
        -0.12,
        z_max + 0.10,
    )
    indices = np.unique(
        np.linspace(0, len(frames) - 1, frame_count, dtype=int)
    )
    label = case.replace("_", " ")
    images = [
        render_frame(
            frames[int(index)],
            final_crest_x,
            limits,
            label,
            show_markers,
        )
        for index in indices
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    palette = images[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    gif = [
        item.quantize(palette=palette, dither=Image.Dither.NONE)
        for item in images
    ]
    gif[0].save(
        output,
        save_all=True,
        append_images=gif[1:],
        duration=round(1000 / fps),
        loop=0,
        disposal=2,
        optimize=False,
    )
    keyframes = output.with_suffix("").with_name(output.stem + "_keyframes")
    keyframes.mkdir(parents=True, exist_ok=True)
    for name, position in zip(
        ("shoaling", "overturning", "plunging", "preimpact"),
        np.linspace(0, len(images) - 1, 4, dtype=int),
    ):
        images[int(position)].save(keyframes / f"{name}.png", optimize=True)
    report = {
        "evidence_class": "external reference benchmark",
        "scope": (
            "qualitative/profile reference; archive geometry must be audited "
            "before comparison with the final article parameter matrix"
        ),
        "case": case,
        "source_frames": len(frames),
        "rendered_frames": len(images),
        "time_normalization": "t_nondimensional = t_seconds*sqrt(9.81/h0)",
        "deep_depth": float(frames[0]["deep_depth"]),
        "first_time_seconds": float(frames[0]["time_dimensional"]),
        "last_time_seconds": float(frames[-1]["time_dimensional"]),
        "first_time_nondimensional": float(frames[0]["time"]),
        "last_time_nondimensional": float(frames[-1]["time"]),
        "article_doi": ARTICLE_DOI,
        "data_doi": ZENODO_DOI,
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), default="gentle_1_50_H0_0p6")
    parser.add_argument("--archive", type=Path, default=CACHE / "mat.zip")
    parser.add_argument("--cache-directory", type=Path, default=CACHE / "extracted")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "jfm2026_reference_plunging.gif",
    )
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--markers", action="store_true")
    args = parser.parse_args()
    archive = fetch_archive(args.archive)
    directory = extract_archive(archive, args.cache_directory)
    frames = load_case(directory, args.case)
    report = build_animation(
        frames, args.output, args.case, args.frames, args.fps, args.markers
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
