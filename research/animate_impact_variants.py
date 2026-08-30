"""Render matched smooth-interface and point-vortex-sheet impact GIFs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from animate_lagrangian_bie import spectral_curve_resample
from animate_results import BACKGROUND, CREST, GRID, INK, MUTED, SURFACE, WATER, _font
from boundary_vortex_sheet import surface_vortex_points
from topographic_wave_solver import reference_energy


def _limits(
    records: list[tuple[float, dict[str, np.ndarray], int]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    crest_x_values, maximum_z = [], -math.inf
    for _, data, index in records:
        crest = int(np.argmax(data["z"][index]))
        crest_x_values.append(float(data["x"][index, crest]))
        maximum_z = max(maximum_z, float(np.max(data["z"][index])))
    crest_x = np.asarray(crest_x_values)
    x_limits = (float(np.min(crest_x) - 3.0), float(np.max(crest_x) + 3.5))
    z_limits = (
        max(
            max(float(np.max(data["bottom_z"])) for _, data, _ in records) - 0.12,
            -0.45,
        ),
        maximum_z + 0.14,
    )
    return x_limits, z_limits


def _energy_drift(data: dict[str, np.ndarray], index: int) -> float:
    initial_energy = float(data.get("global_initial_energy", data["energy"][0]))
    reference = float(data.get("energy_reference", np.asarray(0.0)))
    return (float(data["energy"][index]) - initial_energy) / max(
        abs(initial_energy - reference), np.finfo(float).eps
    )


def render(
    data: dict[str, np.ndarray],
    index: int,
    variant: str,
    x_limits: tuple[float, float],
    z_limits: tuple[float, float],
    circulation_scale: float,
    contact_reached: bool,
) -> Image.Image:
    width, height = 1200, 720
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 90.0, 1150.0, 118.0, 586.0
    title_font = _font(28, bold=True)
    metric_font = _font(18, bold=True)
    label_font = _font(15)
    small_font = _font(13)
    outcome = (
        "first-contact threshold reached"
        if contact_reached
        else "stopped by pre-contact resolution gate"
    )
    title = (
        f"Shoaling-to-steepening evolution · {outcome}"
        if variant == "smooth"
        else "Same Euler–BIE state · discrete boundary vortex sheet"
    )
    draw.text((left, 24), title, fill=INK, font=title_font)
    subtitle = (
        "spectral rendering of the parametric free boundary"
        if variant == "smooth"
        else "points carry panel circulation Γj=ΔΦj; they are not bulk vortices"
    )
    draw.text((left, 68), subtitle, fill=CREST if variant == "vortices" else MUTED, font=label_font)
    x_min, x_max = x_limits
    z_min, z_max = z_limits
    map_x = lambda value: left + (value - x_min) / (x_max - x_min) * (right - left)
    map_z = lambda value: bottom - (value - z_min) / (z_max - z_min) * (bottom - top)
    draw.rectangle((left, top, right, bottom), fill="white", outline=GRID, width=2)
    for tick in np.linspace(x_min, x_max, 6):
        px = map_x(float(tick))
        draw.line((px, top, px, bottom), fill="#edf1f5")
        draw.text((px - 17, bottom + 7), f"{tick:.1f}", fill=MUTED, font=small_font)
    for tick in np.linspace(z_min, z_max, 5):
        pz = map_z(float(tick))
        draw.line((left, pz, right, pz), fill="#edf1f5")
        draw.text((27, pz - 7), f"{tick:.2f}", fill=MUTED, font=small_font)
    length = float(data["length"])
    dense_x, dense_z = spectral_curve_resample(
        data["x"][index], data["z"][index], length, refinement=12
    )
    bed_x, bed_z = spectral_curve_resample(
        data["bottom_x"], data["bottom_z"], length, refinement=12
    )
    surface_points = [(map_x(float(x)), map_z(float(z))) for x, z in zip(dense_x, dense_z)]
    bed_points = [(map_x(float(x)), map_z(float(z))) for x, z in zip(bed_x, bed_z)]
    draw.polygon(surface_points + list(reversed(bed_points)), fill=WATER)
    draw.line(bed_points, fill="#75664d", width=5, joint="curve")
    if variant == "smooth":
        draw.line(surface_points, fill=SURFACE, width=5, joint="curve")
    else:
        draw.line(surface_points, fill="#9ab7c3", width=2, joint="curve")
        vortex_x, vortex_z, circulation = surface_vortex_points(
            data["x"][index],
            data["z"][index],
            data["potential"][index],
            length,
            float(data.get("background_current", np.asarray(0.0))),
        )
        for px_value, pz_value, gamma in zip(vortex_x, vortex_z, circulation):
            px, pz = map_x(float(px_value)), map_z(float(pz_value))
            if px < left - 10 or px > right + 10 or pz < top - 10 or pz > bottom + 10:
                continue
            radius = 2.0 + 7.0 * math.sqrt(min(abs(float(gamma)) / circulation_scale, 1.0))
            color = "#d94841" if gamma >= 0.0 else "#176b87"
            draw.ellipse((px - radius, pz - radius, px + radius, pz + radius), fill=color, outline="white", width=1)
    mapping = float(data["min_x_alpha"][index])
    normalized_gap = float(data.get("normalized_impact_distance", np.full(len(data["time"]), np.inf))[index])
    flux = abs(float(data["surface_flux_defect"][index]))
    drift = _energy_drift(data, index)
    validated = flux <= 2.0e-2 and abs(drift) <= 5.0e-3
    evidence_color = "#197a61" if validated else CREST
    evidence = "CONSERVATION GATES PASS" if validated else "EXPLORATORY: CONSERVATION GATE EXCEEDED"
    draw.text((left, 620), f"t={float(data['time'][index]):.3f} · min xα={mapping:+.3f}", fill=INK, font=metric_font)
    gap_text = "gap: pre-overturn" if not math.isfinite(normalized_gap) else f"nonlocal gap/Δs={normalized_gap:.3f}"
    draw.text((left + 390, 620), gap_text, fill=INK, font=label_font)
    draw.text((left + 730, 620), evidence, fill=evidence_color, font=label_font)
    draw.text(
        (left, 666),
        f"ΔE/Ewave={drift:+.2e} · |flux defect|={flux:.2e} · N={data['x'].shape[1]} · stop before topology change",
        fill=MUTED,
        font=small_font,
    )
    return image


def _save_gif(frames: list[Image.Image], output: Path, fps: int) -> None:
    palette = frames[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    output.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(
        output,
        save_all=True,
        append_images=indexed[1:],
        duration=round(1000 / fps),
        loop=0,
        disposal=2,
        optimize=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, nargs="+")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--start-time", type=float, default=-math.inf)
    parser.add_argument("--hard-max-flux", type=float, default=2.0e-2)
    parser.add_argument("--hard-max-energy-drift", type=float, default=5.0e-3)
    args = parser.parse_args()
    datasets = []
    for input_path in args.input:
        with np.load(input_path) as loaded:
            datasets.append({key: loaded[key] for key in loaded.files})
    earliest = min(datasets, key=lambda values: float(values["time"][0]))
    global_initial_energy = float(
        earliest.get("global_initial_energy", earliest["energy"][0])
    )
    global_target_volume = float(
        earliest.get("global_target_volume", earliest["volume"][0])
    )
    shared_reference = float(
        earliest.get(
            "energy_reference",
            np.asarray(
                reference_energy(
                    earliest["bottom_x"],
                    earliest["bottom_z"],
                    float(earliest["length"]),
                    float(earliest["gravity"]),
                    float(earliest.get("background_current", np.asarray(0.0))),
                )
            ),
        )
    )
    for data in datasets:
        data["global_initial_energy"] = np.asarray(global_initial_energy)
        data["global_target_volume"] = np.asarray(global_target_volume)
        data["energy_reference"] = np.asarray(shared_reference)
    unique_records = []
    for dataset_index, data in enumerate(datasets):
        cumulative_filter = float(
            data.get("cumulative_filter_correction_offset", np.asarray(0.0))
        ) + np.cumsum(data["spectral_filter_relative_corrections"])
        volume_error = np.abs(data["volume"] - global_target_volume) / max(
            abs(global_target_volume), np.finfo(float).eps
        )
        spacing_defect = (
            data["monitor_equidistribution_cv"]
            if float(data.get("reparameterization_strength", np.asarray(0.0))) > 0.0
            else data["marker_cv"]
        )
        next_start = (
            float(datasets[dataset_index + 1]["time"][0])
            if dataset_index + 1 < len(datasets)
            else math.inf
        )
        unique_records.extend(
            (float(data["time"][index]), data, index)
            for index in range(len(data["time"]))
            if (
                args.start_time <= float(data["time"][index]) < next_start
                and abs(float(data["surface_flux_defect"][index]))
                <= args.hard_max_flux
                and abs(_energy_drift(data, index))
                <= args.hard_max_energy_drift
                and volume_error[index] <= 5.0e-5
                and spacing_defect[index] <= 1.0e-1
                and cumulative_filter[index] <= 5.0e-2
                and np.all(np.isfinite(data["x"][index]))
                and np.all(np.isfinite(data["z"][index]))
            )
        )
    unique_records.sort(key=lambda item: item[0])
    positions = np.linspace(0, len(unique_records) - 1, args.frames, dtype=int)
    selected = [unique_records[int(position)] for position in positions]
    all_circulation = []
    for _, data, index in selected:
        all_circulation.append(
            surface_vortex_points(
                data["x"][index],
                data["z"][index],
                data["potential"][index],
                float(data["length"]),
                float(data.get("background_current", np.asarray(0.0))),
            )[2]
        )
    circulation_scale = max(
        float(np.quantile(np.abs(np.concatenate(all_circulation)), 0.98)),
        np.finfo(float).eps,
    )
    contact_reached = any(
        math.isfinite(
            float(
                data.get(
                    "normalized_impact_distance",
                    np.full(len(data["time"]), np.inf),
                )[index]
            )
        )
        and float(data["normalized_impact_distance"][index])
        <= float(data.get("impact_distance_factor", np.asarray(0.15)))
        for _, data, index in selected
    )
    x_limits, z_limits = _limits(selected)
    smooth = [
        render(
            data,
            int(index),
            "smooth",
            x_limits,
            z_limits,
            circulation_scale,
            contact_reached,
        )
        for _, data, index in selected
    ]
    vortices = [
        render(
            data,
            int(index),
            "vortices",
            x_limits,
            z_limits,
            circulation_scale,
            contact_reached,
        )
        for _, data, index in selected
    ]
    _save_gif(smooth, args.output_prefix.with_name(args.output_prefix.name + "_smooth.gif"), args.fps)
    _save_gif(vortices, args.output_prefix.with_name(args.output_prefix.name + "_vortices.gif"), args.fps)
    frame_dir = args.output_prefix.with_name(args.output_prefix.name + "_keyframes")
    frame_dir.mkdir(parents=True, exist_ok=True)
    final_label = "contact" if contact_reached else "last_admissible"
    for name, position in zip(("start", "overturn", final_label), np.linspace(0, len(smooth) - 1, 3, dtype=int)):
        smooth[int(position)].save(frame_dir / f"{name}_smooth.png")
        vortices[int(position)].save(frame_dir / f"{name}_vortices.png")


if __name__ == "__main__":
    main()
