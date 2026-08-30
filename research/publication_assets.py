"""Build the numerical evidence table and a publication summary figure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from animate_results import BACKGROUND, GRID, INK, MUTED, _font
from publication_metrics import _geometry, analyze_archive
from publication_study import _crest_x, _translation_aligned_shape_error
from topographic_wave_solver import reference_energy


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {key: loaded[key] for key in loaded.files}


def relative_energy_error(data: dict[str, np.ndarray]) -> np.ndarray:
    if "energy_reference" in data:
        reference = float(data["energy_reference"])
    elif "bottom_x" in data:
        reference = reference_energy(
            data["bottom_x"],
            data["bottom_z"],
            float(data["length"]),
            float(data["gravity"]),
            float(data.get("background_current", np.asarray(0.0))),
        )
    else:
        reference = 0.0
    scale = max(
        abs(float(data["energy"][0]) - reference), np.finfo(float).eps
    )
    return (data["energy"] - data["energy"][0]) / scale


def flat_record(path: Path) -> dict[str, float | int | str]:
    data = load_archive(path)
    length = float(data["length"])
    final_time = float(data["time"][-1])
    initial_x = _crest_x(data["x"][0], data["z"][0], length)
    final_x = _crest_x(data["x"][-1], data["z"][-1], length)
    displacement = (final_x - initial_x + 0.5 * length) % length - 0.5 * length
    measured_speed = displacement / final_time
    target_speed = float(
        data.get(
            "initial_wave_speed",
            np.asarray(
                math.sqrt(
                    float(data["gravity"])
                    * (float(data["depth"]) + float(np.max(data["z"][0])))
                )
            ),
        )
    )
    return {
        "source": str(path.relative_to(ROOT)),
        "n": int(data["x"].shape[1]),
        "dt": float(data["dt"]),
        "final_time": final_time,
        "initializer": str(data["initial_condition"]),
        "shape_relative_l2_error": _translation_aligned_shape_error(
            data["x"][0],
            data["z"][0],
            data["x"][-1],
            data["z"][-1],
            length,
        ),
        "measured_speed": measured_speed,
        "target_speed": target_speed,
        "relative_speed_error": abs(measured_speed - target_speed) / target_speed,
        "maximum_relative_energy_drift": float(
            np.max(np.abs(relative_energy_error(data)))
        ),
    }


def remeshing_record(path: Path, stop_time: float) -> dict[str, float | str]:
    data = load_archive(path)
    active = data["time"] <= stop_time + 1.0e-12
    return {
        "source": str(path.relative_to(ROOT)),
        "final_time_used": float(data["time"][active][-1]),
        "minimum_x_alpha": float(np.min(data["min_x_alpha"][active])),
        "maximum_marker_spacing_cv": float(np.max(data["marker_cv"][active])),
        "maximum_surface_flux_defect": float(
            np.max(np.abs(data["surface_flux_defect"][active]))
        ),
        "maximum_relative_energy_drift": float(
            np.max(np.abs(relative_energy_error(data)[active]))
        ),
    }


def build_report() -> dict[str, object]:
    exact_directory = RESULTS / "publication_study_exact"
    old_directory = RESULTS / "publication_study_flat"
    exact_spatial = [
        flat_record(exact_directory / f"flat_solitary_n{n}.npz")
        for n in (32, 48, 64)
    ]
    asymptotic_spatial = [
        flat_record(old_directory / f"flat_solitary_n{n}.npz")
        for n in (32, 48, 64)
    ]
    exact_temporal = [
        flat_record(exact_directory / f"flat_solitary_dt{token}.npz")
        for token in ("0p008", "0p004", "0p002", "0p001")
    ]
    close_report = json.loads(
        (RESULTS / "publication_study" / "publication_study.json").read_text(
            encoding="utf-8"
        )
    )["close_quadrature_ablation"]
    stop_time = 5.089134289439374
    without_remeshing = remeshing_record(
        RESULTS / "reef_exact_a06_n96_first_overturn.npz", stop_time
    )
    with_remeshing = remeshing_record(
        RESULTS / "reef_exact_a06_n96_remesh_ablation.npz", stop_time
    )
    validated_shoaling = analyze_archive(
        RESULTS / "reef_exact_a06_n96_physical_break.npz",
        compute_condition_number=False,
    )
    exploratory = analyze_archive(
        RESULTS / "reef_exact_a06_n96_visual_jet.npz",
        compute_condition_number=False,
    )
    implicit_ablation = json.loads(
        (RESULTS / "publication_package" / "implicit_ablation.json").read_text(
            encoding="utf-8"
        )
    )
    impact_resolution = json.loads(
        (
            RESULTS
            / "publication_package"
            / "impact_resolution_ablation.json"
        ).read_text(encoding="utf-8")
    )
    near_self = json.loads(
        (
            RESULTS
            / "publication_package"
            / "near_self_quadrature_ablation.json"
        ).read_text(encoding="utf-8")
    )
    vof_topology = json.loads(
        (
            RESULTS
            / "publication_package"
            / "basilisk_vof_topology.json"
        ).read_text(encoding="utf-8")
    )
    exact_n64 = exact_spatial[-1]
    p0 = (
        validated_shoaling["gate_P0_trajectory"] == "pass"
        and max(
            row["relative_q_error_vs_factor16"]
            for row in close_report
            if row["method"] == "adaptive"
        )
        < 1.0e-6
        and min(
            row["relative_normal_derivative_error"]
            for row in near_self["rows"]
            if row["n"] == 128 and row["factor"] >= 8
        )
        < 4.0e-2
    )
    p1 = (
        exact_n64["shape_relative_l2_error"] < 1.0e-3
        and exact_n64["relative_speed_error"] < 1.0e-3
    )
    return {
        "evidence_class": {
            "flat_translation": "validated computational result",
            "close_quadrature": "validated manufactured/ablation result",
            "remeshing": "validated numerical ablation",
            "reef_1_10_until_t9_49": "validated nonbreaking trajectory",
            "late_overhang_1_10": "exploratory computational observation",
            "implicit_ablation": "validated stability/cost ablation",
            "boundary_vortex_points": "visual representation, not bulk vorticity",
            "legacy_first_contact": (
                "invalidated; closest panels were local three-index folds"
            ),
            "topological_first_contact": (
                "open in BIE; now requires 12-panel arclength separation"
            ),
            "JFM_2026_animation": "external reference benchmark",
            "standalone_VOF_topology": (
                "one-resolution receiving-solver result, not BIE output"
            ),
        },
        "flat_spatial_exact": exact_spatial,
        "flat_spatial_third_order": asymptotic_spatial,
        "flat_temporal_exact": exact_temporal,
        "close_quadrature_ablation": close_report,
        "near_self_quadrature_ablation": near_self,
        "remeshing_ablation": {
            "without_spectral_remeshing": without_remeshing,
            "with_spectral_remeshing_every_4_steps": with_remeshing,
        },
        "validated_shoaling": validated_shoaling,
        "exploratory_overhang": exploratory,
        "implicit_ablation": implicit_ablation,
        "impact_resolution_ablation": impact_resolution,
        "standalone_vof_topology": {
            "criterion": vof_topology["criterion"],
            "minimum_component_pixels": vof_topology[
                "minimum_component_pixels"
            ],
            "frame_count": vof_topology["frame_count"],
            "impact_detected": vof_topology["impact_detected"],
            "impact_time": vof_topology["impact_time"],
            "impact_frame": vof_topology["impact_frame"],
        },
        "publication_gates": {
            "P0_operator_correctness": "pass" if p0 else "fail",
            "P1_solitary_baseline": "pass" if p1 else "fail",
            "P2_JFM_1_30_grid_convergence": "open",
            "P3_BIE_valid_nonlocal_contact": "open",
            "P4_coupled_VOF_grid_convergence": (
                "open; standalone receiver smoke test passes"
            ),
        },
    }


def _panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    x_label: str,
    y_label: str,
) -> tuple[float, float, float, float]:
    left, top, right, bottom = box
    title_font = _font(20, bold=True)
    label_font = _font(13)
    draw.rounded_rectangle(box, radius=10, fill="white", outline=GRID, width=2)
    draw.text((left + 18, top + 12), title, fill=INK, font=title_font)
    plot = (left + 76, top + 58, right - 24, bottom - 58)
    draw.line((plot[0], plot[3], plot[2], plot[3]), fill=INK, width=2)
    draw.line((plot[0], plot[1], plot[0], plot[3]), fill=INK, width=2)
    draw.text((0.5 * (plot[0] + plot[2]) - 28, bottom - 35), x_label, fill=MUTED, font=label_font)
    draw.text((left + 12, top + 42), y_label, fill=MUTED, font=label_font)
    return plot


def _log_map(value: float, minimum: float, maximum: float, high: float, low: float) -> float:
    ratio = (math.log10(value) - math.log10(minimum)) / (
        math.log10(maximum) - math.log10(minimum)
    )
    return low - ratio * (low - high)


def _line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: str,
    width: int = 4,
) -> None:
    if len(points) > 1:
        draw.line(points, fill=color, width=width, joint="curve")
    for x, y in points:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="white", width=1)


def draw_figure(report: dict[str, object], output: Path) -> None:
    image = Image.new("RGB", (1600, 1050), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((55, 22), "Publication evidence summary", fill=INK, font=_font(30, bold=True))
    draw.text(
        (55, 64),
        "Euler–BIE free-surface solver · all curves regenerated from archived NPZ data",
        fill=MUTED,
        font=_font(16),
    )
    boxes = (
        (45, 105, 785, 535),
        (815, 105, 1555, 535),
        (45, 565, 785, 995),
        (815, 565, 1555, 995),
    )

    plot = _panel(draw, boxes[0], "A  Exact Euler initial state", "markers N", "shape L2 error (log)")
    n_values = (32, 48, 64)
    x_map = lambda n: plot[0] + (n - 32) / 32 * (plot[2] - plot[0])
    for exponent in (-5, -4, -3, -2):
        y = _log_map(10.0**exponent, 1.0e-5, 2.0e-2, plot[1], plot[3])
        draw.line((plot[0], y, plot[2], y), fill="#edf1f5", width=1)
        draw.text((plot[0] - 53, y - 7), f"1e{exponent}", fill=MUTED, font=_font(12))
    for n in n_values:
        draw.text((x_map(n) - 9, plot[3] + 9), str(n), fill=MUTED, font=_font(12))
    for key, color, label in (
        ("flat_spatial_exact", "#087ea4", "Babenko exact"),
        ("flat_spatial_third_order", "#ef4444", "third order"),
    ):
        points = [
            (
                x_map(int(row["n"])),
                _log_map(float(row["shape_relative_l2_error"]), 1.0e-5, 2.0e-2, plot[1], plot[3]),
            )
            for row in report[key]
        ]
        _line(draw, points, color)
        draw.text((plot[2] - 165, plot[1] + (18 if label.startswith("B") else 42)), label, fill=color, font=_font(13, bold=True))

    plot = _panel(draw, boxes[1], "B  Close-boundary quadrature", "markers N", "relative q error (log)")
    rows = report["close_quadrature_ablation"]
    for exponent in (-9, -7, -5, -3, -1):
        y = _log_map(10.0**exponent, 1.0e-10, 1.0, plot[1], plot[3])
        draw.line((plot[0], y, plot[2], y), fill="#edf1f5", width=1)
        draw.text((plot[0] - 53, y - 7), f"1e{exponent}", fill=MUTED, font=_font(12))
    x_map_b = lambda n: plot[0] + (n - 24) / 56 * (plot[2] - plot[0])
    for method, color in (("plain", "#ef4444"), ("adaptive", "#087ea4")):
        active = [row for row in rows if row["method"] == method]
        points = [
            (
                x_map_b(int(row["n"])),
                _log_map(max(float(row["relative_q_error_vs_factor16"]), 1.0e-10), 1.0e-10, 1.0, plot[1], plot[3]),
            )
            for row in active
        ]
        _line(draw, points, color)
        draw.text((plot[2] - 145, plot[1] + (18 if method == "adaptive" else 42)), method, fill=color, font=_font(13, bold=True))

    plot = _panel(draw, boxes[2], "C  Remeshing ablation, N=96", "time", "marker CV (log)")
    unstable = load_archive(RESULTS / "reef_exact_a06_n96_first_overturn.npz")
    remeshed = load_archive(RESULTS / "reef_exact_a06_n96_remesh_ablation.npz")
    x_map_c = lambda t: plot[0] + (t - 2.0) / 3.5 * (plot[2] - plot[0])
    for exponent in (-4, -3, -2, -1):
        y = _log_map(10.0**exponent, 1.0e-4, 0.3, plot[1], plot[3])
        draw.line((plot[0], y, plot[2], y), fill="#edf1f5", width=1)
        draw.text((plot[0] - 53, y - 7), f"1e{exponent}", fill=MUTED, font=_font(12))
    for data, color, label in (
        (unstable, "#ef4444", "theta-s only"),
        (remeshed, "#087ea4", "+ spectral remesh"),
    ):
        active = (data["time"] >= 2.0) & (data["time"] <= 5.5)
        points = [
            (
                x_map_c(float(t)),
                _log_map(max(float(cv), 1.0e-4), 1.0e-4, 0.3, plot[1], plot[3]),
            )
            for t, cv in zip(data["time"][active], data["marker_cv"][active])
        ]
        if len(points) > 2:
            draw.line(points, fill=color, width=4)
        draw.text((plot[2] - 180, plot[1] + (18 if label.startswith("+") else 42)), label, fill=color, font=_font(13, bold=True))

    plot = _panel(draw, boxes[3], "D  Validated shoaling trajectory", "time", "normalized drift")
    shoal = load_archive(RESULTS / "reef_exact_a06_n96_physical_break.npz")
    x_alpha, _, _ = _geometry(shoal["x"][0], shoal["z"][0], float(shoal["length"]))
    wave_volume = 2.0 * math.pi / shoal["x"].shape[1] * np.sum(shoal["z"][0] * x_alpha)
    energy_error = np.abs(relative_energy_error(shoal))
    volume_error = np.abs((shoal["volume"] - shoal["volume"][0]) / wave_volume)
    x_map_d = lambda t: plot[0] + float(t) / float(shoal["time"][-1]) * (plot[2] - plot[0])
    for exponent in (-9, -7, -5, -3):
        y = _log_map(10.0**exponent, 1.0e-10, 1.0e-2, plot[1], plot[3])
        draw.line((plot[0], y, plot[2], y), fill="#edf1f5", width=1)
        draw.text((plot[0] - 53, y - 7), f"1e{exponent}", fill=MUTED, font=_font(12))
    for values, color, label in (
        (energy_error, "#087ea4", "energy"),
        (volume_error, "#f59e0b", "wave volume"),
    ):
        points = [
            (
                x_map_d(t),
                _log_map(max(float(value), 1.0e-10), 1.0e-10, 1.0e-2, plot[1], plot[3]),
            )
            for t, value in zip(shoal["time"], values)
        ]
        draw.line(points, fill=color, width=4)
        draw.text((plot[2] - 150, plot[1] + (18 if label == "energy" else 42)), label, fill=color, font=_font(13, bold=True))

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=RESULTS / "publication_package",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    report = build_report()
    safe_report = _json_safe(report)
    (args.output_directory / "publication_results.json").write_text(
        json.dumps(safe_report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    draw_figure(report, args.output_directory / "publication_summary.png")
    print(json.dumps(report["publication_gates"], indent=2))


if __name__ == "__main__":
    main()
