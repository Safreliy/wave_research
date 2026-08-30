"""Audit geometry and event metadata in the open Pick--Feddersen examples.

This deliberately separates the archived example configurations from the
parameter matrix reported in the final JFM article.  It is evidence plumbing,
not a claim that the examples reproduce every published production run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jfm2026_benchmark import (
    ARTICLE_DOI,
    CASES,
    GRAVITY,
    ZENODO_DOI,
    CACHE,
    RESULTS,
    extract_archive,
    fetch_archive,
    load_case,
)


def infer_bed_geometry(boundary: np.ndarray) -> dict[str, float]:
    """Infer flat reaches and mean upslope from an ordered tank boundary."""
    boundary = np.asarray(boundary, dtype=float)
    if boundary.ndim != 2 or boundary.shape[1] != 2 or len(boundary) < 4:
        raise ValueError("boundary must be an N-by-2 polyline")
    delta = np.diff(boundary, axis=0)
    domain_length = float(np.ptp(boundary[:, 0]))
    horizontal_motion = np.abs(delta[:, 0]) > max(
        1.0e-12, 1.0e-8 * domain_length
    )
    segment_x0 = boundary[:-1, 0][horizontal_motion]
    segment_x1 = boundary[1:, 0][horizontal_motion]
    x = 0.5 * (segment_x0 + segment_x1)
    z = 0.5 * (boundary[:-1, 1] + boundary[1:, 1])[horizontal_motion]
    order = np.argsort(x)
    x, z = x[order], z[order]
    deep_z = float(np.min(z))
    shelf_z = float(np.max(z))
    vertical_range = shelf_z - deep_z
    if vertical_range <= 0.0 or shelf_z >= -1.0e-8:
        raise ValueError("could not distinguish deep bed and shallow shelf")
    tolerance = max(1.0e-6, 2.0e-3 * vertical_range)
    deep_mask = np.abs(z - deep_z) <= tolerance
    shelf_mask = np.abs(z - shelf_z) <= tolerance
    if not np.any(deep_mask) or not np.any(shelf_mask):
        raise ValueError("flat bed reaches were not resolved")
    toe = float(
        np.max(np.maximum(segment_x0[order][deep_mask], segment_x1[order][deep_mask]))
    )
    shelf_start = float(
        np.min(np.minimum(segment_x0[order][shelf_mask], segment_x1[order][shelf_mask]))
    )
    slope_length = shelf_start - toe
    if slope_length <= 0.0:
        raise ValueError("invalid inferred upslope interval")
    return {
        "domain_length_over_h0": domain_length / (-deep_z),
        "deep_depth_h0": -deep_z,
        "shallow_depth_over_h0": -shelf_z / (-deep_z),
        "toe_over_h0": toe / (-deep_z),
        "shelf_start_over_h0": shelf_start / (-deep_z),
        "slope_length_over_h0": slope_length / (-deep_z),
        "effective_slope": vertical_range / slope_length,
    }


def surface_full_width(surface: np.ndarray, h0: float, threshold: float = 1e-3) -> float:
    """Return the x-span with eta/h0 at least ``threshold``."""
    surface = np.asarray(surface, dtype=float)
    active = surface[:, 1] / h0 >= threshold
    if not np.any(active):
        return 0.0
    return float(np.ptp(surface[active, 0]) / h0)


def is_overturned(surface: np.ndarray) -> bool:
    dx = np.diff(np.asarray(surface, dtype=float)[:, 0])
    tolerance = max(1.0e-10, 1.0e-8 * float(np.ptp(surface[:, 0])))
    nonzero = dx[np.abs(dx) > tolerance]
    return bool(len(nonzero) and np.min(nonzero) < 0.0 < np.max(nonzero))


def audit_case(frames: list[dict[str, np.ndarray | float]], case: str) -> dict[str, object]:
    initial_surface = np.asarray(frames[0]["surface"], dtype=float)
    geometry = infer_bed_geometry(np.asarray(frames[0]["boundary"], dtype=float))
    h0 = float(geometry["deep_depth_h0"])
    crest = initial_surface[int(np.argmax(initial_surface[:, 1]))]
    spacing = float(np.median(np.abs(np.diff(initial_surface[:, 0]))) / h0)
    first_overturning = next(
        (index for index, frame in enumerate(frames) if is_overturned(np.asarray(frame["surface"]))),
        None,
    )
    event = None
    if first_overturning is not None:
        frame = frames[first_overturning]
        event = {
            "frame": first_overturning,
            "time_seconds": float(frame["time_dimensional"]),
            "time_nondimensional": float(frame["time"]),
        }
    return {
        "evidence_class": "open Zenodo example audit",
        "case": case,
        "archive_directory": CASES[case],
        "article_doi": ARTICLE_DOI,
        "data_doi": ZENODO_DOI,
        "geometry": geometry,
        "initial_surface_spacing_over_h0": spacing,
        "initial_amplitude_over_h0": float(crest[1] / h0),
        "initial_crest_x_over_h0": float(crest[0] / h0),
        "initial_full_width_over_h0_eta_ge_0p001": surface_full_width(initial_surface, h0),
        "frame_count": len(frames),
        "first_time_seconds": float(frames[0]["time_dimensional"]),
        "last_time_seconds": float(frames[-1]["time_dimensional"]),
        "first_time_nondimensional": float(frames[0]["time"]),
        "last_time_nondimensional": float(frames[-1]["time"]),
        "first_overturning": event,
        "time_conversion": f"t*sqrt({GRAVITY}/h0)",
        "comparison_scope": (
            "qualitative/profile regression only unless these inferred "
            "parameters match the separately declared solver case"
        ),
    }


def audit_all(directory: Path) -> dict[str, object]:
    return {
        "schema": "jfm2026-open-example-audit-v1",
        "warning": (
            "The open examples are not assumed to be identical to the final "
            "article's production parameter matrix."
        ),
        "cases": {
            case: audit_case(load_case(directory, case), case) for case in CASES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=CACHE / "mat.zip")
    parser.add_argument("--cache-directory", type=Path, default=CACHE / "extracted")
    parser.add_argument(
        "--output", type=Path, default=RESULTS / "jfm2026_open_examples_audit.json"
    )
    args = parser.parse_args()
    root = extract_archive(fetch_archive(args.archive), args.cache_directory)
    report = audit_all(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
