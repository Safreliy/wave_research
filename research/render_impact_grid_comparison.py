"""Render same-time VOF masks across receiver levels with topology evidence.

The image is deliberately geometric rather than decorative: every panel uses
the same physical extent and an unsmoothed nearest-neighbour mask.  The labels
come from the machine-readable impact audit and distinguish an unfiltered gas
component from a component that passes the physical area and persistence gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from two_phase_basilisk.render_bie_handoff import liquid_mask, solid_mask


def frame_index(time: float, dt: float, frame_count: int) -> int:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if frame_count < 1:
        raise ValueError("at least one frame is required")
    return min(frame_count - 1, max(0, int(round(time / dt))))


def _state_image(
    frame: Path,
    *,
    bottom_x: np.ndarray,
    bottom_z: np.ndarray,
    length: float,
    geometry: dict[str, float],
) -> np.ndarray:
    liquid = liquid_mask(frame)
    solid = solid_mask(
        liquid.shape,
        bottom_x,
        bottom_z,
        length,
        geometry["x_min"],
        geometry["x_max"],
        geometry["y_min"],
        geometry["y_max"],
    )
    state = np.zeros(liquid.shape, dtype=np.uint8)
    state[liquid] = 1
    state[solid] = 2
    return state


def render(
    manifest_path: Path,
    audit_path: Path,
    output: Path,
    times: list[float],
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_by_level = {int(case["level"]): case for case in audit["cases"]}
    geometry = {key: float(value) for key, value in manifest["geometry"].items()}
    dt = float(manifest["thresholds"]["simulation_dt"])
    with np.load(manifest["handoff_npz"]) as source:
        bottom_x = np.asarray(source["bottom_x"], dtype=float)
        bottom_z = np.asarray(source["bottom_z"], dtype=float)
        grid_x = np.asarray(source["grid_x"], dtype=float)
        length = float(grid_x[-1] - grid_x[0] + grid_x[1] - grid_x[0])

    cases = sorted(manifest["cases"], key=lambda case: int(case["level"]))
    fig, axes = plt.subplots(
        len(cases), len(times),
        figsize=(4.25 * len(times), 2.35 * len(cases) + 0.85),
        squeeze=False,
        constrained_layout=True,
    )
    cmap = ListedColormap(["#f7fafc", "#3fa7d6", "#c9b285"])
    extent = [
        geometry["x_min"], geometry["x_max"],
        geometry["y_min"], geometry["y_max"],
    ]
    for row, case in enumerate(cases):
        level = int(case["level"])
        frames = sorted(Path(case["frame_directory"]).glob("vof-*.ppm"))
        if not frames:
            raise ValueError(f"no VOF frames in {case['frame_directory']}")
        primary = audit_by_level[level]["raster_results"][0]
        records = primary["records"]
        delta = float(audit_by_level[level]["delta"])
        for column, requested_time in enumerate(times):
            index = frame_index(requested_time, dt, len(frames))
            state = _state_image(
                frames[index], bottom_x=bottom_x, bottom_z=bottom_z,
                length=length, geometry=geometry,
            )
            ax = axes[row, column]
            ax.imshow(
                state, cmap=cmap, vmin=0, vmax=2, origin="upper",
                extent=extent, interpolation="nearest", aspect="auto",
            )
            record = records[min(index, len(records) - 1)]
            area_ratio = float(record["unfiltered_largest_pocket_area"]) / delta**2
            accepted_count = int(record["pocket_count"])
            ax.text(
                0.02, 0.04,
                rf"raw $A_{{max}}/\Delta^2={area_ratio:.3f}$; accepted={accepted_count}",
                transform=ax.transAxes, fontsize=8.5,
                bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
            )
            if row == 0:
                ax.set_title(rf"continuation $\Delta t={index * dt:.2f}$")
            if column == 0:
                ax.set_ylabel(f"level {level}\n$z/h_0$")
            else:
                ax.set_yticklabels([])
            if row == len(cases) - 1:
                ax.set_xlabel(r"$x/h_0$")
            else:
                ax.set_xticklabels([])
            ax.set_xlim(geometry["x_min"], geometry["x_max"])
            ax.set_ylim(geometry["y_min"], geometry["y_max"])
    fig.suptitle(
        "Conservative-q continuation: same-time grid comparison\n"
        "unsmoothed VOF masks; one-cell physical topology scale",
        fontsize=14, fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--times", type=float, nargs="+", default=[4.0, 5.1, 6.0])
    args = parser.parse_args()
    render(args.manifest, args.audit, args.output, args.times)


if __name__ == "__main__":
    main()
