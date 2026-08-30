"""Plot publication diagnostics for the physical 1:30 JFM benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def admissible_prefix(data: dict[str, np.ndarray]) -> tuple[np.ndarray, int | None]:
    initial_energy = float(
        data.get("global_initial_energy", np.asarray(data["energy"][0]))
    )
    initial_volume = float(
        data.get("global_target_volume", np.asarray(data["volume"][0]))
    )
    energy_scale = max(
        abs(float(initial_energy - data["energy_reference"])),
        np.finfo(float).eps,
    )
    energy = np.abs(data["energy"] - initial_energy) / energy_scale
    volume = np.abs(data["volume"] - initial_volume) / abs(initial_volume)
    cumulative_filter = float(
        data.get("cumulative_filter_correction_offset", np.asarray(0.0))
    ) + np.cumsum(data["spectral_filter_relative_corrections"])
    spacing = (
        data["monitor_equidistribution_cv"]
        if float(data["reparameterization_strength"]) > 0.0
        else data["marker_cv"]
    )
    accepted = (
        (energy <= 5.0e-3)
        & (volume <= 5.0e-5)
        & (np.abs(data["surface_flux_defect"]) <= 2.0e-2)
        & (spacing <= 1.0e-1)
        & (cumulative_filter <= 5.0e-2)
    )
    failure = np.flatnonzero(~accepted)
    stop = int(failure[0]) if len(failure) else len(accepted)
    return np.arange(max(stop, 1)), (stop if len(failure) else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("unfiltered_n128", type=Path)
    parser.add_argument("filtered_n128", type=Path)
    parser.add_argument("filtered_n192", type=Path)
    parser.add_argument("refined_n256", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    series = [
        ("N=128, no filter", load(args.unfiltered_n128), "#c2410c"),
        ("N=128, σ=4", load(args.filtered_n128), "#b7791f"),
        ("N=192, σ=4", load(args.filtered_n192), "#087f5b"),
        ("N=256, σ=4 (refined at t=18)", load(args.refined_n256), "#2563eb"),
    ]

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.8), constrained_layout=True)
    for label, data, color in series:
        indices, failure = admissible_prefix(data)
        time = data["time"][indices]
        initial_energy = float(
            data.get("global_initial_energy", np.asarray(data["energy"][0]))
        )
        energy_scale = max(
            abs(float(initial_energy - data["energy_reference"])),
            np.finfo(float).eps,
        )
        energy = np.abs(data["energy"][indices] - initial_energy) / energy_scale
        crest = np.max(
            resample(data["z"][indices], 8 * data["z"].shape[1], axis=1), axis=1
        )
        axes[0, 0].plot(time, crest, color=color, lw=2, label=label)
        axes[0, 1].semilogy(time, np.maximum(np.abs(data["surface_flux_defect"][indices]), 1e-12), color=color, lw=2)
        axes[1, 0].semilogy(time, np.maximum(energy, 1e-12), color=color, lw=2)
        axes[1, 1].plot(
            time,
            (
                float(data.get("cumulative_filter_correction_offset", np.asarray(0.0)))
                + np.cumsum(data["spectral_filter_relative_corrections"])
            )[indices],
            color=color,
            lw=2,
        )
        if failure is not None and failure < len(data["time"]):
            for axis in axes.ravel():
                axis.axvline(float(data["time"][failure]), color=color, lw=1, ls=":", alpha=0.65)

    axes[0, 0].set_title("Crest elevation (admissible prefix)")
    axes[0, 0].set_ylabel(r"$\max z/h_0$")
    axes[0, 0].legend(frameon=False, loc="best")
    axes[0, 1].set_title("Raw DNO flux defect")
    axes[0, 1].axhline(2.0e-2, color="#374151", ls="--", lw=1, label="gate")
    axes[0, 1].set_ylabel(r"$|\int_{\Gamma_f}q\,ds|$")
    axes[1, 0].set_title("Relative wave-energy drift")
    axes[1, 0].axhline(5.0e-3, color="#374151", ls="--", lw=1)
    axes[1, 0].set_ylabel(r"$|E-E_0|/|E_0-E_{\rm ref}|$")
    axes[1, 1].set_title("Cumulative spectral correction")
    axes[1, 1].axhline(5.0e-2, color="#374151", ls="--", lw=1)
    axes[1, 1].set_ylabel(r"$\sum_k\|\delta y_k\|/\|y_k\|$")
    for axis in axes.ravel():
        axis.set_xlabel(r"$t\sqrt{g/h_0}$")
        axis.grid(True, alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("1:30 shoaling benchmark: conservation-gated resolution audit", fontsize=15, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
