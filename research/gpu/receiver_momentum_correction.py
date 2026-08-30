"""Audit posterior momentum corrections on the actual Basilisk receiver grid."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class CorrectionResult:
    name: str
    corrected_mass_fraction: float
    relative_velocity_correction: float
    norm_ratio_to_global_minimum: float
    relative_momentum_residual: float
    relative_kinetic_energy_change: float
    maximum_speed_correction: float


def apply_profile_correction(
    velocity: np.ndarray,
    weights: np.ndarray,
    target_momentum: np.ndarray,
    profile: np.ndarray,
    name: str,
) -> tuple[CorrectionResult, np.ndarray]:
    """Apply delta-u=lambda*profile while satisfying both momentum constraints."""
    profile = np.asarray(profile, dtype=float)
    if np.any(profile < 0.0):
        raise ValueError("the localization profile must be non-negative")
    weighted_profile = float(np.sum(weights * profile))
    if weighted_profile <= 0.0:
        raise ValueError("the localization profile has zero receiver mass")
    momentum_before = np.sum(weights[:, None] * velocity, axis=0)
    deficit = target_momentum - momentum_before
    correction = profile[:, None] * deficit[None, :] / weighted_profile
    corrected = velocity + correction
    momentum_after = np.sum(weights[:, None] * corrected, axis=0)
    target_scale = max(float(np.linalg.norm(target_momentum)), np.finfo(float).eps)
    kinetic_before = 0.5 * float(np.sum(weights * np.sum(velocity**2, axis=1)))
    kinetic_after = 0.5 * float(np.sum(weights * np.sum(corrected**2, axis=1)))
    velocity_norm = max(np.sqrt(2.0 * kinetic_before), np.finfo(float).eps)
    correction_norm = float(
        np.sqrt(np.sum(weights * np.sum(correction**2, axis=1)))
    )
    mass = float(np.sum(weights))
    global_norm = float(np.linalg.norm(deficit) / np.sqrt(mass))
    result = CorrectionResult(
        name=name,
        corrected_mass_fraction=weighted_profile / mass,
        relative_velocity_correction=correction_norm / velocity_norm,
        norm_ratio_to_global_minimum=correction_norm / global_norm,
        relative_momentum_residual=float(np.linalg.norm(momentum_after - target_momentum))
        / target_scale,
        relative_kinetic_energy_change=abs(kinetic_after - kinetic_before)
        / max(kinetic_before, np.finfo(float).eps),
        maximum_speed_correction=float(np.max(np.linalg.norm(correction, axis=1))),
    )
    return result, corrected


def receiver_crest(data: np.ndarray) -> tuple[float, float]:
    mixed = (data["f"] > 1.0e-6) & (data["f"] < 1.0 - 1.0e-6)
    candidates = data[mixed] if np.any(mixed) else data
    crest_y = float(np.max(candidates["y"]))
    top = candidates[np.abs(candidates["y"] - crest_y) <= 0.51 * candidates["delta"]]
    crest_x = float(np.average(top["x"], weights=top["q"] * top["dv"]))
    return crest_x, crest_y


def periodic_distance(values: np.ndarray, center: float, period: float) -> np.ndarray:
    raw = np.abs(values - center)
    return np.minimum(raw, period - raw)


def run_audit(
    csv_path: Path,
    target_momentum: np.ndarray,
    domain_length: float,
) -> dict[str, object]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    weights = np.asarray(data["q"] * data["dv"], dtype=float)
    velocity = np.column_stack((data["u_x"], data["u_y"]))
    mass = float(np.sum(weights))
    momentum_before = np.sum(weights[:, None] * velocity, axis=0)
    deficit = target_momentum - momentum_before
    crest_x, crest_y = receiver_crest(data)
    distance_x = periodic_distance(np.asarray(data["x"]), crest_x, domain_length)
    distance_y = np.asarray(data["y"]) - crest_y
    distance = np.hypot(distance_x, distance_y)
    results: list[CorrectionResult] = []
    global_result, _ = apply_profile_correction(
        velocity, weights, target_momentum, np.ones_like(weights), "global L2 minimum"
    )
    results.append(global_result)
    for width in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        profile = np.exp(-0.5 * (distance / width) ** 2)
        result, _ = apply_profile_correction(
            velocity,
            weights,
            target_momentum,
            profile,
            f"Gaussian crest sigma={width:g}",
        )
        results.append(result)
    ordering = np.argsort(distance)
    cumulative_mass = np.cumsum(weights[ordering]) / mass
    for requested_fraction in (0.05, 0.10, 0.25, 0.50):
        count = int(np.searchsorted(cumulative_mass, requested_fraction)) + 1
        profile = np.zeros_like(weights)
        profile[ordering[:count]] = 1.0
        result, _ = apply_profile_correction(
            velocity,
            weights,
            target_momentum,
            profile,
            f"hard crest support {requested_fraction:.0%}",
        )
        results.append(result)
    lower_bound_relative = global_result.relative_velocity_correction
    payload: dict[str, object] = {
        "schema": "receiver-posterior-momentum-correction-audit-v1",
        "source": str(csv_path),
        "receiver_cell_count": int(len(data)),
        "receiver_mass": mass,
        "momentum_before": momentum_before.tolist(),
        "target_momentum": target_momentum.tolist(),
        "momentum_deficit": deficit.tolist(),
        "crest_location": [crest_x, crest_y],
        "global_l2_lower_bound_relative_velocity_correction": lower_bound_relative,
        "gate_relative_velocity_correction": 0.005,
        "gate_possible_for_any_exact_posterior_correction": bool(
            lower_bound_relative <= 0.005
        ),
        "result": [asdict(item) for item in results],
        "interpretation": (
            "The uniform shift is the exact weighted-L2 minimizer under the total "
            "momentum equality. Locality, divergence and wall constraints restrict "
            "the feasible set and therefore cannot reduce this lower bound."
        ),
    }
    return payload


def create_figure(payload: dict[str, object], path: Path) -> None:
    result = payload["result"]
    assert isinstance(result, list)
    names = [str(item["name"]) for item in result]
    corrections = np.asarray([item["relative_velocity_correction"] for item in result])
    ratios = np.asarray([item["norm_ratio_to_global_minimum"] for item in result])
    fractions = np.asarray([item["corrected_mass_fraction"] for item in result])
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.3), constrained_layout=True)
    colors = ["#23395B"] + ["#D97706"] * 6 + ["#9B2226"] * 4
    axes[0].barh(np.arange(len(names)), 100.0 * corrections, color=colors)
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1.0, label="0.5% gate")
    axes[0].set_yticks(np.arange(len(names)), names, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("weighted velocity correction [%]")
    axes[0].grid(True, axis="x", alpha=0.25)
    axes[0].legend()
    axes[1].scatter(fractions, ratios, c=colors, s=42)
    support = np.linspace(0.04, 1.0, 300)
    axes[1].plot(support, 1.0 / np.sqrt(support), "k--", linewidth=1.0, label=r"hard support $1/\sqrt{m_S/m}$")
    axes[1].axhline(1.0, color="#23395B", linewidth=1.0)
    axes[1].set_xlabel("profile-weighted receiver mass fraction")
    axes[1].set_ylabel("correction norm / global minimum")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("Receiver-grid momentum correction: localization cannot beat the global L2 bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--target-x", type=float, required=True)
    parser.add_argument("--target-y", type=float, required=True)
    parser.add_argument("--domain-length", type=float, default=64.0)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("research/results/q_transport/receiver_correction_audit"),
    )
    args = parser.parse_args()
    payload = run_audit(
        args.csv,
        np.asarray([args.target_x, args.target_y], dtype=float),
        args.domain_length,
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    create_figure(payload, args.output_prefix.with_suffix(".png"))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
