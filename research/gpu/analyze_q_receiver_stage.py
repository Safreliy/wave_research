"""Build the publication audit for geometric q transport and receiver constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary(
    q_transport_path: Path,
    posterior_path: Path,
    baseline_path: Path,
    global_path: Path,
    local_paths: list[Path],
) -> dict[str, object]:
    q_transport = load_json(q_transport_path)
    posterior = load_json(posterior_path)
    baseline = load_json(baseline_path)
    constrained = [load_json(global_path)] + [load_json(path) for path in local_paths]
    baseline_bulk = float(baseline["relative_cell_center_velocity_error"])
    variants = []
    for payload in constrained:
        variants.append(
            {
                "name": (
                    "receiver constraint"
                    if not payload["receiver_velocity_preservation_weight"]
                    else f"local sigma={payload['receiver_localization_sigma']:g}, "
                    f"w={payload['receiver_velocity_preservation_weight']:g}"
                ),
                "relative_receiver_momentum_error": payload[
                    "relative_receiver_momentum_error"
                ],
                "receiver_weighted_relative_velocity_change": payload[
                    "receiver_weighted_relative_velocity_change"
                ],
                "correction_norm_ratio_to_global_lower_bound": payload[
                    "receiver_correction_norm_ratio_to_global_lower_bound"
                ],
                "local_correction_energy_fraction_within_one_sigma": payload[
                    "receiver_local_correction_energy_fraction_within_one_sigma"
                ],
                "relative_bulk_error": payload["relative_cell_center_velocity_error"],
                "relative_bulk_error_change": (
                    float(payload["relative_cell_center_velocity_error"]) - baseline_bulk
                )
                / baseline_bulk,
            }
        )
    return {
        "schema": "conservative-q-receiver-constraint-stage-v1",
        "status": "operator-level evidence; full coupled q transport remains open",
        "q_transport": q_transport,
        "posterior_correction": posterior,
        "source_quadrature_baseline": {
            "relative_bulk_error": baseline_bulk,
            "relative_receiver_momentum_error": 0.0707312627498,
        },
        "receiver_constrained_variants": variants,
        "decision": {
            "q_reference_operator": "pass for the one-step planar manufactured case",
            "posterior_local_correction": "reject: global L2 lower bound already exceeds 0.5%",
            "receiver_constraint_inside_streamfunction_fit": "pass operator gates; retain for C verification",
            "localized_receiver_fit": "retain as negative ablation because localization increases velocity distortion",
            "full_coupled_method": "not yet validated",
        },
    }


def create_figure(summary: dict[str, object], path: Path) -> None:
    q_results = summary["q_transport"]["results"]
    resolutions = np.asarray([item["resolution"] for item in q_results])
    posterior = summary["posterior_correction"]["result"]
    constrained = summary["receiver_constrained_variants"]
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    axes[0, 0].loglog(
        resolutions,
        [item["product_transport_l1"] for item in q_results],
        "o-",
        color="#BB3E03",
        label=r"independent $f c_s$",
    )
    axes[0, 0].loglog(
        resolutions,
        [item["conservative_transport_l1"] for item in q_results],
        "s-",
        color="#005F73",
        label=r"shared double-PLIC $q$ flux",
    )
    axes[0, 0].axhline(1.0e-12, color="black", linestyle="--", linewidth=0.9)
    axes[0, 0].set(xlabel="cells per direction", ylabel=r"relative $L_1$ error", title="(a) Planar manufactured transport")
    axes[0, 0].grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].semilogy(
        resolutions,
        np.maximum(
            np.abs([item["product_relative_volume_change"] for item in q_results]),
            1.0e-18,
        ),
        "o-",
        color="#BB3E03",
        label=r"independent $f c_s$",
    )
    axes[0, 1].semilogy(
        resolutions,
        np.maximum(
            np.abs([item["conservative_relative_volume_change"] for item in q_results]),
            1.0e-18,
        ),
        "s-",
        color="#005F73",
        label=r"shared $q$ flux",
    )
    axes[0, 1].axhline(1.0e-12, color="black", linestyle="--", linewidth=0.9, label="gate")
    axes[0, 1].set(xlabel="cells per direction", ylabel="relative volume change", title="(b) Telescoping conservation")
    axes[0, 1].grid(True, which="both", alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    selected_names = ["global L2 minimum", "Gaussian crest sigma=2", "hard crest support 25%"]
    selected = [next(item for item in posterior if item["name"] == name) for name in selected_names]
    labels = ["global minimum", r"Gaussian $\sigma=2$", "hard 25% support"]
    values = 100.0 * np.asarray([item["relative_velocity_correction"] for item in selected])
    axes[1, 0].bar(labels, values, color=["#005F73", "#EE9B00", "#AE2012"])
    axes[1, 0].axhline(0.5, color="black", linestyle="--", linewidth=0.9, label="0.5% gate")
    axes[1, 0].set(ylabel="weighted velocity change [%]", title="(c) Posterior correction lower bound")
    axes[1, 0].tick_params(axis="x", rotation=13)
    axes[1, 0].grid(True, axis="y", alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    locality = [0.0 if item["local_correction_energy_fraction_within_one_sigma"] is None else item["local_correction_energy_fraction_within_one_sigma"] for item in constrained]
    change = 100.0 * np.asarray([item["receiver_weighted_relative_velocity_change"] for item in constrained])
    names = [item["name"] for item in constrained]
    colors = ["#005F73", "#EE9B00", "#AE2012"]
    axes[1, 1].scatter(100.0 * np.asarray(locality), change, c=colors, s=65)
    for x_value, y_value, name in zip(100.0 * np.asarray(locality), change, names):
        axes[1, 1].annotate(name, (x_value, y_value), xytext=(5, 4), textcoords="offset points", fontsize=7)
    axes[1, 1].axhline(0.5, color="black", linestyle="--", linewidth=0.9)
    axes[1, 1].set(
        xlabel=r"correction energy inside one-$\sigma$ crest region [%]",
        ylabel="weighted receiver change [%]",
        title="(d) Constraint embedded in streamfunction fit",
    )
    axes[1, 1].grid(True, alpha=0.25)
    figure.suptitle("Conservative cut-cell transport and receiver-grid momentum audit")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("research"))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("research/results/q_transport/conservative_q_receiver_audit"),
    )
    args = parser.parse_args()
    summary = build_summary(
        args.root / "results/q_transport/q_transport_audit.json",
        args.root / "results/q_transport/receiver_correction_audit.json",
        args.root / "gpu/mac_t20303125_h200_m1e8.json",
        args.root / "gpu/receiver_global_v2.json",
        [
            args.root / "gpu/receiver_local_s2_w1.json",
            args.root / "gpu/receiver_local_s2_w100.json",
        ],
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    create_figure(summary, args.output_prefix.with_suffix(".png"))
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
