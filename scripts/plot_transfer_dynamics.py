"""Plot measured native error propagation; no exact-flow accuracy is implied."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05), sharey=True)
    provenance = []
    for ax, level in zip(axes, (9, 10)):
        for suffix, colour, trace_label in (("", "#c46620", "Original trace"),
                                            ("_trace4", "#1674ac", "4x denser trace")):
            folder = args.root / f"flat_native_L{level}{suffix}"
            candidates = [folder / "analysis_full_series.json", folder / "analysis.json"]
            source = next((p for p in candidates if p.exists()), None)
            if source is None:
                raise FileNotFoundError(f"No completed series in {folder}")
            report = json.loads(source.read_text())
            phase = report["phases"]["full"]
            series = phase["time_series"]
            target = report["target_time"]
            if not series or abs(series[-1]["time"] - target) > 1e-12:
                raise ValueError(f"Incomplete endpoint in {source}")
            period = target / report["fraction_linear_wave_period"]
            times = [row["time"] / period for row in series]
            for method, style, marker in (("fitted", "-", "o"),
                                           ("bilinear_centroid", "--", "s")):
                values = [row[method]["global_liquid_velocity_relative_l2"] for row in series]
                if any(value <= 0 for value in values):
                    raise ValueError(f"Nonpositive logarithmic data in {source}")
                name = "Linear (L)" if method == "fitted" else "Bilinear (B)"
                ax.semilogy(times, values, color=colour, linestyle=style,
                            marker=marker, markersize=3, linewidth=1.1,
                            label=f"{name}, {trace_label.lower()}")
            provenance.append({"path": str(source), "sha256": digest(source)})
        ax.set_title(f"$n_x={2 ** level}$")
        ax.set_xlabel("Time / linear wave period")
        ax.set_xlim(-0.002, 0.103)
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(True, which="major", alpha=0.25)
    axes[0].set_ylabel("Global liquid-weighted velocity error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), metadata={"Title": "Native transfer-error propagation"})
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    args.output.with_suffix(".provenance.json").write_text(json.dumps({
        "inputs": provenance, "code_sha256": digest(Path(__file__)),
        "scope": "same-grid reference initialized with exact phase means; not exact Navier-Stokes solution",
        "pdf_sha256": digest(args.output.with_suffix(".pdf")),
        "png_sha256": digest(args.output.with_suffix(".png")),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
