"""Plot every retained time of the common-build three-grid comparison."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if data["schema"] != "auditor-single-runtime-dynamic-raw-audit-v1":
        raise ValueError("Unexpected audit schema")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), layout="constrained")
    for grid, color, marker in zip(sorted(data["grids"].values(), key=lambda row: row["nx"]),
                                   ("#1c6294", "#ac502d", "#367b60"), ("o", "s", "^")):
        period = grid["series"][-1]["time"] / 0.1
        for method, style, name in (("matched", "-", "M"), ("bilinear", "--", "B")):
            for ax, metric in zip(axes, ("global_liquid_velocity_relative_l2",
                                         "column_depth_wave_relative_l2")):
                rows = [row for row in grid["series"] if row[method][metric] is not None]
                ax.semilogy([row["time"]/period for row in rows],
                            [row[method][metric] for row in rows], style,
                            color=color, marker=marker, markersize=4, linewidth=1.4,
                            markerfacecolor=color if method == "matched" else "white",
                            label=f'{grid["nx"]}: {name}')
    for ax, title in zip(axes, ("Liquid velocity", "Surface profile")):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"Time $t/T$")
        ax.set_ylabel("Relative error")
        ax.set_xticks([0, .025, .05, .075, .1])
        ax.set_xlim(-.001, .103)
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=.18, linewidth=.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=8, loc="outside upper center",
               title="Horizontal cells: Matched (M), Bilinear (B)", title_fontsize=8,
               frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = args.output.with_suffix(".png"), args.output.with_suffix(".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf, metadata={"Title": "Common-build three-grid transfer-error propagation"})
    plt.close(fig)
    args.output.with_suffix(".provenance.json").write_text(json.dumps({
        "input_name": args.input.name, "input_sha256": sha(args.input),
        "generator_sha256": sha(Path(__file__)), "pdf_sha256": sha(pdf),
        "png_sha256": sha(png),
        "scope": "same-grid exact-input numerical reference, common corrected receiver runtime",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
