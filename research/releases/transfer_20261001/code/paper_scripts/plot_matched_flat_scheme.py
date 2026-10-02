"""Illustrate trace-offset cancellation for a flat liquid strip."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw_transition(ax, *, lower: float, upper: float, shift_lower: float,
                    shift_upper: float, color: str, heading: str,
                    result: str) -> None:
    """Draw a value-gap segment swept from reference to represented values."""
    y_ref, y_used = 0.285, 0.555
    p0 = (lower, y_ref)
    p1 = (upper, y_ref)
    q0 = (lower + shift_lower, y_used)
    q1 = (upper + shift_upper, y_used)
    ax.add_patch(Polygon((p0, p1, q1, q0), closed=True,
                         facecolor=color, alpha=0.11, edgecolor="none"))
    ax.plot((lower, upper), (y_ref, y_ref), color="#84939a", lw=2.0,
            solid_capstyle="round", zorder=2)
    ax.plot((q0[0], q1[0]), (y_used, y_used), color=color, lw=3.1,
            solid_capstyle="round", zorder=3)
    for start, end in ((p0, q0), (p1, q1)):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                     mutation_scale=8.5, shrinkA=7,
                                     shrinkB=8, color=color, alpha=0.84,
                                     lw=1.3, zorder=2))
    for x, y, filled in ((*p0, False), (*p1, False),
                         (*q0, True), (*q1, True)):
        ax.plot(x, y, marker="o", markersize=7.0,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color if filled else "#84939a",
                markeredgewidth=1.35, zorder=4)
    center = (2 * lower + 2 * upper + shift_lower + shift_upper) / 4
    ax.text(center, 0.625, heading, ha="center", va="center", color=color,
            fontsize=10.4, fontweight="bold")
    ax.text(center, 0.125, result, ha="center", va="center", color=color,
            fontsize=10.0, fontweight="bold")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.2, 3.75), facecolor="white")
    ax = fig.add_axes((0.035, 0.025, 0.93, 0.94))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.09, 1)
    ax.axis("off")
    ink = "#24343f"
    teal = "#267b73"
    rust = "#ae5747"

    ax.text(0.015, 0.978, "One trace. Two edge values.", color=ink,
            fontsize=15.8, family="DejaVu Serif", fontweight="bold",
            va="top")
    ax.text(0.017, 0.884, "Only their difference enters the thin-strip mean",
            color="#61717a", fontsize=9.5, va="top")

    # Spatial inset: d is a geometric thickness. The large constructions
    # below live on separate streamfunction-value axes, not in physical space.
    x0, x1 = 0.688, 0.784
    y0, y1 = 0.807, 0.914
    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                           facecolor="#e9f1f5", edgecolor="none"))
    ax.plot((x0, x1), (y1, y1), color="#556b77", lw=2.3)
    ax.plot((x0, x1), (y0, y0), color="#9cabb3", lw=2.3)
    ax.add_patch(FancyArrowPatch((0.674, y0), (0.674, y1),
                                 arrowstyle="<->", mutation_scale=8,
                                 color=ink, lw=1.05))
    ax.text(0.66, (y0+y1)/2, r"$d$", ha="right", va="center",
            color=ink, fontsize=10)
    ax.text(0.818, 0.873, r"$\bar u=\Delta\psi/d$", color=ink,
            fontsize=11.0, va="center")

    ax.text(0.5, -0.038, r"Streamfunction value $\psi$",
            color="#61717a", fontsize=8.3, ha="center", va="top")
    ax.add_patch(FancyArrowPatch((0.09, 0.025), (0.92, 0.025),
                                 arrowstyle="-|>", mutation_scale=7,
                                 color="#84939a", lw=0.75))

    # Equal endpoint shifts sweep a parallelogram. Independent shifts sweep
    # a trapezoid. The y separation means before/after, not physical height.
    draw_transition(ax, lower=0.090, upper=0.270,
                    shift_lower=0.105, shift_upper=0.105,
                    color=teal, heading="shared trace",
                    result=r"same gap $\Delta\psi$")
    draw_transition(ax, lower=0.585, upper=0.765,
                    shift_lower=0.077, shift_upper=0.150,
                    color=rust, heading="independent edges",
                    result=r"changed gap $\Delta\psi+\delta$")
    ax.text(0.090, 0.212, r"$\psi_{\mathrm{lower}}$", ha="center",
            color="#73838c", fontsize=8.7)
    ax.text(0.270, 0.212, r"$\psi_{\mathrm{top}}$", ha="center",
            color="#73838c", fontsize=8.7)

    ax.plot((0.49, 0.49), (0.18, 0.64), color="#e1e8ea", lw=0.9)
    # Explain marker shape with actual glyphs instead of verbal descriptions
    # of "open" and "filled". Neutral ink leaves case colours to the panels.
    # One enclosed key above the comparison, with both symbols stacked.
    ax.add_patch(Rectangle((0.015, 0.683), 0.335, 0.135,
                           facecolor="#fafbfc", edgecolor="#e1e8ea", lw=0.7))
    legend_y = 0.777
    ax.plot(0.035, legend_y, marker="o", markersize=5,
            markerfacecolor="white", markeredgecolor="#84939a",
            markeredgewidth=1.25, clip_on=False)
    ax.text(0.052, legend_y, "Reference edge values", color="#61717a",
            fontsize=7.7, va="center")
    legend_y = 0.722
    ax.plot(0.035, legend_y, marker="o", markersize=5,
            markerfacecolor=ink, markeredgecolor=ink, clip_on=False)
    ax.text(0.052, legend_y, "Approximated edge values", color="#61717a",
            fontsize=7.7, va="center")

    stem = args.output_dir / "method_principle"
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight",
                metadata={"Title": "Shared trace preserves the streamfunction value gap"})
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    stem.with_suffix(".provenance.json").write_text(json.dumps({
        "schema": "matched-flat-scheme-v7",
        "scope": "conceptual flat-strip comparison isolating a common streamfunction trace offset; not computed data or a curved-interface construction",
        "delta_definition": "difference of independent streamfunction edge approximation errors, not geometric displacement",
        "visual_encoding": "paired markers are edge values on a horizontal psi scale; vertical separation is a before/after stage; parallelogram denotes equal shifts and trapezoid unequal shifts",
        "notation_source": ["eq:matched-jet", "eq:matched-strip-mean"],
        "suggested_caption": "Common-trace cancellation in a flat liquid strip. The inset defines geometric thickness d and horizontal mean velocity from the streamfunction gap. In the value diagram, open markers show reference edge means and filled markers their representations; the vertical separation denotes stages, not spatial height. Equal trace shifts preserve the gap (parallelogram), whereas independent edge errors alter it by delta (trapezoid), contributing delta/d to the mean velocity. Derivative and interior errors remain outside this isolated trace-error term.",
        "code_sha256": sha(Path(__file__)),
        "pdf_sha256": sha(pdf),
        "png_sha256": sha(png),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
