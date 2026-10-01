"""Draw the matched flat-strip construction stated in the manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.3, 4.15), facecolor="white")
    ax = fig.add_axes((0.035, 0.04, 0.93, 0.91))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ink = "#25343d"
    blue = "#176a99"
    orange = "#b55331"
    x0, x1 = 0.10, 0.48
    surface, edge, lower = 0.75, 0.54, 0.39

    ax.text(0.03, 0.97, "One trace, shared across two edge moments", color=ink,
            fontsize=12, fontweight="bold", va="top")
    ax.add_patch(Rectangle((x0, edge), x1 - x0, surface - edge,
                           facecolor="#e9f2f7", edgecolor="none"))
    ax.add_patch(Rectangle((x0, lower), x1 - x0, edge - lower,
                           facecolor="#f4f5f5", edgecolor="none"))
    for x in (x0, x1):
        ax.plot((x, x), (lower, surface), color="#9aa7ad", linewidth=0.85)
    ax.plot((x0, x1), (surface, surface), color=blue, linewidth=2.5)
    ax.plot((x0, x1), (edge, edge), color=orange, linewidth=2.5)
    ax.plot((x0, x1), (lower, lower), color="#aeb8bd", linewidth=0.8)

    ax.add_patch(FancyArrowPatch((x0, 0.845), (x1, 0.845), arrowstyle="<->",
                                 mutation_scale=10, color=ink, linewidth=0.9))
    ax.text((x0 + x1) / 2, 0.855, r"$h$", ha="center", va="bottom", fontsize=11)
    ax.add_patch(FancyArrowPatch((0.071, edge), (0.071, surface), arrowstyle="<->",
                                 mutation_scale=10, color=ink, linewidth=0.9))
    ax.text(0.059, (edge + surface) / 2, r"$d$", ha="right", va="center", fontsize=11)
    ax.text((x0 + x1) / 2, 0.704, "thin liquid strip", ha="center", va="center",
            color="#3b657d", fontsize=9)
    ax.text((x0 + x1) / 2, 0.47, "neighbouring cell", ha="center", va="center",
            color="#66777f", fontsize=8.5)
    ax.text(x0, 0.765, r"physical edge: $\langle H\rangle$", color=blue,
            ha="left", va="bottom", fontsize=9.5)
    ax.text(x0, 0.535, r"shared Cartesian edge: $J(d)$", color=orange,
            ha="left", va="top", fontsize=9.5)

    ax.text(0.56, 0.84, "Source-derived boundary data", fontsize=9,
            color="#607078", va="top")
    ax.text(0.56, 0.745,
            r"$J(d)=\langle H\rangle-d\langle N\rangle+\frac{d^2}{2}\langle S\rangle$",
            fontsize=11.4, color=ink, va="center")
    ax.text(0.56, 0.615,
            r"$\overline{u}_J=\frac{\langle H\rangle-J(d)}{d}$",
            fontsize=11.4, color=ink, va="center")
    ax.text(0.56, 0.525,
            r"$=\langle N\rangle-\frac{d}{2}\langle S\rangle$",
            fontsize=11.4, color=ink, va="center")
    ax.text(0.56, 0.43, "The same trace average cancels.", fontsize=9,
            color="#52636b", va="center")

    ax.plot((0.03, 0.97), (0.345, 0.345), color="#d7dfe3", linewidth=0.9)
    ax.text(0.04, 0.295, "Optional, separate receiver assignment", fontsize=10,
            color=ink, fontweight="bold", va="center")
    ax.text(0.04, 0.205,
            r"$\boldsymbol{v}^{\mathrm{mix}}_K="
            r"\frac{\rho_\ell q_K\overline{\boldsymbol{u}}_{\ell,K}"
            r"+\rho_g(c_{s,K}-q_K)\overline{\boldsymbol{u}}_{g,K}}"
            r"{\rho_\ell q_K+\rho_g(c_{s,K}-q_K)}$",
            fontsize=12, color="#254e65", va="center")
    ax.text(0.04, 0.085,
            "Applied after liquid phase-volume moments are formed; it is not part of their closure identity.",
            fontsize=8.7, color="#56676f", va="center")

    stem = args.output_dir / "method_principle"
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", metadata={"Title": "Matched flat-strip boundary trace"})
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    stem.with_suffix(".provenance.json").write_text(json.dumps({
        "schema": "matched-flat-scheme-v1",
        "scope": "conceptual flat-boundary schematic, not a computed field or curved-interface construction",
        "notation_source": ["eq:matched-jet", "eq:matched-strip-mean", "eq:mixture-projection"],
        "code_sha256": sha(Path(__file__)),
        "pdf_sha256": sha(pdf),
        "png_sha256": sha(png),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
