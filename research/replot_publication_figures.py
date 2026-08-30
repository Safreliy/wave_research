"""Redraw publication summaries from the compact evidence stored in Git."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze_impact_claim import _plot as plot_impact_audit
from analyze_three_grid_morphology import plot as plot_morphology


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "results" / "publication_package"


def main() -> None:
    impact_json = PACKAGE / "impact_claim_q_l8_l9_l10_audit.json"
    impact_png = PACKAGE / "impact_claim_q_l8_l9_l10_audit.png"
    impact = json.loads(impact_json.read_text(encoding="utf-8"))
    plot_impact_audit(impact, impact_png)

    morphology_prefix = PACKAGE / "impact_claim_q_l8_l9_l10_morphology"
    morphology = json.loads(
        morphology_prefix.with_suffix(".json").read_text(encoding="utf-8")
    )
    with np.load(morphology_prefix.with_suffix(".npz")) as archive:
        arrays = {name: archive[name] for name in archive.files}
    plot_morphology(morphology, arrays, morphology_prefix.with_suffix(".png"))

    print(f"wrote {impact_png}")
    print(f"wrote {morphology_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
