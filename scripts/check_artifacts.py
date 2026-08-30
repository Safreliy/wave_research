"""Fail fast when the compact publication package is incomplete or inconsistent."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "research" / "results" / "publication_package"
Q_PACKAGE = ROOT / "research" / "results" / "q_transport"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    required = [
        ROOT / "paper.pdf",
        ROOT / "manuscript" / "main.tex",
        ROOT / "manuscript" / "references.bib",
        PACKAGE / "near_self_quadrature_ablation.json",
        PACKAGE / "mac_face_handoff_audit.json",
        PACKAGE / "transfer_ablation_summary.json",
        PACKAGE / "handoff_time_screen_summary.json",
        Q_PACKAGE / "q_transport_audit.json",
        PACKAGE / "impact_claim_q_l8_l9_l10_audit.json",
        PACKAGE / "impact_claim_q_l8_l9_l10_morphology.json",
        PACKAGE / "impact_claim_q_l8_l9_l10_morphology.npz",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing publication artifacts: {missing}")

    reader = PdfReader(ROOT / "paper.pdf")
    page_count = len(reader.pages)
    if not 10 <= page_count <= 20:
        raise SystemExit(f"expected a 10--20-page paper, found {page_count} pages")
    title = str((reader.metadata or {}).get("/Title", ""))
    if "divergence-conforming state transfer" not in title.lower():
        raise SystemExit(f"unexpected PDF title metadata: {title!r}")

    transfer = load_json(PACKAGE / "transfer_ablation_summary.json")
    if len(transfer["rows"]) != 6:
        raise SystemExit("transfer ablation does not contain all six representations")
    if max(
        level["relative_receiver_momentum_error"]
        for level in transfer["production_receiver_levels"].values()
    ) >= 1e-8:
        raise SystemExit("production receiver momentum gate failed")

    timing = load_json(PACKAGE / "handoff_time_screen_summary.json")
    if timing["all_gates_passed"] is not False:
        raise SystemExit("handoff-time screen is unexpectedly promoted")
    screen = timing["screen_configuration"]
    if (
        screen["conservative_q_embed"] is not False
        or screen["production_comparable"] is not False
    ):
        raise SystemExit("legacy timing screen is incorrectly identified as production evidence")
    production_volume_error = timing["production_reference"][
        "maximum_relative_volume_error_through_time_6"
    ]
    if production_volume_error >= 1e-7:
        raise SystemExit("production level-9 volume record failed its consistency check")

    q_transport = load_json(Q_PACKAGE / "q_transport_audit.json")
    for row in q_transport["results"]:
        if row["conservative_transport_l1"] >= 1e-12:
            raise SystemExit("shared-q transport L1 gate failed")
        if abs(row["conservative_relative_volume_change"]) >= 1e-12:
            raise SystemExit("shared-q volume conservation gate failed")

    impact = load_json(PACKAGE / "impact_claim_q_l8_l9_l10_audit.json")
    levels = [int(case["level"]) for case in impact["cases"]]
    if levels != [8, 9, 10] or impact["impact_claim_accepted"] is not False:
        raise SystemExit("three-grid topology decision is missing or unexpectedly positive")

    morphology = load_json(PACKAGE / "impact_claim_q_l8_l9_l10_morphology.json")
    if morphology["accepted_three_grid_upper_surface"] is not False:
        raise SystemExit("three-grid morphology decision is unexpectedly positive")
    with np.load(PACKAGE / "impact_claim_q_l8_l9_l10_morphology.npz") as arrays:
        if arrays["time"].shape != (301,) or arrays["profiles"].shape[0] != 3:
            raise SystemExit("unexpected morphology array dimensions")

    figures = [
        PACKAGE / "near_self_quadrature_ablation.png",
        PACKAGE / "mac_face_handoff_audit.png",
        PACKAGE / "impact_claim_q_l8_l9_l10_audit.png",
        PACKAGE / "impact_claim_q_l8_l9_l10_grid_comparison.png",
        PACKAGE / "impact_claim_q_l8_l9_l10_morphology.png",
        PACKAGE / "impact_claim_q_l10_vorticity_points_keyframes" / "late_pocket.png",
    ]
    for figure in figures:
        with Image.open(figure) as image:
            image.verify()

    diagnostics_root = ROOT / "research" / "two_phase_basilisk" / "remote_impact_claim"
    cases = {
        8: "impact_claim_q_l8_t60",
        9: "impact_claim_q_l9_t60",
        10: "impact_claim_q_l10_t60_final",
    }
    for level, name in cases.items():
        log = (diagnostics_root / name / "diagnostics.dat").read_text(
            encoding="utf-8", errors="replace"
        )
        if re.search(r"(?m)^RUN 6(?:\.0+)?\s", log) is None:
            raise SystemExit(f"level {level} diagnostic log does not end at t=6")

    oversized = [
        path for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and path.stat().st_size >= 100_000_000
    ]
    if oversized:
        raise SystemExit(f"files exceed GitHub's 100 MB limit: {oversized}")

    print(
        f"publication artifacts: OK ({page_count} pages, L8--L10, "
        "negative convergence claims)"
    )


if __name__ == "__main__":
    main()
