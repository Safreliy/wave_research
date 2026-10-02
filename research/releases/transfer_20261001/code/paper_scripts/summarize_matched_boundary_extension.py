"""Regenerate the complete declared flat-strip comparison for the article."""
import argparse
import hashlib
import json
from pathlib import Path


def number(x):
    return f"{x:.3e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.input.read_text())
    lines = [r"\begin{table}[htbp]", r"\centering\small",
             r"\caption{Initial velocity errors for the matched boundary extension, measured in the liquid-volume-weighted relative norm over all liquid cells. M denotes the manufactured mode-4 field; W denotes the mode-16 wave field. The control uses the same computed streamfunction.}\label{tab:matched-extension}",
             r"\begin{tabular}{@{}lrrllll@{}}\toprule",
             r"Family & $n_x$ & $q_{\rm top}$ & Linear & Matched & Control & Gain\\\midrule"]
    ratios = []
    for row in data["rows"]:
        m = row["metrics"]
        key = "all_q_weighted_relative_l2"
        original = m["linear"]["fitted"][key]
        improved = m["parent_hermite_matched"]["fitted"][key]
        control = m["parent_hermite_matched"]["same_psi_centroid"][key]
        ratios.append(control/improved)
        family = "M" if row["family"] == "manufactured" else "W"
        fields = [family, str(row["nx"]), f'{row["surface_liquid_fraction_requested"]:g}',
                  number(original), number(improved), number(control), f"{control/improved:.2f}"]
        lines.append(" & ".join(fields)+r"\\")
    lines += [r"\bottomrule\end{tabular}", r"\end{table}"]
    args.output.write_text("\n".join(lines)+"\n", encoding="utf-8")
    summary = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
               "rows": len(ratios), "gain_min": min(ratios), "gain_max": max(ratios),
               "all_favourable_against_same_psi_control": all(x > 1 for x in ratios),
               "max_matched_momentum_closure": max(r["matched_momentum_closure"] for r in data["rows"])}
    args.output.with_suffix(".provenance.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
