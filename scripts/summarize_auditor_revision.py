"""Build publication tables from the frozen follow-up experiment audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_table(output: Path, name: str, lines: list[str], source: Path | list[Path]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    table = output / (name + ".tex")
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sources = source if isinstance(source, list) else [source]
    table.with_suffix(".provenance.json").write_text(json.dumps({
        "sources": {path.name: sha(path) for path in sources},
        "generator_sha256": sha(Path(__file__)),
        "table_sha256": sha(table),
    }, indent=2) + "\n", encoding="utf-8")


def initial_refinement(root: Path, output: Path) -> None:
    source = root / "joint_refinement_initial_audit.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    lines = [r"\begin{table}[tbp]\centering\small",
             r"\caption{Initial liquid-volume-weighted velocity error under joint grid and trace refinement (W16). The refined trace spacing is $h/8$ on each grid. Orders compare successive rows; the source boundary state has 512 nodes throughout.}\label{tab:joint-refinement}",
             r"\begin{tabular}{@{}rrrrrrr@{}}\toprule",
             r"$n_x$ & Trace points & $E_M$ & Order & $E_B$ & Order & $E_B/E_M$\\\midrule"]
    for index, row in enumerate(data["rows"]):
        errors = [row["methods"][method]["all_wet"]["liquid_volume_weighted_relative_l2"]
                  for method in ("matched", "bilinear")]
        orders = ["---" if index == 0 else f'{data["observed_two_level_orders"][method][index-1]:.3f}'
                  for method in ("matched", "bilinear")]
        if row["trace_spacing_over_h"] != 0.125:
            raise ValueError("The table requires a fixed h/8 trace spacing")
        lines.append(f'{row["nx"]} & {row["trace_points"]} & {errors[0]:.4e} & {orders[0]} & '
                     f'{errors[1]:.4e} & {orders[1]} & {errors[1]/errors[0]:.2f}' + r"\\")
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    save_table(output, "joint_refinement", lines, source)


def transfer_cost(root: Path, output: Path) -> None:
    sources = [root / f"cost_L{level}.json" for level in (9, 10)]
    data = [json.loads(path.read_text(encoding="utf-8")) for path in sources]
    lines = [r"\begin{table}[tbp]\centering\small",
             r"\caption{CPU transfer cost in seconds: median [minimum, maximum] over three repetitions after one warm-up. Stages exclude file output, checkpoints and CFD evolution. Stage medians need not sum to the median total.}\label{tab:transfer-cost}",
             r"\begin{tabular}{@{}lrr@{}}\toprule",
             r"Stage & $512\times32$ & $1024\times64$\\\midrule"]
    for key, label in (("trace", "Boundary trace"),
                       ("assembly", "Geometry and harmonic-system assembly"),
                       ("solve", "Liquid harmonic solve"),
                       ("gas", "Gas continuation"),
                       ("moments", "Liquid-cell moments"),
                       ("other", "Remaining in-memory operations"),
                       ("total", "Transfer total, excluding trace")):
        values = []
        for run in data:
            stage = run["stages"][key]
            values.append(f'{stage["median_seconds"]:.4f} [{stage["minimum_seconds"]:.4f}, {stage["maximum_seconds"]:.4f}]')
        lines.append(label + " & " + " & ".join(values) + r"\\")
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    save_table(output, "transfer_cost", lines, sources)


def dynamic_comparison(root: Path, output: Path) -> None:
    source = root / "dynamic_three_grid_audit.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    lines = [r"\begin{table}[tbp]\centering\small",
             r"\caption{Propagated transfer error at $t=0.1T$ on the common corrected receiver build. Each reference trajectory starts from exact cell means on the same grid. $E_u$ is the liquid-weighted velocity error and $E_H$ the wave-normalised column-depth discrepancy; $R=E_B/E_M$.}\label{tab:audited-dynamics}",
             r"\begin{tabular}{@{}rrrrrrr@{}}\toprule",
             r"$n_x$ & $E_u^M$ & $E_u^B$ & $R_u$ & $E_H^M$ & $E_H^B$ & $R_H$\\\midrule"]
    for grid in sorted(data["grids"].values(), key=lambda row: row["nx"]):
        end = grid["series"][-1]
        values = [str(grid["nx"])]
        for metric in ("global_liquid_velocity_relative_l2", "column_depth_wave_relative_l2"):
            m, b = (end[method][metric] for method in ("matched", "bilinear"))
            values.extend([f"{m:.4e}", f"{b:.4e}", f"{b/m:.2f}"])
        lines.append(" & ".join(values) + r"\\")
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    save_table(output, "audited_dynamics", lines, source)


def physical_comparison(root: Path, output: Path) -> None:
    source = root / "physical_two_grid_audit.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    lines = [r"\begin{table}[tbp]\centering\small",
             r"\caption{Physical column-depth error against the cell-averaged linear two-fluid wave, with density-weighted receiver assignment held fixed. Errors use the fixed linear-wave RMS amplitude. Every recorded nonzero time is included; the reduction is $100(1-E_M/E_B)$.}\label{tab:physical-refinement}",
             r"\begin{tabular}{@{}rrrrrr@{}}\toprule",
             r"$n_x$ & $t$ & Exact input & Matched (M) & Bilinear (B) & Reduction (\%)\\\midrule"]
    for index, grid in enumerate(sorted(data["grids"].values(), key=lambda row: row["nx"])):
        if index:
            lines.append(r"\midrule")
        for row in grid["series"]:
            if row["time"] <= 0:
                continue
            e, m, b = (row["linear_wave"][method]["normalised_depth_rms"]
                       for method in ("exact", "matched", "bilinear"))
            lines.append(f'{grid["nx"]} & {row["time"]:.5f} & {e:.8f} & {m:.8f} & {b:.8f} & {100*(1-m/b):.2f}' + r"\\")
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    save_table(output, "physical_refinement", lines, source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    initial_refinement(args.root, args.output)
    transfer_cost(args.root, args.output)
    if (args.root / "dynamic_three_grid_audit.json").exists():
        dynamic_comparison(args.root, args.output)
    if (args.root / "physical_two_grid_audit.json").exists():
        physical_comparison(args.root, args.output)


if __name__ == "__main__":
    main()
