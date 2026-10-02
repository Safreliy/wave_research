"""Freeze the declared dynamic comparison matrix and its manuscript table."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CASES = (
    ("flat_native_L9", 512, 8, ("full", "half")),
    ("flat_native_L10", 1024, 8, ("full", "half")),
    ("flat_native_L9_trace4", 512, 32, ("full",)),
    ("flat_native_L10_trace4", 1024, 32, ("full", "half")),
    ("flat_native_L11_resolved", 2048, 64, ("full",)),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    rows, inputs, temporal, cross_time = [], {}, {}, {}
    for name, nx, factor, phases in CASES:
        path = args.root / name / "analysis_full_series.json"
        result = json.loads(path.read_text())
        inputs[name + "/analysis_full_series.json"] = sha(path)
        if "time_step_sensitivity" in result:
            temporal[name] = result["time_step_sensitivity"]
        if "cross_time_reference" in result:
            cross_time[name] = result["cross_time_reference"]
        for phase in phases:
            branch = result["phases"][phase]
            metrics = branch["comparisons"]["final"]
            fit, control = metrics["fitted"], metrics["bilinear_centroid"]
            expected_dt = 0.0025 if phase == "full" else 0.00125
            expected_steps = 210 if phase == "full" else 420
            for metadata in branch["runs"].values():
                if (metadata["status_state"] != "complete"
                        or abs(metadata["time"] - result["target_time"]) > 1e-10
                        or metadata["observed_maximum_dt"] > expected_dt * (1 + 1e-8)
                        or metadata["solver_steps"] < expected_steps):
                    raise ValueError(f"Invalid run coverage: {name}/{phase}")
            vel_key = "global_liquid_velocity_relative_l2"
            wave_key = "column_depth_wave_relative_l2"
            rows.append({"case": name, "nx": nx, "ny": nx // 16,
                         "trace_factor": factor, "phase": phase,
                         "dt_limit": expected_dt,
                         "fitted": fit, "bilinear_centroid": control,
                         "velocity_error_reduction": control[vel_key] / fit[vel_key],
                         "wave_depth_error_reduction": control[wave_key] / fit[wave_key],
                         "run_metadata": branch["runs"]})
    lines = [r"\begin{table}[tbp]\centering\small",
             r"\caption{Global propagated transfer errors at $t=0.1T$. The trace factor",
             r"is the number of trace samples per original source segment. Ratios",
             r"$R_u=E_u^B/E_u^L$ and $R_H$ for wave-normalised",
             r"column depth exceed one when Linear (L) is better than Bilinear (B). Every",
             r"comparison uses a reference with the same grid and time-step limit.}",
             r"\label{tab:dynamic-errors}",
             r"\begin{tabular}{rrrrrrr}\toprule",
             r"$n_x$ & Trace & $10^3\Delta t_{\max}$ & $E_u^L$ & $E_u^B$ & $R_u$ & $R_H$ \\",
             r"\midrule"]
    for row in rows:
        ef = row["fitted"]["global_liquid_velocity_relative_l2"]
        eb = row["bilinear_centroid"]["global_liquid_velocity_relative_l2"]
        lines.append(f'{row["nx"]} & {row["trace_factor"]} & {row["dt_limit"]*1000:g} & '
                     f'{ef:.5f} & {eb:.5f} & {row["velocity_error_reduction"]:.2f} & '
                     f'{row["wave_depth_error_reduction"]:.2f} ' + r"\\")
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table}", ""])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(lines), encoding="utf-8")
    report = {"schema": "transfer-dynamic-summary-v1", "input_hashes": inputs,
              "code_sha256": sha(Path(__file__)), "table_sha256": sha(args.table),
              "reference_limit": "same-grid numerical exact-initial-input trajectory; not physical truth",
              "rows": rows, "time_step_sensitivity": temporal,
              "cross_time_reference": cross_time}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "table": str(args.table)}))


if __name__ == "__main__":
    main()
