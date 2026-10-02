# Reproducing the transfer article

Run the commands below from the **public repository checkout root** (the
directory containing `scripts/`, `manuscript/`, and `research/releases/`).
Set `R` to the frozen follow-up supplement. A separately downloaded supplement
can be used by changing only `R`. The 2026-10-01/v0.31.0 evidence is historical
and is not the source of the new three-grid and two-grid results.

## Level 1: verify archived fields and regenerate article displays

This level does **not** run Aphros or reconstruct the transfer inputs. It
checks 93 manifest-listed files, 180 input members, all 15 native-run
archives, and recomputes the three principal numerical audits from saved raw
fields. It needs Python and NumPy and enough temporary space for extraction.

```sh
R=research/releases/transfer_20261002
export PYTHONDONTWRITEBYTECODE=1
python "$R/verify_release.py"
python "$R/verify_release.py" --replay
python scripts/summarize_auditor_revision.py \
  --root "$R" --output /tmp/transfer-article-tables
python scripts/plot_audited_dynamics.py \
  --input "$R/dynamic_three_grid_audit.json" \
  --output /tmp/audited_native_dynamics
```

The table script writes four `.tex` files and adjacent `.provenance.json`
files. The plot script writes PDF, PNG, and provenance files. To compare with
the submitted source, compare those table files with
`manuscript/tables/` and the plot with
`manuscript/figures/audited_native_dynamics.pdf`.
`SHA256SUMS.json` and `code/MANIFEST.json` record the frozen evidence and
Python-source bytes. The replay verifier re-evaluates numerical checks from
fields; a successful replay is not a new receiver simulation.

| Manuscript item | Numerical data | Audit and display command | Display file |
| --- | --- | --- | --- |
| `tab:joint-refinement` | `R/joint_refinement_inputs.tar.gz`, `R/joint_refinement_initial_audit.json` | `verify_release.py --replay`; `summarize_auditor_revision.py` | `joint_refinement.tex` |
| `tab:audited-dynamics` | `R/dynamic_inputs.tar.gz`, nine `R/L{9,10,11}_{exact,matched,bilinear}_outputs.tar.gz`, `R/dynamic_three_grid_audit.json` | Same verifier and table script | `audited_dynamics.tex` |
| `fig:audited-dynamics` | `R/dynamic_three_grid_audit.json` | `plot_audited_dynamics.py` as above | `audited_native_dynamics.pdf` |
| `tab:physical-refinement` | `R/physical_inputs.tar.gz`, six `R/physical_L{9,10}_{exact,matched,bilinear}_outputs.tar.gz`, `R/physical_two_grid_audit.json` | Same verifier and table script | `physical_refinement.tex` |
| `tab:transfer-cost` | `R/cost_L9.json`, `R/cost_L10.json`, `R/cost_hardware.json` | `summarize_auditor_revision.py` | `transfer_cost.tex` |

The initial three-grid series uses trace factors 8/16/32 at fixed trace/grid
ratio. The native propagation series uses 8/8/16. Its reference is the
same-grid *numerical* trajectory initialized from exact liquid-cell means.
The physical series instead compares with a cell-averaged linear two-fluid
wave. The `normalization` object in `physical_two_grid_audit.json` states
the different denominators; neither reference is an exact nonlinear
Navier–Stokes trajectory. The cost JSON contains measured CPU times, so the
table can be regenerated exactly, while a new timing run need not match it.

Three main-text tables and the inline L/H/M ablation use the earlier evidence,
not the new supplement. The historical archive contains the declared
manufactured/wave result JSON; extract it first, then regenerate the three
historical `.tex` files and provenance sidecars:

```sh
H=research/releases/transfer_20261001
mkdir -p /tmp/transfer-article-tables
tar -xzf "$H/archives/method_trace_hermite_20261001_evidence.tar.gz" \
  -C /tmp/transfer-article-tables \
  method_trace_hermite_20261001/matched_declared/result.json
python scripts/summarize_matched_boundary_extension.py \
  --input /tmp/transfer-article-tables/method_trace_hermite_20261001/matched_declared/result.json \
  --output /tmp/transfer-article-tables/matched_boundary_extension.tex
python scripts/summarize_physical_wave.py \
  --input "$H/results/physical/guarded_transfer_height_analysis.json" \
  --output /tmp/transfer-article-tables/physical_wave_comparison.tex
python scripts/summarize_mixture_factorial.py \
  --input "$H/results/physical/guarded_mass_factorial_complete_height_analysis.json" \
  --output /tmp/transfer-article-tables/mixture_factorial.tex
```

These map respectively to `tab:matched-extension`, `tab:physical-wave`, and
`tab:mixture-factorial`. Their historical source hashes are in the matching
`manuscript/tables/*.provenance.json` files. The inline
`tab:lhm-ablation` uses two rows of the same `matched_declared/result.json`:
`(family,nx,q_top)=(manufactured,128,0.001)` and `(native,512,0.0005)`.
Its L, H, and M cells are `metrics[method].fitted.top_relative_l2` for
`method=linear,parent_hermite,parent_hermite_matched`, respectively. The
values can be printed without relying on the typeset table:

```sh
python - /tmp/transfer-article-tables/method_trace_hermite_20261001/matched_declared/result.json <<'PY'
import json, sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))["rows"]
for family, nx, q in (("manufactured", 128, 0.001), ("native", 512, 0.0005)):
    row = next(r for r in rows if r["family"] == family and r["nx"] == nx
               and r["surface_liquid_fraction_requested"] == q)
    print(family, nx, q, *(
        row["metrics"][method]["fitted"]["top_relative_l2"]
        for method in ("linear", "parent_hermite", "parent_hermite_matched")))
PY
```

Other appendix
tables and figures refer to the earlier protocols identified in their captions
and are not regenerated by the 2026-10-02 supplement.

## Level 2: reconstruct transfer inputs

The supplement contains the three serialized input trees and the frozen
transfer source under `R/code/research/gpu_port/`. The archived
`R/code/research/results/transfer_auditor_revision_20261002/dynamic/source/prepare_flat_dynamic_benchmark.py`
is the exact generator snapshot used before the 8/8/16 native campaign; the later main code adds
an explicit `--trace-factor` option for the separate 8/16/32 initial study.
The scientific transfer modules are hash-linked in `code/MANIFEST.json`.
For a fresh input calculation, use a *new* output directory and retain the
archived exact-input branch as the common receiver geometry and gas field:

```sh
W=/tmp/transfer-input-rebuild
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$W"
tar -xzf "$R/dynamic_inputs.tar.gz" -C "$W"
python "$R/code/research/gpu_port/prepare_flat_dynamic_benchmark.py" \
  --nx 512 --trace-factor 8 --physical-trace parent_hermite_matched \
  --common-input-root "$W/dynamic/L9/exact" --root "$W/rebuilt_L9"
```

The output contains `fitted/` (Matched), `bilinear_centroid/` (same computed
streamfunction control), `exact_initial_means/`, `harmonic_source.npz`, and
`preparation_report.json`. Use `(nx, trace factor)=(1024,8),(2048,16)` for
the other native inputs and `(512,8),(1024,16),(2048,32)` for the separate
fixed-ratio initial study. The same command was checked for L9 against the
frozen input: its Matched and exact velocity arrays matched byte for byte;
the bilinear arrays differed by at most `1.22e-14` in individual components
on the tested local environment. Do not use bitwise equality of every
regenerated floating-point field as a scientific acceptance criterion.

The original `prepare_auditor_runtime_campaign.py` and
`prepare_auditor_twofluid_grid.py` also refer to historical 2026-10-01
workspace inputs, including the earlier physical factorial and the L11
Matched input. They are **not standalone generators from this supplement**.
The six physical input folders and their hashes are already preserved in
`physical_inputs.tar.gz` and `physical/campaign.json`. The executed physical
generator snapshot is under
`R/code/research/results/transfer_auditor_revision_20261002/physical/`.
The current generator corrects a metadata name; see
`physical_initial_metadata_erratum.json`. This does not alter the executed
field bytes. Rebuilding the *original* campaigns from first principles
requires those historical input trees and the source environment; replaying
the archived fields does not.

To measure transfer cost again on a different CPU, use the frozen timing code
with fresh output paths. These commands were checked against its CLI; their
new timing values will reflect the new machine and are not replacements for
the archived measurements:

```sh
python "$R/code/research/gpu_port/benchmark_auditor_transfer_cost.py" \
  --nx 512 --trace-factor 8 --repeats 3 --output /tmp/cost_L9_new.json
python "$R/code/research/gpu_port/benchmark_auditor_transfer_cost.py" \
  --nx 1024 --trace-factor 16 --repeats 3 --output /tmp/cost_L10_new.json
```

## Level 3: repeat receiver runs

The exact 15 supervisor argument lists, branch order, input SHA-256 values,
and executable/library SHA-256 values are in `dynamic/campaign.json` and
`physical/campaign.json`. On a **clean Linux receiver environment**, extract
the two input archives under `/opt/gpu-cfd/transfer_auditor_revision_20261002`
and run each recorded command. The following preserves all scientific
arguments but uses the active Python interpreter in place of the original
machine's hard-coded virtual-environment path:

```sh
N=/opt/gpu-cfd/transfer_auditor_revision_20261002
mkdir -p "$N"
tar -xzf "$R/dynamic_inputs.tar.gz" -C "$N"
tar -xzf "$R/physical_inputs.tar.gz" -C "$N"
mkdir -p /opt/gpu-cfd/prefix_zero_flux_guard_20261001
tar -xzf "$R/runtime_provenance/aphros_guard_guarded_runtime.tar.gz" \
  -C /opt/gpu-cfd/prefix_zero_flux_guard_20261001
install -D "$R/runtime_provenance/aphros_sim_base.conf" \
  /opt/gpu-cfd/aphros/deploy/scripts/sim_base.conf
install -D "$R/runtime_provenance/AMGX_PCG_AGGREGATION_JACOBI.json" \
  /opt/gpu-cfd/prefix/lib/configs/PCG_AGGREGATION_JACOBI.json
python - "$R" <<'PY'
import json, subprocess, sys
from pathlib import Path

release = Path(sys.argv[1])
for campaign in ("dynamic", "physical"):
    plan = json.loads((release / campaign / "campaign.json").read_text())
    for branch in plan["run_order"]:
        command = list(plan["commands"][branch])
        command[0] = sys.executable
        subprocess.run(command, check=True)
PY
```

This expects **fresh run roots** at the recorded absolute paths. It also
requires the AMGX library at the recorded path and SHA-256, a compatible GPU
driver/toolchain, and Python dependencies. Each extracted input grid already
contains its recorded `benchmark_supervisor.py` and base configuration; the
recorded commands point to these copies. `runtime_provenance/` provides the guarded Aphros source,
the narrow zero-displacement patch, separate full-source changes, build
receipt, executable/library hashes, include and AMGX configuration hashes,
and the AMGX source/build-cache evidence. The large prebuilt AMGX library is
not included; `AMGX_DEPENDENCY.json` gives its loaded SHA-256. The archived
source was checked byte for byte against the recorded build source, but no
independent clean rebuild or fresh 15-run reproduction is claimed here.
The command lists and supervisor CLI were inspected; this full native rerun
was not executed as part of preparing these instructions.

Keep new native outputs outside the frozen release. To recompute the two
native audits from them, archive each branch's run folder with the same
internal path as the original `*_outputs.tar.gz` (for example,
`L9/runs/exact/`), place those archives beside the corresponding campaign
JSON and input archive, then run the frozen
`audit_auditor_dynamic_outputs.py` or `audit_auditor_physical_outputs.py`
with `--root` pointing to that new evidence directory. The original
`verify_release.py` is deliberately tied to the original manifest and will
reject replacement results.
