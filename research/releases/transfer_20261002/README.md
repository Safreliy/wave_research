# Transfer follow-up evidence (2026-10-02)

This directory is a reproducible supplement for the matched thin-strip
streamfunction transfer study. It freezes the predeclared protocol, three-grid
initial and dynamic tests, two-grid small-amplitude physical test, transfer
cost timings, full native output fields, and the source/build evidence used to
create them. It is **not yet assigned a DOI**. The earlier `v0.31.0` archive
and DOI `10.5281/zenodo.23104557` document historical evidence; this package
does not alter that archive or make it a source for the new measurements.

## What is measured

- `joint_refinement_initial_audit.json`: initial velocity errors on W16
  grids 512×32, 1024×64, and 2048×128, using a fixed refined trace/grid
  ratio. Its exact reference is the closed-form harmonic liquid-cell mean.
- `dynamic_three_grid_audit.json`: dynamic receiver-field discrepancies on
  three grids against the *same-grid numerical trajectory* initialized from
  exact harmonic cell means. That reference is not an exact nonlinear-flow
  solution. The archived output fields and semantic records are checked by
  `verify_release.py`.
- `physical_two_grid_audit.json`: column-depth error on two grids against a
  cell-averaged linear inviscid two-fluid capillary-gravity wave. Its fixed
  denominator is the RMS linear-wave amplitude, and it is not an exact
  nonlinear Navier–Stokes reference.
- `cost_L9.json`, `cost_L10.json`, `cost_hardware.json`: measured preparation
  costs, separate from native CFD and checkpoint time.

The three dynamic receiver grids follow the trace policy in
`PREDECLARED_PROTOCOL.md` and `PROTOCOL_AT_DYNAMIC_FREEZE.md`; only the
separate initial-refinement test keeps the refined trace/grid ratio fixed on
all three grids. Exact, matched, and bilinear branches are retained, including
unfavourable results. `physical_initial_metadata_erratum.json` clarifies that
one executed metadata field named `liquid_momentum` reported receiver-velocity
loading rather than pure liquid-phase momentum. It does not change any frozen
input or output.

## Integrity and replay

The explicit allowlist and SHA-256 digest of every shipped file are in
`SHA256SUMS.json`. Run from this directory:

```bash
python verify_release.py
python verify_release.py --replay
```

The first command checks all files, every campaign-recorded input-field hash,
branchwise common geometry/face fields, the audit-to-campaign and
audit-to-archive links, all 15 native run statuses, and the raw-field hashes
listed by the auditors. The second
extracts the archived inputs into a temporary directory and re-runs the
frozen initial, dynamic, and physical audit scripts, comparing their output
with the frozen JSON reports. It requires Python and NumPy; the frozen
source environment is recorded in `code/MANIFEST.json`. Replay can take
several minutes and needs enough temporary disk space for extracted fields.

Inputs are `dynamic_inputs.tar.gz`, `physical_inputs.tar.gz`, and
`joint_refinement_inputs.tar.gz`. The first two preserve the complete
campaign input trees; the third retains only the inputs needed to recompute
the separate initial-refinement audit. The 15 `*_outputs.tar.gz` files
preserve native run statuses, semantic audits, checkpoints, and raw fields.
The source files under `code/research/...` are the frozen bytes identified by
`code/MANIFEST.json`, including executed historical generator snapshots.
`review/input_review.json` is an independent construction audit; its script
can be rerun from the original full workspace containing the 2026-10-01
legacy inputs, which are intentionally not duplicated here.

The guarded Aphros source, baseline and corrected runtime bundles, patch,
build receipts, source/build match report, runtime configurations and AMGX
source are in `runtime_provenance/`. The AMGX prebuilt library is not copied:
`AMGX_DEPENDENCY.json` records its exact loaded path and SHA-256. The
campaign commands, initial field hashes, and expected Aphros executable and
library SHA-256 values are in `dynamic/campaign.json` and
`physical/campaign.json`. Rebuilding and repeating the native runs requires
the corresponding GPU toolchain and external libraries; `--replay` recomputes
the scientific audits from the archived raw outputs without rerunning CFD.
