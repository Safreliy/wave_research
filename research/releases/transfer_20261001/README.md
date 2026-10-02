# Conservative BIE-to-VOF transfer: code and numerical evidence (2026-10-01)

This release accompanies `manuscript/transfer_article.tex` at the repository
root. It contains the frozen numerical evidence for the flat-strip matched
boundary operator, the same-grid native propagation comparison, and the
small-amplitude two-fluid test. The older repository files
`manuscript/main.tex` and `paper.pdf` concern an earlier bathymetry study;
they are **not** the manuscript represented by this release. The current
compiled article is [`manuscript/transfer_article.pdf`](../../../manuscript/transfer_article.pdf).
The manuscript and figures were updated on 2 October 2026; the numerical
archives retain their original bytes. This is a research software and data
release accompanying a manuscript, not a journal publication.

## Contents and integrity

`SHA256SUMS.json` is an explicit per-file allowlist with sizes and SHA-256
digests. Run `python research/releases/transfer_20261001/verify_release.py`
from the repository root to verify it. Each selected-output archive has its
own member manifest or comes from a frozen archived campaign. The two
`*_selected_outputs` subset archives were deterministically repacked for Git
size limits; their internal `subset_manifest.json` gives the source archive
hash and SHA-256 for every selected original member. The original full
21-run native archive has SHA-256
`c8a484a279d4b0cd328ea187a0d651eaccc9b1dc277427da0b54c40651ce7ac8`;
this release retains the exact/control branches needed for the L9/L10
matched comparison. The original mass-control archive has SHA-256
`c07f4bc7eb345ee107ced7e68892cdc28810ccf5a2da164b89a353c3347354d6`;
this release additionally retains its two half-amplitude branches.

| Area | Files | Scientific role |
|---|---|---|
| `archives/method_trace_hermite_20261001_evidence.tar.gz` | Immutable 15-case and six held-out CPU campaigns, executed source snapshots, negative Hermite pilot, invariants and audit | Analytic phase-cell means and operator tests |
| `archives/*matched*`, `archives/native_baseline_selected_outputs.tar.gz` | Frozen input and selected raw output fields, logs and native status | Matched versus same-streamfunction centroid control, L9/L10 and L10 half step |
| `archives/twofluid_guarded_mass_factorial_complete_outputs.tar.gz` | Seven native runs, including zero, exact/matched/practical crossed with phase/mass injection | Physical column-depth benchmark against a linear two-fluid reference |
| `archives/twofluid_half_amplitude_outputs.tar.gz` | Two half-amplitude exact-input controls | Small-amplitude scaling check |
| `results/` | Primary and independent JSON analyses, protocols and early transfer arrays | Recompute tables and inspect negative/limited findings |
| `curved_compatibility/` | Frozen BIE state, code, JSON and assessment | Explicit failure of the new flat-only matched construction on a non-graph source |
| `solver_patch/`, `native_source_provenance/` | Aphros zero-flux guard, MIT notice, runtime hashes and historical provenance | Distinguish guarded physical receiver from the earlier installed receiver |
| `code/` | Current readable Python sources and independent audits | Re-run CPU checks and regenerate compatible native input workflows |

The executed matched operator and native builder are specifically frozen
*inside* the immutable method archive. Current readable code in `code/` may
contain later documentation or adjacent changes and should not be mistaken
for every historical executed snapshot. The earlier native input preparation
scripts had a snapshot gap; `results/flat_native/README_DYNAMIC.md` identifies
the byte-identical compatible reconstruction and its limits.

## Replay

With Python 3.11+ and NumPy/SciPy/Matplotlib/Shapely available, run from this
release directory (or set `PYTHONPATH=research/releases/transfer_20261001/code`
from the repository root):

```sh
PYTHONPATH=code python code/research/gpu_port/test_physical_trace.py
PYTHONPATH=code python code/research/gpu_port/test_phase_mixture_transfer.py
PYTHONPATH=code python code/research/gpu_port/audit_curved_matched_compatibility.py \
  --boundary curved_compatibility/boundary_2304.npz \
  --output /tmp/curved_compatibility_replay.json
```

For the numerical paper values, consult the frozen command/protocol files in
the method archive, the corresponding input/output archives and the independent
checkers in `code/independent_audits/`. The seven-run physical analysis is
`results/physical/guarded_mass_factorial_complete_height_analysis.json`;
`results/physical/guarded_mass_factorial_complete_independent_audit.json`
recomputes its endpoint values directly from archived VF/EBVF/VX/VY fields
and checks common inputs and runtime. Re-run that independent checker with
`python replay_physical_audit.py --report /tmp/physical_audit_replay.json`.
The wrapper stages only the three frozen inputs and protocol in a temporary
directory, then invokes the checker explicitly with
`--outputs archives/twofluid_guarded_mass_factorial_complete_outputs.tar.gz`,
`--analysis results/physical/guarded_mass_factorial_complete_height_analysis.json`
and the requested `--report`. The checker defaults point at an *older partial*
factorial archive; do not rely on those defaults. The earlier matched native endpoint
audit is in the method archive. Figure generation sources are in
`code/paper_scripts/`, and the manuscript's figure/table provenance is beside
each figure/table in `manuscript/` at the repository root.

The native simulations need Aphros, MPI and the GPU/AMGX runtime in addition
to the archived inputs. They have **not** been replayed in a fresh third-party
environment. `solver_patch/build_manifest.json` records the guarded receiver
build from upstream Aphros commit `b60ce3da52c19935fa24c778f62f02141eaf7f80`
plus `zero_flux_guard.patch`; the guarded and baseline runtime binaries are
included. `native_source_provenance/README.md` explains why the exact dirty
source snapshot of the *earlier* installed receiver cannot be cryptographically
identified from the available build records. No private connection helpers
or credentials are part of this release.

## Interpretation and limits

The matched boundary construction is proved and tested for horizontal flat
surface strips. `curved_compatibility/ASSESSMENT.md` shows that the current
matched implementation correctly rejects the earlier non-graph overturning
state; this is a compatibility test, not a curved-wave accuracy result.
The linear two-fluid wave supplies an independent small-amplitude reference;
it is not an exact solution of the nonlinear native receiver. At the final
physical time, phase-practical to density-weighted matched reduces normalized
column-depth error by 52.1887%, while the within-density-weighted comparison
reduces it by 5.4896%. The within-projection ranking reverses at one earlier
nonzero output, and no universal trajectory advantage is claimed. The old
geometry-only 12-cell quadrature reference sampled bed cuts only, and the
old 33-step native startup did not validate long-time breaking dynamics.

Original research code in `research/` is MIT-licensed; manuscript, figures
and original derived data are CC BY 4.0 under the repository `LICENSE.md`.
The Aphros material keeps its own MIT licence, copied to `solver_patch/APHROS_LICENSE`
and `native_source_provenance/aphros_LICENSE`. This release does not relicense
third-party code. Cite the repository's `CITATION.cff` and the DOI of the archived version used.

## Publication snapshot

`TRANSFER_PUBLICATION_SHA256SUMS.json` at the repository root covers the
current article PDF, LaTeX source, figure/table assets and publication tools.
Run `python scripts/verify_transfer_publication.py` from the repository root.
The build receipt is `results/manuscript_build_20261002.json`; the older
`results/manuscript_draftmode_validation.json` is an earlier source check,
not the build record for the current PDF. The native experiments were not
rerun for the editorial update.
