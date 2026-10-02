# Data availability

The current flat-strip transfer article is `manuscript/transfer_article.tex`.
Its new [experiment package](research/releases/transfer_20261002/README.md)
contains all fields used by the three-grid dynamic and two-grid physical
audits, initial refinement data, timings, source snapshots, build records,
runtime configuration dependencies and SHA-256 manifests. The included
verifier can repeat the three principal numerical audits from these files.
This evidence is archived as [v0.32.0, doi:10.5281/zenodo.23111007](https://doi.org/10.5281/zenodo.23111007).
The current manuscript adds the DOI citation after archival; the numerical
evidence and frozen release manifests are unchanged.

The [original benchmark package](research/releases/transfer_20261001/README.md)
is archived as [v0.31.0, doi:10.5281/zenodo.23104557](https://doi.org/10.5281/zenodo.23104557).
That DOI does not identify the later experiments. The earlier provenance
limits remain documented with that package. The corrected receiver used
for the new central comparisons has a matched source/build/runtime record;
a new third-party clean GPU build and simulation replay has not been performed.
The material below describes the separate earlier variable-bathymetry study.

## Earlier variable-bathymetry data

This repository keeps the publication package small enough for ordinary Git
while preserving the evidence needed to audit every numerical statement in
the manuscript.

## Included

- the compiled paper and complete LaTeX source;
- all six manuscript figures;
- the close-quadrature JSON table;
- the MAC handoff audit and backend summaries;
- the six-row transfer ablation and the negative pre-production two-time
  screen, including its receiver-configuration provenance;
- the complete three-grid topology audit JSON;
- the complete three-grid morphology JSON and compressed derived arrays;
- the short L9 and L10 OpenMP performance/repeatability summaries and plots;
- a 101-frame, losslessly encoded L10 VOF/vorticity animation derived by
  retaining every third raw output;
- L8, L9, and L10 diagnostic logs through continuation time 6;
- the late active BIE handoff state;
- the generated conservative receiver headers for levels 8--10;
- the Python, C, and header sources used for simulation and analysis.

## Not included in Git

The raw VOF/vorticity raster sequences are omitted because they contain 1,509
frames and occupy about 503 MB before archival compression. They are not
required for the unit tests, manuscript build, or audit of the packaged JSON
and NPZ results. They are required to recompute connected components and
matched-time surface profiles from pixels.

Place a full-data archive under the following paths before running the three
raw-frame analysis commands:

```text
research/two_phase_basilisk/remote_impact_claim/
РІвЂќСљРІвЂќР‚РІвЂќР‚ impact_claim_q_l8_t60/handoff_frames/
РІвЂќСљРІвЂќР‚РІвЂќР‚ impact_claim_q_l9_t60/handoff_frames/
РІвЂќвЂќРІвЂќР‚РІвЂќР‚ impact_claim_q_l10_t60_final/
    РІвЂќСљРІвЂќР‚РІвЂќР‚ handoff_frames/
    РІвЂќвЂќРІвЂќР‚РІвЂќР‚ vorticity_frames/
```

The declared thresholds and expected paths are fixed in
`research/impact_claim_q_l8_l9_l10_manifest.json`. Once the raw archive has a
stable public DOI or GitHub Release URL, record it here and in the manuscript;
do not replace the predeclared thresholds after inspecting a finer-grid run.

The compact code-and-evidence package is versioned by the Git tag
`v0.29.0-openmp-visualization`. It does not yet have an archival DOI. Deposit
the raw frames and this tagged release in Zenodo or an equivalent archive
before journal submission, then record the DOI here, in `CITATION.cff`, and
in the manuscript.
