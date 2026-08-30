# Data availability

This repository keeps the publication package small enough for ordinary Git
while preserving the evidence needed to audit every numerical statement in
the manuscript.

## Included

- the compiled paper and complete LaTeX source;
- all six manuscript figures;
- the close-quadrature JSON table;
- the MAC handoff audit and backend summaries;
- the six-row transfer ablation and the two-time handoff sensitivity summary;
- the complete three-grid topology audit JSON;
- the complete three-grid morphology JSON and compressed derived arrays;
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
├── impact_claim_q_l8_t60/handoff_frames/
├── impact_claim_q_l9_t60/handoff_frames/
└── impact_claim_q_l10_t60_final/
    ├── handoff_frames/
    └── vorticity_frames/
```

The declared thresholds and expected paths are fixed in
`research/impact_claim_q_l8_l9_l10_manifest.json`. Once the raw archive has a
stable public DOI or GitHub Release URL, record it here and in the manuscript;
do not replace the predeclared thresholds after inspecting a finer-grid run.

The compact code-and-evidence package is versioned by the Git tag
`v0.27-audit-revision`. It does not yet have an archival DOI. Deposit the raw
frames and this tagged release in Zenodo or an equivalent archive before
journal submission, then record the DOI here, in `CITATION.cff`, and in the
manuscript.
