# Conservative divergence-conforming BIE-to-VOF state transfer

Research code and evidence package for the manuscript **“Conservative
divergence-conforming state transfer from adaptive potential-flow BIE to
embedded VOF: overturning waves over variable bathymetry.”**

The current manuscript is available as [`paper.pdf`](paper.pdf). Its central
result is a one-way transition that preserves the physical liquid measure,
constructs a divergence-conforming MAC field, and constrains receiver-grid
momentum. The resulting VOF continuation is stable on receiver levels 8--10
through nondimensional time 6. The level-10 gas cavity is not reproduced on
levels 8 and 9, so this repository does **not** claim grid-converged impact or
air entrainment.

## Repository contents

| Path | Contents |
| --- | --- |
| `paper.pdf` | Compiled manuscript linked from GitHub |
| `manuscript/` | LaTeX source and bibliography |
| `research/` | Euler--BIE, handoff, VOF-analysis, plotting, and test code |
| `research/results/` | Compact machine-readable evidence, ablations, timing-screen provenance, and figures |
| `research/two_phase_basilisk/` | Basilisk receiver, conservative aperture operator, and generated L8--L10 inputs |
| `DATA_AVAILABILITY.md` | Included data, omitted raw frames, and the expected archive layout |

Development caches, LaTeX intermediates, rendered QA pages, executables, and
the historical thesis/notebook tree are intentionally excluded.

## Quick verification

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\python -m pip install -r requirements-dev.txt
# Linux/macOS: .venv/bin/python -m pip install -r requirements-dev.txt
python -m pytest -q research
python scripts/check_artifacts.py
```

The packaged state passes 111 tests. The artifact check validates the PDF,
JSON/NPZ evidence, the three receiver levels, and the deliberately negative
topology and morphology decisions.

## Reproduce packaged figures

The close-quadrature experiment is small enough to rerun directly:

```bash
python research/near_self_quadrature_study.py
```

The MAC handoff plot and the two figures backed by packaged derived arrays can
be regenerated without the large raw receiver archives:

```bash
python research/plot_mac_face_handoff_audit.py
python research/replot_publication_figures.py
```

The matched-time VOF comparison and vorticity keyframe require the raw frame
archives described in [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md). Their
published PNGs and the scripts used to create them are included.

## Build the paper

With a LaTeX distribution providing `latexmk` and BibTeX:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-paper.ps1
```

or:

```bash
./scripts/build-paper.sh
```

Both commands compile `manuscript/main.tex` and replace the root `paper.pdf`
only after a successful build.

## Full receiver calculation

The three-grid receiver is a Linux/Basilisk calculation. Exact compile flags,
the pinned container digest, the conservative transport headers, and the
generated L8--L10 handoff inputs are documented in
[`research/two_phase_basilisk/README.md`](research/two_phase_basilisk/README.md).
The full runs are computationally expensive; the checked-in diagnostic logs
and derived evidence allow the reported numerical decisions to be audited
without rerunning them.

## Citation

Use [`CITATION.cff`](CITATION.cff) from GitHub’s “Cite this repository” menu.
The publication revision is tagged `v0.28.1-figure-citations`. Update the
citation metadata with the journal DOI and the repository archive DOI when
those identifiers exist.

## License

Original research code is MIT-licensed; the manuscript, figures, and derived
data are CC BY 4.0. See [`LICENSE.md`](LICENSE.md). No third-party raw dataset
is redistributed here.
