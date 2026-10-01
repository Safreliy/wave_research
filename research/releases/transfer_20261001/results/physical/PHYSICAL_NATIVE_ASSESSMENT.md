# Frozen two-fluid native follow-up (2026-10-01)

This file records a bounded subject-matter test of the transfer operator on a
linearized two-fluid standing capillary–gravity wave. The native receiver solves
its nonlinear discrete equations; the independent analytic reference is a
**linear** solution, so it is a controlled small-amplitude target rather than an
exact nonlinear Navier–Stokes trajectory. The domain length is 64, horizontal
mode 16, liquid depth 0.97006249, `512 × 32` cells, potential amplitude
`A = 2e-5`, and final time `0.1T = 0.5256608823752456`. An `A/2` control and
an exact-zero control were declared before candidate outcomes. Viscosity is zero,
and the bed is slip/impermeable in this experiment.

The primary observable is the RMS column-depth error
`||H_native - H_linear||_RMS / (B sinc(kh/2)/sqrt(2))`, with `B` fixed by the
analytic potential amplitude. The zero run is reported as an absolute floor and
is never subtracted. Every candidate has the same q/cs/body geometry, physical
parameters, runtime and step setting; the serialized face arrays are common,
but native `flux_init zero` **reconstructs the actual initial face flux** from
each branch's cell velocity. The density-weighted input is the candidate
`u = (rho_l q u_l + rho_g (cs-q) u_g)/(rho_l q + rho_g(cs-q))`, with the same
analytic gas phase mean `u_g` in mixed cells. This is a tested projection
hypothesis, not an asserted exact contract for the native velocity variable.

## Native receiver defect and control

The first zero-velocity pilot on the original native runtime lost 0.4029% of
its liquid volume after one step. Single-factor controls removing surface
tension, restoring old wall/viscosity, and widening the top liquid row also
failed. A 2D zero-flux PLIC path returned NaN because a zero swept displacement
made a degenerate line interval; subsequent clipping erased full liquid cells.
An isolated, fresh build from the same source reproduced the native failure.
A guard returning zero transported volume for exactly zero displacement
passed the same native one-step case with zero mass drift, while 16
representative nonzero kernel results remained bitwise unchanged. The original
installed runtime and earlier benchmarks were not modified. Source, patch,
build commands, binaries, logs, independent source diff and kernel audits are
frozen in `source_build_guard/`; the initial failure/control raw fields are in
`twofluid_zero_diagnostic_outputs.tar.gz`.

## Controlled trajectory result

All seven selected runs reached `0.1T`; all their semantic audits are valid.
The zero-run H floor is `1.11e-16` absolute and its mass drift is zero.
The analytic-reference H errors for the two exact-input projection controls
scale with amplitude: at `A`, phase input `0.0353352787008` and mass input
`0.0177546477088`; at `A/2`, respectively `0.0353360055572` and
`0.0177543767329`. This checks the intended linear regime at this resolution.

| Liquid transfer | Phase injection | Density-weighted mixture |
|---|---:|---:|
| Exact phase means | 0.035335278701 | 0.017754647709 |
| Matched boundary trace | 0.035358432260 | 0.017778264816 |
| Same-streamfunction bilinear centroid | 0.037184237465 | 0.018810903971 |

The table gives the **normalised error against analytic column depth** at
`0.1T`, not an error relative to a native branch. With density weighting fixed, matched trace
reduces this error by `5.4896%` against the practical bilinear control
(`1.05808×` error ratio). With the transfer method fixed, density weighting
reduces it by about `49.4–49.7%`. The complete practical phase-injection to
matched density-weighted pipeline reduces the endpoint error by `52.1887%`
(`2.09156×`). Relative to the native exact-mass branch, the endpoint H
difference is `2.36173e-5` for matched and `0.001056264` for practical;
that `44.72×` propagated-input ratio must not be called a `44.72×` physical
accuracy gain. All values, initial/loaded time behavior, mode amplitudes,
velocity errors, mass and divergence audits are in
`guarded_mass_factorial_complete_height_analysis.json`.
An independent checker recomputed the endpoint norms from the frozen raw fields,
verified all seven run statuses and common inputs, and recorded matching results
in `guarded_mass_factorial_complete_independent_audit.json`.

The advantage of matched over practical with density weighting is **time
dependent**. At `t=0.13125`, practical happens to have the smaller physical H
error (`0.000161713` versus `0.000406056`); matched is better at the other
three nonzero dumps, including the final time. The combined phase-practical to
mass-matched pipeline improves physical H at all four nonzero dumps. This is one
grid and one wave geometry, so it does not establish universal physical
accuracy or asymptotic spatial convergence for the two-fluid case. The separate
original-runtime wave benchmarks on L9/L10 and L10 half-step establish robust
reduction of matched-step **propagated input differences**, not an independent
physical solution error. No additional parameter search was run after the
factorial result.

## Frozen data and replay

The final selected-output archive is
`twofluid_guarded_mass_factorial_complete_outputs.tar.gz` with SHA-256
`f6a99920820d5491422d1b71e3f0e7463b1da8e6b4d6a5113bfc49e0cc3de3e0`.
It contains the zero control and all six nonzero 2×3 branches, their actual
native configurations, statuses, audits, logs and raw fields. The final
analysis and source are `guarded_mass_factorial_complete_height_analysis.json`
and `factorial_complete_python_snapshot/`. The two candidate input folders,
preparation report and launch protocol are in
`flat_native_L9_guarded_mass_factorial/`; their input archive is
`flat_native_L9_guarded_mass_factorial_inputs.tar.gz` (SHA-256
`59a282ce2d8335708f96fa79fc3a67a34c482fea69295911b666849f554343a7`).
The analytic base inputs are in `flat_native_L9_twofluid_inputs.tar.gz` (SHA-256
`65631c5876c1b02cba090d71d181702e9cb48579b686b439ed38d49c05d8403e`),
and the phase candidate inputs in
`flat_native_L9_guarded_transfer_inputs.tar.gz` (SHA-256
`399e54d4be03083b92a00a9dc3003ac3ac137001eff468103c8fe44185970493`).

The preparation code, analytic solution, launch helper and analyzers used in
this follow-up are frozen in `executed_python_snapshot/` and
`factorial_complete_python_snapshot/`. Code hashes recorded by each protocol or
analysis file identify the executed/replayed version; other copied dependencies
are provenance aids. The partial factorial archive and JSON are retained under
their original names; use only the `*_complete*` archive and JSON for the final
2×3 table. To recompute that table in this workspace, run:

```sh
python research/gpu_port/analyze_guarded_twofluid_mass_factorial.py \
  --archive research/results/publication_package/transfer_physical_pilot_20261001/twofluid_guarded_mass_factorial_complete_outputs.tar.gz \
  --output research/results/publication_package/transfer_physical_pilot_20261001/guarded_mass_factorial_complete_height_analysis_recomputed.json
```

The isolated receiver build commands and CMake state are in
`source_build_guard/build_manifest.json`; the source archives include the
upstream licence. The original old-runtime source-at-compile-time state is not
cryptographically attested, while the isolated baseline/guarded build contrast
has source and runtime archives. The attempted eighth-step temporal reference
failed a 300-second checkpoint deadline before advancing from t=0; its
status/log and explicit exclusion are frozen separately under
`transfer_temporal_followup_20261001/`.
