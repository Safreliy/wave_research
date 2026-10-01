# Compatibility of the matched trace with the fixed overturning-wave source

The current `parent_hermite_matched` extension is defined for horizontal,
initially flat free-surface strips. It deliberately rejects a curved upper
boundary before generating its matched normal-jet horizontal edge means.
`research/gpu_port/physical_trace.py` requires the surface-height span to be
at most `1e-10`; the negative regression in `test_physical_trace.py` checks
that rejection. The older general fitted-moment transfer is a distinct
component and does support a curved source.

The frozen `boundary_2304.npz` used for the earlier wave experiment has
SHA-256 `0a09d3f9ed3e5e3bd1158d3318780b265ce4bec5edf94abe14cb41961bdd132b`.
Using its actual transfer preprocessing `boundary_traces(..., factor=16)`
yields 36,864 surface nodes, height span `0.7710403162907755`, and 323
nonpositive increments in the ordered surface x coordinates (20 in the
2,304-node source). This is a non-monotone parametrization, so a construction
that assumes a single-valued graph `z = eta(x)` would not cover this fixed
state. Directly calling `flat_surface_jet_means` on these preprocessed traces
raises `ValueError: matched source-jet edges require a flat surface`.
`audit.json` records source and code hashes, geometry metrics and the exact
rejection; `research/gpu_port/audit_curved_matched_compatibility.py` reproduces
it. The independent existing `test_physical_trace.py` passed. This is a
compatibility audit, **not** a native simulation or an accuracy result.

The archived earlier curved-wave evidence establishes initial transfer
conservation and selected local accuracy for the older fitted-moment scheme;
on the native geometry it includes 12 independently integrated **bed** cut
cells and a completed 33-step native startup to `t=0.01`. The reported longer
v71/v72 continuations did not reach their validation target. Their terminal
logs and raw states were not frozen in the local handoff; the currently
archived source is one fixed BIE state, not a converged nonlinear BIE time
series. Consequently neither the old startup nor the flat-wave gains validate
the new matched extension through wave formation or overturning.

For a paper whose claim is confined to conservative transfer plus the flat
thin-strip remedy, this is a stated limit rather than a missing control on
the demonstrated flat result. A claim of improved breaking-wave prediction
would require separate evidence. A bounded next study should proceed in this
order:

1. Define a curved, orientation-aware matched construction on source arc
   segments, including cells with multiple or non-graph surface crossings.
   Specify how the same physical trace enters the cut boundary integral and
   adjacent shared Cartesian edge mean; retain exact shared-edge cancellation
   and a demonstrated liquid-volume/momentum closure. A flat limit must
   reproduce the frozen matched implementation. This is a new method, not a
   flag change to the current code.
2. On an independent manufactured curved source, compare against directly
   integrated phase-volume means on three grids and multiple cut fractions,
   including near-empty slivers. Freeze geometry, source sampling, baseline,
   metrics and error gates before inspecting outcomes. Use a non-overturning
   case and a separate non-graph case; failure on the latter must remain
   visible rather than being silently simplified to a graph.
3. Apply the validated method to the same frozen BIE state and receiver-matched
   geometry as the old transfer. Compare local BIE-reference means across
   surface and bed cuts, including `q <= 0.05`, alongside q-weighted global
   errors, conservation and native loaded state. The old 12-cell reference
   excludes surface cuts and thin slivers, so it cannot alone certify this
   step.
4. Only then compare same-grid native trajectories from the same BIE source,
   geometry, gas field, solver/configuration and timestep, first over a short
   qualified pre-overturning interval and then through the difficult event.
   A finer/halved-step receiver and independently validated physical
   observables are needed to claim prediction accuracy; two branches merely
   diverging do not identify which is closer to the physical solution.

No native wave run was launched in this audit because the present matched
implementation would reject the required input, and bypassing that guard
would turn the run into an unvalidated new numerical method.
