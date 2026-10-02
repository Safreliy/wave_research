# Auditor revision: fixed tests before new outcomes

This campaign uses the existing parent-Hermite matched boundary extension, the
same-streamfunction bilinear centroid control, and closed-form exact initial
phase means. The Aphros executable and library are the single corrected
zero-displacement build from `/opt/gpu-cfd/prefix_zero_flux_guard_20261001`.
The L9, L10 and L11 flat harmonic runs use the same receiver physics and
branchwise common geometry, gas velocity, and face inputs. End time is 0.1 of
the prescribed wave period. Default source sampling is 512 boundary nodes;
trace refinement factor is 8 on L9/L10, reproducing the historical inputs.
The new L11 run uses factor 16 to hold the L10 trace/grid ratio fixed. Thus
trace-spacing/grid-spacing ratios are respectively
0.125, 0.25 and 0.25. The first two levels reproduce the published input
policy; the L10--L11 pair holds this ratio fixed. These are three receiver
levels with a declared trace policy, not a pure fixed-source or uniform
joint-refinement sequence.
An additional, separate initial-error experiment fixes the trace/grid ratio
at 0.125 on all three levels (factors 8, 16 and 32 for 512, 1024 and 2048
columns). It uses the same source nodes, exact initial phase means and
bilinear control. It is reported separately from the 8/8/16 native series,
regardless of whether its observed slopes are favourable. If native runtime
permits, the factor-16 L10 and factor-32 L11 matched branches will also be
continued to 0.1T against exact/control branches with byte-identical common
fields. These runs test a fixed *refined trace/grid ratio* under receiver
refinement, not convergence of boundary-source discretization.
The dynamic primary metric is liquid-volume-weighted global velocity L2
against the same-grid exact-input numerical trajectory. Secondary metrics are
column-depth and VOF differences. At each resolution we retain all three
branches, including unfavourable outcomes. Three-level rates are descriptive;
they do not establish an asymptotic theorem.

For the independent physical test, retain the inviscid, zero-rest-validated,
small-amplitude two-fluid linear wave and the receiver's density-weighted
velocity assignment. Compare exact phase means, matched reconstruction and
same-streamfunction bilinear centroid control at 512x32 and 1024x64 with
identical assignment and same-grid setup. Primary metric: nonzero-time RMS
column depth against the cell-averaged linear solution, divided by the fixed
linear wave RMS amplitude. Report all recorded times and the endpoint,
independently of sign. Also report discrepancy to exact-input native branch,
mass, initial phase-field norms, and zero-amplitude floor where run. No
post-result change in amplitude, assignment or receiver settings.

Transfer cost is measured separately for boundary trace preparation, cell
moment construction and native input serialization where separable. Use
repeat timing with warm-up on named hardware, reporting medians and spread.
Do not conflate transfer cost with CFD, checkpoint or file I/O.

Inputs, scripts, Aphros source/patch, build metadata, executable and library
SHA-256, commands, environment versions, raw-field norms, and output hashes
are frozen in this new campaign. Existing `20261001` archives and the Zenodo
v0.31.0 record are historical immutable references. This revision has no
claim of replacing the original release DOI until it is separately released.
