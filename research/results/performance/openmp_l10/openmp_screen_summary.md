# OpenMP screening

This is a short performance/equivalence screen, not an L11 physics or convergence result.

| threads | wall [s] | speedup | mass drift | rel. kinetic vs 1 | post-div RMS | post-div Linf | VOF mask mismatch | components |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 57.950 | 1.000 | -5.668e-09 | 0.000e+00 | 4.549e-12 | 2.976e-11 | 0.000e+00 | 1 |
| 4 | 19.820 | 2.924 | -5.668e-09 | 2.348e-05 | 4.647e-12 | 2.950e-11 | 0.000e+00 | 1 |

Fastest observed configuration: 4 threads (2.92x).
The trajectories are not bitwise identical: OpenMP reduction order produces a measurable kinetic-energy spread. A longer fixed-threshold L10/L11 equivalence run is required before using the fastest setting for publication data.
