# OpenMP screening

This is a short performance/equivalence screen, not an L11 physics or convergence result.

| threads | wall [s] | speedup | mass drift | rel. kinetic vs 1 | post-div RMS | post-div Linf | VOF mask mismatch | components |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.250 | 1.000 | -6.431e-08 | 0.000e+00 | 1.403e-11 | 5.553e-11 | 0.000e+00 | 1 |
| 2 | 2.650 | 1.604 | -6.431e-08 | 1.939e-05 | 1.255e-11 | 4.447e-11 | 0.000e+00 | 1 |
| 4 | 1.910 | 2.225 | -6.431e-08 | 3.470e-05 | 1.174e-11 | 5.289e-11 | 0.000e+00 | 1 |
| 8 | 2.120 | 2.005 | -6.431e-08 | 7.036e-05 | 1.285e-11 | 7.161e-11 | 0.000e+00 | 1 |

Fastest observed configuration: 4 threads (2.23x).
The trajectories are not bitwise identical: OpenMP reduction order produces a measurable kinetic-energy spread. A longer fixed-threshold L10/L11 equivalence run is required before using the fastest setting for publication data.
