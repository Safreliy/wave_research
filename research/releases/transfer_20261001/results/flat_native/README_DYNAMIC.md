# Flat-wave transfer: frozen native follow-up

This bundle tests whether improved phase-cell velocity initialization persists after native two-phase wave evolution. It is an error-propagation benchmark against a **same-grid Aphros trajectory initialized with closed-form exact phase-cell means**. That reference is numerical after (t=0); it is not an exact physical solution.

The prescribed streamfunction is a smooth mode-16 harmonic wave on a period-64 domain, with potential amplitude 0.02, effective liquid depth 0.97006249, zero background current, a fixed flat embedded bed, and a free surface crossing the top cut-cell row at a small volume fraction. The benchmark ends at (t=0.5250717658), one tenth of the corresponding linear capillary-gravity period. All branches within each comparison use the same geometry, native receiver, physical parameters, and time-step setting. The practical control evaluates the curl of the **same sampled streamfunction** by bilinear interpolation at liquid centroids. The primary error is global liquid-volume-weighted relative (L^2) velocity error, weighted by the reference liquid fractions.

| Grid | Trace factor | Native step | Fitted velocity error | Practical control | Control/fitted |
|---|---:|---:|---:|---:|---:|
| 512×32 | 8 | full | 0.0398835 | 0.0138609 | 0.35 |
| 512×32 | 8 | half | 0.0408403 | 0.0141904 | 0.35 |
| 1024×64 | 8 | full | 0.0433953 | 0.00425970 | 0.10 |
| 1024×64 | 8 | half | 0.0454939 | 0.00446877 | 0.10 |
| 512×32 | 32 | full | 0.0292106 | 0.0138609 | 0.47 |
| 1024×64 | 32 | full | 0.00153448 | 0.00425970 | 2.78 |
| 1024×64 | 32 | half | 0.00159996 | 0.00446877 | 2.79 |
| 2048×128 | 64 | full | 0.000508178 | 0.00174890 | 3.44 |

“Full” uses maximum step 0.0025 and CFL/CFLa 0.1; “half” uses 0.00125 and 0.05. The 512×32 refined branch was deliberately not repeated at half step after its full-step negative result. The original trace factor 8 fails to improve propagation against the practical control. Fourfold trace refinement yields a matched-step advantage at 1024×64 and a second, finer spatial resolution. The refined 512×32 case still loses. No untested case is represented as a win.

The separate cross-time comparison uses full-step candidates against the half-step exact-initial reference. At 1024×64, trace factor 32, fitted error is **0.0209794** versus control **0.0193803**: the fitted candidate is worse in this comparison. The full/half exact-initial reference discrepancy is **0.0204241**, much larger than the matched-step transfer error. Thus the positive matched-step result establishes reduced error *relative to the same native numerical trajectory*, not improved accuracy against a physical trajectory or a temporally converged solution. Cross-time results are included for all four cases with an available half-step reference. There is no half-step reference at 2048×128.

Fourfold trace refinement changes the boundary-trace and edge-moment reconstruction, including values and gradients at boundary intersections. The bulk streamfunction changes by at most 1.40e-12 in relative norm; recomputed preprocessing MAC face velocities change by at most 2.15e-12 in absolute value. Serialized common face arrays are copied unchanged from the original reference, while the native receiver reconstructs its own face fluxes from each branch's cell velocities. Liquid geometry is identical. This check does not isolate one integration subcomponent as the sole cause. The exact initial phase means were independently checked by Gaussian quadrature in the companion article package.

`flat_native_selected_outputs.tar.gz` contains 21 complete native runs across the declared matrix, selected VF/EBVF/VX/VY time-series fields, actual native configuration and log/stat files, semantic audits, analysis JSON, and a SHA256 allowlist for 702 payload files. `frozen_native_endpoint_audit.json` independently recomputes all ten full-step primary errors from fields *inside that frozen archive* and matches their recorded values exactly. The five `flat_native_*_inputs.tar.gz` archives contain frozen initial inputs. `frozen_input_archive_hashes.json` lists their SHA256 values.

The historical preparation-script snapshots for the earlier 512×32 and 1024×64 inputs were not retained. `input_reconstruction_audit.json` in each condition records a current compatible preparation workflow that reproduces all 33 native raw/body inputs byte-for-byte. Original workflows recompute 33 files; refined workflows deliberately copy 29 frozen reference/common files and newly compute four velocity files. The current script must not be described as the exact historical source snapshot. The 2048×128 case used the current recorded preparation script. Native solver source/build provenance and its limitations are documented separately in the companion package.
