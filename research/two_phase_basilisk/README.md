# Basilisk two-phase receiver

`bie_handoff_receiver.c` continues the late Euler--BIE state on an adaptive
embedded two-phase grid. `q_embed_vof.h` transports the full-cell liquid
aperture `q = f cs` with the same geometric flux used by the phase-momentum
tracers.

## Reproducibility pin

The archived L8--L10 calculations used Basilisk patch `ad05f362` (7 April
2022) in this immutable container image:

```text
sgls/basilisk-docker@sha256:86e22069efeacadf32eeffbebd627366ec29212622e856ffc485e2fee1f6b764
```

Compatibility with current upstream Basilisk is not claimed.

## Included receiver inputs

The generated headers are the exact receiver-grid inputs used by the three
reported calculations:

- `bie_handoff_data_receiver_conservative_q_t20303125_l8.h` for L8/L6;
- `bie_handoff_data_receiver_conservative_q_t20303125.h` for L9/L7;
- `bie_handoff_data_receiver_conservative_q_t20303125_l10.h` for L10/L8.

They are intentionally committed even though they are generated files: the
paper promises solver inputs, and regenerating the constrained face field
requires a separate sparse/GPU solve.

## Compile and run

From the repository root on Linux, with `BASILISK` set inside the pinned
container:

```bash
cd research/two_phase_basilisk
./build_three_grid.sh 8
./build_three_grid.sh 9
./build_three_grid.sh 10
```

Pass `--run` after the level to start the expensive continuation. Each run
writes its diagnostics and frames beneath `runs/l<level>/`.

The common production flags are:

```text
CONSERVATIVE_Q_EMBED=1
Q_EMBED_MOMENTUM_DIAGNOSTICS=1
END_TIME=6.0
ADAPT_VOLUME_LEDGER=1
POST_ADAPT_REPROJECT=1
POST_ADAPT_PRESERVE_FACE_FLUX=1
FRAME_X_MIN=32
FRAME_X_MAX=46
```

The archived diagnostic logs are under `remote_impact_claim/`. All three end
with `RUN 6`; they contain 301 output times per receiver level. The raw frame
sequences are external to Git as described in the repository-level
`DATA_AVAILABILITY.md`.
