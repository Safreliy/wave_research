#!/usr/bin/env bash
set -euo pipefail

level="${1:-}"
run_after_build="${2:-}"

case "$level" in
  8)
    minimum_level=6
    input_macro=RECEIVER_CONSERVATIVE_Q_HANDOFF_N576_T20303125_X1024_L8
    ;;
  9)
    minimum_level=7
    input_macro=RECEIVER_CONSERVATIVE_Q_HANDOFF_N576_T20303125_X1024
    ;;
  10)
    minimum_level=8
    input_macro=RECEIVER_CONSERVATIVE_Q_HANDOFF_N576_T20303125_X1024_L10
    ;;
  *)
    echo "usage: $0 {8|9|10} [--run]" >&2
    exit 2
    ;;
esac

: "${BASILISK:?BASILISK must point to the pinned Basilisk checkout}"

source_dir="$(cd "$(dirname "$0")" && pwd)"
run_dir="$source_dir/runs/l$level"
mkdir -p "$run_dir"

"$BASILISK/qcc" -O2 -Wall \
  -DCONSERVATIVE_Q_EMBED=1 \
  -DQ_EMBED_MOMENTUM_DIAGNOSTICS=1 \
  -D"$input_macro"=1 \
  -DLEVEL="$level" -DMINLEVEL="$minimum_level" -DEND_TIME=6.0 \
  -DADAPT_VOLUME_LEDGER=1 \
  -DPOST_ADAPT_REPROJECT=1 \
  -DPOST_ADAPT_PRESERVE_FACE_FLUX=1 \
  -DFRAME_X_MIN=32. -DFRAME_X_MAX=46. \
  "$source_dir/bie_handoff_receiver.c" \
  -o "$run_dir/receiver" -lm

if [[ "$run_after_build" == "--run" ]]; then
  (
    cd "$run_dir"
    ./receiver 2> diagnostics.dat
  )
fi
