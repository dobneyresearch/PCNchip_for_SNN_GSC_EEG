#!/usr/bin/env bash
# 84k FULL-SET headline: jug winning recipe (var_norm+cold, d8) vs Adam. Full 11005 test eval.
set -u
cd "$(dirname "$0")/.."
OUT="paper3_runs"; SUM="$OUT/SUMMARY.txt"
COMMON="--readout leaky --n 84000 --eval_n 11005 --epochs 20 --bs 128"
echo "=== 84k start $(date -Is) ===" >> "$SUM"
run () { local tag="$1"; shift; local log="$OUT/$tag.log"
  echo "[$(date -Is)] START $tag :: $*" | tee -a "$SUM"
  python3 gsc_temporal_harness.py "$@" > "$log" 2>&1
  echo "[$(date -Is)] DONE  $tag :: $(grep -oE 'best val=[0-9.]+' "$log" | tail -1)" | tee -a "$SUM"; }
run full84k_jug_varnorm_cold_d8 $COMMON --pl_buffer_n 8 --rr_cold --var_norm_fold --rr_center --rr_lr_mult 100
run full84k_adam                $COMMON --train adam
echo "=== 84k end $(date -Is) ===" >> "$SUM"
