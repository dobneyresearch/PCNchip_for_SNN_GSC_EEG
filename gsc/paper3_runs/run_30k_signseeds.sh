#!/usr/bin/env bash
# Follow-up: sign+cold on seeds 1,2 to PAIR the sign-vs-var_norm comparison (var_norm has s0,1,2
# from the main battery; sign has only s0). Depth 8. Launch AFTER run_30k_battery.sh finishes.
set -u
cd "$(dirname "$0")/.."
OUT="paper3_runs"; SUM="$OUT/SUMMARY.txt"
COMMON="--readout leaky --n 30000 --eval_n 4000 --epochs 20 --bs 128 --pl_buffer_n 8 --rr_cold --rr_center --rr_lr_mult 100"
echo "=== sign-seeds start $(date -Is) ===" >> "$SUM"
run () { local tag="$1"; shift; local log="$OUT/$tag.log"
  echo "[$(date -Is)] START $tag :: $*" | tee -a "$SUM"
  python3 gsc_temporal_harness.py "$@" > "$log" 2>&1
  echo "[$(date -Is)] DONE  $tag :: $(grep -oE 'best val=[0-9.]+' "$log" | tail -1)" | tee -a "$SUM"; }
run e3_sign_cold_d8_s1 $COMMON --seed 1
run e3_sign_cold_d8_s2 $COMMON --seed 2
echo "=== sign-seeds end $(date -Is) ===" >> "$SUM"
