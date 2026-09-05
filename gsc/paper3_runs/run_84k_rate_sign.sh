#!/usr/bin/env bash
# 84k HARDWARE-MINIMALISM corner: RATE readout (spike-counter) + SIGN write (1-bit) + cold trained head.
# Rate+sign+cold was 0.762 @30k (nearly the full recipe); does the cheapest combo hold at full scale?
# Launch AFTER run_84k.sh (jug leaky + Adam leaky) finishes.
set -u
cd "$(dirname "$0")/.."
OUT="paper3_runs"; SUM="$OUT/SUMMARY.txt"
TAG=full84k_jug_sign_cold_RATE
echo "[$(date -Is)] START $TAG :: --readout rate --n 84000 --eval_n 11005 --epochs 20 --pl_buffer_n 8 --rr_cold --rr_center --rr_lr_mult 100 (SIGN write, no var_norm)" | tee -a "$SUM"
python3 gsc_temporal_harness.py --readout rate --n 84000 --eval_n 11005 --epochs 20 --bs 128 \
  --pl_buffer_n 8 --rr_cold --rr_center --rr_lr_mult 100 > "$OUT/$TAG.log" 2>&1
echo "[$(date -Is)] DONE  $TAG :: $(grep -oE 'best val=[0-9.]+' "$OUT/$TAG.log" | tail -1)" | tee -a "$SUM"
