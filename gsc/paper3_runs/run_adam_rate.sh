#!/usr/bin/env bash
# Complete the 2×2 (jug/adam × rate/leaky) at 30k/20: the missing corner = Adam on RATE.
# Answers the observation "jug rate ≈ jug leaky ≈ 0.77" — is 0.77 a shared, readout-independent
# ceiling ~3pp under Adam? Launch AFTER run_84k.sh finishes.
set -u
cd "$(dirname "$0")/.."
OUT="paper3_runs"; SUM="$OUT/SUMMARY.txt"
echo "[$(date -Is)] START adam_rate_30k :: --train adam --readout rate --n 30000 --epochs 20 --eval_n 4000" | tee -a "$SUM"
python3 gsc_temporal_harness.py --train adam --readout rate --n 30000 --epochs 20 --eval_n 4000 --bs 128 \
  > "$OUT/adam_rate_30k.log" 2>&1
echo "[$(date -Is)] DONE  adam_rate_30k :: $(grep -oE 'best val=[0-9.]+' "$OUT/adam_rate_30k.log" | tail -1)" | tee -a "$SUM"
