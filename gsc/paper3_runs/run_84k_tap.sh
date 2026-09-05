#!/bin/bash
# REGENERATION SCRIPT — GSC 84k, learnable tap on the buffer (A/B vs the committed uniform cell).
# Verified result 2026-08-24: no-tap 0.8340 (reproduces committed 0.832) ; +tap 0.8659 (+0.032).
# Result doc: paper3_runs/RESULTS_tap.md ; reference log: paper3_runs/tap_84k.log
# Run from anywhere: resolves the harness dir relative to this script.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSC="$(dirname "$HERE")"          # neurobench_gsc/
cd "$GSC"

# Committed winning recipe (uniform cell, leaky readout), seed 0. --tap adds the learnable input FIR.
COMMON="--train jug --readout leaky --n 84000 --eval_n 11005 --epochs 20 --bs 128 --seed 0 \
--pl_buffer_n 8 --rr_cold --var_norm_fold --rr_center --rr_lr_mult 100 --buf_uniform"

echo "===== GSC 84k  var_norm  NO-tap  (baseline → expect ~0.832) ====="
python3 gsc_temporal_harness.py $COMMON

echo "===== GSC 84k  var_norm  +tap  (learned delay on the buffer → expect ~0.866) ====="
python3 gsc_temporal_harness.py $COMMON --tap --tap_K 16

echo "########## GSC 84k TAP A/B DONE ##########"
