#!/usr/bin/env bash
# Paper-3 close-out battery — all 30k/20, seed 0 unless noted. Runs SEQUENTIAL (one GPU).
# Each run's full log -> paper3_runs/<tag>.log ; one-line summary appended to SUMMARY.txt.
set -u
cd "$(dirname "$0")/.."
OUT="paper3_runs"
SUM="$OUT/SUMMARY.txt"
COMMON="--readout leaky --n 30000 --eval_n 4000 --epochs 20 --bs 128 --pl_buffer_n"
echo "=== battery start $(date -Is) ===" >> "$SUM"

run () {  # tag  extra-args...
  local tag="$1"; shift
  local log="$OUT/$tag.log"
  echo "[$(date -Is)] START $tag :: $*" | tee -a "$SUM"
  python3 gsc_temporal_harness.py "$@" > "$log" 2>&1
  local best; best=$(grep -oE 'best val=[0-9.]+' "$log" | tail -1)
  echo "[$(date -Is)] DONE  $tag :: $best" | tee -a "$SUM"
}

# --- Exp 1: clean sign vs var_norm write, both with cold gradient readout, depth 8 ---
run e1_sign_cold_d8    $COMMON 8  --rr_cold --rr_center --rr_lr_mult 100
run e1_varnorm_cold_d8 $COMMON 8  --rr_cold --var_norm_fold --rr_center --rr_lr_mult 100   # == winning recipe (also depth-8 point)

# --- Exp 2: depth sweep (var_norm+cold write), 6/10/12 (8 is e1_varnorm_cold_d8) ---
run e2_varnorm_cold_d6  $COMMON 6  --rr_cold --var_norm_fold --rr_center --rr_lr_mult 100
run e2_varnorm_cold_d10 $COMMON 10 --rr_cold --var_norm_fold --rr_center --rr_lr_mult 100
run e2_varnorm_cold_d12 $COMMON 12 --rr_cold --var_norm_fold --rr_center --rr_lr_mult 100

# --- Exp 3: seed check at depth 8 (seed 0 == e1_* runs above); PAIRED sign vs var_norm ---
run e3_varnorm_cold_d8_s1 $COMMON 8 --rr_cold --var_norm_fold --rr_center --rr_lr_mult 100 --seed 1
run e3_varnorm_cold_d8_s2 $COMMON 8 --rr_cold --var_norm_fold --rr_center --rr_lr_mult 100 --seed 2
run e3_sign_cold_d8_s1    $COMMON 8 --rr_cold --rr_center --rr_lr_mult 100 --seed 1
run e3_sign_cold_d8_s2    $COMMON 8 --rr_cold --rr_center --rr_lr_mult 100 --seed 2

echo "=== battery end $(date -Is) ===" >> "$SUM"
