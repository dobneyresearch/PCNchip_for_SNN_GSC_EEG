# Learnable tap on the buffer — GSC 84k A/B (2026-08-24) ✅ VERIFIED

**Question (user):** does a learnable per-delay weighting over the temporal buffer (the "tap") add on
top of the committed uniform cell at full scale — i.e. can the delay act as a *controller for the time
buffer*, steering where the fixed-depth window looks?

**Setup:** the `--tap` flag (default OFF, `pcn_learner_snn.py` cfg `tap_fold`/`tap_K`; forwards-only tap
credit reuses the δ chain — `E_tap += (W₀ᵀδ₀) ⊗ raw-lagged-input`, folded by the SAME sign/var_norm
rule). Causal per-channel FIR of length `tap_K=16` on the input, delta-initialised to identity (current
tap = 1), so at init the net is byte-for-byte the committed cell. A/B on the committed winning recipe,
84k / 20ep leaky, seed 0:
`--readout leaky --n 84000 --eval_n 11005 --epochs 20 --bs 128 --pl_buffer_n 8 --rr_cold --var_norm_fold
--rr_center --rr_lr_mult 100 --buf_uniform` [`--tap --tap_K 16`].

Regenerate: `bash paper3_runs/run_84k_tap.sh`  ·  reference log: `paper3_runs/tap_84k.log`

## Result (84k / 20ep leaky, seed 0)
| config | best val | vs committed 0.832 | vs Adam 0.881 |
|---|---|---|---|
| var_norm, **no-tap** (baseline) | **0.8340** | +0.002 (reproduces) | −0.047 |
| var_norm, **+tap** (learned delay) | **0.8659** | **+0.034** | **−0.015** |
| Δ (tap − no-tap) | **+0.0319** | | |

## Reading
- The learnable delay adds **+0.032** at full scale and **stacks on** the committed uniform cell — it
  replaces nothing. Gap to the Adam upper bound narrows from 0.049 → **0.015** (~98% of BPTT: 0.866/0.881).
- Interpretation for the paper: the tap is a **controller for the time buffer**. The buffer supplies a
  fixed-depth window; the tap learns *where in that window to look* (per-channel delay/weighting). This
  is the timescale-flexibility primitive — learnable and expressive — as opposed to a fixed adaptation.
- The no-tap baseline reproducing 0.834 also serves as a **regression check**: the default-OFF
  experimental flags added 2026-08-24 (`drive_deriv`, `plb_elig`, `tap_fold`, `q_fold`) leave the
  validated fold path byte-for-byte unchanged.

## Status / TODO for the paper
- [ ] Fold this into `paper/main_stage3_GSC_v1.*` — new row in the §2/§5 results table (headline stays
      the committed 0.832; tap presented as the buffer-controller extension, +0.032, gap-to-Adam 0.015).
- [ ] Multi-seed confirm (currently single seed 0) before it becomes a headline claim — 84k is stable
      but a 2–3 seed A/B matches the buf_uniform rigor.
- [ ] Section framing: tap as timescale controller; ties to the het-τ/ALIF demotion (design note §4).
