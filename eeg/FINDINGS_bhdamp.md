# boss_h_damp — findings (2026-08-26, EEG jug) ⚗ EXPERIMENTAL

## ⛔ VERDICT (n=3, BOTH variants complete — supersedes the n=1 section below): NULL on best, HURTS robustness
Full sweep 2026-08-26 (`logs/bhdamp_3seed_20260826-125333.log`), shallow rate+tap K16, sign, boss_lr 3e-4, 60ep, paired:

| variant | best off→on (Δ) | between-seed spread off→on | late_std Δ | per-seed best Δ |
|---|---|---|---|---|
| **/8** floor .125 | 0.6676→0.6686 (**+0.0010**) | ±0.0014 → **±0.0130** | −0.0010 | +0.0197 / −0.0139 / −0.0029 |
| **/4** floor .25  | 0.6676→0.6678 (**+0.0002**) | ±0.0014 → **±0.0156** | −0.0001 | 0.0000 / −0.0203 / +0.0208 |

- **best NULL** on both (both span 0). **Between-seed spread INFLATED ~10×** — the per-seed deltas are pure
  basin-scramble (/4: seed1 −0.020, seed2 +0.021). Marginal /8 late_std smoothing (−0.0010) vanished at /4.
- The seed0 +0.020 that started this did NOT replicate. A single-seed *paired* A/B still misled because the
  damper's *effect itself* is seed-dependent — a paired shuffle controls shuffle noise, NOT basin-selection
  noise. [[feedback_log_exact_run_provenance]]

## Why this axis is CLOSED (the var_norm upper-bound argument — the real reason, task-independent)
Damping the sign step is a *weaker* member of the write-rule family than var_norm (per-synapse bounded
magnitude m/√v = the strongest bounded write). On GSC-leaky var_norm helps the jug a lot (+0.121 vs sign)
but **still leaves the same ~0.05 to Adam**. If the STRONGEST bounded write-rule refinement can't close the
ceiling, a damped sign certainly can't. ⇒ the jug↔Adam gap is **provably not in the write rule / step size /
oscillation** — it is the forwards-only credit **DIRECTION** (windowed adjoint: route Wᵀ + presyn spike,
~8-step reach) vs true BPTT. EEG (multiseed null) and GSC (var_norm tops out below Adam) meet at the same wall.
The write-rule axis (sign / var_norm / damp) is a **robustness / HW-cost** axis, NOT an accuracy-to-Adam axis.

**Disposition:** flag KEPT (DEFAULT OFF, validated path byte-identical when off) as tested-and-parked. NOT
promoted. This is a HOLD POINT — see the handoff note below; the gradient-direction gap is the open question
for a fresh (PhD-level) look, and it is bounded, not undiscovered.

## Handoff — the one open question, stated so the search need not be redone
**Everything measured/eliminated on the jug↔Adam gap:** readout/head co-adaptation (frozen-head), write
precision (sign≥magnitude), var_norm (helps leaky, null rate, never reaches Adam), g(t) shape (paired null),
delta_bits, boss_lr (3e-4 optimal, plateau+cliff), tap K, buffer depth >8, fold rule, classifier, ALIF/q,
heterogeneous-τ, init, normalization, timesteps, loop bounds, determinism, data integrity, **global step
damping (boss_h_damp, this doc)**. **The one thing standing:** the forwards-only windowed-adjoint credit
computes a gradient *direction* that ≈ BPTT but not exactly, and closing it needs a better *direction*
(reach / weighting / routing of the adjoint), not a better *magnitude*. That is the precise, isolated
question to hand to a theorist. Paper 3 banks the ~0.05 as the forwards-only tax; this reasoning justifies it.

---
## (superseded) first look — n=1

**Mechanism (built into validated `pcn_learner_snn.py`, DEFAULT OFF):** multiply the 1-bit sign-fold
step by a GLOBAL scalar `damp = clamp(mean_bh / div, floor, 1)`, where `mean_bh` = aggregate boss_h
severity over the fold window (correct samples ⇒ bh=0, so it tracks the error rate). Anneals the fixed
signSGD step as the model converges. Applied POST-sign (survives momentum/wclip), hidden W folds only.
Flags: `boss_h_damp`, `boss_h_damp_div` (8), `boss_h_damp_floor` (0.125). Probe: `probe_bhdamp.py`.

## Result — VARIANT A (/8, floor 0.125), shallow rate+tap K16, boss_lr 3e-4, 60ep, seed0, PAIRED
| metric | OFF | ON | Δ(on−off) |
|---|---|---|---|
| **best** | 0.6667 | **0.6863** | **+0.0197** |
| final | 0.6493 | 0.6736 | +0.0243 |
| late_m (last 10) | 0.6549 | 0.6756 | +0.0208 |
| late_std | 0.0042 | 0.0043 | +0.0001 |

`damp_last = 0.219` (mean_bh≈1.75 at convergence ⇒ step annealed to ~0.22× near the top).

## Read — this BEAT my prediction (best moved, not just stability)
I predicted best-val would be ~flat (peak already caught by best-of-epochs) and the win, if any, would be
final-state reliability (final↑, late_std↓). Instead **best itself jumped +0.020** — the overshoot-return
dither was NOT benign: it was bouncing the trajectory OFF a better basin, costing ~0.02 accuracy. Damping
the step near the top let it settle deeper. late_std was already tiny (±0.004) and stayed flat, so the win
here is a genuinely better optimum, not a steadier landing.

**Gap impact (IF it holds under multiseed):** jug↔Adam gap was 0.042 (jug 0.6677 / Adam 0.7094). A +0.020
jug gain → gap ~0.023, i.e. jug ≈ 97% of Adam (from ~94%). That would be the FIRST lever to move best-val
toward Adam after the entire signal-leak elimination table — potentially material for Paper 3.

## ⚠ CAVEATS — do not over-claim
- **n=1.** Single seed. It IS a *paired* A/B on an identical seeded shuffle (the reliable kind), so the
  +0.020 is not shuffle noise — but whether seed0 is representative is unconfirmed. **NEEDS multiseed n=4.**
- The 8ep smoke showed /8 damps hard early (0.378) and hurt final at 8ep — yet at 60ep /8 WON. So the early
  climb-tax was either worth it or irrelevant by convergence. A warmup (damp only late) may not be needed
  after all — but test /4 (less early tax) to see if it does even better.

## PENDING (cut off by the power outage)
- **VARIANT B (/4, floor 0.25) ON** — did NOT complete. Its OFF ran and reproduced 0.6667 exactly
  (confirms the paired setup is deterministic). Re-run: `python3 probe_bhdamp.py --seeds 0 --div 4 --floor 0.25 --epochs 60`.
- **Multiseed n=4** on the winning variant (the confirmation that decides if this is real).
- Consider: does the same damper help GSC-leaky and EMNIST-static? (transfer = the rolodex story). It is
  ONE global scalar per fold = "one gain on the error bus" — HW-cheap, keeps the 1-bit direction write.

## NOT YET DONE
- Not written into any checklist/memory as validated — it is n=1 experimental. Promote only after multiseed.
- Default stays OFF; validated sign path is byte-identical when off (damp=1.0).
