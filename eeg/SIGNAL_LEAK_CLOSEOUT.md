# EEG jug↔Adam gap — signal-leak close-out (2026-08-25)

**Verdict:** the jug↔Adam gap on EEG-MI is **~0.044** and decomposes as **~0.013 (time-buffer) +
~0.031 (forwards-only credit-direction residual)**. Every other candidate has been tested and
eliminated. The ~0.031 is **localized** (to the hidden-feature credit) but **not located to a
recoverable lever** — it is the priced-in cost of forwards-only 1-bit credit, not a bug or lost signal.

## Definitive numbers (all shallow 62→16, rate readout, tap K16, sign fold; each with its command)
| what | value | command |
|---|---|---|
| Adam ceiling (K16 rate+tap) | **0.7101** | `probe_tapweight.py --modes learned16 --epochs 60 --lr 1e-3 --seed 0` |
| Adam K-sweep (lr=1e-3, 60ep) | K8 0.7049 · K12 0.7095 · **K16 0.7095** · K24 0.7095 | `probe_adam_ksweep.py --lr 1e-3 --Ks 8,12,16,24 --epochs 60` |
| Jug winning (ep=60) | **0.6655** | `probe_jug.py --shapes shallow --modes tap --readout rate --epochs 60 --seed 0` |
| Jug winning (ep=120) | 0.6753 | `probe_jug.py --shapes shallow --modes tap --readout rate --epochs 120 --seed 0` |
| **gap (Adam − jug)** | **~0.044** | |

## ★ CLOSE-OUT MULTISEED (the definitive matched comparison, n=4, 2026-08-25)
Replaces the flawed `multiseed.log` (whose Adam rows were 40e/1e-3 or 60e/1e-4 — never clean 60e/1e-3).
All 60ep, rate+tap, Adam lr=1e-3, jug boss_lr=3e-4, shuffle seeded. `run_closeout_multiseed.sh`.

| model | K | mean ± std | min | max |
|---|---|---|---|---|
| Adam | 8 | 0.7054 ± 0.0045 | 0.7002 | 0.7118 |
| Adam | 16 | 0.7094 ± 0.0052 | 0.7020 | 0.7153 |
| Jug | 8 | 0.6373 ± 0.0046 | 0.6296 | 0.6418 |
| **Jug** | **16** | **0.6677 ± 0.0012** | 0.6667 | 0.6696 |

**gap K=8 = +0.068 · gap K=16 = +0.042** (jug = **94.1%** of Adam at K=16).

- **Adam is K-insensitive** (0.705 K8 → 0.709 K16); **the jug needs the tap window** (0.637 K8 → 0.668 K16,
  +0.030). The tap is the jug's buffer-controller — reach closes ~0.026 of the gap for the jug alone.
- **No instability.** At K=16 the jug has the TIGHTEST spread of all four cells (±0.0012, 4× tighter than
  jug K=8 and tighter than Adam K=16). The wider spread at K=8 was under-resourcing, not divergence — more
  reach both raises AND stabilizes the jug. (Within-run the 1-bit fold DITHERS ±0.04–0.05/epoch vs Adam's
  ±0.01–0.02, but between-seed the best-of-epochs converges to a stable point; noisy optimizer, stable outcome.)

## The "regression" that started this was a PROVENANCE bug, not a code change
Adam numbers came in ~0.02–0.10 low. Exhaustive audit found NO code/data/hardware/determinism cause.
Root cause = two stacked lost-command provenance failures:
1. **`--epochs 120` dropped** — reference runs were 120ep, re-runs used the 60ep default (~−0.015).
2. **`lr=1e-3` → default `lr=1e-4`** — the true 0.7182 config was lr=1e-3 (recovered from the deleted
   `probe_reconfirm.py`'s log via the transcript + runtime fingerprint 531s≈539s). lr=1e-4 undertrains
   and plateaus ~0.03 low regardless of epochs. The "0.718cfg" label on a lr=1e-4 run was simply wrong.

**At the correct lr=1e-3, ep=60 suffices** (ep=120 was only compensating the too-low lr). This is why
the `multiseed.log` gap looked wrong: its Adam rows were 40e/1e-3 (0.7025) or 60e/1e-4 (0.6198) — never
the clean 60e/1e-3. ⇒ new discipline: **log the exact command (all flags) + date + seed** (CHECKLIST #13).

## Tested and ELIMINATED (this is the value — we know where it ISN'T)
| candidate | verdict | evidence |
|---|---|---|
| readout / head co-adaptation | ✗ not it | frozen-head: best Adam head on frozen jug feats ≈ jug-native (Δ ±0.01), both ≪ Adam ⇒ residual is in FEATURES (`frozen_head.log`) |
| write precision (sign vs magnitude) | ✗ not it | sign ≥ magnitude on this task; jug replicates Adam well on coarse 1-bit data (var_norm only helps GSC) |
| g(t) shape (flat vs leaky) | ✗ NULL | **paired** flat−leaky Δ = **+0.003 ± 0.007** (spans 0; seed3 negative). Earlier "+0.018" = single-shuffle artifact (`b_gt_paired.log`) |
| delta_bits (δ quantization) | ✗ NULL | delta_bits=0 0.6620 vs =6 0.6655 (−0.0035, noise). Earlier "+0.007" = single-shuffle artifact |
| boss_lr (fold pulse rate) | ✗ optimal | sweep: 1e-4 0.672 / 3e-4 0.667 / 1e-3 0.622 / 3e-3 0.539 / 1e-2 0.520 — plateau at ≤3e-4, cliff above ⇒ 3e-4 correct, transfers from GSC |
| tap K | ✗ saturated | Adam K12=K16=K24=0.7095; jug tap learns (peak idx ~13/15) but caps same |
| epochs / lr provenance | ✓ was the APPARENT gap | see above — now fully explained |
| buffer depth >8, fold rule, classifier, ALIF/q, heterogeneous-τ, init, normalization, timesteps (250 not 200), loop bounds, seed/GPU determinism | ✗ | all separately checked earlier this session |

## Where it IS (localized, not fixable within the constraints)
The residual ~0.031 is in the **hidden-feature credit DIRECTION**: the forwards-only windowed adjoint
computes a gradient direction that ≈ BPTT but not exactly. It is **not** a coarseness problem — writing
the slightly-wrong direction more precisely (magnitude/finer bits) does not help; the *direction* is what
differs, by design (forwards-only, 8-step reach vs full BPTT). Closing it would require the exact things
the chip deliberately gives up (true BPTT credit / magnitude write, the latter 5×-refuted). So this is the
**accepted forwards-only tax**, consistent with the GSC diagnosis and the Paper-3 "~94% of BPTT" thesis —
here the jug is 0.665/0.710 ≈ **94%** of the Adam ceiling.

## Engineering-validation spin-off (rolodex substrate)
The boss_lr sweep is a **positive** result for the printed/lithographed A4-sheet ("rolodex") design:
- **Task-transferable:** ~3e-4 optimum across GSC (temporal), EMNIST (static), EEG (band-power) ⇒ boss_lr
  can be a **single hard-wired global constant** for the whole stack — no per-sheet / per-task trimming
  (which a cheap printed substrate can't do well).
- **Broad plateau (1e-4→3e-4, ~3×) then graceful decay:** the tolerance headroom a wide-variation substrate
  needs — sheet-to-sheet variation in the realized boss_lr lands on the flat and the result holds; failure
  above the cliff is a gradual accuracy decay (diagnosable), not silent. See `GENERALISABLE_CELL_DESIGN.md`.

## g(t) design principle (kept, corrected)
g(t) is **auto-matched to the readout for CORRECTNESS** (rate→flat, leaky→leaky = the right ∂feat/∂s_t),
not for benefit (null on EEG). The manual-override escape hatch (`flat_gt`) is **REMOVED** from the module;
`readout_form` (auto-stamped by make_forward_fn) is the sole selector ⇒ a g(t) mismatch **cannot** occur in
production. Retired diagnostics: `probe_gt_rate.py`, `probe_regcheck.py` (both had the stamp-override flaw +
unseeded shuffle); the valid A/B is `probe_gt_paired.py` (leaky_wrap + seeded shuffle).
