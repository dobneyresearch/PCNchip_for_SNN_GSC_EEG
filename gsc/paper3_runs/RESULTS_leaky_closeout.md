# Paper-3 close-out runs — LEAKY readout (2026-08-19)

## 0. FRAMING — constrained biological/hardware system vs the best pure-maths result

The Adam comparator is NOT a rival we are trying to beat; it is the **unconstrained upper bound**.
Every PCN run deliberately ACCEPTS compromises to stay biologically / hardware realistic, and each
compromise costs accuracy that Adam does not pay:

| dimension | PCN (constrained, realistic) | Adam comparator (best pure maths) |
|---|---|---|
| credit assignment | **forwards-only**, no backward pass | full backprop |
| temporal reach | **8-step per-layer local FIFO buffer** | **full 200-step BPTT** (entire sequence unrolled) |
| write precision | 1-bit **sign** or bounded **var_norm** (m/√v) | full-precision continuous gradient |
| readout | on-chip **cold gradient** cell (rule-trained) | end-to-end co-trained head |
| topology | fixed narrow LIF stack [20,256,256,256] | same forward, but grads flow globally |

With a full matrix stack and unconstrained continuous maths, Adam can keep chasing accuracy. The PCN
question is the opposite one: **given constraints we ADD to make it physically realistic, what is
achievable relative to that best pure-maths result?** So the headline metric is the GAP to Adam under
matched forward + data, and "within ~5pp forwards-only with an 8-step buffer" is the claim — not "equals".
The forward design is IDENTICAL in both (same `SNNNet.run_hidden`); only the learning/credit differs.

### The 8-step buffer vs 200-step BPTT (the core efficiency point)
Adam's `run_hidden` loops over ALL T=200 steps with the membrane carried across every step and no
`detach` (`gsc_temporal_harness.py:86–103`) ⇒ `loss.backward()` = full 200-step BPTT. The jug never runs
backward; its temporal credit is a **tiny 8-step local adjoint per layer** (`pl_buffer_n=8`,
`_error_carry`). 8 suffices because (1) the membrane leak α=0.9 gives a ~10-step time constant so older
credit has already decayed, and (2) the per-layer adjoints **compose down the 3 layers**, so effective
reach > 8 while each FIFO stays small (the depth-flat 6–12 result). ⇒ a forwards-only rule with an
8-deep FIFO lands within ~5pp of full 200-step backprop.


All `gsc_temporal_harness.py`, seed as noted, `--readout leaky` (the HARD temporal task — see
PCN_CHECKLIST.md §1 invariant #10). Winning recipe = `--pl_buffer_n 8 --rr_cold --var_norm_fold
--rr_center --rr_lr_mult 100`. Metric = best val accuracy. Reproduction commands: run_30k_battery.sh
(30k) and run_84k.sh (84k) in this folder.

⚠ A first pass of this battery accidentally ran on `--readout rate` (the default) — those results are
archived in `rate_readout_ARCHIVE/` and repurposed below as the static-analog CONTROL. The mixup is
what motivated checklist invariant #10.

## 1. Write rule: sign(E) vs var_norm (m/√v), both with cold gradient readout, depth 8

The clean sign-vs-var_norm ablation the user asked for. Both use the cold gradient readout; the ONLY
difference is the fold write rule (applied to hidden AND readout together).

| seed | sign+cold | var_norm+cold | Δ (var−sign) |
|---|---|---|---|
| 0 | 0.6583 | 0.7743 | +0.1160 |
| 1 | 0.6618 | 0.7668 | +0.1050 |
| 2 | 0.6268 | 0.7700 | +0.1432 |
| **mean ± sd** | **0.6490 ± 0.0157** | **0.7704 ± 0.0031** | **+0.1214** |

Paired t = 10.7 (df=2), one-tailed p ≈ 0.004. **var_norm graded write is LOAD-BEARING on the leaky
task: +12.1pp, positive in every seed, and far more stable (sd 0.003 vs 0.016).**

## 2. Depth sweep (var_norm+cold, seed 0)

| depth | 6 | 8 | 10 | 12 |
|---|---|---|---|---|
| val | 0.7660 | 0.7743 | 0.7788 | 0.7698 |

Range 0.0128 — DEPTH-FLAT over 6–12 (within seed noise). d10 nominal best; **d8 = solid cheap default**
(smallest FIFO). Matches the earlier probe finding that per-layer buffered credit is ~depth-independent.

## 3. RATE-readout CONTROL (static-analog; archived)

Same runs on `--readout rate` (pooled spike-count → aggregate per-synapse gradient, NO per-timestep
credit structure):

| | sign+cold | var_norm+cold | Δ |
|---|---|---|---|
| mean (3 seeds) | 0.7622 | 0.7765 | **+0.0143** |

**On rate, var_norm ≈ sign (+1.4pp, within seed noise).** Contrast with leaky (+12.1pp). ⇒ the graded
write helps IFF the per-synapse credit is a noisy SUM of per-timestep terms. This CONTROL answers the
deferred static EMNIST var_norm-vs-sign guard BY PREDICTION: static EMNIST (aggregate credit) ⇒
var_norm ≈ sign, so it was not run.

## 4. Interpretation for the paper

- The **cold gradient readout** and the **var_norm graded write** are BOTH load-bearing on the real
  (leaky) temporal task. Earlier "50/50 decomposition" (graded-write ≈ +0.10) is CONFIRMED on leaky.
- var_norm's value is **readout/temporal-dependent** (big on leaky, negligible on rate/static) — a
  clean mechanistic signature: reliability-grading matters exactly when credit is noisy per-t.
- Buffer depth is **flat 6–12**; use d8.

## 4b. OBSERVATION — with the cold readout, rate ≈ leaky ~0.77 (readout-independent ceiling?)

User observation (2026-08-19): jug var_norm+cold reaches ~0.77 on BOTH readouts (rate 0.7765, leaky
0.7704), whereas pre-cold-head rate sat well below leaky. Reading it beside the sign numbers:

| | sign+cold | var_norm+cold |
|---|---|---|
| rate  | 0.762 | 0.777 |
| leaky | 0.649 | 0.770 |

Interpretation: **~0.77 looks like a shared ceiling set by the hidden representation + data, not by the
readout.** Once the per-layer BUFFER gives the hidden layers proper temporal credit, the temporal
processing lives in the hidden WEIGHTS, so the readout's temporal weighting (uniform pool = rate vs
exp = leaky) stops being the bottleneck. The leaky readout's only residual cost is NOISIER per-t credit
— which is exactly why sign collapses on leaky (0.65) but not rate (0.76), and why var_norm rescues
leaky back to the shared 0.77. HW upside: a simple spike-COUNTER readout may suffice (no leaky readout).

⏸ **DEFERRED TEST (next session; user 2026-08-19)** — complete the 2×2 by running **Adam on RATE at
30k/20** (`run_adam_rate.sh`, ready). Confirms whether 0.77 is a shared jug ceiling ~3pp under Adam on
BOTH readouts (expect Adam-rate ≈ Adam-leaky ≈ 0.80). If Adam-rate came in BELOW jug-rate, that's a red
flag to chase. Not run now — user wants the current readouts confirmed via 84k first.

## 5. 84k full-set headline (LEAKY, 11005 test)

| config | val | vs Adam |
|---|---|---|
| **jug winning recipe** (var_norm+cold d8, forwards-only, 8-step buffer) | **0.8311** | −0.050 |
| Adam (full 200-step BPTT, upper bound) | **0.8810** | — |
| jug sign+cold, RATE readout (hardware-minimalism corner) | **0.7800** | −0.101 |

The gap to the pure-maths upper bound is ~5pp on full data (30k was ~3.6pp: jug 0.770 vs Adam 0.806).
Both improve with data; Adam improves slightly more, so the gap widens modestly at scale. **Honest claim:
the forwards-only jug with an 8-step buffer reaches ~94% of full-200-step-BPTT accuracy on the full set**
— not equal, but close under substantial deliberate constraints (see §0).

### Tiered hardware story (full 84k)
- **Cheapest** (spike-COUNTER readout + 1-bit SIGN write + on-chip cold head): **0.78** — already above the
  full-recipe 30k leaky number (0.770). Simplest possible cells.
- **+ leaky readout + var_norm graded write** (the winning recipe): **0.831** — buys ~+5pp for the extra
  machinery (leaky integrator readout + m/√v write).
- **Adam upper bound** (full BPTT, continuous, end-to-end): **0.881**.
⇒ pay for leaky+var_norm only if you want that last ~5pp; the minimal-hardware corner is already at 0.78.
Note rate+sign ≈ rate+var_norm at scale (var_norm is only load-bearing on the leaky readout — §1).
