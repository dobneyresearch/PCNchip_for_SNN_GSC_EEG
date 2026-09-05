# neurobench_thor — THOR EEG-MI, a THIRD benchmark for the PCN module

Clean, separable folder for the NeuroBench **THOR EEG motor-imagery** challenge (past the
2026 deadline; we run it as a third generalization datapoint after static EMNIST and audio
GSC). The point: show the ONE forwards-only rule + one leaky-accumulator primitive transfers
to a genuinely different signal (EEG), and get a jug-vs-Adam gap on it.

**▶ Resume anchor = this file.** Harness = `eeg_mi_harness.py`. Data = `data/*.npy`.

## Isolation, without forking the module
The validated learner (`pcn_learner_snn.py`) is **imported from `../neurobench_gsc`** — single
source of truth, NOT copied here. Copying it would fork the validated module and let it drift
([[feedback_pcn_module_first]]). Only NEW EEG code lives in this folder; any module change EEG
ever needs goes into the shared module as a **default-OFF flag** (as `buf_uniform` did), never a
local edit. That is the real "avoid contamination": the new harness is isolated, the asset is not
duplicated.

## The task / data (`ThorEEGMI`, NeuroBench 2.3.0)
- Preprocessed **Lee2019 / OpenBMI** motor imagery, 54 subjects, re-split for THOR. Four `.npy`
  files auto-downloaded from HuggingFace (`data/`, ~1.1 GB total).
- **train (7344, 250, 62), val (1728, 250, 62)** — only train/val are public ⇒ **val = our eval**.
- 2 balanced classes (**0 = right hand, 1 = left hand**); chance = 0.50.
- Input = continuous **8–30 Hz** EEG amplitudes, T=250 @ 100 Hz, 62 channels; zero-mean, std≈5.5,
  heavy-tailed (±500 outliers). Fed straight into the first FC ⇒ **analog-in, spiking-hidden** (our
  layer-0 `v=α·v+X@Wᵀ` handles it identically). We **z-score per channel** (train stats, both splits)
  by default so the operating point is sane; `--no_normalize` for raw.

## The architecture map (baseline ↔ ours)
| | NeuroBench EEG_SNN baseline | our SNNNet (shared jug/Adam forward) |
|---|---|---|
| hidden | fc1 62→256 +LIF, fc2 256→128 +LIF | dims=[62,256,128]: 2 LIF layers, feats=128 |
| output | fc3 128→2 +LIF, **count output spikes** | readout head 128→2 on pooled feats (rate/leaky) |
| neuron | LIF β=0.9, θ=1, fast_sigmoid(25) | α=0.9, θ=1, pseudo_deriv slope **SURR=25** (matched) |
| train | Adam+BPTT, lr 1e-4, bs 64, 120 ep | `--train adam` (same forward) / `--train jug` |

The one structural difference: the baseline's 3rd layer is a spiking 2-neuron output (spike-count
readout); ours is a linear head on the pooled 128-d feature. So we run three modes:
- `--train official` — the EXACT baseline net, to reproduce the published reference number.
- `--train adam`     — our shared SNNNet + Adam (the FAIR drop-in comparator for the jug).
- `--train jug`      — the forwards-only PCN module, identical forward/data.

## Rules (read the official PDF, 2026-08-21 — `THOR_Neurobench_Challenge_2026_V1.0`)
- **Task**: 2-class MI (left/right hand), OpenBMI 54 subjects, 2 sessions, 100 trials/class/session,
  raw 1000 Hz 62-ch. Cross-subject generalization is the point (incl. "BCI illiteracy" subjects).
- **Track 1 (Absolute Accuracy)**: preprocessing UNRESTRICTED — custom spike encodings, spatial
  filters (CSP), any signal processing allowed before the SNN. Highest acc on unseen test subjects.
- **Track 2 (Compute Efficiency)**: accuracy COMBINED with SynOps/sample + footprint; MUST use the
  standardized pipeline = raw 62-ch → **continuous dense float tensors** (== our `ThorEEGMI` data).
  Our first-LIF-layer "Direct Input Coding" is the sanctioned encoding ⇒ **we are Track-2-compliant
  by construction.** Input `[62 × T]`, T = chosen binning.
- **Model**: torch SNN, scored by the NeuroBench harness (SynOps + footprint). **Accuracy is a GATING
  requirement; efficiency decides Track 2.** Training method UNRESTRICTED (Adam/BPTT fine — the
  challenge scores the INFERENCE model, NOT the learning rule).
- **Eval**: final ranking on a HIDDEN test set / newly-generated samples; our public split is
  train/val ⇒ val = our eval. Multi-phase CV for cross-subject generalization.
- ⚠ **Strategic**: forwards-only on-chip learning is NOT a scored feature. Its value here = (1) a
  THIRD paper benchmark (does the one rule reach competitive acc on EEG?), (2) the Track-2
  footprint/SynOps story (our leaky-jug HW / no-per-synapse-state / int8 work is on-metric).
- Past deadline (Phase-1 closed 24 Jul 2026) ⇒ this is a research run, not a submission.

## Metrics (challenge)
- **Track 1**: classification accuracy (val).
- **Track 2**: SynOps + footprint — a property of the trained forward net's spike sparsity (same
  for jug or Adam), where our HW-credibility case (no per-synapse digital state, int8 footprint,
  verified leaky-jug RTL) lands. Scored via NeuroBench's `Benchmark` (to wire in later).

## Status (2026-08-21 — scaffold verified, runs end-to-end)
- ✅ Data downloaded + shapes confirmed; both forwards, the jug credit path, the Adam comparator,
  and the official baseline path all execute on GPU (~16 s/epoch).
- Smoke test (not tuned): adam-rate val 0.546 @3ep (climbing from 0.50); jug leaky+uniform+var_norm
  +cold val **0.591 @2ep**. Early, different readouts — NOT yet a comparison.

## ⚠ The open research question (why sweep, don't assume the GSC recipe)
EEG-MI discriminates **mu/beta band POWER** over motor cortex (event-related desync) — a roughly
**epoch-static** quantity — NOT delta-modulated spike *timing* like GSC. So:
- the **temporal-credit** advantage (pl_buffer_n) may be muted; the leaky vs rate readout question
  is genuinely open (leaky/power-integrating may suit this better; GSC trap T3 likely does NOT apply).
- absolute accuracy is **subject-variability bound** (cross-subject 2-class MI ~0.6–0.8; classical
  CSP/Riemannian are strong). The interesting number is the **jug-vs-Adam gap**, not the ceiling.

## NEXT (comparator-first)
1. **Full baselines**: `--train official` (reproduce the published number) and `--train adam`
   (rate AND leaky), ~120 ep. Establishes the target.
2. **Jug drop-in** vs that Adam, same readout; then sweep the EEG-relevant knobs (readout rate/leaky,
   pl_buffer_n incl. 0, var_norm vs sign, boss_lr, normalize on/off).
3. Only then the Track-2 SynOps/footprint story + NeuroBench `Benchmark` wiring.

Run: `python3 eeg_mi_harness.py --train adam --readout rate --epochs 120`
     `python3 eeg_mi_harness.py --train jug --readout leaky --pl_buffer_n 8 --rr_cold --var_norm_fold --rr_center --rr_lr_mult 100 --buf_uniform --epochs 120`
