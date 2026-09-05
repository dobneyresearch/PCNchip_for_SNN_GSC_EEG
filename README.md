# One Leaky-Accumulator Cell: Forwards-Only Temporal Credit Across Spiking Speech and EEG

**A generalist, forwards-only rule for temporal credit assignment in spiking predictive-coding
networks.**

Code, results, and a bit-faithful RTL implementation for **Part III** of this study — the temporal
(spiking) extension of the forwards-only PCN chip. The paper itself is in [`paper/`](paper/).

> **All results are pre-silicon.** The evidence here is Python behavioural simulation and a
> bit-faithful RTL implementation verified bit-exact against fixed-point reference models. Nothing
> has been fabricated.

---

## What this is

A single reusable primitive — a **leaky accumulator**, `x ← λx + u`, feeding a threshold or a bounded
graded write — assigns temporal credit **forwards-only**: no backward pass, no global unrolled
history, no host in the training loop. The same cell, at different leak time-constants, does spike
coding, temporal credit (a short *local* eligibility window, not backpropagation-through-time),
weight consolidation, reliability grading, and on-chip read-out.

The claim the repository is built to support is **robustness / generality**: one architecture,
tested on two temporal signals of opposite character, with no task-specific redesign.

- **Google Speech Commands (GSC)** — delta-modulated audio, sparse and event-like, deep LIF stack,
  leaky read-out.
- **THOR EEG motor imagery** — continuous band-power, shallow network, rate read-out.

The one setting that differs between them — a reliability-graded versus a 1-bit-sign weight write —
is not tuned but **predicted** by a falsifiable law (grading earns its keep only when temporal credit
is a noisy per-timestep sum), which the EEG task independently confirms.

## Headline results (identical-network comparison against full-BPTT / Adam)

| benchmark | forwards-only rule | full-BPTT (Adam) upper bound | fraction of BPTT |
|---|---|---|---|
| GSC, uniform buffer (84k) | **0.832** | 0.881 | ~94% |
| GSC, **+ learnable tap** (84k) | **0.866** | 0.881 | **~98%** |
| THOR EEG-MI (4 seeds) | **0.668** | 0.709 | ~94% |

The residual to BPTT is localised to the forward credit *direction*, not the write magnitude; the
learnable-delay ("tap") buffer is the shared mechanism that supplies temporal reach on both tasks.

---

## Repository structure

```
.
├── paper/          The paper, its LaTeX/Markdown sources, refs, and build script
├── gsc/            GSC temporal benchmark
│   ├── pcn_learner.py        Static forwards-only learner (shared base, Part II)
│   ├── pcn_learner_snn.py    Temporal/spiking learner — the validated module (single source of truth)
│   ├── gsc_temporal_harness.py   Entry point: --train {jug,adam}, --readout {leaky,rate}, --tap ...
│   └── paper3_runs/          Regeneration scripts (run_*.sh), RESULTS_*.md, and the evidence logs
├── eeg/            THOR EEG-MI benchmark (imports the validated module from ../gsc)
│   ├── eeg_mi_harness.py     Data + forward-model harness
│   ├── probe_jug.py          Forwards-only rule (the 0.668 result)
│   ├── probe_tapweight.py    Adam K-sweep / tap window (K=8 vs 16)
│   ├── probe_adam_ksweep.py  BPTT/Adam comparator across window widths
│   ├── probe_bhdamp.py       The Adam-gap search (write-path damper; null, Sec. 8)
│   ├── probe_synops.py       NeuroBench Track-2 op/footprint counter (SynOps)
│   ├── probe_spike_cov.py    Spike-native covariance ablation (diagonal vs cross-terms)
│   ├── SIGNAL_LEAK_CLOSEOUT.md   Definitive EEG numbers + the gap analysis
│   ├── RESULTS_track2.md     Track-2 leaderboard comparison (SynOps + footprint)
│   └── FINDINGS_bhdamp.md    The write-path search record
└── rtl_temporal/   Bit-exact RTL (paper Sec. 5)
    ├── rtl/        Synthesisable Verilog modules + testbenches + run_all_tb.sh
    └── ref/        Fixed-point Python golden references + hex test vectors
```

`pcn_learner_snn.py` is the **single source of truth** for the learning rule; both benchmarks import
the same file (the EEG harness reaches it at `../gsc`). New behaviour is a default-OFF flag in that
module, never a per-dataset fork.

---

## Reproducing the results

Python 3 with PyTorch and NumPy; the RTL regression needs `iverilog`. The datasets are **not**
included (see [NOTICE.md](NOTICE.md)) — obtain GSC via the NeuroBench protocol and the THOR EEG data
from its challenge source, and point the harnesses at them as documented in each harness header.

**GSC (speech).** The committed uniform cell and the learnable-tap A/B:

```bash
cd gsc
bash paper3_runs/run_84k_tap.sh      # no-tap ≈ 0.832 (committed) ; +tap ≈ 0.866
# or directly:
python3 gsc_temporal_harness.py --train jug --readout leaky --n 84000 --eval_n 11005 \
    --epochs 20 --bs 128 --seed 0 --pl_buffer_n 8 --rr_cold --var_norm_fold \
    --rr_center --rr_lr_mult 100 --buf_uniform            # add: --tap --tap_K 16
```

The Adam / full-BPTT upper bound and the ablations behind the paper's tables are in
`gsc/paper3_runs/` — each `run_*.sh` regenerates a set of `*.log` files, and the `RESULTS_*.md`
notes interpret them.

**EEG (motor imagery).** The forwards-only rule and its Adam comparator:

```bash
cd eeg
python3 probe_jug.py       --shapes shallow --modes tap --readout rate --epochs 60 --seed 0   # ≈ 0.666
python3 probe_tapweight.py --modes learned16 --epochs 60 --lr 1e-3 --seed 0                    # Adam ≈ 0.710
```

`SIGNAL_LEAK_CLOSEOUT.md` records the multi-seed numbers and the exact commands.

The **NeuroBench Track-2** accounting — a shallow spike-rate net beats the reproduced official
baseline (0.682 vs 0.653) at 17.5× fewer ops and 47.8× smaller footprint — is regenerated by
`probe_synops.py`; `RESULTS_track2.md` tabulates it. (This is a property of MI's shallow band-power
structure, not part of the paper's forwards-only headline.)

```bash
python3 probe_synops.py     # Track-2 SynOps + footprint table
```

**RTL (bit-exact).** Every module is checked bit-for-bit against its fixed-point reference:

```bash
cd rtl_temporal/rtl
bash run_all_tb.sh          # pass/fail regression; silence counts as failure
```

---

## Licensing

Source-available, not open source — **noncommercial use is free; commercial use is evaluation-only
without a negotiated licence**. See [NOTICE.md](NOTICE.md) for the plain-language summary and
[LICENSE.md](LICENSE.md) / [LICENSE-COMMERCIAL-EVALUATION.md](LICENSE-COMMERCIAL-EVALUATION.md) for
the canonical PolyForm texts. The paper and the datasets are covered separately (see NOTICE).

This is Part III of a series: Part I presents the analog cell and its unsupervised learning; Part II
the forwards-only supervised architecture (the **W&E** design). Contact: `saul.dobney@dobney.com`.
