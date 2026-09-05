# Temporal RTL — added-complexity audit (2026-08-20)

What the temporal upgrade adds over the static jug chip, in state / signals / widths, and whether it
is sensible-scope and fits the router. Numbers per **cell** = 16×16 = 256 synapses, 16 in / 16 out
neurons. Widths from the RTL params; router/interconnect widths from the sim (`delta_bits=6`, 8-bit W).

## Added state
| new state | scope | width | per cell | verdict |
|---|---|---|---|---|
| membrane | **per-neuron** (16) | MEMW=20 | ~0.3 Kb | ✅ cheap; over-provisioned (~14 b needed) |
| adjoint window (FIFO + acc) | **per-neuron** (16) | (N+1)×DW + AWACC = 9×16+22 = 166 | ~2.7 Kb | ✅ the "genuinely new state," small |
| z-delay (presyn spike) | per-input (16) | N×1 = 8 | 0.13 Kb | ✅ trivial |
| credit accumulator | per-synapse (256) | ECW=32 | — | ✅ **maps to the existing analog E-cap, not new SRAM** |
| **graded-write m, v** | **per-synapse** (256) | MW+VW = 24+32 = **56** | **~14 Kb** | ⚠ big — reintroduces per-synapse digital state (see below) |

**Base temporal cell (LIF + adjoint buffer + sign fold) adds only PER-NEURON state (~3 Kb/cell) and
KEEPS the jug's defining property — no per-synapse digital accumulator (Part II C3).** Only the graded
write breaks it.

## The graded-write cost — OPEN QUESTION / future work (user, 2026-08-20)
`var_norm` (m/√v) adds **56 bits/synapse ≈ 14 Kb/cell = 7× the 8-bit weight SRAM (2 Kb/cell)** — exactly
the per-synapse digital accumulator the leaky jug was built to avoid. It is **optional**: the ablation
(§5) shows sign-write alone = 0.78 and var_norm buys the last **+5pp** to 0.83. **The trade-off is known
and accepted; left as an OPEN QUESTION for future work:** is there a *middle-ground reliability signal* —
more information than a 1-bit sign, but far cheaper than full per-synapse m+v (e.g. a few-bit confidence,
a shared/blocked v, a sign-with-hysteresis) — that recovers most of the +5pp without the full accumulator?
The architecture↔accuracy trade-off is characterised, so this is a refinement, not a blocker.

## The read-out (W_f) head — checked, not a concern
- **Size**: `W_ro = 35×256 = 8,960` weights (~72 Kb at 8-bit) — ~14% of one hidden layer (256×256). New
  *on-chip* only because the cold gradient readout moves it off the host (the intended contribution).
- **Dataflow rate**: runs **per-SAMPLE, not per-timestep** — on the leaky-integrated feature (256-d,
  integrated over T≈200). Its whole cycle (feature gather → `W_ro` MAC → 35 scores → softmax → gradient
  `E_ro += e⊗feat` → top-δ `W_roᵀ·e` to the last hidden) happens once per sample ⇒ **1/T the hidden
  credit rate; outside the hot loop.**
- **Placement**: ~35 analog cells OR (cleaner, since it's per-sample + small) a plain **digital MAC**.
- **Gradient**: `E_ro` maps to the analog cap; `m_ro/v_ro` (if var_norm on the readout) is the *same*
  optional per-synapse trade-off as the hidden layers (sign-write avoids it).

## Router-fit flags (concrete refinements)
1. **Routed-adjoint width. ✅ DONE (`route_quant.v`, bit-exact).** The inter-cell message is now **6-bit**
   (matching `delta_bits`), quantised via a causal running-peak scale (AGC A) + fixed-level Q — the
   streaming analog of the sim's non-causal per-sequence-max quantiser. The wide scale stays local (not
   sent per message). ✅ **WIRED into `pcn_temporal_2layer.v`** (transpose → 6-bit message → reconstruct →
   layer-1 adjoint; regression re-passes bit-exact). Router-fit #1 complete end-to-end.
2. **Per-timestep credit traffic. ✅ ANALYSED (`BANDWIDTH_ANALYSIS.md`) — not a bottleneck.** δ routes
   per-timestep (×T≈200), but the ×T is absorbed by ms-scale timesteps: ~1–5 Mbit/s network-wide, <1% of
   one 100 MHz link (2–3 orders of magnitude headroom). Best-effort/commutative/fixed-divide carry δ
   unchanged (just more often); the 4-bit `frame_seq` (16 frames) covers the N≈8 in-flight window with 2×
   margin. Only a **deeper** temporal window would need a 5-bit `frame_seq`. No router redesign.

## Compute flags ("share / LUT it," never per-synapse)
3. **ψ readiness** = `NUMER/den²` per-neuron per-timestep → a small **LUT** indexed by \|v−θ\| (saturated),
   not a literal divider. ✅ **DONE** — `lif_cell` reads `psi_lut.hex` (512 entries, one shared ROM);
   regression bit-exact.
4. **graded-write √ + divide** → **one shared, swept** unit time-multiplexed across the array (as the jug
   shares its comparator), never replicated per-synapse. ✅ **DONE** — `graded_sweep.v` (one engine, NS
   synapses/fold); bit-exact to NS parallel graded_writes.

## Over-provisioned widths (tighten in a real design; verification widths here)
`MEMW=20`→~14, `AWACC=22`→~20, per-synapse `MW=24`/`VW=32` (tightening these directly cuts the 14 Kb).
`ECW=32` is the analog cap (moot). None are correctness issues.

## Accuracy headroom — future work (user, 2026-08-20)
The **PCN_snn vs full-Adam gap** (~0.05 at 84k; forwards-only 0.83 vs 200-step BPTT 0.88) is left as an
area for **future detailed exploration**, not a blocker — the constraint↔accuracy trade-off is known.

## Bottom line
Fits the router and the jug philosophy well. Base temporal cell preserves "no per-synapse digital state";
the graded write is the one genuine per-synapse cost (optional, +5pp — open question for a cheaper
middle-ground). Two easy router-fit fixes (6-bit routed adjoint; per-t bandwidth check) and two "share/LUT
it" compute notes. The read-out head is modest and per-sample (out of the hot loop).
