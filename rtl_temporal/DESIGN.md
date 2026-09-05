# Sky130A_16x16_4cell_GSCtemporal — temporal-credit RTL (Paper III)

**Purpose.** Hardware-credible RTL for the *temporal* forwards-only PCN of Paper III
(`paper/main_stage3_GSC_v1.md`): the learning that `pcn_learner_snn.py` validates only in
the maths sim. Kept as a **clean, separable** tree (copied the jug base + harness) so temporal
work does not entangle the static jug chip (`Sky130A_16x16_4cell_jug/`).

**▶ Master index + remaining checklist: `../neurobench_gsc/PAPER3_STATUS.md`.** This DESIGN.md is the
RTL resume anchor; the STATUS hub ties the paper, the sim, and this build together.
**▶ Added-complexity / router-fit review: `COMPLEXITY_AUDIT.md`** (state per cell, per-synapse costs,
the 2 router-fit refinements, share/LUT compute notes, open questions).
**▶ Per-timestep router bandwidth: `BANDWIDTH_ANALYSIS.md`** (refinement #2 — ×T absorbed, 2–3 orders
headroom, `frame_seq` fits N≈8 with margin; no redesign).

**Scope (user, 2026-08-20):** FULL integration — all temporal modules **plus the temporal
chip top** (the top-level integration is the real risk). Bit-faithful throughout.

## The one primitive
Everything is the leaky-accumulator + threshold/graded-write cell that already exists as
`jug_ctrl.v` (leak λ=1, threshold → ±1 weight). The temporal roles are configurations of it:

| module | role (Paper III §3) | leak | write | status |
|---|---|---|---|---|
| `lif_cell.v` | temporal coding (membrane) | α≈0.9 | spike (soft reset) | ✅ **P1 DONE — bit-exact** |
| `adjoint_window.v` | temporal credit ADJOINT δ=Σ(ψe) window | window n≈8 | — | ✅ **bit-exact — the credit core** |
| ~~`elig_buffer.v`~~ | (paper eq3: Σ δ·s window) | — | — | ⚠ SUPERSEDED — wrong gate position (see dataflow) |
| ~~`elig_carry.v`~~ | faithful leaky adjoint | α(1−θψ) | — | ❌ NOT NEEDED — uniform ≈ faithful in sim |
| `jug_ctrl.v` (from base) | weight consolidation (fold) | λ=1 | ±1 sign write | ✅ (static, reused) |
| `graded_write.v` | reliability-graded write (var_norm) | β₂≈0.99 | m/√v | ✅ **P3 DONE — bit-exact** |
| `readout_grad.v` | cold on-chip read-out | — | Sᵀ·feat MAC | ✅ **P4 DONE — bit-exact** |
| `pcn_temporal_top.v` | the temporal CHIP (single-lane datapath) | — | — | ✅ **P5 DONE — bit-exact** |
| `pcn_temporal_tile.v` | the temporal chip at WIDTH (NO×NI tile) | — | — | ✅ **P6a DONE — bit-exact** |
| `transpose_route.v` | Wᵀ hop: route adjoint DOWN a layer | — | — | ✅ **P6b DONE — bit-exact** |
| `route_quant.v` | quantise routed adjoint to 6-bit msg (router-fit #1) | — | — | ✅ **DONE — bit-exact** |
| `lif_cell.v` ψ-LUT (`psi_lut.hex`) | readiness via shared LUT, not a divider (#3) | — | — | ✅ **DONE — bit-exact** |
| `graded_sweep.v` | one shared √+÷ swept over synapses (#4) | — | — | ✅ **DONE — bit-exact** |
| `pcn_temporal_2layer.v` | DEEP forwards-only credit (2 layers) | — | — | ✅ **P6b DONE — bit-exact** |

Optional unify: refactor the leaky-accumulate core into one `leaky_mac_cell.v` that lif/jug/
graded instantiate — the paper's one-primitive thesis expressed in RTL.

## P5 — the temporal chip TOP: dataflow (the validated credit, from the sim)
Per SAMPLE (a T-step spike sequence), per layer `l` (top→down), per synapse o←i:
1. **Forward** (per t): analog MAC `W·s` → `lif_cell` → spike `s_o[t]` + readiness `ψ_o[t]`
   (the surrogate derivative). Read-out layer integrates a leaky feature `f`.
2. **Top error** `e_L[t]` from the read-out Jacobian (firmware soft-max forms `S`).
3. **Adjoint** `δ_o[t] = Σ_{k=0}^{N} (ψ_o·e_o)[t+k]` — `adjoint_window` on `a=ψ·e` (N-step
   FORWARD window ⇒ N-cycle learning latency; the FIFO delay).
4. **Weight credit** `E_oi += δ_o[t]·z_i[t]` — a GATED accumulate of the adjoint against the
   CURRENT presyn spike `z_i[t]` (delay z by N to align with δ). Reuses the `readout_grad`
   accumulate datapath (E += a·b, b = spike). Σ over t.
5. **Route down** `e_{l-1,i}[t] = Σ_o δ_o[t]·W_oi` — the existing `pcn_transpose.v` Wᵀ hop.
6. **Read-out** `E_ro += boss_lr·Sᵀ·feat`, `E_b_ro += boss_lr·ΣS` — `readout_grad`.
7. **Fold** (periodic): `E → graded_write → ±code` writes to W-SRAM (the jug fold path).
⚠ This CORRECTS the module decomposition: the credit is `adjoint(ψe) window` × `current presyn
spike`, NOT `elig_buffer`'s `Σ δ_{t-k}·s_{t-k}`. **Paper eq (3) must be fixed to this form.**
Modules still to build for the top: `lif_cell` +ψ output; `credit_accum` (gated E+=δ·z, = readout_grad
core); the sequencing FSM `pcn_temporal_top.v`; then `tb_temporal_e2e` vs a Python composition, then
vs the sim's fixed-point path (P6).

## ⏳ TO-DO (flagged 2026-08-20)
- **Correct paper eq (3)** in `main_stage3_GSC_v1.md` to the adjoint-window form above (+ note the
  reset-product ablation: uniform ≈ faithful).
- **Re-run the 84k headline** with the committed (uniform) buffer for consistency, once the top is built —
  confirm it still reaches ~0.831 so the paper's headline matches the actual hardware mechanism (user).

## Fixed-point conventions (the bit-faithful contract)
- **Membrane** `lif_cell`: signed `MEMW=20`; input current `INW=12` signed (post-ADC MAC);
  `α = ALPHA/2^ASHIFT = 230/256 ≈ 0.898` (the fixed-point image of the sim's 0.9 — a stated
  quantisation, like the static weight-cell curve); `THRESH` in membrane LSBs; positive-threshold
  LIF (membrane may go negative, spike on the up-crossing only), soft reset `mem -= THRESH`.
- **Signedness rule (bit us once):** keep every intermediate FULL-WIDTH and SIGNED; force
  `$signed()` on BOTH multiply operands; a part-select is unsigned and corrupts a negative value —
  truncate ONLY at the register assignment.
- Python `>>` == Verilog `>>>` (arithmetic floor for two's complement), so the integer reference
  and the RTL agree exactly.

## Verification methodology (double-build)
Each module: a fixed-point Python reference in `ref/<mod>_ref.py` computes the integer model AND
dumps two's-complement hex vectors; a `tb_<mod>.v` reads them via `$readmemh("../ref/…hex")`,
drives the DUT, and checks **bit-exact** with `!==` (x-safe). All TBs run under
`rtl/run_all_tb.sh` (PASS/FAIL; *silence is a FAIL*). Regenerate vectors with
`python3 ref/<mod>_ref.py` before running if params change.

Run: `cd rtl && bash run_all_tb.sh`  (or `… tb_lif_cell` to filter).

## The bit-faithful reference and the sim
`ref/*_ref.py` is the integer model the **temporal `pcn_learner_snn.py` must match** when run
bit-faithfully — the same double-build principle as the static side. A follow-up is to give
`pcn_learner_snn.py` a fixed-point path that dumps end-to-end vectors for `tb_…_e2e` (P5).

## Status log
- **2026-08-20 — P1 `lif_cell.v` bit-exact** (`tb_lif_cell`, 64-step trace incl. firing+residue,
  quiet leak decay, negative membrane, random). Caught + fixed a signed-multiply/part-select bug.
  Dir seeded from the jug base (`run_all_tb.sh`, `jug_ctrl.v`).
- **2026-08-20 — P2 `elig_buffer.v` bit-exact** (`tb_elig_buffer`, 48-step; window fill/drain, sign
  changes, sparse spikes). Sliding-window running sum over an N-deep FIFO of gated errors.
- **2026-08-20 — P3 `graded_write.v` bit-exact** (`tb_graded_write`, 40 folds; checks m, v AND step;
  consistent/noisy/negative-E phases). m/√v via floor integer-sqrt (`automatic` fn) + divide.
  Bugs caught+fixed: (a) self-determined 16-bit `$signed(E)*$signed(E)` truncating E² — widen before
  squaring; (b) out-of-range bit-select `CLIP[MW+F-1:0]` on a 32-bit param → x — size via a localparam.
- **2026-08-20 — P4 `readout_grad.v` bit-exact** (`tb_readout_grad`, 40-cycle acc/fold op stream, weight
  AND bias modes; checks E and dW incl. clamped folds). Outer-product accumulate + bootstrapped bounded write.
- **2026-08-20 — P5 STARTED: dataflow locked (above) + `adjoint_window.v` bit-exact** (`tb_adjoint_window`,
  48-step; burst/drain/sign-flip). The corrected credit core (δ=Σ(ψe) window). Superseded `elig_buffer.v`.
- **2026-08-20 — `lif_cell` +ψ readiness output** (`ψ=1/(1+SURR|v−THR|)²`, SURR=5, integer divide;
  bit-exact) — the readiness the credit path needs.
- **2026-08-20 — P5 CHIP TOP `pcn_temporal_top.v` bit-exact** (`tb_temporal_top`, 49-cycle: 40 steps +
  8 flush + fold). Single-lane integrated datapath composing lif(+ψ) → a=ψ·e → adjoint_window → z-delay
  (N-align) → credit accumulate (Σ δ·z) → graded_write fold → dW. Verified vs a CYCLE-ACCURATE Python
  model (`top_ref.py`). First-pass pass (the per-module debugging paid off). The **integration/timing is
  proven**; a full chip replicates this lane across the 16×16 array (swept controller like jug_ctrl) +
  the analog MAC + `pcn_transpose.v` Wᵀ route.
- **2026-08-20 — P6a ARRAY SCALE-OUT `pcn_temporal_tile.v` bit-exact** (`tb_temporal_tile`, NO=2×NI=2,
  41 cyc × 4 synapses). Replicates the lane across a tile: per-neuron lif+adjoint, SHARED per-presyn
  z-delay lines, per-synapse credit + graded write. Proves the array composition. (Resource SHARING —
  one swept comparator/divider time-multiplexed, à la jug_ctrl — is the area optimisation still to do.)
- **2026-08-20 — P6b DEEP CREDIT: `transpose_route.v` + `pcn_temporal_2layer.v` bit-exact.**
  `transpose_route` (`tb_transpose_route`, 24 vec): the Wᵀ hop `e_{l-1,i}=sat(Σ_o δ_o·W_oi >> RSH)`.
  `pcn_temporal_2layer` (`tb_temporal_2layer`, 49 cyc, cycle-accurate): the top error trains BOTH layers
  forwards-only — layer-2's adjoint δ2 updates W2 AND routes down via the transpose to become layer-1's
  error, driving δ1 and W1. δ1 lags δ2 by one pipeline cycle. **The core Paper-III claim in verified RTL.**
- **2026-08-20 — ROUTER-FIT #1: `route_quant.v` bit-exact** (`tb_route_quant`, 48-step). Quantises the
  routed adjoint to the interconnect's **6-bit** message width (`delta_bits`) via a causal running-peak
  scale (the AGC A operator) + fixed-level quantiser (Q) — the streaming-hardware analog of the sim's
  non-causal per-sequence-max quantiser. Message on the wire = 6-bit q; the wide scale stays local.
  Plugs in between `transpose_route` (wide MAC) and the router.
- **2026-08-20 — `route_quant` WIRED into `pcn_temporal_2layer.v`** (regression re-passes bit-exact): the
  routed error now goes transpose → 6-bit message → reconstruct (`q*scale/L`) → layer-1 adjoint. Both
  layers still train correctly through the quantised route. Router-fit #1 is complete end-to-end.
- **2026-08-20 — COMPUTE REFINEMENTS #3, #4 (bit-exact).** #3 **ψ→shared LUT**: `lif_cell` reads
  `psi_lut.hex` (512 entries, `|v−THR|>>3`) instead of the per-neuron `NUMER/den²` divider; all four
  ψ-using refs import the shared `psi_lut` so RTL↔ref stay locked (all vectors regenerated, tops re-pass).
  #4 **`graded_sweep.v`**: ONE shared √+÷ time-multiplexed over NS synapses (one/cycle, NS-cycle fold, FSM
  like `jug_ctrl`'s swept comparator) — bit-exact to NS parallel `graded_write`s.
- **2026-08-21 — E2E MATCH vs the validated sim DONE (`E2E_MATCH.md`).** `ref/e2e_ref.py` runs the
  committed recipe (uniform cell, briefly trained on real GSC) through the SIM's OWN credit path, then
  drives the SAME integer lane as the RTL (`top_ref.Top`) with the sim's real per-t quantities
  (in_cur=Zᵀ·W[o], e_err=d_o·g(t), z_i), quantised to the fixed-point contract. **Fidelity over 24k
  real lanes: sign agreement 93.6%, Pearson 0.992, Spearman 0.998** (sign flips concentrated at
  near-zero credit). ψ cosine 0.979 (α 0.9→230/256 + LUT); **credit cosine ψ-skew-vs-aligned 0.998 ⇒
  the 1-cycle ψ pipeline skew is IMMATERIAL — P6 skew note CLOSED by measurement, no de-skew needed.**
  Sweep credit self-checked bit-exact to `top_ref.Top`. New `tb_temporal_e2e.v` replays the extracted
  real-GSC lane (209 cyc) through the actual Verilog top, bit-exact.
- **Regression: 12/12 pass** (`cd rtl && bash run_all_tb.sh`) — adds `tb_temporal_e2e`.
- **P6 remaining (AREA / completeness only — correctness proven end-to-end):** swept/shared-resource
  controller (one comparator/divider time-multiplexed, à la jug_ctrl); wire `readout_grad` into a top.
  ψ-skew alignment: MEASURED immaterial (above) — not a refinement to make.
- **2026-08-20 — HW-SIMPLICITY WIN (`buf_uniform` A/B).** The validated sim credit (`_error_carry`) is a
  ψ-gated leaky backward adjoint; `elig_buffer.v` is the simpler UNIFORM n-window. A/B in the sim (30k/20ep
  leaky, winning recipe, s0): **faithful 0.7685 vs uniform 0.7633 (Δ 0.5pp, within the ~0.7pp band).** ⇒ the
  reset-product decay is NOT load-bearing at n=8 (the membrane leak already caps reach) ⇒ **`elig_buffer.v`
  is the credit cell; `elig_carry.v` dropped.** Paper eq3 (uniform window) stands as the real mechanism;
  add the reset-product ablation as supporting evidence. (Seeds 1–2 confirmation running.) Sim toggle =
  `pcn_learner_snn.py` `buf_uniform` (default OFF) / harness `--buf_uniform`.
- **House Verilog traps hit (all now in-code):** keep signed intermediates full-width; `$signed()` BOTH
  operands; cast-function args are self-determined (widen before squaring); no out-of-range param
  bit-selects; loop-bearing functions in continuous assigns must be `automatic`.
