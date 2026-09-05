# End-to-end match: validated SIM credit ↔ fixed-point HW (2026-08-21)

The last open validation link. The per-module TBs prove **RTL == its integer reference**;
`top_ref.py` proves the lane **composes**; `tb_temporal_top` proves it on **synthetic random**
vectors. What was still unchecked: does that fixed-point datapath reproduce what the *validated
maths sim* (`pcn_learner_snn.py`) computes for the temporal credit — on **real GSC data**?

Rig: `ref/e2e_ref.py` (+ `tb_temporal_e2e.v`). Committed winning recipe with the committed UNIFORM
cell (`--readout leaky --pl_buffer_n 8 --rr_cold --var_norm_fold --rr_center --rr_lr_mult 100`,
`buf_uniform=True`), briefly trained on GSC (4k samples, 4 ep → val 0.49; enough for realistic
spike/error distributions), then one fresh batch run through the **sim's own** credit path
(`_error_carry`, `_readout_jacobian`, the top-hidden-layer accumulate).

## What is compared
For the TOP hidden layer, per synapse lane (neuron o ← input i) the HW consumes exactly:
`in_cur_o[t] = Zᵀ·W[o]` (membrane current), `e_err_o[t] = d_o·g(t)` (routed top error), `z_i[t]`
(presyn spike). These are quantised to the RTL contract (**1.0 == THRESH=1024 LSB; INW=EEW=12**;
`e_err` scale from the batch's |e| p99.5) and run through the **same integer lane as the RTL**
(`top_ref.Top`, which recomputes membrane + ψ via the shared LUT at **α = 230/256**). The result
`E_int` is compared against the sim's float credit `E_float = Σ_t δ_o[t]·z_i[t]`.

The sweep credit is **bit-exact to `top_ref.Top`** (self-checked: the factored `delta_prev·z_del`
== `Top.step` on the dumped lane, −43864) so the fidelity numbers are against the true RTL, not an
approximation.

## Result (8 samples × 64 neurons × 64 inputs = 24,018 nonzero-credit lanes)
| metric | value | what it measures |
|---|---|---|
| **Sign agreement** (HW vs sim) | **93.6 %** | do HW and sim push the weight the same way |
| **Pearson corr** (credit magnitude) | **0.992** | does the HW credit track the sim credit |
| **Spearman rank corr** | **0.998** | monotone fidelity (robust to the scale) |
| ψ cosine (fixed LUT vs sim ψ) | 0.979 | the α 0.9→230/256 + LUT quantisation |
| credit cosine (ψ-skew vs aligned) | **0.998** | the DESIGN.md 1-cycle ψ pipeline skew |

- **The committed fixed-point cell reproduces the sim's learning signal.** Credit magnitude
  correlates 0.992 / rank 0.998; the direction agrees on 94 % of lanes. The sign disagreements are
  concentrated at **near-zero credit** (|E_float|→0), where quantisation flips a value that carries
  almost no learning — the high magnitude correlation confirms the substantive credit is preserved.
- **The 1-cycle ψ pipeline skew is immaterial** (credit cosine 0.998). DESIGN.md flagged
  "matching the sim's exact per-step ψ·e alignment" as a P6 refinement; measured, the registered-ψ
  RTL and the sim-aligned datapath give credit that differs by <0.2 %. **P6 skew note closed by
  measurement** — no de-skew needed.
- **The α and LUT quantisation is the larger (still small) effect** (ψ cosine 0.979): α = 230/256 ≈
  0.898 vs the sim's 0.9, plus the 512-entry ψ LUT. It does not break the credit fidelity.

## RTL confirmation
`tb_temporal_e2e.v` replays the extracted real-GSC lane (sample 0, neuron 2, input 54; 209 cycles =
200 real + 8 flush + fold) through the actual `pcn_temporal_top` Verilog and checks `E_credit`/`dW`
bit-exact each cycle vs the `top_ref.Top` expected (HW E = −43864, dW = −1024). **PASS.** Regression
now **12/12** (`cd rtl && bash run_all_tb.sh`).

## Bottom line
The chain is closed end-to-end: **validated float sim → fixed-point contract → integer reference →
RTL**, on real GSC data. The committed uniform temporal cell, in silicon-faithful fixed point,
reproduces the sim's forwards-only temporal credit (corr 0.99, sign 94 %), and the two quantisation
choices (α image, ψ skew) are quantified and small. Reproduce: `python3 ref/e2e_ref.py` then
`cd rtl && bash run_all_tb.sh tb_temporal_e2e`.
