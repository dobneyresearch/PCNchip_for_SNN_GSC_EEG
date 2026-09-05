"""
pcn_learner_snn.py — TEMPORAL / SNN variant of the definitive PCN learner.

⚠ THIS IS A FORK for driving a spiking (LIF/ALIF) net on GSC-temporal. The CLEAN, VALIDATED
STATIC solution is pcn_learner.py — do NOT port these temporal settings back into it. The
static module's conclusions (auto_lr does the per-layer job; scaled-RMS not needed) hold for
STATIC wide-input data; they do NOT transfer to the SNN, where the input is narrow (20 S2S
channels) and delta-modulated, so auto_lr mis-scales the first layer badly (~70× η imbalance,
layer0 thrashes / deep layers freeze). This file therefore adds, as TEMPORAL/SNN-ONLY options:
  • forward_fn        — pluggable external forward, to drive the snntorch LIF trunk from self.W
  • auto_lr = False    — DEFAULT here: use a fixed boss_lr (the narrow input breaks auto_lr)
  • unit_rms_presyn    — Concept #1: unit-RMS the presyn activation + RMS-preserving backproj
                         (this is what the prior 44% rate rule used; it normalises the
                          pathological input scale that auto_lr cannot)
  • per_neuron_gain    — Concept #2: per-output-neuron adaptive step on the sign write
                         (running per-row |E|, mean-1 normalised). The "adjust θ per neuron"
                          knob; reserved for the b_eprop-style test. Default OFF.
Everything else is the SAME definitive rule as pcn_learner.py (fit_clf, form-A boss_h,
teach-on-correct, Wᵀ backproj + delta_bits, fold jug + sign_at_fold + momentum).

--- original static-module header (rule ingredients, unchanged) ---
A single, importable, tested class that reproduces the pcn_bigspec.py
`--backprop --bp_full --fold --sign_at_fold ...` recipe (the run that reaches ~82% on
EMNIST) as the reusable base for downstream benchmarks/competitions (GSC, EMNIST, ...).
See DEFINITIVE_ARCHITECTURE.md for the ingredient checklist this implements.

The ONE learning rule — every vital ingredient present by DEFAULT:
  1. fit_clf         standardized logistic (default) or lstsq classifier -> W_f, b_f
  2. boss_h top-δ    S = onehot − softmax(scores)  [logistic/logits]  OR
                     S = onehot − scores           [lstsq/p̂, err_mode='mse'], gated by:
                       - perceptron stop-on-correct (zero the correct samples)
                       - boss_h severity  sev = (bh+1)/4,  bh = min(int(7·gain·margin/span), 7)
                         (bh==0 -> small bh_leak·0.25 step, the beneficial-noise gate)
  3. backprojection  true Wᵀ hops with leaky-ReLU jacobian, delta_bits per-row quantization
  4. fold jug        E += η·(Dᵀ@F) accumulate (clip e_clip); every fold_every samples:
                       quant E to e_bits -> [sign_at_fold: E = sign(E)·boss_lr]
                       -> velocity momentum  v = μ·v + E; W += v; E = 0
  5. auto_lr         per-layer η from measured activity RMS (automatic parameterisation)

Reference (numpy, chip-tiled): ../multi_array_level3_BIGspec/pcn_bigspec.py
Benchmark path: identical topology trained with autograd + Adam, for a validated
second opinion (this is what tells us how far the rule is from backprop).

NOTE on faithfulness: this is the pcn_bigspec FOLD jug (accumulate→fold→momentum), NOT
the continuous-leak per-weight-threshold `SigmaDelta` used in train_ablate.py. They are
different mechanisms; this class is the definitive one.

═══════════════════════════════════════════════════════════════════════════════
TRAPS & CATCH POINTS  — read before changing anything, or re-deriving from memory.
Each of these HAS bitten us and silently produced wrong numbers. They are recorded
here (not just in scattered notes) so the module itself is the definitive reference.
═══════════════════════════════════════════════════════════════════════════════

T1. DEGENERATE-SOFTMAX  (the big estimation error; guarded in __init__).
    lstsq regresses onto one-hot => its outputs are P(class|a), rms~0.05 — PROBABILITIES,
    NOT logits. Feeding them to softmax makes onehot−softmax ≈ a CONSTANT (onehot−1/K)
    class template + an informative part that is ~K× smaller. The rule then degenerates
    into DFA-with-a-fixed-template and LOOKS like "underfitting" while actually not being
    taught. Valid pairings ONLY:  logistic+softmax (BEST, default)  |  lstsq+mse (chip).
    ⇒ Never pair a probability head with softmax. The guard refuses the two bad pairings.
    ⇒ Any "PCN underfits / gap = fitting strength" claim measured under lstsq+softmax is VOID.

T2. CLASS-SORTED CACHE.
    The GSC cache (cache/gsc_*.pt) and some feature dumps are SORTED BY LABEL. X[:N] is
    then the first ~5 classes only — every un-shuffled --subset diagnostic trained on a
    handful of classes and tested on all 35. ALWAYS shuffle (seeded permutation) before
    slicing. All pre-shuffle subset comparisons are RETRACTED.

T3. TIME-COLLAPSING ENCODER kills delta-modulated spikes.
    S2S / speech2spikes emits delta-modulated TERNARY spikes whose time-MEAN is ≈0. A
    'rate' or 'leaky' encoder that collapses the time axis destroys the signal — even Adam
    drops to chance, so it masquerades as a learning-rule failure. Use a TEMPORAL encoder
    (e.g. the 10-bin binned encoder in gap_ladder.py) that preserves within-window timing.
    ⇒ If Adam is also at chance, suspect the ENCODER, not the rule.

T4. sign_at_fold IS LOAD-BEARING (not a chip tax).
    Writing sign(E)·boss_lr at fold is the MAGNITUDE NORMALISER. Removing it (writing raw
    E magnitude) collapses training (~−20pp). It is not an optional quantisation; it is
    how the rule controls step size. sign-at-fold (sign-of-mean) ≫ sign-per-step
    (mean-of-sign) by ~10pp. Keep it.

T5. FOLD CADENCE (REVISED on the corrected/T1-fixed platform). Pre-T1 notes said "fold
    per batch, larger hurts −4..−8pp". That REVERSED once the teaching signal was valid:
    a LARGER fold_every slightly HELPS (+1..+2pp on GSC-12k, +1pp on EMNIST). Effect is
    small; default 128 is safe, not sacred. fold_mean decouples lr from batch size.

T6. TEACH ON CORRECT (stop_on_correct=False) is the DEFINITIVE recipe (--bp_full).
    Perceptron gate (True, teach wrong-only) plateaus early because correct samples stop
    contributing; CE keeps widening margins. Big historic impact. NOTE: its benefit is
    MASKED under T1 — on a constant template, teaching-on-correct does ~nothing — so only
    trust teach-on-correct comparisons on a VALID head/error pairing.

T7. CLASSIFIER REFIT CADENCE: refit ONCE PER EPOCH, then FROZEN within the epoch.
    More-frequent (mid-epoch / per-chunk) refresh is HARMFUL (−2.25 measured; "fresher
    references are worse, stable references win"). Do NOT add chunked refit to accelerate.

T8. auto_lr here recalibrates PER EPOCH (calibrate on a 2000-sample subset). pcn_bigspec
    calibrates ONCE PER STAGE (lr_base reset at stage start). We do NOT implement staging
    here — staging + stage_rollback were an anti-plateau hack SUPERSEDED by the leaky jug
    and dropped. Do not re-add them. "frozen-after-first" auto_lr is NOT a thing we use.

T9. lstsq-vs-logistic in-loop ranking is UNSETTLED post-T1-fix. An earlier "lstsq ≫
    logistic" result was measured with the degenerate softmax active and is not to be
    trusted; re-measure both under valid pairings before ranking heads.

T10. METRIC for a sign rule: cosine(update, true grad) is the WRONG metric (signing a
    heavy-tailed gradient always destroys cosine, torch too). Use per-coordinate SIGN
    AGREEMENT when auditing whether the update tracks the gradient.

T11. boss_h has TWO forms; this module implements form (A):
    (A) severity-gated softmax error S=onehot−softmax, gated by perceptron + severity —
        this is the --backprop --bp_full path that reaches ~82%. (USED HERE.)
    (B) local-settle sign(W_f[label]−W_f[pred]) — the older forwards-only settle direction.
    Don't conflate them; (A) is the definitive teaching signal.

Always compare PCN against benchmark_adam() on the SAME features/topology/encoder — it is
what caught T1 and T3, and it frames every gap number.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PCNConfig:
    # activation / signal
    leaky_alpha: float = 0.1          # leaky-ReLU slope for x<0
    delta: float = 0.3                # DELTA: top-δ amplitude
    # classifier + teaching-error mode  (MUST be a valid pairing — see guard in __init__)
    #   DEFINITIVE recipe (pcn_bigspec CLF_LOGISTIC=True since 2026-07-13):
    #     clf='logistic' + err_mode='softmax'  -> real LOGITS into softmax  [BEST, faithful]
    #   chip-cheap alternative:
    #     clf='lstsq'    + err_mode='mse'      -> S = onehot - scores (no exp, difference amp)
    #   BROKEN (refused): clf='lstsq' + 'softmax' = the 2026-07-13 degenerate-softmax bug
    #   (lstsq outputs are P(class|a) rms~0.05, NOT logits; softmaxing them makes the
    #    informative gradient ~K× smaller than a constant onehot-1/K class template).
    clf: str = "logistic"             # 'logistic' (faithful default) | 'lstsq'
    err_mode: str = "softmax"         # 'softmax' (needs logits) | 'mse' (needs lstsq p̂)
    clf_refit_every: int = 1          # refit once/epoch then FROZEN within epoch.
                                      # ⚠ T7: MORE-frequent refit is HARMFUL (−2.25); do
                                      # NOT lower this / add chunked refit to "accelerate".
    clf_C: float = 10.0               # logistic inverse-reg
    # boss_h teaching-signal gate
    # DEFINITIVE recipe uses --bp_full => stop_on_correct=False: TEACH ON CORRECT samples too
    # (literal CE backprop, "what the torch rig does"). True = perceptron gate (a TESTED VARIANT,
    # not the default). Do NOT set this True and call it definitive.
    stop_on_correct: bool = False
    bh_gate: bool = True              # apply boss_h severity gate
    bh_leak: float = 0.25             # small step when bh quantises to 0
    boss_h_max: int = 7               # 3-bit severity ceiling
    bh_gain: float = 1.0
    # boss_h convergence damper (self-annealing sign-write scale) — DEFAULT OFF ⚗
    boss_h_damp: bool = False         # multiply the sign-write step by a GLOBAL scalar in [floor,1]
                                      # derived from the AGGREGATE boss_h severity over the fold
                                      # window. Anneals the fixed sign step as the model converges
                                      # (fewer / less-confident errors ⇒ smaller step), damping the
                                      # signSGD overshoot-return dither near the optimum. Keeps the
                                      # 1-bit DIRECTION write — one shared gain on the error bus,
                                      # HW-cheap; NOT a per-synapse graded write (that's var_norm).
                                      # Revives the signal auto_lr wants but that sign() eats: it is
                                      # applied POST-sign, so momentum/wclip don't renormalise it.
                                      # Hidden W folds only (readout/tap untouched). See probe_bhdamp.py.
    boss_h_damp_div: float = 8.0      # damp = clamp(mean_bh / div, floor, 1). mean_bh∈[0,7] ⇒ /8 caps
                                      # at 0.875 (always damping); /4 lets confident-wrong batches reach
                                      # the full step (boost-on-worse). Start /8 (easy quick check).
    boss_h_damp_floor: float = 0.125  # step never anneals below this ⇒ direction stays alive (no
                                      # plateau freeze). Try 0.25 paired with div=4.
    # delta transport
    delta_bits: int = 6               # per-row quantisation of the transported δ (0 = off)
    # fold jug
    e_bits: int = 4                   # E integrator resolution (0 = off). NB: a REGULARISER,
                                      # dataset-dependent — near-free on GSC (−e_bits ≈ +0.4)
                                      # but LOAD-BEARING on EMNIST (−e_bits = −8pp). NOT the
                                      # universal "free" that the old GAP_DIAGNOSIS claimed.
    e_clip: float = 100.0             # E clip during accumulation
    sign_at_fold: bool = True         # write sign(E)·boss_lr at fold (1-bit write).
                                      # ⚠ T4: LOAD-BEARING — this is the magnitude
                                      # normaliser; False (raw-E write) collapses (~−20pp).
    stoch_fold: bool = False          # ⚠ DIAGNOSTIC ONLY (default OFF; NOT a validated recipe knob).
                                      # Replaces the 1-bit sign write with STOCHASTIC ROUNDING: write
                                      # ±boss_lr with prob p=clamp(|E|/(stoch_scale·mean|E|),0,1), else 0.
                                      # Unbiased in expectation ⇒ probes whether the sign fold's gap to
                                      # Adam is lost magnitude/sign-noise. Hidden weights only.
    stoch_scale: float = 1.0          # normaliser for stoch_fold write probability
    fold_momentum: float = 0.9        # velocity momentum μ
    fold_every: int = 128             # fold cadence in SAMPLES.
                                      # ⚠ T5 (revised): on the T1-fixed platform a LARGER
                                      # fold_every slightly HELPS (+1..2pp), not hurts.
    fold_mean: bool = True            # divide accumulation lr by fold_every (batch-decoupled)
    # steps
    auto_lr: bool = False             # ⚠ TEMPORAL/SNN: DEFAULT OFF (was True in the static
                                      # module). auto_lr's fanin-normalised per-layer η
                                      # mis-scales the narrow 20-ch SNN input (~70× imbalance)
                                      # ⇒ use a fixed boss_lr instead. Set True only for static.
    auto_frac: float = 2e-3           # target per-step output move fraction (dense-calibrated)
    boss_lr: float = 3e-3             # fixed accumulation lr AND sign-at-fold step. Higher than
                                      # the static 3e-4 default (SNN, auto_lr off) — SWEEP it.
    wclip: Optional[float] = None     # symmetric weight clip (None = off)
    # ── TEMPORAL / SNN-ONLY normalisation (see module header) ─────────────────────
    unit_rms_presyn: bool = True      # Concept #1: unit-RMS presyn activation before the
                                      # outer product + RMS-preserving on the backprojected
                                      # error. Normalises the pathological narrow/delta-mod
                                      # input scale that auto_lr can't. What the 44% rule used.
    per_neuron_gain: bool = False     # Concept #2: per-output-neuron adaptive step on the sign
                                      # write (running per-row |E|, normalised to mean 1). The
                                      # "adjust θ per neuron" knob; reserved for b_eprop test.
    per_neuron_beta: float = 0.99     # EMA decay for the per-neuron running |E|
    per_neuron_cap: float = 8.0       # clip the per-neuron step scale (avoid a few dominating)
    # ═══════════════════════ TEMPORAL DROP-IN RECIPE (2026-08-19) ═══════════════════════════
    # For a NEW temporal dataset, use the VALIDATED recipe below and IGNORE everything marked
    # "✗ DO NOT USE" further down (those are refuted dead-ends kept only for history/ablation).
    #
    #   WINNING RECIPE (forwards-only jug ≈ Adam on GSC; 30k/20 jug 0.777±0.009 vs Adam 0.806):
    #     temporal=True + a temporal forward_fn returning feats+Z+PSI, then:
    #       pl_buffer_n=8, rule_readout=True, rr_warm=False, rr_center=True, rr_lr_mult=100,
    #       var_norm_fold=True.
    #     harness: --pl_buffer_n 8 --rr_cold --var_norm_fold --rr_center --rr_lr_mult 100
    #
    #   THREE ORTHOGONAL TOGGLES (each independently on/off — no interaction tuning needed):
    #     (1) LIF + LOCAL BUFFER  = pl_buffer_n : temporal-credit buffer depth. 0 = OFF (falls back
    #         to ψ-gated per-t routing); 8 = ON. Depth is FLAT over 6–12 (all within seed noise) —
    #         use 8 (smallest solid FIFO). This is the single on/off switch for the temporal upgrade.
    #     (2) COLD GRADIENT READOUT = rr_warm=False (--rr_cold) : train the classifier on-chip by its
    #         own gradient (no logistic fit). THE key lever for matching Adam. Needs rr_lr_mult≈100
    #         to bootstrap from small init, and rr_center=True for the leaky feature's DC.
    #     (3) WRITE RULE = var_norm_fold : True = graded m/√v (winning), False = sign. MINOR: var_norm
    #         beats sign by only ~1.4pp (consistent across seeds but < the seed spread). Sign is a
    #         perfectly viable simplification. NB: the write rule applies to BOTH hidden AND readout.
    #   Everything else below defaults OFF and keeps the validated static baseline untouched.
    # ════════════════════════════════════════════════════════════════════════════════════════
    # ── TEMPORAL CREDIT (e-prop) — active only when temporal=True. See T12/T13 below. ──────
    temporal: bool = False            # drive the credit assignment over TIME (ε eligibility +
                                      # ψ-gated per-t routing), not the collapsed static path.
                                      # Requires a temporal forward_fn (returns feats+Z+PSI).
    elig_alpha: float = 0.9           # ε fast presyn-echo leak (= membrane β). ε_t = α ε_{t-1}+z
    elig_slow: float = 0.0            # ε slow leak (0=off; ALIF two-timescale, e.g. 0.99)
    per_t_route: bool = True          # ⚠ T12 LOAD-BEARING: route the per-TIMESTEP ψ-gated signal
                                      # down (broadcast top-δ over t, gate each Wᵀ hop by per-t ψ)
                                      # vs a single time-collapsed δ. per_t=True is what lifts the
                                      # clean fo rule from ~0.45 toward BPTT-level; False = the
                                      # cheap time-collapsed hop (caps low).
    psi_route: bool = True            # gate each Wᵀ down-hop by ψ (as true backprop does)
    # ✗ DO NOT USE — REFUTED (kept for history). g(t) near-flat → no lift; superseded by pl_buffer_n.
    readout_jacobian: bool = False    # B4: multiply the top learning signal by the readout's
                                      # TEMPORAL JACOBIAN g(t)=(1/T)Σ_{τ≥t}β^{τ−t} before broadcasting
                                      # over t — the first-order rate↔leaky term the jug otherwise
                                      # drops (uniform broadcast). g(t) derived from (readout_beta,T),
                                      # nothing tuned; mean-normalised so only the SHAPE acts (scale
                                      # is a sign_at_fold no-op). HW = one global gain on the error bus.
    readout_beta: float = 0.9         # β of the leaky-integrator readout (= membrane ALPHA); sets g(t)
    # ★ g(t) FORM MUST MATCH THE READOUT DESIGN (2026-08-24 — the recovered term-3 loss, +0.018 on rate).
    # The readout's temporal Jacobian ∂feat/∂s_t differs by readout type:
    #   • LEAKY readout → g(t) = (1−β^{T−t})/(1−β)   (early spikes weighted more; _readout_jacobian)
    #   • RATE  readout → g(t) = FLAT 1/T            (every step weighted equally)
    # Using the leaky g(t) on a rate pool is a WRONG derivative. NOTE (2026-08-25): the g(t) SHAPE is
    # empirically NULL on EEG (paired flat-vs-leaky Δ = +0.003 ± 0.007, spans zero; the earlier "+0.018"
    # was a single-shuffle artifact). We keep g(t) matched to the readout for CORRECTNESS (it is the right
    # ∂feat/∂s_t), not for benefit — and it may matter on a genuinely g(t)-sensitive readout.
    # `readout_form` is the SWITCH; it MUST equal the forward_fn's readout. To make that automatic and
    # UN-MISMATCHABLE, make_forward_fn STAMPS learner._readout_form = its own readout on every call, and
    # _readout_jacobian prefers that stamp over this cfg field — there is NO manual override, so only the
    # matched g(t) can run. Default "leaky" = validated GSC path (leaky readout, leaky g(t)) unchanged.
    readout_form: str = "leaky"       # ★ "leaky" | "rate" — selects g(t) form; auto-stamped to match readout
    # ✗ DO NOT USE — REFUTED (kept for history). Raw per-step δ[t] collapses (DC class-bias dominates).
    running_delta: bool = False       # L1 (PREDICTIVE_DELTA_DESIGN.md §4): use the leaky readout's
                                      # RUNNING per-step delta δ[t]=onehot−softmax(W_f·r[t]) instead of
                                      # ONE pooled δ broadcast flat over t. r[t] = the per-step running
                                      # readout membrane (cap['RO_t']). Turns time-CONSTANT credit (B1)
                                      # into time-VARYING causal credit — the leaky readout's temporal
                                      # signal the jug currently averages away (⇒ jug-leaky ≈ jug-rate).
                                      # Needs a temporal forward_fn that returns RO_t (leaky readout);
                                      # falls back to the broadcast δ when absent or rule_readout.
    # ✗ DO NOT USE — REFUTED (kept for history). Velocity δ[t]−a·δ[t−1] amplifies noise; no net gain.
    pred_lead: float = 0.0            # L2 (PREDICTIVE_DELTA_DESIGN.md §5,§10b): on the running δ[t],
                                      # use signal[t] = δ[t] − pred_lead·δ[t−1]. a=0 ⇒ L1 level (raw
                                      # per-step, collapses — DC class-bias dominates); a=1 ⇒ PURE
                                      # VELOCITY δ[t]−δ[t−1], which DIFFERENCES OUT the low-evidence DC
                                      # bias and keeps only the discriminative change (and provides the
                                      # anticipatory lead). Only active with running_delta. The velocity
                                      # term amplifies noise ⇒ a is a lead/lag dial (§5).
    # ✗ DO NOT USE — REFUTED (kept for history). SnAp gate-side carry too shallow; superseded by pl_buffer_n.
    influence_n: int = 0              # (b) FORWARD INFLUENCE trace (PREDICTIVE_DELTA_DESIGN.md §b;
                                      # SnAp-n / RTRL-truncated). Replace the readiness gate ψ_o[t]
                                      # with Ψ̃_o[t]=Σ_{k=0}^{n}(Π_{j=1}^{k}α(1−ψ[t+j]))·ψ[t+k]: credit o
                                      # at t for its readiness propagated FORWARD n steps, discounted by
                                      # membrane leak α and CUT by reset (1−ψ). Keeps the good terminal
                                      # error d_o, enriches the temporal weighting with the causal carry.
                                      # n=0 ⇒ the jug unchanged; uses only t..t+n ⇒ causal, n-lag,
                                      # deterministic (no UORO variance). n≈1/(1−α)=10 covers ~90%.
    # ✗ DO NOT USE — REFUTED (kept for history). Eligibility double-count; pl_buffer_n is the corrected version.
    error_carry_n: int = 0            # FORK (a) — the TRUE ERROR-SIDE BPTT carry (temporal_credit_maths.pdf
                                      # next-session fork (a)). The SnAp buffer (influence_n) put the
                                      # temporal carry on the GATE side (ψ→Ψ̃, carrying future READINESS ×
                                      # a constant error). This puts it on the ERROR side: build the true
                                      # unrolled adjoint δ_o[t]=Σ_{k=0}^{n}(Π_{j=0}^{k−1}α(1−θ·ψ[t+j]))·
                                      # ψ[t+k]·e[t+k], where e[t]=d_o·g(t) at the top (pooled top error ×
                                      # readout Jacobian) and e[t]=routed-adjoint below. Both ψ[t+k] AND
                                      # e[t+k] are BUFFER READS (learning-delay FIFO, t..t+n) — causal, O(n),
                                      # NOT a forward-time lookahead. The adjoint δ (not a gate×error) is
                                      # what multiplies eligibility ε AND what routes down through Wᵀ. This
                                      # is the one structurally-correct thing untried. Reset index j=0..k−1
                                      # (includes the CURRENT step's reset) — vs the buffer's j=1..k offset.
                                      # g(t) enters at the TOP only (readout_beta), applied unconditionally
                                      # in this path (near-flat ⇒ small, but it is the true term). n=0 ⇒ jug.
    carry_theta: float = 1.0          # θ in the reset factor α(1−θ·ψ) of the error carry. 1.0 = full reset
                                      # (matches the SnAp buffer's (1−ψ)); the LIF surrogate-BPTT value.
    pl_buffer_n: int = 0              # ★ PER-LAYER BUFFER (Q2, BUFFER_AS_DESIGN_TOOL.md §3) — the CORRECTED
                                      # temporal credit, validated == full BPTT at n≥T and ~depth-independent
                                      # (d≈8 matches BPTT; `probe_perlayer_buffer.py`). Each layer computes its
                                      # OWN local n-step adjoint δ_l = error_carry(e_l, ψ_l, n) and pairs it with
                                      # the DIRECT presyn SPIKE z_l (NOT the leaky eligibility ε — that
                                      # double-counts α; ✅ CONFIRMED 2026-08-24, ε HURTS −0.065/−0.076, see
                                      # `plb_elig`): ΔW_l = Σ_t δ_l[t]⊗z_l[t]. Then routes the adjoint
                                      # down the forwards-only Wᵀ hop: e_{l-1}=δ_l·W_l. Top error e_L=d_o·g(t)
                                      # (readout Jacobian). This is the "LIF + small local buffer" temporal
                                      # upgrade: n≈8 is a tiny per-tile FIFO. n=0 ⇒ off. ✅ RESOLVED
                                      # (2026-08-19): the buffered credit SURVIVES the fold (sign or
                                      # var_norm) and depth is FLAT over 6–12 (all within seed noise) ⇒
                                      # use 8. This is the LIF+buffer on/off toggle.
    buf_uniform: bool = False         # ⚗ HW-SIMPLICITY PROBE (2026-08-20) — default OFF. When True,
                                      # _error_carry drops the reset-product decay Π α(1−θψ) (sets it to 1),
                                      # so δ[t] = Σ_{k=0}^{n} ψ[t+k]·e[t+k] — a UNIFORM n-window of the gated
                                      # error (the simpler `elig_buffer.v` hardware) instead of the leaky
                                      # backward adjoint. A/B this vs the faithful carry to see whether the
                                      # cheaper cell recovers the accuracy. Does NOT change the validated
                                      # default (False = the exact surrogate-BPTT recurrence).
    # ✅ DOUBLE-COUNT CONFIRMED (2026-08-24) — default OFF. The pl_buffer credit pairs the adjoint δ_l
    # (α carried BACKWARD in _error_carry) with the RAW presyn spike z_l. Pairing it with the leaky
    # eligibility ε (α carried FORWARD) instead DOUBLE-COUNTS α. This was previously only asserted; now
    # MEASURED via plb_elig=True (z → ε = _leaky_trace): ε HURTS −0.065 (shallow EEG) / −0.076 (deep),
    # `probe_doublecount.py`. ⇒ the raw presyn spike is CORRECT; there is NO recoverable signal on the
    # presyn side (the "richer" full-length ε is strictly worse). Flag kept for reproducibility only.
    plb_elig: bool = False            # ✅ confirmed double-count (ε HURTS); raw presyn is correct. A/B only.
    # ✗ DO NOT USE — REFUTED for float (kept for history / HW interest). Coarse ±1 credit lost to var_norm.
    delta_jug: bool = False           # ★★ CREDIT DELTA-JUG (2026-08-18, user) — the SigmaDelta cell applied
                                      # to the TIME/ERROR axis (separate from the weight E-jug). A per-neuron
                                      # BACKWARD leaky accumulator with THRESHOLD-FIRE runs the temporal credit:
                                      #   J_o[t] = α(1−θψ_o[t])·J_o[t+1] + ψ_o[t]·e_o[t];  fire ±1 when |J|>θ_dj,
                                      #   subtract θ_dj (error-feedback residual carried on).
                                      # = a coarse spike-train credit δ̃_o[t]∈{−1,0,+1}: HIGH temporal accuracy
                                      # (WHEN it fires), LOW magnitude accuracy (±1) — the biological/hardware
                                      # regime. The leak α sets the effective buffer depth (~1/(1−α)≈10, matches
                                      # the Q2 "depth≈8" result), so the jug IS the buffer AND the coarsener in
                                      # ONE primitive. Fires drive the UNCHANGED sign-fold E-jug: ΔW=Σ δ̃⊗z; route
                                      # fires down the Wᵀ hop. Input ψ·e is per-neuron RMS-normalised so θ_dj is
                                      # self-scaling (robust, no fine tuning). Breaks the sign-fold's TIME-COLLAPSE
                                      # (it sums δ over t BEFORE signing); the delta-jug coarsens ON the time axis.
    dj_theta: float = 1.0             # delta-jug fire threshold on the RMS-normalised gated error. Self-scaling.
    var_norm_fold: bool = False       # ★★ RELIABILITY-GRADED FOLD (2026-08-18) — the ONE thing structurally
                                      # missing vs Adam (cell-level maths, BUFFER_AS_DESIGN_TOOL.md §4). The
                                      # sign-fold writes a UNIFORM step (boss_lr) for every synapse regardless of
                                      # how CONSISTENT its accumulated credit is; Adam scales the step by
                                      # reliability: step = m/√v = (smoothed gradient)/(typical magnitude). This
                                      # replaces sign(E)·lr with an Adam-style per-synapse graded step on the
                                      # fold-accumulated E:  m←β1·m+(1−β1)·E; v←β2·v+(1−β2)·E²;
                                      # W += boss_lr·m/(√v+ε), clamped to ±vn_clip. |m/√v|≤~1 so the step is
                                      # STILL bounded like sign (robust, no raw-magnitude blow-up / overfit) but
                                      # GRADED: full step on consistent credit, ~0 on noisy credit. Tests whether
                                      # reliability-grading (not the credit, not the buffer) is the 0.16 gap.
                                      # Default OFF — the validated sign-fold is untouched.
    vn_beta1: float = 0.9             # var-norm fold: EMA decay for m (smoothed gradient / direction)
    vn_beta2: float = 0.99            # var-norm fold: EMA decay for v (gradient energy / typical magnitude)
    vn_clip: float = 1.0              # var-norm fold: clamp the graded step |m/√v| (keeps it sign-bounded)
    rule_readout: bool = False        # design A: the (leaky-integrator) READOUT weight W_ro is
                                      # RULE-TRAINED forwards-only (its own jug), NOT a fit logistic
                                      # head. scores = feats@W_roᵀ+b_ro; dW_ro = Sᵀ·feats, db_ro = Σ S
                                      # (= the readout gradient delta (1), readout_learning_maths.pdf).
    rr_warm: bool = True              # rule_readout: WARM-START W_ro from the logistic fit each epoch
                                      # (then co-adapt within epoch). False = ★ OPTION 1: PURE COLD
                                      # GRADIENT READOUT — no fit ever; W_ro,b_ro trained ONLY by their
                                      # own gradient (delta (1)) + var-norm graded write, exactly what
                                      # Adam does to W_f, forwards-only. The untried clean shot at the
                                      # readout residual (warm-start co-adapt only PERTURBED the fit).
    rr_center: bool = False           # rule_readout: subtract a running feature mean in the readout
                                      # gradient (conditioning for the leaky feature's DC; the trained
                                      # bias b_ro usually handles it, so default OFF — enable if cold collapses).
    rr_lr_mult: float = 1.0           # rule_readout: readout LR = boss_lr·rr_lr_mult. A COLD readout must
                                      # grow from a small init to an O(1) classifier — the ±boss_lr graded
                                      # step is far too small to bootstrap it, so cold needs a larger readout
                                      # LR (Adam's readout effectively has one). Scale knob for the bootstrap.
    fold_mode: str = "sign"           # ⚠ VALIDATED FLOAT FOLD = "sign" (the pcn_bigspec
                                      # sign_at_fold jug + momentum; the definitive rule — see
                                      # header lines 44-46). Temporal changes only the CREDIT
                                      # (ε + ψ-gated per-t routing into E); the FOLD stays this.
                                      # "threshold"/"sgd" are NON-validated experiments (below).
    # ⚠ NON-VALIDATED for float. "threshold" = the HARDWARE-cell SigmaDelta (E←λE+g; fire ±LSB when
    # |E|>θ; subtract θ). That mechanism is a HARDWARE ENABLER (coarse analog writes) — in FLOAT it
    # NEVER beats the fold rule ([[project_pcnchip_leaky_jug]] close-out). Kept only for HW-faithful
    # experiments; do NOT use it as the float comparator. "sgd" is a credit-only debug path.
    sd_theta: float = 1.0             # (threshold expt) fire threshold on the per-neuron accumulator
    sd_lsb: float = 0.02              # (threshold expt) fixed weight step per fired write
    sd_leak: float = 0.9             # (threshold expt) jug leak
    sd_wclip: float = 2.0             # (threshold expt) weight clamp
    sd_beta_norm: float = 0.99        # (threshold expt) per-output-neuron running-RMS EMA
    # init
    seed: int = 0
    init_gain: float = 1.0
    init: str = "ortho"               # "ortho" (validated static default) | "snn" = normal(0,
                                      # 1/√fanin). ⚠ SNN needs "snn": orthogonal under-drives the
                                      # tall input layer (per-elt 1/√max(in,out)) -> a SILENT net
                                      # the local jug can't bootstrap. "snn" = train_alif's init.
    # ─────────────────────────────────────────────────────────────────────────
    # ⚗⚗ EXPERIMENTAL — DEFAULT OFF, **NOT MAIN PATH** (2026-08-24, GENERALISABLE_CELL_DESIGN.md §8).
    # Learnable input FIR ("tap-weights"/delay) folded FORWARDS-ONLY. Adds a per-input-channel causal
    # kernel W_tap (C,K) applied to the input BEFORE the stack; its credit reuses the EXISTING δ chain
    # (route δ_0 one more Wᵀ hop to the input) ⊗ the RAW lagged input — NO new error channel. Folded by
    # the SAME sign rule into a small E_tap accumulator. Requires: pl_buffer_n>0 + per_t_route (uses the
    # validated windowed-adjoint path), and a forward_fn that APPLIES learner.W_tap and returns
    # cap['X_raw'] (the pre-tap input). When tap_fold=False NOTHING below allocates or runs — the
    # validated path is byte-for-byte unchanged.
    tap_fold: bool = False            # ⚗ enable the experimental learnable-delay fold
    tap_K: int = 16                   # FIR window length (taps); delay of tap k = tap_K-1-k
    # ─────────────────────────────────────────────────────────────────────────
    # ⚠ CURRENTLY REDUNDANT FOLLOWING TESTING (2026-08-24) — DEFAULT OFF, kept for reproducibility only.
    # Learnable LIF/ALIF blend q (ALIF DEMOTED to a conditional enrichment; GENERALISABLE_CELL_DESIGN.md §4).
    #   Verdict: (1) LIF≈ALIF standalone; ALIF HURTS EEG −0.04/−0.05 ⇒ nothing to drive q toward on our
    #   tasks; (2) the forwards-only credit CANNOT drive q — q_lr_mult sweep {1..200} leaves q̄ plateaued at
    #   0.53→0.74 (sublinear, accuracy declines), signal-limited not step-limited (`probe_qmult.py`). ⇒ q is
    #   NOT part of the core cell; the tap-weight is the timescale primitive instead. Would only earn its
    #   place on genuine long working-memory tasks (SHD/SSC-with-gaps, delayed-match), which GSC/EEG are not.
    # Mechanism (unchanged, for reference): per-neuron q∈(0,1), output = q·s_LIF + (1−q)·s_ALIF; credit reuses
    # the δ chain E_q += Σ_t δ_l·(s_LIF−s_ALIF); folded by the sign rule w/ bootstrap q_lr_mult. Requires
    # pl_buffer_n>0 + per_t_route + a forward_fn returning cap['DQ']. OFF ⇒ nothing allocates; path unchanged.
    q_fold: bool = False              # ⚠ REDUNDANT after testing (ALIF demoted); experimental, kept for repro
    q_lr_mult: float = 1.0            # bootstrap LR for q — swept {1..200}: q̄ plateaus, does NOT rescue q
    q_alif_rho: float = 0.98          # ALIF adaptation-trace decay
    q_alif_wa: float = 0.5            # ALIF self-inhibition gain

    # ⚗⚗ EXPERIMENTAL — DEFAULT OFF, **NOT MAIN PATH** (2026-08-24, GENERALISABLE_CELL_DESIGN.md §4).
    # AMPLITUDE-DERIVATIVE in the credit. The LIF-rate feature is a (crude) half-SQUARE of the drive, so
    # the exact gradient of a squared feature ∂(Wx)²/∂W = 2·(Wx)·x carries a DRIVE factor the step
    # surrogate ψ only emulates crudely (accumulated threshold bumps). Blend a smooth drive-proportional
    # piece into the LOCAL surrogate that gates each layer's OWN credit: ψ_eff = (1−c)·ψ_step + c·(v/θ),
    # v = pre-reset membrane. Timing tasks (GSC) want ψ_step (c→0); power tasks (EEG band-power) want the
    # drive-derivative (c>0). Requires a forward_fn that returns cap['VDR']=[v/θ per layer]. OFF (c=0 or
    # flag False) ⇒ ψ untouched, validated path byte-for-byte unchanged. LOCAL ONLY — does NOT touch the
    # routed error down the Wᵀ hop.
    drive_deriv: bool = False         # ⚗ enable the experimental amplitude-derivative surrogate blend
    drive_c: float = 0.0              # blend weight c: 0 = pure ψ_step (validated), >0 mixes in v/θ


# ─────────────────────────────────────────────────────────────────────────────
def _default_encoder(kind: str, alpha: float = 0.9):
    """Temporal front-end (N,T,D) -> static features (N,F). 'rate' = mean firing rate;
    'leaky' = leaky-integrated final state concatenated with the rate.

    ⚠ TRAP T3: both 'rate' and 'leaky' COLLAPSE the time axis. For delta-modulated spike
    data (S2S/speech2spikes) whose time-mean is ≈0 this DESTROYS the signal (Adam→chance
    too). For such data pass a TEMPORAL encoder (e.g. the binned encoder in gap_ladder.py)
    instead of these. These two are only appropriate for rate-coded / non-delta input."""
    def rate(X):
        return X.mean(dim=1)
    def leaky(X):
        B, T, D = X.shape
        s = torch.zeros(B, D, device=X.device, dtype=X.dtype)
        for t in range(T):
            s = alpha * s + X[:, t, :]
        return torch.cat([s / T, X.mean(dim=1)], dim=1)   # recency + rate
    return {"rate": rate, "leaky": leaky}[kind]


class PCNLearner:
    def __init__(self, layer_dims: List[int], n_classes: int,
                 cfg: Optional[PCNConfig] = None,
                 encoder: Optional[Callable] = None,
                 device: Optional[str] = None,
                 forward_fn: Optional[Callable] = None):
        """layer_dims: [in, h1, ..., feat] — the hidden stack producing classifier
        features. encoder: callable (N,T,D)->(N,in) for temporal input, or a string
        'rate'/'leaky', or None (static input fed straight in).
        forward_fn: OPTIONAL external forward `fn(self, X)->Fs` (activation stack
        [input,...,feat]) that REPLACES the internal leaky-ReLU stack — used to drive an
        SNN (or any net) with THIS rule: forward_fn runs the net FROM self.W and returns
        the captured per-layer activations, so fit()/auto_lr/fold all update self.W and the
        external net reads them back. dims must match the external net's layer widths."""
        self.cfg = cfg or PCNConfig()
        # ── Guard the 2026-07-13 degenerate-softmax bug (mirror pcn_bigspec.py:2462) ──
        #     lstsq    + softmax = ~46  <- THE BUG (p̂ is not a logit)
        #     lstsq    + mse     = ~54     correct pairing (chip-cheap)
        #     logistic + softmax = ~69     correct pairing  <- BEST (definitive default)
        #     logistic + mse     = ~10     collapses (a logit is not a p̂)
        if self.cfg.clf == "lstsq" and self.cfg.err_mode == "softmax":
            raise ValueError(
                "REFUSING: clf='lstsq' + err_mode='softmax' is the degenerate-softmax bug. "
                "lstsq outputs are probabilities (rms~0.05), not logits. "
                "Use err_mode='mse' with lstsq, or clf='logistic' with softmax.")
        if self.cfg.clf == "logistic" and self.cfg.err_mode == "mse":
            raise ValueError(
                "REFUSING: clf='logistic' + err_mode='mse'. Logistic outputs are LOGITS; "
                "subtracting them from one-hot collapses. Use err_mode='softmax' with logistic.")
        self.dims = list(layer_dims)
        self.n_classes = n_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(encoder, str):
            encoder = _default_encoder(encoder)
        self.encoder = encoder
        self.forward_fn = forward_fn      # external forward (SNN etc.); None = internal stack
        self.a = self.cfg.leaky_alpha
        g = torch.Generator().manual_seed(self.cfg.seed)
        self.W, self.E, self.vel = [], [], []
        for i in range(len(self.dims) - 1):
            d_in, d_out = self.dims[i], self.dims[i + 1]
            m = torch.empty(d_out, d_in)
            if self.cfg.init == "snn":
                torch.nn.init.normal_(m, 0.0, 1.0 / d_in ** 0.5, generator=g)  # 1/√fanin (train_alif)
            else:
                torch.nn.init.orthogonal_(m, gain=self.cfg.init_gain)  # random orthogonal (NOT GHA)
            self.W.append(m.to(self.device))
            self.E.append(torch.zeros(d_out, d_in, device=self.device))
            self.vel.append(torch.zeros(d_out, d_in, device=self.device))
        self.W_f = None
        self.b_f = None
        self.eta = [self.cfg.boss_lr] * len(self.W)
        self._acc_ct = 0
        self._bh_sum = 0.0; self._bh_cnt = 0   # boss_h_damp: aggregate severity over the fold window
        if self.cfg.boss_h_damp:
            print(f"[⚗ boss_h_damp ON] sign step ×clamp(mean_bh/{self.cfg.boss_h_damp_div}, "
                  f"{self.cfg.boss_h_damp_floor}, 1) — hidden W only, self-annealing damper", flush=True)
        self._pn_ms = [torch.zeros(w.shape[0], device=self.device) for w in self.W]  # per-neuron |E| EMA
        self._pn_init = [False] * len(self.W)
        # ── temporal (e-prop) state ──
        self._jug = [torch.zeros_like(w) for w in self.W]            # SigmaDelta persistent jug
        self._vn_m = [torch.zeros_like(w) for w in self.W]           # var-norm fold: momentum (m)
        self._vn_v = [torch.zeros_like(w) for w in self.W]           # var-norm fold: grad energy (v)
        self._sd_ms = [torch.zeros(w.shape[0], 1, device=self.device) for w in self.W]  # per-neuron RMS
        self._Mc = None                                             # cached (T,T) causal leaky matrix
        self._gjac = None                                           # cached (T,) readout Jacobian g(t)
        self._cap = None                                           # last temporal forward capture
        # ⚗ EXPERIMENTAL learnable input FIR (default OFF). delta-init = identity (current tap = 1).
        if self.cfg.tap_fold:
            C, K = self.dims[0], self.cfg.tap_K
            wt = torch.zeros(C, K); wt[:, -1] = 1.0                # identity ⇒ starts as no-op, then adapts
            self.W_tap = wt.to(self.device)
            self.E_tap = torch.zeros(C, K, device=self.device)
            self.vel_tap = torch.zeros(C, K, device=self.device)
            self._vn_m_tap = torch.zeros(C, K, device=self.device)   # var-norm tap momentum (if var_norm_fold)
            self._vn_v_tap = torch.zeros(C, K, device=self.device)   # var-norm tap grad-energy
        # ⚠ REDUNDANT after testing (ALIF demoted, see q_fold cfg banner). EXPERIMENTAL, default OFF.
        if self.cfg.q_fold:
            self.q_raw = [torch.zeros(self.dims[l + 1], device=self.device) for l in range(len(self.W))]
            self.E_q = [torch.zeros_like(q) for q in self.q_raw]
            self.vel_q = [torch.zeros_like(q) for q in self.q_raw]
            self._vn_m_q = [torch.zeros_like(q) for q in self.q_raw]   # var-norm q state (if var_norm_fold)
            self._vn_v_q = [torch.zeros_like(q) for q in self.q_raw]
        # rule-trained readout (design A): W_ro replaces the fit logistic head, with its own jug
        if self.cfg.rule_readout:
            ro = torch.empty(n_classes, self.dims[-1])
            torch.nn.init.normal_(ro, 0.0, 1.0 / self.dims[-1] ** 0.5, generator=g)  # small (Adam-like) cold init
            self.W_ro = ro.to(self.device)
            self.b_ro = torch.zeros(n_classes, device=self.device).double()   # readout bias (trained by grad Σ S)
            self.E_ro = torch.zeros_like(self.W_ro)
            self.E_b_ro = torch.zeros_like(self.b_ro)     # bias gradient accumulator
            self.vel_ro = torch.zeros_like(self.W_ro)
            self._vn_m_ro = torch.zeros_like(self.W_ro)   # var-norm fold: readout weight momentum
            self._vn_v_ro = torch.zeros_like(self.W_ro)   # var-norm fold: readout weight grad energy
            self._vn_m_b = torch.zeros_like(self.b_ro)    # var-norm fold: readout bias momentum
            self._vn_v_b = torch.zeros_like(self.b_ro)    # var-norm fold: readout bias grad energy
            self._feat_mean = torch.zeros(self.dims[-1], device=self.device).double()  # running feature mean (rr_center)
            self._feat_mean_init = False

    # ── input / forward ──────────────────────────────────────────────────────
    def _encode(self, X):
        X = X.to(self.device).float()
        if self.forward_fn is not None:
            return X                      # external forward consumes raw (temporal) input
        if X.dim() == 3:
            if self.encoder is None:
                raise ValueError("3-D (temporal) input needs an encoder")
            X = self.encoder(X)
        return X

    @torch.no_grad()
    def _forward(self, X):
        """Return activation stack F[0..L]; F[0]=input, F[-1]=classifier features."""
        if self.forward_fn is not None:
            return self.forward_fn(self, X)   # SNN/external forward driven from self.W
        Fs = [X]
        f = X
        for W in self.W:
            f = F.leaky_relu(f @ W.t(), negative_slope=self.a)
            Fs.append(f)
        return Fs

    def _leaky_deriv(self, f):
        return torch.where(f >= 0.0, torch.ones_like(f), torch.full_like(f, self.a))

    # ── classifier ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def _fit_clf(self, feats, y):
        A = feats.double()
        mean = A.mean(0)
        scale = A.std(0).clamp_min(1e-4)
        A_sc = (A - mean) / scale
        if self.cfg.clf == "logistic":
            import warnings
            from sklearn.linear_model import LogisticRegression
            # PERF NOTE (the logits-slowdown): the per-epoch lbfgs fit is the bottleneck when
            # features are wide (EMNIST 1152-d). We A/B'd sklearn warm_start (reuse prev coef):
            # it gave only 1.38× AND −0.33pp — the speed came FROM under-convergence (tight tol
            # removed the win). Non-neutral => it would inflate the PCN:Adam GAP (Adam has no
            # such fit). So we keep the FULL COLD fit for clean definitive numbers. --clf_sub
            # (subsampling) is worse still: it does NOT transfer (−5pp on wide readouts). A real
            # fit speed-up would be a torch/GPU multinomial logistic, as a separate vetted change.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LogisticRegression(solver="lbfgs", C=self.cfg.clf_C,
                                         max_iter=2000, random_state=0)
                clf.fit(A_sc.cpu().numpy(), y.cpu().numpy())
            coef = torch.tensor(clf.coef_, device=self.device).double()
            icpt = torch.tensor(clf.intercept_, device=self.device).double()
            # ⚠ BINARY GUARD: sklearn LogisticRegression returns a SINGLE (1,feat) coef for 2
            # classes (one logit), not one row per class. Left as-is, W_f is (1,feat) => scores
            # are (B,1) and _top_delta's S[arange, y] indexes out of bounds for y==1 (device
            # assert). Expand to the equivalent 2-class softmax logits [0, w·x+b] (class 0 =
            # reference). Multiclass (coef rows == n_classes) is untouched. Keeps the module
            # usable on 2-class tasks (e.g. EEG-MI), not just the multiclass EMNIST/GSC it grew on.
            if coef.shape[0] == 1 and self.n_classes == 2:
                coef = torch.cat([torch.zeros_like(coef), coef], 0)   # (2, feat)
                icpt = torch.cat([torch.zeros_like(icpt), icpt], 0)   # (2,)
            self.W_f = coef / scale
            self.b_f = icpt - coef @ (mean / scale)
        else:  # lstsq multiclass
            Y = F.one_hot(y.to(self.device), self.n_classes).double()
            A_aug = torch.cat([A_sc, torch.ones(len(A_sc), 1, device=self.device).double()], 1)
            Wb = torch.linalg.lstsq(A_aug, Y).solution
            W_raw, b_raw = Wb[:-1], Wb[-1]                    # (feat,cls),(cls,)
            self.W_f = (W_raw / scale[:, None]).t()           # (cls, feat)
            self.b_f = b_raw - W_raw.t() @ (mean / scale)

    @torch.no_grad()
    def _scores(self, feats):
        if self.cfg.rule_readout:
            fc = (feats.double() - self._feat_mean) if self.cfg.rr_center else feats.double()
            return fc @ self.W_ro.double().t() + self.b_ro              # rule-trained readout (+bias, +opt centering)
        return feats.double() @ self.W_f.t() + self.b_f

    # ── boss_h top-δ ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def _top_delta(self, scores, y, track_bh=False):
        """Teaching error S, gated by perceptron stop-on-correct + boss_h severity.
        err_mode 'softmax': S = onehot − softmax(scores)  (scores must be LOGITS => logistic)
        err_mode 'mse':     S = onehot − scores           (scores are p̂ => lstsq; no exp).
        track_bh: accumulate the AGGREGATE boss_h severity over the fold window (boss_h_damp).
        Set True ONLY at the boss/classification level — NOT the per-timestep route call."""
        S = -scores.clone() if self.cfg.err_mode == "mse" else -F.softmax(scores, dim=1)
        S[torch.arange(len(y)), y] += 1.0
        pred = scores.argmax(1)
        wrong = pred != y
        if track_bh and self.cfg.boss_h_damp:            # boss_h_damp: mean severity over the batch
            mg = scores[torch.arange(len(y)), pred] - scores[torch.arange(len(y)), y]  # ≥0 (0 if correct)
            span = (scores.amax(1) - scores.amin(1)).clamp_min(1e-6)
            bh = (7.0 * self.cfg.bh_gain * mg / span).int().clamp(0, self.cfg.boss_h_max)
            self._bh_sum += float(bh.double().sum()); self._bh_cnt += len(y)  # correct⇒bh=0 ⇒ tracks err rate
        if self.cfg.stop_on_correct:
            S[~wrong] = 0.0
        if self.cfg.bh_gate:
            mg = scores[torch.arange(len(y)), pred] - scores[torch.arange(len(y)), y]
            span = (scores.amax(1) - scores.amin(1)).clamp_min(1e-6)
            bh = (7.0 * self.cfg.bh_gain * mg / span).int().clamp(max=self.cfg.boss_h_max)
            sev = torch.where(bh > 0, (bh.double() + 1) / 4.0,
                              torch.full_like(mg, self.cfg.bh_leak * 0.25))
            S = S * torch.where(wrong, sev, torch.ones_like(sev))[:, None]
        return S, int(wrong.sum())

    def _constrain(self, D):
        """Per-row (per-sample) quantise the transported δ to delta_bits levels."""
        b = self.cfg.delta_bits
        if b <= 0:
            return D
        scale = D.abs().amax(1, keepdim=True).clamp_min(1e-12)
        L = 2 ** (b - 1) - 1
        return torch.round(D / scale * L) / L * scale

    # ── auto_lr ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _calibrate_auto_lr(self, X, y):
        if self.cfg.temporal:                            # temporal: per-layer POOLED rates as Fs
            cap = self.forward_fn(self, X, capture=True)
            Fs = [z.double().mean(1) for z in cap['Z']] + [cap['feats'].double()]  # L+1 rate maps
            scores = self._scores(cap['feats'])
        else:
            Fs = self._forward(X)
            scores = self._scores(Fs[-1])
        pred = scores.argmax(1); wrong = pred != y
        if wrong.any():
            mg = scores[wrong, pred[wrong]] - scores[wrong, y[wrong]]
            span = (scores[wrong].amax(1) - scores[wrong].amin(1)).clamp_min(1e-6)
            bh = (7.0 * mg / span).int().clamp(0, self.cfg.boss_h_max)
            st = ((bh.double() + 1) / 4.0)[bh > 0]
            mean_step = float(st.mean()) if len(st) else 1.0
        else:
            mean_step = 1.0
        rms_delta = mean_step * self.cfg.delta
        rms = lambda A: float(A.double().pow(2).mean().sqrt())
        for l in range(len(self.W)):
            fanin = self.dims[l]
            self.eta[l] = self.cfg.auto_frac * rms(Fs[l + 1]) / max(
                rms_delta * fanin * rms(Fs[l]), 1e-12)

    # ── TEMPORAL/SNN normalisation helpers (Concept #1) ───────────────────────
    @staticmethod
    def _unit_rms(v, eps=1e-4):
        """Row-wise unit-RMS: v / (||v_row|| / sqrt(dim)). Normalises the pathological
        narrow/delta-mod presyn scale that auto_lr cannot (per-sample)."""
        rms = v.pow(2).mean(dim=1, keepdim=True).sqrt()
        return torch.where(rms > eps, v / rms, v)

    @staticmethod
    def _rms_preserving(src, out, eps=1e-8):
        """Scale `out` so each row's RMS matches `src`'s — RMS-preserving backprojection."""
        rs = src.pow(2).mean(dim=1, keepdim=True).sqrt()
        ro = out.pow(2).mean(dim=1, keepdim=True).sqrt()
        return out * torch.where(ro > eps, rs / ro, torch.ones_like(ro))

    # ── accumulate + fold ────────────────────────────────────────────────────
    @torch.no_grad()
    def _accumulate(self, Fs, S, bs):
        """Backproject S down the stack, accumulate outer products into E."""
        lr = self.eta if self.cfg.auto_lr else [self.cfg.boss_lr] * len(self.W)
        scale = (1.0 / self.cfg.fold_every) if (self.cfg.fold_mean and self.cfg.fold_every > 1) else 1.0
        D = (S @ self.W_f) * self._leaky_deriv(Fs[-1]).double()      # δ at top features
        D = self._constrain(D)
        for l in reversed(range(len(self.W))):
            pre = Fs[l].double()
            if self.cfg.unit_rms_presyn:                # Concept #1: unit-RMS presyn (SNN)
                pre = self._unit_rms(pre)
            self.E[l] += (lr[l] * scale) * (D.t() @ pre)
            self.E[l].clamp_(-self.cfg.e_clip, self.cfg.e_clip)
            if l > 0:
                G = D @ self.W[l].double()
                if self.cfg.unit_rms_presyn:            # Concept #1: RMS-preserving backproj
                    G = self._rms_preserving(D, G)
                D = self._constrain(G * self._leaky_deriv(Fs[l]).double())

    # ── TEMPORAL credit (e-prop): ε eligibility + ψ-gated per-t routing ───────
    def _leaky_trace(self, Z, alpha):
        """Causal leaky eligibility trace of presyn spikes Z (B,T,in):
        ε_t = alpha·ε_{t-1} + Z_t, via a cached (T,T) causal-decay matrix. = ∂v/∂W echo."""
        T = Z.shape[1]
        if self._Mc is None or self._Mc.shape[0] != T or float(self._Mc[1, 0]) != alpha:
            idx = torch.arange(T, device=self.device)
            diff = (idx[:, None] - idx[None, :]).float()          # t - s
            self._Mc = torch.where(diff >= 0, alpha ** diff.clamp(min=0),
                                   torch.zeros_like(diff))         # (T,T) causal
        return torch.einsum('ts,bsi->bti', self._Mc.to(Z.dtype), Z)

    def _readout_jacobian(self, T, device):
        """B4: the leaky-integrator readout's temporal Jacobian g(t) = ∂feat/∂z_t.
        A hidden spike at t feeds the running leaky integral for every later step, so
        g(t) = (1/T)·Σ_{τ=t}^{T-1} β^{τ−t} = (1/T)(1−β^{T−t})/(1−β). Early spikes weighted more,
        tapering over the last ~1/(1−β) steps. Rate readout would give a flat g(t)=1/T. Closed-form
        from readout_beta (nothing tuned); MEAN-normalised so only the SHAPE acts (overall scale is a
        sign_at_fold no-op). Cached on self."""
        # ★ g(t) form MUST match the readout: RATE → flat 1/T, LEAKY → leaky kernel. The forward_fn stamps
        # learner._readout_form (authority); fall back to cfg.readout_form. No override ⇒ always matched.
        form = getattr(self, "_readout_form", None) or self.cfg.readout_form
        if form == "rate":                                        # flat 1/T g(t) — correct for RATE readout
            return torch.ones(T, device=device, dtype=torch.double)   # uniform (mean-norm=1); no leaky cache
        b = self.cfg.readout_beta
        if getattr(self, "_gjac", None) is None or self._gjac.shape[0] != T:
            k = torch.arange(T, device=device).double()           # T−t = steps remaining after t
            rem = (T - k)                                          # τ from t..T-1  → count = T−t
            a = (1.0 - b ** rem) / (1.0 - b)                       # Σ_{τ≥t} β^{τ−t}  (drop 1/T const)
            self._gjac = a / a.mean()                              # mean-normalise: shape only
        return self._gjac

    def _forward_influence(self, psi, n, alpha):
        """(b) SnAp-n forward influence gate Ψ̃ (PREDICTIVE_DELTA_DESIGN.md §b).
        Ψ̃_o[t] = Σ_{k=0}^{n} (Π_{j=1}^{k} α(1−ψ_o[t+j])) · ψ_o[t+k]  — the neuron's readiness
        propagated FORWARD n steps, discounted by membrane leak α, CUT by reset (1−ψ). = the causal,
        n-truncated forward influence of o's activity at t on the readout. Uses only t..t+n (n-lag,
        physical). n=0 ⇒ ψ (the jug). psi: (B,T,O)."""
        T = psi.shape[1]
        out = psi.clone()                                       # k=0 term
        P = torch.ones_like(psi)                                # running product Π_{j=1}^{k} α(1−ψ[t+j])
        decay = alpha * (1.0 - psi)                             # per-step forward carry α(1−ψ[t])
        for k in range(1, n + 1):
            dshift = torch.zeros_like(decay); dshift[:, :T - k] = decay[:, k:]   # decay[t+k], tail 0
            pshift = torch.zeros_like(psi);   pshift[:, :T - k] = psi[:, k:]     # ψ[t+k], tail 0
            P = P * dshift
            out = out + P * pshift
        return out

    def _error_carry(self, e, psi, n, alpha, theta):
        """FORK (a): the TRUE unrolled BPTT adjoint on the ERROR side, via the causal buffer.
        δ_o[t] = Σ_{k=0}^{n} (Π_{j=0}^{k−1} α(1−θ·ψ_o[t+j])) · ψ_o[t+k] · e_o[t+k]
        where e is the incoming error at this layer (top: d_o·g(t); below: the routed adjoint).
        Both ψ[t+k] and e[t+k] are BUFFER READS (indices t..t+n only ⇒ causal, n-lag, physical) —
        this is a FIFO learning-delay, NOT a forward-time lookahead. Contrast _forward_influence,
        which carried READINESS ψ[t+k] on the GATE side with a constant error and a j=1..k reset
        offset; here the carry is the ERROR term and the reset product is j=0..k−1 (includes the
        current step, exactly matching the surrogate-BPTT recurrence δ[t]=ψ[t]e[t]+α(1−θψ[t])δ[t+1]).
        The returned δ multiplies eligibility ε (the weight update) AND routes down through Wᵀ.
        e, psi: (B,T,O). n=0 ⇒ δ=ψ·e (the plain per-t jug signal). O(n) time/memory."""
        T = e.shape[1]
        w = psi * e                                             # ψ[t]·e[t]  (k=0 term)
        delta = w.clone()
        # ⚗ HW-simplicity probe: buf_uniform drops the reset-product decay (r=1) ⇒ a plain
        #   uniform n-window Σ ψ·e (the simpler elig_buffer.v cell). Default = faithful carry.
        r = (torch.ones_like(psi) if self.cfg.buf_uniform
             else alpha * (1.0 - theta * psi))                  # reset carry α(1−θψ[t])  (B,T,O)
        P = torch.ones_like(psi)                                # running Π_{j=0}^{k−1} r[t+j]
        for k in range(1, n + 1):
            rshift = torch.zeros_like(r); rshift[:, :T - (k - 1)] = r[:, k - 1:]   # r[t+k−1]
            wshift = torch.zeros_like(w); wshift[:, :T - k] = w[:, k:]             # (ψ·e)[t+k]
            P = P * rshift                                      # extend product by the t+k−1 reset
            delta = delta + P * wshift                          # + carry of the future error·readiness
        return delta

    def _delta_jug(self, u, carry, sd_theta):
        """★★ CREDIT DELTA-JUG — SigmaDelta on the credit/time axis. A per-neuron BACKWARD leaky
        accumulator with threshold-fire that runs the temporal buffer AND coarsens it in one primitive.
        u = ψ·e (gated error, per-neuron RMS-normalised); carry = α(1−θψ) (the membrane leak/reset).
          J[t] = carry[t]·J[t+1] + u[t];  fire = sign(J)·1[|J|>θ_dj];  J −= θ_dj·fire  (residual carried).
        Returns fires (B,T,O) ∈ {−1,0,+1}: coarse spike-train credit — timing precise, magnitude ±1.
        The leak in `carry` gives ~1/(1−α) effective depth ⇒ the jug is the buffer. Backward-in-time =
        the adjoint direction, realised over the learning-delay buffer (read newest→oldest)."""
        B, T, O = u.shape
        J = torch.zeros(B, O, device=u.device, dtype=u.dtype)
        fires = torch.zeros_like(u)
        for t in range(T - 1, -1, -1):                          # backward: adjoint direction
            J = carry[:, t] * J + u[:, t]                       # leaky-integrate the gated error
            fire = torch.sign(J) * (J.abs() > sd_theta).to(J.dtype)
            J = J - sd_theta * fire                             # SigmaDelta error-feedback reset
            fires[:, t] = fire
        return fires

    @torch.no_grad()
    def _accumulate_temporal(self, cap, S, y=None):
        """Forwards-only temporal credit. cap = {'feats',(B,F); 'Z':[.(B,T,in)]; 'PSI':[.(B,T,out)]}.
        Route the top learning signal S down the hidden stack (ψ-gated per-t if per_t_route),
        accumulate dW = Σ_t (δ·ψ)ᵀ ε into the jug E. Mirrors train_alif.fo_update but per-t.
        `y` (labels) is only needed for the L1 running-delta path (per-step teaching error)."""
        Z, PSI = cap['Z'], cap['PSI']
        VDR = cap.get('VDR')                                     # ⚗ [v/θ per layer] for the drive-derivative
        L = len(self.W); B, T = Z[0].shape[0], Z[0].shape[1]
        pt, pr = self.cfg.per_t_route, self.cfg.psi_route
        # E += η·dW, accumulated into the SAME jug the validated sign_at_fold fold reads (auto_lr
        # off for the SNN fork ⇒ η = boss_lr; magnitude is a no-op under sign_at_fold but sets e_clip).
        eta = self.eta if self.cfg.auto_lr else [self.cfg.boss_lr] * L   # faithful to static _accumulate
        # top-δ at the feature layer. Design B: W_f (fit logistic) transports. Design A: the
        # rule-trained W_ro transports AND is itself updated (dW_ro = Sᵀ·feats into its own jug).
        if self.cfg.rule_readout:
            feats = cap['feats'].double()                         # (B, feat) = ro_feat
            Bf = feats.shape[0]
            if self.cfg.rr_center:                               # subtract running feature mean (conditioning)
                fm = feats.mean(0)
                self._feat_mean = fm if not self._feat_mean_init else 0.99 * self._feat_mean + 0.01 * fm
                self._feat_mean_init = True
                fc = feats - self._feat_mean
            else:
                fc = feats
            self.E_ro += self.cfg.boss_lr * (S.double().t() @ fc) / Bf   # readout WEIGHT gradient  Sᵀ·feat
            self.E_ro.clamp_(-self.cfg.e_clip, self.cfg.e_clip)
            self.E_b_ro += self.cfg.boss_lr * S.double().sum(0) / Bf     # readout BIAS gradient  Σ S
            self.E_b_ro.clamp_(-self.cfg.e_clip, self.cfg.e_clip)
            Wtop = self.W_ro.double()
        else:
            Wtop = self.W_f
        # ── L1 running-delta (PREDICTIVE_DELTA_DESIGN.md §4): per-step δ[t] from the running readout ──
        use_rd = (self.cfg.running_delta and pt and (not self.cfg.rule_readout)
                  and (cap.get('RO_t') is not None) and (y is not None))
        use_ec = (self.cfg.error_carry_n > 0 and pt and not use_rd)   # FORK (a): true error-side carry
        use_plb = (self.cfg.pl_buffer_n > 0 and pt and not use_rd)    # ★ Q2 PER-LAYER BUFFER (δ⊗direct presyn)
        use_dj = (self.cfg.delta_jug and pt and not use_rd)          # ★★ CREDIT DELTA-JUG (SigmaDelta on credit)
        if use_rd:
            RO = cap['RO_t'].double()                             # (B,T,feat) running readout r[t]
            scr = RO @ self.W_f.t() + self.b_f                    # (B,T,ncls) per-step logits (same fit head)
            yv = y[:, None].expand(B, T).reshape(B * T)           # label repeated over t
            St, _ = self._top_delta(scr.reshape(B * T, -1), yv)   # per-(sample,t) teaching error δ[t]
            St = St.reshape(B, T, -1)                             # (B,T,ncls) per-step δ[t]
            a = self.cfg.pred_lead                                # L2: signal[t] = δ[t] − a·δ[t−1]
            if a != 0.0:                                          # a=1 ⇒ pure velocity (differences OUT
                Sh = torch.zeros_like(St); Sh[:, 1:] = St[:, :-1] # the low-evidence DC class-bias that
                St = St - a * Sh                                  # collapses raw L1)
            s = self._constrain(St.reshape(B * T, -1) @ self.W_f).reshape(B, T, -1)  # route — NO broadcast
        else:
            d = self._constrain(S.double() @ Wtop)                # (B, feat) = δ into last hidden
            if pt:
                s = d[:, None, :].expand(B, T, d.shape[1])        # broadcast over t (time-CONSTANT credit)
                if self.cfg.readout_jacobian or use_ec or use_plb or use_dj:   # shape by readout Jacobian g(t)
                    a = self._readout_jacobian(T, Z[0].device)    # (T,) g(t), mean-normalised
                    s = s * a[None, :, None]                      # top error e_L[t] = d_o·g(t)
            else:
                s = d                                             # time-collapsed δ
        tap_delta0 = None                                        # ⚗ capture δ_0 for the experimental tap fold
        q_deltas = [None] * L                                     # ⚗ capture δ_l per layer for the q blend fold
        for l in reversed(range(L)):
            psi = PSI[l].double()                                 # (B,T,out_l) readiness gate
            if self.cfg.drive_deriv and self.cfg.drive_c > 0.0 and VDR is not None:
                c = self.cfg.drive_c                              # ⚗ ψ_eff = (1−c)·ψ_step + c·(v/θ) — LOCAL
                psi = (1.0 - c) * psi + c * VDR[l].double()       # amplitude-derivative for the SQUARE feature
            if use_dj:                                            # ★★ CREDIT DELTA-JUG (SigmaDelta on the credit)
                u = psi * s                                       # gated error ψ·e  (B,T,out_l)
                rms = u.pow(2).mean(dim=(0, 1), keepdim=True).sqrt().clamp_min(1e-6)   # per-neuron scale
                carry = self.cfg.elig_alpha * (1.0 - self.cfg.carry_theta * psi)       # membrane leak α(1−θψ)
                fires = self._delta_jug(u / rms, carry, self.cfg.dj_theta)   # coarse ±1 temporal credit δ̃
                z = Z[l].double()                                 # DIRECT presyn spike
                if self.cfg.unit_rms_presyn:
                    z = self._unit_rms(z.reshape(B * T, -1)).reshape(B, T, -1)
                self.E[l] += eta[l] * torch.einsum('bto,bti->oi', fires, z) / (B * T)   # ΔW = Σ_t δ̃⊗z
                self.E[l].clamp_(-self.cfg.e_clip, self.cfg.e_clip)
                if l > 0:                                         # route the FIRES down the Wᵀ hop
                    s2 = torch.einsum('bto,oi->bti', fires, self.W[l].double())
                    if self.cfg.unit_rms_presyn:
                        s2 = self._rms_preserving(fires.reshape(B * T, -1),
                                                  s2.reshape(B * T, -1)).reshape(B, T, -1)
                    s = self._constrain(s2.reshape(B * T, -1)).reshape(B, T, -1)
                continue
            if use_plb:                                           # ★ Q2 PER-LAYER BUFFER (the corrected credit)
                delta = self._error_carry(s, psi, self.cfg.pl_buffer_n,
                                          self.cfg.elig_alpha, self.cfg.carry_theta)   # local n-step adjoint δ_l
                if self.cfg.tap_fold and l == 0:
                    tap_delta0 = delta                            # ⚗ δ_0 for input-tap credit (after the loop)
                if self.cfg.q_fold:
                    q_deltas[l] = delta                           # ⚗ δ_l for the q-blend credit (after the loop)
                if self.cfg.plb_elig:                             # ⚗ double-count A/B: FORWARD-α eligibility ε
                    z = self._leaky_trace(Z[l].double(), self.cfg.elig_alpha)   # full-length presyn integration
                else:
                    z = Z[l].double()                             # DIRECT presyn SPIKE (validated; NOT the echo)
                if self.cfg.unit_rms_presyn:
                    z = self._unit_rms(z.reshape(B * T, -1)).reshape(B, T, -1)
                self.E[l] += eta[l] * torch.einsum('bto,bti->oi', delta, z) / (B * T)   # ΔW_l = Σ_t δ_l⊗z_l
                self.E[l].clamp_(-self.cfg.e_clip, self.cfg.e_clip)
                if l > 0:                                         # route the adjoint down the forwards-only Wᵀ hop
                    s2 = torch.einsum('bto,oi->bti', delta, self.W[l].double())
                    if self.cfg.unit_rms_presyn:
                        s2 = self._rms_preserving(delta.reshape(B * T, -1),
                                                  s2.reshape(B * T, -1)).reshape(B, T, -1)
                    s = self._constrain(s2.reshape(B * T, -1)).reshape(B, T, -1)
                continue
            eps = self._leaky_trace(Z[l].double(), self.cfg.elig_alpha)   # (B,T,in_l)
            if self.cfg.elig_slow > 0:
                eps = eps + self._leaky_trace(Z[l].double(), self.cfg.elig_slow)
            if self.cfg.unit_rms_presyn:
                eps = self._unit_rms(eps.reshape(B * T, -1)).reshape(B, T, -1)
            if use_ec:                                            # FORK (a): true error-side BPTT carry
                delta = self._error_carry(s, psi, self.cfg.error_carry_n,
                                          self.cfg.elig_alpha, self.cfg.carry_theta)   # (B,T,out) adjoint δ
                self.E[l] += eta[l] * torch.einsum('bto,bti->oi', delta, eps) / (B * T)
                self.E[l].clamp_(-self.cfg.e_clip, self.cfg.e_clip)
                if l > 0:                                         # route the ADJOINT δ down (not a gate)
                    s2 = torch.einsum('bto,oi->bti', delta, self.W[l].double())         # Wᵀ hop (per-t)
                    if self.cfg.unit_rms_presyn:
                        s2 = self._rms_preserving(delta.reshape(B * T, -1),
                                                  s2.reshape(B * T, -1)).reshape(B, T, -1)
                    s = self._constrain(s2.reshape(B * T, -1)).reshape(B, T, -1)
                continue
            psi_g = (self._forward_influence(psi, self.cfg.influence_n, self.cfg.elig_alpha)
                     if self.cfg.influence_n > 0 else psi)        # (b) forward-influence gate Ψ̃ (n>0)
            if pt:
                g = psi_g * s                                     # (B,T,out_l) influence-gated per-t signal
                self.E[l] += eta[l] * torch.einsum('bto,bti->oi', g, eps) / (B * T)
            else:
                g = s                                             # (B,out_l) time-collapsed δ
                psibar = psi_g.mean(1)                            # (B,out_l) mean readiness
                self.E[l] += eta[l] * ((g * psibar).t() @ eps.mean(1)) / B
            self.E[l].clamp_(-self.cfg.e_clip, self.cfg.e_clip)
            if l > 0:                                             # route the signal down one layer
                gate = (psi_g * s) if (pr and pt) else (s if pt else (g * psi_g.mean(1)))
                if pt:
                    s2 = torch.einsum('bto,oi->bti', gate, self.W[l].double())     # Wᵀ hop (per-t)
                    if self.cfg.unit_rms_presyn:
                        s2 = self._rms_preserving(gate.reshape(B * T, -1),
                                                  s2.reshape(B * T, -1)).reshape(B, T, -1)
                    s = self._constrain(s2.reshape(B * T, -1)).reshape(B, T, -1)
                else:
                    s2 = gate @ self.W[l].double()
                    if self.cfg.unit_rms_presyn:
                        s2 = self._rms_preserving(gate, s2)
                    s = self._constrain(s2)
        # ⚗ EXPERIMENTAL input-tap credit (default OFF). Route δ_0 ONE more Wᵀ hop to the input node, then
        # correlate that input-error with the RAW lagged input — reuses the existing δ chain, no new channel.
        if self.cfg.tap_fold and tap_delta0 is not None and cap.get('X_raw') is not None:
            err_in = torch.einsum('bto,oi->bti', tap_delta0, self.W[0].double())   # (B,T,C) error at input
            Xr = cap['X_raw'].double()                                             # (B,T,C) pre-tap input
            Bn, Tn, _ = Xr.shape; K = self.cfg.tap_K
            for k in range(K):
                d = K - 1 - k                                                      # delay of tap k
                corr = ((err_in * Xr) if d == 0
                        else (err_in[:, d:, :] * Xr[:, :Tn - d, :])).sum(dim=(0, 1))
                self.E_tap[:, k] += self.cfg.boss_lr * corr / (Bn * Tn)
            self.E_tap.clamp_(-self.cfg.e_clip, self.cfg.e_clip)
        # ⚠ REDUNDANT after testing (ALIF demoted; see q_fold cfg banner). q-blend credit, default OFF.
        if self.cfg.q_fold and cap.get('DQ') is not None:
            DQ = cap['DQ']
            for l in range(L):
                if q_deltas[l] is None:
                    continue
                dq = DQ[l].double()
                Bn, Tn = dq.shape[0], dq.shape[1]
                self.E_q[l] += self.cfg.boss_lr * (q_deltas[l] * dq).sum(dim=(0, 1)) / (Bn * Tn)
                self.E_q[l].clamp_(-self.cfg.e_clip, self.cfg.e_clip)

    @torch.no_grad()
    def _fold(self):
        if self.cfg.var_norm_fold:
            self._fold_var_norm(); return
        if self.cfg.temporal and self.cfg.fold_mode in ("threshold", "sgd"):
            self._fold_temporal(); return
        eb, bl = self.cfg.e_bits, self.cfg.boss_lr
        damp = 1.0                                       # boss_h_damp: self-annealing global sign-step scale
        if self.cfg.boss_h_damp:
            agg = (self._bh_sum / self._bh_cnt) if self._bh_cnt > 0 else 0.0   # mean_bh over the fold window
            damp = max(self.cfg.boss_h_damp_floor, min(1.0, agg / self.cfg.boss_h_damp_div))
            self._bh_sum = 0.0; self._bh_cnt = 0
            self._last_damp = damp                       # exposed for probes / logging
        for l in range(len(self.W)):
            EL = self.E[l]
            if eb > 0:                                   # quantise E to e_bits vs its max
                sc = EL.abs().max()
                if sc > 1e-12:
                    L = 2 ** (eb - 1) - 1
                    EL = torch.round(EL / sc * L) / L * sc if eb > 1 else torch.sign(EL) * sc
            if self.cfg.per_neuron_gain:                 # Concept #2: per-output-neuron step
                row = self.E[l].abs().mean(dim=1)        # this-fold |E| per output neuron
                if not self._pn_init[l]:
                    self._pn_ms[l] = row.clone(); self._pn_init[l] = True
                else:
                    b = self.cfg.per_neuron_beta
                    self._pn_ms[l] = b * self._pn_ms[l] + (1 - b) * row
                gain = self._pn_ms[l] / self._pn_ms[l].mean().clamp_min(1e-12)   # mean-1
                gain = gain.clamp(max=self.cfg.per_neuron_cap)
                if self.cfg.sign_at_fold:
                    EL = torch.sign(EL) * (bl * damp) * gain[:, None]   # per-neuron adaptive sign step
                else:
                    EL = EL * gain[:, None]
            elif self.cfg.sign_at_fold:
                if self.cfg.stoch_fold:                  # DIAGNOSTIC: stochastic-rounded 1-bit write
                    s = EL.abs().mean().clamp_min(1e-12) * self.cfg.stoch_scale
                    p = (EL.abs() / s).clamp(0.0, 1.0)   # write prob ∝ |accumulated credit|
                    EL = torch.sign(EL) * (bl * damp) * (torch.rand_like(EL) < p).to(EL.dtype)
                else:
                    EL = torch.sign(EL) * (bl * damp)    # 1-bit write, step = boss_lr·damp
            self.vel[l] = self.cfg.fold_momentum * self.vel[l] + EL
            self.W[l] = (self.W[l].double() + self.vel[l]).float()
            if self.cfg.wclip is not None:
                self.W[l].clamp_(-self.cfg.wclip, self.cfg.wclip)
            self.E[l].zero_()
        if self.cfg.rule_readout:                            # fold the rule-trained readout jug
            # readout gets the SAME cold-start LR bootstrap (rr_lr_mult) under the sign write as it
            # does under var_norm — so a sign-vs-var_norm ablation isolates the WRITE RULE only.
            ELr = (torch.sign(self.E_ro) * bl if self.cfg.sign_at_fold else self.E_ro) * self.cfg.rr_lr_mult
            self.vel_ro = self.cfg.fold_momentum * self.vel_ro + ELr
            self.W_ro = (self.W_ro.double() + self.vel_ro).float()
            if self.cfg.wclip is not None:
                self.W_ro.clamp_(-self.cfg.wclip, self.cfg.wclip)
            self.E_ro.zero_()
        if self.cfg.tap_fold:                                # ⚗ fold the experimental tap jug (SAME sign rule)
            ELt = torch.sign(self.E_tap) * bl if self.cfg.sign_at_fold else self.E_tap
            self.vel_tap = self.cfg.fold_momentum * self.vel_tap + ELt
            self.W_tap = (self.W_tap.double() + self.vel_tap).float()
            if self.cfg.wclip is not None:
                self.W_tap.clamp_(-self.cfg.wclip, self.cfg.wclip)
            self.E_tap.zero_()
        if self.cfg.q_fold:                                  # ⚠ REDUNDANT (ALIF demoted); q blend sign fold
            for l in range(len(self.W)):
                ELq = (torch.sign(self.E_q[l]) * bl if self.cfg.sign_at_fold else self.E_q[l]) * self.cfg.q_lr_mult
                self.vel_q[l] = self.cfg.fold_momentum * self.vel_q[l] + ELq
                self.q_raw[l] = (self.q_raw[l] + self.vel_q[l]).clamp(-12.0, 12.0)   # keep sigmoid in range
                self.E_q[l].zero_()

    @torch.no_grad()
    def _fold_var_norm(self):
        """★★ RELIABILITY-GRADED (Adam-style) fold. Replaces sign(E)·lr with a per-synapse graded
        step m/√v: full step on CONSISTENT accumulated credit, ~0 on noisy credit — the one thing
        the sign-fold lacks vs Adam (uniform step size). Still bounded (|m/√v|≲1, clamped) so it is
        NOT raw-magnitude (which overfits). E = this fold-window's accumulated credit."""
        b1, b2, bl = self.cfg.vn_beta1, self.cfg.vn_beta2, self.cfg.boss_lr
        for l in range(len(self.W)):
            E = self.E[l].double()
            self._vn_m[l] = (b1 * self._vn_m[l].double() + (1 - b1) * E).to(self._vn_m[l].dtype)
            self._vn_v[l] = (b2 * self._vn_v[l].double() + (1 - b2) * E * E).to(self._vn_v[l].dtype)
            step = self._vn_m[l].double() / (self._vn_v[l].double().sqrt() + 1e-8)   # m/√v ∈ ~[-1,1]
            step = step.clamp(-self.cfg.vn_clip, self.cfg.vn_clip)                   # keep sign-bounded
            self.W[l] = (self.W[l].double() + bl * step).float()                    # graded step
            if self.cfg.wclip is not None:
                self.W[l].clamp_(-self.cfg.wclip, self.cfg.wclip)
            self.E[l].zero_()
        if self.cfg.rule_readout:                                                   # CO-ADAPT the readout too
            Er = self.E_ro.double()                                                 # (same graded m/√v step)
            self._vn_m_ro = (b1 * self._vn_m_ro.double() + (1 - b1) * Er).to(self._vn_m_ro.dtype)
            self._vn_v_ro = (b2 * self._vn_v_ro.double() + (1 - b2) * Er * Er).to(self._vn_v_ro.dtype)
            blr = bl * self.cfg.rr_lr_mult                                            # readout LR (bootstrap scale)
            stepr = (self._vn_m_ro.double() / (self._vn_v_ro.double().sqrt() + 1e-8)).clamp(
                -self.cfg.vn_clip, self.cfg.vn_clip)
            self.W_ro = (self.W_ro.double() + blr * stepr).float()
            if self.cfg.wclip is not None:
                self.W_ro.clamp_(-self.cfg.wclip, self.cfg.wclip)
            self.E_ro.zero_()
            Eb = self.E_b_ro.double()                                                # readout BIAS (graded step)
            self._vn_m_b = (b1 * self._vn_m_b.double() + (1 - b1) * Eb).to(self._vn_m_b.dtype)
            self._vn_v_b = (b2 * self._vn_v_b.double() + (1 - b2) * Eb * Eb).to(self._vn_v_b.dtype)
            stepb = (self._vn_m_b.double() / (self._vn_v_b.double().sqrt() + 1e-8)).clamp(
                -self.cfg.vn_clip, self.cfg.vn_clip)
            self.b_ro = self.b_ro + blr * stepb
            self.E_b_ro.zero_()
        if self.cfg.tap_fold:                                                    # ⚗ graded tap write (m/√v)
            Et = self.E_tap.double()
            self._vn_m_tap = (b1 * self._vn_m_tap.double() + (1 - b1) * Et).to(self._vn_m_tap.dtype)
            self._vn_v_tap = (b2 * self._vn_v_tap.double() + (1 - b2) * Et * Et).to(self._vn_v_tap.dtype)
            stept = (self._vn_m_tap.double() / (self._vn_v_tap.double().sqrt() + 1e-8)).clamp(
                -self.cfg.vn_clip, self.cfg.vn_clip)
            self.W_tap = (self.W_tap.double() + bl * stept).float()
            if self.cfg.wclip is not None:
                self.W_tap.clamp_(-self.cfg.wclip, self.cfg.wclip)
            self.E_tap.zero_()
        if self.cfg.q_fold:                                              # ⚠ REDUNDANT (ALIF demoted); graded q
            blq = bl * self.cfg.q_lr_mult
            for l in range(len(self.W)):
                Eq = self.E_q[l].double()
                self._vn_m_q[l] = (b1 * self._vn_m_q[l].double() + (1 - b1) * Eq).to(self._vn_m_q[l].dtype)
                self._vn_v_q[l] = (b2 * self._vn_v_q[l].double() + (1 - b2) * Eq * Eq).to(self._vn_v_q[l].dtype)
                stepq = (self._vn_m_q[l].double() / (self._vn_v_q[l].double().sqrt() + 1e-8)).clamp(
                    -self.cfg.vn_clip, self.cfg.vn_clip)
                self.q_raw[l] = (self.q_raw[l].double() + blq * stepq).float().clamp(-12.0, 12.0)
                self.E_q[l].zero_()

    @torch.no_grad()
    def _fold_temporal(self):
        """Temporal fold. 'sgd' = plain W += boss_lr·E (credit-only debug). 'threshold' = the
        robust SigmaDelta leaky-jug: per-output-neuron RMS gain (one θ fires everywhere) →
        leaky-accumulate into a persistent jug → fire a fixed ±LSB when |jug|>θ → subtract the
        charge (self-scaling, convergent). E is this fold-window's accumulated gradient (ascent)."""
        for l in range(len(self.W)):
            g = self.E[l]
            if self.cfg.fold_mode == "sgd":
                self.W[l] = (self.W[l].double() + self.cfg.boss_lr * g).float()
                if self.cfg.wclip is not None:
                    self.W[l].clamp_(-self.cfg.wclip, self.cfg.wclip)
                self.E[l].zero_(); continue
            # threshold-write leaky jug (SigmaDelta)
            row_ms = g.pow(2).mean(1, keepdim=True)                       # (out,1) per output neuron
            self._sd_ms[l].mul_(self.cfg.sd_beta_norm).add_((1 - self.cfg.sd_beta_norm) * row_ms)
            gn = g / (self._sd_ms[l].sqrt() + 1e-8)                       # each row → unit scale
            self._jug[l].mul_(self.cfg.sd_leak).add_(gn)                  # leaky accumulate (jug)
            fired = (self._jug[l].abs() > self.cfg.sd_theta).double()
            step = torch.sign(self._jug[l]) * fired
            self.W[l] = (self.W[l].double() + self.cfg.sd_lsb * step).float()   # ascent ±LSB
            self._jug[l].sub_(self.cfg.sd_theta * step)                   # error-feedback reset
            self.W[l].clamp_(-self.cfg.sd_wclip, self.cfg.sd_wclip)
            self.E[l].zero_()

    # ── feature extraction (temporal returns pooled feats via forward_fn(capture=False)) ──
    @torch.no_grad()
    def _all_feats(self, X, bs=512):
        """Pooled classifier features for the whole set. Temporal: batch the forward_fn WITHOUT
        capturing Z/PSI (eval-cheap). Static: the collapsed forward stack's top."""
        if not self.cfg.temporal:
            return self._forward(X)[-1]
        out = []
        for i in range(0, len(X), bs):
            out.append(self.forward_fn(self, X[i:i + bs], capture=False)['feats'])
        return torch.cat(out, 0)

    # ── public API ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def fit(self, X, y, epochs=8, bs=128, X_val=None, y_val=None,
            probe=False, probe_every=0, verbose=True):
        X = self._encode(X); y = y.to(self.device)
        if X_val is not None:
            X_val = self._encode(X_val); y_val = y_val.to(self.device)
        n = len(X); hist = []
        for ep in range(epochs):
            if (not self.cfg.rule_readout or self.cfg.rr_warm) and ep % self.cfg.clf_refit_every == 0:
                self._fit_clf(self._all_feats(X), y)          # strong logistic classifier
                if self.cfg.rule_readout:                     # WARM-START the co-adapting readout from it,
                    self.W_ro = self.W_f.clone()              # then the per-batch rule co-adapts it within
                    self.b_ro = self.b_f.clone()              # the epoch. (rr_warm=False ⇒ pure cold gradient)
            if self.cfg.auto_lr:                          # temporal calibration handled in _calibrate
                cal = torch.randperm(n)[:min(2000, n)]
                self._calibrate_auto_lr(X[cal], y[cal])
            perm = torch.randperm(n)
            self._acc_ct = 0; moves = 0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if self.cfg.temporal:
                    cap = self.forward_fn(self, X[idx], capture=True)
                    S, w = self._top_delta(self._scores(cap['feats']), y[idx], track_bh=True)
                    self._accumulate_temporal(cap, S, y[idx]); moves += w
                else:
                    Fs = self._forward(X[idx])
                    S, w = self._top_delta(self._scores(Fs[-1]), y[idx], track_bh=True)
                    self._accumulate(Fs, S, len(idx)); moves += w
                self._acc_ct += len(idx)
                if self._acc_ct >= self.cfg.fold_every:
                    self._fold(); self._acc_ct = 0
                    if probe and probe_every and (i // bs) % probe_every == 0:
                        self._probe(ep, i // bs)
            if self._acc_ct > 0:
                self._fold(); self._acc_ct = 0
            tr = self.score(X, y, encoded=True)
            va = self.score(X_val, y_val, encoded=True) if X_val is not None else float("nan")
            hist.append((ep, tr, va))
            if verbose:
                print(f"[pcn ep{ep:2d}] train={tr:.4f} val={va:.4f} "
                      f"moves/ep~{moves} fold={self.cfg.fold_mode if self.cfg.temporal else 'sign'}",
                      flush=True)
        return hist

    @torch.no_grad()
    def predict(self, X, encoded=False):
        if not encoded:
            X = self._encode(X)
        return self._scores(self._all_feats(X)).argmax(1)

    @torch.no_grad()
    def score(self, X, y, encoded=False):
        if X is None:
            return float("nan")
        return float((self.predict(X, encoded=encoded) == y.to(self.device)).float().mean())

    @torch.no_grad()
    def _probe(self, ep, b):
        print(f"    [pcn probe ep{ep} b{b}]", flush=True)
        for l in range(len(self.W)):
            e, w, v = self.E[l], self.W[l], self.vel[l]
            print(f"      L{l} E_rms={e.pow(2).mean().sqrt():.2e} "
                  f"vel_rms={v.pow(2).mean().sqrt():.2e} W_rms={w.pow(2).mean().sqrt():.3f} "
                  f"η={self.eta[l]:.2e}", flush=True)

    # ── Adam/backprop benchmark (same topology, autograd) ────────────────────
    def benchmark_adam(self, X, y, epochs=8, bs=128, lr=1e-3,
                       X_val=None, y_val=None, verbose=True):
        """Second opinion: identical dense topology + a linear head, trained with
        autograd + Adam + cross-entropy. Returns (epoch, train, val) history."""
        Xt = self._encode(X); yt = y.to(self.device)
        if X_val is not None:
            Xv = self._encode(X_val); yv = y_val.to(self.device)
        layers = []
        for i in range(len(self.dims) - 1):
            layers += [torch.nn.Linear(self.dims[i], self.dims[i + 1], bias=False),
                       torch.nn.LeakyReLU(self.a)]
        net = torch.nn.Sequential(*layers, torch.nn.Linear(self.dims[-1], self.n_classes))
        net.to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        n = len(Xt); hist = []
        for ep in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = F.cross_entropy(net(Xt[idx]), yt[idx])
                loss.backward(); opt.step()
            with torch.no_grad():
                tr = float((net(Xt).argmax(1) == yt).float().mean())
                va = float((net(Xv).argmax(1) == yv).float().mean()) if X_val is not None else float("nan")
            hist.append((ep, tr, va))
            if verbose:
                print(f"[adam ep{ep:2d}] train={tr:.4f} val={va:.4f}", flush=True)
        return hist
