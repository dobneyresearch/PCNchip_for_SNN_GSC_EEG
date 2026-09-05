"""
pcn_learner.py — Definitive forwards-only leaky-jug PCN learner (PyTorch, dense).

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
    var_norm_fold: bool = False       # GUARD TEST (2026-08-18): reliability-graded write m/√v instead of
                                      # sign(E)·lr — the temporal-module write, tested here on STATIC to check
                                      # it does NOT regress the validated EMNIST ~82%. Default OFF (sign unchanged).
    vn_beta1: float = 0.9
    vn_beta2: float = 0.99
    vn_clip: float = 1.0
    fold_momentum: float = 0.9        # velocity momentum μ
    fold_every: int = 128             # fold cadence in SAMPLES.
                                      # ⚠ T5 (revised): on the T1-fixed platform a LARGER
                                      # fold_every slightly HELPS (+1..2pp), not hurts.
    fold_mean: bool = True            # divide accumulation lr by fold_every (batch-decoupled)
    # steps
    auto_lr: bool = True              # per-layer η from measured RMS. ⚠ T8: recalibrated
                                      # PER EPOCH here (pcn_bigspec: once per STAGE). No
                                      # staging/stage_rollback here — dropped, superseded
                                      # by the leaky jug. No "frozen-after-first" variant.
    auto_frac: float = 2e-3           # target per-step output move fraction (dense-calibrated)
    boss_lr: float = 3e-4             # accumulation lr (auto_lr off) AND sign-at-fold step
    wclip: Optional[float] = None     # symmetric weight clip (None = off)
    # init
    seed: int = 0
    init_gain: float = 1.0


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
                 device: Optional[str] = None):
        """layer_dims: [in, h1, ..., feat] — the hidden stack producing classifier
        features. encoder: callable (N,T,D)->(N,in) for temporal input, or a string
        'rate'/'leaky', or None (static input fed straight in).
        NB: the SNN/temporal variant (pluggable forward, unit-RMS, per-neuron gain) lives
        in pcn_learner_snn.py — this module stays the clean, validated STATIC solution."""
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
        self.a = self.cfg.leaky_alpha
        g = torch.Generator().manual_seed(self.cfg.seed)
        self.W, self.E, self.vel = [], [], []
        for i in range(len(self.dims) - 1):
            d_in, d_out = self.dims[i], self.dims[i + 1]
            m = torch.empty(d_out, d_in)
            torch.nn.init.orthogonal_(m, gain=self.cfg.init_gain)   # random orthogonal (NOT GHA)
            self.W.append(m.to(self.device))
            self.E.append(torch.zeros(d_out, d_in, device=self.device))
            self.vel.append(torch.zeros(d_out, d_in, device=self.device))
        self.W_f = None
        self.b_f = None
        self.eta = [self.cfg.boss_lr] * len(self.W)
        self._acc_ct = 0

    # ── input / forward ──────────────────────────────────────────────────────
    def _encode(self, X):
        X = X.to(self.device).float()
        if X.dim() == 3:
            if self.encoder is None:
                raise ValueError("3-D (temporal) input needs an encoder")
            X = self.encoder(X)
        return X

    @torch.no_grad()
    def _forward(self, X):
        """Return activation stack F[0..L]; F[0]=input, F[-1]=classifier features."""
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
        return feats.double() @ self.W_f.t() + self.b_f

    # ── boss_h top-δ ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def _top_delta(self, scores, y):
        """Teaching error S, gated by perceptron stop-on-correct + boss_h severity.
        err_mode 'softmax': S = onehot − softmax(scores)  (scores must be LOGITS => logistic)
        err_mode 'mse':     S = onehot − scores           (scores are p̂ => lstsq; no exp)."""
        S = -scores.clone() if self.cfg.err_mode == "mse" else -F.softmax(scores, dim=1)
        S[torch.arange(len(y)), y] += 1.0
        pred = scores.argmax(1)
        wrong = pred != y
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

    # ── accumulate + fold ────────────────────────────────────────────────────
    @torch.no_grad()
    def _accumulate(self, Fs, S, bs):
        """Backproject S down the stack, accumulate outer products into E."""
        lr = self.eta if self.cfg.auto_lr else [self.cfg.boss_lr] * len(self.W)
        scale = (1.0 / self.cfg.fold_every) if (self.cfg.fold_mean and self.cfg.fold_every > 1) else 1.0
        D = (S @ self.W_f) * self._leaky_deriv(Fs[-1]).double()      # δ at top features
        D = self._constrain(D)
        for l in reversed(range(len(self.W))):
            self.E[l] += (lr[l] * scale) * (D.t() @ Fs[l].double())
            self.E[l].clamp_(-self.cfg.e_clip, self.cfg.e_clip)
            if l > 0:
                G = D @ self.W[l].double()
                D = self._constrain(G * self._leaky_deriv(Fs[l]).double())

    @torch.no_grad()
    def _fold(self):
        eb, bl = self.cfg.e_bits, self.cfg.boss_lr
        if self.cfg.var_norm_fold:                       # GUARD: graded write (Adam-style m/√v) vs sign
            if not hasattr(self, "_vn_m"):
                self._vn_m = [torch.zeros_like(w) for w in self.W]
                self._vn_v = [torch.zeros_like(w) for w in self.W]
            b1, b2 = self.cfg.vn_beta1, self.cfg.vn_beta2
            for l in range(len(self.W)):
                E = self.E[l].double()
                self._vn_m[l] = (b1 * self._vn_m[l].double() + (1 - b1) * E).to(self._vn_m[l].dtype)
                self._vn_v[l] = (b2 * self._vn_v[l].double() + (1 - b2) * E * E).to(self._vn_v[l].dtype)
                step = (self._vn_m[l].double() / (self._vn_v[l].double().sqrt() + 1e-8)).clamp(
                    -self.cfg.vn_clip, self.cfg.vn_clip)
                self.W[l] = (self.W[l].double() + bl * step).float()
                if self.cfg.wclip is not None:
                    self.W[l].clamp_(-self.cfg.wclip, self.cfg.wclip)
                self.E[l].zero_()
            return
        for l in range(len(self.W)):
            EL = self.E[l]
            if eb > 0:                                   # quantise E to e_bits vs its max
                sc = EL.abs().max()
                if sc > 1e-12:
                    L = 2 ** (eb - 1) - 1
                    EL = torch.round(EL / sc * L) / L * sc if eb > 1 else torch.sign(EL) * sc
            if self.cfg.sign_at_fold:
                EL = torch.sign(EL) * bl                 # 1-bit write, step = boss_lr
            self.vel[l] = self.cfg.fold_momentum * self.vel[l] + EL
            self.W[l] = (self.W[l].double() + self.vel[l]).float()
            if self.cfg.wclip is not None:
                self.W[l].clamp_(-self.cfg.wclip, self.cfg.wclip)
            self.E[l].zero_()

    # ── public API ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def fit(self, X, y, epochs=8, bs=128, X_val=None, y_val=None,
            probe=False, probe_every=0, verbose=True):
        X = self._encode(X); y = y.to(self.device)
        if X_val is not None:
            X_val = self._encode(X_val); y_val = y_val.to(self.device)
        n = len(X); hist = []
        for ep in range(epochs):
            if ep % self.cfg.clf_refit_every == 0:
                self._fit_clf(self._forward(X)[-1], y)
            if self.cfg.auto_lr:
                cal = torch.randperm(n)[:min(2000, n)]
                self._calibrate_auto_lr(X[cal], y[cal])
            perm = torch.randperm(n)
            self._acc_ct = 0; moves = 0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                Fs = self._forward(X[idx])
                S, w = self._top_delta(self._scores(Fs[-1]), y[idx])
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
                      f"moves/ep~{moves} η={[f'{e:.1e}' for e in self.eta]}", flush=True)
        return hist

    @torch.no_grad()
    def predict(self, X, encoded=False):
        if not encoded:
            X = self._encode(X)
        return self._scores(self._forward(X)[-1]).argmax(1)

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
