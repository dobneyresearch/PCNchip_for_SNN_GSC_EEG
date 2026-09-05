"""
GSC temporal harness for the PCN module (pcn_learner_snn.py).

The THIN, swappable SNN benchmark harness the module design calls for. ONE shared forward
(`SNNNet`) is driven two ways so they are a TRUE drop-in of each other (same forward/readout/data,
only the UPDATE differs):
  • --train jug   : the forwards-only PCN module drives SNNNet.run_hidden FROM learner.W (no grad),
                    gets feats + per-t captures (Z, PSI) for the temporal e-prop credit.
  • --train adam  : SNNNet trained end-to-end by Adam + surrogate-gradient BPTT (the comparator).
Readout switch (the A/B designs):
  • --readout rate  (B): pooled last-hidden firing rate -> head.  Simpler; may cap ~0.71.
  • --readout leaky (A): learned leaky-integrator readout, logits = mean membrane over T.  The
                         design known to reach ~0.856. [scaffolded — filled in the A step.]
Swap THIS file for another dataset/net; the rule/fold/jug stays in the module.
"""
import os, argparse, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn as nn
import torch.nn.functional as F
from pcn_learner_snn import PCNLearner, PCNConfig

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA, THR, SURR = 0.9, 1.0, 5.0        # membrane leak, threshold, surrogate slope (= train_alif)


class SurrSpike(torch.autograd.Function):
    """Hard threshold forward, surrogate gradient backward (= train_alif). For --train jug the
    forward is run under no_grad so backward is never used; for --train adam it carries BPTT."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1.0 + SURR * x.abs()) ** 2


spike = SurrSpike.apply


def pseudo_deriv(x):
    return 1.0 / (1.0 + SURR * x.abs()) ** 2


class SNNNet(nn.Module):
    """The ONE SNN forward. Hidden = bias-free LIF stack (β=0.9, θ=1). Readout switchable."""
    def __init__(self, dims=(20, 256, 256, 256), n_classes=35, readout="rate", seed=0):
        super().__init__()
        self.dims = list(dims); self.readout = readout
        g = torch.Generator().manual_seed(seed)
        self.W = nn.ParameterList()
        for i in range(len(self.dims) - 1):
            w = torch.empty(self.dims[i + 1], self.dims[i])
            torch.nn.init.normal_(w, 0.0, 1.0 / self.dims[i] ** 0.5, generator=g)  # 1/√fanin
            self.W.append(nn.Parameter(w))                 # (= train_alif; orthogonal under-drives
            #                                                 the tall input layer -> a silent net)
        # readout head (used by --train adam; the jug uses the module's fit W_f on feats instead)
        self.head = nn.Linear(self.dims[-1], n_classes)
        if readout == "leaky":
            self.ro_beta = ALPHA                            # leaky-integrator readout leak

    def run_hidden(self, X, capture=False):
        """Run the LIF hidden stack. Returns (feats, ro_feat, Z, PSI, RO_t).
        feats   = pooled last-hidden RATE (B, dims[-1])            — design B feature.
        ro_feat = leaky-integrated last-hidden feature (B, dims[-1]) — design A feature (recency-
                  weighted; = the readout membrane BEFORE the readout weight, since the readout is
                  linear so leaky∘Wro = Wro∘leaky). logits are then just head(ro_feat) / ro_feat@Wroᵀ.
        RO_t    = the RUNNING readout membrane r[t]=β r[t-1]+z_L per step (B,T,dims[-1]) — the per-step
                  running feature the jug's L1 running-delta reads (PREDICTIVE_DELTA_DESIGN.md §4).
                  ro_feat is just its time-mean; RO_t keeps every step. captured only if capture=True.
        Z/PSI captured per-t only if capture=True (jug path)."""
        B, T, _ = X.shape
        qv = getattr(self, "_q_vecs", None)                          # ⚗ per-layer LIF/ALIF blend (None ⇒ plain)
        v = [torch.zeros(B, w.shape[0], device=X.device) for w in self.W]
        va = [torch.zeros(B, w.shape[0], device=X.device) for w in self.W] if qv is not None else None
        ba = [torch.zeros(B, w.shape[0], device=X.device) for w in self.W] if qv is not None else None
        Z = [torch.zeros(B, T, w.shape[1], device=X.device) for w in self.W] if capture else None
        PSI = [torch.zeros(B, T, w.shape[0], device=X.device) for w in self.W] if capture else None
        DQ = ([torch.zeros(B, T, w.shape[0], device=X.device) for w in self.W]
              if (qv is not None and capture) else None)
        RO = torch.zeros(B, T, self.W[-1].shape[0], device=X.device) if capture else None
        ratesum = torch.zeros(B, self.W[-1].shape[0], device=X.device)
        rof_v = torch.zeros(B, self.W[-1].shape[0], device=X.device)     # leaky-integrated feature
        rof_sum = torch.zeros(B, self.W[-1].shape[0], device=X.device)
        beta = ALPHA if self.readout != "leaky" else self.ro_beta
        for t in range(T):
            z = X[:, t, :]
            for l, w in enumerate(self.W):
                pre = z
                drive = z @ w.t()
                v[l] = ALPHA * v[l] + drive
                vmthr = v[l] - THR
                s_lif = spike(vmthr)
                v[l] = v[l] - s_lif * THR
                if qv is not None:                                   # ⚗ parallel ALIF branch + blend
                    va[l] = ALPHA * va[l] + drive - self._q_wa * ba[l]
                    vamt = va[l] - THR
                    s_alif = spike(vamt)
                    va[l] = va[l] - s_alif * THR
                    ba[l] = self._q_rho * ba[l] + s_alif
                    q = qv[l]
                    zt = q * s_lif + (1.0 - q) * s_alif
                    if capture:
                        Z[l][:, t, :] = pre
                        PSI[l][:, t, :] = q * pseudo_deriv(vmthr) + (1.0 - q) * pseudo_deriv(vamt)
                        DQ[l][:, t, :] = s_lif - s_alif
                else:
                    zt = s_lif
                    if capture:
                        Z[l][:, t, :] = pre
                        PSI[l][:, t, :] = pseudo_deriv(vmthr)
                z = zt
            ratesum = ratesum + z
            rof_v = beta * rof_v + z                          # leaky integral of last-hidden spikes
            rof_sum = rof_sum + rof_v
            if capture:
                RO[:, t, :] = rof_v                           # r[t] — running readout membrane
        self._DQ_out = DQ                                     # ⚗ stash for make_forward_fn (None if no q)
        return ratesum / T, rof_sum / T, Z, PSI, RO

    def forward(self, X):
        """Adam path: differentiable logits via head on the design's feature."""
        feats, ro_feat, _, _, _ = self.run_hidden(X, capture=False)
        return self.head(ro_feat if self.readout == "leaky" else feats)


# ── jug path: the module's forward_fn drives SNNNet.run_hidden from learner.W ──────────────
def make_forward_fn(readout="rate", seed=0):
    net_box = {}

    @torch.no_grad()
    def forward_fn(learner, X, capture=False):
        learner._readout_form = readout      # ★ stamp g(t) form to MATCH this forward's readout (rate/leaky)
        X = X.to(learner.device).float()
        X_raw = X
        Wt = getattr(learner, "W_tap", None)            # ⚗ EXPERIMENTAL learnable input FIR (delay)
        if Wt is not None:                              # causal per-channel conv; identity at init
            C = X.shape[2]; K = Wt.shape[1]
            xt = F.pad(X.transpose(1, 2), (K - 1, 0))
            X = F.conv1d(xt, Wt.unsqueeze(1), groups=C).transpose(1, 2)
        if "net" not in net_box:
            net_box["net"] = SNNNet(learner.dims, learner.n_classes, readout, seed).to(learner.device)
        net = net_box["net"]
        for wp, wl in zip(net.W, learner.W):            # copy the rule-trained weights in
            wp.data.copy_(wl)
        qr = getattr(learner, "q_raw", None)            # ⚗ sync the learnable LIF/ALIF blend
        if qr is not None:
            net._q_vecs = [torch.sigmoid(q) for q in qr]
            net._q_rho = learner.cfg.q_alif_rho; net._q_wa = learner.cfg.q_alif_wa
        feats_rate, ro_feat, Z, PSI, RO = net.run_hidden(X, capture=capture)
        # design B uses the plain rate; design A uses the leaky-integrated feature (rule-trained W_ro)
        out = {"feats": ro_feat if readout == "leaky" else feats_rate}
        if capture:
            out["Z"] = Z; out["PSI"] = PSI
            if Wt is not None:
                out["X_raw"] = X_raw   # ⚗ pre-tap input for the forwards-only tap credit
            if getattr(net, "_DQ_out", None) is not None:
                out["DQ"] = net._DQ_out   # ⚗ per-layer (s_LIF−s_ALIF) for the q credit
            if readout == "leaky":
                out["RO_t"] = RO       # L1 running-delta reads the per-step running readout r[t]
        return out
    return forward_fn


def load(n, seed=0):
    tr = torch.load(os.path.join(CACHE, "gsc_training.pt"))
    te = torch.load(os.path.join(CACHE, "gsc_testing.pt"))
    Xtr, Ytr, Xte, Yte = tr["X"], tr["Y"], te["X"], te["Y"]
    if n and n < len(Xtr):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(Xtr), generator=g)[:n]
        Xtr, Ytr = Xtr[idx], Ytr[idx]
    return Xtr, Ytr, Xte, Yte


@torch.no_grad()
def _adam_eval(net, X, y, bs=500):
    corr = 0
    for i in range(0, len(X), bs):
        logits = net(X[i:i + bs].to(DEV).float())
        corr += (logits.argmax(1).cpu() == y[i:i + bs]).sum().item()
    return corr / len(X)


def train_adam(dims, ncls, Xtr, Ytr, Xte, Yte, epochs, bs, lr, readout, seed):
    """The COMPARATOR: SNNNet trained end-to-end by Adam + surrogate-grad BPTT (CE loss)."""
    net = SNNNet(dims, ncls, readout, seed).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(Xtr); best = 0.0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n)
        t0 = time.time()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logits = net(Xtr[idx].to(DEV).float())
            loss = F.cross_entropy(logits, Ytr[idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); va = _adam_eval(net, Xte, Yte)
        best = max(best, va)
        print(f"[adam {readout} ep{ep:2d}] val={va:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", choices=["jug", "adam"], default="jug")
    ap.add_argument("--readout", choices=["rate", "leaky"], default="rate")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--eval_n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3, help="adam lr")
    ap.add_argument("--fold", type=str, default="sign", choices=["sign", "threshold", "sgd"],
                    help="jug fold (sign = VALIDATED)")
    ap.add_argument("--no_per_t", action="store_true")
    ap.add_argument("--boss_lr", type=float, default=3e-4,
                    help="FAITHFUL default = 3e-4 (static default; the diagnostic-matrix value). "
                         "3e-3 was a silent-init-era artifact and oscillates to chance under sign-fold.")
    ap.add_argument("--rr_cold", action="store_true",
                    help="★ COLD GRADIENT READOUT (rule_readout with rr_warm=False) — train W_ro,b_ro "
                         "ONLY by their own gradient, no logistic fit. Implies --rule_readout. The write "
                         "rule is orthogonal: add --var_norm_fold for the graded (winning) write, omit it "
                         "for the sign write (ablation). WINNING RECIPE = --pl_buffer_n 8 --rr_cold "
                         "--var_norm_fold --rr_center --rr_lr_mult 100.")
    ap.add_argument("--rr_center", action="store_true", help="subtract running feature mean in readout gradient")
    ap.add_argument("--rr_lr_mult", type=float, default=1.0, help="cold readout LR multiplier (bootstrap scale)")
    ap.add_argument("--rule_readout", action="store_true",
                    help="design A jug: RULE-TRAIN the readout (default OFF = leaky feature + fit "
                         "logistic head, the stable drop-in)")
    ap.add_argument("--unit_rms", action="store_true",
                    help="use unit_rms_presyn (SNN-fork feature); default OFF = static-faithful")
    ap.add_argument("--no_auto_lr", action="store_true",
                    help="disable auto_lr; default auto_lr ON = static-faithful")
    ap.add_argument("--readout_jacobian", action="store_true",
                    help="B4: inject the leaky readout's temporal Jacobian g(t) into the per-t signal "
                         "(first-order rate↔leaky term; HW = one global gain on the error bus)")
    ap.add_argument("--running_delta", action="store_true",
                    help="L1 (PREDICTIVE_DELTA_DESIGN.md §4): use the leaky readout's per-step running "
                         "delta δ[t] instead of one pooled δ broadcast flat — time-varying causal credit")
    ap.add_argument("--pred_lead", type=float, default=0.0,
                    help="L2 (§5,§10b): on running δ[t], signal=δ[t]−pred_lead·δ[t−1]. 1.0 = pure "
                         "velocity (differences out the DC class-bias that collapses raw L1). Implies "
                         "--running_delta.")
    ap.add_argument("--influence_n", type=int, default=0,
                    help="(b) forward-influence gate: replace ψ[t] with the SnAp-n forward carry over "
                         "n steps (leak+reset). 0=jug; n≈10 covers the leaky readout's horizon.")
    ap.add_argument("--error_carry_n", type=int, default=0,
                    help="FORK (a): TRUE ERROR-SIDE BPTT carry via the causal buffer. Build the unrolled "
                         "adjoint δ[t]=Σ_k(Πα(1−θψ))·ψ[t+k]·e[t+k] (buffer reads t..t+n) and route the "
                         "adjoint (not a gate) down. 0=jug; the structurally-correct untried option.")
    ap.add_argument("--carry_theta", type=float, default=1.0,
                    help="θ in the error-carry reset α(1−θ·ψ). 1.0 = full reset (LIF surrogate-BPTT).")
    ap.add_argument("--buf_uniform", action="store_true",
                    help="HW-simplicity probe: uniform n-window credit (drop the reset-product decay in "
                         "the buffer) — the simpler elig_buffer.v cell. Default off = faithful leaky adjoint.")
    ap.add_argument("--pl_buffer_n", type=int, default=0,
                    help="★ Q2 PER-LAYER BUFFER: each layer's local n-step adjoint δ_l paired with DIRECT presyn "
                         "(the corrected credit; Adam-validated ~depth-indep, d≈8 matches BPTT). Tests survival "
                         "of the sign-fold. 0=off; try 8.")
    ap.add_argument("--delta_jug", action="store_true",
                    help="★★ CREDIT DELTA-JUG: SigmaDelta (leaky accumulator + threshold-fire) on the credit/time "
                         "axis, feeding the unchanged sign-fold E-jug. Coarse ±1 temporal credit — timing precise, "
                         "magnitude coarse (the biological/hardware regime). Tests if this beats jug 0.56 → Adam 0.72.")
    ap.add_argument("--dj_theta", type=float, default=1.0, help="delta-jug fire threshold (on RMS-norm'd input).")
    ap.add_argument("--tap", action="store_true",
                    help="⚗ EXPERIMENTAL learnable input FIR (delay), folded forwards-only (default OFF).")
    ap.add_argument("--tap_K", type=int, default=16, help="tap FIR window length (delay of tap k = tap_K-1-k).")
    ap.add_argument("--q_fold", action="store_true",
                    help="⚗ EXPERIMENTAL learnable LIF/ALIF blend q, folded forwards-only (default OFF).")
    ap.add_argument("--q_lr_mult", type=float, default=1.0, help="bootstrap LR for q (set from the LR sweep).")
    ap.add_argument("--var_norm_fold", action="store_true",
                    help="★★ RELIABILITY-GRADED fold: replace sign(E)·lr with Adam-style m/√v (full step on "
                         "consistent credit, ~0 on noisy). The one thing structurally missing vs Adam. Bounded "
                         "(not raw magnitude). Tests if reliability-grading is the 0.16 gap.")
    ap.add_argument("--fold_mult", type=int, default=1,
                    help="accumulate E over fold_mult BATCHES before the single sign-write "
                         "(fold_every = fold_mult·bs). Enabler for the buffer's small per-time signal.")
    ap.add_argument("--save_state", type=str, default="",
                    help="if best val > save_thresh, torch.save the learner state here (warm-start).")
    ap.add_argument("--save_thresh", type=float, default=0.5)
    args = ap.parse_args()


    Xtr, Ytr, Xte, Yte = load(args.n, args.seed)
    Xte, Yte = Xte[:args.eval_n], Yte[:args.eval_n]
    dims, ncls = [20, 256, 256, 256], 35
    print(f"[gsc-temporal] train={tuple(Xtr.shape)} eval={tuple(Xte.shape)} "
          f"TRAIN={args.train} READOUT={args.readout} dev={DEV}", flush=True)

    if args.train == "adam":
        best = train_adam(dims, ncls, Xtr, Ytr, Xte, Yte, args.epochs, args.bs,
                          args.lr, args.readout, args.seed)
        print(f"\n=== ADAM (BPTT) on SNN forward, readout={args.readout} ===", flush=True)
        print(f"  best val={best:.4f}   [design target A~0.86]", flush=True)
    else:
        cfg = PCNConfig(temporal=True, fold_mode=args.fold, per_t_route=not args.no_per_t,
                        boss_lr=args.boss_lr, fold_every=args.fold_mult * args.bs, seed=args.seed,
                        rule_readout=args.rule_readout or args.rr_cold,   # off = leaky feature + fit head
                        rr_warm=not args.rr_cold, rr_center=args.rr_center, rr_lr_mult=args.rr_lr_mult,  # ★ opt 1
                        init="snn",                              # 1/√fanin — SNN must spike at init
                        unit_rms_presyn=args.unit_rms,           # default OFF = static-faithful
                        auto_lr=not args.no_auto_lr,             # default ON  = static-faithful
                        readout_jacobian=args.readout_jacobian,  # B4: readout temporal Jacobian g(t)
                        readout_beta=ALPHA,                      # readout β = membrane leak
                        running_delta=args.running_delta or args.pred_lead > 0,  # L1: per-step δ[t]
                        pred_lead=args.pred_lead,                # L2: velocity lead δ[t]−a·δ[t−1]
                        influence_n=args.influence_n,            # (b): SnAp-n forward influence gate
                        error_carry_n=args.error_carry_n,        # FORK (a): true error-side BPTT carry
                        carry_theta=args.carry_theta,
                        pl_buffer_n=args.pl_buffer_n,           # ★ Q2 per-layer buffer (δ⊗direct presyn)
                        buf_uniform=args.buf_uniform,           # ⚗ HW-simplicity probe (uniform vs leaky adjoint)
                        delta_jug=args.delta_jug,               # ★★ credit delta-jug (SigmaDelta on credit)
                        dj_theta=args.dj_theta,
                        var_norm_fold=args.var_norm_fold,   # ★★ graded write — orthogonal to rr_cold (see help)
                        tap_fold=args.tap, tap_K=args.tap_K,   # ⚗ learnable input FIR (delay), default OFF
                        q_fold=args.q_fold, q_lr_mult=args.q_lr_mult)   # ⚗ learnable LIF/ALIF blend, default OFF
        learner = PCNLearner(dims, ncls, cfg=cfg, device=DEV,
                             forward_fn=make_forward_fn(args.readout, args.seed))
        hist = learner.fit(Xtr, Ytr, epochs=args.epochs, bs=args.bs, X_val=Xte, y_val=Yte)
        best = max(v for _, _, v in hist)
        print(f"\n=== PCN JUG (forwards-only) on SNN forward, readout={args.readout} ===", flush=True)
        print(f"  fold={args.fold} per_t={not args.no_per_t} influence_n={args.influence_n} "
              f"error_carry_n={args.error_carry_n} pl_buffer_n={args.pl_buffer_n} "
              f"delta_jug={args.delta_jug} fold_mult={args.fold_mult}  best val={best:.4f}", flush=True)
        if args.q_fold:                                          # ⚗ where did the LIF/ALIF blend settle?
            qm = [torch.sigmoid(q).mean().item() for q in learner.q_raw]
            print("  q̄ per layer = [" + ", ".join(f"{m:.3f}" for m in qm) +
                  "]  (→1 = LIF/off, <1 = ALIF/on)", flush=True)
        if args.save_state and best > args.save_thresh:      # warm-start checkpoint (speed only)
            torch.save({"W": [w.cpu() for w in learner.W], "W_f": learner.W_f.cpu(),
                        "b_f": learner.b_f.cpu(), "E": [e.cpu() for e in learner.E],
                        "vel": [v.cpu() for v in learner.vel], "best": best}, args.save_state)
            print(f"  [saved plateau state → {args.save_state} @ best {best:.3f}]", flush=True)


if __name__ == "__main__":
    main()
