"""
THOR EEG-MI harness for the PCN module (pcn_learner_snn.py) — a THIRD benchmark.

Same drop-in discipline as the GSC harness: ONE shared SNN forward (`SNNNet`) driven two
ways so jug and Adam are a TRUE drop-in of each other (identical forward / readout / data,
only the UPDATE differs):
  • --train jug    : the forwards-only PCN module drives SNNNet.run_hidden FROM learner.W
                     (no grad), gets feats + per-t (Z, PSI) for the temporal credit.
  • --train adam   : the SAME SNNNet trained end-to-end by Adam + surrogate BPTT (comparator).
Plus:
  • --train official : the EXACT NeuroBench baseline net (62→256→128→2 spiking, count spikes)
                       to reproduce the published reference number (sanity / context).

Dataset: NeuroBench `ThorEEGMI` (preprocessed Lee2019 OpenBMI motor-imagery; 4 .npy from HF).
  train (7344, 250, 62), val (1728, 250, 62); 2 balanced classes (0=right,1=left); continuous
  8–30 Hz EEG amplitudes (zero-mean, std≈5.5, heavy-tailed) fed into the first FC (analog-in,
  spiking hidden) — matches the baseline. We z-score per channel (train stats) by default so the
  operating point is sane; the SAME normalisation is applied to jug, our-adam AND official.

The VALIDATED learner (`pcn_learner_snn.py`) is imported from ../gsc — single-sourced,
NOT copied (a per-dataset fork rots; [[feedback_pcn_module_first]]). New behaviour, if ever needed,
goes into that module as a default-OFF flag, not here.

Winning GSC recipe carried over as the starting point (--readout leaky --pl_buffer_n 8 --rr_cold
--var_norm_fold --rr_center --rr_lr_mult 100 --buf_uniform), but EEG is a DIFFERENT signal class
(mu/beta band POWER, ~epoch-static, not delta-spike timing) so the readout/credit knobs are open
questions to sweep — establish Adam first (--train adam), then the jug drop-in.
"""
import os, sys, argparse, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if "--strict" in sys.argv:   # must be set BEFORE torch import for cuBLAS determinism
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gsc"))   # single-sourced validated module
from pcn_learner_snn import PCNLearner, PCNConfig
from neurobench.datasets import ThorEEGMI

DATA = os.path.join(HERE, "data")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA, THR, SURR = 0.9, 1.0, 25.0        # LIF leak, threshold, surrogate slope (= baseline fast_sigmoid(25))


class SurrSpike(torch.autograd.Function):
    """Hard-threshold forward, fast-sigmoid surrogate backward (slope SURR). Under --train jug
    the forward runs no_grad (backward unused); under --train adam it carries surrogate BPTT."""
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
    """The ONE shared SNN forward. Hidden = bias-free LIF stack (β=0.9, θ=1). Readout switchable
    (rate = pooled spike count; leaky = leaky-integrated feature). dims = hidden stack widths, the
    readout head maps dims[-1] → n_classes. Mirrors the NeuroBench EEG_SNN hidden path."""
    def __init__(self, dims=(62, 256, 128), n_classes=2, readout="rate", seed=0, alpha_vecs=None,
                 thr_vecs=None):
        super().__init__()
        self.dims = list(dims); self.readout = readout
        g = torch.Generator().manual_seed(seed)
        self.W = nn.ParameterList()
        for i in range(len(self.dims) - 1):
            w = torch.empty(self.dims[i + 1], self.dims[i])
            torch.nn.init.normal_(w, 0.0, 1.0 / self.dims[i] ** 0.5, generator=g)   # 1/√fanin
            self.W.append(nn.Parameter(w))
        self.head = nn.Linear(self.dims[-1], n_classes)
        if readout == "leaky":
            self.ro_beta = ALPHA
        # OPTIONAL frozen per-neuron leak (heterogeneous τ). None ⇒ scalar ALPHA (default, unchanged).
        self._has_het = alpha_vecs is not None
        if self._has_het:
            for l, av in enumerate(alpha_vecs):
                self.register_buffer(f"alpha_{l}", av)
        # OPTIONAL frozen per-neuron threshold (device mismatch). None ⇒ scalar THR (default, unchanged).
        self._has_thr = thr_vecs is not None
        if self._has_thr:
            for l, tv in enumerate(thr_vecs):
                self.register_buffer(f"thr_{l}", tv)

    def run_hidden(self, X, capture=False):
        """Run the LIF hidden stack. Returns (feats_rate, ro_feat, Z, PSI, RO_t). ⚗ If self._q_vecs is set
        (q-blend experiment) each layer runs a parallel ALIF branch and outputs q·s_LIF+(1−q)·s_ALIF; the
        per-layer (s_LIF−s_ALIF) is stashed on self._DQ_out for the forwards-only q credit."""
        B, T, _ = X.shape
        qv = getattr(self, "_q_vecs", None)                          # ⚗ per-layer blend q (None ⇒ plain LIF)
        v = [torch.zeros(B, w.shape[0], device=X.device) for w in self.W]
        va = [torch.zeros(B, w.shape[0], device=X.device) for w in self.W] if qv is not None else None
        ba = [torch.zeros(B, w.shape[0], device=X.device) for w in self.W] if qv is not None else None
        Z = [torch.zeros(B, T, w.shape[1], device=X.device) for w in self.W] if capture else None
        PSI = [torch.zeros(B, T, w.shape[0], device=X.device) for w in self.W] if capture else None
        VDR = [torch.zeros(B, T, w.shape[0], device=X.device) for w in self.W] if capture else None  # ⚗ v/θ drive
        DQ = ([torch.zeros(B, T, w.shape[0], device=X.device) for w in self.W]
              if (qv is not None and capture) else None)
        RO = torch.zeros(B, T, self.W[-1].shape[0], device=X.device) if capture else None
        ratesum = torch.zeros(B, self.W[-1].shape[0], device=X.device)
        rof_v = torch.zeros(B, self.W[-1].shape[0], device=X.device)
        rof_sum = torch.zeros(B, self.W[-1].shape[0], device=X.device)
        beta = ALPHA if self.readout != "leaky" else self.ro_beta
        for t in range(T):
            z = X[:, t, :]
            for l, w in enumerate(self.W):
                pre = z
                a = getattr(self, f"alpha_{l}") if self._has_het else ALPHA   # per-neuron τ if het
                th = getattr(self, f"thr_{l}") if self._has_thr else THR      # per-neuron θ if mismatch
                drive = z @ w.t()
                v[l] = a * v[l] + drive
                vmthr = v[l] - th
                if VDR is not None:
                    VDR[l][:, t, :] = v[l] / THR                  # ⚗ pre-reset drive v/θ (amplitude-derivative)
                s_lif = spike(vmthr)
                v[l] = v[l] - s_lif * th
                if qv is not None:                                   # ⚗ parallel ALIF branch + blend
                    va[l] = a * va[l] + drive - self._q_wa * ba[l]
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
            rof_v = beta * rof_v + z
            rof_sum = rof_sum + rof_v
            if capture:
                RO[:, t, :] = rof_v
        self._DQ_out = DQ                                            # ⚗ stash for make_forward_fn (None if no q)
        self._VDR_out = VDR                                          # ⚗ stash v/θ drive (None if not capturing)
        return ratesum / T, rof_sum / T, Z, PSI, RO

    def forward(self, X):
        feats, ro_feat, _, _, _ = self.run_hidden(X, capture=False)
        return self.head(ro_feat if self.readout == "leaky" else feats)


def make_forward_fn(readout="rate", seed=0, het=False, het_lo=0.70, het_hi=0.98,
                    w_sigma=0.0, thr_sigma=0.0, alpha_sigma=0.0, mism_seed=0):
    """w_sigma/thr_sigma/alpha_sigma = FROZEN per-device relative mismatch (Gaussian σ, sample-and-freeze)
    modelling an analog/printed substrate. Weights ×(1+N(0,w_sigma)) per synapse; per-neuron θ×(1+N(0,
    thr_sigma)); per-neuron leak α×(1+N(0,alpha_sigma)). Frozen ⇒ the jug TRAINS on the mismatched
    substrate and can learn around it. All 0 ⇒ default forward unchanged. (`probe_variability.py`.)"""
    net_box = {}

    @torch.no_grad()
    def forward_fn(learner, X, capture=False):
        learner._readout_form = readout      # ★ stamp g(t) form to MATCH this forward's readout (rate/leaky)
        X = X.to(learner.device).float()
        X_raw = X
        Wt = getattr(learner, "W_tap", None)          # ⚗ EXPERIMENTAL learnable input FIR (delay)
        if Wt is not None:                            # causal per-channel conv; identity at init
            C = X.shape[2]; K = Wt.shape[1]
            xt = F.pad(X.transpose(1, 2), (K - 1, 0))
            X = F.conv1d(xt, Wt.unsqueeze(1), groups=C).transpose(1, 2)
        if "net" not in net_box:
            gm = torch.Generator().manual_seed(seed + 777 + mism_seed)   # frozen device-mismatch draws
            L = len(learner.dims) - 1
            av = None
            if alpha_sigma > 0:                       # per-neuron leak mismatch (Gaussian around ALPHA)
                av = [(ALPHA * (1 + alpha_sigma * torch.randn(learner.dims[i+1], generator=gm))).clamp(0.01, 0.995)
                      for i in range(L)]
            elif het:                                 # (legacy) uniform τ spread
                av = [het_lo + (het_hi - het_lo) * torch.rand(learner.dims[i+1], generator=gm) for i in range(L)]
            tv = None
            if thr_sigma > 0:                         # per-neuron threshold mismatch
                tv = [(THR * (1 + thr_sigma * torch.randn(learner.dims[i+1], generator=gm))).clamp(0.1, 5.0)
                      for i in range(L)]
            net_box["net"] = SNNNet(learner.dims, learner.n_classes, readout, seed,
                                    alpha_vecs=av, thr_vecs=tv).to(learner.device)
            net_box["wmask"] = None
            if w_sigma > 0:                           # frozen per-synapse weight mismatch (multiplicative)
                net_box["wmask"] = [(1 + w_sigma * torch.randn(learner.dims[i+1], learner.dims[i], generator=gm)
                                     ).to(learner.device) for i in range(L)]
        net = net_box["net"]
        for wp, wl in zip(net.W, learner.W):
            wp.data.copy_(wl)
        if net_box.get("wmask") is not None:          # effective HW weight = learned ⊙ frozen mismatch
            for wp, m in zip(net.W, net_box["wmask"]):
                wp.data.mul_(m)
        qr = getattr(learner, "q_raw", None)          # ⚗ sync the learnable LIF/ALIF blend
        if qr is not None:
            net._q_vecs = [torch.sigmoid(q) for q in qr]
            net._q_rho = learner.cfg.q_alif_rho; net._q_wa = learner.cfg.q_alif_wa
        feats_rate, ro_feat, Z, PSI, RO = net.run_hidden(X, capture=capture)
        out = {"feats": ro_feat if readout == "leaky" else feats_rate}
        if capture:
            out["Z"] = Z; out["PSI"] = PSI
            if Wt is not None:
                out["X_raw"] = X_raw                   # ⚗ pre-tap input for the forwards-only tap credit
            if getattr(net, "_DQ_out", None) is not None:
                out["DQ"] = net._DQ_out               # ⚗ per-layer (s_LIF−s_ALIF) for the q credit
            if getattr(net, "_VDR_out", None) is not None:
                out["VDR"] = net._VDR_out             # ⚗ per-layer v/θ drive for the amplitude-derivative
            if readout == "leaky":
                out["RO_t"] = RO
        return out
    return forward_fn


# ── official NeuroBench baseline net (exact), for the reference number ──────────────────────
class EEG_SNN_official(nn.Module):
    """The exact NeuroBench eeg_mi baseline: 62→256→128→2, snn-style LIF, count output spikes."""
    def __init__(self, n_inputs=62, n_hidden=256, n_outputs=2, beta=0.9):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden);      self.b1 = beta
        self.fc2 = nn.Linear(n_hidden, n_hidden // 2); self.b2 = beta
        self.fc3 = nn.Linear(n_hidden // 2, n_outputs); self.b3 = beta

    def forward(self, X):
        B, T, _ = X.shape
        m1 = torch.zeros(B, self.fc1.out_features, device=X.device)
        m2 = torch.zeros(B, self.fc2.out_features, device=X.device)
        m3 = torch.zeros(B, self.fc3.out_features, device=X.device)
        out = []
        for t in range(T):
            m1 = self.b1 * m1 + self.fc1(X[:, t, :]); s1 = spike(m1 - THR); m1 = m1 - s1 * THR
            m2 = self.b2 * m2 + self.fc2(s1);         s2 = spike(m2 - THR); m2 = m2 - s2 * THR
            m3 = self.b3 * m3 + self.fc3(s2);         s3 = spike(m3 - THR); m3 = m3 - s3 * THR
            out.append(s3)
        return torch.stack(out, 1)      # (B,T,2)


# ── data ────────────────────────────────────────────────────────────────────────────────────
def load(normalize=True):
    tr = ThorEEGMI(root=DATA, split="train", download=True)
    va = ThorEEGMI(root=DATA, split="val", download=True)
    Xtr, Ytr = tr.data.float(), tr.targets.long()
    Xva, Yva = va.data.float(), va.targets.long()
    if normalize:                                    # per-channel z-score on TRAIN stats (both splits)
        mu = Xtr.mean(dim=(0, 1), keepdim=True)
        sd = Xtr.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        Xtr = (Xtr - mu) / sd
        Xva = (Xva - mu) / sd
    return Xtr, Ytr, Xva, Yva


@torch.no_grad()
def _eval(net, X, y, bs=256):
    net.eval(); corr = 0
    for i in range(0, len(X), bs):
        out = net(X[i:i + bs].to(DEV)).sum(1) if isinstance(net, EEG_SNN_official) else net(X[i:i + bs].to(DEV))
        corr += (out.argmax(1).cpu() == y[i:i + bs]).sum().item()
    return corr / len(X)


def _write_csv(path, header, rows):
    import csv as _csv
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["# " + " ".join(sys.argv)])                        # exact command (provenance)
        w.writerow(["# run " + time.strftime("%Y-%m-%d %H:%M:%S")])    # date-time
        w.writerow(header)
        w.writerows(rows)
    print(f"[csv] per-epoch trace -> {path} ({len(rows)} epochs)", flush=True)


def train_adam(Xtr, Ytr, Xva, Yva, dims, ncls, epochs, bs, lr, readout, seed, official=False, csv=None):
    if official:
        net = EEG_SNN_official(dims[0], 256, ncls).to(DEV); tag = "adam-official"
    else:
        net = SNNNet(dims, ncls, readout, seed).to(DEV); tag = f"adam-{readout}"
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(Xtr); best = 0.0; rows = []
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n); t0 = time.time()
        run_loss = 0.0; run_corr = 0; run_n = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            out = net(Xtr[idx].to(DEV))
            if official:
                out = out.sum(1)
            yb = Ytr[idx].to(DEV)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item() * len(idx); run_n += len(idx)
            run_corr += (out.argmax(1) == yb).sum().item()
        tr_loss = run_loss / run_n; tr_acc = run_corr / run_n
        va = _eval(net, Xva, Yva); best = max(best, va); sec = time.time() - t0
        print(f"[{tag} ep{ep:3d}] train_acc={tr_acc:.4f} loss={tr_loss:.4f} "
              f"val={va:.4f} best={best:.4f} ({sec:.0f}s)", flush=True)
        rows.append((ep, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{va:.6f}", f"{best:.6f}", f"{sec:.1f}"))
    if csv:
        _write_csv(csv, ["epoch", "train_loss", "train_acc", "val_acc", "best_val", "sec"], rows)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", choices=["jug", "adam", "official"], default="adam")
    ap.add_argument("--readout", choices=["rate", "leaky"], default="rate")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4, help="adam lr (baseline uses 1e-4)")
    ap.add_argument("--no_normalize", action="store_true", help="skip per-channel z-score (raw EEG)")
    # jug recipe knobs (start from the GSC winner; EEG is a different signal ⇒ sweep)
    ap.add_argument("--boss_lr", type=float, default=3e-4)
    ap.add_argument("--pl_buffer_n", type=int, default=8)
    ap.add_argument("--buf_uniform", action="store_true")
    ap.add_argument("--rr_cold", action="store_true")
    ap.add_argument("--rr_center", action="store_true")
    ap.add_argument("--rr_lr_mult", type=float, default=100.0)
    ap.add_argument("--var_norm_fold", action="store_true")
    ap.add_argument("--unit_rms", action="store_true")
    ap.add_argument("--no_auto_lr", action="store_true")
    ap.add_argument("--fold_mult", type=int, default=1, help="fold_every = fold_mult·bs")
    ap.add_argument("--strict", action="store_true",
                    help="force the correct/slower deterministic GPU path (no TF32, no cuDNN autotune, "
                         "deterministic kernels, seeded shuffle) — removes run-to-run GPU variance")
    ap.add_argument("--csv", default=None,
                    help="write per-epoch trace (train_loss/train_acc/val_acc/best_val/sec) to this path; "
                         "'auto' names it logs/<train><readout>_s<seed>_<timestamp>.csv (learning-curve ready)")
    args = ap.parse_args()

    if args.csv == "auto":
        os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
        args.csv = os.path.join(HERE, "logs",
            f"{args.train}_{args.readout}_e{args.epochs}_s{args.seed}"
            f"{'_strict' if args.strict else ''}_{time.strftime('%Y%m%d-%H%M%S')}.csv")

    if args.strict:
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[strict] deterministic path ON (no TF32, cuDNN deterministic, seeded, "
              f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')})", flush=True)

    Xtr, Ytr, Xva, Yva = load(normalize=not args.no_normalize)
    dims, ncls = [62, 256, 128], 2
    print(f"[thor-eeg] train={tuple(Xtr.shape)} val={tuple(Xva.shape)} TRAIN={args.train} "
          f"READOUT={args.readout} norm={not args.no_normalize} dev={DEV}", flush=True)

    if args.train in ("adam", "official"):
        best = train_adam(Xtr, Ytr, Xva, Yva, dims, ncls, args.epochs, args.bs, args.lr,
                          args.readout, args.seed, official=(args.train == "official"), csv=args.csv)
        print(f"\n=== {'OFFICIAL BASELINE' if args.train=='official' else 'ADAM (BPTT)'} "
              f"readout={args.readout} === best val={best:.4f}", flush=True)
    else:
        cfg = PCNConfig(temporal=True, fold_mode="sign", per_t_route=True, boss_lr=args.boss_lr,
                        fold_every=args.fold_mult * args.bs, seed=args.seed,
                        rule_readout=args.rr_cold, rr_warm=not args.rr_cold, rr_center=args.rr_center,
                        rr_lr_mult=args.rr_lr_mult, init="snn", unit_rms_presyn=args.unit_rms,
                        auto_lr=not args.no_auto_lr, readout_beta=ALPHA, pl_buffer_n=args.pl_buffer_n,
                        carry_theta=1.0, buf_uniform=args.buf_uniform, var_norm_fold=args.var_norm_fold)
        lrn = PCNLearner(dims, ncls, cfg=cfg, device=DEV,
                         forward_fn=make_forward_fn(args.readout, args.seed))
        hist = lrn.fit(Xtr, Ytr, epochs=args.epochs, bs=args.bs, X_val=Xva, y_val=Yva)
        best = max(v for _, _, v in hist)
        if args.csv:
            run = 0.0; rows = []
            for (ep, tr, va) in hist:
                run = max(run, va)
                rows.append((ep, f"{tr:.6f}", f"{va:.6f}", f"{run:.6f}"))
            _write_csv(args.csv, ["epoch", "train_acc", "val_acc", "best_val"], rows)
        print(f"\n=== PCN JUG (forwards-only) readout={args.readout} pl_buffer_n={args.pl_buffer_n} "
              f"var_norm={args.var_norm_fold} rr_cold={args.rr_cold} === best val={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
