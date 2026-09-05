"""
Adam SCREEN — does a LEARNABLE temporal tap-weighting (fixed-window delay) help EEG-MI?

The jug's credit buffer (pl_buffer_n=8, buf_uniform=True) is a flat, all-taps-on delay line. A learnable
delay = replace the flat weighting with a LEARNED kernel over the same fixed window. Forward-path analog:
a per-channel (depthwise) causal FIR over time, kernel length K, applied to the input before the shallow
LIF+rate net. A free FIR can express delay, smoothing OR differencing, so it also tells us WHICH temporal
op the signal wants. Modes:
  none      : no kernel (baseline = shallow rate net, 0.682)
  uniform   : FIXED boxcar K (is it just low-pass smoothing that helps? — the control)
  learned   : learnable per-channel FIR K=8, init = delta (identity ⇒ starts as 'none', then adapts)
  learned16 : learnable per-channel FIR K=16 (longer reach)
Threshold FIXED at 1.0, alpha FIXED at 0.9 (isolate the tap-weighting from the leak). Comparator-first Adam.
Run:  python3 probe_tapweight.py --modes none,uniform,learned,learned16
"""
import argparse, time, os, sys
if "--strict" in sys.argv:   # must precede torch import for cuBLAS determinism
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gsc"))
import torch
import torch.nn as nn
import torch.nn.functional as F
from eeg_mi_harness import load, spike, THR, ALPHA, DEV

DIMS, NCLS = [62, 16], 2


class TapNet(nn.Module):
    def __init__(self, mode, K=8, seed=0, het=False, lo=0.70, hi=0.98):
        super().__init__()
        self.mode, self.K, self.C = mode, K, DIMS[0]
        g = torch.Generator().manual_seed(seed)
        # OPTIONAL frozen per-neuron τ spread (heterogeneous τ) — test complementarity with taps
        self.het = het
        if het:
            self.register_buffer("alpha_vec", lo + (hi - lo) * torch.rand(DIMS[1], generator=g))
        if mode.startswith("learned"):
            w = torch.zeros(self.C, 1, K); w[:, 0, -1] = 1.0     # delta init = identity (causal: last tap = t)
            self.kern = nn.Parameter(w)
        elif mode == "uniform":
            self.register_buffer("kern", torch.full((self.C, 1, K), 1.0 / K))
        else:
            self.kern = None
        w0 = torch.empty(DIMS[1], DIMS[0]); torch.nn.init.normal_(w0, 0.0, 1.0 / DIMS[0] ** 0.5, generator=g)
        self.W0 = nn.Parameter(w0)
        self.head = nn.Linear(DIMS[1], NCLS)

    def temporal(self, X):
        if self.kern is None:
            return X
        xt = X.transpose(1, 2)                                   # (B,C,T)
        xt = F.pad(xt, (self.K - 1, 0))                          # causal: only look back
        xt = F.conv1d(xt, self.kern, groups=self.C)             # depthwise FIR over time
        return xt.transpose(1, 2)                                # (B,T,C)

    def forward(self, X):
        X = self.temporal(X)
        B, T, _ = X.shape
        v = torch.zeros(B, DIMS[1], device=X.device)
        ratesum = torch.zeros(B, DIMS[1], device=X.device)
        a = self.alpha_vec if self.het else ALPHA
        for t in range(T):
            v = a * v + X[:, t, :] @ self.W0.t()
            zt = spike(v - THR); v = v - zt * THR
            ratesum = ratesum + zt
        return self.head(ratesum / T)


@torch.no_grad()
def evl(net, X, y, bs=256):
    net.eval(); c = 0
    for i in range(0, len(X), bs):
        c += (net(X[i:i + bs].to(DEV)).argmax(1).cpu() == y[i:i + bs]).sum().item()
    return c / len(X)


def train(net, Xtr, Ytr, Xva, Yva, epochs, bs, lr, trace=None, tag=""):
    opt = torch.optim.Adam(net.parameters(), lr=lr); n = len(Xtr); best = 0.0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n); t0 = time.time()
        run_loss = 0.0; run_corr = 0; run_n = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            yb = Ytr[idx].to(DEV); out = net(Xtr[idx].to(DEV))
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item() * len(idx); run_n += len(idx)
            run_corr += (out.argmax(1) == yb).sum().item()
        va = evl(net, Xva, Yva); best = max(best, va)
        if trace is not None:
            trace.append((tag, ep, f"{run_loss/run_n:.6f}", f"{run_corr/run_n:.6f}",
                          f"{va:.6f}", f"{best:.6f}", f"{time.time()-t0:.1f}"))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", type=str, default="none,uniform,learned,learned16")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--het", action="store_true", help="frozen per-neuron τ spread (complementarity test)")
    ap.add_argument("--strict", action="store_true", help="deterministic GPU path (no TF32, seeded, cuDNN det.)")
    ap.add_argument("--csv", default=None, help="per-epoch trace path; 'auto' -> logs/tap_<modes>_...csv")
    args = ap.parse_args()
    if args.strict:
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        print(f"[strict] deterministic path ON (CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')})", flush=True)
    if args.csv == "auto":
        HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
        args.csv = os.path.join(HERE, "logs",
            f"tap_{args.modes.replace(',', '-')}_e{args.epochs}_s{args.seed}"
            f"{'_strict' if args.strict else ''}_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    Xtr, Ytr, Xva, Yva = load(normalize=True)
    print(f"[tapweight] dims={DIMS} rate-readout het={args.het}  train={tuple(Xtr.shape)} dev={DEV}")
    print("  (comparator: shallow rate net, no kernel = 0.682)\n", flush=True)
    res = {}; trace = [] if args.csv else None
    for m in [s.strip() for s in args.modes.split(",") if s.strip()]:
        t0 = time.time()
        K = 16 if m == "learned16" else 8
        torch.manual_seed(args.seed)          # ★ seed the fit shuffle per (mode,seed) ⇒ reproducible/comparable
        net = TapNet(m, K=K, seed=args.seed, het=args.het).to(DEV)
        res[m] = train(net, Xtr, Ytr, Xva, Yva, args.epochs, args.bs, args.lr, trace=trace, tag=m)
        print(f"  >>> {m:10s} best val={res[m]:.4f}  (K={K})  ({time.time()-t0:.0f}s)", flush=True)
    print("\n=== LEARNABLE TAP-WEIGHTS / DELAY (Adam) ===")
    for m, v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {m:10s} {v:.4f}")
    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["# " + " ".join(sys.argv)]); w.writerow(["# run " + time.strftime("%Y-%m-%d %H:%M:%S")])
            w.writerow(["mode", "epoch", "train_loss", "train_acc", "val_acc", "best_val", "sec"])
            w.writerows(trace)
        print(f"[csv] per-epoch trace -> {args.csv} ({len(trace)} rows)", flush=True)


if __name__ == "__main__":
    main()
