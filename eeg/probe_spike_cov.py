"""
Experiment B — spike-native covariance. Build the spatial covariance MI needs FROM SPIKES, shallow
and SynOps-cheap (Track-2), and see if it matches the continuous square (energy net 0.710) — possibly
beating CSP, since coincidences give the FULL covariance, not CSP's few filters.

Exp A showed the deep LIF stack doesn't augment the square, so B is shallow + second-order-first:

    X (B,T,62) --spatial filter W0 (62->K, learned)--> proj --LIF spike--> S (B,T,K)
        diagonal      : rate_k = <S_k>            (per-filter spike rate = the crude/half-wave square)
        off-diagonal  : C_jk   = <S_j · S_k>      (pairwise COINCIDENCE = spike-domain cross-term)
    features -> BN -> linear head -> 2

⚠ Squaring a single spike train is degenerate (s²=s ⇒ rate). The non-degenerate second-order signal
lives in the OFF-diagonal (cross-channel coincidences) + the rate diagonal (a power proxy). Together
they are the spike-domain spatial covariance.

Ablations (Adam + surrogate BPTT through the LIF, same data/split):
  • rates    : diagonal only  (K feats)             — shallow spiking rate code, the crude square
  • coinc    : off-diagonal only (K(K-1)/2 feats)   — pure cross-channel coincidences
  • full     : rates ⊕ coinc                        — the spike-native covariance (the target)
  • cont_cov : continuous full covariance of proj    — upper bound (what spike-quantisation gives up)

K=16 ⇒ 16 rates + 120 coincidences = 136 feats. Comparators: energy(square) 0.710, CSP 0.675, SNN 0.667.
Run:  python3 probe_spike_cov.py --K 16 --epochs 60
"""
import argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from eeg_mi_harness import load, spike, DEV      # spike = SurrSpike.apply (slope 25)

THR = 1.0


def slog(x):                                      # signed-log, for the continuous covariance scale
    return torch.sign(x) * torch.log1p(x.abs())


class SpikeCovNet(nn.Module):
    def __init__(self, n_ch=62, K=16, n_cls=2, mode="full", alpha=0.9, seed=0):
        super().__init__()
        self.mode, self.alpha, self.K = mode, alpha, K
        g = torch.Generator().manual_seed(seed)
        w = torch.empty(K, n_ch); torch.nn.init.orthogonal_(w, generator=g)
        self.W0 = nn.Parameter(w)
        iu = torch.triu_indices(K, K, offset=1)
        self.register_buffer("iu0", iu[0]); self.register_buffer("iu1", iu[1])
        n_off = K * (K - 1) // 2
        d = {"rates": K, "coinc": n_off, "full": K + n_off, "cont_cov": K + n_off}[mode]
        self.bn = nn.BatchNorm1d(d)
        self.head = nn.Linear(d, n_cls)

    def forward(self, X):                          # X (B,T,62)
        proj = X @ self.W0.t()                      # (B,T,K) spatially-filtered signal
        if self.mode == "cont_cov":                 # continuous full covariance (upper bound)
            pc = proj - proj.mean(1, keepdim=True)
            C = torch.einsum("btj,btk->bjk", pc, pc) / proj.shape[1]
            diag = slog(torch.diagonal(C, dim1=1, dim2=2))       # (B,K) log-power
            off = slog(C[:, self.iu0, self.iu1])                 # (B,n_off) cross-cov
            feat = torch.cat([diag, off], 1)
        else:                                       # spike path
            B, T, K = proj.shape
            v = torch.zeros(B, K, device=proj.device)
            S = torch.empty(B, T, K, device=proj.device)
            for t in range(T):
                v = self.alpha * v + proj[:, t, :]
                z = spike(v - THR); v = v - z * THR
                S[:, t, :] = z
            rates = S.mean(1)                                    # (B,K) diagonal (rate = power proxy)
            Cc = torch.einsum("btj,btk->bjk", S, S) / T          # (B,K,K) coincidence matrix
            coinc = Cc[:, self.iu0, self.iu1]                    # (B,n_off) off-diagonal
            feat = {"rates": rates, "coinc": coinc,
                    "full": torch.cat([rates, coinc], 1)}[self.mode]
        return self.head(self.bn(feat))


@torch.no_grad()
def evaluate(net, X, y, bs=256):
    net.eval(); corr = 0
    for i in range(0, len(X), bs):
        corr += (net(X[i:i+bs].to(DEV)).argmax(1).cpu() == y[i:i+bs]).sum().item()
    return corr / len(X)


def run(mode, Xtr, Ytr, Xva, Yva, K, epochs, bs, lr, alpha, seed):
    net = SpikeCovNet(Xtr.shape[2], K, int(Yva.max()) + 1, mode, alpha, seed).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(Xtr); best = 0.0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            loss = F.cross_entropy(net(Xtr[idx].to(DEV)), Ytr[idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        best = max(best, evaluate(net, Xva, Yva))
        if ep % 15 == 0 or ep == epochs - 1:
            print(f"    [{mode} ep{ep:3d}] best={best:.4f}", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.9, help="LIF leak (sweep if coincidences flat)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", type=str, default="rates,coinc,full,cont_cov")
    args = ap.parse_args()

    Xtr, Ytr, Xva, Yva = load(normalize=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"[spike-cov] train={tuple(Xtr.shape)} K={args.K} alpha={args.alpha} modes={modes} dev={DEV}")
    print("  (comparators: energy/square 0.710, CSP 0.675, SNN 0.667)\n", flush=True)
    res = {}
    for m in modes:
        t0 = time.time()
        res[m] = run(m, Xtr, Ytr, Xva, Yva, args.K, args.epochs, args.bs, args.lr, args.alpha, args.seed)
        print(f"  >>> {m:9s} best val={res[m]:.4f}   ({time.time()-t0:.0f}s)\n", flush=True)
    print("=== SUMMARY (K={}) ===".format(args.K))
    for m in modes:
        print(f"  {m:9s} {res[m]:.4f}")


if __name__ == "__main__":
    main()
