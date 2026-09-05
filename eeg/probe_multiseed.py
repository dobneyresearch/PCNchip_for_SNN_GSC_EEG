#!/usr/bin/env python3
"""
DEFINITIVE multi-seed jug-vs-Adam sweep — pin down the effect + expected range (mean±std over 4 seeds).
Deterministic flags reduce GPU run-noise; 4 seeds give error bars. rate+tap, flat g(t) auto for the jug.
  Jug  : K=8, K=16   (40ep, cold_sign)
  Adam : K=8, K=16   (40ep, lr=1e-3, matched budget)
  Adam : K=16 @ lr=1e-4/60ep  (the historical "0.718" config — is it reproducible or lucky?)
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # must precede cuda init
import time, sys, statistics
import torch, torch.nn as nn, torch.nn.functional as F
torch.use_deterministic_algorithms(True, warn_only=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gsc"))
from dataclasses import replace
from pcn_learner_snn import PCNLearner
from eeg_mi_harness import load, make_forward_fn, DEV, spike, ALPHA, THR
from probe_jug_shallow import cfg_for
Xtr, Ytr, Xva, Yva = load(normalize=True); NCLS = int(Yva.max()) + 1
SEEDS = [0, 1, 2, 3]


class TapNet(nn.Module):
    def __init__(self, dims, K, seed=0):
        super().__init__(); g = torch.Generator().manual_seed(seed); self.C = dims[0]; self.K = K
        w = torch.zeros(self.C, 1, K); w[:, 0, -1] = 1.0; self.kern = nn.Parameter(w)
        self.W = nn.ParameterList()
        for i in range(len(dims)-1):
            wi = torch.empty(dims[i+1], dims[i]); nn.init.normal_(wi, 0.0, 1.0/dims[i]**0.5, generator=g); self.W.append(nn.Parameter(wi))
        self.head = nn.Linear(dims[-1], NCLS)
    def forward(self, X):
        xt = F.pad(X.transpose(1,2), (self.K-1,0)); X = F.conv1d(xt, self.kern, groups=self.C).transpose(1,2)
        B,T,_ = X.shape; v = [torch.zeros(B,w.shape[0],device=X.device) for w in self.W]; rs = torch.zeros(B,self.W[-1].shape[0],device=X.device)
        for t in range(T):
            z = X[:,t,:]
            for l,w in enumerate(self.W):
                v[l] = ALPHA*v[l] + z@w.t(); s = spike(v[l]-THR); v[l] = v[l]-s*THR; z = s
            rs = rs + z
        return self.head(rs/T)

@torch.no_grad()
def evl(net,X,y,bs=256):
    net.eval(); c=0
    for i in range(0,len(X),bs): c+=(net(X[i:i+bs].to(DEV)).argmax(1).cpu()==y[i:i+bs]).sum().item()
    return c/len(X)

def jug_run(K, seed):
    torch.manual_seed(seed)
    cfg = replace(cfg_for("cold_sign_lr10", seed), tap_fold=True, tap_K=K, readout_form="rate")
    lrn = PCNLearner([62,16], NCLS, cfg, device=DEV, forward_fn=make_forward_fn("rate", seed))
    h = lrn.fit(Xtr,Ytr,epochs=40,bs=64,X_val=Xva,y_val=Yva,verbose=False)
    return max(v for _,_,v in h)

def adam_run(K, seed, ep, lr):
    torch.manual_seed(seed)
    net = TapNet([62,16], K, seed).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=lr); n=len(Xtr); best=0
    for _ in range(ep):
        net.train(); perm = torch.randperm(n)
        for i in range(0,n,64):
            idx=perm[i:i+64]; loss=F.cross_entropy(net(Xtr[idx].to(DEV)),Ytr[idx].to(DEV)); opt.zero_grad(); loss.backward(); opt.step()
        best=max(best,evl(net,Xva,Yva))
    return best

CFG = [
    ("jug  K=8",        lambda s: jug_run(8, s)),
    ("jug  K=16",       lambda s: jug_run(16, s)),
    ("adam K=8  (40e/1e-3)",  lambda s: adam_run(8, s, 40, 1e-3)),
    ("adam K=16 (40e/1e-3)",  lambda s: adam_run(16, s, 40, 1e-3)),
    ("adam K=16 (60e/1e-4 *0.718cfg)", lambda s: adam_run(16, s, 60, 1e-4)),
]
res = {name: [] for name, _ in CFG}
print(f"[multiseed] seeds={SEEDS} deterministic  train={tuple(Xtr.shape)} dev={DEV}\n", flush=True)
for s in SEEDS:
    for name, fn in CFG:
        t0=time.time(); v=fn(s); res[name].append(v)
        print(f"  seed{s} {name:34} val={v:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    print("", flush=True)
print("=== MULTI-SEED SUMMARY (mean ± std, n=%d) ===" % len(SEEDS), flush=True)
for name,_ in CFG:
    xs=res[name]; print(f"  {name:34} {statistics.mean(xs):.4f} ± {statistics.pstdev(xs):.4f}  min={min(xs):.4f} max={max(xs):.4f}", flush=True)
print("########## MULTISEED DONE ##########", flush=True)
