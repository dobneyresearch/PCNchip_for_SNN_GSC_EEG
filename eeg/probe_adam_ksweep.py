#!/usr/bin/env python3
"""Adam envelope: rate+tap, shallow. CORRECTED 2026-08-25: lr=1e-3 is the TRUE 0.7182 provenance
(the old lr=1e-4 undertrained everything ⇒ the 'low Adam' scare). K∈{8,12,16,24}, 60ep default.
Usage: probe_adam_ksweep.py [--lr 1e-3] [--Ks 8,12,16,24] [--epochs 60]"""
import time, os, sys, argparse
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gsc"))
from eeg_mi_harness import load, spike, DEV, ALPHA, THR
DIMS=[62,16]; NCLS=2
ap=argparse.ArgumentParser()
ap.add_argument("--lr",type=float,default=1e-3)
ap.add_argument("--Ks",type=str,default="8,12,16,24")
ap.add_argument("--epochs",type=int,default=60)
args=ap.parse_args()
Xtr,Ytr,Xva,Yva=load(normalize=True)
class TapNet(nn.Module):
    def __init__(self,K,seed=0):
        super().__init__(); g=torch.Generator().manual_seed(seed); self.C=DIMS[0]; self.K=K
        w=torch.zeros(self.C,1,K); w[:,0,-1]=1.0; self.kern=nn.Parameter(w)
        w0=torch.empty(DIMS[1],DIMS[0]); nn.init.normal_(w0,0.0,1.0/DIMS[0]**0.5,generator=g); self.W0=nn.Parameter(w0)
        self.head=nn.Linear(DIMS[1],NCLS)
    def forward(self,X):
        xt=F.pad(X.transpose(1,2),(self.K-1,0)); X=F.conv1d(xt,self.kern,groups=self.C).transpose(1,2)
        B,T,_=X.shape; v=torch.zeros(B,DIMS[1],device=X.device); rs=torch.zeros(B,DIMS[1],device=X.device)
        for t in range(T):
            v=ALPHA*v+X[:,t,:]@self.W0.t(); z=spike(v-THR); v=v-z*THR; rs=rs+z
        return self.head(rs/T)
@torch.no_grad()
def evl(net,X,y,bs=256):
    net.eval(); c=0
    for i in range(0,len(X),bs): c+=(net(X[i:i+bs].to(DEV)).argmax(1).cpu()==y[i:i+bs]).sum().item()
    return c/len(X)
Ks=tuple(int(k) for k in args.Ks.split(","))
print(f"[adam-ksweep] rate+tap, {args.epochs}ep, lr={args.lr:g}  Ks={Ks}  train={tuple(Xtr.shape)} dev={DEV}\n",flush=True)
for K in Ks:
    net=TapNet(K,0).to(DEV); opt=torch.optim.Adam(net.parameters(),lr=args.lr); n=len(Xtr); best=0.0; t0=time.time()
    for ep in range(args.epochs):
        net.train(); perm=torch.randperm(n)
        for i in range(0,n,64):
            idx=perm[i:i+64]; loss=F.cross_entropy(net(Xtr[idx].to(DEV)),Ytr[idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        best=max(best,evl(net,Xva,Yva))
    print(f"  >>> ADAM K={K:2}  best val={best:.4f}  ({time.time()-t0:.0f}s)",flush=True)
print("\n########## ADAMKSWEEP DONE ##########",flush=True)
