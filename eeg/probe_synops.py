"""
Track-2 accounting — SynOps + footprint of the official deep baseline vs our shallow spike-rates net.

NeuroBench convention: a connection driven by a CONTINUOUS input costs a dense MAC per application;
a connection driven by SPIKES costs one AC per actual presynaptic spike (measured at the trained
firing rate). Footprint = weight count. Per SAMPLE (T=250 timesteps).

  deep (official 62→256→128→2, all LIF):
     fc1 62→256   continuous EEG in  → MACs = 62·256·T
     fc2 256→128  spikes in          → ACs  = (layer-1 spikes/sample)·128
     fc3 128→2    spikes in          → ACs  = (layer-2 spikes/sample)·2
  shallow (rates 62→16 LIF → rate → head 16→2):
     W0  62→16    continuous EEG in  → MACs = 62·16·T
     head 16→2    pooled rates in    → MACs = 16·2   (once/sample)

Firing rates measured on val after a short train (sparsity is stable early). Accuracies from RESULTS.md
(official 0.653, shallow-rates 0.682).
"""
import time
import torch
import torch.nn.functional as F
from eeg_mi_harness import load, EEG_SNN_official, spike, DEV, ALPHA, THR
from probe_spike_cov import SpikeCovNet

T = 250


def train_deep(Xtr, Ytr, epochs, bs=64, lr=1e-4):
    net = EEG_SNN_official(62, 256, 2).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr); n = len(Xtr)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            loss = F.cross_entropy(net(Xtr[idx].to(DEV)).sum(1), Ytr[idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def train_shallow(Xtr, Ytr, epochs, bs=64, lr=1e-3):
    net = SpikeCovNet(62, 16, 2, mode="rates").to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr); n = len(Xtr)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            loss = F.cross_entropy(net(Xtr[idx].to(DEV)), Ytr[idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def deep_spikes(net, X, bs=256):
    """Avg per-sample total spikes in layer-1 (256) and layer-2 (128)."""
    net.eval(); s1t = s2t = 0.0; N = len(X)
    for i in range(0, N, bs):
        xb = X[i:i+bs].to(DEV); B = xb.shape[0]
        m1 = torch.zeros(B, 256, device=DEV); m2 = torch.zeros(B, 128, device=DEV); m3 = torch.zeros(B, 2, device=DEV)
        for t in range(T):
            m1 = net.b1*m1 + net.fc1(xb[:, t]); z1 = spike(m1-THR); m1 = m1 - z1*THR
            m2 = net.b2*m2 + net.fc2(z1);       z2 = spike(m2-THR); m2 = m2 - z2*THR
            m3 = net.b3*m3 + net.fc3(z2);       z3 = spike(m3-THR); m3 = m3 - z3*THR
            s1t += z1.sum().item(); s2t += z2.sum().item()
    return s1t/N, s2t/N


@torch.no_grad()
def shallow_spikes(net, X, bs=256):
    net.eval(); st = 0.0; N = len(X)
    for i in range(0, N, bs):
        xb = X[i:i+bs].to(DEV); B = xb.shape[0]
        proj = xb @ net.W0.t(); v = torch.zeros(B, net.K, device=DEV)
        for t in range(T):
            v = net.alpha*v + proj[:, t]; z = spike(v-THR); v = v - z*THR
            st += z.sum().item()
    return st/N


def main():
    Xtr, Ytr, Xva, Yva = load(normalize=True)
    t0 = time.time()
    print("[synops] training deep (15ep) + shallow (25ep) for representative firing rates...", flush=True)
    deep = train_deep(Xtr, Ytr, 15); shallow = train_shallow(Xtr, Ytr, 25)
    s1, s2 = deep_spikes(deep, Xva); ssh = shallow_spikes(shallow, Xva)
    print(f"  (trained in {time.time()-t0:.0f}s)\n")

    # deep accounting (per sample)
    d_mac = 62*256*T
    d_ac  = s1*128 + s2*2
    d_params = 62*256 + 256*128 + 128*2
    d_fire1 = s1/(256*T); d_fire2 = s2/(128*T)
    # shallow accounting
    sh_mac = 62*16*T + 16*2
    sh_ac  = 0                        # rates are pooled (adds), no spike-driven synapse layer
    sh_params = 62*16 + 16*2
    sh_fire = ssh/(16*T)

    print("=== Track-2 accounting (per sample, T=250) ===\n")
    print(f"  DEEP baseline (62→256→128→2, all LIF)   acc 0.653   firing L1={d_fire1:.1%} L2={d_fire2:.1%}")
    print(f"     dense MACs (fc1, EEG in) : {d_mac:>12,}")
    print(f"     spike ACs  (fc2+fc3)     : {int(d_ac):>12,}")
    print(f"     total ops                : {int(d_mac+d_ac):>12,}")
    print(f"     footprint (weights)      : {d_params:>12,}\n")
    print(f"  SHALLOW spike-rates (62→16 LIF→rate→2)  acc 0.682   firing {sh_fire:.1%}")
    print(f"     dense MACs (W0 + head)   : {sh_mac:>12,}")
    print(f"     spike ACs                : {sh_ac:>12,}")
    print(f"     total ops                : {sh_mac:>12,}")
    print(f"     footprint (weights)      : {sh_params:>12,}\n")
    print("=== ratios (deep / shallow) ===")
    print(f"     total ops : {(d_mac+d_ac)/sh_mac:>6.1f}×   footprint : {d_params/sh_params:>5.1f}×")
    print(f"     ... and the shallow net is MORE accurate (0.682 vs 0.653, +0.029).")


if __name__ == "__main__":
    main()
