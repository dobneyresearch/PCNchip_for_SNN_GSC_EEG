#!/usr/bin/env python3
"""END-TO-END match: the validated SIM credit  ↔  the fixed-point HW (top_ref.py == RTL).

The per-module TBs prove RTL == its integer reference. `top_ref.py` proves the single
lane composes. What was still UNCHECKED is the last link: does that fixed-point datapath
actually reproduce what the *validated maths sim* (`pcn_learner_snn.py`) computes for the
temporal credit — on REAL GSC data, not synthetic random vectors?

This script closes that loop. It:
  1. builds the committed winning recipe (`--readout leaky --pl_buffer_n 8 --rr_cold
     --var_norm_fold --rr_center --rr_lr_mult 100`) with `buf_uniform=True` (the committed
     UNIFORM HW cell — the A/B showed uniform≈faithful), briefly trains it on GSC so the
     spike/error distributions are realistic, then runs one fresh batch through the SIM's
     OWN credit path (`_error_carry`, `_readout_jacobian`, the top-layer accumulate);
  2. for the TOP hidden layer, extracts per-lane, per-timestep the exact quantities the HW
     lane consumes:  in_cur_o[t] = Zᵀ·W[o]  (membrane current),  e_err_o[t] = d_o·g(t)
     (the routed top error),  z_i[t] (presyn spike);
  3. quantises them to the RTL fixed-point contract (1.0 == THRESH LSBs; INW/EEW = 12) and
     runs the SAME integer lane as the RTL (`top_ref.Top`, which recomputes membrane+ψ via
     the shared LUT with α=230/256), giving the HW credit E_int;
  4. reports FIDELITY across thousands of real lanes — does E_int track the sim's float
     credit E_float? sign agreement, correlation — and isolates the two quantisation
     effects (α/LUT membrane vs the DESIGN.md-flagged 1-cycle ψ pipeline skew);
  5. dumps ONE real lane's stimulus + expected E/dW (via top_ref.Top ⇒ bit-exact to the
     RTL) as e2e_*.hex for tb_temporal_e2e.v — the RTL replaying a real GSC lane.

Run:  python3 e2e_ref.py            (writes e2e_*.hex + prints the fidelity report)
Then: cd ../rtl && bash run_all_tb.sh tb_temporal_e2e
"""
import os, sys, math, random
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GSC = os.path.abspath(os.path.join(HERE, "..", "..", "neurobench_gsc"))
sys.path.insert(0, HERE)          # top_ref, psi_lut
sys.path.insert(0, GSC)           # pcn_learner_snn, gsc_temporal_harness

from top_ref import Top, wrap, hexs, ALPHA, ASHIFT, THRESH, PW, N, DW, ECW, EEW, INW, GEW
from psi_lut import psi_lut as psi_of
from pcn_learner_snn import PCNLearner, PCNConfig
import gsc_temporal_harness as H

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IN_SCALE = float(THRESH)          # fixed-point: membrane 1.0 == THRESH LSBs (so in_cur too)
INMAX = (1 << (INW - 1)) - 1      # INW-bit signed clip
EMAX = (1 << (EEW - 1)) - 1       # EEW-bit signed clip


# ── fixed-point membrane + readiness for one (sample,neuron): a vector of ψ,spikes ──
def lif_fixed(in_cur_int):
    """Replay the RTL lif (α=230/256, LUT ψ) over an integer current sequence.
    Returns (psi[t], spike[t]) — psi from the PRE-reset membrane, exactly as lif_cell.v."""
    mem = 0; psi = []; spk = []
    for ic in in_cur_int:
        integ = ((mem * ALPHA) >> ASHIFT) + int(ic)
        p = psi_of(integ)
        fire = integ >= THRESH
        mem = (integ - THRESH) if fire else integ
        psi.append(p); spk.append(1 if fire else 0)
    return psi, spk


from top_ref import AWACC


def top_lane_delta(in_cur_int, e_int, n, skew=True):
    """The EXACT `top_ref.Top`/RTL registered adjoint the credit accumulator pairs with z,
    factored out of z (which only depends on the presyn input i). Runs T real + n flush
    en-cycles and returns:
      delta_prev[c] = the `self.delta` register value at the START of en-cycle c (= delta[c-1],
                      what the RTL's `E += z_del ? delta : 0` actually adds), and
      psi[t]        = the fixed-point LUT readiness (for the ψ-fidelity report).
    Then E_int[i] = Σ_c delta_prev[c]·z[c-N]  reproduces Top bit-for-bit (verified: the dumped
    lane's Σ matches top_ref.Top). skew=True ⇒ a=(ψ[t-1]·e)>>PW (registered ψ, the RTL);
    skew=False ⇒ a=(ψ[t]·e)>>PW (sim-faithful alignment) — isolates the DESIGN.md ψ skew."""
    T = len(e_int); NC = T + n                       # real + flush en-cycles
    mem = 0; psi_reg = 0; psi = []
    fifo = [0] * (n + 1); delta = 0
    delta_prev = []
    for c in range(NC):
        ic = int(in_cur_int[c]) if c < T else 0
        ee = int(e_int[c]) if c < T else 0
        integ = ((mem * ALPHA) >> ASHIFT) + ic
        psi_next = psi_of(integ)
        fire = integ >= THRESH
        mem = (integ - THRESH) if fire else integ
        a = wrap(((psi_reg if skew else psi_next) * ee) >> PW, DW)
        delta_prev.append(delta)                     # register value paired THIS cycle
        a_old = fifo[n]
        delta = wrap(delta + a - a_old, AWACC)
        fifo = [a] + fifo[:n]
        psi_reg = psi_next
        if c < T:
            psi.append(psi_next)
    return delta_prev, psi


def main():
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    dims, ncls = [20, 256, 256, 256], 35
    n_train, epochs = 4000, 4                       # brief: realistic spiking, not accuracy
    Xtr, Ytr, Xte, Yte = H.load(n_train, seed=0)
    print(f"[e2e] dev={DEV} train={tuple(Xtr.shape)} — committed recipe (uniform HW cell)")

    # committed winning recipe, buf_uniform=True (the A/B-chosen simpler HW cell)
    cfg = PCNConfig(temporal=True, fold_mode="sign", per_t_route=True, boss_lr=3e-4,
                    fold_every=128, seed=0, rule_readout=True, rr_warm=False,
                    rr_center=True, rr_lr_mult=100.0, init="snn", unit_rms_presyn=False,
                    auto_lr=True, readout_beta=H.ALPHA, pl_buffer_n=8, carry_theta=1.0,
                    buf_uniform=True, var_norm_fold=True)
    lrn = PCNLearner(dims, ncls, cfg=cfg, device=DEV,
                     forward_fn=H.make_forward_fn("leaky", 0))
    lrn.fit(Xtr, Ytr, epochs=epochs, bs=128, X_val=Xte[:2000], y_val=Yte[:2000], verbose=True)

    # ── one fresh batch through the SIM's own credit path (top hidden layer) ──
    B = 8
    Xb = Xte[:B].to(DEV).float(); yb = Yte[:B].to(DEV)   # forward_fn consumes raw temporal input
    cap = lrn.forward_fn(lrn, Xb, capture=True)
    S, _ = lrn._top_delta(lrn._scores(cap["feats"]), yb)
    Z, PSI = cap["Z"], cap["PSI"]
    L = len(lrn.W); l = L - 1                        # TOP hidden layer
    T = Z[l].shape[1]
    Wtop = lrn.W_ro.double()                         # rr_cold ⇒ the rule-trained readout transports
    d = lrn._constrain(S.double() @ Wtop)            # (B, feat)  δ into last hidden
    g = lrn._readout_jacobian(T, Z[l].device)        # (T,) readout Jacobian, applied for pl_buffer
    s = d[:, None, :] * g[None, :, None]             # e_o[t] = d_o·g(t)   (B,T,out)
    psi_sim = PSI[l].double()                        # (B,T,out) sim readiness
    delta = lrn._error_carry(s, psi_sim, cfg.pl_buffer_n, cfg.elig_alpha, cfg.carry_theta)
    # sim float credit per lane E_float[b,o,i] = Σ_t δ[b,t,o]·z[b,t,i]
    zf = Z[l].double()                               # (B,T,in) presyn spikes (0/1)
    E_float = torch.einsum("bto,bti->boi", delta, zf).cpu().numpy()   # (B,out,in)

    # per (b,o) inputs the HW lane consumes: in_cur (Zᵀ·W[o]) and e_err (s[:,:,o])
    Wl = lrn.W[l].double()                           # (out,in)
    in_cur_f = torch.einsum("bti,oi->bto", zf, Wl).cpu().numpy()      # (B,T,out) membrane current
    e_f = s.cpu().numpy()                            # (B,T,out) top error
    z_np = (zf.cpu().numpy() > 0.5).astype(np.int64) # (B,T,in)
    psi_sim_np = psi_sim.cpu().numpy()               # (B,T,out)

    # e_err integer scale: map the 99.5th pct of |e| to ~1500 LSB (headroom under EMAX)
    e_abs = np.abs(e_f); e_hi = np.percentile(e_abs[e_abs > 0], 99.5) if (e_abs > 0).any() else 1.0
    S_e = 1500.0 / max(e_hi, 1e-9)
    print(f"[e2e] fixed-point scales: in 1.0=={THRESH}LSB, e_err S_e={S_e:.1f} "
          f"(|e| p99.5={e_hi:.4g}); T={T}, N={N}, lanes={B}·{min(64,dims[l+1])}·{min(64,dims[l])}")

    # ── FIDELITY sweep over real lanes (integer window is (b,o)-local ⇒ vectorise over i) ──
    OO = min(64, dims[l + 1]); II = min(64, dims[l])   # subset of neurons/inputs
    sign_ok = tot = 0
    E_i_all, E_f_all = [], []
    psi_num = psi_da = psi_db = 0.0                  # cosine accum for ψ (sim vs fixed)
    skew_num = skew_da = skew_db = 0.0              # cosine: skew vs align credit
    E_int_grid = {}                                 # cache (b,o) -> E_int over i (for the dump)

    def zdel_matrix(b):
        """z delayed by N: row c pairs with delta_prev[c]. c<N → 0; else z[c-N]."""
        Zd = np.zeros((T + N, II), dtype=np.int64)
        Zd[N:N + T, :] = z_np[b, :, :II]
        return Zd

    for b in range(B):
        Zd = zdel_matrix(b)
        for o in range(OO):
            ic = np.clip(np.rint(in_cur_f[b, :, o] * IN_SCALE), -INMAX, INMAX).astype(np.int64)
            ee = np.clip(np.rint(e_f[b, :, o] * S_e), -EMAX, EMAX).astype(np.int64)
            dprev_s, psi_fx = top_lane_delta(ic.tolist(), ee.tolist(), N, skew=True)
            dprev_a, _ = top_lane_delta(ic.tolist(), ee.tolist(), N, skew=False)
            dprev_s = np.array(dprev_s); dprev_a = np.array(dprev_a)
            E_int = dprev_s @ Zd                     # (II,) EXACT RTL credit per input i
            E_alg = dprev_a @ Zd                     # (II,) sim-aligned fixed-point credit
            E_ref = E_float[b, o, :II]               # (II,) sim float credit
            E_int_grid[(b, o)] = E_int
            m = np.abs(E_ref) > 1e-9
            sign_ok += int((np.sign(E_int[m]) == np.sign(E_ref[m])).sum()); tot += int(m.sum())
            E_i_all.append(E_int[m]); E_f_all.append(E_ref[m])
            # ψ fidelity (sim float ψ vs fixed-point LUT ψ, in Q_PW) — cosine over t
            psf = psi_sim_np[b, :, o]; pfx = np.array(psi_fx) / (1 << PW)
            psi_num += float((psf * pfx).sum()); psi_da += float((psf * psf).sum()); psi_db += float((pfx * pfx).sum())
            # skew-vs-align credit similarity (does the 1-cycle ψ skew change the credit?)
            skew_num += float((E_int * E_alg).sum()); skew_da += float((E_int ** 2).sum()); skew_db += float((E_alg ** 2).sum())
    E_i_all = np.concatenate(E_i_all); E_f_all = np.concatenate(E_f_all)
    corr = float(np.corrcoef(E_i_all.astype(float), E_f_all.astype(float))[0, 1])
    # rank corr
    ri = np.argsort(np.argsort(E_i_all)); rf = np.argsort(np.argsort(E_f_all))
    rank = float(np.corrcoef(ri.astype(float), rf.astype(float))[0, 1])
    psi_cos = psi_num / (math.sqrt(psi_da * psi_db) + 1e-12)
    skew_cos = skew_num / (math.sqrt(skew_da * skew_db) + 1e-12)

    print("\n===== E2E FIDELITY: fixed-point HW credit  vs  validated sim credit =====")
    print(f"  lanes with nonzero sim credit : {tot}")
    print(f"  SIGN agreement (HW vs sim)     : {100.0*sign_ok/max(tot,1):.2f}%   ({sign_ok}/{tot})")
    print(f"  Pearson corr  (credit magnitude): {corr:.4f}")
    print(f"  Spearman rank corr             : {rank:.4f}")
    print(f"  ψ cosine (fixed LUT vs sim ψ)  : {psi_cos:.4f}   (α 0.9→230/256 + LUT quant)")
    print(f"  credit cosine (ψ-skew vs aligned): {skew_cos:.4f}   (DESIGN.md 1-cycle skew effect)")

    # ── dump ONE real lane (largest |E_int|) as bit-exact RTL vectors via top_ref.Top ──
    best = None
    for (b, o), E_int in E_int_grid.items():
        i = int(np.argmax(np.abs(E_int)))
        if best is None or abs(E_int[i]) > abs(best[0]):
            best = (E_int[i], b, o, i)
    _, b, o, i = best
    ic = np.clip(np.rint(in_cur_f[b, :, o] * IN_SCALE), -INMAX, INMAX).astype(np.int64)
    ee = np.clip(np.rint(e_f[b, :, o] * S_e), -EMAX, EMAX).astype(np.int64)
    zi = z_np[b, :, i]
    print(f"\n[e2e] dump lane: sample {b}, neuron {o}, input {i}  (E_int={best[0]})")

    # cycle stream: T real (en=1) + N flush (en=1, zeros) + 1 fold — matches top_ref.main
    cyc = [(1, 0, int(ic[t]), int(ee[t]), int(zi[t])) for t in range(T)]
    cyc += [(1, 0, 0, 0, 0) for _ in range(N)]
    cyc += [(0, 1, 0, 0, 0)]
    top = Top(); Es, dWs = [], []
    for (en, fold, a, c, zp) in cyc:
        E, dW = top.step(en, fold, a, c, zp)
        Es.append(E); dWs.append(dW)
    assert Es[-2] == int(best[0]), \
        f"factored sweep credit {best[0]} != top_ref.Top {Es[-2]} — sweep model not bit-exact"
    print(f"[e2e] self-check: factored sweep credit == top_ref.Top ({Es[-2]}) ✓")
    cols = [("e2e_en", [c[0] for c in cyc], 1), ("e2e_fold", [c[1] for c in cyc], 1),
            ("e2e_in", [c[2] for c in cyc], INW), ("e2e_e", [c[3] for c in cyc], EEW),
            ("e2e_z", [c[4] for c in cyc], 1),
            ("e2e_E_exp", Es, ECW), ("e2e_dW_exp", dWs, GEW + 4)]
    for name, seq, w in cols:
        with open(os.path.join(HERE, name + ".hex"), "w") as f:
            f.write("\n".join(hexs(x, w) for x in seq) + "\n")
    print(f"[e2e] wrote e2e_*.hex: {len(cyc)} cycles (T={T}+flush{N}+fold), "
          f"HW E={Es[-2]}, dW={dWs[-1]}  ⇒ tb_temporal_e2e replays this real lane bit-exact")


if __name__ == "__main__":
    main()
