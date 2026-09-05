#!/usr/bin/env python3
"""Cycle-accurate bit-faithful reference for pcn_temporal_top.v (P5 integration).

Models the RTL registers exactly (all next-states from current state, committed
together each cycle) so the composition/timing is verified end to end:
  lif(mem,psi) -> a=(psi*e)>>PW -> adjoint_window(delta) -> z-delay -> E_credit
  -> (fold) graded_write -> dW.
NB: psi is a REGISTERED lif output, so a[t] uses psi[t-1] (a 1-cycle pipeline skew);
the model mirrors that. Sim-semantic alignment is a later refinement (P6).

Dumps a per-cycle stimulus + expected E_credit and dW for tb_temporal_top.v.
"""
import os, math, random

# lif
MEMW, INW, ALPHA, ASHIFT, THRESH, PW, SURR = 20, 12, 230, 8, 1024, 8, 5
NUMER = (THRESH * THRESH) << PW
# credit
EEW, DW, N, AWACC, ECW, GEW = 12, 16, 8, 22, 32, 16
# graded write (inside)
B1, BSH1, B2, BSH2, F, CLIP, EPS = 230, 8, 1014, 10, 10, 1024, 1
MW, VW = 24, 32
T = 40                                  # real timesteps in the sequence

def wrap(v, w):
    v &= (1 << w) - 1
    return v - (1 << w) if v >= (1 << (w - 1)) else v

from psi_lut import psi_lut as psi_of               # refinement #3: ψ via shared LUT

class Top:
    def __init__(self):
        self.mem = 0; self.psi = 0
        self.fifo = [0] * (N + 1); self.delta = 0
        self.zsr = 0; self.E = 0
        self.m = 0; self.v = 0; self.dW = 0

    def step(self, en, fold, in_cur, e_err, z_pre):
        if en:
            integ = ((self.mem * ALPHA) >> ASHIFT) + in_cur
            psi_next = psi_of(integ)
            fire = integ >= THRESH
            mem_next = (integ - THRESH) if fire else integ
            a = wrap((self.psi * e_err) >> PW, DW)          # uses OLD psi register
            a_old = self.fifo[N]
            delta_next = wrap(self.delta + a - a_old, AWACC)
            fifo_next = [a] + self.fifo[:N]
            z_del = (self.zsr >> (N - 1)) & 1
            E_next = wrap(self.E + (self.delta if z_del else 0), ECW)   # uses OLD delta,z
            zsr_next = ((self.zsr << 1) | (1 if z_pre else 0)) & ((1 << N) - 1)
            self.mem, self.psi = wrap(mem_next, MEMW), psi_next
            self.fifo, self.delta = fifo_next, delta_next
            self.zsr, self.E = zsr_next, E_next
        elif fold:
            sp, sn = (1 << (GEW - 1)) - 1, -(1 << (GEW - 1))
            ef = sp if self.E > sp else sn if self.E < sn else self.E
            self.m = wrap((B1 * self.m + ((1 << BSH1) - B1) * ef) >> BSH1, MW)
            self.v = (B2 * self.v + ((1 << BSH2) - B2) * (ef * ef)) >> BSH2
            q = (abs(self.m) << F) // math.isqrt(self.v + EPS)
            q = min(q, CLIP)
            self.dW = -q if self.m < 0 else q
            self.E = 0
        return self.E, self.dW

def hexs(v, w):
    return f"{v & ((1 << w) - 1):0{(w + 3) // 4}x}"

def main():
    random.seed(21)
    # one sample: T real steps, then N flush steps (drain window), then 1 fold.
    cyc = []   # (en, fold, in_cur, e_err, z_pre)
    for _ in range(T):
        cyc.append((1, 0, random.randint(-200, 900), random.randint(-800, 800),
                    1 if random.random() < 0.5 else 0))
    for _ in range(N):
        cyc.append((1, 0, 0, 0, 0))     # flush
    cyc.append((0, 1, 0, 0, 0))         # fold

    top = Top()
    Es, dWs = [], []
    for (en, fold, ic, ee, zp) in cyc:
        E, dW = top.step(en, fold, ic, ee, zp)
        Es.append(E); dWs.append(dW)

    d = os.path.dirname(os.path.abspath(__file__))
    cols = [("top_en",  [c[0] for c in cyc], 1), ("top_fold", [c[1] for c in cyc], 1),
            ("top_in",  [c[2] for c in cyc], INW), ("top_e", [c[3] for c in cyc], EEW),
            ("top_z",   [c[4] for c in cyc], 1),
            ("top_E_exp", Es, ECW), ("top_dW_exp", dWs, GEW + 4)]
    for name, seq, w in cols:
        with open(os.path.join(d, name + ".hex"), "w") as f:
            f.write("\n".join(hexs(x, w) for x in seq) + "\n")
    print(f"wrote top vectors: cycles={len(cyc)} (T={T}+flush{N}+fold), "
          f"final E={Es[-2]}, dW={dWs[-1]}")

if __name__ == "__main__":
    main()
