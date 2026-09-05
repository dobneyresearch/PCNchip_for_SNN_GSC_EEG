#!/usr/bin/env python3
"""Cycle-accurate bit-faithful reference for pcn_temporal_tile.v (P6 array scale-out).

NO postsyn neurons x NI presyn inputs. Mirrors the RTL registers exactly (same as
top_ref.py, replicated): per-neuron lif+adjoint (delta_o), shared per-presyn z-delay
lines, per-synapse credit E_oi and graded write dW_oi.

Dumps stimulus + expected E and dW laid out cycle-major, synapse-minor (index
c*NSYN + (o*NI+i)), matching the RTL flattening. Params IDENTICAL to tb_temporal_tile.v.
"""
import os, math, random

NO, NI = 2, 2
MEMW, INW, ALPHA, ASHIFT, THRESH, PW, SURR = 20, 12, 230, 8, 1024, 8, 5
NUMER = (THRESH * THRESH) << PW
EEW, DW, N, AWACC, ECW, GEW, DWW = 12, 16, 8, 22, 32, 16, 20
B1, BSH1, B2, BSH2, F, CLIP, EPS, MW, VW = 230, 8, 1014, 10, 10, 1024, 1, 24, 32
T = 32
NSYN = NO * NI

def wrap(v, w):
    v &= (1 << w) - 1
    return v - (1 << w) if v >= (1 << (w - 1)) else v

from psi_lut import psi_lut as psi_of               # refinement #3: ψ via shared LUT

class Tile:
    def __init__(self):
        self.mem = [0]*NO; self.psi = [0]*NO
        self.fifo = [[0]*(N+1) for _ in range(NO)]; self.delta = [0]*NO
        self.zsr = [0]*NI
        self.E = [[0]*NI for _ in range(NO)]
        self.m = [[0]*NI for _ in range(NO)]; self.v = [[0]*NI for _ in range(NO)]
        self.dW = [[0]*NI for _ in range(NO)]

    def step(self, en, fold, in_cur, e_err, z_pre):
        if en:
            nmem, npsi, nfifo, ndelta = [0]*NO, [0]*NO, [None]*NO, [0]*NO
            for o in range(NO):
                integ = ((self.mem[o]*ALPHA) >> ASHIFT) + in_cur[o]
                npsi[o] = psi_of(integ)
                nmem[o] = wrap((integ - THRESH) if integ >= THRESH else integ, MEMW)
                a = wrap((self.psi[o]*e_err[o]) >> PW, DW)        # OLD psi
                a_old = self.fifo[o][N]
                ndelta[o] = wrap(self.delta[o] + a - a_old, AWACC)
                nfifo[o] = [a] + self.fifo[o][:N]
            zdel = [(self.zsr[i] >> (N-1)) & 1 for i in range(NI)]
            nE = [[wrap(self.E[o][i] + (self.delta[o] if zdel[i] else 0), ECW)  # OLD delta,z
                   for i in range(NI)] for o in range(NO)]
            nzsr = [((self.zsr[i] << 1) | (1 if z_pre[i] else 0)) & ((1 << N)-1) for i in range(NI)]
            self.mem, self.psi, self.fifo, self.delta = nmem, npsi, nfifo, ndelta
            self.zsr, self.E = nzsr, nE
        elif fold:
            sp, sn = (1 << (GEW-1))-1, -(1 << (GEW-1))
            for o in range(NO):
                for i in range(NI):
                    ef = sp if self.E[o][i] > sp else sn if self.E[o][i] < sn else self.E[o][i]
                    self.m[o][i] = wrap((B1*self.m[o][i] + ((1<<BSH1)-B1)*ef) >> BSH1, MW)
                    self.v[o][i] = (B2*self.v[o][i] + ((1<<BSH2)-B2)*(ef*ef)) >> BSH2
                    q = min((abs(self.m[o][i]) << F) // math.isqrt(self.v[o][i]+EPS), CLIP)
                    self.dW[o][i] = -q if self.m[o][i] < 0 else q
                    self.E[o][i] = 0
        return ([self.E[o][i] for o in range(NO) for i in range(NI)],
                [self.dW[o][i] for o in range(NO) for i in range(NI)])

def hexs(v, w):
    return f"{v & ((1 << w) - 1):0{(w + 3) // 4}x}"

def main():
    random.seed(31)
    cyc = []
    for _ in range(T):
        cyc.append((1, 0, [random.randint(-200, 900) for _ in range(NO)],
                    [random.randint(-800, 800) for _ in range(NO)],
                    [1 if random.random() < 0.5 else 0 for _ in range(NI)]))
    for _ in range(N):
        cyc.append((1, 0, [0]*NO, [0]*NO, [0]*NI))
    cyc.append((0, 1, [0]*NO, [0]*NO, [0]*NI))

    tile = Tile()
    Es, dWs = [], []
    for (en, fold, ic, ee, zp) in cyc:
        E, dW = tile.step(en, fold, ic, ee, zp)
        Es.extend(E); dWs.extend(dW)     # cycle-major, synapse-minor

    d = os.path.dirname(os.path.abspath(__file__))
    def dump(name, seq, w):
        with open(os.path.join(d, name + ".hex"), "w") as f:
            f.write("\n".join(hexs(x, w) for x in seq) + "\n")
    # stimulus (per cycle; per-neuron / per-presyn flattened)
    dump("tile_en",   [c[0] for c in cyc], 1)
    dump("tile_fold", [c[1] for c in cyc], 1)
    dump("tile_in",   [v for c in cyc for v in c[2]], INW)
    dump("tile_e",    [v for c in cyc for v in c[3]], EEW)
    dump("tile_z",    [ (sum((1 if c[4][i] else 0) << i for i in range(NI))) for c in cyc], NI)
    dump("tile_E_exp",  Es, ECW)
    dump("tile_dW_exp", dWs, DWW)
    print(f"wrote tile vectors: NO={NO} NI={NI} cycles={len(cyc)}, "
          f"final dW={dWs[-NSYN:]}")

if __name__ == "__main__":
    main()
