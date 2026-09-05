#!/usr/bin/env python3
"""Bit-faithful reference for graded_sweep.v (refinement #4, shared swept √+÷).

Per synapse (persistent m,v across folds), per fold:
    m = (B1*m + (2^BSH1-B1)*E) >> BSH1        # signed floor
    v = (B2*v + (2^BSH2-B2)*E^2) >> BSH2       # unsigned floor
    dW = sign(m) * clamp((|m|<<F)//isqrt(v+EPS), CLIP)
identical to graded_write — the sweep just time-multiplexes one engine over NS
synapses, so the result is bit-exact to NS parallel graded_writes.

Dumps NFOLDS folds: gs_E.hex (NFOLDS*NS, EW), gs_dW_exp.hex (NFOLDS*NS, F+4).
Params IDENTICAL to tb_graded_sweep.v.
"""
import os, math, random

NS, EW, MW, VW = 4, 16, 24, 32
B1, BSH1, B2, BSH2, F, CLIP, EPS = 230, 8, 1014, 10, 10, 1024, 1
NFOLDS = 12

def wrap(v, w):
    v &= (1 << w) - 1
    return v - (1 << w) if v >= (1 << (w - 1)) else v

def sweep(folds):
    m = [0]*NS; vv = [0]*NS; out = []
    for E in folds:
        dW = [0]*NS
        for s in range(NS):
            e = E[s]
            m[s]  = wrap((B1*m[s] + ((1<<BSH1)-B1)*e) >> BSH1, MW)
            vv[s] = (B2*vv[s] + ((1<<BSH2)-B2)*(e*e)) >> BSH2
            q = min((abs(m[s]) << F) // math.isqrt(vv[s]+EPS), CLIP)
            dW[s] = -q if m[s] < 0 else q
        out.append(dW)
    return out

def hexs(v, w):
    return f"{v & ((1 << w) - 1):0{(w + 3) // 4}x}"

def main():
    random.seed(71)
    # per fold, per synapse E: give each synapse a distinct regime (consistent +, noisy,
    # consistent -, random) so the sweep is exercised across synapses.
    folds = []
    for f in range(NFOLDS):
        E = [ 400,                                   # syn0: consistent +
             (700 if f % 2 == 0 else -700),          # syn1: noisy alternating
             -500,                                   # syn2: consistent -
              random.randint(-1200, 1200)]           # syn3: random
        folds.append(E)
    for E in folds:
        assert all(-(1 << (EW-1)) <= e < (1 << (EW-1)) for e in E), "E range"

    out = sweep(folds)
    d = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(d, "gs_E.hex"), "w") as fh:
        fh.write("\n".join(hexs(e, EW) for E in folds for e in E) + "\n")
    with open(os.path.join(d, "gs_dW_exp.hex"), "w") as fh:
        fh.write("\n".join(hexs(x, F+4) for dW in out for x in dW) + "\n")
    print(f"wrote gsweep vectors: NS={NS} folds={NFOLDS}, last-fold dW={out[-1]}")

if __name__ == "__main__":
    main()
