#!/usr/bin/env python3
"""Bit-faithful reference for transpose_route.v (Paper III deep credit Wᵀ hop).

    e_{l-1,i} = sat( (sum_o delta_o * W_oi) >> RSH , EEW )

Python `>>` == Verilog `>>>` (arithmetic floor). Dumps NV random test vectors:
    tr_delta.hex  NO adjoints per vector (AWACC two's complement)
    tr_w.hex      NO*NI weights per vector (WW two's complement, row-major o*NI+i)
    tr_e_exp.hex  NI routed errors per vector (EEW two's complement)
Layout is vector-major. Params IDENTICAL to tb_transpose_route.v.
"""
import os, random

NO, NI, AWACC, WW, EEW, RSH = 2, 2, 22, 8, 12, 16
NV = 24
EMAX, EMIN = (1 << (EEW-1)) - 1, -(1 << (EEW-1))

def route(delta, W):
    out = []
    for i in range(NI):
        acc = sum(delta[o] * W[o][i] for o in range(NO))
        sc = acc >> RSH                          # arithmetic floor
        out.append(max(EMIN, min(EMAX, sc)))
    return out

def hexs(v, w):
    return f"{v & ((1 << w) - 1):0{(w + 3) // 4}x}"

def main():
    random.seed(41)
    D, Wv, E = [], [], []
    for _ in range(NV):
        delta = [random.randint(-(1 << 20), (1 << 20)) for _ in range(NO)]
        W = [[random.randint(-128, 127) for _ in range(NI)] for _ in range(NO)]
        D.extend(delta)
        Wv.extend(W[o][i] for o in range(NO) for i in range(NI))
        E.extend(route(delta, W))
    d = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(d, "tr_delta.hex"), "w") as f:
        f.write("\n".join(hexs(x, AWACC) for x in D) + "\n")
    with open(os.path.join(d, "tr_w.hex"), "w") as f:
        f.write("\n".join(hexs(x, WW) for x in Wv) + "\n")
    with open(os.path.join(d, "tr_e_exp.hex"), "w") as f:
        f.write("\n".join(hexs(x, EEW) for x in E) + "\n")
    print(f"wrote transpose vectors: NV={NV}, e range [{min(E)},{max(E)}]")

if __name__ == "__main__":
    main()
