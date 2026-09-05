#!/usr/bin/env python3
"""Bit-faithful reference for adjoint_window.v (Paper III corrected credit core).

Running windowed sum of the gated error a = psi*e:
    S[t] = sum_{k=0}^{min(t,N)} a[t-k]   (add-new / subtract-old over an N+1 FIFO)

Dumps: adj_a.hex (DW), adj_delta_exp.hex (ACCW). Params IDENTICAL to tb_adjoint_window.v.
This running sum is the credit adjoint; the TOP delays z by N to pair delta[t] with
z[t] (Sec. dataflow in DESIGN.md).
"""
import os, random

DW, N, ACCW, T = 16, 8, 22, 48

def adjoint(a_seq):
    fifo = [0] * (N + 1)
    acc, out = 0, []
    for a in a_seq:
        a_old = fifo[N]
        acc = acc + a - a_old
        fifo = [a] + fifo[:N]
        assert -(1 << (ACCW-1)) <= acc < (1 << (ACCW-1)), f"acc overflow {acc}"
        out.append(acc)
    return out

def hexs(v, w):
    return f"{v & ((1 << w) - 1):0{(w + 3) // 4}x}"

def main():
    random.seed(13)
    # a = psi*e: signed, varied magnitude; a burst then quiet (window drains), sign flips.
    a = [1200, -800, 1500, -400, 900, -1100, 700, 1300, -600]   # dense
    a += [0] * 10                                                # quiet: window drains
    a += [random.randint(-1500, 1500) for _ in range(T - len(a))]
    a = a[:T]
    assert all(-(1 << (DW-1)) <= x < (1 << (DW-1)) for x in a), "a range"
    out = adjoint(a)
    d = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(d, "adj_a.hex"), "w") as f:
        f.write("\n".join(hexs(x, DW) for x in a) + "\n")
    with open(os.path.join(d, "adj_delta_exp.hex"), "w") as f:
        f.write("\n".join(hexs(x, ACCW) for x in out) + "\n")
    print(f"wrote adjoint vectors: T={T}, delta range [{min(out)},{max(out)}]")

if __name__ == "__main__":
    main()
