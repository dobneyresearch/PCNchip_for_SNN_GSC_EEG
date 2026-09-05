#!/usr/bin/env python3
"""Bit-faithful reference for route_quant.v (router-fit refinement #1).

Causal 6-bit routed-message quantiser (hardware analog of the sim's delta_bits):
    s    = max(|value|, s - (s >> LEAK))         # leaky running peak
    s_use= s if s>0 else 1
    q    = clamp( round_half_up(|value|*L / s_use), L ) * sign(value)   # L = 2^(OUTB-1)-1

round_half_up(a/b) == (2a + b) // (2b). Python `//`/`>>` == Verilog `/`/`>>>` (floor,
non-negative) so this is bit-exact. Dumps: rq_in.hex (VW), rq_q_exp.hex (OUTB),
rq_scale_exp.hex (VW). Params IDENTICAL to tb_route_quant.v.
"""
import os, random

VW, OUTB, LEAK, T = 24, 6, 6, 48
L = (1 << (OUTB - 1)) - 1                       # 31

def routeq(vals):
    s = 0; qs, ss = [], []
    for v in vals:
        av = abs(v)
        s = max(av, s - (s >> LEAK))            # running peak (incl. current)
        s_use = s if s > 0 else 1
        num = av * L
        qmag = min((2 * num + s_use) // (2 * s_use), L)
        qs.append(-qmag if v < 0 else qmag); ss.append(s)
    return qs, ss

def hexs(v, w):
    return f"{v & ((1 << w) - 1):0{(w + 3) // 4}x}"

def main():
    random.seed(61)
    # a big transient (sets the peak), decay, sign flips, small values (should still span +-L
    # relative to the decaying peak), and a fresh spike.
    vals = [50000, -8000, 30000, -2000, 100, -100, 40000]
    vals += [random.randint(-60000, 60000) for _ in range(T - len(vals))]
    vals = vals[:T]
    assert all(-(1 << (VW-1)) <= v < (1 << (VW-1)) for v in vals), "value range"
    qs, ss = routeq(vals)
    d = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(d, "rq_in.hex"), "w") as f:
        f.write("\n".join(hexs(v, VW) for v in vals) + "\n")
    with open(os.path.join(d, "rq_q_exp.hex"), "w") as f:
        f.write("\n".join(hexs(q, OUTB) for q in qs) + "\n")
    with open(os.path.join(d, "rq_scale_exp.hex"), "w") as f:
        f.write("\n".join(hexs(s, VW) for s in ss) + "\n")
    print(f"wrote routeq vectors: T={T}, q range [{min(qs)},{max(qs)}] (msg is {OUTB}-bit)")

if __name__ == "__main__":
    main()
