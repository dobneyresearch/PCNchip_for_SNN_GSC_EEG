#!/usr/bin/env python3
"""Bit-faithful reference for graded_write.v (Paper III eq. 5, var_norm write).

Reliability-graded weight step m/sqrt(v), both m and v leaky-accumulator EMAs:
    m = (B1*m + (2^BSH1 - B1)*E)   >> BSH1        # signed floor  (beta1 ~ 0.9)
    v = (B2*v + (2^BSH2 - B2)*E^2) >> BSH2        # unsigned floor (beta2 ~ 0.99)
    step = sign(m) * clamp( (|m| << F) // isqrt(v + EPS), CLIP )

isqrt = floor integer sqrt (math.isqrt == the Verilog digit-by-digit function).
Python `>>`/`//` == Verilog `>>>`/`/` for these (floor / non-negative), so exact.

Dumps for tb_graded_write.v:  graded_E.hex, graded_m_exp.hex, graded_v_exp.hex,
graded_step_exp.hex.  Keep params IDENTICAL to tb_graded_write.v.
"""
import os, math, random

EW, MW, VW = 16, 24, 32
B1, BSH1 = 230, 8
B2, BSH2 = 1014, 10
F, CLIP, EPS, T = 10, 1024, 1, 40

def graded(E_seq):
    m = v = 0
    ms, vs, steps = [], [], []
    for E in E_seq:
        m = (B1 * m + ((1 << BSH1) - B1) * E) >> BSH1
        v = (B2 * v + ((1 << BSH2) - B2) * (E * E)) >> BSH2
        m &= (1 << MW) - 1                       # MW two's-complement wrap
        if m >= (1 << (MW - 1)):
            m -= (1 << MW)
        assert 0 <= v < (1 << VW), f"v overflow: {v}"
        vs_ = math.isqrt(v + EPS)
        q = (abs(m) << F) // vs_
        if q > CLIP:
            q = CLIP
        step = -q if m < 0 else q
        ms.append(m); vs.append(v); steps.append(step)
    return ms, vs, steps

def hexs(v, w):
    return f"{v & ((1 << w) - 1):0{(w + 3) // 4}x}"

def main():
    random.seed(3)
    # (a) consistent positive E: m grows, v moderate -> LARGE, reliable step.
    E = [400] * 8
    # (b) noisy alternating E of the same magnitude: m ~ 0, v large -> SMALL step.
    E += [800, -800, 800, -800, 800, -800, 800, -800]
    # (c) consistent negative E: step turns negative.
    E += [-500] * 8
    # (d) random signed tail.
    E += [random.randint(-1200, 1200) for _ in range(T - len(E))]
    E = E[:T]
    assert all(-(1 << (EW - 1)) <= e < (1 << (EW - 1)) for e in E), "E out of range"

    ms, vs, steps = graded(E)
    d = os.path.dirname(os.path.abspath(__file__))
    for name, seq, w in [("graded_E", E, EW), ("graded_m_exp", ms, MW),
                         ("graded_v_exp", vs, VW), ("graded_step_exp", steps, F + 4)]:
        with open(os.path.join(d, name + ".hex"), "w") as f:
            f.write("\n".join(hexs(x, w) for x in seq) + "\n")
    print(f"wrote graded vectors: T={T}, step range [{min(steps)},{max(steps)}], "
          f"v max {max(vs)}")

if __name__ == "__main__":
    main()
