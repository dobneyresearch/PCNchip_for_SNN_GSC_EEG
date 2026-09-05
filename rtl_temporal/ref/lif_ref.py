#!/usr/bin/env python3
"""Bit-faithful reference for lif_cell.v (Paper III temporal LIF membrane).

Integer model of the digital LIF neuron:
    decayed = (mem * ALPHA) >> ASHIFT        # floor  == Verilog >>>
    integ   = decayed + in_cur
    spike   = integ >= THRESH
    mem     = integ - THRESH if spike else integ

Dumps vectors for tb_lif_cell.v (all two's-complement hex, MSB-first, one/line):
    lif_in.hex       INW-bit input current per step
    lif_mem_exp.hex  MEMW-bit membrane after each step
    lif_spk_exp.hex  1-bit spike per step

Keep the params here IDENTICAL to tb_lif_cell.v.  This same integer model is the
one pcn_learner_snn.py must match when the temporal path is run bit-faithfully.
"""
import os, random

MEMW, INW, ALPHA, ASHIFT, THRESH, PW, SURR, T = 20, 12, 230, 8, 1024, 8, 5, 64
NUMER = (THRESH * THRESH) << PW

from psi_lut import psi_lut as psi_of               # refinement #3: ψ via shared LUT

def lif(seq):
    mem, mems, spks, psis = 0, [], [], []
    for u in seq:
        decayed = (mem * ALPHA) >> ASHIFT          # floor; matches Verilog >>>
        integ = decayed + u
        psi = psi_of(integ)                        # readiness on the pre-reset membrane
        if integ >= THRESH:
            spk, mem = 1, integ - THRESH
        else:
            spk, mem = 0, integ
        assert -(1 << (MEMW - 1)) <= mem < (1 << (MEMW - 1)), f"membrane overflow: {mem}"
        mems.append(mem); spks.append(spk); psis.append(psi)
    return mems, spks, psis

def hexs(v, w):
    return f"{v & ((1 << w) - 1):0{(w + 3) // 4}x}"

def main():
    random.seed(7)
    # a sequence that exercises: sub-threshold build-up, firing + residue,
    # a quiet tail (pure leak decay), and negative input (membrane goes < 0).
    seq  = [220] * 8            # ramp toward threshold over several steps
    seq += [900, 300, 300]      # push over -> fire, then residue-driven fires
    seq += [0] * 12             # quiet: watch the leak decay the residue
    seq += [-400, -400, 200]    # drive negative, then recover
    seq += [random.randint(-500, 600) for _ in range(T - len(seq))]
    seq = seq[:T]
    assert all(-(1 << (INW - 1)) <= u < (1 << (INW - 1)) for u in seq), "input out of INW range"

    mems, spks, psis = lif(seq)
    d = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(d, "lif_in.hex"), "w") as f:
        f.write("\n".join(hexs(u, INW) for u in seq) + "\n")
    with open(os.path.join(d, "lif_mem_exp.hex"), "w") as f:
        f.write("\n".join(hexs(m, MEMW) for m in mems) + "\n")
    with open(os.path.join(d, "lif_spk_exp.hex"), "w") as f:
        f.write("\n".join(hexs(s, 1) for s in spks) + "\n")
    with open(os.path.join(d, "lif_psi_exp.hex"), "w") as f:
        f.write("\n".join(hexs(p, PW + 1) for p in psis) + "\n")
    print(f"wrote lif vectors: T={T}, fires={sum(spks)}, "
          f"mem range [{min(mems)},{max(mems)}], psi range [{min(psis)},{max(psis)}]")

if __name__ == "__main__":
    main()
