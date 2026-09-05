#!/usr/bin/env python3
"""Bit-faithful reference for elig_buffer.v (Paper III eq. 3, temporal credit).

Windowed eligibility for one synapse:
    p_t = delta_t if spk_t else 0
    g_t = g_{t-1} + p_t - p_{t-N}     (== sum of the last N gated errors)

Dumps vectors for tb_elig_buffer.v:
    elig_delta.hex   DW-bit transported error per step (two's complement)
    elig_spk.hex     1-bit presynaptic spike per step
    elig_g_exp.hex   ACCW-bit windowed eligibility after each step

Keep params IDENTICAL to tb_elig_buffer.v. This integer model is what the
temporal pcn_learner_snn.py must match when run bit-faithfully.
"""
import os, random

DW, N, ACCW, T = 12, 8, 18, 48

def elig(deltas, spks):
    fifo = [0] * N
    acc, gs = 0, []
    for d, s in zip(deltas, spks):
        p_new = d if s else 0
        p_old = fifo[N - 1]
        acc = acc + p_new - p_old
        fifo = [p_new] + fifo[:N - 1]
        assert -(1 << (ACCW - 1)) <= acc < (1 << (ACCW - 1)), f"acc overflow: {acc}"
        gs.append(acc)
    return gs

def hexs(v, w):
    return f"{v & ((1 << w) - 1):0{(w + 3) // 4}x}"

def main():
    random.seed(11)
    # exercise: a spike burst (window fills), a quiet tail (window drains as
    # terms fall off), sign changes in delta, and sparse spikes.
    deltas = [300, -250, 400, -100, 500, -600, 200, 350]   # 8 steps, dense spikes
    spks   = [1,    1,    1,   1,    1,   1,    1,   1]
    deltas += [700, -700, 300]     # more, still spiking -> old terms start leaving window
    spks   += [1,   1,    1]
    deltas += [999, 999, 999, 999, 999, 999, 999, 999, 999]  # quiet (no spikes): window drains
    spks   += [0]*9
    # fill the remainder with sparse random spikes + random signed deltas
    while len(deltas) < T:
        deltas.append(random.randint(-800, 800))
        spks.append(1 if random.random() < 0.5 else 0)
    deltas, spks = deltas[:T], spks[:T]
    assert all(-(1 << (DW-1)) <= d < (1 << (DW-1)) for d in deltas), "delta out of DW range"

    gs = elig(deltas, spks)
    d = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(d, "elig_delta.hex"), "w") as f:
        f.write("\n".join(hexs(x, DW) for x in deltas) + "\n")
    with open(os.path.join(d, "elig_spk.hex"), "w") as f:
        f.write("\n".join(hexs(s, 1) for s in spks) + "\n")
    with open(os.path.join(d, "elig_g_exp.hex"), "w") as f:
        f.write("\n".join(hexs(g, ACCW) for g in gs) + "\n")
    print(f"wrote elig vectors: T={T}, spikes={sum(spks)}, g range [{min(gs)},{max(gs)}]")

if __name__ == "__main__":
    main()
