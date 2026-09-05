#!/usr/bin/env python3
"""Bit-faithful reference for readout_grad.v (Paper III eq. 6, cold read-out).

Per read-out synapse:
    acc:  E += e*f   (weight)   |   E += e   (bias)
    fold: dW = clamp((LR_MULT * E) >> LSH, +-CLIP);  E = 0

Python `>>` == Verilog `>>>` (arithmetic floor). Dumps an op stream so the TB
replays exactly the same accumulate/fold sequence:
    ro_op.hex    per cycle: 1=acc, 2=fold      (hex)
    ro_e.hex     e per cycle (EW two's complement)
    ro_f.hex     f per cycle (FW two's complement)
    ro_bias.hex  bias_mode per cycle (0/1)
    ro_E_exp.hex E after each cycle (ACCW two's complement)
    ro_dW_exp.hex dW after each cycle (9-bit two's complement; holds last on non-fold)
Keep params IDENTICAL to tb_readout_grad.v.
"""
import os, random

EW, FW, ACCW, LR_MULT, LSH, CLIP = 12, 12, 32, 100, 16, 127

def wrap(v, w):
    v &= (1 << w) - 1
    return v - (1 << w) if v >= (1 << (w - 1)) else v

def run(ops):
    E, dW = 0, 0
    Es, dWs = [], []
    for kind, e, f, bias in ops:
        if kind == 1:                    # acc
            E = E + (e if bias else e * f)
            E = wrap(E, ACCW)
        elif kind == 2:                  # fold
            step = (LR_MULT * E) >> LSH
            step = max(-CLIP, min(CLIP, step))
            dW = step
            E = 0
        Es.append(E); dWs.append(dW)
    return Es, dWs

def hexs(v, w):
    return f"{v & ((1 << w) - 1):0{(w + 3) // 4}x}"

def main():
    random.seed(5)
    ops = []
    # 4 weight-mode folds of 6 accumulate samples, mixed sign (E swings, dW sometimes clamps)
    for _ in range(4):
        for _ in range(6):
            ops.append((1, random.randint(-120, 120), random.randint(-120, 120), 0))
        ops.append((2, 0, 0, 0))
    # 2 bias-mode folds of 5 samples
    for _ in range(2):
        for _ in range(5):
            ops.append((1, random.randint(-200, 200), 0, 1))
        ops.append((2, 0, 0, 1))
    for _, e, f, _b in ops:
        assert -(1 << (EW-1)) <= e < (1 << (EW-1)), "e range"
        assert -(1 << (FW-1)) <= f < (1 << (FW-1)), "f range"

    Es, dWs = run(ops)
    d = os.path.dirname(os.path.abspath(__file__))
    dump = [("ro_op",  [o[0] for o in ops], 4),
            ("ro_e",   [o[1] for o in ops], EW),
            ("ro_f",   [o[2] for o in ops], FW),
            ("ro_bias",[o[3] for o in ops], 1),
            ("ro_E_exp", Es, ACCW),
            ("ro_dW_exp", dWs, 9)]
    for name, seq, w in dump:
        with open(os.path.join(d, name + ".hex"), "w") as fh:
            fh.write("\n".join(hexs(x, w) for x in seq) + "\n")
    print(f"wrote readout vectors: T={len(ops)}, folds={sum(1 for o in ops if o[0]==2)}, "
          f"dW range [{min(dWs)},{max(dWs)}]")

if __name__ == "__main__":
    main()
