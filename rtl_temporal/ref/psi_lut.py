#!/usr/bin/env python3
"""Readiness ψ as a shared LUT (refinement #3) — replaces the per-neuron divider.

ψ(d) = 1/(1+SURR|d|/THRESH)^2 in Q_PW; formerly a per-neuron divide (NUMER/den^2).
Here it is a table indexed by |d|>>PSI_IDXSH (ψ→0 past the table), sampled from the
exact formula. One shared ROM for all neurons.

Used two ways:
  * `from psi_lut import psi_lut as psi_of`  — the reference drop-in (in-memory LUT).
  * `python3 ref/psi_lut.py`                 — dumps psi_lut.hex for the RTL ($readmemh).
Both index identically, so RTL and reference stay bit-exact.
"""
import os

THRESH, SURR, PW = 1024, 5, 8
NUMER = (THRESH * THRESH) << PW
PSI_N, PSI_IDXSH = 512, 3                       # 512 entries × step 8 → |d| 0..4095, ψ=0 beyond

def psi_exact(d):
    ad = min(abs(d), 0xFFFF)
    return min(NUMER // ((THRESH + SURR * ad) ** 2), 1 << PW)

_LUT = [psi_exact(i << PSI_IDXSH) for i in range(PSI_N)]

def psi_lut(integ):
    """ψ for a membrane `integ` (pre-reset), matching the RTL LUT index exactly."""
    ad = abs(integ - THRESH)
    return _LUT[min(ad >> PSI_IDXSH, PSI_N - 1)]

def main():
    d = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(d, "psi_lut.hex"), "w") as f:
        f.write("\n".join(f"{v & ((1 << (PW + 1)) - 1):03x}" for v in _LUT) + "\n")
    print(f"wrote psi_lut.hex: {PSI_N} entries, psi[0]={_LUT[0]}, psi[8]={_LUT[1]}, "
          f"psi[last]={_LUT[-1]}")

if __name__ == "__main__":
    main()
