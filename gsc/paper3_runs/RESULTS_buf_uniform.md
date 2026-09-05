# buf_uniform A/B — does the simpler credit cell work in the sim? (2026-08-20)

**Question (user):** the validated sim credit (`_error_carry`, the `pl_buffer_n` path) is a ψ-gated
*leaky backward adjoint* — richer than the plain uniform window the simpler `adjoint_window.v` hardware
computes. Does dropping the reset-product decay (the simpler cell) cost accuracy in the sim?

**Setup:** the `--buf_uniform` flag (default OFF, `pcn_learner_snn.py._error_carry`) sets the reset carry
`r = 1`, so `δ[t] = Σ_{k=0}^{n} ψ[t+k]·e[t+k]` — a uniform n-window instead of the decaying adjoint.
A/B on the winning recipe, 30k/20ep leaky, matched seeds:
`--readout leaky --n 30000 --eval_n 4000 --epochs 20 --bs 128 --pl_buffer_n 8 --rr_cold --var_norm_fold
--rr_center --rr_lr_mult 100` [`--buf_uniform`].  Logs: `paper3_runs/buf_{faithful,uniform}_30k[_s{1,2}].log`.

## Result (30k/20ep leaky, 3 seeds)
| seed | faithful (leaky adjoint) | uniform (simple cell) | Δ (uniform − faithful) |
|---|---|---|---|
| 0 | 0.7685 | 0.7633 | −0.0052 |
| 1 | 0.7803 | 0.7875 | +0.0072 |
| 2 | 0.7683 | 0.7783 | +0.0100 |
| **mean** | **0.7724** | **0.7764** | **+0.0040** |

Paired t ≈ 0.85 (df=2), **p ≈ 0.4 — NOT significant.** Uniform is if anything marginally *higher* on the
mean, well inside the ~0.7pp run-to-run band.

## Reading
**The sign of the gap FLIPS across seeds** and the mean difference (+0.004) is not significant — the
classic signature of "no real difference, both inside the noise band." So the **reset-product decay is
NOT load-bearing at n=8**: the membrane leak (α=0.9, ~10-step time constant) already caps the effective
reach over the 8-step window, so re-weighting the terms inside it buys nothing. The simpler uniform cell
is statistically equivalent to the faithful leaky adjoint.

## 84k consistency check (full set, leaky, winning recipe + `--buf_uniform`)
**uniform 0.8321** vs faithful 0.8311 vs Adam 0.8810 (`paper3_runs/buf_uniform_84k.log`). The committed
uniform cell MATCHES (marginally beats) the faithful adjoint at full scale, confirming the 30k finding —
~94% of BPTT (0.832/0.881). Headline in the paper updated to 0.832 (the committed mechanism); gap to Adam
0.049.

## Consequences
- **Hardware:** commit to the simpler `adjoint_window.v` (uniform window). The faithful `elig_carry.v`
  (ψ-gated leaky adjoint) is NOT needed — dropped. Simpler cell (no state-dependent leak / reset product).
- **Paper:** eq (3) stated as the uniform adjoint window; this A/B is the supporting ablation (§5).
- Confirms the recurring theme that a simpler hardware design can remove maths complexity at no accuracy cost.

⚠ Both arms use the corrected credit *structure* (`δ=Σ(ψe)` window, then `Σ_t δ·z` with the current
presyn spike); the only difference is the per-term decay. The structural correction to eq (3) is
independent of which arm wins.
