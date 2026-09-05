"""
boss_h_damp A/B — does self-annealing the sign step tame the jug's overshoot-return dither?

HYPOTHESIS (this session): the 1-bit sign fold is signSGD — a FIXED step that never shrinks, so it
dithers ±0.04-0.05/epoch around the optimum (overshoot, return). boss_h already measures how
confidently-WRONG the batch is (severity from the classifier margin), but sign() throws that magnitude
away at the fold. boss_h_damp multiplies the sign write by a GLOBAL scalar = clamp(mean_bh/div, floor, 1):
big step when the batch is badly wrong (boost-on-worse), small step as the model converges (damp) —
the annealer auto_lr wanted but sign() ate. One shared gain on the error bus (HW-cheap, still 1-bit dir).

HONEST metrics (the dither is WITHIN-run; best-of-epochs already catches each bounce peak):
  best  : max val over epochs           (what we report — may be INSENSITIVE to damping)
  final : val at the last epoch         (where a deployed rolodex chip actually LANDS — the real win)
  late  : mean ± std of the last N vals  (final-state reliability; damping should shrink the std)

Winning EEG config: shallow 62→16, rate readout, tap K16, sign fold, boss_lr 3e-4, 60ep.
Run:  python3 probe_bhdamp.py --seeds 0,1,2,3 --div 8 --floor 0.125
      python3 probe_bhdamp.py --seeds 0,1,2,3 --div 4 --floor 0.25    # boost-on-worse variant
"""
import argparse, time, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gsc"))
import torch
from pcn_learner_snn import PCNLearner, PCNConfig
from eeg_mi_harness import load, make_forward_fn, ALPHA, DEV

SHAPE = [62, 16]


def cfg_for(seed, damp=False, div=8.0, floor=0.125, tap_K=16, boss_lr=3e-4):
    return PCNConfig(temporal=True, fold_mode="sign", per_t_route=True, boss_lr=boss_lr,
                     fold_every=128, fold_mean=True, seed=seed, init="snn",
                     unit_rms_presyn=False, auto_lr=False, readout_beta=ALPHA, carry_theta=1.0,
                     buf_uniform=True, wclip=5.0, rule_readout=True, rr_warm=False, rr_center=True,
                     rr_lr_mult=10.0, pl_buffer_n=8, var_norm_fold=False,
                     tap_fold=True, tap_K=tap_K,
                     boss_h_damp=damp, boss_h_damp_div=div, boss_h_damp_floor=floor)


def stats(vals):
    import statistics as st
    return (max(vals), vals[-1],
            st.mean(vals[-N_LATE:]), (st.pstdev(vals[-N_LATE:]) if N_LATE > 1 else 0.0))


N_LATE = 10


def run_one(Xtr, Ytr, Xva, Yva, seed, damp, div, floor, epochs, bs):
    lrn = PCNLearner(SHAPE, 2, cfg=cfg_for(seed, damp=damp, div=div, floor=floor),
                     device=DEV, forward_fn=make_forward_fn("rate", seed, het=False))
    torch.manual_seed(seed)                 # seed the fit shuffle (module uses global RNG) ⇒ paired A/B
    hist = lrn.fit(Xtr, Ytr, epochs=epochs, bs=bs, X_val=Xva, y_val=Yva, verbose=False)
    vals = [v for _, _, v in hist]
    dlast = getattr(lrn, "_last_damp", 1.0)
    return vals, dlast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3")
    ap.add_argument("--div", type=float, default=8.0)
    ap.add_argument("--floor", type=float, default=0.125)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    Xtr, Ytr, Xva, Yva = load(normalize=True)
    print(f"[bhdamp] shallow {SHAPE} rate+tap K16 sign boss_lr=3e-4  div={args.div} floor={args.floor}")
    print(f"  paired A/B (off vs on), seeds={seeds}, {args.epochs}ep  train={tuple(Xtr.shape)} dev={DEV}")
    print("  metrics: best / final / late(mean±std over last 10)\n", flush=True)

    rows = {"off": [], "on": []}
    for s in seeds:
        for tag, damp in (("off", False), ("on", True)):
            t0 = time.time()
            vals, dlast = run_one(Xtr, Ytr, Xva, Yva, s, damp, args.div, args.floor, args.epochs, args.bs)
            b, f, lm, ls = stats(vals)
            rows[tag].append((b, f, lm, ls))
            dinfo = f" damp_last={dlast:.3f}" if damp else ""
            print(f"  seed{s} {tag:3s}  best={b:.4f} final={f:.4f} late={lm:.4f}±{ls:.4f}"
                  f"  ({time.time()-t0:.0f}s){dinfo}", flush=True)

    import statistics as st
    print("\n=== boss_h_damp A/B (n={} seeds) ===".format(len(seeds)))
    print(f"  {'metric':6s} {'OFF':>18s} {'ON':>18s} {'Δ(on−off)':>12s}")
    for j, name in enumerate(["best", "final", "late_m", "late_std"]):
        off = [r[j] for r in rows["off"]]; on = [r[j] for r in rows["on"]]
        om, osd = st.mean(off), (st.pstdev(off) if len(off) > 1 else 0.0)
        nm, nsd = st.mean(on), (st.pstdev(on) if len(on) > 1 else 0.0)
        print(f"  {name:6s} {om:.4f}±{osd:.4f}   {nm:.4f}±{nsd:.4f}   {nm-om:+.4f}")
    print("\n  read: best≈flat is EXPECTED (peak already caught); the win is final↑ toward best")
    print("  and late_std↓ (steadier landing = rolodex reliability). late_std↑ ⇒ damper hurt.", flush=True)


if __name__ == "__main__":
    main()
