"""
Forwards-only JUG on the shallow AND deep shapes — the comparator pairing for the Adam re-confirm.

Everything here uses the VALIDATED module `pcn_learner_snn.py` UNEDITED (imported, single-sourced). The
only variation is a frozen per-neuron τ spread supplied through the harness forward (het=True) — that is
a FORWARD-model change, the fold is untouched. The jug uses its own validated boss_lr=3e-4 (NOT the Adam
lr=1e-3 fix — that was an Adam-comparator artifact).

Recipe = the EEG cold-sign fold that scored ~0.64 (probe_jug_shallow `cold_sign_lr10`): sign fold,
boss_lr 3e-4, fold_every 128, init snn, cold rule-readout (rr_center, rr_lr_mult 10), pl_buffer_n 8,
buf_uniform, wclip 5, carry_theta 1.0, auto_lr off, var_norm off. θ fixed at 1.0.

  plain : scalar α=0.9              (the drop-in jug baseline, pairs with Adam)
  het   : frozen per-neuron α~U[0.70,0.98]  (does the free timescale spread help the jug too?)

Comparators (Adam, lr=1e-3): shallow none 0.667 / learned-taps 0.702 ; deep fixed ~0.66 / spread ~0.67.
Run:  python3 probe_jug.py --shapes shallow,deep --modes plain,het
"""
import argparse, time, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gsc"))
import torch
from pcn_learner_snn import PCNLearner, PCNConfig
from eeg_mi_harness import load, make_forward_fn, ALPHA, DEV

SHAPES = {"shallow": [62, 16], "deep": [62, 256, 128]}


def cfg_for(seed, tap=False, tap_K=16, vn=False, q=False, q_lr_mult=1.0, delta_bits=6, boss_lr=3e-4):
    return PCNConfig(temporal=True, fold_mode="sign", per_t_route=True, boss_lr=boss_lr,
                     fold_every=128, fold_mean=True, seed=seed, init="snn",
                     unit_rms_presyn=False, auto_lr=False, readout_beta=ALPHA, carry_theta=1.0,
                     buf_uniform=True, wclip=5.0, rule_readout=True, rr_warm=False, rr_center=True,
                     rr_lr_mult=10.0, pl_buffer_n=8, var_norm_fold=vn,       # graded m/√v write when vn
                     tap_fold=tap, tap_K=tap_K,          # ⚗ tap_fold default OFF
                     q_fold=q, q_lr_mult=q_lr_mult,      # ⚗ q-blend fold default OFF
                     delta_bits=delta_bits)              # per-row δ quantisation (6=default, 0=off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", type=str, default="shallow,deep")
    ap.add_argument("--modes", type=str, default="plain,het")
    ap.add_argument("--readout", choices=["rate", "leaky"], default="rate")
    ap.add_argument("--var_norm", action="store_true", help="graded m/√v write instead of 1-bit sign")
    ap.add_argument("--q_lr_mult", type=float, default=1.0, help="bootstrap LR for the q blend (from LR sweep)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta_bits", type=int, default=6, help="per-row δ quantisation (6=default, 0=off)")
    ap.add_argument("--boss_lr", type=float, default=3e-4, help="jug fold pulse rate (GSC-swept 3e-4; sweep on EEG)")
    ap.add_argument("--tap_K", type=int, default=16, help="tap FIR window length K")
    args = ap.parse_args()
    Xtr, Ytr, Xva, Yva = load(normalize=True)
    write = "var_norm" if args.var_norm else "sign"
    print(f"[jug] forwards-only  write={write} readout={args.readout}  train={tuple(Xtr.shape)} dev={DEV}")
    print("  (Adam comparators lr=1e-3: shallow 0.667 / taps 0.702 ; deep ~0.66)\n", flush=True)
    res = {}
    for shape in [s.strip() for s in args.shapes.split(",") if s.strip()]:
        for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
            t0 = time.time()
            tap, het, qf = ("tap" in mode), ("het" in mode), ("q" in mode)
            lrn = PCNLearner(SHAPES[shape], 2,
                             cfg=cfg_for(args.seed, tap=tap, tap_K=args.tap_K, vn=args.var_norm, q=qf, q_lr_mult=args.q_lr_mult, delta_bits=args.delta_bits, boss_lr=args.boss_lr),
                             device=DEV, forward_fn=make_forward_fn(args.readout, args.seed, het=het))
            torch.manual_seed(args.seed)          # ★ seed the fit shuffle (module uses global RNG) ⇒ reproducible/comparable
            hist = lrn.fit(Xtr, Ytr, epochs=args.epochs, bs=args.bs, X_val=Xva, y_val=Yva, verbose=False)
            res[(shape, mode)] = max(v for _, _, v in hist)
            tapinfo = ""
            if tap:                                   # did the delay actually get LEARNED (moved off identity)?
                Wt = lrn.W_tap.cpu()
                cur = Wt[:, -1].abs().mean().item(); past = Wt[:, :-1].abs().mean().item()
                peak = Wt.abs().argmax(dim=1).float().mean().item()
                tapinfo += f"  [|cur|={cur:.2f} |past|={past:.2f} peakidx={peak:.1f}/{lrn.cfg.tap_K-1}]"
            if qf:                                    # where did q settle? (→1 = LIF/off, <1 = ALIF/on)
                qm = [torch.sigmoid(q).mean().item() for q in lrn.q_raw]
                tapinfo += "  q̄=[" + ",".join(f"{m:.2f}" for m in qm) + "]"
            print(f"  >>> {shape:7s} {mode:5s} best val={res[(shape, mode)]:.4f}   "
                  f"({time.time()-t0:.0f}s){tapinfo}", flush=True)
    print("\n=== JUG (forwards-only) shallow/deep × plain/het ===")
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {k[0]:7s} {k[1]:5s} {v:.4f}")


if __name__ == "__main__":
    main()
