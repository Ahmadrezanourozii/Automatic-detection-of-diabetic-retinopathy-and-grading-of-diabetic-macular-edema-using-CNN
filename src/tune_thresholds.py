"""
tune_thresholds.py — pick the decision cut-points properly, from archived predictions.

Two separate problems, both visible in E05:

1. **The grade cut-points are not optimal for QWK.** Decoding at sigmoid > 0.5 puts the cut
   between grade k and k+1 at expected-grade 0.5, 1.5, 2.5 ... That is arbitrary. On an
   ordinal target with heavy class imbalance, moving the cuts is worth several QWK points
   and costs nothing — it is the standard final step in every DR grading competition.

2. **The screening operating point is far too conservative.** E05 referred only 73.7 % of
   referable-DR cases at 98.0 % specificity. In a screening programme a missed referral is
   the expensive error and an extra referral is cheap, so the model should sit somewhere
   near 90 % sensitivity. That is a threshold choice, not a modelling problem.

THE PART THAT MATTERS: cut-points are chosen CROSS-FITTED. For fold f, the cuts come from
the out-of-fold predictions of every OTHER fold, then are applied to fold f. Fold f's images
never take part in choosing their own thresholds. Tuning on the pooled predictions and then
reporting on those same predictions would be selection on the evaluation set, which is the
exact error PROTOCOL.md §3 exists to prevent — and it would inflate QWK by a visible amount
while looking perfectly reasonable.

Usage:
    python src/tune_thresholds.py --run runs/E05 --datasets <root> [--target-sens 0.90]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpora
import metrics as M
from model import expected_grade, decode, N_DR, N_DME


def cuts_to_grades(score, cuts):
    """cuts is ascending, length K-1: grade = number of cuts the score exceeds."""
    return np.searchsorted(np.asarray(cuts), score, side="right")


def fit_cuts(score, y, k, seed=0, rounds=8):
    """Coordinate ascent on QWK over K-1 cut-points. Deterministic, seconds to run."""
    cuts = [i + 0.5 for i in range(k - 1)]
    best = M.quadratic_weighted_kappa(y, cuts_to_grades(score, cuts), k)
    lo, hi = float(np.min(score)) - 0.05, float(np.max(score)) + 0.05
    for _ in range(rounds):
        improved = False
        for i in range(k - 1):
            left = cuts[i - 1] if i > 0 else lo
            right = cuts[i + 1] if i < k - 2 else hi
            if right <= left:
                continue
            for cand in np.linspace(left, right, 60)[1:-1]:
                trial = list(cuts)
                trial[i] = float(cand)
                q = M.quadratic_weighted_kappa(y, cuts_to_grades(score, trial), k)
                if q > best + 1e-9:
                    best, cuts, improved = q, trial, True
        if not improved:
            break
    return cuts, best


def fit_referral_cut(score, y_ref, target_sens):
    """Lowest threshold whose sensitivity is at least `target_sens` (highest specificity
    among the thresholds that clear the sensitivity bar)."""
    order = np.unique(np.round(score, 4))
    best = (None, -1.0)
    for t in order:
        p = score >= t
        tp = int((y_ref & p).sum()); fn = int((y_ref & ~p).sum())
        tn = int((~y_ref & ~p).sum()); fp = int((~y_ref & p).sum())
        sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
        if sens >= target_sens and spec > best[1]:
            best = (float(t), spec)
    return best[0] if best[0] is not None else float(np.min(score))


def load(run, datasets):
    rows = corpora.build(datasets, ("IDRiD", "Messidor-2"))
    by = {r["uid"]: r for r in rows}
    split = json.load(open("data/splits/dev_v1.json"))
    uids, dr_l, dme_l, fold_of = [], [], [], {}
    for p in sorted(glob.glob(os.path.join(run, "oof_*.npz"))):
        f = int(os.path.basename(p).split("_")[1].split(".")[0])
        z = np.load(p, allow_pickle=True)
        us = [str(u) for u in z["uids"]]
        uids += us
        for u in us:
            fold_of[u] = f
        dr_l.append(z["dr_logits"]); dme_l.append(z["dme_logits"])
    if not uids:
        raise SystemExit(f"no oof_*.npz in {run}")
    sub = [by[u] for u in uids]
    for r in sub:
        r["group"] = split["groups"].get(r["uid"], r["uid"])
    return (sub, np.array([fold_of[u] for u in uids]),
            torch.from_numpy(np.concatenate(dr_l)),
            torch.from_numpy(np.concatenate(dme_l)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--head", default="ordinal")
    ap.add_argument("--target-sens", type=float, default=0.90)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    rows, folds, dr_l, dme_l = load(a.run, a.datasets)
    dr_s = expected_grade(dr_l, a.head).numpy()
    dme_s = expected_grade(dme_l, a.head).numpy()
    dr_y = np.array([r["dr"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    three = np.array([r.get("dme_label_space") == "3class" and r["dme"] is not None
                      for r in rows])
    dme_y = np.array([r["dme"] if r["dme"] is not None else -1 for r in rows])

    dr_pred = np.zeros_like(dr_y)
    dme_pred = np.zeros_like(dr_y)
    ref_pred = np.zeros(len(rows), dtype=int)
    chosen = {}

    for f in sorted(set(folds)):
        te = folds == f
        tr = ~te
        c_dr, q_dr = fit_cuts(dr_s[tr], dr_y[tr], N_DR)
        dr_pred[te] = cuts_to_grades(dr_s[te], c_dr)

        m = tr & three
        if m.sum() > 20:
            c_dme, q_dme = fit_cuts(dme_s[m], dme_y[m], N_DME)
        else:
            c_dme, q_dme = [0.5, 1.5], float("nan")
        dme_pred[te] = cuts_to_grades(dme_s[te], c_dme)

        t_ref = fit_referral_cut(dr_s[tr], dr_y[tr] >= 2, a.target_sens)
        ref_pred[te] = (dr_s[te] >= t_ref).astype(int)

        chosen[int(f)] = {"dr_cuts": [round(c, 3) for c in c_dr],
                          "dme_cuts": [round(c, 3) for c in c_dme],
                          "referral_cut": round(t_ref, 3),
                          "dr_qwk_on_other_folds": round(q_dr, 4)}
        print(f"fold {f}: DR cuts {[round(c,2) for c in c_dr]}  "
              f"DME cuts {[round(c,2) for c in c_dme]}  referral@{t_ref:.2f}   "
              f"(chosen on the other {int(tr.sum())} images)")

    print(f"\n{'='*78}\nCROSS-FITTED THRESHOLDS — {a.run}\n{'='*78}")
    base_dr = decode(dr_l, N_DR, a.head).numpy()
    base_dme = decode(dme_l, N_DME, a.head).numpy()

    out = {}
    for name, y, p0, p1, k, mask in (
        ("DR 5-class", dr_y, base_dr, dr_pred, N_DR, np.ones(len(rows), bool)),
        ("DME 3-class (ungated)", dme_y, base_dme, dme_pred, N_DME, three),
    ):
        r0 = M.report(y[mask], p0[mask], k, groups=groups[mask], n_boot=2000)
        r1 = M.report(y[mask], p1[mask], k, groups=groups[mask], n_boot=2000)
        d, lo, hi, sig = M.paired_bootstrap_diff(
            y[mask], p1[mask], p0[mask],
            lambda t, q: M.quadratic_weighted_kappa(t, q, k), groups[mask], n_boot=2000)
        print(f"\n  {name}   n={int(mask.sum())}")
        print(f"    default cuts : acc {r0['accuracy']*100:5.1f}%  QWK {r0['qwk']:.3f}")
        print(f"    tuned cuts   : acc {r1['accuracy']*100:5.1f}%  QWK {r1['qwk']:.3f}")
        print(f"    QWK change   : {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
              f"{'SIGNIFICANT' if sig else 'indistinguishable'}")
        out[name] = {"default": r0, "tuned": r1,
                     "qwk_diff": d, "qwk_diff_ci95": [lo, hi], "significant": sig}

    y_ref = (dr_y >= 2).astype(int)
    for label, p in (("default decode", (base_dr >= 2).astype(int)),
                     (f"tuned for >={a.target_sens:.0%} sens", ref_pred)):
        tp = int(((y_ref == 1) & (p == 1)).sum()); fn = int(((y_ref == 1) & (p == 0)).sum())
        tn = int(((y_ref == 0) & (p == 0)).sum()); fp = int(((y_ref == 0) & (p == 1)).sum())
        print(f"\n  referable DR, {label:28s} sens {tp/max(1,tp+fn)*100:5.1f}%  "
              f"spec {tn/max(1,tn+fp)*100:5.1f}%   (missed {fn} of {tp+fn} referable)")
        out[f"referable_{label}"] = {"sensitivity": tp / max(1, tp + fn),
                                     "specificity": tn / max(1, tn + fp),
                                     "missed": fn, "n_referable": tp + fn}

    print("\n  Cut-points were chosen on the other folds' predictions, never on the fold "
          "they were applied to (PROTOCOL.md §3).")

    if a.write:
        rj = os.path.join(a.run, "results.json")
        res = json.load(open(rj))
        res["threshold_tuning"] = {"per_fold": chosen, "target_sensitivity": a.target_sens,
                                   "summary": out, "cross_fitted": True}
        json.dump(res, open(rj, "w"), indent=1, default=str)
        print(f"\nwrote threshold tuning into {rj}")


if __name__ == "__main__":
    main()
