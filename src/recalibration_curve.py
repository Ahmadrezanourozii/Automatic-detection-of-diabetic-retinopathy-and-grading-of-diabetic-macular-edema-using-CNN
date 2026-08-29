"""
recalibration_curve.py — how many labelled local images does recalibration need?

F3 established that moving the decision cut-points is worth several times more than changing
the backbone, and costs nothing. That turns "you must recalibrate before deployment" into a
recommendation only if we can say *how much local labelled data it takes*. This measures it.

Design, and the part that makes it an honest estimate of deployment rather than a ceiling:

  * Draw n images at random as the "local labelled sample".
  * Fit ordinal cut-points on those n alone, maximising macro-recall.
  * Evaluate on the images NOT drawn — so every reported number is out of sample with
    respect to the cut-points, exactly as a deployment would be.
  * Repeat over many random draws, because *which* n images a clinic happens to label is
    itself a source of variation, and at small n it is the dominant one.
  * Compare against the shipped 0.5 cut-points evaluated on the same held-out images, so the
    gain is measured on identical data.

Also reported: the fraction of draws where recalibration *hurts*. At small n that is the
number a clinician actually needs — not the average gain, but the risk of making it worse.

Usage:
    python src/recalibration_curve.py --runs runs/E08X2 runs/E11X [--draws 200]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
from analyse_mild import fit_cuts, cuts_to_grades, macro_recall

K = 5
NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
SIZES = [25, 50, 100, 200, 400]


def load(run, corpus="aptos"):
    z = np.load(os.path.join(run, f"external_{corpus}_predictions.npz"), allow_pickle=True)
    y = np.asarray(z["y_dr"]).astype(int)
    lg = np.asarray(z["dr_logits"], dtype=np.float64)
    score = (1.0 / (1.0 + np.exp(-lg))).sum(axis=1)
    shipped = np.clip(np.cumprod((1.0 / (1.0 + np.exp(-lg))) > 0.5, axis=1).sum(1), 0, K - 1)
    return y, score, shipped


def study(y, score, shipped, draws, seed=0):
    rng = np.random.default_rng(seed)
    n_all = len(y)
    out = {}
    for n in SIZES:
        gains, milds, qwks, base_mr = [], [], [], []
        for _ in range(draws):
            idx = rng.choice(n_all, size=n, replace=False)
            mask = np.ones(n_all, bool); mask[idx] = False        # evaluate off-sample
            if len(np.unique(y[idx])) < 2:
                continue
            cuts, _ = fit_cuts(score[idx], y[idx], K, "mr", rounds=5, grid=40)
            pred = cuts_to_grades(score[mask], cuts)
            b, t = macro_recall(y[mask], shipped[mask], K), macro_recall(y[mask], pred, K)
            gains.append((t - b) * 100)
            base_mr.append(b * 100)
            milds.append(((pred[y[mask] == 1] == 1).mean()
                          - (shipped[mask][y[mask] == 1] == 1).mean()) * 100)
            qwks.append(M.quadratic_weighted_kappa(y[mask], pred, K)
                        - M.quadratic_weighted_kappa(y[mask], shipped[mask], K))
        g = np.array(gains); m = np.array(milds); q = np.array(qwks)
        out[n] = {"n_draws": len(g),
                  "macro_recall_gain": [float(np.mean(g)), *np.percentile(g, [2.5, 97.5])],
                  "mild_recall_gain": [float(np.mean(m)), *np.percentile(m, [2.5, 97.5])],
                  "qwk_change": [float(np.mean(q)), *np.percentile(q, [2.5, 97.5])],
                  "p_harmful": float((g < 0).mean()),
                  "baseline_macro_recall": float(np.mean(base_mr))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/E08X2", "runs/E11X"])
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--out", default="docs/generated/recalibration_curve.json")
    a = ap.parse_args()

    all_res = {}
    for run in a.runs:
        y, score, shipped = load(run)
        tag = os.path.basename(run)
        print(f"\n═══ {tag}  (n={len(y)} external images, {a.draws} draws per size) ═══")
        print(f"  shipped macro-recall: {macro_recall(y, shipped, K)*100:.2f}%")
        res = study(y, score, shipped, a.draws)
        all_res[tag] = res
        print(f"\n  {'n labelled':>11s}  {'macro-recall gain':>28s}  {'Mild recall gain':>26s}"
              f"  {'ΔQWK':>8s}  {'P(harm)':>8s}")
        for n, r in res.items():
            mg, mlo, mhi = r["macro_recall_gain"]; dg, dlo, dhi = r["mild_recall_gain"]
            print(f"  {n:11d}  {mg:+7.2f} [{mlo:+6.2f},{mhi:+6.2f}] pts  "
                  f"{dg:+7.1f} [{dlo:+6.1f},{dhi:+6.1f}]  "
                  f"{r['qwk_change'][0]:+8.4f}  {r['p_harmful']*100:7.1f}%")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"sizes": SIZES, "draws": a.draws, "results": all_res},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    print("Every gain is measured on images NOT used to fit the cut-points, against the "
          "shipped 0.5 thresholds on those same images.")


if __name__ == "__main__":
    main()
