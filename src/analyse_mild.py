"""
analyse_mild.py — is the external Mild collapse calibration, or capacity?

Mild recall falls from 45-52 % internally to 5-6 % on APTOS, an eight-fold collapse in one
class, while every other class stays within ~26 % of its internal value. The larger backbone
(E11) did not help. Two explanations, and they are distinguishable:

  CALIBRATION — the model ranks Mild correctly but its grade boundaries sit in the wrong
    place for this corpus, so Mild lands on the far side of a cut. Recoverable by moving the
    cut-points, which needs a small labelled sample at deployment.
  CAPACITY / DOMAIN SHIFT — the model cannot separate Mild from its neighbours in this corpus
    at all. Moving cut-points then trades other classes away without recovering Mild.

The test: fit ordinal cut-points on one half of APTOS, apply to the other, and swap
(2-fold cross-fitting, so no image influences its own cut-points). Fit twice, under two
objectives, because they answer different questions:

  * maximise QWK           — what a metric-driven recalibration would do
  * maximise macro-recall  — what a recalibration that is *asked* to care about Mild would do

If Mild recovers under macro-recall, this is calibration and the fix is cheap. If it does not
recover under either, it is a genuine domain-shift limit and must be reported as one.

NOTE ON WHAT THIS CAN AND CANNOT CLAIM. Fitting cut-points on APTOS labels means this is a
*diagnostic*, not an external result: it measures the ceiling recoverable by recalibration
given labelled local data. It does not license quoting the recalibrated numbers as external
validation, and they are not written into any results table.

Usage:
    python src/analyse_mild.py --run runs/E11X [--corpus APTOS]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def expected_grade_np(logits):
    return 1.0 / (1.0 + np.exp(-logits)).sum(axis=1) * 0 + (1.0 / (1.0 + np.exp(-logits))).sum(axis=1)


def cuts_to_grades(score, cuts):
    return np.searchsorted(np.asarray(cuts), score, side="right")


def macro_recall(y, p, k):
    out = []
    for c in range(k):
        m = y == c
        if m.sum():
            out.append((p[m] == c).mean())
    return float(np.mean(out))


def fit_cuts(score, y, k, objective, rounds=8, grid=80):
    fn = ((lambda a, b: M.quadratic_weighted_kappa(a, b, k)) if objective == "qwk"
          else (lambda a, b: macro_recall(a, b, k)))
    cuts = [i + 0.5 for i in range(k - 1)]
    best = fn(y, cuts_to_grades(score, cuts))
    lo, hi = float(score.min()) - 0.05, float(score.max()) + 0.05
    for _ in range(rounds):
        improved = False
        for i in range(k - 1):
            left = cuts[i - 1] if i > 0 else lo
            right = cuts[i + 1] if i < k - 2 else hi
            if right <= left:
                continue
            for cand in np.linspace(left, right, grid)[1:-1]:
                trial = list(cuts); trial[i] = float(cand)
                v = fn(y, cuts_to_grades(score, trial))
                if v > best + 1e-9:
                    best, cuts, improved = v, trial, True
        if not improved:
            break
    return cuts, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/E11X")
    ap.add_argument("--corpus", default="APTOS")
    a = ap.parse_args()

    z = np.load(os.path.join(a.run, f"external_{a.corpus.lower()}_predictions.npz"),
                allow_pickle=True)
    y = np.asarray(z["y_dr"]).astype(int)
    logits = np.asarray(z["dr_logits"], dtype=np.float64)
    k = logits.shape[1] + 1
    score = (1.0 / (1.0 + np.exp(-logits))).sum(axis=1)      # E[grade] under the ordinal head

    base = np.clip(np.cumprod((1.0 / (1.0 + np.exp(-logits))) > 0.5, axis=1).sum(1), 0, k - 1)
    print(f"{a.corpus}: n={len(y)}  k={k}")
    print(f"\n{'':26s}" + "".join(f"{n:>13s}" for n in NAMES) + "     QWK   macro-rec")

    def row(label, pred):
        r = [(pred[y == c] == c).mean() if (y == c).sum() else np.nan for c in range(k)]
        q = M.quadratic_weighted_kappa(y, pred, k)
        print(f"{label:26s}" + "".join(f"{v*100:9.1f}%    " for v in r)
              + f"  {q:.3f}    {macro_recall(y, pred, k):.3f}")
        return r, q

    row("as shipped (0.5 cuts)", base)

    rng = np.random.default_rng(0)
    half = rng.permutation(len(y)) % 2                     # 2-fold cross-fitting
    for objective in ("qwk", "macro-recall"):
        pred = np.zeros_like(y)
        chosen = []
        for h in (0, 1):
            tr, te = half != h, half == h
            cuts, _ = fit_cuts(score[tr], y[tr], k, "qwk" if objective == "qwk" else "mr")
            pred[te] = cuts_to_grades(score[te], cuts)
            chosen.append([round(c, 2) for c in cuts])
        row(f"recalibrated for {objective}", pred)
        print(f"{'':26s}cut-points per half: {chosen[0]} / {chosen[1]}")

    print("\nDIAGNOSTIC ONLY. Cut-points were fitted on APTOS labels, so these recalibrated "
          "numbers are NOT external validation and are not quoted as results. They measure "
          "the ceiling recoverable by recalibration given labelled local data.")


if __name__ == "__main__":
    main()
