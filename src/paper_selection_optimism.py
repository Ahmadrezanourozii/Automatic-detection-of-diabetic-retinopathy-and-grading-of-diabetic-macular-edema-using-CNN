"""
paper_selection_optimism.py — how optimistic is selecting each fold's epoch on that fold?

train.py keeps, for every fold, the epoch with the best validation score ON THAT FOLD, and
the out-of-fold (OOF) predictions are taken from that epoch. The fold therefore helped pick
its own checkpoint, so every development-pool number is mildly optimistic. External numbers
are not affected: no external image influenced any choice.

The size of the optimism is estimated from the archived per-epoch histories, without
retraining: for fold f, pick the epoch that maximises the selection score averaged over the
OTHER folds (cross-fitted epoch selection), and compare fold f's DR QWK at that epoch with its
QWK at its own best epoch. Per-epoch histories are decoded at the shipped cut-points and
without TTA, so this measures the selection effect on that quantity.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np


def optimism(results_paths):
    folds = []
    for p in results_paths:
        folds += json.load(open(p))["folds"]
    by = {f["fold"]: f for f in folds}
    n_ep = min(len(f["history"]) for f in folds)
    S = np.array([[h["score"] for h in by[k]["history"][:n_ep]] for k in sorted(by)])
    Q = np.array([[h["dr_qwk"] for h in by[k]["history"][:n_ep]] for k in sorted(by)])
    rows = []
    for i, k in enumerate(sorted(by)):
        own = int(np.argmax(S[i]))
        others = np.delete(S, i, axis=0).mean(axis=0)
        cross = int(np.argmax(others))
        rows.append({"fold": k, "own_epoch": own, "crossfit_epoch": cross,
                     "qwk_own": float(Q[i, own]), "qwk_crossfit": float(Q[i, cross]),
                     "qwk_last": float(Q[i, -1])})
    d = np.array([r["qwk_own"] - r["qwk_crossfit"] for r in rows])
    return {"per_fold": rows, "mean_optimism_dr_qwk": float(d.mean()),
            "max_optimism_dr_qwk": float(d.max()), "n_epochs": n_ep}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="NAME=path/results.json[+path2/results.json]")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = {}
    for spec in a.runs:
        name, paths = spec.split("=")
        out[name] = optimism(paths.split("+"))
        r = out[name]
        print(f"{name:9s} mean optimism {r['mean_optimism_dr_qwk']:+.4f}  max "
              f"{r['max_optimism_dr_qwk']:+.4f}  "
              + " ".join(f"f{x['fold']}:{x['own_epoch']}/{x['crossfit_epoch']}"
                         for x in r["per_fold"]))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
