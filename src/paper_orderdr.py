"""
paper_orderdr.py — the decision-rule analysis applied to a PUBLISHED model family.

Sheng et al. (ORDER-DR, Front Endocrinol 2026) trained on APTOS 2019 and validated externally
on Messidor-2 -- the direction in which the field's external grade-1 collapses are reported --
and released, per seed, the image-level class probabilities of their models on their APTOS
validation split and on all 1 744 Messidor-2 images (github.com/Hajimi-Sudo/ORDER-DR,
results/results_dr). No weights are needed: every rule below is a function of those
probabilities.

Models: the class-balanced EfficientNet-B0 branch (CB-384), the ordinal-risk branch
(LORS-384), Swin-T, and the ORDER-DR fusion w*CB + (1-w)*LORS with the authors' own per-seed
cut-points for the fixed 0.5/0.5 fusion (order_dr_final/order_dr_seedwise_weights.csv,
"ORDER-DR-mean2"), which reproduces their
reported external numbers before anything else is computed.

Rules (as in docs/PAPER_EXPERIMENTS_preregistration.md, adapted to softmax outputs):
  R1   argmax
  R2   QWK-maximising cut-points on the expected grade sum_k k*p_k, fitted on THEIR APTOS
       validation predictions (this is the ORDER-DR rule)
  R4   QWK-maximising cut-points, 2-fold cross-fitted on Messidor-2 labels (diagnostic)
  R4m  as R4, maximising macro-recall (diagnostic)

Writes docs/generated/paper/orderdr_reanalysis.json.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_rules as P
import metrics as M

SEEDS = ("20260617", "20260618", "20260619")
BRANCH = {"CB-384": ("order_dr_eval", "paper_v2_efficientnet_b0_balanced_384_seed{s}"),
          "LORS-384": ("order_dr_eval", "paper_v2_lors_efficientnet_b0_balanced_384_seed{s}"),
          "Swin-T": ("swin_t_r1_eval_final", "paper_swin_t_balanced_384_seed{s}")}


def read_probs(path):
    rows = list(csv.DictReader(open(path)))
    p = np.array([[float(r[f"prob_grade_{k}"]) for k in range(5)] for r in rows])
    y = np.array([int(float(r["label"])) for r in rows])
    ids = [r.get("id_code", str(i)) for i, r in enumerate(rows)]
    return p, y, ids


def egrade(p):
    return (p * np.arange(5)).sum(axis=1)


def f1_class(y, pred, c):
    tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum()
    fn = ((pred != c) & (y == c)).sum()
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def summarise(y, pred):
    b = P.block(y, pred)
    b["mild_f1"] = f1_class(y, pred, 1)
    b["mild_pred"] = int((pred == 1).sum())
    b["mild_precision"] = float(((pred == 1) & (y == 1)).sum() / max(1, (pred == 1).sum()))
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="clone of github.com/Hajimi-Sudo/ORDER-DR")
    ap.add_argument("--out", default="docs/generated/paper/orderdr_reanalysis.json")
    a = ap.parse_args()
    R = os.path.join(a.repo, "results", "results_dr")

    weights = {}
    for r in csv.DictReader(open(os.path.join(R, "order_dr_final", "order_dr_seedwise_weights.csv"))):
        # the published ORDER-DR is the FIXED 0.5/0.5 fusion: the repository's grouped table
        # gives external QWK 0.6423 +/- 0.0364 for "ORDER-DR-mean2", the paper's Table 6
        # value, whereas its "ORDER-DR" row (tuned weights) gives 0.6357 +/- 0.0416
        if r["method"] == "ORDER-DR-mean2":
            weights[r["seed"]] = (float(r["efficientnet_weight"]), float(r["lors_weight"]),
                                  [float(t) for t in r["thresholds"].split("|")])

    out = {"source": "github.com/Hajimi-Sudo/ORDER-DR results/results_dr", "models": {}}
    for s in SEEDS:
        ext, val, ids0 = {}, {}, None
        for name, (sub, pat) in BRANCH.items():
            run = pat.format(s=s)
            pe, ye, ids = read_probs(os.path.join(R, sub, f"{run}_external_predictions.csv"))
            pv, yv, _ = read_probs(os.path.join(R, run, "valid_predictions.csv"))
            if ids0 is None:
                ids0, y_ext = ids, ye
            assert ids == ids0 and (ye == y_ext).all(), "external rows not aligned across models"
            ext[name], val[name] = pe, (pv, yv)
        w_cb, w_lors, cuts = weights[s]
        ext["ORDER-DR"] = w_cb * ext["CB-384"] + w_lors * ext["LORS-384"]
        val["ORDER-DR"] = (w_cb * val["CB-384"][0] + w_lors * val["LORS-384"][0], val["CB-384"][1])
        for name, pe in ext.items():
            pv, yv = val[name]
            se, sv = egrade(pe), egrade(pv)
            if name == "ORDER-DR":
                c2 = cuts                                  # the authors' published cut-points
            else:
                c2, _ = P.fit_cuts(sv, yv, "qwk", grid=P.GRID_DEV)
            preds = {"R1": pe.argmax(1), "R2": P.cuts_to_grades(se, c2),
                     "R4": P.crossfit_external(se, y_ext, "qwk"),
                     "R4m": P.crossfit_external(se, y_ext, "mr")}
            entry = {r: summarise(y_ext, p) for r, p in preds.items()}
            entry["_threshold_free"] = {
                "ref_auroc": P.auroc(pe[:, 2:].sum(1), y_ext >= 2),
                "grade_spearman": float(__import__("scipy.stats").stats.spearmanr(se, y_ext).correlation)}
            out["models"].setdefault(name, {})[s] = entry
    # seed means and SDs, the way the paper reports them
    agg = {}
    for name, by_seed in out["models"].items():
        agg[name] = {}
        for r in ("R1", "R2", "R4", "R4m"):
            vals = {k: [by_seed[s][r][k] if not isinstance(by_seed[s][r][k], list) else None
                        for s in SEEDS] for k in ("qwk", "accuracy", "mild_f1", "mild_pred",
                                                    "mild_precision", "ref_sens", "ref_spec")}
            vals["mild_recall"] = [by_seed[s][r]["recall"][1] for s in SEEDS]
            agg[name][r] = {k: [float(np.mean(v)), float(np.std(v, ddof=1))]
                            for k, v in vals.items()}
    out["seed_mean_sd"] = agg
    out["n_external"] = int(len(y_ext))
    out["n_mild"] = int((y_ext == 1).sum())
    json.dump(out, open(a.out, "w"), indent=1, default=float)

    print(f"Messidor-2 n={len(y_ext)}, grade 1 n={(y_ext == 1).sum()}")
    for name, rr in agg.items():
        for r, v in rr.items():
            print(f"{name:9s} {r:4s} QWK {v['qwk'][0]:.4f}±{v['qwk'][1]:.4f}  acc {v['accuracy'][0]:.3f}  "
                  f"mild recall {v['mild_recall'][0]*100:5.1f}  F1 {v['mild_f1'][0]:.3f}  "
                  f"pred-1 {v['mild_pred'][0]:6.1f}  prec {v['mild_precision'][0]:.3f}  "
                  f"sens {v['ref_sens'][0]*100:5.1f} spec {v['ref_spec'][0]*100:5.1f}")


if __name__ == "__main__":
    main()
