"""
paper_revision.py — the no-training analyses of docs/PAPER_REVISION_preregistration.md
(R-A1, R-A4, R-B, R-C, R-D), computed from the archived per-image predictions.

For every pre-registered development model and every external corpus with predictions:
  * discrimination per threshold (AUROC of P(y > k), k = 0..3) and Harrell's C of the expected
    grade, development (out of fold) and external                                  [R-D]
  * the mean expected grade given the true grade, development vs external          [R-A1]
  * grade distributions                                                             [R-A4]
  * label-free decision rules: EM prior adaptation (R5) and black-box shift estimation (R5b),
    with the unadjusted argmax (R0a) as a decoding reference                       [R-B]
  * mild-grade precision / F1 and full confusion matrices under every rule          [R-D]
  * referable sensitivity at a fixed 90 % specificity, and the model ranking by it vs by QWK
                                                                                    [R-D]
  * recalibration sample size with the referral objective (selected and reference model)
                                                                                    [R-C]
Writes docs/generated/paper/revision.json.

Usage: python src/paper_revision.py --labels <dir> [--corpora APTOS,DDR,DeepDRiD,SUSTech]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from scipy.stats import kendalltau, rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_rules as P
import metrics as M
from paper_phase0 import load_model, load_ext

MODELS = ["E08", "E09", "E10", "E11FULL", "E14MAC", "E15LPFT", "E17NAT", "E20CORAL",
          "E22CORN", "E21CNXA"]
K = 5


# ── class probabilities from an ordinal head ──────────────────────────────────
def class_probs(logits, head):
    g = np.minimum.accumulate(P.p_greater(logits, head), axis=1)    # P(y > k), monotone
    n = len(g)
    p = np.empty((n, K))
    p[:, 0] = 1 - g[:, 0]
    for k in range(1, K - 1):
        p[:, k] = g[:, k - 1] - g[:, k]
    p[:, K - 1] = g[:, K - 2]
    p = np.clip(p, 1e-6, None)
    return p / p.sum(axis=1, keepdims=True)


def em_prior(p, prior_s, iters=1000, tol=1e-8):
    """Saerens, Latinne & Decaestecker (2002): EM estimate of the target prior."""
    pt = prior_s.copy()
    for _ in range(iters):
        q = p * (pt / prior_s)
        q /= q.sum(axis=1, keepdims=True)
        new = q.mean(axis=0)
        if np.abs(new - pt).max() < tol:
            pt = new
            break
        pt = new
    return pt


def bbse_prior(y_dev, pred_dev, pred_ext, prior_true_dev):
    """Lipton, Wang & Smola (2018): w = C^-1 mu with C[i, j] = P(yhat = i, y = j) on the
    development set; negative weights clipped at 0. Returns the implied target prior."""
    C = np.zeros((K, K))
    for i, j in zip(pred_dev, y_dev):
        C[i, j] += 1
    C /= len(y_dev)
    mu = np.bincount(pred_ext, minlength=K) / len(pred_ext)
    w, *_ = np.linalg.lstsq(C, mu, rcond=None)
    w = np.clip(w, 0, None)
    pt = w * prior_true_dev
    return pt / pt.sum() if pt.sum() > 0 else prior_true_dev


def reweight(p, pt, ps):
    q = p * (pt / ps)
    return q / q.sum(axis=1, keepdims=True)


# ── metrics ───────────────────────────────────────────────────────────────────
def mild_prf(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y != 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred != 1)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "n_pred": int((pred == 1).sum())}


def summary(y, pred):
    b = P.block(y, pred)
    return {"qwk": b["qwk"], "accuracy": b["accuracy"], "ref_sens": b["ref_sens"],
            "ref_spec": b["ref_spec"], "recall": b["recall"],
            "mean_pred_minus_true": b["mean_pred_minus_true"], "mild": mild_prf(y, pred),
            "confusion": M.confusion(y, pred, K).tolist()}


def cindex(score, y):
    """Harrell's C for an ordinal outcome: concordance over pairs with different grades, ties
    in score counted as one half (pooled pairwise Mann-Whitney)."""
    num = den = 0.0
    for j in range(K):
        for k in range(j + 1, K):
            a, b = score[y == j], score[y == k]
            if not len(a) or not len(b):
                continue
            r = rankdata(np.concatenate([a, b]))
            u = r[len(a):].sum() - len(b) * (len(b) + 1) / 2
            num += u; den += len(a) * len(b)
    return float(num / den) if den else float("nan")


def discrimination(logits, head, y):
    g = P.p_greater(logits, head)
    return {"auc_k": [P.auroc(g[:, k], y > k) for k in range(K - 1)],
            "cindex": cindex(P.expected_grade(logits, head), y)}


def sens_at_spec(score, y, spec=0.90):
    neg, pos = score[y < 2], score[y >= 2]
    if not len(neg) or not len(pos):
        return float("nan")
    t = np.quantile(neg, spec)
    return float((pos > t).mean())


def es_by_grade(logits, head, y):
    s = P.expected_grade(logits, head)
    return [float(s[y == k].mean()) if (y == k).any() else float("nan") for k in range(K)]


# ── R-C: referral recalibration sample size ──────────────────────────────────
def sample_size(score, y, sizes=(25, 50, 100, 200, 400, 800), draws=500, target=0.85,
                harm=0.80, seed=0):
    rng = np.random.default_rng(seed)
    ref = y >= 2
    out = {}
    for n in sizes:
        if n >= len(y) - 50:
            continue
        sens, spec, skipped = [], [], 0
        for _ in range(draws):
            idx = rng.permutation(len(y))
            tr, te = idx[:n], idx[n:]
            pos = np.sort(score[tr][ref[tr]])[::-1]
            if not len(pos):
                skipped += 1
                continue
            k = int(np.ceil(target * len(pos)))
            t = pos[k - 1]                                  # largest t with sens >= target
            st, rt = score[te], ref[te]
            sens.append(float((st[rt] >= t).mean())); spec.append(float((st[~rt] < t).mean()))
        sens, spec = np.array(sens), np.array(spec)
        out[str(n)] = {"draws": int(len(sens)), "skipped_no_positive": skipped,
                       "sens_mean": float(sens.mean()), "sens_lo": float(np.percentile(sens, 2.5)),
                       "sens_hi": float(np.percentile(sens, 97.5)),
                       "spec_mean": float(spec.mean()), "spec_lo": float(np.percentile(spec, 2.5)),
                       "spec_hi": float(np.percentile(spec, 97.5)),
                       "p_harm": float((sens < harm).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pred-root", default="results/predictions")
    ap.add_argument("--corpora", default="APTOS,DDR,DeepDRiD,SUSTech")
    ap.add_argument("--out", default="docs/generated/paper/revision.json")
    a = ap.parse_args()
    L = P.dev_labels(a.labels, "data/splits/dev_v1.json")
    reg = json.load(open(os.path.join(a.pred_root, "registry.json")))
    out = {"models": MODELS, "dev": {}, "corpora": {}}

    dev = {}
    for m in MODELS:
        d = load_model(a.pred_root, m, L)
        head = reg[m]["head"]
        p = class_probs(d["dr"], head)
        dev[m] = {"d": d, "head": head, "p": p, "prior_s": p.mean(axis=0),
                  "prior_true": np.bincount(d["y"], minlength=K) / len(d["y"]),
                  "pred_r1": P.shipped(d["dr"], head)}
        out["dev"][m] = {**discrimination(d["dr"], head, d["y"]),
                         "es_by_grade": es_by_grade(d["dr"], head, d["y"]),
                         "grade_dist": np.bincount(d["y"], minlength=K).tolist(),
                         "summary_R1": summary(d["y"], dev[m]["pred_r1"])}

    for corpus in [c for c in a.corpora.split(",") if c]:
        have = [m for m in MODELS if corpus in reg[m].get("external", {})]
        if not have:
            print(f"{corpus}: no predictions yet")
            continue
        C = {"models": {}, "n": None}
        for m in have:
            e = load_ext(a.pred_root, m, corpus)
            v = dev[m]; head = v["head"]; y = e["y"]
            C["n"] = int(len(y)); C["grade_dist"] = np.bincount(y, minlength=K).tolist()
            rm, preds = P.rule_matrix(v["d"], e, head)
            p = class_probs(e["dr"], head)
            pt_em = em_prior(p, v["prior_s"])
            pt_bb = bbse_prior(v["d"]["y"], v["pred_r1"], P.shipped(e["dr"], head), v["prior_true"])
            preds["R0a"] = p.argmax(axis=1)
            preds["R5"] = reweight(p, pt_em, v["prior_s"]).argmax(axis=1)
            preds["R5b"] = reweight(p, pt_bb, v["prior_s"]).argmax(axis=1)
            rules = {r: summary(y, preds[r]) for r in ("R1", "R2", "R3", "R4", "R4m", "R0a", "R5", "R5b")}
            s1, s4 = rules["R1"]["ref_sens"], rules["R4"]["ref_sens"]
            gap = {r: ((rules[r]["ref_sens"] - s1) / (s4 - s1) if abs(s4 - s1) >= 0.02 else None)
                   for r in ("R0a", "R5", "R5b")}
            es_e = es_by_grade(e["dr"], head, y)
            es_d = out["dev"][m]["es_by_grade"]
            C["models"][m] = {
                **discrimination(e["dr"], head, y),
                "es_by_grade": es_e,
                "n_grades_above_dev": int(sum(x > z for x, z in zip(es_e, es_d))),
                "n_grades_below_dev": int(sum(x < z for x, z in zip(es_e, es_d))),
                "prior_em": pt_em.tolist(), "prior_bbse": pt_bb.tolist(),
                "rules": rules, "gap_closed_ref_sens": gap,
                "sens_at_spec90": sens_at_spec(P.p_greater(e["dr"], head)[:, 1], y),
            }
            if m in ("E08", "E11FULL"):
                C["models"][m]["sample_size"] = sample_size(P.p_greater(e["dr"], head)[:, 1], y)
            r = C["models"][m]
            print(f"{corpus:8s} {m:9s} C {r['cindex']:.3f} AUCk {[round(x, 3) for x in r['auc_k']]} "
                  f"above/below {r['n_grades_above_dev']}/{r['n_grades_below_dev']}  sens R1 {s1:.3f} "
                  f"R4 {s4:.3f} R5 {rules['R5']['ref_sens']:.3f} R5b {rules['R5b']['ref_sens']:.3f} "
                  f"gap {gap}", flush=True)
        ms = list(C["models"])
        q = [C["models"][m]["rules"]["R1"]["qwk"] for m in ms]
        s = [C["models"][m]["sens_at_spec90"] for m in ms]
        tau = kendalltau(q, s)
        C["kendall_qwkR1_vs_sens_at_spec90"] = {"tau": float(tau.statistic), "p": float(tau.pvalue),
                                                "best_by_qwk": ms[int(np.argmax(q))],
                                                "best_by_sens": ms[int(np.argmax(s))]}
        C["A1_above"] = sum(C["models"][m]["n_grades_above_dev"] >= 3 for m in ms)
        C["A1_below"] = sum(C["models"][m]["n_grades_below_dev"] >= 3 for m in ms)
        g = [C["models"][m]["gap_closed_ref_sens"]["R5"] for m in ms]
        C["B_n_models_R5_closes_25pct"] = sum(1 for x in g if x is not None and x >= 0.25)
        out["corpora"][corpus] = C
        print(f"{corpus}: Kendall tau {tau.statistic:.2f}; A1 above {C['A1_above']} below "
              f"{C['A1_below']}; R5 closes >=25% in {C['B_n_models_R5_closes_25pct']} models")
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
