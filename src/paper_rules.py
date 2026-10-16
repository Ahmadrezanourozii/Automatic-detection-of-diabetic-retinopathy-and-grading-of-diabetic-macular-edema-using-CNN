"""
paper_rules.py — the decision-rule analyses for the journal paper, from archived predictions.

Everything here is re-analysis of per-image logits that the training and external-evaluation
runs already wrote (oof_<fold>.npz, external_<corpus>_predictions.npz). No model is run.

The four decision rules are the ones fixed in docs/PAPER_EXPERIMENTS_preregistration.md:

  R1   shipped: sigmoid > 0.5 on every ordinal threshold (model.decode)
  R2   four cut-points on the expected grade maximising QWK, fitted on the development OOF
  R3   R2 with the grade-1|2 cut moved to the lowest value giving development referable
       sensitivity >= 0.90
  R4   oracle: QWK-maximising cuts, 2-fold cross-fitted on the EXTERNAL labels (diagnostic)
  R4m  as R4 for macro-recall (diagnostic; the F1 recovery analysis)

R4/R4m use external labels and are never results; they bound what local recalibration buys.

Labels come from the corpora's own label files, not from a re-run of corpora.build (which
needs the images). The development uid scheme is the one corpora.py writes:
IDRiD_<train|test>_<Image name>, MESSIDOR2_<image_id without extension>.
"""
from __future__ import annotations
import csv, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

K_DR = 5
NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]


# ── labels ────────────────────────────────────────────────────────────────────
def _rows(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def dev_labels(labels_dir, split_path):
    """uid -> dict(dr, dme, space, corpus, fold) for the 2 260 development images."""
    out = {}
    for split, frag in (("train", "Training Labels"), ("test", "Testing Labels")):
        hits = glob.glob(os.path.join(labels_dir, "**", f"*Disease Grading_{frag}.csv"),
                         recursive=True)
        if not hits:
            raise SystemExit(f"IDRiD {frag} CSV not found under {labels_dir}")
        for r in _rows(hits[0]):
            name = (r.get("Image name") or "").strip()
            if not name:
                continue
            dme_key = [c for c in r if c and "macular" in c.lower()][0]
            out[f"IDRiD_{split}_{name}"] = {
                "dr": int(r["Retinopathy grade"]), "dme": int(r[dme_key]),
                "space": "3class", "corpus": "IDRiD"}
    hits = glob.glob(os.path.join(labels_dir, "**", "messidor_data.csv"), recursive=True)
    if not hits:
        raise SystemExit(f"messidor_data.csv not found under {labels_dir}")
    for r in _rows(hits[0]):
        if str(r.get("adjudicated_gradable", "")).strip() != "1":
            continue
        fn = r["image_id"].strip()
        if not str(r.get("adjudicated_dr_grade", "")).strip():
            continue
        out[f"MESSIDOR2_{os.path.splitext(fn)[0]}"] = {
            "dr": int(r["adjudicated_dr_grade"]),
            "dme": 2 if int(r["adjudicated_dme"]) else None,
            "ref_dme": int(r["adjudicated_dme"]),
            "space": "binary", "corpus": "Messidor-2"}
    split = json.load(open(split_path))
    keep = {}
    for u, f in split["folds"].items():
        if u not in out:
            raise SystemExit(f"split uid {u} has no label — label files and split disagree")
        keep[u] = dict(out[u], fold=int(f))
    return keep


# ── probabilities and decodes ────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def p_greater(logits, head):
    """(n, K-1) array of P(y > k). CORN's logits are conditionals and compose by product."""
    s = sigmoid(logits)
    return np.cumprod(s, axis=1) if head == "corn" else s


def expected_grade(logits, head):
    return p_greater(logits, head).sum(axis=1)


def shipped(logits, head):
    """model.decode, reimplemented in numpy: leading thresholds that fire, cummin-repaired."""
    fired = (p_greater(logits, head) > 0.5).astype(int)
    return np.minimum.accumulate(fired, axis=1).sum(axis=1).clip(0, K_DR - 1)


def cuts_to_grades(score, cuts):
    return np.searchsorted(np.asarray(cuts), score, side="right")


def macro_recall(y, p, k=K_DR):
    out = [(p[y == c] == c).mean() for c in range(k) if (y == c).any()]
    return float(np.mean(out))


def fit_cuts(score, y, objective="qwk", k=K_DR, rounds=8, grid=80):
    """Coordinate ascent over K-1 ordered cut-points (same search as tune_thresholds.py /
    analyse_mild.py, so every number in the paper comes from one optimiser)."""
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


def referral_cut(score, y, target=0.90):
    """Lowest expected-grade threshold whose development referable sensitivity >= target,
    i.e. the most specific threshold that clears the bar."""
    ref = y >= 2
    best = None
    for t in np.unique(np.round(score, 4)):
        sens = (score[ref] >= t).mean()
        if sens >= target:
            best = float(t)
        else:
            break
    return best if best is not None else float(score.min())


def crossfit_external(score, y, objective, seed=0):
    """2-fold cross-fitting on the external set: no image influences its own cuts."""
    rng = np.random.default_rng(seed)
    half = rng.permutation(len(y)) % 2
    pred = np.zeros(len(y), dtype=int)
    for h in (0, 1):
        tr, te = half != h, half == h
        cuts, _ = fit_cuts(score[tr], y[tr], objective)
        pred[te] = cuts_to_grades(score[te], cuts)
    return pred


# ── metrics ───────────────────────────────────────────────────────────────────
def cohen_kappa(y, p, k=K_DR):
    O = M.confusion(y, p, k).astype(float)
    n = O.sum()
    po = np.trace(O) / n
    pe = (O.sum(0) * O.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def auroc(score, pos):
    """Mann–Whitney AUROC with average ranks for ties."""
    from scipy.stats import rankdata
    pos = np.asarray(pos, bool)
    r = rankdata(score)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(prob, outcome, bins=10):
    prob, outcome = np.asarray(prob, float), np.asarray(outcome, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(prob[m].mean() - outcome[m].mean())
    return float(e)


def block(y, pred):
    rec, sup = M.per_class_recall(y, pred, K_DR)
    s, sp = M.binary_sens_spec(y, pred, 2)
    return {"qwk": M.quadratic_weighted_kappa(y, pred, K_DR),
            "kappa": cohen_kappa(y, pred),
            "accuracy": M.accuracy(y, pred),
            "macro_f1": M.macro_f1(y, pred, K_DR),
            "macro_recall": macro_recall(y, pred),
            "recall": rec, "support": sup,
            "ref_sens": s, "ref_spec": sp,
            "pred_dist": np.bincount(pred, minlength=K_DR).tolist(),
            "mean_pred_minus_true": float(np.mean(pred - y))}


def threshold_free(logits, head, y):
    p_ref = p_greater(logits, head)[:, 1]          # P(grade >= 2)
    ref = (y >= 2).astype(int)
    return {"ref_auroc": auroc(p_ref, ref),
            "ref_brier": float(np.mean((p_ref - ref) ** 2)),
            "ref_ece": ece(p_ref, ref),
            "grade_spearman": float(__import__("scipy.stats").stats.spearmanr(
                expected_grade(logits, head), y).correlation)}


# ── loading archived predictions ─────────────────────────────────────────────
def load_oof(dirs, labels):
    """Pool oof_<fold>.npz across one or more run directories (an assembled run's folds can
    live in several). Refuses a fold index supplied twice."""
    uids, dr, dme, seen = [], [], [], {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "oof_*.npz"))):
            f = os.path.basename(p)
            if f in seen:
                raise SystemExit(f"{f} supplied by both {seen[f]} and {d}")
            seen[f] = d
            z = np.load(p, allow_pickle=True)
            uids += [str(u) for u in z["uids"]]
            dr.append(z["dr_logits"]); dme.append(z["dme_logits"])
    if not uids:
        raise SystemExit(f"no oof_*.npz under {dirs}")
    y = np.array([labels[u]["dr"] for u in uids])
    folds = np.array([labels[u]["fold"] for u in uids])
    return {"uids": np.array(uids), "y": y, "folds": folds,
            "dr": np.concatenate(dr).astype(np.float64),
            "dme": np.concatenate(dme).astype(np.float64)}


def load_external(path):
    z = np.load(path, allow_pickle=True)
    return {"uids": np.array([str(u) for u in z["uids"]]), "y": np.asarray(z["y_dr"]).astype(int),
            "dr": np.asarray(z["dr_logits"], dtype=np.float64)}


# ── the rule matrix for one model on one external corpus ────────────────────
def rule_matrix(dev, ext, head, target_sens=0.90):
    s_dev = expected_grade(dev["dr"], head)
    s_ext = expected_grade(ext["dr"], head)
    y_dev, y_ext = dev["y"], ext["y"]
    cuts_qwk, _ = fit_cuts(s_dev, y_dev, "qwk")
    t_ref = referral_cut(s_dev, y_dev, target_sens)
    cuts_r3 = list(cuts_qwk)
    cuts_r3[1] = t_ref
    cuts_r3[0] = min(cuts_r3[0], t_ref)
    cuts_r3[2] = max(cuts_r3[2], t_ref)
    cuts_r3[3] = max(cuts_r3[3], cuts_r3[2])
    preds = {
        "R1": shipped(ext["dr"], head),
        "R2": cuts_to_grades(s_ext, cuts_qwk),
        "R3": cuts_to_grades(s_ext, cuts_r3),
        "R4": crossfit_external(s_ext, y_ext, "qwk"),
        "R4m": crossfit_external(s_ext, y_ext, "mr"),
    }
    out = {r: block(y_ext, p) for r, p in preds.items()}
    out["_cuts"] = {"R2": [round(c, 4) for c in cuts_qwk],
                    "R3": [round(c, 4) for c in cuts_r3]}
    out["_threshold_free"] = threshold_free(ext["dr"], head, y_ext)
    return out, preds


def dev_matched(dev, head):
    """Development QWK at shipped cuts and at cross-fitted (other-fold) QWK cuts."""
    s = expected_grade(dev["dr"], head)
    y, folds = dev["y"], dev["folds"]
    pred = np.zeros(len(y), dtype=int)
    for f in sorted(set(folds.tolist())):
        te = folds == f
        cuts, _ = fit_cuts(s[~te], y[~te], "qwk")
        pred[te] = cuts_to_grades(s[te], cuts)
    return {"shipped": block(y, shipped(dev["dr"], head)), "matched": block(y, pred),
            "n": int(len(y)), "folds": sorted(set(folds.tolist()))}, pred
