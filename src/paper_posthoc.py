"""
paper_posthoc.py — analyses added after an internal editorial assessment of the manuscript.
They were NOT pre-specified; the manuscript labels them as post hoc. All are computed from the
archived per-image predictions; nothing is retrained.

  1. Referral recalibration sample size counted in referable cases. The threshold rule of R-C
     ("largest cut giving >= 85 % sensitivity on the labelled sample") depends only on the
     referable images in the sample, so its precision is set by their number, not by the number
     of images. For k referable cases drawn at random, the threshold's true sensitivity is
     measured on every referable image of the corpus that was not drawn (k is capped at a quarter
     of the corpus's referable images so that at least three quarters remain for evaluation).
     Harm = true sensitivity below 80 %. Screening prevalence: n images at a referable
     prevalence of 5, 10 or 20 % contain k ~ Binomial(n, prevalence) referable cases.
     With continuous scores the risk is distribution-free -- the population sensitivity of the
     cut at the j-th largest of k referable scores is Beta(j, k - j + 1) -- so the closed form
     is recorded beside the simulations, which check it on real scores (ties, finite corpora).
  2. Label-free prior adaptation against the decoding it uses: R5 and R5b decode the adapted
     posteriors by argmax, so they are compared with the argmax of the unadapted posteriors
     (R0a), on every external corpus, rather than with the 0.5 rule R1.
  3. Spectrum of the external corpora: referral AUC standardised to the development grade mix
     (each image weighted by development share / corpus share of its grade, within referable
     and within non-referable images), and AUC between adjacent grades.
  4. Kendall's tau between the model rankings by QWK at R1 and by referable sensitivity at 90 %
     specificity, with a 95 % interval from 1 000 image bootstrap resamples and a two-sided
     permutation p-value (10 000 permutations of one ranking; ties allowed).
  5. Mild-grade F1 on development data under the same cut-point objective as R4m (macro-averaged
     recall, cut-points cross-fitted across folds), for a like-for-like comparison with R4m.

Writes docs/generated/paper/posthoc.json.
Usage: python src/paper_posthoc.py --labels <dir>
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from scipy.stats import kendalltau, beta as beta_dist, binom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_rules as P
import metrics as M
from paper_phase0 import load_model, load_ext
from paper_revision import MODELS, K, mild_prf, sens_at_spec

CORPORA = ["APTOS", "DDR", "DeepDRiD", "SUSTech"]
KS = [20, 40, 60, 80, 100, 120, 160, 200, 240, 300]      # 0.85 k is an integer: exactly 85 %
PREV = [0.05, 0.10, 0.20]
NS = [100, 200, 500, 1000, 2000]
TARGET, HARM = 0.85, 0.80


# ── 1. sample size in referable cases ─────────────────────────────────────────
def threshold(pos_scores):
    s = np.sort(pos_scores)[::-1]
    return s[int(np.ceil(TARGET * len(s))) - 1]          # largest cut with sample sens >= target


def by_cases(score, y, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = score[y >= 2], score[y < 2]
    cap = len(pos) // 4
    out = {}
    for k in [k for k in KS if k <= cap]:
        sens, spec = np.empty(draws), np.empty(draws)
        for d in range(draws):
            idx = rng.permutation(len(pos))
            t = threshold(pos[idx[:k]])
            sens[d] = (pos[idx[k:]] >= t).mean()
            spec[d] = (neg < t).mean()
        out[str(k)] = {"sens_mean": float(sens.mean()), "sens_lo": float(np.percentile(sens, 2.5)),
                       "sens_hi": float(np.percentile(sens, 97.5)), "spec_mean": float(spec.mean()),
                       "p_harm": float((sens < HARM).mean())}
    return out, cap


def by_prevalence(score, y, draws=2000, seed=1):
    rng = np.random.default_rng(seed)
    pos = score[y >= 2]
    cap = len(pos) // 4
    out = {}
    for p in PREV:
        for n in NS:
            ks = rng.binomial(n, p, size=draws)
            if np.percentile(ks, 99) > cap:
                out[f"{p:.2f}_{n}"] = None                 # the corpus has too few referable cases
                continue
            harm = 0
            for k in ks:
                if k == 0:
                    harm += 1                              # no threshold can be set
                    continue
                idx = rng.permutation(len(pos))
                harm += (pos[idx[k:]] >= threshold(pos[idx[:k]])).mean() < HARM
            out[f"{p:.2f}_{n}"] = {"p_harm": harm / draws, "k_mean": float(ks.mean())}
    return out


def harm_exact(k, target=TARGET, floor=HARM):
    """Distribution-free: with continuous scores, the population sensitivity of the cut at the
    j-th largest of k referable scores (j = ceil(target k)) is Beta(j, k - j + 1)."""
    if k == 0:
        return 1.0
    j = int(np.ceil(target * k))
    return float(beta_dist.cdf(floor, j, k - j + 1))


def harm_prevalence_exact(n, p):
    ks = np.arange(n + 1)
    w = binom.pmf(ks, n, p)
    keep = w > 1e-12
    return float(sum(wi * harm_exact(int(k)) for k, wi in zip(ks[keep], w[keep])))


def required_sample_sens(k, risk):
    """Smallest sample sensitivity j/k whose cut keeps P(population sensitivity < floor) <= risk."""
    for j in range(1, k + 1):
        if beta_dist.cdf(HARM, j, k - j + 1) <= risk:
            return j / k
    return None


# ── 3. spectrum ───────────────────────────────────────────────────────────────
def weighted_auc(score, pos, w):
    """Mann-Whitney AUC with case weights; ties count one half."""
    sp, wp = score[pos], w[pos]
    sn, wn = score[~pos], w[~pos]
    o = np.argsort(sn)
    sn, wn = sn[o], wn[o]
    cw = np.concatenate([[0.0], np.cumsum(wn)])
    lo = np.searchsorted(sn, sp, side="left")
    hi = np.searchsorted(sn, sp, side="right")
    num = (wp * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))).sum()
    return float(num / (wp.sum() * wn.sum()))


def spectrum(score_ref, g, y, dev_share):
    w = np.zeros(len(y))
    for grades in ((0, 1), (2, 3, 4)):                     # non-referable, referable
        in_grp = np.isin(y, grades)
        d_tot = sum(dev_share[k] for k in grades)
        for k in grades:
            if (y == k).any():
                w[y == k] = (dev_share[k] / d_tot) / ((y == k).sum() / in_grp.sum())
    pair = lambda a, b, col: (P.auroc(g[(y == a) | (y == b), col], y[(y == a) | (y == b)] == b)
                              if (y == a).any() and (y == b).any() else float("nan"))
    return {"auc_ref": P.auroc(score_ref, y >= 2),
            "auc_ref_devmix": weighted_auc(score_ref, y >= 2, w),
            "auc_0v1": pair(0, 1, 0), "auc_1v2": pair(1, 2, 1), "auc_0v2": pair(0, 2, 1)}


# ── 4. Kendall tau with an image bootstrap ───────────────────────────────────
def tau_boot(ext, heads, n_boot=1000, seed=0):
    ms = list(ext)
    y = ext[ms[0]]["y"]
    pred = {m: P.shipped(ext[m]["dr"], heads[m]) for m in ms}
    ref = {m: P.p_greater(ext[m]["dr"], heads[m])[:, 1] for m in ms}

    def stat(idx):
        q = [M.quadratic_weighted_kappa(y[idx], pred[m][idx], K) for m in ms]
        s = [sens_at_spec(ref[m][idx], y[idx]) for m in ms]
        return kendalltau(q, s).statistic

    rng = np.random.default_rng(seed)
    full = stat(np.arange(len(y)))
    boots = np.array([stat(rng.integers(0, len(y), len(y))) for _ in range(n_boot)])
    q = [M.quadratic_weighted_kappa(y, pred[m], K) for m in ms]
    s = np.array([sens_at_spec(ref[m], y) for m in ms])
    perm = np.array([kendalltau(q, rng.permutation(s)).statistic for _ in range(10000)])
    return {"tau": float(full), "lo": float(np.nanpercentile(boots, 2.5)),
            "hi": float(np.nanpercentile(boots, 97.5)),
            "p_perm": float((np.abs(perm) >= abs(full) - 1e-12).mean())}   # two-sided, ties allowed


# ── 5. like-for-like mild F1 on development data ─────────────────────────────
def dev_mild_mr(d, head):
    s = P.expected_grade(d["dr"], head)
    pred = np.zeros(len(s), dtype=int)
    for f in sorted(set(d["folds"].tolist())):
        te = d["folds"] == f
        cuts, _ = P.fit_cuts(s[~te], d["y"][~te], "mr")
        pred[te] = P.cuts_to_grades(s[te], cuts)
    return mild_prf(d["y"], pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pred-root", default="results/predictions")
    ap.add_argument("--out", default="docs/generated/paper/posthoc.json")
    a = ap.parse_args()
    L = P.dev_labels(a.labels, "data/splits/dev_v1.json")
    reg = json.load(open(os.path.join(a.pred_root, "registry.json")))
    rev = json.load(open("docs/generated/paper/revision.json"))
    heads = {m: reg[m]["head"] for m in MODELS}
    dev = {m: load_model(a.pred_root, m, L) for m in MODELS}
    dev_share = np.bincount(dev["E08"]["y"], minlength=K) / len(dev["E08"]["y"])
    out = {"note": "post hoc; not pre-specified", "target": TARGET, "harm": HARM,
           "dev": {}, "corpora": {}}

    for m in MODELS:
        g = P.p_greater(dev[m]["dr"], heads[m])
        out["dev"][m] = {"mild_mr": dev_mild_mr(dev[m], heads[m]),
                         "spectrum": spectrum(g[:, 1], g, dev[m]["y"], dev_share)}
        print(f"dev {m:9s} mild F1 (macro-recall cuts) {out['dev'][m]['mild_mr']['f1']:.3f}", flush=True)

    for c in CORPORA:
        ext = {m: load_ext(a.pred_root, m, c) for m in MODELS}
        ext = {m: e for m, e in ext.items() if e is not None}
        y = next(iter(ext.values()))["y"]
        assert all((e["y"] == y).all() for e in ext.values())
        C = {"n": int(len(y)), "n_ref": int((y >= 2).sum()), "grade_dist": np.bincount(y, minlength=K).tolist(),
             "models": {}}
        for m, e in ext.items():
            g = P.p_greater(e["dr"], heads[m])
            R = rev["corpora"][c]["models"][m]["rules"]
            s0a, s5, s5b, s4 = (R[r]["ref_sens"] for r in ("R0a", "R5", "R5b", "R4"))
            closed = lambda s: (s - s0a) / (s4 - s0a) if abs(s4 - s0a) >= 0.02 else None
            C["models"][m] = {
                "spectrum": spectrum(g[:, 1], g, y, dev_share),
                "R5_vs_R0a": {"sens": {r: R[r]["ref_sens"] for r in ("R1", "R0a", "R5", "R5b", "R4")},
                              "qwk": {r: R[r]["qwk"] for r in ("R1", "R0a", "R5", "R5b", "R4")},
                              "mild_recall": {r: R[r]["recall"][1] for r in ("R1", "R0a", "R5", "R5b", "R4")},
                              "mild_npred": {r: R[r]["mild"]["n_pred"] for r in ("R0a", "R5", "R5b")},
                              "closed_R5": closed(s5), "closed_R5b": closed(s5b)},
                "mild_f1_R4m": R["R4m"]["mild"]["f1"],
            }
            if m in ("E08", "E11FULL"):
                cases, cap = by_cases(g[:, 1], y)
                C["models"][m]["by_cases"] = cases
                C["models"][m]["by_prevalence"] = by_prevalence(g[:, 1], y)
                pos, neg = g[y >= 2, 1], g[y < 2, 1]
                C["models"][m]["spec_at_sens"] = {str(t): float((neg < np.quantile(pos, 1 - t)).mean())
                                                  for t in (0.80, 0.85, 0.90, 0.95)}
                C["k_cap"] = cap
                print(f"{c:8s} {m:8s} harm by k " +
                      " ".join(f"{k}:{v['p_harm']:.3f}" for k, v in cases.items()), flush=True)
        C["tau"] = tau_boot(ext, heads)
        sp = [v["spectrum"] for v in C["models"].values()]
        print(f"{c:8s} tau {C['tau']['tau']:+.2f} [{C['tau']['lo']:+.2f}, {C['tau']['hi']:+.2f}] "
              f"p {C['tau']['p_perm']:.2f}; AUC ref {min(s['auc_ref'] for s in sp):.3f}-"
              f"{max(s['auc_ref'] for s in sp):.3f}, dev mix {min(s['auc_ref_devmix'] for s in sp):.3f}-"
              f"{max(s['auc_ref_devmix'] for s in sp):.3f}, 1v2 {min(s['auc_1v2'] for s in sp):.3f}-"
              f"{max(s['auc_1v2'] for s in sp):.3f}", flush=True)
        out["corpora"][c] = C
    out["exact"] = {
        "harm_by_k": {str(k): harm_exact(k) for k in KS + [10, 30, 50]},
        "harm_by_prevalence": {f"{p:.2f}_{n}": harm_prevalence_exact(n, p) for p in PREV for n in NS},
        "required_sample_sens": {str(k): {"0.05": required_sample_sens(k, 0.05),
                                          "0.10": required_sample_sens(k, 0.10)}
                                 for k in (20, 40, 60, 80, 100, 120, 160, 200, 300)},
    }
    dev_ = [abs(v["by_cases"][k]["p_harm"] - out["exact"]["harm_by_k"][k])
            for C in out["corpora"].values() for v in C["models"].values() if "by_cases" in v
            for k in v["by_cases"]]
    out["exact"]["max_abs_dev_sim"] = float(max(dev_))
    print("closed form vs simulation, max |difference|:", round(max(dev_), 3))
    print("harm at n images:", {k: round(v, 3) for k, v in out["exact"]["harm_by_prevalence"].items()})
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
