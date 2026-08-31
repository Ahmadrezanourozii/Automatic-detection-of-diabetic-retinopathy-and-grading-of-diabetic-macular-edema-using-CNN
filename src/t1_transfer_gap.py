"""
t1_transfer_gap.py — T1's actual deliverable: how far a threshold fitted on the development
pool lands from its target on the external corpus.

WHY THIS AND NOT "the model's sensitivity". `sigmoid > 0.5` gives 86.28 % referable-DR
sensitivity on the development pool and 99.53 % on APTOS — the same model at the same
cut-point, at opposite ends of its own operating curve. A single achieved sensitivity is
therefore not a property of the model and cannot be reported as one. What can be reported is
the *transfer gap*: choose a target, fit the cut-point where PROTOCOL.md allows it (the
development pool, cross-fitted, never APTOS), then measure how far the achieved sensitivity
lands from the target on a corpus the threshold has never seen. That distance, with its sign
and its interval, is the result.

Targets come from `docs/T1_referral_threshold_candidates.md`, which sources them to published
standards and flags the one that could not be verified at primary source.

Usage:
    python src/t1_transfer_gap.py --dev runs/E08 --external runs/E08X2 --datasets <root>
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpora
import metrics

REFERABLE_FROM = 2          # referable DR is grade >= 2; the ordinal cut is P(y > 1)
CUT_INDEX = 1

TARGETS = [
    (0.80, "A", "British Diabetic Association 1997 — minimum standard, referable DR "
                "(PROVENANCE UNVERIFIED at primary source)"),
    (0.85, "B", "IDx-DR pivotal trial — FDA pre-specified superiority endpoint (> 85 %), mtmDR"),
    (0.872, "B*", "IDx-DR pivotal trial — achieved sensitivity, mtmDR"),
    (0.955, "C", "EyeArt pivotal trial — reported performance, mtmDR (not a standard)"),
]


def sens_spec(score, y_ref, thr):
    pred = score > thr
    P, N = y_ref.sum(), (~y_ref).sum()
    return (pred & y_ref).sum() / max(1, P), (~pred & ~y_ref).sum() / max(1, N)


def fit_threshold(score, y_ref, target):
    """Lowest threshold whose sensitivity is still >= target (highest specificity that
    meets the target). Deterministic; ties broken toward specificity."""
    order = np.argsort(-score)
    tp = np.cumsum(y_ref[order])
    sens = tp / max(1, y_ref.sum())
    i = int(np.argmax(sens >= target)) if (sens >= target).any() else len(sens) - 1
    return float(score[order][i])


def load_dev(run, datasets):
    rows = corpora.build(datasets, ("IDRiD", "Messidor-2"))
    by = {r["uid"]: r for r in rows}
    split = json.load(open("data/splits/dev_v1.json"))
    uids, logits, folds = [], [], []
    for p in sorted(glob.glob(os.path.join(run, "oof_*.npz"))):
        f = int(os.path.basename(p).split("_")[1].split(".")[0])
        z = np.load(p, allow_pickle=True)
        us = [str(u) for u in z["uids"]]
        uids += us; folds += [f] * len(us); logits.append(z["dr_logits"])
    L = np.concatenate(logits)
    score = 1 / (1 + np.exp(-L[:, CUT_INDEX]))
    y = np.array([by[u]["dr"] for u in uids])
    groups = np.array([split["groups"].get(u, u) for u in uids])
    return score, (y >= REFERABLE_FROM), np.array(folds), groups


def load_external(run, fname="external_aptos_predictions.npz"):
    z = np.load(os.path.join(run, fname), allow_pickle=True)
    score = 1 / (1 + np.exp(-z["dr_logits"][:, CUT_INDEX]))
    y = np.asarray(z["y_dr"])
    groups = np.asarray([str(g) for g in z["groups"]])
    return score, (y >= REFERABLE_FROM), groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--external", required=True)
    ap.add_argument("--ext-file", default="external_aptos_predictions.npz",
                    help="prediction file inside --external. I22 writes recipe-named "
                         "files, e.g. external_aptos_f0_notta_predictions.npz")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="docs/generated/t1_transfer_gap.md")
    a = ap.parse_args()

    ds, dy, dfold, dgrp = load_dev(a.dev, a.datasets)
    xs, xy, xgrp = load_external(a.external, a.ext_file)
    print(f"dev  n={len(ds)}  referable prevalence {dy.mean()*100:.1f}%")
    print(f"ext  n={len(xs)}  referable prevalence {xy.mean()*100:.1f}%")

    rng = np.random.default_rng(0)
    xindex = metrics._group_index(xgrp)

    rows = []
    for target, tag, prov in TARGETS:
        # (a) cross-fitted on dev: fold f's threshold comes from the other folds only.
        #     This is the honest estimate of what the procedure achieves in-domain.
        cf_sens, cf_spec = [], []
        for f in sorted(set(dfold.tolist())):
            te, tr = dfold == f, dfold != f
            thr = fit_threshold(ds[tr], dy[tr], target)
            s, sp = sens_spec(ds[te], dy[te], thr)
            cf_sens.append(s); cf_spec.append(sp)
        cf_sens, cf_spec = float(np.mean(cf_sens)), float(np.mean(cf_spec))

        # (b) the threshold you would actually ship: fitted on the whole dev pool.
        thr_ship = fit_threshold(ds, dy, target)
        d_sens, d_spec = sens_spec(ds, dy, thr_ship)

        # (c) applied unchanged to the external corpus, which it has never seen.
        x_sens, x_spec = sens_spec(xs, xy, thr_ship)

        # interval on the external sensitivity, resampling whole groups
        boots = []
        for _ in range(a.n_boot):
            idx = metrics._resample_groups(xindex, rng)
            s, _ = sens_spec(xs[idx], xy[idx], thr_ship)
            boots.append(s)
        lo, hi = np.percentile(boots, [2.5, 97.5])

        rows.append(dict(target=target, tag=tag, provenance=prov, threshold=thr_ship,
                         dev_crossfit_sens=cf_sens, dev_crossfit_spec=cf_spec,
                         dev_sens=float(d_sens), dev_spec=float(d_spec),
                         ext_sens=float(x_sens), ext_spec=float(x_spec),
                         ext_sens_lo=float(lo), ext_sens_hi=float(hi),
                         gap=float(x_sens - target)))

    hdr = ["# T1 — transfer gap: a threshold fitted on the development pool, measured on APTOS",
           "",
           f"Model: `{os.path.basename(a.dev)}` weights. Dev pool n={len(ds)} "
           f"(referable prevalence {dy.mean()*100:.1f} %); APTOS n={len(xs)} "
           f"(prevalence {xy.mean()*100:.1f} %). Thresholds are fitted on the development "
           "pool only — never on APTOS (`PROTOCOL.md` §3, §6.1).", "",
           "| target | source | threshold | dev sens (cross-fitted) | dev sens (shipped fit) | dev spec | **APTOS sens** | 95 % CI | APTOS spec | **transfer gap** |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        hdr.append(
            f"| **{r['target']*100:.1f} %** | {r['tag']} | {r['threshold']:.4f} | "
            f"{r['dev_crossfit_sens']*100:.2f} % | {r['dev_sens']*100:.2f} % | "
            f"{r['dev_spec']*100:.2f} % | **{r['ext_sens']*100:.2f} %** | "
            f"[{r['ext_sens_lo']*100:.2f}, {r['ext_sens_hi']*100:.2f}] | "
            f"{r['ext_spec']*100:.2f} % | **{r['gap']*100:+.2f} pts** |")
    body = "\n".join(hdr)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(body + "\n")
    json.dump(rows, open(a.out.replace(".md", ".json"), "w"), indent=1)
    print("\n" + body)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
