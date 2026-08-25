"""
compare_variants.py — paired bootstrap between configurations of one run.

Two configurations are compared on the SAME units, resampling whole groups. If the interval
of the difference contains zero, they are indistinguishable and must be reported that way —
not ranked anyway (PROTOCOL.md §4).

This exists as its own script because the tempting alternative — eyeballing two overlapping
confidence intervals — is wrong. Overlapping intervals do not imply a non-significant
difference, and non-overlapping ones are not required for significance. Only the paired
interval of the difference answers the question.

Usage:
    python src/compare_variants.py runs/E01/results.json
"""
from __future__ import annotations
import itertools, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

HEADS = [
    ("dr_official", 5, "DR 5-class, official test"),
    ("dme_official_ungated", 3, "DME 3-class ungated [PRIMARY]"),
    ("dme_official_gated", 3, "DME 3-class gated DR>=1 [secondary]"),
]


def main(path):
    with open(path) as f:
        res = json.load(f)
    variants = list(res["results"].keys())
    print(f"paired bootstrap over groups — {os.path.basename(path)}  "
          f"(commit {res['commit'][:10]})")

    out = {}
    for head, k, title in HEADS:
        print(f"\n{'='*76}\n{title}\n{'='*76}")
        base = res["results"][variants[0]][head]
        y = np.array(base["y_true"])
        groups = np.array(base["groups"])
        print(f"  n={len(y)}   majority floor {base['majority_floor']*100:.1f}%")

        print(f"\n  {'variant':20s} {'acc':>7s} {'acc 95% CI':>16s} "
              f"{'QWK':>7s} {'QWK 95% CI':>16s}")
        for v in variants:
            r = res["results"][v][head]
            print(f"  {v:20s} {r['accuracy']*100:6.1f}% "
                  f"[{r['accuracy_ci95'][0]*100:5.1f},{r['accuracy_ci95'][1]*100:5.1f}] "
                  f"{r['qwk']:7.3f} [{r['qwk_ci95'][0]:6.3f},{r['qwk_ci95'][1]:6.3f}]")

        print(f"\n  pairwise differences (A - B), paired over the same {len(set(groups))} groups:")
        for a, b in itertools.combinations(variants, 2):
            ra, rb = res["results"][a][head], res["results"][b][head]
            assert ra["y_true"] == rb["y_true"], "variants evaluated on different rows"
            pa, pb = np.array(ra["y_pred"]), np.array(rb["y_pred"])
            for mname, fn in (("acc", M.accuracy),
                              ("QWK", lambda t, p: M.quadratic_weighted_kappa(t, p, k))):
                d, lo, hi, sig = M.paired_bootstrap_diff(y, pa, pb, fn, groups, seed=0)
                scale = 100 if mname == "acc" else 1
                verdict = "SIGNIFICANT" if sig else "indistinguishable"
                print(f"    {a:18s} - {b:18s}  {mname:3s}  "
                      f"{d*scale:+7.2f}  [{lo*scale:+7.2f},{hi*scale:+7.2f}]  {verdict}")
                out[f"{head}|{a}-{b}|{mname}"] = {
                    "diff": d, "ci95": [lo, hi], "significant": sig}

    dst = os.path.join(os.path.dirname(path), "comparisons.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {dst}")
    n_sig = sum(1 for v in out.values() if v["significant"])
    print(f"{n_sig} of {len(out)} pairwise comparisons are significant at 95%.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/E01/results.json")
