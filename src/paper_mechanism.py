"""
paper_mechanism.py — revision outcomes A2 (image degradation) and A3 (image statistics), from the
outputs of src/paper_degrade.py (kernel X5NEW). Writes docs/generated/paper/mechanism.json.

A2 holds for a model if at least one degradation lowers the mean predicted-minus-true grade at the
shipped rule by at least 0.10 AND lowers referable sensitivity by at least 10 points, relative to
the undegraded out-of-fold re-scoring; the outcome is reported per model and held overall only if
it holds for both re-scored models.

Usage: python src/paper_mechanism.py --search <dirs>
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_rules as P

CONDS = ["orig", "jpeg30_512", "down256", "blur2", "gamma18", "contrast60"]
LABELS = {"orig": "None (re-scored control)", "jpeg30_512": "JPEG quality 30 at 512 px",
          "down256": "Downsampled to 256 px", "blur2": "Gaussian blur, sigma 2 px",
          "gamma18": "Gamma 1.8 (darker)", "contrast60": "Contrast reduced to 60%"}
SHORT = {"jpeg30_512": "jpeg", "down256": "down", "blur2": "blur", "gamma18": "gamma", "contrast60": "contrast"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", nargs="+", required=True)
    ap.add_argument("--out", default="docs/generated/paper/mechanism.json")
    a = ap.parse_args()
    files = [f for r in a.search for f in glob.glob(os.path.join(r, "**", "degrade_*.npz"), recursive=True)]
    stats = [f for r in a.search for f in glob.glob(os.path.join(r, "**", "image_stats.json"), recursive=True)]
    out = {"conditions": [c for c in CONDS if c != "orig"], "labels": LABELS, "short": SHORT,
           "models": {}, "stats": json.load(open(stats[0])) if stats else {}}
    by = {}
    for f in files:
        name = os.path.basename(f)[len("degrade_"):-4]
        m, cond = name.split("_", 1)
        by.setdefault(m, {})[cond] = f
    verdicts = []
    for m, conds in by.items():
        if "orig" not in conds:
            continue
        res = {}
        for cond in CONDS:
            if cond not in conds:
                continue
            z = np.load(conds[cond], allow_pickle=True)
            y = np.asarray(z["y_dr"]).astype(int)
            lg = np.asarray(z["dr_logits"], dtype=np.float64)
            b = P.block(y, P.shipped(lg, "ordinal"))
            res[cond] = {"mean": b["mean_pred_minus_true"], "sens": b["ref_sens"], "spec": b["ref_spec"],
                         "qwk": b["qwk"], "auroc": P.auroc(P.p_greater(lg, "ordinal")[:, 1], y >= 2)}
        for cond in res:
            res[cond]["dmean"] = res[cond]["mean"] - res["orig"]["mean"]
            res[cond]["dsens"] = res[cond]["sens"] - res["orig"]["sens"]
        hits = [c for c in res if c != "orig" and res[c]["dmean"] <= -0.10 and res[c]["dsens"] <= -0.10]
        best = min((c for c in res if c != "orig"), key=lambda c: res[c]["dmean"])
        res["_A2_holds"] = bool(hits); res["_hits"] = hits; res["_most_negative"] = best
        verdicts.append(bool(hits))
        out["models"][m] = res
        print(m, {c: (round(res[c]["dmean"], 3), round(100 * res[c]["dsens"], 1)) for c in CONDS if c in res},
              "A2", bool(hits), hits)
    out["A2_holds"] = bool(verdicts) and all(verdicts)
    json.dump(out, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
