"""
paper_crossfold_dupes.py — sensitivity of development estimates to the exact-duplicate pairs
that the overlap audit found split across folds of the frozen development split.

Recomputes each 5-fold model's matched development QWK with both members of every such pair
removed. Writes docs/generated/paper/crossfold_dupes.json.

Usage: python src/paper_crossfold_dupes.py --labels <dir> --audit <overlap_audit.json>
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_rules as P
from paper_phase0 import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--audit", required=True)
ap.add_argument("--pred-root", default="results/predictions")
ap.add_argument("--max-bits", type=int, default=8)
a = ap.parse_args()
L = P.dev_labels(a.labels, "data/splits/dev_v1.json")
audit = json.load(open(a.audit))
pairs = [p for p in audit["dev_within_cross_fold_examples"] if p[2] <= a.max_bits]
drop = {u for p in pairs for u in p[:2]}
reg = json.load(open(os.path.join(a.pred_root, "registry.json")))
out = {"n_pairs": len(pairs), "n_images": len(drop), "pairs": pairs, "models": {}}
for m in ("E08", "E09", "E10", "E11FULL", "E14MAC", "E15LPFT", "E17NAT", "E20CORAL", "E22CORN"):
    d = load_model(a.pred_root, m, L)
    full, _ = P.dev_matched(d, reg[m]["head"])
    keep = np.array([u not in drop for u in d["uids"]])
    sub = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in d.items()}
    red, _ = P.dev_matched(sub, reg[m]["head"])
    out["models"][m] = {"full": full["matched"]["qwk"], "deduplicated": red["matched"]["qwk"],
                        "diff": red["matched"]["qwk"] - full["matched"]["qwk"]}
    print(f"{m:9s} {full['matched']['qwk']:.4f} -> {red['matched']['qwk']:.4f}")
diffs = [v["diff"] for v in out["models"].values()]
out["max_abs_diff"] = float(max(abs(x) for x in diffs))
json.dump(out, open("docs/generated/paper/crossfold_dupes.json", "w"), indent=1, default=float)
print(len(pairs), "pairs,", len(drop), "images; max |diff|", out["max_abs_diff"])
