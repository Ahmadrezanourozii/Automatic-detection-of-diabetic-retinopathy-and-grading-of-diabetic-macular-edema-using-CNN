"""
paper_dev_summary.py — development out-of-fold DR QWK (shipped and matched) for every model in
results/predictions/registry.json. Writes docs/generated/paper/dev_summary.json.

Usage: python src/paper_dev_summary.py --labels <dir with IDRiD + Messidor-2 label CSVs>
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_rules as P
from paper_phase0 import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--pred-root", default="results/predictions")
a = ap.parse_args()
L = P.dev_labels(a.labels, "data/splits/dev_v1.json")
reg = json.load(open(os.path.join(a.pred_root, "registry.json")))
out = {}
for m, e in reg.items():
    if not e.get("oof_n") or e.get("reverse"):
        continue
    dv, _ = P.dev_matched(load_model(a.pred_root, m, L), e["head"])
    out[m] = {"n": dv["n"], "folds": dv["folds"], "qwk_shipped": dv["shipped"]["qwk"],
              "qwk_matched": dv["matched"]["qwk"], "accuracy_matched": dv["matched"]["accuracy"]}
    print(f"{m:9s} n={dv['n']:4d} folds={dv['folds']} shipped {dv['shipped']['qwk']:.4f} "
          f"matched {dv['matched']['qwk']:.4f}")
json.dump(out, open("docs/generated/paper/dev_summary.json", "w"), indent=1, default=float)
