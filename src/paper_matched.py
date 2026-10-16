"""
paper_matched.py — recompute a matched-calibration comparison from results/predictions/ in the
archived file format of compare_matched.py, head-aware (ISSUES.md §30). Output goes to
docs/generated/paper/<name>.json and supersedes the archived docs/generated/<name>.json in
src/paper_tables.py.

Usage: python src/paper_matched.py --labels <dir> --a E22CORN --b E08 --name matched_corn_e08
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_rules as P
from paper_phase0 import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--pred-root", default="results/predictions")
a = ap.parse_args()
L = P.dev_labels(a.labels, "data/splits/dev_v1.json")
reg = json.load(open(os.path.join(a.pred_root, "registry.json")))
A, B = load_model(a.pred_root, a.a, L), load_model(a.pred_root, a.b, L)
out = []
for head, dme in (("dr", False), ("dme_ungated", True)):
    r = P.matched_compare(A, B, reg[a.a]["head"], reg[a.b]["head"], labels=L, dme=dme)
    for mode, key in (("default", "shipped"), ("matched", "matched")):
        v = r[key]
        out.append({"head": head, "mode": mode, "metric": "QWK", "a": v["a"], "b": v["b"],
                    "diff": v["diff"], "lo": v["lo"], "hi": v["hi"],
                    "significant": v["significant"], "n": v["n"],
                    "heads": [reg[a.a]["head"], reg[a.b]["head"]]})
os.makedirs("docs/generated/paper", exist_ok=True)
json.dump(out, open(f"docs/generated/paper/{a.name}.json", "w"), indent=1)
for o in out:
    print(f"{o['head']:12s} {o['mode']:8s} {a.a} {o['a']:.4f}  {a.b} {o['b']:.4f}  "
          f"{o['diff']:+.4f} [{o['lo']:+.4f}, {o['hi']:+.4f}]")
