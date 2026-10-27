"""
paper_acquisition.py — post-hoc shortcut check (added after an internal editorial assessment).

Can the way an image was acquired or stored predict its grade without a model? For every corpus
with per-image acquisition statistics (src/paper_degrade.py --per-image, run X6ACQ), the AUC with
which each single statistic separates referable (grade >= 2) from non-referable images, and any
DR from no DR, is computed in whichever direction separates better (max(AUC, 1 - AUC)).
Development = IDRiD and Messidor-2 pooled. Writes docs/generated/paper/acquisition.json.

Usage: python src/paper_acquisition.py --dir <folder with per_image_*.csv>
"""
from __future__ import annotations
import argparse, csv, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_rules as P

FEATURES = ["height", "width", "file_bytes", "jpeg_quality", "retina_fraction", "brightness",
            "rms_contrast", "laplacian_var"]
NAMES = {"sustech": "SUSTech", "aptos": "APTOS", "deepdrid": "DeepDRiD", "ddr": "DDR",
         "idrid": "IDRiD", "messidor-2": "Messidor-2"}


def read(path):
    rows = list(csv.DictReader(open(path)))
    y = np.array([int(r["dr"]) for r in rows])
    X = {f: np.array([float(r[f]) if r[f] not in ("", None) else np.nan for r in rows]) for f in FEATURES}
    return y, X


def separation(y, X):
    out = {}
    for f, x in X.items():
        ok = ~np.isnan(x)
        if ok.sum() < 20 or np.nanstd(x) == 0:
            continue
        r = {}
        for name, pos in (("ref", y >= 2), ("any", y >= 1)):
            if pos[ok].all() or not pos[ok].any():
                continue
            a = P.auroc(x[ok], pos[ok])
            r[name] = max(a, 1 - a)
        out[f] = r
    best = lambda k: max(((f, v[k]) for f, v in out.items() if k in v), key=lambda t: t[1], default=(None, None))
    fr, mr = best("ref"); fa, ma = best("any")
    return {"by_feature": out, "max_ref": mr, "max_ref_feature": fr, "max_any": ma, "max_any_feature": fa,
            "n": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="docs/generated/paper/acquisition.json")
    a = ap.parse_args()
    data = {}
    for p in sorted(glob.glob(os.path.join(a.dir, "per_image_*.csv"))):
        key = os.path.basename(p)[len("per_image_"):-4]
        data[NAMES.get(key, key)] = read(p)
    out = {}
    for c, (y, X) in data.items():
        if c in ("IDRiD", "Messidor-2"):
            continue
        out[c] = separation(y, X)
    if "IDRiD" in data and "Messidor-2" in data:
        y = np.concatenate([data["IDRiD"][0], data["Messidor-2"][0]])
        X = {f: np.concatenate([data["IDRiD"][1][f], data["Messidor-2"][1][f]]) for f in FEATURES}
        out["dev"] = separation(y, X)
    for c, v in out.items():
        print(f"{c:10s} n {v['n']:6d}  referable: {v['max_ref_feature']} {v['max_ref']:.3f}   "
              f"any DR: {v['max_any_feature']} {v['max_any']:.3f}")
    json.dump(out, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
