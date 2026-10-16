"""
paper_overlap_audit.py — content-level near-duplicate audit across every corpus the paper uses.

verify_external.py proved that no external image shares a uid with the development pool. A uid
is a filename; it says nothing about whether the same photograph sits in two corpora under two
names. This audit compares image CONTENT, with the 256-bit difference hash calibrated on fundus
images in ISSUES.md §7 (distinct pairs >= 28 bits apart, re-encodings of one image 0-8;
threshold 16), and reports:

  * near-duplicate pairs WITHIN the development pool (the frozen split treats every image as
    its own group, so any pair found here is a leakage risk across folds);
  * pairs BETWEEN the development pool and each external set (APTOS, DDR) -- leakage into the
    external estimate;
  * pairs between the EyePACS pretraining corpus and the external sets -- contamination of the
    external estimate through pretraining;
  * pairs between APTOS and DDR.

Runs on CPU. Writes overlap_audit.json into --out.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpora
from dedup_groups import dhash, _POPCOUNT
from preprocess import crop_retina

THRESH = 16


def hashes(rows):
    H, keep = [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        g = cv2.cvtColor(crop_retina(img), cv2.COLOR_BGR2GRAY)
        H.append(dhash(g)); keep.append(r["uid"])
        if (i + 1) % 2000 == 0:
            print(f"  hashed {i + 1}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    return np.stack(H), keep


def cross_pairs(A, B, same=False, chunk=256):
    """(i, j, d) with Hamming distance <= THRESH between rows of A and rows of B."""
    out = []
    for s in range(0, len(A), chunk):
        blk = A[s:s + chunk]
        d = _POPCOUNT[np.bitwise_xor(blk[:, None, :], B[None, :, :])].sum(axis=2)
        ii, jj = np.nonzero(d <= THRESH)
        for i, j in zip(ii, jj):
            gi = s + int(i)
            if same and j <= gi:
                continue
            out.append((gi, int(j), int(d[i, j])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["/kaggle/input"])
    ap.add_argument("--splits", default="data/splits/dev_v1.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eyepacs-limit", type=int, default=0,
                    help="debug: hash only the first N EyePACS images")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sets = {}
    for name, corp in (("dev", ("IDRiD", "Messidor-2")), ("APTOS", ("APTOS",)),
                       ("DDR", ("DDR",)), ("EyePACS", ("EyePACS",))):
        rows = corpora.build(a.datasets, corp)
        if name == "EyePACS" and a.eyepacs_limit:
            rows = rows[:a.eyepacs_limit]
        print(f"[audit] hashing {name}: {len(rows)} images", flush=True)
        sets[name] = hashes(rows)

    report = {"threshold_bits": THRESH, "n": {k: len(v[1]) for k, v in sets.items()},
              "pairs": {}}

    def record(tag, pairs, ua, ub):
        cap = None if len(pairs) <= 5000 else 50       # small sets are archived in full
        report["pairs"][tag] = {"n_pairs": len(pairs),
                                "examples": [[ua[i], ub[j], d] for i, j, d in pairs[:cap]]}
        print(f"[audit] {tag:22s} {len(pairs):6d} pairs", flush=True)

    H, U = sets["dev"]
    record("dev_within", cross_pairs(H, H, same=True), U, U)
    for ext in ("APTOS", "DDR"):
        HE, UE = sets[ext]
        record(f"dev_vs_{ext}", cross_pairs(H, HE), U, UE)
        record(f"{ext}_within", cross_pairs(HE, HE, same=True), UE, UE)
    HA, UA = sets["APTOS"]; HD, UD = sets["DDR"]
    record("APTOS_vs_DDR", cross_pairs(HA, HD), UA, UD)
    HP, UP = sets["EyePACS"]
    for ext, (HE, UE) in (("APTOS", sets["APTOS"]), ("DDR", sets["DDR"]), ("dev", sets["dev"])):
        record(f"EyePACS_vs_{ext}", cross_pairs(HE, HP), UE, UP)

    # the split treats every development image as its own group; a within-pool pair whose two
    # images sit in different folds is cross-fold leakage
    split = json.load(open(a.splits))
    cross_fold = [p for p in report["pairs"]["dev_within"]["examples"]
                  if split["folds"].get(p[0]) != split["folds"].get(p[1])]
    report["dev_within_cross_fold_examples"] = cross_fold
    json.dump(report, open(os.path.join(a.out, "overlap_audit.json"), "w"), indent=1)
    print(json.dumps({k: v["n_pairs"] for k, v in report["pairs"].items()}, indent=1))


if __name__ == "__main__":
    main()
