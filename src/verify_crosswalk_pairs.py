"""
verify_crosswalk_pairs.py — is a matched (segmentation, grading) pair the same eye?

Part A's verdict rests on two contradicting images, and the first competing explanation is
that the crosswalk put the wrong grade next to the mask. That was checked once by overlaying
the pair and reporting a mean absolute difference of 0.78--0.96 of 255 -- JPEG re-encoding
noise, so the same photograph. The check was real; the session that ran it is gone and no
artefact survived it, so the number could not be regenerated and the write-up was quoting a
figure nothing on disk supported.

This reproduces it for every informative image and archives the result. The comparison is
deliberately crude -- greyscale, resized to a common frame, mean |difference| -- because the
question is not "how similar" but "is this the same photograph re-encoded".

Usage:
    python3 src/verify_crosswalk_pairs.py --idrid "<IDRiD root>"
"""
from __future__ import annotations
import argparse, glob, json, os
import cv2
import numpy as np

OUT = "docs/generated/crosswalk_pairs.json"
GATE = "docs/generated/idrid_derivation_gate.json"


def find(root, pattern, key):
    return {key(os.path.basename(p)): p
            for p in glob.glob(os.path.join(root, "**", pattern), recursive=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idrid", required=True)
    a = ap.parse_args()

    gate = json.load(open(GATE))
    majority = max(range(3), key=lambda g: sum(x["true"] == g for x in gate["per_image"]))
    # every image whose expert grade is not the majority grade: the only ones on which the
    # derivation can be distinguished from a constant predictor, and so the only ones whose
    # crosswalk the verdict actually depends on
    informative = [x for x in gate["per_image"] if x["true"] != majority]

    seg_img = {}
    for p in glob.glob(os.path.join(a.idrid, "**", "*.jpg"), recursive=True):
        if "Segmentation" in p and "Original Images" in p:
            seg_img[os.path.basename(p)[:-4]] = p
    grade_img = {}
    for p in glob.glob(os.path.join(a.idrid, "**", "*.jpg"), recursive=True):
        if "Disease Grading" in p and "Original Images" in p:
            grade_img[os.path.basename(p)[:-4]] = p
    if not seg_img or not grade_img:
        raise SystemExit(f"found {len(seg_img)} segmentation and {len(grade_img)} grading "
                         f"originals under {a.idrid} -- check the path")

    rows = []
    for x in informative:
        s, g = x["seg"], x["grading"]
        A = cv2.imread(seg_img[s], cv2.IMREAD_GRAYSCALE)
        B = cv2.imread(grade_img[g], cv2.IMREAD_GRAYSCALE)
        if A is None or B is None:
            raise SystemExit(f"could not read the pair {s} / {g}")
        if A.shape != B.shape:
            B = cv2.resize(B, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_AREA)
        d = float(np.abs(A.astype(np.float32) - B.astype(np.float32)).mean())
        rows.append({"seg": s, "grading": g, "expert": x["true"],
                     "derived": x["equivalent_area"], "mean_abs_diff": d,
                     "agrees": x["true"] == x["equivalent_area"]})
        print(f"  {s:10s} <-> {g:12s} expert {x['true']} derived {x['equivalent_area']}  "
              f"mean |diff| {d:6.2f} / 255")

    out = {"note": "mean absolute greyscale difference between the segmentation original and "
                   "the grading original the crosswalk matched it to. A value of a few units "
                   "out of 255 is JPEG re-encoding noise, i.e. the same photograph.",
           "n_informative": len(rows),
           "max_mean_abs_diff": max(r["mean_abs_diff"] for r in rows),
           "min_mean_abs_diff": min(r["mean_abs_diff"] for r in rows),
           "pairs": rows}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}: mean |diff| ranges {out['min_mean_abs_diff']:.2f}"
          f"--{out['max_mean_abs_diff']:.2f} of 255 across {len(rows)} informative pairs")


if __name__ == "__main__":
    main()
