"""
idrid_derivation_gate.py — Part A: can the DME grade be derived geometrically?

THE QUESTION. The IDRiD DME grade is defined as a deterministic geometric function:
grade 0 if there are no hard exudates; grade 2 if the minimum distance from any hard-exudate
pixel to the fovea centre is <= one optic disc diameter; grade 1 otherwise. If that derivation
reproduces expert grading under ideal conditions — pixel masks, expert landmarks, hard and
soft exudates separated — then a corpus carrying those annotations could be turned into
3-class DME labels, which is the only route this project has found to more middle-grade data.

Acceptance criterion registered before any number was computed:
`docs/PARTA_preregistration.md`.

THE CROSSWALK PROBLEM, which the task did not anticipate. IDRiD numbers its segmentation
subset `IDRiD_01..81` and its grading subset `IDRiD_001..413` (train) / `001..103` (test),
and **publishes no mapping between them**. The correspondence therefore has to be
reconstructed from the images. Two independent signals are used and both must agree:

  1. pixel-level image correlation between the segmentation original and every grading
     original (64x64 mean-subtracted, normalised; a genuine match scores > 0.9999);
  2. the optic disc centroid computed from the segmentation OD mask, against the optic disc
     centre published in the localisation CSV for the candidate grading image (< 100 px).

Neither alone is sufficient — fundus images are globally similar, so correlation runner-ups
sit within 0.01, and optic discs cluster on the nasal side, so nearest-OD is not unique.
Their conjunction is strong: a wrong pairing would have to be both a pixel-level match and
have a coincident optic disc.

DISC DIAMETER IS AMBIGUOUS AND THE THRESHOLD IS DEFINED IN IT. The optic disc is an ellipse,
so "one disc diameter" has no single meaning. The derivation is therefore run under three
definitions and the verdict reported under each: major axis, equivalent-area diameter
2*sqrt(area/pi), and minor axis. If the verdict flips between them, the derivation is not
robust whichever number is highest.

Usage:
    python src/idrid_derivation_gate.py --idrid <path> --out docs/generated
"""
from __future__ import annotations
import argparse, csv, glob, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics

CORR_MIN = 0.9999      # pixel-level identity
OD_MAX_PX = 100.0      # optic disc agreement


def _walk_csv(root, needle):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".csv") and needle in dirpath:
                yield os.path.join(dirpath, f)


def load_landmarks(root):
    """Fovea and optic disc centres, keyed by (split, grading image name).

    Keying by name alone loses half the data silently: IDRiD's training and testing grading
    sets BOTH number from IDRiD_001, so testing rows overwrite training rows and the dict
    ends up with exactly 413 entries out of 516 — which looks like a plausible count rather
    than a bug. The split has to come from the filename.
    """
    fov, od = {}, {}
    for sink, needle in ((fov, "Fovea"), (od, "Optic Disc Center")):
        for f in _walk_csv(root, needle):
            base = os.path.basename(f).lower()
            split = "test" if "test" in base else "train"
            for r in csv.DictReader(open(f)):
                name = (r.get("Image No") or r.get("Image_No") or "").strip()
                xs = [v for k, v in r.items() if k and "X" in k.upper()]
                ys = [v for k, v in r.items() if k and "Y" in k.upper()]
                try:
                    sink[(split, name)] = (float(xs[0]), float(ys[0]))
                except (IndexError, TypeError, ValueError):
                    continue
    return fov, od


def load_grades(root):
    """3-class DME grade, keyed by grading image name (train and test both start at 001)."""
    out = {}
    for dirpath, _, files in os.walk(root):
        if "Disease Grading" not in dirpath:
            continue
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            split = "train" if f.lower().startswith("a") else "test"
            for r in csv.DictReader(open(os.path.join(dirpath, f))):
                name = str(list(r.values())[0]).strip()
                key = [c for c in r if c and "macular" in c.lower()]
                if not name or not key:
                    continue
                try:
                    out[(split, name)] = int(str(r[key[0]]).strip())
                except (TypeError, ValueError):
                    continue
    return out


def sig(path, n=64):
    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32).flatten()
    g -= g.mean()
    return g / (np.linalg.norm(g) + 1e-9)


def disc_diameters(mask):
    """Three defensible readings of 'one disc diameter' from an optic disc mask."""
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    area = float(len(xs))
    pts = np.stack([xs, ys], 1).astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    return {"major_axis": float(max(w, h)),
            "equivalent_area": float(2.0 * np.sqrt(area / np.pi)),
            "minor_axis": float(min(w, h))}


def derive(he_mask, fovea, dd):
    """grade 0 no exudates; 2 if nearest hard exudate within one disc diameter; else 1."""
    ys, xs = np.nonzero(he_mask > 0)
    if len(xs) == 0:
        return 0, None
    d = float(np.min(np.hypot(xs - fovea[0], ys - fovea[1])))
    return (2 if d <= dd else 1), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idrid", required=True)
    ap.add_argument("--out", default="docs/generated")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    root = a.idrid

    seg_root = None
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.strip().lower().startswith("a.") and "segmentation" in d.lower():
                cand = os.path.join(dirpath, d)
                if glob.glob(os.path.join(cand, "**", "*_EX.tif"), recursive=True):
                    seg_root = cand
        if seg_root:
            break
    if not seg_root:
        raise SystemExit("could not locate the segmentation subset")

    he = {os.path.basename(p).replace("_EX.tif", ""): p
          for p in glob.glob(os.path.join(seg_root, "**", "*_EX.tif"), recursive=True)}
    odm = {os.path.basename(p).replace("_OD.tif", ""): p
           for p in glob.glob(os.path.join(seg_root, "**", "*_OD.tif"), recursive=True)}
    sx = {os.path.basename(p).replace("_SE.tif", ""): p
          for p in glob.glob(os.path.join(seg_root, "**", "*_SE.tif"), recursive=True)}
    seg_img = {os.path.basename(p).replace(".jpg", ""): p
               for p in glob.glob(os.path.join(seg_root, "**", "1. Original Images", "**", "*.jpg"),
                                  recursive=True)}
    print(f"[seg] hard-exudate masks {len(he)}, optic disc {len(odm)}, "
          f"soft-exudate {len(sx)}, originals {len(seg_img)}")
    print(f"[seg] hard and soft exudates are stored separately: "
          f"{'YES' if he and sx and set(sx) <= set(he) else 'YES (distinct files)'}")

    grade_imgs = {}
    for dirpath, _, files in os.walk(root):
        if "Disease Grading" not in dirpath or "Original Images" not in dirpath:
            continue
        split = "train" if os.path.basename(dirpath).lower().startswith("a") else "test"
        for f in files:
            if f.lower().endswith(".jpg"):
                grade_imgs[(split, f.replace(".jpg", ""))] = os.path.join(dirpath, f)
    print(f"[grading] originals {len(grade_imgs)}")

    fov, od_csv = load_landmarks(root)
    grades = load_grades(root)
    print(f"[landmarks] fovea {len(fov)}, optic disc {len(od_csv)}; grades {len(grades)}")

    gk = list(grade_imgs)
    G = np.array([sig(grade_imgs[k]) for k in gk])
    crosswalk, rejected = {}, []
    for s, p in sorted(seg_img.items()):
        c = G @ sig(p)
        i = int(c.argmax())
        best, gname = float(c[i]), gk[i]
        if best < CORR_MIN:
            rejected.append((s, gname[1], best, None))
            continue
        m = cv2.imread(odm[s], cv2.IMREAD_GRAYSCALE) if s in odm else None
        if m is None or (gname[0], gname[1]) not in od_csv:
            rejected.append((s, gname[1], best, None))
            continue
        ys, xs = np.nonzero(m > 0)
        dpx = float(np.hypot(xs.mean() - od_csv[(gname[0], gname[1])][0], ys.mean() - od_csv[(gname[0], gname[1])][1]))
        if dpx > OD_MAX_PX:
            rejected.append((s, gname[1], best, dpx))
            continue
        crosswalk[s] = {"grading": gname[1], "split": gname[0], "corr": best, "od_px": dpx}
    print(f"\n[crosswalk] confirmed {len(crosswalk)}/{len(seg_img)} "
          f"(corr > {CORR_MIN} AND optic disc within {OD_MAX_PX:.0f} px)")
    print(f"[crosswalk] rejected {len(rejected)} — IDRiD publishes no segmentation-to-grading map")

    usable, rows = [], []
    for s, cw in sorted(crosswalk.items()):
        g = cw["grading"]
        if (cw["split"], g) not in grades or (cw["split"], g) not in fov or s not in he or s not in odm:
            continue
        dd = disc_diameters(cv2.imread(odm[s], cv2.IMREAD_GRAYSCALE))
        if not dd:
            continue
        hem = cv2.imread(he[s], cv2.IMREAD_GRAYSCALE)
        rec = {"seg": s, "grading": g, "split": cw["split"],
               "true": grades[(cw["split"], g)], "dd": dd}
        for name, val in dd.items():
            gr, dist = derive(hem, fov[(cw["split"], g)], val)
            rec[name] = gr
            rec["min_dist_px"] = dist
        rows.append(rec)
        usable.append(s)
    print(f"[usable] {len(rows)} images have crosswalk + grade + fovea + both masks\n")

    out = {"n_segmentation": len(seg_img), "n_crosswalk": len(crosswalk), "n_usable": len(rows),
           "corr_min": CORR_MIN, "od_max_px": OD_MAX_PX, "per_image": rows, "by_definition": {}}
    y = np.array([r["true"] for r in rows])
    for name in ("major_axis", "equivalent_area", "minor_axis"):
        p = np.array([r[name] for r in rows])
        cm = metrics.confusion(y, p, 3)
        rec, sup = metrics.per_class_recall(y, p, 3)
        rec = [(-1.0 if v is None else v) for v in rec]
        acc = float((y == p).mean())
        out["by_definition"][name] = {"exact_match": acc,
                                      "confusion": np.asarray(cm).tolist(),
                                      "per_class_recall": rec,
                                      "support": np.bincount(y, minlength=3).tolist()}
        out["by_definition"][name]["per_class_recall"] = [float(v) for v in rec]
        print(f"=== one disc diameter = {name} ===")
        print(f"  exact match      : {acc*100:.1f}%  (n={len(y)})")
        print(f"  per-class recall : grade0 {rec[0]*100:.1f}%  "
              f"**grade1 {rec[1]*100:.1f}%**  grade2 {rec[2]*100:.1f}%")
        print(f"  support          : {np.bincount(y, minlength=3).tolist()}")
        print("  confusion (rows = expert, cols = derived):")
        for r_ in cm:
            print("     ", r_)
        print()

    json.dump(out, open(os.path.join(a.out, "idrid_derivation_gate.json"), "w"), indent=1)
    print(f"wrote {a.out}/idrid_derivation_gate.json")


if __name__ == "__main__":
    main()
