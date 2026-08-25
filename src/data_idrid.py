"""
data_idrid.py — the IDRiD manifest, built once and cached.

One row per image: uid, name, DR grade, DME grade, official split, fovea centre,
optic-disc centre, and the absolute path. Nothing here loads pixels; that is the caller's
job.

`uid` exists because **IDRiD image names are not unique**. The training set and the testing
set each number their images from IDRiD_001, so 103 names appear twice with different
patients, different images and different labels (ISSUES.md §8). Key on `uid`, never on
`name`.

Label semantics (ISSUES.md §2 — the old code got these wrong):
  DR  0..4  ICDR: none / mild / moderate / severe / proliferative
  DME 0..2  "risk of macular edema", by hard-exudate distance to the macula centre:
            0 = no visible hard exudates          -> No_DME
            1 = exudates present, > 1 disc diam.  -> Non_referable_DME
            2 = exudates within 1 disc diameter   -> Referable_DME
"""
from __future__ import annotations
import csv, json, os

DR_CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]
DME_CLASSES = ["No_DME", "Non_referable_DME", "Referable_DME"]

# The official IDRiD release nests content twice, and the macOS Google Drive mount
# URL-encodes the first level. Try every known spelling.
_GRADING_VARIANTS = [
    ("B.%20Disease%20Grading", "B. Disease Grading"),
    ("B. Disease Grading", "B. Disease Grading"),
    ("B. Disease Grading",),
]
_LOC_VARIANTS = [
    ("C.%20Localization", "C. Localization"),
    ("C. Localization", "C. Localization"),
    ("C. Localization",),
]


def _first_dir(root, variants):
    for parts in variants:
        p = os.path.join(root, *parts)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"none of {variants} found under {root}")


def _read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f)]


def _coords(path):
    """Read a localization markup CSV -> {image name: (x, y)}."""
    out = {}
    for r in _read_csv(path):
        name = (r.get("Image No") or "").strip()
        if not name:
            continue
        xs = [v for k, v in r.items() if k and "X-" in k]
        ys = [v for k, v in r.items() if k and "Y " in k or (k and "Y-" in k)]
        try:
            out[name] = (int(float(xs[0])), int(float(ys[0])))
        except (ValueError, IndexError):
            continue
    return out


def build_manifest(idrid_root: str) -> list[dict]:
    grading = _first_dir(idrid_root, _GRADING_VARIANTS)
    gt = os.path.join(grading, "2. Groundtruths")
    imgs = os.path.join(grading, "1. Original Images")

    try:
        loc = _first_dir(idrid_root, _LOC_VARIANTS)
        fov = {}
        od = {}
        for sub, sink in [("2. Fovea Center Location", fov),
                          ("1. Optic Disc Center Location", od)]:
            d = os.path.join(loc, "2. Groundtruths", sub)
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith(".csv"):
                    sink.update(_coords(os.path.join(d, fn)))
    except FileNotFoundError:
        fov, od = {}, {}

    rows = []
    for split, csv_name, img_dir in [
        ("train", "a. IDRiD_Disease Grading_Training Labels.csv", "a. Training Set"),
        ("test",  "b. IDRiD_Disease Grading_Testing Labels.csv",  "b. Testing Set"),
    ]:
        for r in _read_csv(os.path.join(gt, csv_name)):
            name = (r.get("Image name") or "").strip()
            if not name:
                continue
            dme_key = [c for c in r if c and "macular" in c.lower()][0]
            path = os.path.join(imgs, img_dir, name + ".jpg")
            if not os.path.exists(path):
                alt = os.path.join(imgs, img_dir, name + ".JPG")
                path = alt if os.path.exists(alt) else path
            rows.append({
                # unique across the whole corpus; `name` alone is NOT (ISSUES.md §8)
                "uid": f"IDRiD_{split}_{name}",
                "name": name,
                "dr": int(r["Retinopathy grade"].strip()),
                "dme": int(r[dme_key].strip()),
                "official_split": split,
                "fovea": list(fov.get(name, ())) or None,
                "optic_disc": list(od.get(name, ())) or None,
                "path": path,
                # IDRiD publishes no patient IDs and one image per eye, so each image is
                # its own group until the near-duplicate pass says otherwise.
                # PROTOCOL.md §1. Grouping on `name` would merge the train and test images
                # that share a number into one group -- different patients, different
                # labels (ISSUES.md §8).
                "group": f"IDRiD_{split}_{name}",
                "source": "IDRiD",
            })
    return rows


def load_manifest(idrid_root: str, cache: str | None = None) -> list[dict]:
    if cache and os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    rows = build_manifest(idrid_root)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            json.dump(rows, f, indent=1)
    return rows


if __name__ == "__main__":
    import sys, collections
    rows = build_manifest(sys.argv[1])
    print(f"{len(rows)} images")
    for s in ("train", "test"):
        sub = [r for r in rows if r["official_split"] == s]
        print(f"  {s:5s} n={len(sub):4d} "
              f"DR={dict(sorted(collections.Counter(r['dr'] for r in sub).items()))} "
              f"DME={dict(sorted(collections.Counter(r['dme'] for r in sub).items()))}")
    print(f"  with fovea coords     : {sum(1 for r in rows if r['fovea'])}")
    print(f"  with optic-disc coords: {sum(1 for r in rows if r['optic_disc'])}")
    print(f"  missing image files   : {sum(1 for r in rows if not os.path.exists(r['path']))}")
    print(f"  unique uids           : {len(set(r['uid'] for r in rows))}  "
          f"(unique bare names: {len(set(r['name'] for r in rows))} -- "
          f"why uid exists, ISSUES.md §8)")
