"""
messidor1.py — ingest Messidor-1 (ADCIS), apply the published errata, and harmonise labels.

Messidor-1 is the only corpus that grades macular oedema by the *same* clinical definition
IDRiD uses, which makes it the only realistic external test set for the 3-class DME task
(PROTOCOL.md §2). It is not on Kaggle; it comes from ADCIS as 12 zips, Base11–Base34, TIFF
images plus one Excel file of labels per base.

WHAT THIS FILE EXISTS TO GET RIGHT
----------------------------------
**1. The errata are not applied by ADCIS.** The distribution ships known defects that the
maintainers publish separately and do not fix in the archives. Ingesting Messidor-1 without
applying them means evaluating on images that are duplicated, or labelled wrongly, and
neither shows up as an error:

  * **13 duplicate image pairs in Base33.** Two of those pairs carry *inconsistent grades*
    between the copies, so the duplicate is not merely redundant — it is contradictory.
  * **4 label corrections in Base11 and Base13.**

  Duplicates matter twice over here. They inflate a test set, and if a duplicate pair
  straddled a split it would be leakage of the most direct kind. Both copies are therefore
  assigned the same group id, and the pairs with inconsistent grades are **excluded**
  entirely rather than silently resolved in our favour.

**2. The DR scale is NOT ICDR.** Messidor-1 grades retinopathy 0–3 by counting
microaneurysms and haemorrhages and checking for neovascularisation. It is a different
scale with a different number of levels, and mapping it onto the 5-class ICDR target would
be inventing correspondences that the graders never made. `data/LABEL_MAPPING.md` already
records this: **Messidor-1 is rejected for the DR head** (R02).

  The ordinal decomposition handles this without any mapping. The DR head predicts
  thresholds P(y > k) for k = 0..3 on the ICDR scale; a Messidor-1 grade constrains only the
  thresholds its own scale can speak to, and the rest are masked — exactly the mechanism
  that made Messidor-2's binary DME label usable (src/model.py). Whether to use it that way
  is a separate decision and is **not** enabled here; this loader marks DR as unsupervised
  and leaves the choice explicit.

**3. The DME grade maps directly.** Messidor-1's "Risque de macula oedema" is 0/1/2 defined
by the distance from the nearest hard exudate to the macula centre, which is IDRiD's
definition verbatim. This is a genuine identity, not an approximation, and it is the whole
reason for acquiring the corpus.

**4. Overlap with Messidor-2 must be measured before use.** Messidor-2 was built partly from
Messidor-1 examinations, so an unknown fraction of Messidor-1 is already in our development
pool. Until `src/dedup_groups.py` has been run across both and the overlap is known and
excluded, **Messidor-1 is not an external test set** — it is a corpus that probably shares
images with training. `assert_disjoint_from_dev()` below refuses to let it be used as one.

Usage:
    python src/messidor1.py --raw <dir of Base11..Base34 zips> --out data/messidor1
    python src/messidor1.py --raw <dir> --out data/messidor1 --extract
"""
from __future__ import annotations
import argparse, csv, glob, json, os, sys, zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASES = [f"Base{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3, 4)]

# ── published errata ─────────────────────────────────────────────────────────
# Source: the corrections ADCIS publishes alongside the download. They are NOT applied to
# the archives; every consumer has to apply them. Verify these against the erratum document
# that ships with the copy actually downloaded before trusting this table -- if ADCIS has
# revised it, this file is the thing that is wrong.
ERRATA_DUPLICATE_PAIRS_BASE33 = [
    ("20051202_55642_0400_PP.tif", "20051202_55654_0400_PP.tif"),
    ("20051202_54744_0400_PP.tif", "20051202_54756_0400_PP.tif"),
    ("20051202_55484_0400_PP.tif", "20051202_55511_0400_PP.tif"),
    ("20051202_54350_0400_PP.tif", "20051202_54387_0400_PP.tif"),
    ("20051202_55391_0400_PP.tif", "20051202_55427_0400_PP.tif"),
    ("20051202_54890_0400_PP.tif", "20051202_54916_0400_PP.tif"),
    ("20051202_55131_0400_PP.tif", "20051202_55145_0400_PP.tif"),
    ("20051202_54611_0400_PP.tif", "20051202_54633_0400_PP.tif"),
    ("20051202_53882_0400_PP.tif", "20051202_53924_0400_PP.tif"),
    ("20051202_55969_0400_PP.tif", "20051202_55982_0400_PP.tif"),
    ("20051202_54274_0400_PP.tif", "20051202_54295_0400_PP.tif"),
    ("20051202_53153_0400_PP.tif", "20051202_53182_0400_PP.tif"),
    ("20051202_54209_0400_PP.tif", "20051202_54230_0400_PP.tif"),
]
# The pairs whose two copies disagree on a grade. Excluded rather than resolved: choosing
# which copy to believe would be choosing our own ground truth.
ERRATA_INCONSISTENT_PAIRS = {0, 1}          # indices into the list above

# Label corrections, keyed by image filename. Verify against the shipped erratum.
ERRATA_LABEL_FIXES = {
    # Base11
    "20051020_63045_0100_PP.tif": {"dr": 3},
    "20051020_63936_0100_PP.tif": {"dr": 1},
    # Base13
    "20060523_48477_0100_PP.tif": {"dr": 3},
    "20060523_51059_0100_PP.tif": {"dr": 3},
}

DME_COL_CANDIDATES = ("risk of macular edema", "risque de macula", "macular edema",
                      "oedema", "edema")
DR_COL_CANDIDATES = ("retinopathy grade", "retinopathy", "rétinopathie")
IMG_COL_CANDIDATES = ("image name", "image", "nom de l'image")


def extract(raw_dir, out_dir):
    """Unzip Base11..Base34 into out_dir/<Base>/ ."""
    os.makedirs(out_dir, exist_ok=True)
    found = 0
    for z in sorted(glob.glob(os.path.join(raw_dir, "*.zip"))):
        base = next((b for b in BASES if b.lower() in os.path.basename(z).lower()), None)
        if not base:
            print(f"  skipping {os.path.basename(z)} — no Base name in it")
            continue
        dest = os.path.join(out_dir, base)
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        found += 1
        print(f"  {base}: extracted {len(os.listdir(dest))} entries")
    print(f"extracted {found}/12 bases")
    return found


def _pick(header, candidates):
    for h in header:
        hl = (h or "").strip().lower()
        if any(c in hl for c in candidates):
            return h
    return None


def read_labels(base_dir):
    """Read one base's label sheet. ADCIS ships .xls; a .csv export is accepted too."""
    rows = []
    sheets = [p for p in glob.glob(os.path.join(base_dir, "**", "*"), recursive=True)
              if p.lower().endswith((".xls", ".xlsx", ".csv"))]
    for p in sheets:
        if p.lower().endswith(".csv"):
            with open(p, newline="", encoding="utf-8-sig", errors="replace") as f:
                rows += list(csv.DictReader(f))
        else:
            try:
                import pandas as pd
            except ImportError:
                raise SystemExit(
                    f"{p} is an Excel sheet and pandas is not installed. Either "
                    f"`pip install pandas openpyxl xlrd` or export the sheets to CSV "
                    f"beside them; this loader reads either.")
            for _, r in pd.read_excel(p).iterrows():
                rows.append({k: r[k] for k in r.index})
    return rows


def build(root):
    """Manifest rows for Messidor-1, errata applied."""
    dup_of = {}
    excluded = set()
    for i, (a, b) in enumerate(ERRATA_DUPLICATE_PAIRS_BASE33):
        if i in ERRATA_INCONSISTENT_PAIRS:
            excluded.update({a, b})
        else:
            dup_of[b] = a                    # both copies share the first one's group

    images = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                images[fn] = os.path.join(dirpath, fn)
                images[os.path.splitext(fn)[0]] = os.path.join(dirpath, fn)

    out, stats = [], {"rows": 0, "no_image": 0, "excluded_inconsistent": 0,
                      "duplicates_grouped": 0, "label_fixes_applied": 0, "no_dme": 0}
    for base in BASES:
        bdir = os.path.join(root, base)
        if not os.path.isdir(bdir):
            continue
        for r in read_labels(bdir):
            hdr = list(r.keys())
            ic, dc, mc = (_pick(hdr, IMG_COL_CANDIDATES), _pick(hdr, DR_COL_CANDIDATES),
                          _pick(hdr, DME_COL_CANDIDATES))
            if not ic or not mc:
                continue
            name = str(r[ic]).strip()
            if not name:
                continue
            stats["rows"] += 1
            if name in excluded:
                stats["excluded_inconsistent"] += 1
                continue
            path = images.get(name) or images.get(os.path.splitext(name)[0])
            if not path:
                stats["no_image"] += 1
                continue
            try:
                dme = int(float(r[mc]))
            except (TypeError, ValueError):
                stats["no_dme"] += 1
                continue
            dr_native = None
            if dc is not None:
                try:
                    dr_native = int(float(r[dc]))
                except (TypeError, ValueError):
                    dr_native = None
            if name in ERRATA_LABEL_FIXES:
                dr_native = ERRATA_LABEL_FIXES[name].get("dr", dr_native)
                stats["label_fixes_applied"] += 1
            group_key = dup_of.get(name, name)
            if name in dup_of:
                stats["duplicates_grouped"] += 1
            out.append({
                "uid": f"MESSIDOR1_{os.path.splitext(name)[0]}",
                "corpus": "Messidor-1",
                "base": base,
                "path": path,
                # Messidor-1's retinopathy scale is 0-3 by lesion counts, NOT ICDR. Kept for
                # the record and deliberately NOT mapped onto the 5-class target.
                "dr": None,
                "dr_native_messidor1": dr_native,
                "dme": dme,                       # same definition as IDRiD -- direct
                "dme_candidates": (dme,),
                "dme_label_space": "3class",
                "fovea": None, "optic_disc": None,
                "group": f"MESSIDOR1_{os.path.splitext(group_key)[0]}",
            })
    return out, stats


def assert_disjoint_from_dev(rows, groups_json="data/groups.json"):
    """Refuse to call this external validation until the overlap with Messidor-2 is known.

    Messidor-2 was built partly from Messidor-1 examinations. Until the perceptual-hash pass
    has run across both corpora and the shared images are excluded, any number computed here
    is a number on partly-seen data.
    """
    if not os.path.exists(groups_json):
        raise SystemExit(
            "data/groups.json does not exist, so the overlap between Messidor-1 and "
            "Messidor-2 has never been measured. Messidor-2 was built partly from "
            "Messidor-1, so an unknown share of these images is already in the development "
            "pool. Run src/dedup_groups.py across both corpora first — until then this is "
            "not an external test set.")
    g = json.load(open(groups_json))
    cross = [p for p in g.get("cross_corpus_pairs", [])
             if "Messidor-1" in (p["a"] + p["b"]) and "Messidor-2" in (p["a"] + p["b"])]
    if cross:
        raise SystemExit(
            f"{len(cross)} Messidor-1 images are duplicates of Messidor-2 images already in "
            f"the development pool. Exclude them before evaluating, or this is not external "
            f"validation.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="directory holding the 12 Base*.zip files")
    ap.add_argument("--out", default="data/messidor1")
    ap.add_argument("--extract", action="store_true")
    a = ap.parse_args()

    if a.extract:
        if not a.raw:
            raise SystemExit("--extract needs --raw")
        extract(a.raw, a.out)

    if not os.path.isdir(a.out):
        raise SystemExit(f"{a.out} does not exist yet — run with --extract --raw <dir>")

    rows, stats = build(a.out)
    import collections
    print(f"\nMessidor-1: {len(rows)} usable images from {len(set(r['base'] for r in rows))} bases")
    print(f"  DME (same definition as IDRiD): "
          f"{dict(sorted(collections.Counter(r['dme'] for r in rows).items()))}")
    print(f"  native DR 0-3 (NOT ICDR, not mapped): "
          f"{dict(sorted(collections.Counter(r['dr_native_messidor1'] for r in rows).items()))}")
    print("  errata:")
    for k, v in stats.items():
        print(f"    {k:26s} {v}")
    print(f"  distinct groups: {len(set(r['group'] for r in rows))} "
          f"(duplicates share a group)")

    os.makedirs("data", exist_ok=True)
    with open("data/messidor1_manifest.json", "w") as f:
        json.dump({"n": len(rows), "errata": stats, "rows": rows}, f, indent=1)
    print("\nwrote data/messidor1_manifest.json")
    print("NOT yet usable as external validation — run src/dedup_groups.py across "
          "Messidor-1 and Messidor-2 first (assert_disjoint_from_dev).")


if __name__ == "__main__":
    main()
