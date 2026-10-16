"""
splits.py — build the frozen split assignment, once, and write it to a file.

Every model in this project loads its folds from that file. Nothing generates a split at
training time. This is the rule that makes two runs comparable at all (PROTOCOL.md §1).

Design
  * 5-fold, grouped and stratified. The group is the unit of independence; the stratum is
    (corpus, DR grade) so that no fold ends up with a different corpus mix or a different
    class balance from the others. Corpus is in the stratum deliberately: Messidor-2, IDRiD
    and APTOS differ in camera, field of view and class distribution, so "which dataset"
    is a shortcut feature, and letting it vary across folds would confound every comparison.
  * Every image is predicted exactly once as out-of-fold, so the reported metric is over
    all 2 260 development images (+/- 1.3 pts) rather than over one 452-image slice
    (+/- 3.0 pts). PROTOCOL.md §7.
  * The external test sets (APTOS for DR, Messidor-1 for DME) never appear here.

Usage:
    python src/splits.py --datasets <root> [--folds 5] [--seed 0] --out data/splits/dev_v1.json
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpora


def grouped_stratified_folds(rows, n_folds=5, seed=0):
    """Assign each GROUP to a fold, balancing strata across folds.

    Greedy: walk the strata, and inside each stratum hand groups to whichever fold is
    currently smallest for that stratum. Deterministic given the seed.
    """
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)

    # a group's stratum is its most common (corpus, dr); groups are almost always size 1
    strata = collections.defaultdict(list)
    for g, items in by_group.items():
        key = collections.Counter((i["corpus"], i["dr"]) for i in items).most_common(1)[0][0]
        strata[key].append(g)

    rng = random.Random(seed)
    counts = collections.defaultdict(lambda: [0] * n_folds)
    assignment = {}
    for key in sorted(strata, key=lambda k: (str(k[0]), k[1])):
        groups = sorted(strata[key])
        rng.shuffle(groups)
        for g in groups:
            f = min(range(n_folds), key=lambda i: (counts[key][i], i))
            assignment[g] = f
            counts[key][f] += len(by_group[g])
    return assignment


def dedup_group(rows, thresh=16):
    """Merge visually near-identical images into one group (256-bit dHash, union-find).

    The calibration of the hash and of the 16-bit threshold is in src/dedup_groups.py
    (ISSUES.md §7): distinct fundus pairs sit at >= 28 bits, re-encodings of one image at
    0-8. APTOS is known to carry duplicates, so a split over it that ignored them would put
    copies of one photograph on both sides of a fold boundary. Returns the number of images
    that joined a multi-image group.
    """
    import cv2, numpy as np
    from dedup_groups import dhash, pairwise_within, DSU
    from preprocess import crop_retina
    H = []
    for i, r in enumerate(rows):
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"unreadable image while hashing: {r['path']}")
        grey = cv2.cvtColor(crop_retina(img), cv2.COLOR_BGR2GRAY)
        H.append(dhash(grey))
        if (i + 1) % 500 == 0:
            print(f"[dedup] hashed {i + 1}/{len(rows)}", flush=True)
    H = np.stack(H)
    dsu = DSU(len(rows))
    n_pairs = 0
    for i, j, _ in pairwise_within(H, thresh):
        dsu.union(i, j); n_pairs += 1
    for i, r in enumerate(rows):
        r["group"] = rows[dsu.find(i)]["uid"]
    sizes = collections.Counter(r["group"] for r in rows)
    joined = sum(v for v in sizes.values() if v > 1)
    print(f"[dedup] {n_pairs} near-duplicate pairs at <= {thresh} bits; {joined} images in "
          f"{sum(1 for v in sizes.values() if v > 1)} multi-image groups", flush=True)
    return joined


def build(datasets_roots, n_folds=5, seed=0, corpora_names=("IDRiD", "Messidor-2"),
          dedup_hamming=0):
    rows = corpora.build(datasets_roots, corpora_names)
    if dedup_hamming:
        dedup_group(rows, dedup_hamming)
    assignment = grouped_stratified_folds(rows, n_folds, seed)

    for r in rows:
        r["fold"] = assignment[r["group"]]

    # a fingerprint of the membership, so a later run can prove it used the same split
    digest = hashlib.sha256(
        "|".join(f"{r['uid']}:{r['fold']}" for r in sorted(rows, key=lambda x: x["uid"]))
        .encode()).hexdigest()[:16]

    return rows, {
        "n_folds": n_folds, "seed": seed, "corpora": list(corpora_names),
        "n_images": len(rows), "n_groups": len(set(r["group"] for r in rows)),
        "fingerprint": digest,
        "folds": {r["uid"]: r["fold"] for r in rows},
        "groups": {r["uid"]: r["group"] for r in rows},
    }


def report(rows, n_folds):
    out = []
    out.append(f"{'fold':>5s} {'n':>6s}  " +
               "  ".join(f"DR{i}" for i in range(5)) + "   " +
               "  ".join(f"DME{i}" for i in range(3)) + "   corpora")
    for f in range(n_folds):
        sub = [r for r in rows if r["fold"] == f]
        dr = collections.Counter(r["dr"] for r in sub)
        dme = collections.Counter(r["dme"] for r in sub if r["dme"] is not None)
        cor = collections.Counter(r["corpus"] for r in sub)
        out.append(f"{f:5d} {len(sub):6d}  " +
                   "  ".join(f"{dr.get(i,0):3d}" for i in range(5)) + "   " +
                   "  ".join(f"{dme.get(i,0):4d}" for i in range(3)) + "   " +
                   " ".join(f"{k}={v}" for k, v in sorted(cor.items())))
    return "\n".join(out)


def verify(meta, rows):
    """Assertions that must hold before this file is committed."""
    problems = []
    g2f = collections.defaultdict(set)
    for r in rows:
        g2f[r["group"]].add(r["fold"])
    straddling = [g for g, fs in g2f.items() if len(fs) > 1]
    if straddling:
        problems.append(f"{len(straddling)} groups appear in more than one fold")
    if len(meta["folds"]) != meta["n_images"]:
        problems.append("fold map size != image count")
    per_fold = collections.Counter(r["fold"] for r in rows)
    if max(per_fold.values()) > 1.35 * min(per_fold.values()):
        problems.append(f"folds badly unbalanced: {dict(sorted(per_fold.items()))}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/splits/dev_v1.json")
    ap.add_argument("--corpora", default="IDRiD,Messidor-2",
                    help="which corpora the split covers. The frozen development split is "
                         "IDRiD,Messidor-2; APTOS is split only for the reverse-direction "
                         "experiment (docs/PAPER_EXPERIMENTS_preregistration.md, G2)")
    ap.add_argument("--dedup-hamming", type=int, default=0,
                    help="merge near-duplicates into one group at this 256-bit dHash "
                         "distance before splitting (16 is the calibrated value; 0 = off)")
    a = ap.parse_args()

    rows, meta = build(a.datasets, a.folds, a.seed, tuple(a.corpora.split(",")),
                       a.dedup_hamming)
    print(corpora.summarise(rows))
    print()
    print(report(rows, a.folds))

    problems = verify(meta, rows)
    print()
    if problems:
        print("REFUSING TO WRITE — invariants violated:")
        for p in problems:
            print("   ", p)
        sys.exit(1)
    print("invariants: no group straddles a fold; folds balanced")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\nwrote {a.out}  fingerprint={meta['fingerprint']}  "
          f"({meta['n_images']} images, {meta['n_groups']} groups)")
    print("This file is now FROZEN. Every run loads its folds from here.")


if __name__ == "__main__":
    main()
