"""
dedup_groups.py — build the grouping that every split in this project is made on.

Why this exists (PROTOCOL.md §1): none of our three corpora publishes patient IDs, and two
of them are known to contain near-duplicates —

  * Messidor-2 pairs two images per examination (one per eye) upstream, but our mirror's
    CSV carries no examination column, so the pairing has to be reconstructed.
  * 687 of the 1 744 files in that same mirror use Messidor-1 filenames, so cross-corpus
    overlap with any future Messidor-1 acquisition is likely (ISSUES.md §3).
  * APTOS is known to contain left/right pairs of the same patient.

Split an image pair like that across the train/test boundary and the model recognises the
individual rather than the disease. This pass finds them.

Method — and why it is NOT the textbook one
  The standard recipe (64-bit dHash/pHash, Hamming <= 6) DOES NOT WORK on fundus images and
  would have quietly poisoned every split in this project. Every fundus photograph is a
  bright ellipse on a black ground, so at 8x8 they all look alike. Measured on 60 IDRiD
  images, all of them different patients (ISSUES.md §7):

      64-bit dHash, distinct pairs : min distance 2 bits
      64-bit dHash, true duplicate : 0-2 bits  (re-encode / resize of the SAME image)
                                     -> no separation whatsoever; 98 of 1 770 distinct
                                        pairs fell inside the usual threshold of 6

  Raising the resolution fixes it. A 256-bit dHash (16x16 grid) separates cleanly:

      256-bit dHash, distinct pairs: min 28, 1st percentile 36, median 66
      256-bit dHash, true duplicate: 0-8

  So: **256-bit dHash, threshold 16 bits**, which sits in a wide empty gap between the two
  populations. Normalised cross-correlation of a 64x64 contrast-normalised thumbnail is
  recorded alongside as a second opinion, but is NOT used as the criterion — it does not
  separate (distinct pairs reach 0.991, true duplicates start at 0.992).

  Near-duplicates are merged transitively into groups by union-find. A group is the unit
  that splits and bootstraps resample (PROTOCOL.md §1, §4).

IMPORTANT: this finds *visually* near-identical images. Two eyes of one patient are NOT
visually near-identical, so this catches re-encodings, mirrored copies and cross-corpus
overlap — not fellow eyes. Fellow-eye pairing needs the vessel/optic-disc geometry pass
flagged at the bottom of this file, or the upstream Messidor-2 pairing file. Until one of
those exists, any Messidor-2 result carries that caveat explicitly.

Usage:
    python src/dedup_groups.py --datasets <Datasets dir> --out data/groups.json
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import crop_retina


# ── hashes ────────────────────────────────────────────────────────────────────
HASH_GRID = 16          # 16x16 -> 240 bits. See the calibration in the module docstring.
NCC_SIZE = 64

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(grey: np.ndarray, n: int = HASH_GRID) -> np.ndarray:
    """256-bit horizontal-gradient hash, packed into 30 uint8s."""
    r = cv2.resize(grey, (n + 1, n), interpolation=cv2.INTER_AREA)
    return np.packbits((r[:, 1:] > r[:, :-1]).flatten())


def signature(grey: np.ndarray, n: int = NCC_SIZE) -> np.ndarray:
    """Contrast-normalised thumbnail, for the correlation second opinion."""
    r = cv2.resize(grey, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)
    r = cv2.GaussianBlur(r, (0, 0), 1.0)
    r -= r.mean()
    s = r.std()
    return (r / (s if s > 1e-6 else 1.0)).astype(np.float32)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(_POPCOUNT[np.bitwise_xor(a, b)].sum())


def pairwise_within(H: np.ndarray, thresh: int, chunk: int = 512):
    """Yield (i, j, distance) for every pair closer than `thresh`.

    Brute force, but vectorised and chunked: prefix bucketing is unsafe at a 16-bit
    threshold, and 6 000 images is only 18 M comparisons.
    """
    n = len(H)
    for s in range(0, n, chunk):
        block = H[s:s + chunk]
        d = _POPCOUNT[np.bitwise_xor(block[:, None, :], H[None, :, :])].sum(axis=2)
        for bi, gi in enumerate(range(s, min(s + chunk, n))):
            hits = np.flatnonzero(d[bi] <= thresh)
            for j in hits:
                if j > gi:
                    yield gi, int(j), int(d[bi, j])


# ── union-find ────────────────────────────────────────────────────────────────
class DSU:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra


# ── corpus enumeration ────────────────────────────────────────────────────────
def enumerate_images(datasets_root: str) -> list[dict]:
    """Every image we hold, with its corpus tag. Paths only — no pixels read here."""
    out = []

    idrid = os.path.join(datasets_root, "IDRiD Indian Diabetic Retinopathy Image Dataset")
    for enc in ("B.%20Disease%20Grading", "B. Disease Grading"):
        base = os.path.join(idrid, enc, "B. Disease Grading", "1. Original Images")
        if os.path.isdir(base):
            for sub in ("a. Training Set", "b. Testing Set"):
                d = os.path.join(base, sub)
                for fn in sorted(os.listdir(d)):
                    if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        out.append({"corpus": "IDRiD", "name": os.path.splitext(fn)[0],
                                    "path": os.path.join(d, fn)})
            break

    m2 = os.path.join(datasets_root, "Messidor-2", "messidor-2", "messidor-2", "preprocess")
    if os.path.isdir(m2):
        for fn in sorted(os.listdir(m2)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                out.append({"corpus": "Messidor-2", "name": fn,
                            "path": os.path.join(m2, fn),
                            # flags the Messidor-1-style filenames from ISSUES.md §3
                            "filename_style": "messidor1" if fn.upper().startswith("IM")
                                              else "messidor2"})

    ap = os.path.join(datasets_root, "APTOS 2019 dataset")
    for sub in ("train_images", "val_images", "test_images"):
        for d in (os.path.join(ap, sub, sub), os.path.join(ap, sub)):
            if os.path.isdir(d):
                for fn in sorted(os.listdir(d)):
                    if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                        out.append({"corpus": "APTOS", "name": os.path.splitext(fn)[0],
                                    "path": os.path.join(d, fn), "aptos_split": sub})
                break
    return out


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--out", default="data/groups.json")
    ap.add_argument("--hamming", type=int, default=16,
                    help="bits of tolerance out of 256; 16 sits in the empty gap between "
                         "true duplicates (<=8) and distinct fundus images (>=28)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only hash the first N")
    args = ap.parse_args()

    imgs = enumerate_images(args.datasets)
    if args.limit:
        imgs = imgs[:args.limit]
    print(f"enumerated {len(imgs)} images")
    by_corpus = {}
    for r in imgs:
        by_corpus[r["corpus"]] = by_corpus.get(r["corpus"], 0) + 1
    for k, v in sorted(by_corpus.items()):
        print(f"  {k:12s} {v:5d}")

    t0 = time.time()
    H, S, keep = [], [], []
    for i, r in enumerate(imgs, 1):
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            r["unreadable"] = True
            continue
        grey = cv2.cvtColor(crop_retina(img), cv2.COLOR_BGR2GRAY)
        H.append(dhash(grey))
        S.append(signature(grey))
        keep.append(i - 1)
        if i % 250 == 0 or i == len(imgs):
            el = time.time() - t0
            print(f"  hashed {i}/{len(imgs)}  {el:.0f}s, ~{el/i*(len(imgs)-i):.0f}s left",
                  flush=True)
    H = np.stack(H) if H else np.zeros((0, 30), np.uint8)
    print(f"hashed {len(keep)}/{len(imgs)} ({time.time()-t0:.0f}s)")
    unreadable = [r for r in imgs if r.get("unreadable")]
    if unreadable:
        print(f"  WARNING: {len(unreadable)} unreadable files, excluded from grouping")

    dsu = DSU(len(imgs))
    pairs = []
    for a, b, d in pairwise_within(H, args.hamming):
        i, j = keep[a], keep[b]
        dsu.union(i, j)
        pairs.append({"a": f"{imgs[i]['corpus']}/{imgs[i]['name']}",
                      "b": f"{imgs[j]['corpus']}/{imgs[j]['name']}",
                      "dhash256_distance": d,
                      "ncc": round(float((S[a] * S[b]).mean()), 4)})

    groups = {}
    for i, r in enumerate(imgs):
        groups.setdefault(dsu.find(i), []).append(i)

    multi = {k: v for k, v in groups.items() if len(v) > 1}
    cross = [p for p in pairs if p["a"].split("/")[0] != p["b"].split("/")[0]]

    print(f"\ngroups: {len(groups)} for {len(imgs)} images")
    print(f"  groups with >1 image : {len(multi)}")
    print(f"  duplicate pairs found: {len(pairs)}  (cross-corpus: {len(cross)})")
    for p in cross[:20]:
        print(f"    CROSS-CORPUS  {p['a']}  <->  {p['b']}  "
              f"(d={p['dhash256_distance']}, ncc={p['ncc']})")
    if len(cross) > 20:
        print(f"    ... and {len(cross)-20} more")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "method": f"{HASH_GRID}x{HASH_GRID} dHash (256-bit), Hamming <= threshold; "
                      f"NCC on {NCC_SIZE}x{NCC_SIZE} contrast-normalised thumbnails "
                      f"recorded as a second opinion only",
            "hamming_threshold": args.hamming,
            "n_images": len(imgs),
            "n_groups": len(groups),
            "duplicate_pairs": pairs,
            "cross_corpus_pairs": cross,
            "assignment": {f"{r['corpus']}/{r['name']}": int(dsu.find(i))
                           for i, r in enumerate(imgs)},
        }, f, indent=1)
    print(f"\nwrote {args.out}")
    print("\nNOTE: this finds visually near-identical images. Fellow-eye pairs are NOT "
          "visually near-identical and are NOT caught here — that still needs the upstream "
          "Messidor-2 pairing file or an optic-disc-laterality pass. Until then, any "
          "Messidor-2 result carries that caveat (PROTOCOL.md §1).")


if __name__ == "__main__":
    main()
