"""
audit_baseline.py — reproduce, from files on disk only, every number this project
has ever claimed. Run this before believing any prior result.

Usage:  python3 src/audit_baseline.py --drive "<path to Google Drive Alireza folder>"

Outputs a plain-text report. Computes nothing that is not derived from the
label CSVs and the archived TensorBoard event files.
"""
import argparse, csv, collections, glob, os, struct, sys

# ── minimal pure-python TFRecord/Event parser (no tensorflow dependency) ──────
def _varint(b, i):
    r = s = 0
    while True:
        x = b[i]; i += 1; r |= (x & 0x7F) << s; s += 7
        if not x & 0x80:
            return r, i

def _fields(b):
    i = 0
    while i < len(b):
        k, i = _varint(b, i); f, w = k >> 3, k & 7
        if w == 0:   v, i = _varint(b, i)
        elif w == 1: v = b[i:i+8]; i += 8
        elif w == 2: n, i = _varint(b, i); v = b[i:i+n]; i += n
        elif w == 5: v = b[i:i+4]; i += 4
        else: return
        yield f, w, v

def _records(path):
    b = open(path, 'rb').read(); i = 0
    while i + 12 <= len(b):
        n = struct.unpack('<Q', b[i:i+8])[0]; i += 12
        yield b[i:i+n]; i += n + 4

def read_events(path):
    """Return {tag: [(step, value), ...]} of scalar summaries."""
    out = collections.defaultdict(list)
    for rec in _records(path):
        step, summ = 0, None
        for f, w, v in _fields(rec):
            if f == 2 and w == 0: step = v
            elif f == 5 and w == 2: summ = v
        if summ is None: continue
        for f, w, v in _fields(summ):
            if f != 1: continue
            tag = val = None
            for f2, w2, v2 in _fields(v):
                # Summary.Value: tag = field 1, simple_value = 2, node_name = 7, tensor = 8
                if f2 == 1 and w2 == 2: tag = v2.decode('utf8', 'replace')
                elif f2 == 2 and w2 == 5: val = struct.unpack('<f', v2)[0]
                elif f2 == 8 and w2 == 2:
                    for f3, w3, v3 in _fields(v2):
                        if f3 == 4 and w3 == 2 and len(v3) >= 4:
                            val = struct.unpack('<f', v3[:4])[0]
                        elif f3 == 5 and w3 == 5:
                            val = struct.unpack('<f', v3)[0]
            if tag and val is not None:
                out[tag].append((step, val))
    return out

# ── label loading ─────────────────────────────────────────────────────────────
def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return [r for r in csv.DictReader(f)]

IDRID_SUB = "B.%20Disease%20Grading/B. Disease Grading/2. Groundtruths"

def load_idrid(ds_root):
    root = os.path.join(ds_root, "IDRiD Indian Diabetic Retinopathy Image Dataset", IDRID_SUB)
    out = {}
    for split, fn in [("train", "a. IDRiD_Disease Grading_Training Labels.csv"),
                      ("test",  "b. IDRiD_Disease Grading_Testing Labels.csv")]:
        rows = [r for r in read_csv(os.path.join(root, fn)) if r.get("Image name")]
        key = [c for c in rows[0] if c and "macular" in c.lower()][0]
        out[split] = [(r["Image name"].strip(),
                       int(r["Retinopathy grade"].strip()),
                       int(r[key].strip())) for r in rows]
    return out

def dist(vals, k):
    c = collections.Counter(vals)
    return {i: c.get(i, 0) for i in range(k)}

def majority_acc(vals):
    c = collections.Counter(vals)
    return max(c.values()) / len(vals)

# ── the two confusion matrices archived in results/ (transcribed from the PNGs,
#    which are the only saved evaluation output in the whole project) ──────────
DR_CM_ARCHIVED = [   # rows = true 0..4, cols = pred 0..4
    [0, 0, 27, 0, 7],
    [0, 0,  4, 0, 1],
    [0, 1, 28, 0, 3],
    [0, 0, 17, 0, 2],
    [0, 0, 13, 0, 0],
]
DME_CM_ARCHIVED = [  # rows = true 0..2, cols = pred 0..2
    [11, 0, 0],
    [10, 0, 0],
    [46, 2, 0],
]

def cm_stats(cm):
    n = sum(sum(r) for r in cm)
    correct = sum(cm[i][i] for i in range(len(cm)))
    support = [sum(r) for r in cm]
    return n, correct, correct / n, support

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True, help="path to the Google Drive 'Alireza' folder")
    a = ap.parse_args()
    ds = os.path.join(a.drive, "Datasets")
    W = 78

    print("=" * W); print("BASELINE AUDIT — every number reproduced from files on disk"); print("=" * W)

    # 1 — IDRiD labels
    idr = load_idrid(ds)
    print("\n[1] IDRiD Disease Grading — official splits")
    for s in ("train", "test"):
        dr = [x[1] for x in idr[s]]; dme = [x[2] for x in idr[s]]
        print(f"    {s:5s} n={len(dr):4d}  DR={dist(dr,5)}  DME={dist(dme,3)}")

    allr = idr["train"] + idr["test"]
    print("\n    joint DR x DME (all 516):")
    j = collections.Counter((r[1], r[2]) for r in allr)
    print("      DR\\DME " + "".join(f"{d:>6}" for d in range(3)))
    for d in range(5):
        print(f"        {d}   " + "".join(f"{j[(d,m)]:6d}" for m in range(3)))
    leak = sum(v for (d, m), v in j.items() if d == 0 and m > 0)
    print(f"    DR=0 images with DME>0 (lost to the DR>=1 gate): {leak}")

    # 2 — majority-class floors
    print("\n[2] Trivial baselines (majority class) — any model must beat these")
    te_dr  = [x[1] for x in idr["test"]]
    te_dme = [x[2] for x in idr["test"] if x[1] >= 1]
    print(f"    DR, IDRiD official test (n={len(te_dr)})           : {majority_acc(te_dr)*100:5.1f}%")
    print(f"    DME, gated DR>=1, IDRiD test (n={len(te_dme)})      : {majority_acc(te_dme)*100:5.1f}%")
    tr_dme = [x[2] for x in allr if x[1] >= 1]
    print(f"    DME, gated DR>=1, all 516 (n={len(tr_dme)})        : {majority_acc(tr_dme)*100:5.1f}%")

    # 3 — the only archived evaluation output
    print("\n[3] Archived confusion matrices (results/*.png) — the ONLY saved evaluation")
    n, c, acc, sup = cm_stats(DR_CM_ARCHIVED)
    print(f"    DR : n={n:3d} correct={c:3d} -> accuracy {acc*100:5.1f}%   support={sup}")
    print(f"         (matches IDRiD official test set exactly: {dist(te_dr,5)})")
    n2, c2, acc2, sup2 = cm_stats(DME_CM_ARCHIVED)
    print(f"    DME: n={n2:3d} correct={c2:3d} -> accuracy {acc2*100:5.1f}%   support={sup2}")
    print(f"         (matches IDRiD test filtered to DR>=1: {dist(te_dme,3)})")

    # 4 — training curves
    print("\n[4] Archived TensorBoard runs")
    logroot = os.path.join(a.drive, "results", "logs")
    logs = sorted(os.path.join(r, f) for r, _, fs in os.walk(logroot)
                  for f in fs if f.startswith("events"))
    if not logs:
        print("    none found")
    for p in logs:
        v = read_events(p)
        keys = [k for k in v if "accuracy" in k and k.startswith("epoch_")]
        if not keys: continue
        run = "/".join(p.split(os.sep)[-3:-1])
        print(f"    {run}")
        for k in sorted(keys):
            s = sorted(v[k])
            print(f"        {k:28s} epochs={len(s):2d} first={s[0][1]:.3f} "
                  f"last={s[-1][1]:.3f} best={max(x[1] for x in s):.3f}")

    # 5 — verdict
    print("\n" + "=" * W)
    print("[5] VERDICT vs. the numbers claimed in thesis chapter 4")
    print(f"    claimed DR  91.6% on n=262   |  reproducible: NO — no split of any")
    print(f"    claimed DME 87.6% on n=89    |  dataset in this project yields n=262 or n=89,")
    print(f"    claimed ablation 10/7,6/8,3/4 |  and no run producing them exists on disk.")
    print(f"    best archived DR  accuracy   : {acc*100:.1f}%  (below the {majority_acc(te_dr)*100:.1f}% majority floor)")
    print(f"    best archived DME accuracy   : {acc2*100:.1f}%  (below the {majority_acc(te_dme)*100:.1f}% majority floor)")
    print("=" * W)

if __name__ == "__main__":
    main()
