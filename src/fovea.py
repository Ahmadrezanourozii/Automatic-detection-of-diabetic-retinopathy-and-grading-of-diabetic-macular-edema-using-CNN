"""
fovea.py — locate the macula centre, and measure the error in the units the grade uses.

WHY THIS EXISTS
The DME grade is defined by the distance from the nearest hard exudate to the macula centre,
in disc diameters. Global average pooling over the whole image dilutes that region by roughly
16x at 448 px (1 DD is about 55 px there, so the decisive region is a ~110 px disc in a 448 px
image). Pooling the DME head's features around the macula instead is therefore the one queued
idea that addresses *position* rather than capacity — and DME is the head nothing has moved.

IDRiD publishes fovea coordinates for all 516 images. Messidor-2 does not. So the idea needs a
localiser, and the localiser needs a gate before anything is built on it.

THE GATE, AND WHY IT IS IN DISC DIAMETERS
Measured on IDRiD: the fovea sits a median 0.112 of the short side from the image centre, and
one disc diameter is 0.122 of image width. So the fovea is typically displaced from centre by
about one disc diameter — the same scale as the grading tolerance. A centre-crop proxy is
therefore wrong by about as much as the criterion allows, which is why a localiser is needed
rather than a centre crop.

That also fixes the acceptance threshold. An error of 1 DD is the size of the whole decision
region; an error of 0.5 DD still moves the window by half of it. **Pass at median error below
0.5 DD and 90th percentile below 1.0 DD**, stated before the number is seen.

WHAT CANNOT BE VALIDATED
The localiser is trained and evaluated on IDRiD, out of fold. Messidor-2 has no fovea ground
truth, so its predictions cannot be checked — applying it there is an assumption, and any
result built on it inherits that assumption. This is stated rather than hidden.

Usage:
    python src/fovea.py --datasets /kaggle/input --out runs/E13gate
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpora
from preprocess import crop_retina, IMAGENET_MEAN, IMAGENET_STD

cv2.setNumThreads(0)
DD_FRACTION_OF_WIDTH = 0.122      # one disc diameter, measured from IDRiD fovea-to-disc spans


class FoveaSet(Dataset):
    """Returns the image and the fovea position in [0,1] coordinates of the CROPPED image."""

    def __init__(self, rows, size, train):
        self.rows, self.size, self.train = rows, size, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        h0, w0 = img.shape[:2]
        # crop_retina removes the black border; the label must move with it
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = grey > 12
        rr, cc = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
        y0, y1, x0, x1 = rr[0], rr[-1] + 1, cc[0], cc[-1] + 1
        img = img[y0:y1, x0:x1]
        if r.get("fovea"):
            fx, fy = r["fovea"]
            tx = (fx - x0) / max(1, x1 - x0)
            ty = (fy - y0) / max(1, y1 - y0)
        else:
            tx = ty = -1.0          # prediction mode: no target, never used for a loss

        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.train and np.random.rand() < 0.5:
            img = cv2.flip(img, 1); tx = 1.0 - tx
        if self.train:
            f = img.astype(np.float32) * np.random.uniform(0.9, 1.1)
            img = np.clip(f, 0, 255).astype(np.uint8)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
                torch.tensor([tx, ty], dtype=torch.float32),
                torch.tensor(float(x1 - x0)))          # cropped width, for DD conversion


def build(size=224):
    import torchvision.models as tvm
    m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 128), nn.ReLU(inplace=True),
                         nn.Linear(128, 2), nn.Sigmoid())
    return m


def fit_predict(rows_all, size=224, epochs=25, batch=32, workers=2, device=None,
                verbose=True):
    """Train the localiser on every IDRiD row that HAS ground truth, then predict (fx, fy)
    for every row in `rows_all`.

    Returns {uid: [fx, fy]} in [0,1] coordinates of the retina-cropped image -- the same
    space `preprocess.crop_retina` produces, so the coordinates apply unchanged to the
    training cache (both use the identical threshold-12 bounding box, and normalised
    coordinates are invariant to the cache's downscale).

    THE ASSUMPTION THIS CARRIES, stated because no experiment available to this project can
    check it: the localiser is trained and validated on IDRiD only. Messidor-2 has no fovea
    ground truth. Applying these predictions there is unvalidated transfer. E13gate measured
    the IDRiD accuracy (median 0.196 DD, 90th 0.433 DD out of fold); it says nothing about
    Messidor-2. Any result built on this must report the IDRiD-only number beside it.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    labelled = [r for r in rows_all if r.get("fovea")]
    if verbose:
        print(f"[fovea] training localiser on {len(labelled)} labelled rows, "
              f"predicting for {len(rows_all)}", flush=True)
    dl_tr = DataLoader(FoveaSet(labelled, size, True), batch_size=batch, shuffle=True,
                       num_workers=workers, drop_last=True)
    m = build(size).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4,
                                                total_steps=max(1, len(dl_tr)) * epochs)
    t0 = time.time()
    for ep in range(epochs):
        m.train()
        for x, t, _ in dl_tr:
            loss = nn.functional.smooth_l1_loss(m(x.to(dev)), t.to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    if verbose:
        print(f"[fovea] localiser trained in {time.time()-t0:.0f}s", flush=True)

    # predict for everything, including the labelled rows -- using the SAME predicted
    # coordinates everywhere keeps IDRiD and Messidor-2 on one footing, so a difference
    # between them cannot be an artefact of one corpus getting better coordinates.
    out = {}
    m.eval()
    dl_all = DataLoader(FoveaSet(rows_all, size, False), batch_size=batch,
                        num_workers=workers)
    i = 0
    with torch.no_grad():
        for x, _, _ in dl_all:
            p = m(x.to(dev)).cpu().numpy()
            for row in p:
                out[rows_all[i]["uid"]] = [float(row[0]), float(row[1])]
                i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["/kaggle/input"])
    ap.add_argument("--splits", default="data/splits/dev_v1.json")
    ap.add_argument("--out", default="runs/E13gate")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={dev}")
    rows = [r for r in corpora.build(a.datasets, ("IDRiD",)) if r.get("fovea")]
    folds = json.load(open(a.splits))["folds"]
    for r in rows:
        r["fold"] = folds[r["uid"]]
    print(f"IDRiD images with fovea coordinates: {len(rows)}")
    print(f"ACCEPTANCE, fixed before the numbers: median error < 0.5 DD, 90th pct < 1.0 DD")

    errs_dd, errs_frac = [], []
    for fold in sorted({r["fold"] for r in rows}):
        tr = [r for r in rows if r["fold"] != fold]
        te = [r for r in rows if r["fold"] == fold]
        dl_tr = DataLoader(FoveaSet(tr, a.size, True), batch_size=a.batch, shuffle=True,
                           num_workers=a.workers, drop_last=True)
        dl_te = DataLoader(FoveaSet(te, a.size, False), batch_size=a.batch,
                           num_workers=a.workers)
        m = build(a.size).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=max(1, len(dl_tr)) * a.epochs)
        t0 = time.time()
        for ep in range(a.epochs):
            m.train()
            for x, t, _ in dl_tr:
                loss = nn.functional.smooth_l1_loss(m(x.to(dev)), t.to(dev))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        m.eval()
        with torch.no_grad():
            for x, t, wpx in dl_te:
                p = m(x.to(dev)).cpu().numpy(); t = t.numpy()
                d = np.hypot(*(p - t).T)                       # error in fraction of width
                errs_frac += list(d)
                errs_dd += list(d / DD_FRACTION_OF_WIDTH)      # in disc diameters
        print(f"  fold {fold}: {len(te)} images, {time.time()-t0:.0f}s, "
              f"median {np.median(errs_dd[-len(te):]):.3f} DD")

    e = np.array(errs_dd)
    med, p90, p99 = np.median(e), np.percentile(e, 90), np.percentile(e, 99)
    ok = med < 0.5 and p90 < 1.0
    print(f"\n{'='*70}\nFOVEA LOCALISER — out of fold on {len(e)} IDRiD images\n{'='*70}")
    print(f"  error in disc diameters : median {med:.3f}   90th {p90:.3f}   99th {p99:.3f}")
    print(f"  error as fraction of width: median {np.median(errs_frac):.4f}")
    print(f"  fraction within 0.5 DD: {(e < 0.5).mean()*100:.1f}%   within 1 DD: {(e < 1.0).mean()*100:.1f}%")
    print(f"\n  GATE: {'PASS' if ok else 'FAIL'} (median < 0.5 DD and 90th < 1.0 DD)")
    if not ok:
        print("  A localiser this imprecise would place the DME pooling window off the macula")
        print("  by a clinically meaningful distance. Do not build the macula-pooled head on")
        print("  it for corpora without ground-truth coordinates.")
    json.dump({"n": len(e), "median_dd": float(med), "p90_dd": float(p90),
               "p99_dd": float(p99), "frac_within_0.5dd": float((e < 0.5).mean()),
               "gate_pass": bool(ok),
               "acceptance": "median < 0.5 DD and 90th percentile < 1.0 DD, fixed in advance",
               "caveat": "trained and evaluated on IDRiD only; Messidor-2 has no fovea "
                         "ground truth, so applying it there is an unvalidated assumption"},
              open(os.path.join(a.out, "fovea_gate.json"), "w"), indent=1)
    print(f"\nwrote {a.out}/fovea_gate.json")


if __name__ == "__main__":
    main()
