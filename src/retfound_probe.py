"""
retfound_probe.py — I24 Stage 1: cached linear probe on RETFound CFP features.

WHY A PROBE FIRST. A full fine-tune of ViT-L/16 at 224 costs about 5.3x E08's per-image
compute — roughly 19 h for 5 folds, over the run cap. A linear probe needs **one forward pass
per image**, cached once, after which fitting the head is seconds of CPU. It is decisive about
the thing actually in question (does this representation carry more DR/DME signal than ours?)
and it *is* the LP half of LP-FT, so it answers the staging question by measurement instead of
argument.

THE CONTROL IS E09, NOT E08. RETFound is 224-native — ViT-L/16 with a 197-token position
embedding — and interpolating to 448 would quadruple the token count and the compute. So the
clean one-change comparison is against **E09 (densenet121 @224, 5 folds, EyePACS-pretrained,
DR QWK 0.8389 at matched calibration)**. Comparing against E08 @448 would confound backbone
with resolution.

THE HANDICAP, STATED IN ADVANCE. Our own 224 -> 448 jump is worth about +0.04 QWK on this head
(E09 0.8389 vs E08 0.8646, matched). RETFound is locked to 224, so it must overcome that
before it can beat our best. **Clearing E09 but not E08 or the 0.8828 ensemble is therefore an
informative result about representations, not a loss** — it would say the backbone is worth
more than a doubling of resolution, while still not being the best available pipeline.

PROVENANCE. The checkpoint's sha256 is pinned below and verified at load time: what the kernel
reads is checked, not what its path claims (PROTOCOL.md §9, docs/RETFound_provenance.md).

Usage:
    python src/retfound_probe.py --datasets /kaggle/input --weights <dir-or-file> --out runs/I24
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpora
import manifest
import metrics
from preprocess import crop_retina, IMAGENET_MEAN, IMAGENET_STD

cv2.setNumThreads(0)

ENCODER_SHA256 = "847f9dd0e33bf8d450cc6121295d2919fc4bba3c185757a17ce6427bfa14ed37"
SOURCE_SHA256 = "e1e4f66a1b792eeb6e2efaf158f33be35c8255f36b3d17ed67cd5129da246485"
N_DR, N_DME = 5, 3
SIZE = 224          # RETFound is 224-native; see the module docstring


WEIGHT_BASENAME = "retfound_cfp_encoder.pth"


def find_weights(path):
    """Locate the encoder without assuming where Kaggle mounted it.

    ISSUES.md §15: Kaggle sometimes mounts every attached dataset under a single
    /kaggle/input/datasets/ directory rather than one top-level directory each, so a path
    built as /kaggle/input/<dataset-slug> simply does not exist. A one-level assumption cost
    E08 an entire external evaluation, and it cost this probe its first launch. Search
    recursively, by filename, and say what was found.
    """
    if os.path.isfile(path):
        return path
    exact = sorted(glob.glob(os.path.join(path, "**", WEIGHT_BASENAME), recursive=True))
    if exact:
        print(f"[weights] found {len(exact)} candidate(s) by name; using {exact[0]}", flush=True)
        return exact[0]
    any_pth = sorted(glob.glob(os.path.join(path, "**", "*.pth"), recursive=True))
    if any_pth:
        print(f"[weights] {WEIGHT_BASENAME} not found; {len(any_pth)} other .pth present: "
              f"{[os.path.basename(h) for h in any_pth[:5]]}", flush=True)
        return any_pth[0]
    listing = []
    for root, dirs, files in os.walk(path):
        listing.append(root)
        if len(listing) > 12:
            break
    raise SystemExit(f"no .pth anywhere under {path}. Directories seen:\n  " +
                     "\n  ".join(listing))


def load_encoder(path, device):
    """Load the RETFound encoder into a timm ViT-L/16, verifying the file by hash first."""
    import timm
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"[weights] {os.path.basename(path)}  sha256 {h}", flush=True)
    if h != ENCODER_SHA256:
        raise SystemExit(
            f"checkpoint hash mismatch — refusing to run.\n"
            f"  expected {ENCODER_SHA256}\n  got      {h}\n"
            f"A path is not provenance (PROTOCOL.md §9). If the artefact legitimately "
            f"changed, update ENCODER_SHA256 deliberately and say why.")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    if ck.get("source_sha256") and ck["source_sha256"] != SOURCE_SHA256:
        raise SystemExit("stripped encoder does not carry the expected source sha256")

    m = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    # A silent partial load would give ImageNet-random features and look like a null result.
    real_missing = [k for k in missing if not k.startswith("head")]
    print(f"[weights] loaded: {len(sd)} tensors; missing {len(real_missing)}, "
          f"unexpected {len(unexpected)}", flush=True)
    if real_missing:
        raise SystemExit(f"encoder did not fully load; missing {real_missing[:8]} ... "
                         f"a partially loaded backbone would be indistinguishable from a "
                         f"null result")
    return m.to(device).eval()


class Plain(torch.utils.data.Dataset):
    """Deterministic, no augmentation — a probe caches features once."""

    def __init__(self, rows, size=SIZE):
        self.rows, self.size = rows, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = cv2.imread(self.rows[i]["path"], cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.size, self.size, 3), np.uint8)
        img = crop_retina(img)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), i


@torch.no_grad()
def extract(model, rows, device, batch, workers):
    dl = torch.utils.data.DataLoader(Plain(rows), batch_size=batch, shuffle=False,
                                     num_workers=workers, pin_memory=True)
    feats = np.zeros((len(rows), model.num_features), dtype=np.float32)
    t0 = time.time()
    for x, idx in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16,
                            enabled=device.startswith("cuda")):
            f = model(x)
        feats[idx.numpy()] = f.float().cpu().numpy()
    print(f"[extract] {len(rows)} images in {time.time()-t0:.0f}s "
          f"-> {feats.shape}", flush=True)
    return feats


def ordinal_probe(ftr, y, folds, k, epochs=300, lr=1e-3, wd=1e-4, seed=0):
    """Cross-fitted ordinal head on frozen features: K-1 thresholds, P(y > j).

    Fold f's head is trained on the other folds only, so no image contributes to its own
    prediction (PROTOCOL.md §3).
    """
    torch.manual_seed(seed)
    out = np.zeros((len(y), k - 1), dtype=np.float32)
    mu, sd = ftr.mean(0, keepdims=True), ftr.std(0, keepdims=True) + 1e-6
    for f in sorted(set(folds.tolist())):
        te, tr = folds == f, folds != f
        # standardisation fitted on TRAIN folds only (PROTOCOL.md §8.6)
        mu_f = ftr[tr].mean(0, keepdims=True)
        sd_f = ftr[tr].std(0, keepdims=True) + 1e-6
        Xtr = torch.tensor((ftr[tr] - mu_f) / sd_f)
        Xte = torch.tensor((ftr[te] - mu_f) / sd_f)
        ks = torch.arange(k - 1).unsqueeze(0)
        Ttr = (torch.tensor(y[tr]).unsqueeze(1) > ks).float()
        head = nn.Linear(ftr.shape[1], k - 1)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(head(Xtr), Ttr)
            loss.backward()
            opt.step()
        with torch.no_grad():
            out[te] = head(Xte).numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["/kaggle/input"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--splits", default="data/splits/dev_v1.json")
    ap.add_argument("--out", default="runs/I24")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)
    model = load_encoder(find_weights(a.weights), dev)

    rows = corpora.build(a.datasets, ("IDRiD", "Messidor-2"))
    split = json.load(open(a.splits))
    rows = [r for r in rows if r["uid"] in split["folds"]]
    folds = np.array([split["folds"][r["uid"]] for r in rows])
    print(f"{len(rows)} images, folds {sorted(set(folds.tolist()))}", flush=True)

    feats = extract(model, rows, dev, a.batch, a.workers)
    np.savez_compressed(os.path.join(a.out, "features.npz"),
                        uids=np.array([r["uid"] for r in rows]), feats=feats, folds=folds)

    res = {"run_id": "I24_RETFOUND_PROBE",
           "backbone": "RETFound CFP ViT-L/16 @224 (frozen), linear ordinal probe",
           "encoder_sha256": ENCODER_SHA256, "source_sha256": SOURCE_SHA256,
           "split_fingerprint": split.get("fingerprint"),
           "consumption": manifest.build(rows, a.datasets),
           "metrics": {}}

    y_dr = np.array([r["dr"] if r["dr"] is not None else -1 for r in rows])
    y_dme = np.array([r["dme"] if (r.get("dme") is not None and
                                   r.get("dme_label_space") == "3class") else -1
                      for r in rows])
    groups = np.array([split["groups"].get(r["uid"], r["uid"]) for r in rows])

    for head, y, k in (("dr", y_dr, N_DR), ("dme_ungated", y_dme, N_DME)):
        keep = y >= 0
        logits = ordinal_probe(feats[keep], y[keep], folds[keep], k, epochs=a.epochs)
        score = torch.sigmoid(torch.tensor(logits)).sum(1).numpy()
        grades = np.clip(np.floor(score + 0.5).astype(int), 0, k - 1)
        rep = metrics.report(y[keep], grades, k, groups=groups[keep].tolist())
        res["metrics"][head] = rep
        np.savez_compressed(os.path.join(a.out, f"probe_{head}.npz"),
                            logits=logits, y=y[keep],
                            uids=np.array([r["uid"] for r, m in zip(rows, keep) if m]),
                            folds=folds[keep])
        print(f"\n{head}: n={int(keep.sum())}  acc {rep['accuracy']*100:.2f}  "
              f"QWK {rep['qwk']:.4f} {[round(x, 4) for x in rep['qwk_ci95']]}  "
              f"floor {rep['majority_floor']*100:.1f}", flush=True)

    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=str)
    print(f"\nwrote {a.out}/results.json")


if __name__ == "__main__":
    main()
