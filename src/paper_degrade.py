"""
paper_degrade.py — can image quality alone produce the under-grading seen on DDR?
(docs/PAPER_REVISION_preregistration.md, A2 and A3)

A2: the development pool is re-scored OUT OF FOLD -- every image by the fold model that did not
train on it -- after each of five fixed degradations of the cached image, plus the undegraded
cache as a control. Writes <out>/degrade_<MODEL>_<condition>.npz (uids, y_dr, folds, dr_logits).

A3: per-corpus image statistics on up to --stats-n images per corpus: source resolution, JPEG
quality estimated from the luminance quantisation table, mean brightness and RMS contrast
inside the retinal mask, and Laplacian variance at a common 512 px scale. Writes
<out>/image_stats.json.

Shortcut check (added after the second-stage plan, post hoc): with --per-image, acquisition
statistics of every image of the named corpora, with its grade, in <out>/per_image_<corpus>.csv.

The analysis of the predictions is done offline (src/paper_mechanism.py).

Usage (inside a Kaggle kernel):
    python src/paper_degrade.py --datasets /kaggle/input --out OUT \
        --model E08=E08 --model E11FULL=E11+E19E11C --stats-corpora IDRiD,Messidor-2,APTOS,DDR
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpora
from model import MultiOutputNet
from train import FundusDataset, build_cache, predict, CACHE_EXT
from preprocess import crop_retina


def _jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _resize_long(img, side):
    h, w = img.shape[:2]
    s = side / max(h, w)
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


# fixed before the run; see the revision plan
CONDITIONS = {
    "orig": lambda im: im,
    "jpeg30_512": lambda im: _jpeg(_resize_long(im, 512), 30),
    "down256": lambda im: _resize_long(im, 256),
    "blur2": lambda im: cv2.GaussianBlur(im, (0, 0), 2.0),
    "gamma18": lambda im: np.clip(255.0 * (im / 255.0) ** 1.8, 0, 255).astype(np.uint8),
    "contrast60": lambda im: np.clip(im.astype(np.float32).mean() +
                                     0.6 * (im.astype(np.float32) - im.astype(np.float32).mean()),
                                     0, 255).astype(np.uint8),
}

# IJG standard luminance quantisation table (quality 50)
_STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
                      14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
                      18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                      49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
                     dtype=np.float64)


def jpeg_quality(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            q = getattr(im, "quantization", None)
            if not q:
                return None
            t = np.array(q[0], dtype=np.float64)
    except Exception:
        return None
    s = 100.0 * t.mean() / _STD_LUMA.mean()
    return float((200.0 - s) / 2.0) if s <= 100 else float(5000.0 / s)


def image_stats(rows, n, seed=0):
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(rows))[:n]
    res, q, bright, contrast, sharp = [], [], [], [], []
    for i in pick:
        r = rows[i]
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        res.append(max(img.shape[:2]))
        jq = jpeg_quality(r["path"]) if r["path"].lower().endswith((".jpg", ".jpeg")) else None
        if jq is not None:
            q.append(jq)
        c = crop_retina(img)
        g = cv2.cvtColor(_resize_long(c, 512), cv2.COLOR_BGR2GRAY).astype(np.float64)
        m = g > 12
        if m.sum() < 100:
            continue
        bright.append(float(g[m].mean()))
        contrast.append(float(g[m].std()))
        lap = cv2.Laplacian(g, cv2.CV_64F)
        sharp.append(float(lap[m].var()))

    def summ(v):
        v = np.asarray(v, dtype=np.float64)
        if not len(v):
            return None
        return {"n": int(len(v)), "median": float(np.median(v)),
                "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75))}
    return {"resolution": summ(res), "jpeg_quality": summ(q), "png_or_unknown": int(len(pick) - len(q)),
            "brightness": summ(bright), "rms_contrast": summ(contrast), "laplacian_var": summ(sharp)}


def per_image_stats(rows, path):
    """Acquisition statistics of EVERY image, with its grade, for a shortcut check: can the way
    an image was acquired or stored predict its grade without looking at the retina? Columns:
    uid, dr, height, width, file_bytes, jpeg_quality, retina_fraction (share of the frame inside
    the retinal mask), brightness, rms_contrast, laplacian_var (the last three as in A3)."""
    import csv
    cols = ["uid", "dr", "height", "width", "file_bytes", "jpeg_quality", "retina_fraction",
            "brightness", "rms_contrast", "laplacian_var"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, wd = img.shape[:2]
            jq = jpeg_quality(r["path"]) if r["path"].lower().endswith((".jpg", ".jpeg")) else None
            full = cv2.cvtColor(_resize_long(img, 512), cv2.COLOR_BGR2GRAY)
            frac = float((full > 12).mean())
            g = cv2.cvtColor(_resize_long(crop_retina(img), 512), cv2.COLOR_BGR2GRAY).astype(np.float64)
            m = g > 12
            if m.sum() < 100:
                stats = ["", "", ""]
            else:
                lap = cv2.Laplacian(g, cv2.CV_64F)
                stats = [f"{g[m].mean():.3f}", f"{g[m].std():.3f}", f"{lap[m].var():.3f}"]
            w.writerow([r["uid"], r["dr"], h, wd, os.path.getsize(r["path"]),
                        "" if jq is None else f"{jq:.1f}", f"{frac:.4f}", *stats])


def find_ckpts(srcdirs, root="/kaggle/input"):
    found = {}
    for dirpath, _, files in os.walk(root):
        if os.path.basename(dirpath) in srcdirs:
            for fn in files:
                if fn.startswith("best_") and fn.endswith(".pt"):
                    if fn in found:
                        raise SystemExit(f"{fn} supplied twice ({found[fn]} and {dirpath})")
                    found[fn] = os.path.join(dirpath, fn)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["/kaggle/input"])
    ap.add_argument("--splits", default="data/splits/dev_v1.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/kaggle/temp/cache_dev560")
    ap.add_argument("--model", action="append", default=[],
                    help="NAME=rundir[+rundir...]; fold weights best_<f>.pt found under /kaggle/input")
    ap.add_argument("--stats-corpora", default="")
    ap.add_argument("--stats-n", type=int, default=1000)
    ap.add_argument("--per-image", default="",
                    help="corpora whose every image gets acquisition statistics (shortcut check)")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.stats_corpora:
        stats = {}
        for c in a.stats_corpora.split(","):
            rows = corpora.build(a.datasets, (c,))
            t0 = time.time()
            stats[c] = image_stats(rows, a.stats_n)
            stats[c]["n_corpus"] = len(rows)
            print(f"[stats] {c}: {json.dumps(stats[c])[:300]}  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(stats, open(os.path.join(a.out, "image_stats.json"), "w"), indent=1)

    for c in [c for c in a.per_image.split(",") if c]:
        t0 = time.time()
        rows = corpora.build(a.datasets, (c,))
        per_image_stats(rows, os.path.join(a.out, f"per_image_{c.lower()}.csv"))
        print(f"[per-image] {c}: {len(rows)} images ({time.time() - t0:.0f}s)", flush=True)

    if not a.model:
        return
    rows = corpora.build(a.datasets, ("IDRiD", "Messidor-2"))
    split = json.load(open(a.splits))
    for r in rows:
        r["fold"] = split["folds"][r["uid"]]
    build_cache(rows, a.cache)
    caches = {}
    for name, fn in CONDITIONS.items():
        d = a.cache if name == "orig" else f"{a.cache}_{name}"
        if name != "orig":
            os.makedirs(d, exist_ok=True)
            for r in rows:
                dst = os.path.join(d, r["uid"] + CACHE_EXT)
                if os.path.exists(dst):
                    continue
                im = cv2.imread(os.path.join(a.cache, r["uid"] + CACHE_EXT), cv2.IMREAD_COLOR)
                if im is None:
                    continue
                cv2.imwrite(dst, fn(im), [cv2.IMWRITE_JPEG_QUALITY, 100])
        caches[name] = d
        print(f"[degrade] condition {name} ready in {d}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.float16 if device == "cuda" else None
    for spec in a.model:
        name, dirs = spec.split("=")
        ck = find_ckpts(set(dirs.split("+")))
        print(f"[degrade] {name}: {sorted(ck)}", flush=True)
        for cond, cdir in caches.items():
            out_l, out_u, out_y, out_f = [], [], [], []
            for fn_ck, path in sorted(ck.items()):
                f = int(fn_ck.split("_")[1].split(".")[0])
                st = torch.load(path, map_location=device, weights_only=False)
                cfg = st["config"]
                model = MultiOutputNet(cfg["backbone"], False, cfg["head"], cfg["hidden"],
                                       cfg["dropout"]).to(device)
                model.load_state_dict({k: v.to(device) for k, v in st["state_dict"].items()})
                fr = [r for r in rows if r["fold"] == f]
                dl = DataLoader(FundusDataset(fr, cdir, cfg["size"], False), batch_size=a.batch,
                                shuffle=False, num_workers=2, pin_memory=True)
                dr_l, _, order = predict(model, dl, device, cfg["head"], amp, tta=True)
                inv = torch.argsort(order)
                out_l.append(dr_l[inv].numpy()); out_u += [r["uid"] for r in fr]
                out_y += [r["dr"] for r in fr]; out_f += [f] * len(fr)
                del model; torch.cuda.empty_cache()
            np.savez_compressed(os.path.join(a.out, f"degrade_{name}_{cond}.npz"),
                                uids=np.array(out_u), y_dr=np.array(out_y), folds=np.array(out_f),
                                dr_logits=np.concatenate(out_l))
            print(f"[degrade] wrote {name} {cond} ({len(out_u)} images)", flush=True)


if __name__ == "__main__":
    main()
