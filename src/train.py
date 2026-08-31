"""
train.py — the training run. One fold per invocation, or all folds in sequence.

Runs identically on Kaggle and locally. Writes, for every run:
  runs/<ID>/results.json   metrics, full config, git SHA, env versions, runtime, seed
  runs/<ID>/train.log      the stdout of the run
  runs/<ID>/oof_<fold>.npz out-of-fold predictions, so folds can be pooled and compared
                           by paired bootstrap later without retraining
  runs/<ID>/ckpt_<fold>.pt resumable checkpoint (Kaggle kills sessions at the wall clock)

The first line printed is the git SHA, so any log traces back to the code that made it.

Usage
    python src/train.py --datasets /kaggle/input --splits data/splits/dev_v1.json \
        --run-id E05 --folds 0,1,2,3,4 --epochs 30 --size 448 --backbone densenet121
"""
from __future__ import annotations
import argparse, json, math, os, platform, random, subprocess, sys, time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpora
import metrics as M
from model import (MultiOutputNet, multitask_loss, decode, expected_grade,
                   N_DR, N_DME)
from preprocess import crop_retina, IMAGENET_MEAN, IMAGENET_STD

cv2.setNumThreads(0)          # DataLoader workers must not each spawn an OpenCV pool


def git_sha():
    try:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sha = subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "-C", here, "status", "--porcelain"],
                                        stderr=subprocess.DEVNULL).decode().strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "UNKNOWN"


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── image cache ───────────────────────────────────────────────────────────────
CACHE_EXT = ".jpg"
CACHE_QUALITY = 95


def build_cache(rows, cache_dir, cache_size=560, train_size=None):
    """Decode, border-crop and downscale every source image once.

    IDRiD is 4288x2848; decoding it 30 times per fold would dominate the run and burn the
    weekly GPU quota on JPEG decoding. Keyed on uid, which is unique -- keying on the
    filename silently overwrote 103 images once already (ISSUES.md §8).
    """
    if train_size is not None and cache_size < train_size:
        raise SystemExit(
            f"cache_size={cache_size} is below train size={train_size}: every image would "
            f"be UPSAMPLED from the cache and the extra pixels would carry no information. "
            f"E10 ran this way and was a {cache_size}px experiment wearing a "
            f"{train_size}px label (ISSUES.md §18). Raise --cache-size to at least "
            f"--size, and higher where the source images support it.")
    os.makedirs(cache_dir, exist_ok=True)
    assert len({r["uid"] for r in rows}) == len(rows), "uids are not unique"
    todo = [r for r in rows
            if not os.path.exists(os.path.join(cache_dir, r["uid"] + CACHE_EXT))]
    if not todo:
        print(f"[cache] all {len(rows)} images already cached", flush=True)
        return
    print(f"[cache] decoding {len(todo)} images -> {cache_dir}", flush=True)
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            print(f"[cache] WARNING unreadable: {r['path']}")
            continue
        img = crop_retina(img)
        h, w = img.shape[:2]
        if max(h, w) > cache_size:
            s = cache_size / max(h, w)
            img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(cache_dir, r["uid"] + CACHE_EXT), img,
                    [cv2.IMWRITE_JPEG_QUALITY, CACHE_QUALITY])
        if i % 250 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"[cache]   {i}/{len(todo)}  {el:.0f}s, ~{el/i*(len(todo)-i):.0f}s left",
                  flush=True)


# ── augmentation ──────────────────────────────────────────────────────────────
def augment(img, rng):
    """Fundus-appropriate augmentation, in numpy/cv2 so it has no extra dependencies.

    Rotation is unrestricted: a retina has no canonical up, and the graders' criteria
    (lesion counts, exudate distance to the macula) are rotation-invariant. Flips are
    likewise safe -- a mirrored right eye looks like a left eye, which the corpus already
    contains. Colour jitter is kept mild because DR grading depends on subtle red-lesion
    contrast that aggressive jitter destroys.
    """
    h, w = img.shape[:2]
    angle = rng.uniform(0, 360)
    scale = rng.uniform(0.88, 1.12)
    tx, ty = rng.uniform(-0.04, 0.04) * w, rng.uniform(-0.04, 0.04) * h
    Mrot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    Mrot[0, 2] += tx; Mrot[1, 2] += ty
    img = cv2.warpAffine(img, Mrot, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
    if rng.random() < 0.5:
        img = cv2.flip(img, 0)

    f = img.astype(np.float32)
    f *= rng.uniform(0.88, 1.12)                                  # brightness
    mean = f.mean()
    f = (f - mean) * rng.uniform(0.88, 1.12) + mean                # contrast
    if rng.random() < 0.3:                                         # gamma
        f = 255.0 * np.power(np.clip(f, 0, 255) / 255.0, rng.uniform(0.85, 1.18))
    return np.clip(f, 0, 255).astype(np.uint8)


DD_FRACTION_OF_WIDTH = 0.122      # one disc diameter, measured from IDRiD fovea-to-disc spans


def macula_crop(img, fovea, dd=3.0):
    """Square crop of side `dd` disc diameters centred on the fovea (I07).

    `fovea` is (fx, fy) in [0,1] of this image. The window is clamped to stay inside the
    frame rather than zero-padded, so the crop always contains real retina -- a padded
    border would be a constant the DME head could key on, and it would differ
    systematically between central and peripheral foveae.
    """
    h, w = img.shape[:2]
    half = 0.5 * dd * DD_FRACTION_OF_WIDTH * w
    cx, cy = fovea[0] * w, fovea[1] * h
    x0 = int(round(min(max(cx - half, 0), max(0, w - 2 * half))))
    y0 = int(round(min(max(cy - half, 0), max(0, h - 2 * half))))
    x1 = min(w, x0 + int(round(2 * half)))
    y1 = min(h, y0 + int(round(2 * half)))
    crop = img[y0:y1, x0:x1]
    return crop if crop.size else img


class FundusDataset(Dataset):
    def __init__(self, rows, cache_dir, size, train, seed=0,
                 fovea=None, macula_size=224, macula_dd=3.0):
        self.rows, self.cache_dir, self.size, self.train = rows, cache_dir, size, train
        self.seed = seed
        self.fovea, self.macula_size, self.macula_dd = fovea, macula_size, macula_dd

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(os.path.join(self.cache_dir, r["uid"] + CACHE_EXT), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.size, self.size, 3), np.uint8)
        # The macula window is cut from the UNAUGMENTED cache, then augmented on its own.
        # Cutting after a random rotation would move the fovea out of the window, which is
        # the one thing this experiment must not do.
        xm = None
        if self.fovea is not None:
            f = self.fovea.get(r["uid"])
            crop = macula_crop(img, f, self.macula_dd) if f else img
            if self.train:
                rng_m = random.Random((self.seed * 7_654_321 + i * 104_729 +
                                       torch.initial_seed()) % (2 ** 31))
                crop = augment(crop, rng_m)
            crop = cv2.resize(crop, (self.macula_size, self.macula_size),
                              interpolation=cv2.INTER_AREA)
            xm = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            xm = (xm - IMAGENET_MEAN) / IMAGENET_STD

        if self.train:
            rng = random.Random((self.seed * 1_000_003 + i * 7919 +
                                 torch.initial_seed()) % (2 ** 31))
            img = augment(img, rng)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD

        cands = r["dme_candidates"]
        lo, hi = (min(cands), max(cands)) if cands else (-1, -1)
        out = {
            "x": torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
            "dr": torch.tensor(r["dr"] if r["dr"] is not None else -1, dtype=torch.long),
            "dr_mask": torch.tensor(1.0 if r["dr"] is not None else 0.0),
            "dme_lo": torch.tensor(lo, dtype=torch.long),
            "dme_hi": torch.tensor(hi, dtype=torch.long),
            "idx": torch.tensor(i, dtype=torch.long),
        }
        if xm is not None:
            out["xm"] = torch.from_numpy(np.ascontiguousarray(xm.transpose(2, 0, 1)))
        return out


def make_sampler(rows, seed=0):
    """Balance DR classes. The rare grades (severe, proliferative) are 3-5 % of the pool,
    and a model that never sees them cannot be scored on per-class recall."""
    counts = np.bincount([r["dr"] for r in rows], minlength=N_DR).astype(np.float64)
    counts[counts == 0] = 1
    w = np.array([1.0 / counts[r["dr"]] for r in rows])
    g = torch.Generator(); g.manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 num_samples=len(rows), replacement=True, generator=g)


# ── EMA ───────────────────────────────────────────────────────────────────────
class EMA:
    """Exponential moving average of the weights; evaluated instead of the raw model.

    Worth the twenty lines on a 2 260-image corpus: single-batch noise at batch 16 makes
    the raw weights jump around enough that 'best epoch' becomes partly luck."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def copy_to(self, model):
        sd = model.state_dict()
        merged = {k: (self.shadow[k].to(sd[k].dtype) if k in self.shadow else v)
                  for k, v in sd.items()}
        model.load_state_dict(merged)


# ── evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, loader, device, head, amp_dtype, tta=False):
    """Forward pass over a loader. With `tta`, average logits over the four dihedral flips.

    Flips are the safe augmentation to average over here: a retina has no canonical
    orientation, and a mirrored right eye is a plausible left eye, so all four views are
    in-distribution rather than adversarial.
    """
    model.eval()
    dr_logits, dme_logits, order = [], [], []
    for b in loader:
        x = b["x"].to(device, non_blocking=True)
        xm = b["xm"].to(device, non_blocking=True) if "xm" in b else None
        views = [x]
        if tta:
            views += [torch.flip(x, [3]), torch.flip(x, [2]), torch.flip(x, [2, 3])]
        # the macula crop is flipped the same way, so both branches see the same view
        views_m = ([xm] if xm is not None else [None])
        if tta and xm is not None:
            views_m += [torch.flip(xm, [3]), torch.flip(xm, [2]), torch.flip(xm, [2, 3])]
        elif tta:
            views_m = [None] * len(views)
        dl_sum = ml_sum = None
        for v, vm in zip(views, views_m):
            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                dl, ml = model(v, vm)
            dl, ml = dl.float(), ml.float()
            dl_sum = dl if dl_sum is None else dl_sum + dl
            ml_sum = ml if ml_sum is None else ml_sum + ml
        dr_logits.append((dl_sum / len(views)).cpu())
        dme_logits.append((ml_sum / len(views)).cpu())
        order.append(b["idx"])
    return (torch.cat(dr_logits), torch.cat(dme_logits), torch.cat(order))


def evaluate(rows, dr_logits, dme_logits, head, n_boot=1000):
    """The standard metric block for both heads. DME ungated is primary (PROTOCOL.md §5.1)."""
    dr_pred = decode(dr_logits, N_DR, head).numpy()
    dme_pred = decode(dme_logits, N_DME, head).numpy()
    dr_true = np.array([r["dr"] for r in rows])
    groups = np.array([r["group"] for r in rows])

    out = {"dr": M.report(dr_true, dr_pred, N_DR, groups=groups,
                          n_boot=n_boot, referable_from=2)}

    # 3-class DME is scored ONLY on corpora that grade all three levels. Messidor-2's
    # referable rows do carry an exact grade -- but they are all grade 2, so including
    # them would lift the majority floor from 47 % to 59 % and hand the model free
    # accuracy on a biased set. See corpora.py.
    exact = np.array([r.get("dme_label_space") == "3class" and r["dme"] is not None
                      for r in rows])
    if exact.any():
        dme_true = np.array([r["dme"] for r in rows])[exact]
        out["dme_ungated"] = M.report(dme_true, dme_pred[exact], N_DME,
                                      groups=groups[exact], n_boot=n_boot,
                                      referable_from=2)
        gated = exact & (dr_true >= 1)
        if gated.sum() > 10:
            out["dme_gated"] = M.report(np.array([r["dme"] for r in rows])[gated],
                                        dme_pred[gated], N_DME, groups=groups[gated],
                                        n_boot=n_boot, referable_from=2)

    # referable DME as a binary call, which is the corpus-wide screening decision and the
    # only DME quantity Messidor-2 can be scored on at all
    ref_known = np.array([r["dme_candidates"] is not None and
                          (min(r["dme_candidates"]) == 2 or max(r["dme_candidates"]) < 2)
                          for r in rows])
    if ref_known.any():
        ref_true = np.array([1 if min(r["dme_candidates"]) == 2 else 0
                             for r in rows if r["dme_candidates"] is not None and
                             (min(r["dme_candidates"]) == 2 or max(r["dme_candidates"]) < 2)])
        ref_pred = (dme_pred[ref_known] >= 2).astype(int)
        out["dme_referable_binary"] = M.report(ref_true, ref_pred, 2,
                                               groups=groups[ref_known], n_boot=n_boot)
    return out


def fmt(head_name, rep):
    lo, hi = rep["accuracy_ci95"]
    return (f"    {head_name:22s} n={rep['n']:5d}  acc {rep['accuracy']*100:5.1f}% "
            f"[{lo*100:.1f},{hi*100:.1f}]  floor {rep['majority_floor']*100:5.1f}%  "
            f"QWK {rep['qwk']:6.3f}  F1 {rep['macro_f1']:.3f}"
            + ("  BEATS" if rep["beats_floor"] else "  below-floor"))


# ── pretraining ───────────────────────────────────────────────────────────────
def pretrain(rows, args, device, out_dir, amp_dtype):
    """Train the shared extractor on a large DR corpus before the folds.

    EyePACS is ~35 k images with DR grades and no DME labels at all. Because the loss masks
    each head per sample, those rows train the backbone and the DR head and contribute
    nothing to the DME head -- no special code path is needed.

    Done ONCE and reused by every fold. EyePACS is disjoint from the development pool, so
    this leaks nothing; running it per fold would cost five times as much for the same
    weights. The result is cached to disk so a killed Kaggle session does not repeat it.
    """
    ck = os.path.join(out_dir, "pretrained.pt")
    if os.path.exists(ck):
        print(f"[pretrain] reusing {ck}", flush=True)
        return torch.load(ck, map_location="cpu", weights_only=False)

    print(f"\n{'='*78}\nPRETRAIN on {len(rows)} images "
          f"({args.pretrain_epochs} epochs)\n{'='*78}", flush=True)
    ds = FundusDataset(rows, args.cache, args.size, True, args.seed)
    dl = DataLoader(ds, batch_size=args.batch, sampler=make_sampler(rows, args.seed),
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0)
    model = MultiOutputNet(args.backbone, True, args.head, args.hidden, args.dropout).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_pretrain,
                            weight_decay=args.weight_decay)
    steps = max(1, len(dl)) * args.pretrain_epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr_pretrain,
                                                total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(device.split(":")[0], enabled=amp_dtype == torch.float16)
    for ep in range(args.pretrain_epochs):
        model.train(); t0, tot = time.time(), 0.0
        for b in dl:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            if args.channels_last:
                b["x"] = b["x"].to(memory_format=torch.channels_last)
                if "xm" in b:
                    b["xm"] = b["xm"].to(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                dl_, ml_ = model(b["x"], b.get("xm"))
                loss, _, _ = multitask_loss(dl_, ml_, b, 1.0, 0.0,
                                            smoothing=args.smoothing, head=args.head)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        print(f"  pretrain ep {ep:2d}/{args.pretrain_epochs}  "
              f"loss {tot/max(1,len(dl)):.4f}  {time.time()-t0:.0f}s", flush=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(state, ck)
    print(f"[pretrain] saved {ck}", flush=True)
    return state


# ── one fold ──────────────────────────────────────────────────────────────────
def run_fold(rows, fold, args, device, out_dir, amp_dtype, init_state=None):
    tr = [r for r in rows if r["fold"] != fold]
    va = [r for r in rows if r["fold"] == fold]
    print(f"\n{'='*78}\nFOLD {fold}   train={len(tr)}  val={len(va)}\n{'='*78}", flush=True)

    fov = getattr(args, "fovea_map", None)
    ds_tr = FundusDataset(tr, args.cache, args.size, True, args.seed,
                          fovea=fov, macula_size=args.macula_size,
                          macula_dd=args.macula_dd)
    ds_va = FundusDataset(va, args.cache, args.size, False, args.seed,
                          fovea=fov, macula_size=args.macula_size,
                          macula_dd=args.macula_dd)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, sampler=make_sampler(tr, args.seed),
                       num_workers=args.workers, pin_memory=True, drop_last=True,
                       persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=args.batch * 2, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    model = MultiOutputNet(args.backbone, True, args.head, args.hidden, args.dropout).to(device)
    if init_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in init_state.items()})
        print("  initialised from the pretrained backbone", flush=True)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    head_params = list(model.dr_head.parameters()) + list(model.dme_head.parameters())
    head_ids = {id(p) for p in head_params}
    bb_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([
        {"params": bb_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    steps = max(1, len(dl_tr)) * args.epochs
    warm = max(1, int(0.05 * steps))

    def lr_at(step):
        if step < warm:
            return step / warm
        p = (step - warm) / max(1, steps - warm)
        return 0.5 * (1 + math.cos(math.pi * p)) * (1 - args.min_lr_frac) + args.min_lr_frac

    # ── I21: LP-FT. Train the heads on a frozen backbone first, then unfreeze at a much
    # lower LR with its own warmup. The point is to stop fine-tuning from distorting a
    # representation that E01 showed is already linearly separable. Note what this does NOT
    # claim: D01 exonerated the fine-tuning mechanism as the cause of the archived collapse,
    # so the floor-lift argument is withdrawn and this is an open question, not a favourite.
    if args.lpft:
        probe_steps = max(1, len(dl_tr)) * args.lpft_epochs
        ft_warm = max(1, len(dl_tr) * args.lpft_warmup_epochs)
        bb_scale = args.lpft_lr_backbone / max(1e-12, args.lr_backbone)

        def lr_bb(step):
            if step < probe_steps:
                return 0.0                     # probe phase: backbone frozen
            w = step - probe_steps
            if w < ft_warm:
                return (w / ft_warm) * bb_scale
            q = (step - probe_steps - ft_warm) / max(1, steps - probe_steps - ft_warm)
            cos = 0.5 * (1 + math.cos(math.pi * q)) * (1 - args.min_lr_frac) + args.min_lr_frac
            return cos * bb_scale

        sched = torch.optim.lr_scheduler.LambdaLR(opt, [lr_bb, lr_at])
        print(f"  [LP-FT] probe {args.lpft_epochs} epochs (backbone frozen), then unfreeze "
              f"at {args.lpft_lr_backbone:g} with {args.lpft_warmup_epochs}-epoch warmup",
              flush=True)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler(device.split(":")[0], enabled=amp_dtype == torch.float16)
    ema = EMA(model, args.ema) if args.ema > 0 else None

    ckpt_path = os.path.join(out_dir, f"ckpt_{fold}.pt")
    start_epoch, best = 0, -1e9
    if args.resume and os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); start_epoch = st["epoch"] + 1
        best = st.get("best", -1e9)
        if ema and st.get("ema"):
            ema.shadow = {k: v.to(device) for k, v in st["ema"].items()}
        print(f"[resume] fold {fold} from epoch {start_epoch}", flush=True)

    # built once, not per epoch: instantiating a backbone every epoch cost more than the
    # EMA itself and re-triggered the timm/torchvision probe each time
    ema_model = None
    if ema:
        ema_model = MultiOutputNet(args.backbone, False, args.head,
                                   args.hidden, args.dropout).to(device)

    best_state, history = None, []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.lpft:
            # zeroing the backbone LR alone would still backprop through it every step;
            # detaching saves that compute and makes the probe phase genuinely a probe
            probing = epoch < args.lpft_epochs
            for prm in bb_params:
                prm.requires_grad_(not probing)
            if epoch in (0, args.lpft_epochs):
                print(f"  [LP-FT] epoch {epoch}: backbone "
                      f"{'FROZEN (probe)' if probing else 'UNFROZEN (fine-tune)'}",
                      flush=True)
        t0, tot, ndr, ndme = time.time(), 0.0, 0.0, 0.0
        for b in dl_tr:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            if args.channels_last:
                b["x"] = b["x"].to(memory_format=torch.channels_last)
                if "xm" in b:
                    b["xm"] = b["xm"].to(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                dl_, ml_ = model(b["x"], b.get("xm"))
                loss, l_dr, l_dme = multitask_loss(
                    dl_, ml_, b, args.alpha, args.beta,
                    smoothing=args.smoothing, head=args.head)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt); scaler.update(); sched.step()
            if ema:
                ema.update(model)
            tot += loss.item(); ndr += l_dr.item(); ndme += l_dme.item()

        n = max(1, len(dl_tr))
        eval_model = model
        if ema:
            ema_model.load_state_dict(model.state_dict())
            ema.copy_to(ema_model)
            eval_model = ema_model
        dr_lg, dme_lg, order = predict(eval_model, dl_va, device, args.head, amp_dtype)
        inv = torch.argsort(order)
        rep = evaluate(va, dr_lg[inv], dme_lg[inv], args.head, n_boot=200)

        # selection score: the two primary quantities, weighted as the loss is
        score = args.alpha * rep["dr"]["qwk"] + args.beta * rep.get(
            "dme_ungated", {"qwk": 0.0})["qwk"]
        history.append({"epoch": epoch, "loss": tot / n, "loss_dr": ndr / n,
                        "loss_dme": ndme / n, "dr_qwk": rep["dr"]["qwk"],
                        "dr_acc": rep["dr"]["accuracy"],
                        "dme_qwk": rep.get("dme_ungated", {}).get("qwk"),
                        "dme_acc": rep.get("dme_ungated", {}).get("accuracy"),
                        "score": score, "lr": sched.get_last_lr()[1]})
        star = ""
        if score > best:
            best, star = score, "  *best"
            best_state = {k: v.detach().cpu().clone()
                          for k, v in eval_model.state_dict().items()}
            np.savez_compressed(os.path.join(out_dir, f"oof_{fold}.npz"),
                                uids=np.array([r["uid"] for r in va]),
                                dr_logits=dr_lg[inv].numpy(),
                                dme_logits=dme_lg[inv].numpy())
        print(f"  ep {epoch:3d}/{args.epochs}  loss {tot/n:.4f} "
              f"(dr {ndr/n:.4f} dme {ndme/n:.4f})  "
              f"DR qwk {rep['dr']['qwk']:.3f} acc {rep['dr']['accuracy']*100:.1f}%  "
              f"DME qwk {(rep.get('dme_ungated',{}).get('qwk') or 0):.3f} "
              f"acc {(rep.get('dme_ungated',{}).get('accuracy') or 0)*100:.1f}%  "
              f"score {score:.4f}  {time.time()-t0:.0f}s{star}", flush=True)

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "best": best,
                    "ema": ema.shadow if ema else None}, ckpt_path)

    if best_state is not None:
        # The resumable ckpt_<fold>.pt carries optimiser and scheduler state and is ~90 MB;
        # this is the selected weights alone (~28 MB), which is what external validation,
        # ensembling and Grad-CAM actually need. Without it a finished run leaves no model.
        torch.save({"state_dict": best_state, "config": vars(args), "fold": fold,
                    "score": best},
                   os.path.join(out_dir, f"best_{fold}.pt"))
        final_model = ema_model if ema_model is not None else model
        final_model.load_state_dict(best_state)
        dr_lg, dme_lg, order = predict(final_model, dl_va, device, args.head,
                                       amp_dtype, tta=args.tta)
        inv = torch.argsort(order)
        final = evaluate(va, dr_lg[inv], dme_lg[inv], args.head, n_boot=1000)
    else:
        final = rep

    print(f"\n  fold {fold} best (selected on validation score, never on test):")
    for k in ("dr", "dme_ungated", "dme_gated", "dme_referable_binary"):
        if k in final:
            print(fmt(k, final[k]), flush=True)
    return {"fold": fold, "n_train": len(tr), "n_val": len(va),
            "history": history, "metrics": final, "best_score": best}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["/kaggle/input"])
    p.add_argument("--splits", default="data/splits/dev_v1.json")
    p.add_argument("--run-id", default="E05")
    p.add_argument("--out", default=None)
    p.add_argument("--cache", default="/kaggle/temp/cache560")
    p.add_argument("--cache-size", type=int, default=560,
                   help="long side of the cached image; must be >= --size or the run "
                        "upsamples and the resolution label is a lie (ISSUES.md §18)")
    p.add_argument("--folds", default="0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--size", type=int, default=448)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--backbone", default="densenet121")
    p.add_argument("--head", default="ordinal", choices=["ordinal", "softmax"])
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--min-lr-frac", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--smoothing", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--pretrain-corpora", default="",
                   help="e.g. EyePACS -- trained on once, before the folds, then reused")
    p.add_argument("--pretrain-epochs", type=int, default=6)
    p.add_argument("--lr-pretrain", type=float, default=3e-4)
    p.add_argument("--lpft", action="store_true",
                   help="I21: linear-probe-then-fine-tune. Heads first on a frozen "
                        "backbone, then unfreeze at --lpft-lr-backbone with warmup.")
    p.add_argument("--lpft-epochs", type=int, default=5,
                   help="probe epochs with the backbone frozen")
    p.add_argument("--lpft-lr-backbone", type=float, default=1e-5,
                   help="backbone LR after unfreezing (the point is that it is far below "
                        "--lr-backbone)")
    p.add_argument("--lpft-warmup-epochs", type=int, default=2,
                   help="warmup epochs for the backbone after unfreezing")
    p.add_argument("--macula", action="store_true",
                   help="I07: give the DME head a macula-centred crop instead of the whole "
                        "fundus. The backbone is shared, so the only change is what the DME "
                        "head looks at.")
    p.add_argument("--macula-size", type=int, default=224,
                   help="pixel size of the macula crop fed to the DME head")
    p.add_argument("--macula-dd", type=float, default=3.0,
                   help="side of the macula window in disc diameters (the DME grade is "
                        "defined within 1 DD of the macula centre, so 3 DD gives context)")
    p.add_argument("--fovea-epochs", type=int, default=25,
                   help="epochs for the fovea localiser trained inside a --macula run")
    p.add_argument("--tta", action="store_true",
                   help="average logits over the four dihedral flips at final evaluation")
    p.add_argument("--messidor-source", default="prefer-native",
                   choices=["prefer-native", "native-only", "default-only"],
                   help="which files back the Messidor-2 labels; the two -only modes are "
                        "the source control of IDEAS.md I19")
    p.add_argument("--corpora", default="IDRiD,Messidor-2",
                   help="comma-separated; used for fast local smoke tests")
    p.add_argument("--limit", type=int, default=0, help="debug: keep only N images")
    p.add_argument("--hypothesis", default="")
    args = p.parse_args()

    os.environ["MESSIDOR_SOURCE"] = args.messidor_source
    import importlib
    importlib.reload(corpora)

    sha = git_sha()
    print(f"COMMIT {sha}")                 # first line, always
    print(f"RUN {args.run_id}")
    if args.hypothesis:
        print(f"HYPOTHESIS {args.hypothesis}")
    t_start = time.time()
    seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else \
             ("mps" if torch.backends.mps.is_available() else "cpu")
    amp_dtype = torch.float16 if device == "cuda" else None
    print(f"device={device}  torch={torch.__version__}  "
          f"gpu={torch.cuda.get_device_name(0) if device=='cuda' else '-'}", flush=True)
    if device == "cuda":
        # A GPU that torch can see is not necessarily one it can run on: Kaggle's P100 is
        # sm_60 and the preinstalled cu128 build ships sm_70+ kernels only. Without this
        # check the run dies 2 minutes in, after rebuilding the whole image cache
        # (ISSUES.md §9).
        cap = torch.cuda.get_device_capability(0)
        try:
            (torch.zeros(8, 8, device="cuda") @ torch.zeros(8, 8, device="cuda")).sum().item()
            print(f"[gpu] sm_{cap[0]}{cap[1]} usable", flush=True)
        except Exception as e:
            raise SystemExit(
                f"GPU {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]}) cannot execute "
                f"kernels from this torch build ({torch.__version__}): {e}\n"
                f"Pin a supported accelerator in kernel-metadata.json "
                f'("accelerator": "nvidiaTeslaT4") and relaunch.')

    out_dir = args.out or os.path.join("runs", args.run_id)
    os.makedirs(out_dir, exist_ok=True)

    rows = corpora.build(args.datasets, tuple(args.corpora.split(",")))
    nat = sum(1 for r in rows if r.get("native_res"))
    if nat:
        print(f"[corpora] {nat}/{sum(1 for r in rows if r['corpus']=='Messidor-2')} "
              f"Messidor-2 images come from the native-resolution mirror; the rest are "
              f"512 px. Report resolution results per corpus, not pooled.", flush=True)
    print(corpora.summarise(rows), flush=True)

    with open(args.splits) as f:
        split_meta = json.load(f)
    folds = split_meta["folds"]
    missing = [r["uid"] for r in rows if r["uid"] not in folds]
    if missing:
        raise SystemExit(f"{len(missing)} images are not in the frozen split "
                         f"({args.splits}); e.g. {missing[:3]}. Rebuild the split "
                         f"deliberately -- do not generate one at training time.")
    for r in rows:
        r["fold"] = folds[r["uid"]]
        r["group"] = split_meta["groups"][r["uid"]]
    if args.limit:
        rows = rows[:args.limit]
        print(f"[debug] --limit {args.limit}: keeping {len(rows)} images")
    print(f"[split] {args.splits} fingerprint={split_meta['fingerprint']} "
          f"{split_meta['n_folds']} folds", flush=True)

    pre_rows = []
    if args.pretrain_corpora:
        pre_rows = corpora.build(args.datasets, tuple(args.pretrain_corpora.split(",")))
        if not pre_rows:
            # A silent zero here would turn this run into an unlabelled duplicate of the
            # no-pretraining baseline, and the comparison would then "show" that
            # pretraining does not help -- when it simply never happened.
            raise SystemExit(
                f"--pretrain-corpora {args.pretrain_corpora} matched 0 images under "
                f"{args.datasets}. Is the corpus attached to this kernel? Refusing to "
                f"run, because a silently skipped pretraining phase produces a result "
                f"that looks like evidence and is not.")
        overlap = {r["uid"] for r in pre_rows} & {r["uid"] for r in rows}
        if overlap:
            raise SystemExit(f"{len(overlap)} pretraining images are also in the "
                             f"development pool -- that is leakage, not pretraining")
        for r in pre_rows:
            r.setdefault("group", r["uid"])
        print(corpora.summarise(pre_rows), flush=True)

    build_cache(rows + pre_rows, args.cache, args.cache_size, args.size)

    # ── I07: fovea coordinates for every row, so the DME head can be given a
    # macula-centred crop. Trained here rather than fetched, so the run is reproducible
    # from one commit. Cached to the run directory: a killed Kaggle session does not
    # repeat it, and the coordinates the run actually used are archived beside its results.
    args.fovea_map = None
    if args.macula:
        import fovea as fovea_mod
        fj = os.path.join(out_dir, "fovea_coords.json")
        if os.path.exists(fj):
            args.fovea_map = json.load(open(fj))
            print(f"[fovea] reusing {fj} ({len(args.fovea_map)} rows)", flush=True)
        else:
            args.fovea_map = fovea_mod.fit_predict(
                rows, size=224, epochs=args.fovea_epochs, workers=args.workers,
                device=device)
            json.dump(args.fovea_map, open(fj, "w"))
            print(f"[fovea] wrote {fj} ({len(args.fovea_map)} rows)", flush=True)
        n_gt = sum(1 for r in rows if r.get("fovea"))
        print(f"[fovea] {n_gt} of {len(rows)} rows have GROUND-TRUTH coordinates "
              f"(IDRiD); the remaining {len(rows)-n_gt} use PREDICTED coordinates, "
              f"which is unvalidated transfer -- report the IDRiD-only result too.",
              flush=True)

    results = {
        "run_id": args.run_id, "commit": sha, "hypothesis": args.hypothesis,
        # fovea_map holds 2 260 coordinate pairs; it is archived as fovea_coords.json
        # beside the results, not inlined into every config block
        "config": {k: v for k, v in vars(args).items() if k != "fovea_map"},
        "split_fingerprint": split_meta["fingerprint"],
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "platform": platform.platform(),
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None},
        "folds": [],
    }
    init_state = pretrain(pre_rows, args, device, out_dir, amp_dtype) if pre_rows else None
    results["pretrain"] = {"corpora": args.pretrain_corpora, "n_images": len(pre_rows),
                           "epochs": args.pretrain_epochs if pre_rows else 0}

    for fold in [int(x) for x in args.folds.split(",") if x != ""]:
        results["folds"].append(run_fold(rows, fold, args, device, out_dir, amp_dtype,
                                         init_state))
        results["runtime_sec"] = round(time.time() - t_start, 1)
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=1, default=str)

    # pool every out-of-fold prediction written so far and score the whole corpus at once
    pooled = pool_oof(rows, out_dir, args.head)
    if pooled:
        results["pooled_oof"] = pooled
        print(f"\n{'='*78}\nPOOLED OUT-OF-FOLD ({pooled['n_images']} images, "
              f"folds {pooled['folds']})\n{'='*78}")
        for k in ("dr", "dme_ungated", "dme_gated", "dme_referable_binary"):
            if k in pooled["metrics"]:
                print(fmt(k, pooled["metrics"][k]), flush=True)

    results["runtime_sec"] = round(time.time() - t_start, 1)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\nwrote {out_dir}/results.json  ({results['runtime_sec']:.0f}s)")


def pool_oof(rows, out_dir, head):
    """Every image predicted exactly once, across all folds present on disk."""
    by_uid = {r["uid"]: r for r in rows}
    uids, dr_l, dme_l, got = [], [], [], []
    for fold in range(10):
        p = os.path.join(out_dir, f"oof_{fold}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        uids += list(z["uids"]); dr_l.append(z["dr_logits"]); dme_l.append(z["dme_logits"])
        got.append(fold)
    if not uids:
        return None
    sub = [by_uid[u] for u in uids]
    return {"folds": got, "n_images": len(sub),
            "metrics": evaluate(sub, torch.from_numpy(np.concatenate(dr_l)),
                                torch.from_numpy(np.concatenate(dme_l)), head,
                                n_boot=2000)}


if __name__ == "__main__":
    main()
