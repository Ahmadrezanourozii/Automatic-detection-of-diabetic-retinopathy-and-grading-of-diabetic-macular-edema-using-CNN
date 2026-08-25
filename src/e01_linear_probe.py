"""
E01 — does the representation carry DR/DME signal at all?

HYPOTHESIS
    The collapse in every archived run (ISSUES.md §1) was an optimisation failure, not a
    data failure. A logistic regression on frozen ImageNet DenseNet121 features should beat
    the majority-class floor on both heads by more than its bootstrap interval.

FALSIFIED IF
    The probe fails to beat the floor. That would mean the signal is not in the
    representation at this resolution, and the problem is preprocessing or resolution rather
    than the training loop — which changes the whole plan.

DECLARED FACTORIAL
    Three preprocessing variants are run, not one, because feature extraction shares the
    expensive image decode and the second factor answers E02 at no extra cost:
        rgb                — plain RGB, ImageNet normalisation (the trivial baseline)
        green_clahe        — the thesis chain, ImageNet normalisation
        green_clahe_raw01  — the thesis chain exactly as the old code fed it ([0,1], no
                             ImageNet normalisation)
    Reported as three rows, attributable individually. See src/preprocess.py.

PROTOCOL COMPLIANCE
    - Scaler and classifier C are fitted on training folds only; C is chosen by inner CV on
      the training data, never on test (PROTOCOL.md §3).
    - Bootstrap intervals resample groups, not rows (PROTOCOL.md §4).
    - DME is reported ungated as primary and gated DR>=1 as secondary (PROTOCOL.md §5.1).
    - Every number is quoted against its majority-class floor.

Usage:
    python src/e01_linear_probe.py --idrid <path> [--size 224] [--out runs/E01]
"""
from __future__ import annotations
import argparse, json, os, platform, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
from data_idrid import load_manifest, DR_CLASSES, DME_CLASSES
from preprocess import VARIANTS, VARIANT_DOC, apply_variant, load_cropped

EXPERIMENT_ID = "E01"
SEED = 0


def git_sha() -> str:
    """The first thing this script prints, so any log traces back to its code."""
    try:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sha = subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "-C", here, "status", "--porcelain"],
                                        stderr=subprocess.DEVNULL).decode().strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "UNKNOWN-not-a-git-repo"


# ── image cache ───────────────────────────────────────────────────────────────
def build_cache(rows, cache_dir, cache_size=640):
    """Decode each source image once, border-crop it, and store it locally.

    The source images are 4288x2848 on a network mount; decoding them per variant would
    dominate the runtime and make every rerun expensive.
    """
    os.makedirs(cache_dir, exist_ok=True)
    assert len({r["uid"] for r in rows}) == len(rows), "uids are not unique"
    todo = [r for r in rows if not os.path.exists(os.path.join(cache_dir, r["uid"] + ".png"))]
    if not todo:
        print(f"[cache] {len(rows)} images already cached in {cache_dir}")
        return
    print(f"[cache] decoding {len(todo)} images (once) -> {cache_dir}")
    import cv2
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        img = load_cropped(r["path"], cache_size)
        cv2.imwrite(os.path.join(cache_dir, r["uid"] + ".png"), img)
        if i % 50 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"[cache]   {i}/{len(todo)}  {el:.0f}s elapsed, "
                  f"~{el / i * (len(todo) - i):.0f}s left", flush=True)


# ── features ──────────────────────────────────────────────────────────────────
def extract_features(rows, cache_dir, variant, size, device, batch=32):
    import cv2, torch
    from torchvision.models import densenet121, DenseNet121_Weights

    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()   # -> 1024-d after the built-in GAP
    model.eval().to(device)

    feats = np.zeros((len(rows), 1024), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            chunk = rows[s:s + batch]
            xs = np.stack([
                apply_variant(cv2.imread(os.path.join(cache_dir, r["uid"] + ".png"),
                                         cv2.IMREAD_COLOR), variant, size)
                for r in chunk])
            out = model(torch.from_numpy(xs).to(device))
            feats[s:s + len(chunk)] = out.float().cpu().numpy()
    print(f"[feat] {variant:18s} {len(rows)} images in {time.time() - t0:.0f}s")
    return feats


# ── probe ─────────────────────────────────────────────────────────────────────
def fit_predict(Xtr, ytr, Xte, seed=SEED):
    """Standardise + multinomial logistic regression, C chosen by inner CV on TRAIN only.

    Every quantity that touches the test rows (scaler statistics, C, class weights) is
    estimated on the training rows and then applied unchanged (PROTOCOL.md §3, 'fit nothing
    on data you will evaluate on').
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_min = int(np.bincount(ytr).min())
    inner = StratifiedKFold(n_splits=min(3, max(2, n_min)), shuffle=True, random_state=seed)
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
    )
    gs = GridSearchCV(pipe, {"logisticregression__C": [0.001, 0.01, 0.1, 1.0]},
                      cv=inner, scoring="balanced_accuracy", n_jobs=-1)
    gs.fit(Xtr, ytr)
    return gs.predict(Xte), gs.best_params_["logisticregression__C"]


def eval_head(rows, feats, key, k, tr_idx, te_idx, label, referable_from):
    y = np.array([r[key] for r in rows])
    pred, C = fit_predict(feats[tr_idx], y[tr_idx], feats[te_idx])
    groups = np.array([r["group"] for r in rows])[te_idx]
    rep = M.report(y[te_idx], pred, k, groups=groups, seed=SEED,
                   referable_from=referable_from)
    rep["chosen_C"] = C
    rep["label"] = label
    return rep, pred


def print_block(title, rep, classes):
    lo, hi = rep["accuracy_ci95"]
    qlo, qhi = rep["qwk_ci95"]
    verdict = "BEATS floor" if rep["beats_floor"] else "does NOT beat floor"
    print(f"\n  {title}  (n={rep['n']})")
    print(f"    accuracy {rep['accuracy']*100:5.1f}%  [{lo*100:.1f}, {hi*100:.1f}]"
          f"   floor {rep['majority_floor']*100:5.1f}%   -> {verdict}")
    print(f"    QWK      {rep['qwk']:5.3f}   [{qlo:.3f}, {qhi:.3f}]"
          f"        macro-F1 {rep['macro_f1']:.3f}")
    rec = "  ".join(f"{c}={'--' if r is None else f'{r*100:.0f}%'}"
                    for c, r in zip(classes, rep["per_class_recall"]))
    print(f"    recall   {rec}")
    if "referable_sensitivity" in rep:
        print(f"    referable  sens {rep['referable_sensitivity']*100:.1f}%   "
              f"spec {rep['referable_specificity']*100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idrid", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", default="runs/E01")
    ap.add_argument("--cache", default="data/cache/idrid_640")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()

    sha = git_sha()
    print(f"COMMIT {sha}")            # first line, always
    print(f"EXPERIMENT {EXPERIMENT_ID} — frozen-feature linear probe")
    t_start = time.time()

    import torch
    device = "mps" if torch.backends.mps.is_available() else \
             ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  size={args.size}  seed={SEED}")

    rows = load_manifest(args.idrid, cache=os.path.join(args.out, "manifest_idrid.json"))
    print(f"IDRiD manifest: {len(rows)} images")
    build_cache(rows, args.cache)

    idx = np.arange(len(rows))
    tr_official = idx[[r["official_split"] == "train" for r in rows]]
    te_official = idx[[r["official_split"] == "test" for r in rows]]
    dr = np.array([r["dr"] for r in rows])
    gated = np.flatnonzero(dr >= 1)

    results = {
        "experiment": EXPERIMENT_ID,
        "commit": sha,
        "hypothesis": "Frozen ImageNet features beat the majority-class floor on both "
                      "heads; the archived collapse was an optimisation failure.",
        "config": {"size": args.size, "seed": SEED, "backbone": "densenet121/IMAGENET1K_V1",
                   "probe": "StandardScaler + multinomial LogisticRegression "
                            "(class_weight=balanced), C by inner 3-fold CV on train",
                   "split": "IDRiD official 413/103", "device": device},
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "platform": platform.platform()},
        "variant_docs": VARIANT_DOC,
        "results": {},
    }

    for variant in args.variants.split(","):
        print(f"\n{'='*72}\nVARIANT  {variant}\n{'='*72}")
        fcache = os.path.join(args.out, f"features_{variant}_{args.size}.npy")
        if os.path.exists(fcache):
            feats = np.load(fcache)
            print(f"[feat] loaded cached features {feats.shape}")
        else:
            feats = extract_features(rows, args.cache, variant, args.size, device)
            os.makedirs(args.out, exist_ok=True)
            np.save(fcache, feats)

        v = {}
        rep, _ = eval_head(rows, feats, "dr", 5, tr_official, te_official,
                           "DR 5-class, IDRiD official test", referable_from=2)
        print_block("DR — 5-class, official test", rep, DR_CLASSES)
        v["dr_official"] = rep

        rep, _ = eval_head(rows, feats, "dme", 3, tr_official, te_official,
                           "DME 3-class UNGATED (primary), IDRiD official test",
                           referable_from=2)
        print_block("DME — 3-class, ungated  [PRIMARY]", rep, DME_CLASSES)
        v["dme_official_ungated"] = rep

        tr_g = np.array([i for i in tr_official if dr[i] >= 1])
        te_g = np.array([i for i in te_official if dr[i] >= 1])
        rep, _ = eval_head(rows, feats, "dme", 3, tr_g, te_g,
                           "DME 3-class GATED DR>=1 (secondary), IDRiD official test",
                           referable_from=2)
        print_block("DME — 3-class, gated DR>=1  [secondary]", rep, DME_CLASSES)
        v["dme_official_gated"] = rep

        results["results"][variant] = v

    results["runtime_sec"] = round(time.time() - t_start, 1)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {os.path.join(args.out, 'results.json')}  "
          f"({results['runtime_sec']:.0f}s total)")


if __name__ == "__main__":
    main()
