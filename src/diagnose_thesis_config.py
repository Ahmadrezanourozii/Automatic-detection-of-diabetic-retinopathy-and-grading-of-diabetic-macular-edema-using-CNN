"""
diagnose_thesis_config.py — can the ORIGINAL thesis training config memorise 20 images?

This settles the oldest open question in the project (`ISSUES.md` §1): the archived runs
collapsed to 27.2 % DR / 15.9 % DME, both below the majority-class floor, while E01's frozen
linear probe reached 47.6 % on the same backbone's features. Either the fine-tuning recipe
destroyed a representation that was already good enough, or something in the training loop
was miswired. A capacity or data explanation cannot survive an overfit-a-tiny-batch test:
**20 images, regularisation and augmentation off, is a memorisation task.** Any correctly
wired network drives training accuracy to 100 %. One that cannot has a defect in its
optimisation or its loss wiring, and the size of the training set is irrelevant to it.

This runs the ORIGINAL Keras code path, not a PyTorch restatement of it, because two of the
three suspects (`ISSUES.md` §1) are Keras-specific: sample weights supplied as a flat array
to a multi-output model, and BatchNormalization frozen by `layer.trainable = False`. A
reimplementation would quietly not have those bugs and would answer a different question.

CPU only. No GPU quota is spent deciding whether LP-FT is the right lever.

Variants, each starting from the same ImageNet weights and the same 20 images:

  A  head_only     backbone frozen, Adam(1e-4) on the heads.  This is the original's
                   phase 1, and it is the control: E01 says these features are separable,
                   so failure HERE means the loss/label wiring is broken, not fine-tuning.
  B  two_phase     A, then `unfreeze_backbone()` at 1e-5 with clipnorm=1.0 and BN frozen.
                   The original recipe in full. If A memorises and B does not, the
                   fine-tuning phase is what destroys the representation, which is exactly
                   the failure LP-FT is designed to prevent.
  C  two_phase_cw  B plus the archived class-weight path: weights computed from the DR
                   label distribution and passed as a FLAT array to a two-output model,
                   which Keras broadcasts to both heads. Isolates that suspect on its own.

Usage:
    python src/diagnose_thesis_config.py --idrid <path to IDRiD root> --out runs/D01

Needs tensorflow, opencv-python-headless, pandas, scikit-learn. These are NOT project
dependencies -- the project is PyTorch. Use a throwaway venv on Python 3.12.
"""
from __future__ import annotations
import argparse, json, os, sys, time

import numpy as np
import pandas as pd

# ── the original config, transcribed verbatim from config.py on the Drive ────────────
REAL_IMG_SIZE = (512, 512)
DR_NUM_CLASSES = 5
DME_NUM_CLASSES = 3
BATCH_SIZE = 16          # "Q2 report explicitly states batch size = 16"
LEARNING_RATE = 1e-4     # head / frozen-backbone phase
FINETUNE_LR = 1e-5       # unfreeze phase
ALPHA, BETA = 0.6, 0.4   # DR / DME loss weights
CLAHE_CLIP, CLAHE_TILE = 2.0, (8, 8)
GAUSSIAN_KERN = (5, 5)


def preprocess_file(path, img_size=REAL_IMG_SIZE):
    """The original 'green' pipeline: resize -> green -> CLAHE -> Gaussian -> /255 -> x3."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)
    green = img[:, :, 1].copy()
    green = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE).apply(green)
    green = cv2.GaussianBlur(green, GAUSSIAN_KERN, 0)
    out = green.astype(np.float32) / 255.0
    return np.stack([out, out, out], axis=-1)


def build_model(dropout, freeze_backbone=True, lr=LEARNING_RATE):
    """model.py's build_multi_output_model, with dropout exposed so it can be turned off."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import DenseNet121

    inputs = layers.Input(shape=(*REAL_IMG_SIZE, 3), name="input_image")
    backbone = DenseNet121(include_top=False, weights="imagenet", input_tensor=inputs)
    backbone.trainable = not freeze_backbone

    x = layers.GlobalAveragePooling2D(name="shared_gap")(backbone.output)

    dr = layers.Dense(256, activation="relu", name="dr_dense")(x)
    dr = layers.BatchNormalization(name="dr_bn")(dr)
    dr = layers.Dropout(dropout, name="dr_dropout")(dr)
    dr_out = layers.Dense(DR_NUM_CLASSES, activation="softmax", name="dr_output")(dr)

    dme = layers.Dense(256, activation="relu", name="dme_dense")(x)
    dme = layers.BatchNormalization(name="dme_bn")(dme)
    dme = layers.Dropout(dropout, name="dme_dropout")(dme)
    dme_out = layers.Dense(DME_NUM_CLASSES, activation="softmax", name="dme_output")(dme)

    model = Model(inputs=inputs, outputs=[dr_out, dme_out], name="MultiOutput_DenseNet121")
    _compile(model, lr)
    return model


def _compile(model, lr, clipnorm=None):
    import tensorflow as tf
    opt = (tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm) if clipnorm
           else tf.keras.optimizers.Adam(learning_rate=lr))
    model.compile(
        optimizer=opt,
        loss={"dr_output": tf.keras.losses.SparseCategoricalCrossentropy(),
              "dme_output": tf.keras.losses.SparseCategoricalCrossentropy()},
        loss_weights={"dr_output": ALPHA, "dme_output": BETA},
        metrics={"dr_output": ["accuracy"], "dme_output": ["accuracy"]},
    )


def unfreeze_backbone(model, lr=FINETUNE_LR):
    """model.py's unfreeze_backbone, verbatim: every non-BN layer trainable, BN frozen."""
    import tensorflow as tf
    bn = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            bn += 1
        else:
            layer.trainable = True
    _compile(model, lr, clipnorm=1.0)
    return model, bn


def load_20(idrid_root, n=20, seed=0, natural=False):
    """20 IDRiD training images, spread across DR grades so the task is not degenerate."""
    grading = None
    for dirpath, dirnames, _ in os.walk(idrid_root):
        # the mirror uses URL-escaped directory names ("B.%20Disease%20Grading")
        for d in dirnames:
            if "Disease" in d and "Grading" in d:
                grading = os.path.join(dirpath, d)
                break
        if grading:
            break
    if grading is None:
        raise SystemExit(f"could not find the Disease Grading directory under {idrid_root}")

    csv_path = img_dir = None
    for dirpath, _, filenames in os.walk(grading):
        for fn in filenames:
            if fn.lower().endswith(".csv") and "train" in fn.lower():
                csv_path = os.path.join(dirpath, fn)
        if any(f.lower().endswith((".jpg", ".jpeg")) for f in filenames) and "rain" in dirpath:
            img_dir = dirpath
    if not csv_path or not img_dir:
        raise SystemExit(f"csv={csv_path} img_dir={img_dir} -- one is missing under {grading}")

    df = pd.read_csv(csv_path)
    rng = np.random.RandomState(seed)
    picked = []
    if natural:
        # Draw with IDRiD's own class distribution. A balanced draw makes
        # compute_class_weight('balanced') return exactly 1.0 for every class, so the
        # class-weight code path becomes a no-op and cannot be tested at all.
        idx = df.index.tolist()
        rng.shuffle(idx)
        picked = idx[:n]
    # take round-robin across DR grades so all five classes appear if possible
    by_grade = {g: df[df.iloc[:, 1] == g].index.tolist() for g in range(DR_NUM_CLASSES)}
    for g in by_grade:
        rng.shuffle(by_grade[g])
    while len(picked) < n and not natural:
        progressed = False
        for g in range(DR_NUM_CLASSES):
            if by_grade[g] and len(picked) < n:
                picked.append(by_grade[g].pop())
                progressed = True
        if not progressed:
            break

    X, y_dr, y_dme, names = [], [], [], []
    for i in picked:
        row = df.loc[i]
        name = str(row.iloc[0]).strip()
        path = os.path.join(img_dir, name + ".jpg")
        if not os.path.exists(path):
            path = os.path.join(img_dir, name + ".JPG")
        if not os.path.exists(path):
            continue
        X.append(preprocess_file(path))
        y_dr.append(int(row.iloc[1]))
        y_dme.append(int(row.iloc[2]))
        names.append(name)
    return (np.array(X, dtype=np.float32), np.array(y_dr, dtype=np.int32),
            np.array(y_dme, dtype=np.int32), names)


def memorisation(model, X, y_dr, y_dme):
    """Accuracy on the very images that were trained on. 100 % is the only passing score."""
    p_dr, p_dme = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    return (float((p_dr.argmax(1) == y_dr).mean()),
            float((p_dme.argmax(1) == y_dme).mean()),
            p_dr.argmax(1).tolist(), p_dme.argmax(1).tolist())


def _fit_input(X, y_dr, y_dme, sw):
    """train.py's _make_tf_dataset: a flat weight array as the third tuple element, beside
    a DICT of two outputs. This is the archived path, and it is the one under suspicion --
    Keras has to decide what a single weight vector means for two heads."""
    import tensorflow as tf
    labels = {"dr_output": y_dr, "dme_output": y_dme}
    if sw is not None:
        ds = tf.data.Dataset.from_tensor_slices((X, labels, sw))
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, labels))
    return ds.shuffle(len(X), reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE)


def run_variant(name, X, y_dr, y_dme, epochs_head, epochs_ft, dropout, class_weights):
    import tensorflow as tf
    print(f"\n{'='*74}\nVARIANT {name}\n{'='*74}", flush=True)
    tf.keras.utils.set_random_seed(0)

    sw = None
    if class_weights:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.arange(DR_NUM_CLASSES)
        present = np.unique(y_dr)
        w = compute_class_weight(class_weight="balanced", classes=present, y=y_dr)
        cw = {int(c): float(x) for c, x in zip(present, w)}
        for c in classes:
            cw.setdefault(int(c), 1.0)
        # THE ARCHIVED BUG, reproduced exactly: a flat per-sample array handed to a
        # two-output model. Keras broadcasts it to BOTH heads, so DR class imbalance
        # silently reweights the DME loss too.
        sw = np.array([cw[int(l)] for l in y_dr], dtype=np.float32)
        print(f"  DR class weights: {cw}")
        print(f"  sample_weight shape {sw.shape} -> broadcast across both outputs")

    model = build_model(dropout=dropout, freeze_backbone=True, lr=LEARNING_RATE)
    hist = {"head": [], "finetune": []}

    def record(phase, h):
        for i in range(len(h.history["loss"])):
            hist[phase].append({k: float(v[i]) for k, v in h.history.items()})

    t0 = time.time()
    print(f"  phase 1: backbone FROZEN, Adam(lr={LEARNING_RATE}), {epochs_head} epochs",
          flush=True)
    h = model.fit(_fit_input(X, y_dr, y_dme, sw), epochs=epochs_head, verbose=0)
    record("head", h)
    a_dr, a_dme, _, _ = memorisation(model, X, y_dr, y_dme)
    print(f"  after phase 1: DR {a_dr*100:.0f}%  DME {a_dme*100:.0f}%  "
          f"loss {h.history['loss'][-1]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    phase1 = {"dr_acc": a_dr, "dme_acc": a_dme, "loss": float(h.history["loss"][-1])}

    phase2 = None
    if epochs_ft:
        model, bn = unfreeze_backbone(model, FINETUNE_LR)
        print(f"  phase 2: backbone UNFROZEN, Adam(lr={FINETUNE_LR}, clipnorm=1.0), "
              f"{bn} BN layers frozen, {epochs_ft} epochs", flush=True)
        h2 = model.fit(_fit_input(X, y_dr, y_dme, sw), epochs=epochs_ft, verbose=0)
        record("finetune", h2)
        b_dr, b_dme, _, _ = memorisation(model, X, y_dr, y_dme)
        print(f"  after phase 2: DR {b_dr*100:.0f}%  DME {b_dme*100:.0f}%  "
              f"loss {h2.history['loss'][-1]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        phase2 = {"dr_acc": b_dr, "dme_acc": b_dme, "loss": float(h2.history["loss"][-1])}

    final = phase2 or phase1
    passed = final["dr_acc"] >= 0.99 and final["dme_acc"] >= 0.99
    print(f"  MEMORISED: {'YES' if passed else 'NO'}")
    return {"variant": name, "dropout": dropout, "class_weights": class_weights,
            "epochs_head": epochs_head, "epochs_finetune": epochs_ft,
            "phase1": phase1, "phase2": phase2, "memorised": passed,
            "history": hist, "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idrid", required=True)
    ap.add_argument("--out", default="runs/D01")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--natural", action="store_true",
                    help="sample with IDRiD's own class distribution instead of "
                         "one image per grade; required for the class-weight variant "
                         "to be anything but a no-op")
    ap.add_argument("--epochs-head", type=int, default=40)
    ap.add_argument("--epochs-finetune", type=int, default=40)
    ap.add_argument("--variants", default="head_only,two_phase,two_phase_cw")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")     # CPU only, deliberately
    print(f"tensorflow {tf.__version__}  devices={[d.device_type for d in tf.config.get_visible_devices()]}")

    X, y_dr, y_dme, names = load_20(a.idrid, a.n, natural=a.natural)
    print(f"loaded {len(X)} IDRiD images at {X.shape[1:]}  "
          f"DR grades {np.bincount(y_dr, minlength=5).tolist()}  "
          f"DME grades {np.bincount(y_dme, minlength=3).tolist()}")
    print("ACCEPTANCE, fixed before the numbers: a correctly wired network memorises 20 "
          "images with regularisation and augmentation off. Pass = 100% on BOTH heads.")

    specs = {
        # regularisation off => dropout 0.0, no augmentation anywhere, no early stopping
        "head_only":    dict(epochs_head=a.epochs_head, epochs_ft=0,
                             dropout=0.0, class_weights=False),
        "two_phase":    dict(epochs_head=a.epochs_head, epochs_ft=a.epochs_finetune,
                             dropout=0.0, class_weights=False),
        "two_phase_cw": dict(epochs_head=a.epochs_head, epochs_ft=a.epochs_finetune,
                             dropout=0.0, class_weights=True),
    }
    results = []
    for name in a.variants.split(","):
        name = name.strip()
        if name not in specs:
            raise SystemExit(f"unknown variant {name!r}; choose from {list(specs)}")
        results.append(run_variant(name, X, y_dr, y_dme, **specs[name]))

    out = {
        "diagnostic": "overfit-a-tiny-batch on the ORIGINAL thesis config",
        "question": "ISSUES.md §1 -- was the 27.2%/15.9% collapse a training-loop defect?",
        "acceptance": "a correctly wired network memorises 20 images with regularisation "
                      "and augmentation off; pass = 100% train accuracy on BOTH heads",
        "n_images": int(len(X)), "image_names": names,
        "dr_grades": np.bincount(y_dr, minlength=5).tolist(),
        "dme_grades": np.bincount(y_dme, minlength=3).tolist(),
        "config_source": "config.py / model.py / train.py as archived on Google Drive",
        "tensorflow": __import__("tensorflow").__version__,
        "device": "CPU",
        "variants": results,
    }
    with open(os.path.join(a.out, "tinybatch.json"), "w") as f:
        json.dump(out, f, indent=1)

    print(f"\n{'='*74}\nSUMMARY\n{'='*74}")
    for r in results:
        f_ = r["phase2"] or r["phase1"]
        print(f"  {r['variant']:15s} DR {f_['dr_acc']*100:5.1f}%  DME {f_['dme_acc']*100:5.1f}%"
              f"   memorised={'YES' if r['memorised'] else 'NO'}")
    print(f"\nwrote {a.out}/tinybatch.json")


if __name__ == "__main__":
    main()
