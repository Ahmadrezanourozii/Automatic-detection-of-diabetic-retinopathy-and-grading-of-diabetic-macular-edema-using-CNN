"""
eval_external.py — score a finished run on a corpus it has never seen.

This is the number that survives a defence. Everything else in this project is measured on
images from the same two corpora the model trained on; APTOS is a different population,
different cameras, different graders, and it has been held out entirely since the protocol
was frozen (PROTOCOL.md §2/§6).

Two honest caveats it must always be reported with:
  * APTOS labels are single-grader, and noisier than IDRiD's or Messidor-2's adjudicated
    ones. Part of any internal-to-external drop is label noise, not generalisation failure.
  * APTOS has DR grades only. There is no external 3-class DME number until Messidor-1 is
    acquired, and its absence is a limitation to state, not to paper over.

The five folds are ensembled by averaging logits, which is what the folds are for once
selection is done.

Usage:
    python src/eval_external.py --run runs/E06 --datasets /kaggle/input --corpus APTOS
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpora
import metrics as M
from model import MultiOutputNet, decode, N_DR, N_DME
from train import FundusDataset, build_cache, predict, fmt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--datasets", nargs="+", default=["/kaggle/input"])
    ap.add_argument("--corpus", default="APTOS")
    ap.add_argument("--cache", default="/kaggle/temp/cache_ext")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tta", action="store_true")
    a = ap.parse_args()

    # Config comes from the checkpoints themselves. Locating a results.json by path
    # matching found an unrelated one inside an input dataset and crashed on a missing
    # run_id (ISSUES.md §16). Every best_<fold>.pt already carries the exact config it was
    # trained under, so the weights and their configuration cannot drift apart.
    ckpts = sorted(glob.glob(os.path.join(a.run, "best_*.pt")))
    if not ckpts:
        raise SystemExit(f"no best_*.pt in {a.run} — the run kept no weights")
    _first = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    cfg = _first.get("config")
    if not cfg:
        raise SystemExit(f"{ckpts[0]} carries no config; cannot rebuild the model")
    res = {}
    rj = os.path.join(a.run, "results.json")
    if os.path.exists(rj):
        try:
            cand = json.load(open(rj))
            if cand.get("run_id"):
                res = cand
        except Exception:
            pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.float16 if device == "cuda" else None
    print(f"run {res.get('run_id', '?')}  commit {res.get('commit', '?')[:10]}  "
          f"device={device}  backbone={cfg['backbone']} head={cfg['head']} "
          f"size={cfg['size']}")

    rows = corpora.build(a.datasets, (a.corpus,))
    if not rows:
        raise SystemExit(f"{a.corpus} matched 0 images — is it attached to this kernel?")
    for r in rows:
        r.setdefault("group", r["uid"])
    print(corpora.summarise(rows))

    # a corpus used in development is not an external test set, and saying otherwise is the
    # single easiest way to put an indefensible number in a thesis
    dev = corpora.build(a.datasets, tuple(cfg.get("corpora", "IDRiD,Messidor-2").split(",")))
    clash = {r["uid"] for r in rows} & {r["uid"] for r in dev}
    if clash:
        raise SystemExit(f"{len(clash)} images appear in both {a.corpus} and the "
                         f"development pool — this is not external validation")

    build_cache(rows, a.cache)
    ds = FundusDataset(rows, a.cache, cfg["size"], False)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                    pin_memory=True)

    print(f"ensembling {len(ckpts)} folds: {[os.path.basename(c) for c in ckpts]}")

    dr_sum = dme_sum = None
    for c in ckpts:
        st = torch.load(c, map_location=device, weights_only=False)
        model = MultiOutputNet(cfg["backbone"], False, cfg["head"],
                               cfg["hidden"], cfg["dropout"]).to(device)
        model.load_state_dict({k: v.to(device) for k, v in st["state_dict"].items()})
        dr_l, dme_l, order = predict(model, dl, device, cfg["head"], amp, tta=a.tta)
        inv = torch.argsort(order)
        dr_l, dme_l = dr_l[inv], dme_l[inv]
        dr_sum = dr_l if dr_sum is None else dr_sum + dr_l
        dme_sum = dme_l if dme_sum is None else dme_sum + dme_l
    dr_l, dme_l = dr_sum / len(ckpts), dme_sum / len(ckpts)

    y = np.array([r["dr"] for r in rows])
    g = np.array([r["group"] for r in rows])
    rep = M.report(y, decode(dr_l, N_DR, cfg["head"]).numpy(), N_DR,
                   groups=g, n_boot=2000, referable_from=2)
    print(f"\n{'='*78}\nEXTERNAL — {a.corpus}, never seen in training\n{'='*78}")
    print(fmt("dr", rep))
    print(f"{'':26s}recall " +
          "  ".join("--" if x is None else f"{x*100:.0f}%" for x in rep["per_class_recall"]) +
          f"   support {rep['support']}")
    print(f"{'':26s}referable-DR sens {rep['referable_sensitivity']*100:.1f}%  "
          f"spec {rep['referable_specificity']*100:.1f}%")

    internal = (res.get("recomputed") or res.get("pooled_oof") or {}).get("metrics", {})
    if internal.get("dr"):
        d_acc = rep["accuracy"] - internal["dr"]["accuracy"]
        d_qwk = rep["qwk"] - internal["dr"]["qwk"]
        print(f"\n  internal (pooled OOF): acc {internal['dr']['accuracy']*100:.1f}%  "
              f"QWK {internal['dr']['qwk']:.3f}")
        print(f"  external - internal  : acc {d_acc*100:+.1f} pts  QWK {d_qwk:+.3f}")
        print("  Part of any drop is APTOS's single-grader label noise rather than "
              "generalisation failure; report it that way.")

    out = os.path.join(a.run, f"external_{a.corpus.lower()}.json")
    json.dump({"corpus": a.corpus, "n_folds_ensembled": len(ckpts), "tta": a.tta,
               "metrics": {"dr": rep}}, open(out, "w"), indent=1, default=str)

    # Per-image predictions, so the reported INTERVALS can be re-derived and not merely
    # trusted. The first external run archived only the aggregate confusion matrix, which
    # left its bootstrap intervals unverifiable (EXPERIMENTS.md, E08X verification).
    preds = os.path.join(a.run, f"external_{a.corpus.lower()}_predictions.npz")
    np.savez_compressed(
        preds,
        uids=np.array([r["uid"] for r in rows]),
        groups=np.array([r["group"] for r in rows]),
        y_dr=y,
        dr_logits=dr_l.numpy(),
        dme_logits=dme_l.numpy())
    print(f"\nwrote {out}\nwrote {preds}")


if __name__ == "__main__":
    main()
