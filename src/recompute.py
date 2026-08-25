"""
recompute.py — re-score an archived run from its saved out-of-fold logits.

Every run writes oof_<fold>.npz with the uids and the raw logits, so a change to the
*evaluation* definition never requires spending GPU quota again. This is the payoff for
archiving predictions rather than only metrics.

Used when the metric definition is corrected after a run has already happened — see
ISSUES.md §12, where the 3-class DME metric was being computed on a biased 667-image set
instead of the 516-image one it should use.

Usage:
    python src/recompute.py --run runs/E05 --datasets <root> [--head ordinal]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpora
from train import evaluate, fmt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--head", default=None, help="defaults to the run's own config")
    ap.add_argument("--write", action="store_true",
                    help="write the corrected metrics back into results.json")
    a = ap.parse_args()

    rj = os.path.join(a.run, "results.json")
    res = json.load(open(rj)) if os.path.exists(rj) else {}
    head = a.head or res.get("config", {}).get("head", "ordinal")

    rows = corpora.build(a.datasets, ("IDRiD", "Messidor-2"))
    by_uid = {r["uid"]: r for r in rows}
    split = json.load(open("data/splits/dev_v1.json"))
    for r in rows:
        r["group"] = split["groups"].get(r["uid"], r["uid"])

    uids, dr_l, dme_l, folds = [], [], [], []
    for p in sorted(glob.glob(os.path.join(a.run, "oof_*.npz"))):
        z = np.load(p, allow_pickle=True)
        uids += [str(u) for u in z["uids"]]
        dr_l.append(z["dr_logits"]); dme_l.append(z["dme_logits"])
        folds.append(int(os.path.basename(p).split("_")[1].split(".")[0]))
    if not uids:
        raise SystemExit(f"no oof_*.npz in {a.run} — nothing to recompute")

    missing = [u for u in uids if u not in by_uid]
    if missing:
        raise SystemExit(f"{len(missing)} uids in the archive are not in the manifest, "
                         f"e.g. {missing[:3]} — the corpora changed under the archive")

    sub = [by_uid[u] for u in uids]
    m = evaluate(sub, torch.from_numpy(np.concatenate(dr_l)),
                 torch.from_numpy(np.concatenate(dme_l)), head, n_boot=2000)

    print(f"\n{'='*78}\nRECOMPUTED  {a.run}  folds {sorted(folds)}  "
          f"{len(sub)} images  head={head}\n{'='*78}")
    for k in ("dr", "dme_ungated", "dme_gated", "dme_referable_binary"):
        if k in m:
            print(fmt(k, m[k]))
            rec = m[k].get("per_class_recall") or []
            print(f"{'':26s}recall " +
                  "  ".join("--" if x is None else f"{x*100:.0f}%" for x in rec) +
                  f"   support {m[k].get('support')}")

    if a.write:
        res.setdefault("recomputed", {})
        res["recomputed"] = {"folds": sorted(folds), "n_images": len(sub),
                             "head": head, "metrics": m,
                             "note": "re-scored from archived OOF logits after the "
                                     "3-class DME evaluation set was corrected "
                                     "(ISSUES.md §12)"}
        json.dump(res, open(rj, "w"), indent=1, default=str)
        print(f"\nwrote corrected metrics into {rj} under 'recomputed'")


if __name__ == "__main__":
    main()
