"""
paper_reverse.py — the reverse-direction experiment (pre-registered outcomes O4-O6).

The E08 recipe trained on APTOS 2019 (its own 5-fold split, R1APTOS) and evaluated on
Messidor-2 and IDRiD. APTOS labels come from the forward model's APTOS prediction file
(results/predictions/E08/external_aptos.npz), which carries y_dr for every APTOS image.

Writes docs/generated/paper/reverse.json.

Usage:
    python src/paper_reverse.py [--model R1APTOS]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_rules as P


def mild_f1(y, p):
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y != 1) & (p == 1)))
    fn = int(np.sum((y == 1) & (p != 1)))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", default="results/predictions")
    ap.add_argument("--model", default="R1APTOS")
    ap.add_argument("--labels-from", default="E08")
    ap.add_argument("--out", default="docs/generated/paper/reverse.json")
    a = ap.parse_args()

    lab = np.load(os.path.join(a.pred_root, a.labels_from, "external_aptos.npz"), allow_pickle=True)
    ylab = {str(u): int(y) for u, y in zip(lab["uids"], lab["y_dr"])}
    reg = json.load(open(os.path.join(a.pred_root, "registry.json")))[a.model]
    head = reg["head"]
    z = np.load(os.path.join(a.pred_root, a.model, "oof.npz"), allow_pickle=True)
    uids = [str(u) for u in z["uids"]]
    missing = [u for u in uids if u not in ylab]
    if missing:
        raise SystemExit(f"{len(missing)} OOF uids lack APTOS labels, e.g. {missing[:3]}")
    dev = {"uids": np.array(uids), "y": np.array([ylab[u] for u in uids]),
           "folds": np.asarray(z["folds"]).astype(int),
           "dr": np.asarray(z["dr_logits"], dtype=np.float64)}
    dv, _ = P.dev_matched(dev, head)
    out = {"model": a.model, "head": head,
           "dev": {"n": dv["n"], "folds": dv["folds"], "qwk_shipped": dv["shipped"]["qwk"],
                   "qwk_matched": dv["matched"]["qwk"]}}

    for corpus in ("Messidor-2", "IDRiD"):
        p = os.path.join(a.pred_root, a.model, f"external_{corpus.lower()}.npz")
        if not os.path.exists(p):
            continue
        ext = P.load_external(p)
        rm, preds = P.rule_matrix(dev, ext, head)
        for r in ("R1", "R2", "R3", "R4", "R4m"):
            rm[r]["mild_f1"] = mild_f1(ext["y"], preds[r])
        # where do true-Mild images go under the shipped rule?
        m = ext["y"] == 1
        dest = np.bincount(preds["R1"][m], minlength=P.K_DR).tolist()
        out[corpus] = {"n": int(len(ext["y"])), "n_mild": int(m.sum()),
                       "rules": {r: rm[r] for r in ("R1", "R2", "R3", "R4", "R4m")},
                       "cuts": rm["_cuts"], "threshold_free": rm["_threshold_free"],
                       "mild_destination_R1": dest,
                       "mean_true": float(ext["y"].mean()),
                       "mean_pred_R1": float(preds["R1"].mean())}
        print(f"{corpus}: n={len(ext['y'])} " + "  ".join(
            f"{r}: QWK {rm[r]['qwk']:.3f} mild {rm[r]['recall'][1]*100:.1f}"
            for r in ("R1", "R2", "R4", "R4m")) + f"  mild->{dest}")

    m2 = out.get("Messidor-2")
    if m2:
        r1 = m2["rules"]["R1"]["recall"][1]
        r4m = m2["rules"]["R4m"]["recall"][1]
        dest = m2["mild_destination_R1"]
        out["o4"] = bool(r1 < 0.20)
        out["o5"] = bool(r4m >= 3 * r1 and r4m >= 0.40)
        out["o6"] = bool(m2["rules"]["R1"]["mean_pred_minus_true"] < 0 and dest[0] > dest[2])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print({k: out.get(k) for k in ("o4", "o5", "o6")}, "dev", out["dev"])


if __name__ == "__main__":
    main()
