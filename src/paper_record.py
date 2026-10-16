"""
paper_record.py — the matched-calibration record, regenerated from per-image predictions.

PROTOCOL.md §4.1 / §10: the rule "no difference is attributed to representation until it
survives matched calibration" is evidence only if it has been seen to return both answers.
This script recomputes each entry of that record with one code path:

  * per-grade recall differences between two models on an external set at the shipped rule
    (R1) and with both models' cut-points refitted the same way on the external labels (R4,
    QWK objective; R4m, macro-recall objective; 2-fold cross-fitting with the same halves for
    both models, so the comparison is paired);
  * development-pool accuracy at shipped versus matched (cross-fitted) cut-points;
  * the E12 source control, Moderate recall native-minus-default at shipped and matched cuts.

Writes docs/generated/paper/record.json.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_rules as P
import metrics as M
from paper_phase0 import load_model, load_ext


def recall_fn(c):
    return lambda y, p: float((p[y == c] == c).mean()) if (y == c).any() else float("nan")


def paired(y, pa, pb, fn, n_boot=2000):
    d, lo, hi, sig = M.paired_bootstrap_diff(y, pa, pb, fn, groups=None, n_boot=n_boot)
    return {"a": fn(y, pa), "b": fn(y, pb), "diff": d, "lo": lo, "hi": hi, "significant": sig}


def external_pair(ea, eb, ha, hb):
    assert (ea["uids"] == eb["uids"]).all()
    y = ea["y"]
    sa, sb = P.expected_grade(ea["dr"], ha), P.expected_grade(eb["dr"], hb)
    preds = {"R1": (P.shipped(ea["dr"], ha), P.shipped(eb["dr"], hb)),
             "R4": (P.crossfit_external(sa, y, "qwk"), P.crossfit_external(sb, y, "qwk")),
             "R4m": (P.crossfit_external(sa, y, "mr"), P.crossfit_external(sb, y, "mr"))}
    out = {}
    for rule, (pa, pb) in preds.items():
        out[rule] = {"qwk": paired(y, pa, pb, lambda t, p: M.quadratic_weighted_kappa(t, p, 5)),
                     "macro_recall": paired(y, pa, pb, P.macro_recall),
                     "recall": [paired(y, pa, pb, recall_fn(c)) for c in range(5)]}
    return out


def dev_accuracy(a, b, ha, hb, procedures=("shipped", "matched")):
    """'matched' is the fold-wise cross-fit every development comparison uses; 'insample'
    (cuts fitted on all rows) and 'randomhalf' (the external two-fold procedure, grid 80)
    exist to show how a per-class verdict depends on the fitting procedure (ISSUES.md §31)."""
    ib = {u: i for i, u in enumerate(b["uids"])}
    ja = np.array([i for i, u in enumerate(a["uids"]) if u in ib])
    jb = np.array([ib[a["uids"][i]] for i in ja])
    y, f = a["y"][ja], a["folds"][ja]
    out = {}
    sa, sb = P.expected_grade(a["dr"][ja], ha), P.expected_grade(b["dr"][jb], hb)
    for mode in procedures:
        if mode == "shipped":
            pa, pb = P.shipped(a["dr"][ja], ha), P.shipped(b["dr"][jb], hb)
        elif mode == "insample":
            pa = P.cuts_to_grades(sa, P.fit_cuts(sa, y, "qwk")[0])
            pb = P.cuts_to_grades(sb, P.fit_cuts(sb, y, "qwk")[0])
        elif mode == "randomhalf":
            pa, pb = P.crossfit_external(sa, y, "qwk"), P.crossfit_external(sb, y, "qwk")
        else:
            pa = P.crossfit_dev(P.expected_grade(a["dr"][ja], ha), y, f)
            pb = P.crossfit_dev(P.expected_grade(b["dr"][jb], hb), y, f)
        out[mode] = {"accuracy": paired(y, pa, pb, M.accuracy),
                     "recall": [paired(y, pa, pb, recall_fn(c)) for c in range(5)]}
    return out


def oof_from_dir(d, labels):
    uids, dr, folds = [], [], []
    for p in sorted(glob.glob(os.path.join(d, "oof_*.npz"))):
        k = int(os.path.basename(p).split("_")[1].split(".")[0])
        z = np.load(p, allow_pickle=True)
        uids += [str(u) for u in z["uids"]]; dr.append(z["dr_logits"]); folds += [k] * len(z["uids"])
    return {"uids": np.array(uids), "y": np.array([labels[u]["dr"] for u in uids]),
            "folds": np.array(folds), "dr": np.concatenate(dr).astype(np.float64)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--e12", nargs=2, required=True, help="E12def and E12nat run dirs")
    ap.add_argument("--pred-root", default="results/predictions")
    a = ap.parse_args()
    L = P.dev_labels(a.labels, "data/splits/dev_v1.json")
    out = {}
    # F3: EfficientNet-B3 (E11FULL) vs DenseNet121 (E08) on APTOS
    e11 = load_ext(a.pred_root, "E11FULL", "APTOS"); e08 = load_ext(a.pred_root, "E08", "APTOS")
    out["backbone_aptos"] = external_pair(e11, e08, "ordinal", "ordinal")
    # LP-FT vs E08, development accuracy
    out["lpft_dev"] = dev_accuracy(load_model(a.pred_root, "E15LPFT", L),
                                   load_model(a.pred_root, "E08", L), "ordinal", "ordinal")
    # E12 source control: native minus default
    nat, de = oof_from_dir(a.e12[1], L), oof_from_dir(a.e12[0], L)
    out["source_control"] = dev_accuracy(nat, de, "ordinal", "ordinal",
                                         ("shipped", "matched", "insample", "randomhalf"))
    json.dump(out, open("docs/generated/paper/record.json", "w"), indent=1, default=float)

    names = ["NoDR", "Mild", "Mod", "Sev", "PDR"]
    for rule, v in out["backbone_aptos"].items():
        print(f"E11FULL-E08 APTOS {rule}: QWK {v['qwk']['diff']:+.4f} [{v['qwk']['lo']:+.4f},"
              f"{v['qwk']['hi']:+.4f}]  " + "  ".join(
                  f"{n} {r['diff']*100:+.1f}{'*' if r['significant'] else ''}"
                  for n, r in zip(names, v["recall"])))
    for mode, v in out["lpft_dev"].items():
        r = v["accuracy"]
        print(f"LPFT-E08 dev accuracy {mode}: {r['diff']*100:+.2f} [{r['lo']*100:+.2f}, {r['hi']*100:+.2f}]")
    for mode, v in out["source_control"].items():
        r = v["recall"][2]
        print(f"E12 native-default Moderate {mode}: {r['diff']*100:+.1f} [{r['lo']*100:+.1f}, {r['hi']*100:+.1f}]")


if __name__ == "__main__":
    main()
