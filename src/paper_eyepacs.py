"""
paper_eyepacs.py — the models trained on EyePACS (revision outcome E2).

Each model (EYEDN: DenseNet121, EYEEB3: EfficientNet-B3) was trained on folds 1-4 of a
patient-grouped EyePACS split; fold 0 (held out) chose the checkpoint and is the development set
from which rules are fitted here. For every external corpus: referral AUROC, mean
predicted-minus-true grade and referable sensitivity/specificity at the shipped rule (R1), with
cut-points refitted on external labels (R4), and with a referral threshold fitted on fold 0 for
85 % sensitivity. Writes docs/generated/paper/eyepacs.json.

Usage: python src/paper_eyepacs.py --labels <dir with eyepacs/trainLabels.csv> --search <dirs>
"""
from __future__ import annotations
import argparse, csv, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_rules as P
from paper_transfer import fit_threshold, sens_spec

MODELS = {"EYEDN": "DenseNet121, trained on EyePACS", "EYEEB3": "EfficientNet-B3, trained on EyePACS"}
CORPORA = ["APTOS", "DDR", "Messidor-2", "IDRiD", "DeepDRiD", "SUSTech"]


def find(roots, run, fname):
    for r in roots:
        for p in glob.glob(os.path.join(r, "**", run, fname), recursive=True):
            if f"{os.sep}repo{os.sep}" not in p:
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--search", nargs="+", required=True)
    ap.add_argument("--out", default="docs/generated/paper/eyepacs.json")
    a = ap.parse_args()
    lab = {}
    for r in csv.DictReader(open(os.path.join(a.labels, "eyepacs", "trainLabels.csv"))):
        lab[f"EYEPACS_{r['image']}"] = int(r["level"])
    out = {"models": {}}
    for m, label in MODELS.items():
        p = find(a.search, m, "oof_0.npz")
        if not p:
            print(f"{m}: no oof_0.npz yet")
            continue
        z = np.load(p, allow_pickle=True)
        uids = [str(u) for u in z["uids"]]
        res = json.load(open(find(a.search, m, "results.json")))
        head = res.get("config", {}).get("head", "ordinal")
        dev = {"y": np.array([lab[u] for u in uids]), "dr": np.asarray(z["dr_logits"], dtype=np.float64),
               "folds": np.zeros(len(uids), dtype=int)}
        s_dev = P.expected_grade(dev["dr"], head)
        matched = P.crossfit_external(s_dev, dev["y"], "qwk")          # two-fold within the held-out fold
        sp_path = find(a.search, m, "split.json") or "data/splits/eyepacs_v1.json"
        n_split = json.load(open(sp_path))["n_images"] if sp_path else None
        v = {"label": label, "head": head,
             "train_n": int(n_split - len(uids)) if n_split else None,
             "dev": {"n": len(uids), "qwk_shipped": P.block(dev["y"], P.shipped(dev["dr"], head))["qwk"],
                     "qwk_matched": P.block(dev["y"], matched)["qwk"],
                     "auroc": P.auroc(P.p_greater(dev["dr"], head)[:, 1], dev["y"] >= 2)},
             "external": {}}
        thr = fit_threshold(P.p_greater(dev["dr"], head)[:, 1], dev["y"] >= 2, 0.85)
        for c in CORPORA:
            pe = find(a.search, m, f"external_{c.lower()}_predictions.npz")
            if not pe:
                continue
            e = P.load_external(pe)
            rm, preds = P.rule_matrix(dev, e, head)
            xs = P.p_greater(e["dr"], head)[:, 1]
            se, sp = sens_spec(xs, e["y"] >= 2, thr)
            v["external"][c] = {"n": int(len(e["y"])), "auroc": P.auroc(xs, e["y"] >= 2),
                                "mean_R1": rm["R1"]["mean_pred_minus_true"],
                                "sens_R1": rm["R1"]["ref_sens"], "spec_R1": rm["R1"]["ref_spec"],
                                "sens_R4": rm["R4"]["ref_sens"], "spec_R4": rm["R4"]["ref_spec"],
                                "qwk_R1": rm["R1"]["qwk"], "qwk_R4": rm["R4"]["qwk"],
                                "mild_R1": rm["R1"]["recall"][1], "mild_R4m": rm["R4m"]["recall"][1],
                                "sens_tr85": float(se), "spec_tr85": float(sp)}
            r = v["external"][c]
            print(f"{m} {c:10s} AUC {r['auroc']:.3f} mean {r['mean_R1']:+.2f} sens R1 {r['sens_R1']:.3f} "
                  f"R4 {r['sens_R4']:.3f} thr85 {r['sens_tr85']:.3f}")
        gaps = [abs(r["sens_tr85"] - 0.85) for r in v["external"].values()]
        aucs = [r["auroc"] for r in v["external"].values()]
        v["n_gap_gt5"] = int(sum(g > 0.05 for g in gaps))
        v["E2_holds"] = bool(v["n_gap_gt5"] >= 2 and min(aucs) >= 0.90) if aucs else None
        out["models"][m] = v
        print(f"{m}: dev QWK {v['dev']['qwk_matched']:.4f}; {v['n_gap_gt5']} corpora with gap > 5 points; "
              f"min AUC {min(aucs) if aucs else float('nan'):.3f}; E2 {v['E2_holds']}")
    json.dump(out, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
