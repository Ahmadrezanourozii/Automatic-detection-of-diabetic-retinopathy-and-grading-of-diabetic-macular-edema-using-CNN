"""
paper_seeds.py — pre-registered outcomes O1-O3 (docs/PAPER_EXPERIMENTS_preregistration.md).

Five models that differ ONLY in the random seed (E08 = seed 0, S1E08..S4E08 = seeds 1-4).

  O1  noise floor: SD across seeds of development DR QWK at matched (fold-wise cross-fitted)
      cut-points, and of external DR QWK under the shipped rule, per external corpus.
  O2  spurious per-grade significance: over the 10 seed pairs x 5 grades on APTOS under the
      shipped rule, how many paired-bootstrap recall differences are "significant" at 95 %?
      Pre-registered prediction: more than 5 of 50. Also reported under R4 (external
      cross-fitted, QWK) and on DDR, as descriptive context.
  O3  every seed: APTOS Mild recall < 15 % under R1 and > 50 % under R4m.

Writes docs/generated/paper/seeds.json.
"""
from __future__ import annotations
import argparse, itertools, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_rules as P
import metrics as M
from paper_phase0 import load_model, load_ext
from paper_record import paired, recall_fn

SEEDS = ["E08", "S1E08", "S2E08", "S3E08", "S4E08"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pred-root", default="results/predictions")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    L = P.dev_labels(a.labels, "data/splits/dev_v1.json")
    reg = json.load(open(os.path.join(a.pred_root, "registry.json")))
    seeds = [s for s in SEEDS if s in reg and reg[s].get("oof_n") == 2260]
    out = {"seeds": seeds}

    dev = {s: load_model(a.pred_root, s, L) for s in seeds}
    devq = {s: P.dev_matched(dev[s], "ordinal")[0] for s in seeds}
    out["dev"] = {s: {"qwk_shipped": devq[s]["shipped"]["qwk"],
                      "qwk_matched": devq[s]["matched"]["qwk"],
                      "acc_matched": devq[s]["matched"]["accuracy"],
                      "mild_shipped": devq[s]["shipped"]["recall"][1]} for s in seeds}
    vals = [out["dev"][s]["qwk_matched"] for s in seeds]
    out["dev_matched_qwk_sd"] = float(np.std(vals, ddof=1))
    out["dev_matched_qwk_range"] = [min(vals), max(vals)]
    vals = [out["dev"][s]["qwk_shipped"] for s in seeds]
    out["dev_shipped_qwk_sd"] = float(np.std(vals, ddof=1))

    for corpus in ("APTOS", "DDR"):
        have = [s for s in seeds if corpus in reg[s].get("external", {})]
        if len(have) < 2:
            continue
        ext = {s: load_ext(a.pred_root, s, corpus) for s in have}
        y = ext[have[0]]["y"]
        for s in have:
            assert (ext[s]["uids"] == ext[have[0]]["uids"]).all()
        preds = {}
        for s in have:
            sc = P.expected_grade(ext[s]["dr"], "ordinal")
            preds[s] = {"R1": P.shipped(ext[s]["dr"], "ordinal"),
                        "R4": P.crossfit_external(sc, y, "qwk"),
                        "R4m": P.crossfit_external(sc, y, "mr")}
        c = {"seeds": have, "per_seed": {}}
        for s in have:
            c["per_seed"][s] = {r: P.block(y, preds[s][r]) for r in ("R1", "R4", "R4m")}
        q1 = [c["per_seed"][s]["R1"]["qwk"] for s in have]
        c["qwk_R1_sd"] = float(np.std(q1, ddof=1)); c["qwk_R1_range"] = [min(q1), max(q1)]
        for r in ("R1", "R4"):
            sig, rows = 0, []
            for s1, s2 in itertools.combinations(have, 2):
                for k in range(5):
                    d = paired(y, preds[s1][r], preds[s2][r], recall_fn(k), n_boot=a.n_boot)
                    rows.append({"pair": [s1, s2], "grade": k, **d})
                    sig += d["significant"]
            c[f"pairwise_{r}"] = {"n_tests": len(rows), "n_significant": sig,
                                  "max_abs_diff": max(abs(x["diff"]) for x in rows),
                                  "rows": rows}
        mild1 = [c["per_seed"][s]["R1"]["recall"][1] for s in have]
        mild4 = [c["per_seed"][s]["R4m"]["recall"][1] for s in have]
        c["mild_R1"] = [min(mild1), max(mild1)]; c["mild_R4m"] = [min(mild4), max(mild4)]
        if corpus == "APTOS":
            c["O2_prediction_holds"] = c["pairwise_R1"]["n_significant"] > 5
            c["O3_holds"] = bool(max(mild1) < 0.15 and min(mild4) > 0.50)
        out[corpus] = c
        print(f"{corpus}: seeds {have}  R1 QWK {min(q1):.4f}-{max(q1):.4f} (SD {c['qwk_R1_sd']:.4f})"
              f"  Mild R1 {min(mild1)*100:.1f}-{max(mild1)*100:.1f}  R4m {min(mild4)*100:.1f}-{max(mild4)*100:.1f}")
        for r in ("R1", "R4"):
            pw = c[f"pairwise_{r}"]
            print(f"  {r}: {pw['n_significant']} of {pw['n_tests']} per-grade seed-pair differences "
                  f"significant; max |diff| {pw['max_abs_diff']*100:.1f} pts")
    print(f"dev matched QWK: {[round(out['dev'][s]['qwk_matched'], 4) for s in seeds]}  "
          f"SD {out['dev_matched_qwk_sd']:.4f}")
    json.dump(out, open("docs/generated/paper/seeds.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
