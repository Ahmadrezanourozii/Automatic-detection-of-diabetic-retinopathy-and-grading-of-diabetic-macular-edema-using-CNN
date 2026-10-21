"""
paper_overlap_sensitivity.py — do the external results depend on images that the content audit
placed within re-encoding distance (<= 8 bits) of an EyePACS image (pretraining corpus of the
development models, training corpus of the EyePACS models)?

Recomputes, for every development model and both EyePACS models, the shipped-rule referable sensitivity, mild recall,
QWK and referral AUROC with those images removed, and the 85 % development-threshold transfer for
E08. Writes docs/generated/paper/overlap_sensitivity.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_rules as P
from paper_phase0 import load_ext

GP = "docs/generated/paper"
MODELS = ["E08", "E09", "E10", "E11FULL", "E14MAC", "E15LPFT", "E17NAT", "E20CORAL", "E22CORN", "E21CNXA"]
EYE = ["EYEDN", "EYEEB3"]          # trained on EyePACS itself: the models an overlap would favour most
reg = json.load(open("results/predictions/registry.json"))
for m in EYE:
    reg[m] = {"head": json.load(open(f"results/predictions/{m}/results.json")).get("config", {}).get("head", "ordinal")}


def load_any(m, corpus):
    if m not in EYE:
        return load_ext("results/predictions", m, corpus)
    z = np.load(f"results/predictions/{m}/external_{corpus.lower()}_predictions.npz", allow_pickle=False)
    return {"uids": np.array([str(u) for u in z["uids"]]), "y": np.asarray(z["y_dr"]).astype(int),
            "dr": np.asarray(z["dr_logits"], dtype=np.float64)}


out = {"threshold_bits": 8, "corpora": {}}
# X4AUDIT covered APTOS and DDR; X7AUDIT (post hoc) DeepDRiD and SUSTech-SYSU
for corpus, afile, tag in (("DDR", "overlap_audit.json", "EyePACS_vs_DDR"),
                           ("APTOS", "overlap_audit.json", "EyePACS_vs_APTOS"),
                           ("DeepDRiD", "overlap_audit_x7.json", "EyePACS_vs_DeepDRiD"),
                           ("SUSTech", "overlap_audit_x7.json", "EyePACS_vs_SUSTech")):
    audit = json.load(open(f"{GP}/{afile}"))
    flagged = {p[0] for p in audit["pairs"][tag]["le8"]}
    c = {"n_flagged": len(flagged), "models": {}}
    for m in MODELS + EYE:
        e = load_any(m, corpus)
        keep = np.array([u not in flagged for u in e["uids"]])
        res = {}
        for name, k in (("all", np.ones_like(keep)), ("excluded", keep)):
            y, lg = e["y"][k], e["dr"][k]
            b = P.block(y, P.shipped(lg, reg[m]["head"]))
            tf = P.threshold_free(lg, reg[m]["head"], y)
            res[name] = {"n": int(k.sum()), "ref_sens": b["ref_sens"], "mild": b["recall"][1],
                         "qwk": b["qwk"], "auroc": tf["ref_auroc"]}
        res["max_abs_change"] = max(abs(res["excluded"][x] - res["all"][x])
                                    for x in ("ref_sens", "mild", "qwk", "auroc"))
        c["models"][m] = res
    c["max_abs_change"] = max(v["max_abs_change"] for v in c["models"].values())
    c["max_abs_change_eyepacs_models"] = max(c["models"][m]["max_abs_change"] for m in EYE)
    out["corpora"][corpus] = c
    print(corpus, "flagged", len(flagged), "max |change| over models and metrics",
          round(c["max_abs_change"], 4))
json.dump(out, open(f"{GP}/overlap_sensitivity.json", "w"), indent=1, default=float)
