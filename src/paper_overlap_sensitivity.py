"""
paper_overlap_sensitivity.py — do the DDR and APTOS results depend on external images that the
content audit placed within re-encoding distance (<= 8 bits) of an EyePACS pretraining image?

Recomputes, for every pre-registered model, the shipped-rule referable sensitivity, mild recall,
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
audit = json.load(open(f"{GP}/overlap_audit.json"))
reg = json.load(open("results/predictions/registry.json"))
out = {"threshold_bits": 8, "corpora": {}}
for corpus, tag in (("DDR", "EyePACS_vs_DDR"), ("APTOS", "EyePACS_vs_APTOS")):
    flagged = {p[0] for p in audit["pairs"][tag]["le8"]}
    c = {"n_flagged": len(flagged), "models": {}}
    for m in MODELS:
        e = load_ext("results/predictions", m, corpus)
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
    out["corpora"][corpus] = c
    print(corpus, "flagged", len(flagged), "max |change| over models and metrics",
          round(c["max_abs_change"], 4))
json.dump(out, open(f"{GP}/overlap_sensitivity.json", "w"), indent=1, default=float)
