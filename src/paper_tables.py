"""
paper_tables.py — the pre-registered intervention table (Table 2 / Fig. 4) from archived
matched-calibration comparisons.

Each row: intervention, its control, the development DR QWK difference (intervention minus
control) at matched calibration with its paired-bootstrap interval, and the pre-registered
verdict. Differences are re-signed so that positive always means "the intervention helped".
Where docs/generated/paper/matched_<tag>.json exists (recomputed for the paper, e.g. the
five-fold RETFound comparison) it supersedes the archived file.
"""
from __future__ import annotations
import json, os

G, GP = "docs/generated", "docs/generated/paper"

# (label, archived file, intervention is side "a" or "b", control, folds note)
ROWS = [
    ("EfficientNet-B3 backbone", "matched_e11full_e10.json", "a", "E10", "5"),
    ("Macula-centred crop (DME QWK)", "matched_e08_e14mac.json", "b", "E08", "5"),
    ("Linear probe then fine-tune", "matched_comparison.json", "b", "E08", "5"),
    ("Native 640-px resolution", "matched_e10_e17nat.json", "b", "E10", "5"),
    ("CORAL ordinal head", "matched_coral_e08.json", "a", "E08", "5"),
    ("CORN ordinal head", "matched_corn_e08.json", "a", "E08", "5"),
    ("ConvNeXt-tiny backbone", "matched_cnx_e08.json", "a", "E08", "2"),
    ("RETFound ViT-L/16 (fine-tuned)", "matched_retfound_e09.json", "a", "E09", "2"),
]


def pick(rows, head, mode, metric="QWK"):
    for r in rows:
        if r["head"] == head and r["mode"] == mode and r["metric"] == metric:
            return r
    raise KeyError((head, mode, metric))


def main():
    out = []
    for label, fn, side, control, folds in ROWS:
        override = os.path.join(GP, fn.replace(".json", "_5fold.json"))
        src = override if os.path.exists(override) else os.path.join(G, fn)
        data = json.load(open(src))
        if os.path.exists(override):
            folds = "5"
        sgn = 1 if side == "a" else -1
        row = {"label": label, "control": control, "folds": folds, "source": src}
        for head in ("dr", "dme_ungated"):
            for mode in ("default", "matched"):
                r = pick(data, head, mode)
                d, lo, hi = sgn * r["diff"], sgn * r["lo"], sgn * r["hi"]
                lo, hi = min(lo, hi), max(lo, hi)
                row[f"{head}_{mode}"] = {"diff": d, "lo": lo, "hi": hi, "n": r["n"]}
        # the macula crop was pre-registered on the DME head; every other row on DR
        m = row["dme_ungated_matched"] if "DME" in label else row["dr_matched"]
        row.update(diff=m["diff"], lo=m["lo"], hi=m["hi"])
        row["verdict"] = ("confirmed" if m["lo"] > 0 else
                          "falsified, significantly worse" if m["hi"] < 0 else "falsified")
        out.append(row)
    e = json.load(open(os.path.join(G, "ensemble_external.json")))
    out.append({"label": "Six-model ensemble (APTOS, external)", "control": "E10",
                "folds": "5", "source": os.path.join(G, "ensemble_external.json"),
                "diff": e["vs_validation_selected"]["diff"], "lo": e["vs_validation_selected"]["lo"],
                "hi": e["vs_validation_selected"]["hi"],
                "verdict": "falsified", "external": True})
    os.makedirs(GP, exist_ok=True)
    json.dump(out, open(os.path.join(GP, "interventions.json"), "w"), indent=1)
    for r in out:
        print(f"{r['label']:38s} {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  {r['verdict']}")


if __name__ == "__main__":
    main()
