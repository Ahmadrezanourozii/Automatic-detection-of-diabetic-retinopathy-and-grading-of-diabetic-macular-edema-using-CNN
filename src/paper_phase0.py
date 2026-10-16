"""
paper_phase0.py — the decision-rule versus model analysis (pre-registered outcome O7).

For every model in results/predictions/registry.json with development OOF predictions and
predictions on an external corpus, compute the rule matrix R1-R4m (src/paper_rules.py), then
ask the pre-registered question: does the choice among DEPLOYABLE decision rules (R1-R3)
move a clinical metric more than the choice among MODELS under a fixed rule?

Writes docs/generated/paper/phase0_<corpus>.json and .md.

Usage:
    python src/paper_phase0.py --labels <dir with IDRiD + Messidor-2 label CSVs> \
        --corpus APTOS [--models E08,E09,...]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_rules as P

DEPLOYABLE = ("R1", "R2", "R3")
METRICS = {"mild_recall": lambda b: b["recall"][1],
           "ref_sens": lambda b: b["ref_sens"],
           "ref_spec": lambda b: b["ref_spec"],
           "qwk": lambda b: b["qwk"],
           "accuracy": lambda b: b["accuracy"]}


def load_model(pred_root, name, labels):
    z = np.load(os.path.join(pred_root, name, "oof.npz"), allow_pickle=True)
    uids = [str(u) for u in z["uids"]]
    return {"uids": np.array(uids), "y": np.array([labels[u]["dr"] for u in uids]),
            "folds": np.asarray(z["folds"]).astype(int),
            "dr": np.asarray(z["dr_logits"], dtype=np.float64),
            "dme": np.asarray(z["dme_logits"], dtype=np.float64)}


def load_ext(pred_root, name, corpus):
    p = os.path.join(pred_root, name, f"external_{corpus.lower()}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {"uids": np.array([str(u) for u in z["uids"]]), "y": np.asarray(z["y_dr"]).astype(int),
            "dr": np.asarray(z["dr_logits"], dtype=np.float64)}


def spread(per_model, metric):
    """within-model range across R1-R3 (one per model) and between-model range per rule."""
    f = METRICS[metric]
    within = {m: max(f(r[x]) for x in DEPLOYABLE) - min(f(r[x]) for x in DEPLOYABLE)
              for m, r in per_model.items()}
    between = {x: max(f(r[x]) for r in per_model.values()) -
                  min(f(r[x]) for r in per_model.values()) for x in DEPLOYABLE}
    w = np.array(list(within.values()))
    return {"within_model_range": within,
            "within_median": float(np.median(w)),
            "within_iqr": [float(np.percentile(w, 25)), float(np.percentile(w, 75))],
            "between_model_range": between,
            # O7 compares the median within-model range against the between-model range
            # under the shipped rule R1, as pre-registered
            "o7_holds": bool(np.median(w) > between["R1"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--splits", default="data/splits/dev_v1.json")
    ap.add_argument("--pred-root", default="results/predictions")
    ap.add_argument("--corpus", default="APTOS")
    ap.add_argument("--models", default="")
    ap.add_argument("--out-dir", default="docs/generated/paper")
    a = ap.parse_args()

    labels = P.dev_labels(a.labels, a.splits)
    reg = json.load(open(os.path.join(a.pred_root, "registry.json")))
    names = [m for m in (a.models.split(",") if a.models else reg)
             if a.corpus in reg[m].get("external", {}) and reg[m].get("oof_n")]
    per_model, dev = {}, {}
    for m in names:
        head = reg[m]["head"]
        d = load_model(a.pred_root, m, labels)
        e = load_ext(a.pred_root, m, a.corpus)
        rm, _ = P.rule_matrix(d, e, head)
        per_model[m] = rm
        dv, _ = P.dev_matched(d, head)
        dev[m] = {"n": dv["n"], "folds": dv["folds"],
                  "qwk_shipped": dv["shipped"]["qwk"], "qwk_matched": dv["matched"]["qwk"]}
        print(f"{m:9s} dev {dv['n']:4d} QWK {dv['matched']['qwk']:.4f} | {a.corpus} "
              + "  ".join(f"{r}: QWK {rm[r]['qwk']:.3f} mild {rm[r]['recall'][1]*100:4.1f} "
                          f"sens {rm[r]['ref_sens']*100:5.1f}"
                          for r in ("R1", "R2", "R3", "R4m")))

    out = {"corpus": a.corpus, "models": names, "dev": dev, "rule_matrix": per_model,
           "spread": {k: spread(per_model, k) for k in METRICS}}
    # rank stability: do the rules agree on WHICH model is best?
    full = [m for m in names if len(dev[m]["folds"]) == 5]
    q = {r: [per_model[m][r]["qwk"] for m in full] for r in ("R1", "R2", "R3", "R4")}
    devq = [dev[m]["qwk_matched"] for m in full]
    out["rank_models_5fold"] = full
    out["spearman"] = {f"{x}_vs_{y}": float(spearmanr(q[x], q[y]).correlation)
                       for x, y in (("R1", "R2"), ("R1", "R3"), ("R2", "R3"), ("R1", "R4"))}
    out["spearman"]["dev_matched_vs_R1"] = float(spearmanr(devq, q["R1"]).correlation)
    out["spearman"]["dev_matched_vs_R4"] = float(spearmanr(devq, q["R4"]).correlation)

    os.makedirs(a.out_dir, exist_ok=True)
    jp = os.path.join(a.out_dir, f"phase0_{a.corpus.lower()}.json")
    json.dump(out, open(jp, "w"), indent=1, default=float)

    lines = [f"# Phase 0 — decision rule vs model on {a.corpus}", "",
             "Generated by `src/paper_phase0.py`; rules as pre-registered in "
             "`docs/PAPER_EXPERIMENTS_preregistration.md`. R4/R4m use external labels "
             "(diagnostic only).", "",
             "| model | dev QWK (matched) | rule | QWK | κ | acc | Mild recall | ref. sens | ref. spec |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in names:
        for r in ("R1", "R2", "R3", "R4", "R4m"):
            b = per_model[m][r]
            lines.append(f"| {m if r == 'R1' else ''} | "
                         f"{dev[m]['qwk_matched']:.4f} | {r} | {b['qwk']:.4f} | {b['kappa']:.3f} | "
                         f"{b['accuracy']*100:.1f} | {b['recall'][1]*100:.1f} | "
                         f"{b['ref_sens']*100:.2f} | {b['ref_spec']*100:.2f} |"
                         if r == "R1" else
                         f"| | | {r} | {b['qwk']:.4f} | {b['kappa']:.3f} | {b['accuracy']*100:.1f} | "
                         f"{b['recall'][1]*100:.1f} | {b['ref_sens']*100:.2f} | {b['ref_spec']*100:.2f} |")
    lines += ["", "## O7 — within-model (across R1–R3) vs between-model (under R1) range", "",
              "| metric | median within-model range [IQR] | between-model range R1 / R2 / R3 | O7 holds |",
              "|---|---|---|---|"]
    for k, s in out["spread"].items():
        sc = 100 if k != "qwk" else 1
        fmt = (lambda v: f"{v*sc:.1f}") if k != "qwk" else (lambda v: f"{v:.4f}")
        lines.append(f"| {k} | {fmt(s['within_median'])} [{fmt(s['within_iqr'][0])}, "
                     f"{fmt(s['within_iqr'][1])}] | "
                     + " / ".join(fmt(s['between_model_range'][r]) for r in DEPLOYABLE)
                     + f" | {'yes' if s['o7_holds'] else 'NO'} |")
    lines += ["", "## Rank agreement (Spearman, 5-fold models only: " + ", ".join(full) + ")", ""]
    lines += [f"* {k}: {v:+.3f}" for k, v in out["spearman"].items()]
    open(os.path.join(a.out_dir, f"phase0_{a.corpus.lower()}.md"), "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {jp}")
    for k, s in out["spread"].items():
        print(f"  {k:12s} within-median {s['within_median']:.4f}  between R1 "
              f"{s['between_model_range']['R1']:.4f}  O7 {'holds' if s['o7_holds'] else 'FAILS'}")
    print("  spearman", {k: round(v, 3) for k, v in out["spearman"].items()})


if __name__ == "__main__":
    main()
