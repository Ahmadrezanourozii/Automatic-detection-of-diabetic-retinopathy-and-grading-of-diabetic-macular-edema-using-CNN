"""
compare_probes.py — RETFound vs ImageNet, frozen probe against frozen probe.

WHY THIS AND NOT RETFound-vs-E09. The Stage 1 probe scored DR QWK 0.7008 against E09's
0.8389. That number is **not reportable as a verdict on RETFound**, because it compares a
frozen linear probe against a fully fine-tuned network: it measures probing versus
fine-tuning, not one representation against another. The comparison that isolates the
representation holds everything else fixed — same images, same cross-fitted ordinal head,
same matched calibration, same folds — and changes only the frozen backbone.

Both probes are decoded with cut-points cross-fitted on the other folds (PROTOCOL.md §4.1),
because a fixed decode is what made the raw Stage 1 number look like a 23 % catastrophe when
the ranking was in fact carrying real signal.

Usage:
    python src/compare_probes.py --a runs/I24PROBE --b runs/I24BASEPROBE
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics
from compare_matched import crossfit_grades, expected_grade

HEADS = (("dr", 5, "DR, 5-class"), ("dme_ungated", 3, "DME, 3-class ungated"))


def load(run, head):
    p = os.path.join(run, f"probe_{head}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return dict(logits=z["logits"], y=z["y"], folds=z["folds"],
                uids=[str(u) for u in z["uids"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="RETFound probe run dir")
    ap.add_argument("--b", required=True, help="ImageNet probe run dir")
    ap.add_argument("--name-a", default="RETFound CFP ViT-L/16")
    ap.add_argument("--name-b", default="ImageNet DenseNet121")
    ap.add_argument("--splits", default="data/splits/dev_v1.json")
    ap.add_argument("--out", default="docs/generated/probe_vs_probe.md")
    a = ap.parse_args()

    split = json.load(open(a.splits))
    lines = ["# Frozen probe vs frozen probe — does RETFound carry more signal than ImageNet?",
             "",
             f"**{a.name_a}** against **{a.name_b}**. Same 2 260 images, same cross-fitted "
             "linear ordinal head, same matched calibration, same folds — **only the frozen "
             "backbone differs.**", "",
             "> The Stage 1 figure of DR QWK 0.7008 against E09's 0.8389 is **not** a verdict "
             "on RETFound: it compares a frozen probe against a fully fine-tuned network, so "
             "it measures probing versus fine-tuning. This table is the comparison that "
             "isolates the representation.", "",
             "| head | n | " + a.name_a + " | " + a.name_b + " | A − B | 95 % interval | verdict |",
             "|---|---|---|---|---|---|---|"]
    out = {}
    for head, k, label in HEADS:
        ra, rb = load(a.a, head), load(a.b, head)
        if ra is None or rb is None:
            lines.append(f"| {label} | — | (missing) | (missing) | — | — | — |")
            continue
        assert ra["uids"] == rb["uids"], f"{head}: the two probes cover different images"
        y, folds = ra["y"], ra["folds"]
        groups = [split["groups"].get(u, u) for u in ra["uids"]]

        def grades(r):
            s = np.asarray(expected_grade(torch.as_tensor(r["logits"])))
            return crossfit_grades(s, y, folds, k)

        ga, gb = grades(ra), grades(rb)
        qa = metrics.quadratic_weighted_kappa(y, ga, k)
        qb = metrics.quadratic_weighted_kappa(y, gb, k)
        d, lo, hi, sig = metrics.paired_bootstrap_diff(
            y, ga, gb, lambda t, p: metrics.quadratic_weighted_kappa(t, p, k),
            groups=groups, n_boot=2000, seed=0)
        lines.append(f"| {label} | {len(y)} | **{qa:.4f}** | **{qb:.4f}** | {d:+.4f} | "
                     f"[{lo:+.4f}, {hi:+.4f}] | "
                     f"{'**significant**' if sig else 'indistinguishable'} |")
        out[head] = dict(a=float(qa), b=float(qb), diff=float(d), lo=float(lo),
                         hi=float(hi), significant=bool(sig), n=int(len(y)))

    dr = out.get("dr")
    if dr:
        lines += ["", "## What follows, by the decision rule fixed before the numbers", ""]
        if dr["significant"] and dr["diff"] > 0:
            lines.append("**RETFound beats ImageNet beyond the paired interval.** The "
                         "representation is genuinely better, and Stage 2 (fine-tuning) is "
                         "worth the quota — split across runs to stay under the 10 h cap.")
        elif dr["significant"] and dr["diff"] < 0:
            lines.append("**ImageNet beats RETFound.** Stage 2 is closed, and this is stated "
                         "plainly rather than hedged: on this pool, retinal self-supervised "
                         "pretraining carried *less* usable DR signal than generic ImageNet "
                         "weights.")
        else:
            lines.append("**The two are indistinguishable.** Stage 2 is closed. A model "
                         "self-supervised on ~1.6 M retinal images carries no more DR signal "
                         "than ImageNet weights on our 2 260-image pool — a measured negative "
                         "about what foundation-model pretraining buys at this data scale, "
                         "not a failed experiment.")

    body = "\n".join(lines)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(body + "\n")
    json.dump(out, open(a.out.replace(".md", ".json"), "w"), indent=1)
    print(body)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
