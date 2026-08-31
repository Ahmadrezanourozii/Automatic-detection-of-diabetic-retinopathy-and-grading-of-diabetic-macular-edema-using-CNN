"""
ensemble_oof.py — logit-average archived out-of-fold predictions across runs.

Costs no GPU: every run already wrote per-image logits for the whole development pool, and
averaging them is a deliberate method rather than an artefact of one evaluation path.

SELECTION DISCIPLINE (PROTOCOL.md §3). Which runs go into an ensemble must be decided by a
rule fixed in advance, not by trying subsets and keeping the best — that would be selection
on the evaluation set, which is precisely the optimism §3 exists to prevent. Two rules are
pre-specified here and both are reported, whatever they say:

    all-5fold   every run with a complete 5-fold OOF on the frozen split
    matched-448 every 5-fold run trained at size >= 448

Anything else is exploratory and must be labelled as such.

Comparisons use matched calibration: cut-points cross-fitted on the other folds, the same
objective for the ensemble and for each member (PROTOCOL.md §4.1).

Usage:
    python src/ensemble_oof.py --datasets <root>
"""
from __future__ import annotations
import argparse, glob, itertools, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpora
import metrics
from compare_matched import crossfit_grades, default_grades, expected_grade, N_DR, N_DME


def load_run(run, by, split):
    uids, dr, dme, folds = [], [], [], []
    files = sorted(glob.glob(os.path.join(run, "oof_*.npz")))
    if len(files) != 5:
        return None
    for p in files:
        f = int(os.path.basename(p).split("_")[1].split(".")[0])
        z = np.load(p, allow_pickle=True)
        us = [str(u) for u in z["uids"]]
        uids += us; folds += [f] * len(us)
        dr.append(z["dr_logits"]); dme.append(z["dme_logits"])
    return dict(uids=uids, folds=np.array(folds),
                dr=np.concatenate(dr), dme=np.concatenate(dme))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="docs/generated/ensemble_oof.md")
    a = ap.parse_args()

    rows_all = corpora.build(a.datasets, ("IDRiD", "Messidor-2"))
    by = {r["uid"]: r for r in rows_all}
    split = json.load(open("data/splits/dev_v1.json"))

    loaded = {}
    for d in sorted(glob.glob(os.path.join(a.runs_dir, "*"))):
        if not os.path.isdir(d):
            continue
        r = load_run(d, by, split)
        if r is None:
            continue
        cfg_path = os.path.join(d, "results.json")
        cfg = json.load(open(cfg_path))["config"] if os.path.exists(cfg_path) else {}
        r["name"] = os.path.basename(d)
        r["size"] = cfg.get("size")
        r["backbone"] = cfg.get("backbone")
        loaded[r["name"]] = r

    if not loaded:
        raise SystemExit("no run has a complete 5-fold OOF")

    # a common image order, so logits from different runs are aligned per image
    ref = next(iter(loaded.values()))
    order = ref["uids"]
    pos = {u: i for i, u in enumerate(order)}
    for r in loaded.values():
        idx = np.array([pos[u] for u in r["uids"]])
        inv = np.argsort(idx)
        r["dr"], r["dme"] = r["dr"][inv], r["dme"][inv]
        assert [r["uids"][i] for i in inv] == order, f"{r['name']} does not cover the pool"
        r["folds"] = r["folds"][inv]
    folds = ref["folds"][np.argsort(np.array([pos[u] for u in ref["uids"]]))]

    y_dr = np.array([by[u]["dr"] for u in order])
    y_dme = np.array([by[u]["dme"] if by[u].get("dme_label_space") == "3class" else -1
                      for u in order])
    groups = np.array([split["groups"].get(u, u) for u in order])

    SETS = {
        "all-5fold": sorted(loaded),
        "matched-448": sorted(n for n, r in loaded.items() if (r["size"] or 0) >= 448),
    }

    def score(dr_logits, dme_logits):
        out = {}
        for head, L, y, k in (("dr", dr_logits, y_dr, N_DR),
                              ("dme_ungated", dme_logits, y_dme, N_DME)):
            keep = y >= 0
            s = expected_grade(torch.as_tensor(L))[keep]
            g = crossfit_grades(s, y[keep], folds[keep], k)
            out[head] = dict(
                n=int(keep.sum()),
                qwk=float(metrics.quadratic_weighted_kappa(y[keep], g, k)),
                acc=float(metrics.accuracy(y[keep], g)))
        return out

    lines = ["# Ensembling archived out-of-fold predictions (no GPU)", "",
             "Logit-averaged across runs, decoded with cut-points **cross-fitted on the other "
             "folds** so the ensemble and every member sit at matched operating points "
             "(`PROTOCOL.md` §4.1). Membership follows rules fixed before the numbers were "
             "seen (`PROTOCOL.md` §3) — no subset search.", "",
             "## Members, scored individually at matched calibration", "",
             "| run | backbone | size | DR QWK | DR acc | DME QWK | DME acc |",
             "|---|---|---|---|---|---|---|"]
    singles = {}
    for n in sorted(loaded):
        r = loaded[n]
        s = score(r["dr"], r["dme"])
        singles[n] = s
        lines.append(f"| {n} | {r['backbone']} | {r['size']} | {s['dr']['qwk']:.4f} | "
                     f"{s['dr']['acc']*100:.2f} % | {s['dme_ungated']['qwk']:.4f} | "
                     f"{s['dme_ungated']['acc']*100:.2f} % |")

    lines += ["", "## Pre-specified ensembles", "",
              "| rule | members | DR QWK | DR acc | DME QWK | DME acc | best single DR QWK | Δ vs best single |",
              "|---|---|---|---|---|---|---|---|"]
    results = {}
    for rule, names in SETS.items():
        if len(names) < 2:
            continue
        dr = np.mean([loaded[n]["dr"] for n in names], axis=0)
        dme = np.mean([loaded[n]["dme"] for n in names], axis=0)
        s = score(dr, dme)
        best = max(singles[n]["dr"]["qwk"] for n in names)
        results[rule] = dict(members=names, ens=s, best_single=best)
        lines.append(f"| `{rule}` | {len(names)}: {', '.join(names)} | **{s['dr']['qwk']:.4f}** | "
                     f"{s['dr']['acc']*100:.2f} % | {s['dme_ungated']['qwk']:.4f} | "
                     f"{s['dme_ungated']['acc']*100:.2f} % | {best:.4f} | "
                     f"**{s['dr']['qwk']-best:+.4f}** |")

        # paired bootstrap of the ensemble against its own best member
        bn = max(names, key=lambda n: singles[n]["dr"]["qwk"])
        keep = y_dr >= 0
        se = expected_grade(torch.as_tensor(dr))[keep]
        sb = expected_grade(torch.as_tensor(loaded[bn]["dr"]))[keep]
        ge = crossfit_grades(se, y_dr[keep], folds[keep], N_DR)
        gb = crossfit_grades(sb, y_dr[keep], folds[keep], N_DR)
        d, lo, hi, sig = metrics.paired_bootstrap_diff(
            y_dr[keep], ge, gb,
            lambda t, p: metrics.quadratic_weighted_kappa(t, p, N_DR),
            groups=groups[keep], n_boot=2000, seed=0)
        results[rule]["vs_best"] = dict(best=bn, diff=float(d), lo=float(lo), hi=float(hi),
                                        significant=bool(sig))
        lines.append(f"| | *vs best member `{bn}`, paired bootstrap* | | | | | | "
                     f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}] "
                     f"{'**significant**' if sig else 'indistinguishable'} |")

    body = "\n".join(lines)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(body + "\n")
    json.dump({"singles": singles, "ensembles": results},
              open(a.out.replace(".md", ".json"), "w"), indent=1)
    print(body)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
