"""
ensemble_external.py — I23: does the ensembling gain survive on APTOS?

The development-pool ensemble is not a held-out result: every member was trained and selected
on that pool. This scores the same logit-averaging on **APTOS, held out in its entirety since
the protocol was frozen**, where no member has ever seen an image.

It also settles a caveat the dev-pool number carries. On the development pool the members did
not all read the same files — some mounted the native-resolution Messidor-2 mirror, some did
not — so their average was over different renderings of the same eyes (PROTOCOL.md §9).
**APTOS has a single source, so every member necessarily sees identical images**, and the
external number is free of that objection by construction.

MEMBERSHIP. Whichever members have archived APTOS predictions — a mechanical fact, never a
score. The matching development-pool ensemble over exactly the same members is computed
alongside, so dev and external are like-for-like.

Usage:
    python src/ensemble_external.py --datasets <root>
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpora
import metrics
from compare_matched import crossfit_grades, expected_grade, N_DR
from ensemble_oof import load_run

REFERABLE_FROM = 2

# run directory holding APTOS predictions -> the development run whose weights produced them
EXTERNAL_OF = {
    "E08X2": "E08",
    "EXTE09": "E09",
    "EXTE10": "E10",
    "EXTE14MAC": "E14MAC",
    "EXTE15LPFT": "E15LPFT",
    "EXTE17NAT": "E17NAT",
}


def load_ext(d):
    hits = sorted(glob.glob(os.path.join(d, "external_aptos*_predictions.npz")))
    # prefer the full-ensemble TTA artefact when a recipe-control file sits beside it
    plain = [h for h in hits if os.path.basename(h) == "external_aptos_predictions.npz"]
    p = (plain or hits)[0] if hits else None
    if p is None:
        return None
    z = np.load(p, allow_pickle=True)
    return dict(file=os.path.basename(p),
                uids=[str(u) for u in z["uids"]],
                groups=np.array([str(g) for g in z["groups"]]),
                y=np.asarray(z["y_dr"]),
                dr=np.asarray(z["dr_logits"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="docs/generated/ensemble_external.md")
    a = ap.parse_args()

    ext = {}
    for d, dev_run in EXTERNAL_OF.items():
        e = load_ext(os.path.join(a.runs_dir, d))
        if e:
            ext[dev_run] = e
    if len(ext) < 2:
        raise SystemExit(f"only {len(ext)} member(s) have APTOS predictions")
    members = sorted(ext)
    print(f"members with APTOS predictions: {members}")

    ref = ext[members[0]]
    order = ref["uids"]
    pos = {u: i for i, u in enumerate(order)}
    for n in members:
        e = ext[n]
        assert sorted(e["uids"]) == sorted(order), f"{n} covers different APTOS images"
        inv = np.argsort(np.array([pos[u] for u in e["uids"]]))
        e["dr"], e["y"], e["groups"] = e["dr"][inv], e["y"][inv], e["groups"][inv]
    y = ref["y"]
    groups = ref["groups"]

    def qwk_of(logits):
        s = np.asarray(expected_grade(torch.as_tensor(logits)))
        g = np.clip(np.floor(s + 0.5).astype(int), 0, N_DR - 1)
        return metrics.quadratic_weighted_kappa(y, g, N_DR), g

    singles = {}
    for n in members:
        q, _ = qwk_of(ext[n]["dr"])
        singles[n] = q
    ens_logits = np.mean([ext[n]["dr"] for n in members], axis=0)
    q_ens, g_ens = qwk_of(ens_logits)

    # matching development-pool ensemble over exactly the same members
    rows = corpora.build(a.datasets, ("IDRiD", "Messidor-2"))
    by = {r["uid"]: r for r in rows}
    split = json.load(open("data/splits/dev_v1.json"))
    dev = {}
    for n in members:
        r = load_run(os.path.join(a.runs_dir, n), by, split)
        if r:
            dev[n] = r
    dev_line, selected = None, None
    if len(dev) == len(members):
        o = dev[members[0]]["uids"]
        p2 = {u: i for i, u in enumerate(o)}
        for n in members:
            inv = np.argsort(np.array([p2[u] for u in dev[n]["uids"]]))
            dev[n]["dr"] = dev[n]["dr"][inv]
            dev[n]["folds"] = dev[n]["folds"][inv]
        folds = dev[members[0]]["folds"]
        ydev = np.array([by[u]["dr"] for u in o])
        gdev = np.array([split["groups"].get(u, u) for u in o])
        keep = ydev >= 0

        def dev_qwk(L):
            s = np.asarray(expected_grade(torch.as_tensor(L)))[keep]
            g = crossfit_grades(s, ydev[keep], folds[keep], N_DR)
            return metrics.quadratic_weighted_kappa(ydev[keep], g, N_DR)

        d_singles = {n: dev_qwk(dev[n]["dr"]) for n in members}
        d_ens = dev_qwk(np.mean([dev[n]["dr"] for n in members], axis=0))
        dev_line = (d_ens, max(d_singles.values()), max(d_singles, key=d_singles.get))
        selected = dev_line[2]

    # ── THE comparison, and the only protocol-legal one (PROTOCOL.md §3) ──────────────
    # The single model to beat is the one selected on VALIDATION, not the one that happens
    # to score highest on APTOS. Picking the external winner and comparing to it would be
    # model selection on the test set — the exact optimism §3 exists to prevent, and it
    # would make the ensemble look worse than it is.
    if selected is None:
        raise SystemExit("cannot identify the validation-selected member")
    _, g_sel = qwk_of(ext[selected]["dr"])
    d, lo, hi, sig = metrics.paired_bootstrap_diff(
        y, g_ens, g_sel, lambda t, p: metrics.quadratic_weighted_kappa(t, p, N_DR),
        groups=groups.tolist(), n_boot=2000, seed=0)
    best_on_test = max(singles, key=singles.get)

    lines = ["# I23 — does the ensembling gain survive on APTOS?", "",
             f"Members: **{', '.join(members)}** — whichever have archived APTOS predictions, "
             "a mechanical fact rather than a score. APTOS is held out in its entirety and no "
             "member has seen an image of it.", "",
             "**Every member sees identical images here**, because APTOS has a single source. "
             "The development-pool ensemble could not say that (`PROTOCOL.md` §9), so this "
             "number is free of the mixed-input caveat by construction.", "",
             "## APTOS (external, n = %d)" % len(y), "",
             "| member | APTOS DR QWK |", "|---|---|"]
    for n in members:
        lines.append(f"| {n} | {singles[n]:.4f} |")
    lines += [f"| **ensemble of {len(members)}** | **{q_ens:.4f}** |", "",
              f"The single model to beat is **{selected}**, chosen because it scores highest "
              f"on the **development pool** — never on APTOS (`PROTOCOL.md` §3).", "",
              f"**Ensemble − {selected} on APTOS: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] — "
              f"{'significant' if sig else 'indistinguishable'}**", "",
              f"> For completeness: the member scoring highest *on APTOS* is "
              f"**{best_on_test}** ({singles[best_on_test]:.4f}). Comparing against that "
              f"would be selection on the test set and is not the reported result.", ""]
    if dev_line:
        e_, b_, bn_ = dev_line
        lines += ["## The same members on the development pool, for comparison", "",
                  f"| | ensemble | best single ({bn_}) | gain |", "|---|---|---|---|",
                  f"| development pool | {e_:.4f} | {b_:.4f} | {e_-b_:+.4f} |",
                  f"| APTOS (held out) | {q_ens:.4f} | {singles[selected]:.4f} | {d:+.4f} |", ""]

    body = "\n".join(lines)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(body + "\n")
    json.dump({"members": members, "aptos_singles": singles, "aptos_ensemble": float(q_ens),
               "vs_validation_selected": {"member": selected, "diff": float(d),
                                          "lo": float(lo), "hi": float(hi),
                                          "significant": bool(sig)},
               "best_on_test_not_used_for_selection": best_on_test},
              open(a.out.replace(".md", ".json"), "w"), indent=1)
    print(body)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
