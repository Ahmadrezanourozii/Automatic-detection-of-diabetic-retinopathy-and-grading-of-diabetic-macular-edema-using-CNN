"""
compare_matched.py — paired bootstrap between two runs at MATCHED operating points.

`compare_runs.py` compares two runs as they shipped, at whatever cut-points `sigmoid > 0.5`
happens to give each of them. `PROTOCOL.md` §4.1 says that comparison is not evidence about
representations: an untuned default is a hyper-parameter that has been chosen, not one that
has been avoided, and a difference measured at two arbitrary thresholds is a statement about
where those thresholds fell.

This script recalibrates BOTH runs the same way — cut-points fitted cross-fitted, fold f's
cuts taken from every other fold's out-of-fold predictions, under an identical objective —
and repeats the paired bootstrap. Only what survives is a statement about representation.

Usage:
    python src/compare_matched.py --a runs/E08 --b runs/E15LPFT --datasets <root>
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics
import manifest
from tune_thresholds import load, fit_cuts, cuts_to_grades

N_DR, N_DME = 5, 3


def expected_grade(logits):
    """Ordinal head: score = sum_k P(y > k), which is the expected grade."""
    return torch.sigmoid(logits).sum(1).numpy()


def crossfit_grades(score, y, folds, k, seed=0):
    """Grades decoded with cut-points fitted on the OTHER folds only (PROTOCOL.md §3)."""
    out = np.zeros(len(score), dtype=int)
    for f in sorted(set(folds.tolist())):
        te = folds == f
        tr = ~te
        if not tr.any() or not te.any():
            continue
        cuts, _ = fit_cuts(score[tr], y[tr], k, seed=seed)   # returns (cuts, qwk)
        out[te] = cuts_to_grades(score[te], cuts)
    return out


def default_grades(logits, k):
    """The SHIPPED decode, taken from model.decode -- count leading thresholds that fire,
    made rank-consistent by cummin. Reimplementing it as round(expected_grade) would give
    a slightly different number and the 'default' row would not be the operating point the
    run actually shipped."""
    from model import decode
    return decode(logits, k, head="ordinal").numpy()


def head_arrays(rows, logits, head):
    """Return (score, y, keep-mask) for one head, dropping rows with no label."""
    score = expected_grade(logits)
    if head == "dr":
        y = np.array([r["dr"] if r["dr"] is not None else -1 for r in rows])
    else:
        # The 3-class DME evaluation is IDRiD-only. Messidor-2 rows carry a BINARY
        # (referable / not) label in dme_label_space="binary"; counting them here would
        # score a 3-class metric on rows that have no 3-class label and inflate n from
        # 516 to 667. PROTOCOL.md §5.1 and data/LABEL_MAPPING.md.
        y = np.array([r["dme"] if (r.get("dme") is not None and
                                   r.get("dme_label_space") == "3class") else -1
                      for r in rows])
    keep = y >= 0
    return score[keep], y[keep], keep


def compare(a_run, b_run, datasets, n_boot=2000, seed=0, allow_unmanifested=True,
            acknowledge_consumption_diff=False):
    """Returns (results, consumption_note).

    Configuration is not consumption (PROTOCOL.md §9). Two runs may only be compared when
    they read the same files; identical config blocks do not establish that (ISSUES.md §26).

    Some comparisons are *deliberately* across sources — I20 compared E10 against E17NAT
    precisely to measure a resolution change that also changed the image source. Those are
    legitimate, so the override exists; but it does not silence the difference, it *forces it
    into the report*. A caveat the reader cannot see is not a caveat.
    """
    names = {os.path.basename(a_run.rstrip("/")): a_run,
             os.path.basename(b_run.rstrip("/")): b_run}
    note = None
    try:
        manifest.require_same(names, allow_unmanifested=allow_unmanifested)
    except SystemExit as e:
        if not acknowledge_consumption_diff:
            raise
        note = str(e)
        print("[manifest] consumption differs; proceeding under explicit acknowledgement.\n"
              "           The difference will be stamped into the output.", flush=True)
    rows_a, folds_a, dr_a, dme_a = load(a_run, datasets)
    rows_b, folds_b, dr_b, dme_b = load(b_run, datasets)

    # align on uid: the two runs predict the same pool but not necessarily in the same order
    idx_b = {r["uid"]: i for i, r in enumerate(rows_b)}
    common = [i for i, r in enumerate(rows_a) if r["uid"] in idx_b]
    if len(common) != len(rows_a):
        print(f"  note: comparing on {len(common)} rows present in both runs")
    ja = np.array(common)
    jb = np.array([idx_b[rows_a[i]["uid"]] for i in common])
    rows = [rows_a[i] for i in common]
    folds = folds_a[ja]
    assert (folds_b[jb] == folds).all(), \
        "the two runs disagree about which fold a row is in -- different splits?"

    out = []
    for head, la, lb, k in (("dr", dr_a[ja], dr_b[jb], N_DR),
                            ("dme_ungated", dme_a[ja], dme_b[jb], N_DME)):
        sa, y, keep = head_arrays(rows, la, head)
        sb = expected_grade(lb)[keep]
        groups = [r["group"] for r, m in zip(rows, keep) if m]
        f = folds[keep]

        pred = {
            "default": (default_grades(la[keep], k), default_grades(lb[keep], k)),
            "matched": (crossfit_grades(sa, y, f, k, seed),
                        crossfit_grades(sb, y, f, k, seed)),
        }
        for mode, (pa, pb) in pred.items():
            for name, fn in (("QWK", lambda t, p: metrics.quadratic_weighted_kappa(t, p, k)),
                             ("accuracy", metrics.accuracy)):
                d, lo, hi, sig = metrics.paired_bootstrap_diff(
                    y, pa, pb, fn, groups=groups, n_boot=n_boot, seed=seed)
                out.append({
                    "head": head, "mode": mode, "metric": name,
                    "a": float(fn(y, pa)), "b": float(fn(y, pb)),
                    "diff": float(d), "lo": float(lo), "hi": float(hi),
                    "significant": bool(sig), "n": int(len(y)),
                })
    return out, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", default="docs/generated/matched_comparison.md")
    ap.add_argument("--acknowledge-consumption-diff", action="store_true",
                    help="compare two runs that read DIFFERENT files. Legitimate when the "
                         "difference is the experiment (I20), but the difference is then "
                         "printed in the output document -- it cannot be hidden.")
    a = ap.parse_args()

    res, note = compare(a.a, a.b, a.datasets,
                        acknowledge_consumption_diff=a.acknowledge_consumption_diff)
    na, nb = os.path.basename(a.a.rstrip("/")), os.path.basename(a.b.rstrip("/"))

    warn = []
    if note:
        warn = ["", "> ## ⚠️ THESE TWO RUNS DID NOT READ THE SAME FILES", ">",
                "> This comparison was run with `--acknowledge-consumption-diff`. Any "
                "difference below confounds the intended change with a change of input "
                "data (`ISSUES.md` §26, `PROTOCOL.md` §9). Details:", ">",
                "> ```", *[f"> {l}" for l in note.strip().splitlines()], "> ```", ""]
    lines = [f"# {na} vs {nb} — shipped cut-points and matched cut-points", *warn, "",
             "`PROTOCOL.md` §4.1: a difference measured at two arbitrary thresholds is not "
             "evidence about representations. **matched** rows recalibrate both runs the "
             "same way — cut-points cross-fitted on the other folds under one objective — "
             "and repeat the paired bootstrap over groups.", "",
             f"| head | cut-points | metric | {na} | {nb} | A − B | 95 % interval | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for r in res:
        fmt = (lambda v: f"{v*100:.2f} pts") if r["metric"] == "accuracy" else (lambda v: f"{v:+.4f}")
        val = (lambda v: f"{v*100:.2f} %") if r["metric"] == "accuracy" else (lambda v: f"{v:.4f}")
        lines.append(
            f"| {r['head']} | {r['mode']} | {r['metric']} | {val(r['a'])} | {val(r['b'])} | "
            f"{fmt(r['diff'])} | [{fmt(r['lo'])}, {fmt(r['hi'])}] | "
            f"{'**significant**' if r['significant'] else 'indistinguishable'} |")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(lines) + "\n")
    json.dump(res, open(a.out.replace(".md", ".json"), "w"), indent=1)
    print("\n".join(lines))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
