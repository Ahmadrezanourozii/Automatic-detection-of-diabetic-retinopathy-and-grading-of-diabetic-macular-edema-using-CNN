"""
report.py — regenerate every table and figure from the archived results.json files.

Nothing in the thesis is typed by hand. This script is the only path from a run to a number
in the document, so the document cannot drift from the experiments — which is exactly how
the previous version of this thesis ended up reporting figures that no run had produced
(ISSUES.md §1).

Emits, into docs/generated/:
    experiments_table.md    the EXPERIMENTS.md ledger rows
    results_dr.tex          per-class DR table for chapter 4
    results_dme.tex         per-class DME table for chapter 4
    comparison.tex          our numbers against the floors and against prior work
    confusion_dr.png        confusion matrices, from the archived predictions
    confusion_dme.png
    summary.json            every headline number, keyed, so two places cannot disagree

Usage:
    python src/report.py [--runs runs] [--out docs/generated]
"""
from __future__ import annotations
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DR_CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
DME_CLASSES = ["No DME", "Non-referable", "Referable"]
HEAD_TITLES = {
    "dr": ("DR, 5-class", DR_CLASSES),
    "dme_ungated": ("DME, 3-class (ungated — primary)", DME_CLASSES),
    "dme_gated": ("DME, 3-class (gated DR>=1 — secondary)", DME_CLASSES),
    "dme_referable_binary": ("Referable DME (binary)", ["Not referable", "Referable"]),
}


def load_runs(runs_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "results.json"))):
        try:
            r = json.load(open(p))
        except Exception as e:
            print(f"  skipping {p}: {e}")
            continue
        if "run_id" not in r:
            continue
        r["_path"] = p
        out.append(r)
    return out


def headline(run):
    """The corrected re-scoring if present, then pooled out-of-fold, then a fold mean.

    `recomputed` wins because it is the run re-scored under the current evaluation
    definition from its archived logits (ISSUES.md §12). Preferring the stale `pooled_oof`
    would quietly report a number computed under a definition we have since rejected.
    """
    if run.get("recomputed"):
        return (run["recomputed"]["metrics"], run["recomputed"]["n_images"],
                "pooled OOF, re-scored")
    if run.get("pooled_oof"):
        return run["pooled_oof"]["metrics"], run["pooled_oof"]["n_images"], "pooled OOF"
    folds = [f["metrics"] for f in run.get("folds", []) if f.get("metrics")]
    if not folds:
        return None, 0, "none"
    keys = set().union(*[set(f) for f in folds])
    merged = {}
    for k in keys:
        vals = [f[k] for f in folds if k in f]
        n = sum(v["n"] for v in vals)
        merged[k] = {
            "n": n,
            "accuracy": sum(v["accuracy"] * v["n"] for v in vals) / max(1, n),
            "qwk": sum(v["qwk"] * v["n"] for v in vals) / max(1, n),
            "macro_f1": sum(v["macro_f1"] * v["n"] for v in vals) / max(1, n),
            "majority_floor": sum(v["majority_floor"] * v["n"] for v in vals) / max(1, n),
            "accuracy_ci95": [None, None], "qwk_ci95": [None, None],
            "per_class_recall": vals[0].get("per_class_recall"),
            "support": vals[0].get("support"),
            "confusion": None, "beats_floor": None,
        }
    return merged, sum(f["n"] for f in [merged[k] for k in list(merged)[:1]]), "fold mean"


def ci(v, key, scale=100, fmt="{:.1f}"):
    lo, hi = v.get(key + "_ci95", [None, None])
    if lo is None:
        return "—"
    return f"[{fmt.format(lo*scale)}, {fmt.format(hi*scale)}]"


# ── markdown ledger ───────────────────────────────────────────────────────────
def experiments_table(runs):
    lines = ["| Run | Commit | Hypothesis | Backbone / head | n | DR acc | DR QWK | "
             "DME acc | DME QWK | vs floor |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        m, n, kind = headline(r)
        if not m:
            continue
        c = r.get("config", {})
        dr, dme = m.get("dr", {}), m.get("dme_ungated", {})
        tuned = (r.get("threshold_tuning") or {}).get("summary", {})
        beats = []
        for k, lbl in (("dr", "DR"), ("dme_ungated", "DME")):
            if m.get(k, {}).get("beats_floor") is True:
                beats.append(f"{lbl} yes")
            elif m.get(k, {}).get("beats_floor") is False:
                beats.append(f"**{lbl} NO**")
        lines.append(
            f"| `{r['run_id']}` | `{r['commit'][:7]}` | {r.get('hypothesis','—')} | "
            f"{c.get('backbone','?')} / {c.get('head','?')} @{c.get('size','?')}px | "
            f"{dr.get('n','—')} | "
            f"{dr.get('accuracy',0)*100:.1f}% | {dr.get('qwk',0):.3f} | "
            f"{dme.get('accuracy',0)*100:.1f}% | {dme.get('qwk',0):.3f} | "
            f"{', '.join(beats) or '—'} |")
        if tuned.get("DR 5-class", {}).get("tuned"):
            t = tuned["DR 5-class"]
            sig = "significant" if t["significant"] else "n.s."
            lines.append(
                f"| `{r['run_id']}`+cuts | `{r['commit'][:7]}` | "
                f"cross-fitted decision cut-points | tuned on other folds only | "
                f"{t['tuned']['n']} | {t['tuned']['accuracy']*100:.1f}% | "
                f"{t['tuned']['qwk']:.3f} | "
                f"{tuned.get('DME 3-class (ungated)',{}).get('tuned',{}).get('accuracy',0)*100:.1f}% | "
                f"{tuned.get('DME 3-class (ungated)',{}).get('tuned',{}).get('qwk',0):.3f} | "
                f"QWK {t['qwk_diff']:+.3f} {sig} |")
    return "\n".join(lines)


# ── LaTeX ─────────────────────────────────────────────────────────────────────
def latex_per_class(run, head_key, caption, label):
    m, _, kind = headline(run)
    if not m or head_key not in m:
        return None
    v = m[head_key]
    title, classes = HEAD_TITLES[head_key]
    cm = v.get("confusion")
    rows = []
    for i, name in enumerate(classes[:len(v.get("support", []))]):
        rec = v["per_class_recall"][i] if v.get("per_class_recall") else None
        sup = v["support"][i] if v.get("support") else 0
        prec = None
        if cm:
            col = sum(cm[r][i] for r in range(len(cm)))
            prec = cm[i][i] / col if col else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
        rows.append((name, sup,
                     "—" if prec is None else f"{prec:.3f}",
                     "—" if rec is None else f"{rec:.3f}",
                     "—" if f1 is None else f"{f1:.3f}"))
    body = "\n".join(f"    {n} & {s} & {p} & {r} & {f} \\\\" for n, s, p, r, f in rows)
    return f"""% generated by src/report.py from {os.path.basename(os.path.dirname(run['_path']))}/results.json
% run {run['run_id']}  commit {run['commit'][:10]}  ({kind}, n={v['n']})
\\begin{{table}}[H]
\\caption{{{caption}}}
\\centering
\\begin{{tabular}}{{|l|c|c|c|c|}}
\\hline
    \\textbf{{Class}} & \\textbf{{n}} & \\textbf{{Precision}} & \\textbf{{Recall}} & \\textbf{{F1}} \\\\ \\hline
{body}
\\hline
    \\textbf{{Overall}} & {v['n']} & \\multicolumn{{3}}{{c|}}{{
        accuracy {v['accuracy']*100:.1f}\\% {ci(v,'accuracy')},
        QWK {v['qwk']:.3f} {ci(v,'qwk',1,'{:.3f}')},
        macro-F1 {v['macro_f1']:.3f}}} \\\\ \\hline
    \\textit{{Majority-class floor}} & & \\multicolumn{{3}}{{c|}}{{
        \\textit{{{v['majority_floor']*100:.1f}\\%}}}} \\\\ \\hline
\\end{{tabular}}
\\label{{{label}}}
\\end{{table}}
"""


def latex_comparison(runs):
    """Our numbers against their floors. External numbers are NOT put in this table --
    they live in LITERATURE.md, because a comparison against a different dataset under a
    different protocol is not a comparison."""
    best = None
    for r in runs:
        m, _, _ = headline(r)
        if m and m.get("dr", {}).get("qwk") is not None:
            if best is None or m["dr"]["qwk"] > best[1]["dr"]["qwk"]:
                best = (r, m)
    if not best:
        return None
    r, m = best
    lines = []
    for k in ("dr", "dme_ungated", "dme_gated", "dme_referable_binary"):
        if k not in m:
            continue
        v = m[k]
        title = HEAD_TITLES[k][0]
        lines.append(
            f"    {title} & {v['n']} & {v['majority_floor']*100:.1f}\\% & "
            f"{v['accuracy']*100:.1f}\\% & {v['qwk']:.3f} \\\\")
    body = "\n".join(lines)
    return f"""% generated by src/report.py -- best run {r['run_id']} (commit {r['commit'][:10]})
\\begin{{table}}[H]
\\caption{{Proposed model against the majority-class floor for each task. Every accuracy is
quoted next to the floor it must beat, because the floors differ by more than twenty points
between the gated and ungated definitions of the DME task.}}
\\centering
\\begin{{tabular}}{{|l|c|c|c|c|}}
\\hline
    \\textbf{{Task}} & \\textbf{{n}} & \\textbf{{Floor}} & \\textbf{{Accuracy}} & \\textbf{{QWK}} \\\\ \\hline
{body}
\\hline
\\end{{tabular}}
\\label{{tab:vs-floor}}
\\end{{table}}
"""


# ── figures ───────────────────────────────────────────────────────────────────
def confusion_figure(run, head_key, path):
    m, _, _ = headline(run)
    if not m or head_key not in m or not m[head_key].get("confusion"):
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False
    cm = np.array(m[head_key]["confusion"], dtype=float)
    title, classes = HEAD_TITLES[head_key]
    classes = classes[:cm.shape[0]]
    norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(1.35 * len(classes) + 2.2, 1.2 * len(classes) + 1.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{title}\n{run['run_id']} — n={int(cm.sum())}", fontsize=10)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{int(cm[i,j])}\n{norm[i,j]*100:.0f}%",
                    ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else "#123")
    fig.colorbar(im, ax=ax, fraction=0.045, label="row-normalised")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="docs/generated")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = load_runs(a.runs)
    if not runs:
        print(f"no results.json with a run_id under {a.runs}/ — nothing to report")
        return
    print(f"{len(runs)} run(s): " + ", ".join(r["run_id"] for r in runs))

    tbl = experiments_table(runs)
    open(os.path.join(a.out, "experiments_table.md"), "w").write(tbl + "\n")
    print("\n" + tbl)

    best = max((r for r in runs if headline(r)[0]),
               key=lambda r: headline(r)[0].get("dr", {}).get("qwk", -9), default=None)
    summary = {}
    if best:
        m, n, kind = headline(best)
        summary = {"best_run": best["run_id"], "commit": best["commit"],
                   "basis": kind, "metrics": m}
        for key, fn, cap, lab in (
            ("dr", "results_dr.tex",
             "Per-class performance on 5-class diabetic retinopathy grading, "
             "out-of-fold over the full development pool.", "tab:dr-metrics"),
            ("dme_ungated", "results_dme.tex",
             "Per-class performance on 3-class diabetic macular edema grading "
             "(ungated: every image, grade 0 = no DME).", "tab:dme-metrics"),
        ):
            tex = latex_per_class(best, key, cap, lab)
            if tex:
                open(os.path.join(a.out, fn), "w").write(tex)
                print(f"wrote {a.out}/{fn}")
        comp = latex_comparison(runs)
        if comp:
            open(os.path.join(a.out, "comparison.tex"), "w").write(comp)
            print(f"wrote {a.out}/comparison.tex")
        for key, fn in (("dr", "confusion_dr.png"), ("dme_ungated", "confusion_dme.png")):
            if confusion_figure(best, key, os.path.join(a.out, fn)):
                print(f"wrote {a.out}/{fn}")

    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"wrote {a.out}/summary.json")
    print("\nEvery number above came from an archived results.json. "
          "Nothing here is typed by hand.")


if __name__ == "__main__":
    main()
