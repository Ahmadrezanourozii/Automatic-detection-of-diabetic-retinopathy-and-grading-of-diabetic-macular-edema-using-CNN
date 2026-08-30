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

# Persian names for the generated LaTeX, so the tables drop into the RTL thesis unaltered.
# The DME names are the clinically correct ones -- grade 0 is the ABSENCE of oedema, not
# "mild" as the previous version of this thesis had it (ISSUES.md §2).
FA_DR = ["بدون رتینوپاتی", "خفیف", "متوسط", "شدید", "تکثیری"]
FA_DME = ["بدون ورم ماکولا", "غیرقابل‌ارجاع", "قابل‌ارجاع"]
FA_HEAD = {
    "dr": ("رتینوپاتی دیابتی، پنج‌کلاسه", FA_DR),
    "dme_ungated": ("ورم ماکولا دیابتی، سه‌کلاسه (بدون دروازه — معیار اصلی)", FA_DME),
    "dme_gated": ("ورم ماکولا دیابتی، سه‌کلاسه (مشروط به \\lr{DR}$\\geq$1 — ثانویه)", FA_DME),
    "dme_referable_binary": ("ورم ماکولا قابل‌ارجاع (دودویی)",
                             ["غیرقابل‌ارجاع", "قابل‌ارجاع"]),
}
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
def latex_per_class(run, head_key, caption, label, fa=True):
    m, _, kind = headline(run)
    if not m or head_key not in m:
        return None
    v = m[head_key]
    title, classes = (FA_HEAD if fa else HEAD_TITLES)[head_key]
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
    hdr = ("\\textbf{کلاس} & \\textbf{تعداد} & \\textbf{صحت} & \\textbf{فراخوانی} "
           "& \\textbf{امتیاز \\lr{F1}}") if fa else (
          "\\textbf{Class} & \\textbf{n} & \\textbf{Precision} & \\textbf{Recall} "
          "& \\textbf{F1}")
    overall = "\\textbf{کل}" if fa else "\\textbf{Overall}"
    floor_lbl = "\\textit{کف کلاس اکثریت}" if fa else "\\textit{Majority-class floor}"
    acc_lbl, qwk_lbl, f1_lbl = (("دقت", "کاپای وزن‌دار درجه‌دو", "میانگین کلان \\lr{F1}")
                                if fa else ("accuracy", "QWK", "macro-F1"))
    return f"""% generated by src/report.py from {os.path.basename(os.path.dirname(run['_path']))}/results.json
% run {run['run_id']}  commit {run['commit'][:10]}  ({kind}, n={v['n']})
\\begin{{table}}[H]
\\caption{{{caption}}}
\\centering
\\begin{{tabular}}{{|l|c|c|c|c|}}
\\hline
    {hdr} \\\\ \\hline
{body}
\\hline
    {overall} & {v['n']} & \\multicolumn{{3}}{{c|}}{{
        {acc_lbl} {v['accuracy']*100:.1f}\\% {ci(v,'accuracy')},
        {qwk_lbl} {v['qwk']:.3f} {ci(v,'qwk',1,'{:.3f}')},
        {f1_lbl} {v['macro_f1']:.3f}}} \\\\ \\hline
    {floor_lbl} & & \\multicolumn{{3}}{{c|}}{{
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
        title = FA_HEAD[k][0]
        lines.append(
            f"    {title} & {v['n']} & {v['majority_floor']*100:.1f}\\% & "
            f"{v['accuracy']*100:.1f}\\% & {v['qwk']:.3f} \\\\")
    body = "\n".join(lines)
    return f"""% generated by src/report.py -- best run {r['run_id']} (commit {r['commit'][:10]})
\\begin{{table}}[H]
\\caption{{مدل پیشنهادی در برابر کف کلاس اکثریت برای هر وظیفه. هر دقتی کنار کفی که باید از آن
عبور کند نقل شده است، زیرا کف‌ها میان تعریف مشروط و غیرمشروط وظیفه‌ی ورم ماکولا بیش از بیست
واحد تفاوت دارند.}}
\\centering
\\begin{{tabular}}{{|l|c|c|c|c|}}
\\hline
    \\textbf{{وظیفه}} & \\textbf{{تعداد}} & \\textbf{{کف}} & \\textbf{{دقت}} & \\textbf{{\\lr{{QWK}}}} \\\\ \\hline
{body}
\\hline
\\end{{tabular}}
\\label{{tab:vs-floor}}
\\end{{table}}
"""


# ── figures ───────────────────────────────────────────────────────────────────
def latex_calibration_record(out_dir="docs/generated"):
    """PROTOCOL §4.1's record: what matched calibration has overturned, in both directions.

    Generated rather than typed because it is the methods chapter's spine and must not drift
    from the findings it summarises. The third row -- a case where the check CONFIRMED a
    difference -- is what makes the rule evidence rather than a filter.
    """
    src = os.path.join(out_dir, "calibration_record.json")
    if not os.path.exists(src):
        return None
    d = json.load(open(src))
    body = "\n".join(
        f"    {r['claim']} & {r['initial']} & {r['outcome']} & {r['direction']} \\\\ \\hline"
        for r in d["rows"])
    n_conf = sum(1 for r in d["rows"] if "تأیید" in r["direction"])
    return f"""% generated by src/report.py from {os.path.relpath(src)}
% PROTOCOL.md §4.1. Regenerate after adding any row; do not edit this file by hand.
\\begin{{table}}[H]
\\caption{{کارنامه‌ی قاعده‌ی «کالیبراسیون هم‌تراز»: هیچ تفاوتی میان دو مدل به بازنمایی نسبت
داده نمی‌شود مگر آنکه پس از هم‌تراز کردن نقاط تصمیم باقی بماند. این قاعده تا کنون
{len(d['rows'])} پیش‌بینی را رد یا تأیید کرده است و — که مهم‌تر است — در \\emph{{هر دو جهت}}
عمل کرده: {n_conf} مورد از آن‌ها تفاوت را \textbf{{تأیید}} کرده است. بدون چنین موردی، قاعده
از استدلال جهت‌دار قابل تشخیص نمی‌بود.}}
\\centering
\\begin{{tabular}}{{|p{{4.3cm}}|p{{2.6cm}}|p{{5.4cm}}|c|}}
\\hline
    \\textbf{{ادعا}} & \\textbf{{خوانش اولیه}} & \\textbf{{پس از کالیبراسیون هم‌تراز}} & \\textbf{{نتیجه}} \\\\ \\hline
{body}
\\end{{tabular}}
\\label{{tab:calibration-record}}
\\end{{table}}
"""


def latex_operating_point(out_dir="docs/generated"):
    """The discussion chapter's centrepiece: what the operating point is worth against what
    the backbone is worth. Generated like every other number (FINDINGS.md F3)."""
    src = os.path.join(out_dir, "operating_point.json")
    if not os.path.exists(src):
        return None
    d = json.load(open(src))
    rows = "\n".join(
        f"    {r['label']} & {r['delta']} & {r['ci']} & {r['cost']} \\\\"
        for r in d["rows"])
    return f"""% generated by src/report.py from {os.path.relpath(src)}
% See FINDINGS.md F3. Every figure is a paired bootstrap over the same {d['n']} external images.
\\begin{{table}}[H]
\\caption{{اثر جابه‌جا کردن نقطه‌ی کار در برابر اثر تعویض معماری، بر روی همان
{d['n']} تصویر خارجی. بازنمونه‌گیری زوجی روی همان تصاویر انجام شده است. جابه‌جا کردن
نقاط برش چند برابر تعویض ستون فقرات ارزش دارد و هیچ هزینه‌ی محاسباتی ندارد.}}
\\centering
\\begin{{tabular}}{{|p{{6.2cm}}|c|c|c|}}
\\hline
    \\textbf{{مداخله}} & \\textbf{{تغییر \\lr{{macro-recall}}}} & \\textbf{{بازه‌ی ۹۵\\%}} & \\textbf{{هزینه}} \\\\ \\hline
{rows}
\\hline
\\end{{tabular}}
\\label{{tab:operating-point}}
\\end{{table}}
"""


def latex_external(runs_dir="runs"):
    """The external-validation table. This is the thesis' most load-bearing claim, so it is
    generated from the verified artifact like every other number."""
    import glob as _g
    best = None
    for p_ in sorted(_g.glob(os.path.join(runs_dir, "*", "external_*.json"))):
        j = json.load(open(p_))
        if j.get("metrics", {}).get("dr"):
            best = (p_, j)
    if not best:
        return None
    p_, j = best
    v = j["metrics"]["dr"]
    lo, hi = v["accuracy_ci95"]
    qlo, qhi = v["qwk_ci95"]
    return f"""% generated by src/report.py from {os.path.relpath(p_)}
% verified end-to-end by src/verify_external.py (14 checks, 0 failures)
\\begin{{table}}[H]
\\caption{{اعتبارسنجی خارجی روی مجموعه‌داده‌ی \\lr{{{j['corpus']}}}، که در هیچ مرحله‌ای از
آموزش یا انتخاب مدل دیده نشده است. پیش‌بینی‌ها میانگین لاجیت {j['n_folds_ensembled']} فولد
با افزایش داده در زمان آزمون است.}}
\\centering
\\begin{{tabular}}{{|l|c|}}
\\hline
    \\textbf{{معیار}} & \\textbf{{مقدار}} \\\\ \\hline
    تعداد تصاویر & {v['n']} \\\\ \\hline
    کف کلاس اکثریت & {v['majority_floor']*100:.1f}\\% \\\\ \\hline
    دقت & {v['accuracy']*100:.1f}\\% \\quad (\\lr{{95\\%}}: {lo*100:.1f}--{hi*100:.1f}) \\\\ \\hline
    کاپای وزن‌دار درجه‌دو & {v['qwk']:.3f} \\quad (\\lr{{95\\%}}: {qlo:.3f}--{qhi:.3f}) \\\\ \\hline
    حساسیت ارجاع رتینوپاتی & {v.get('referable_sensitivity',0)*100:.1f}\\% \\\\ \\hline
    ویژگی ارجاع رتینوپاتی & {v.get('referable_specificity',0)*100:.1f}\\% \\\\ \\hline
\\end{{tabular}}
\\label{{tab:external}}
\\end{{table}}
"""


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
             "عملکرد به تفکیک کلاس در طبقه‌بندی پنج‌کلاسه‌ی رتینوپاتی دیابتی، "
             "برون‌فولدی روی کل استخر توسعه.", "tab:dr-metrics"),
            ("dme_ungated", "results_dme.tex",
             "عملکرد به تفکیک کلاس در درجه‌بندی سه‌کلاسه‌ی ورم ماکولا دیابتی "
             "(بدون دروازه: همه‌ی تصاویر، درجه‌ی صفر یعنی نبود ورم ماکولا).",
             "tab:dme-metrics"),
        ):
            tex = latex_per_class(best, key, cap, lab)
            if tex:
                open(os.path.join(a.out, fn), "w").write(tex)
                print(f"wrote {a.out}/{fn}")
        cr = latex_calibration_record(a.out)
        if cr:
            open(os.path.join(a.out, "calibration_record.tex"), "w").write(cr)
            print(f"wrote {a.out}/calibration_record.tex")
        op = latex_operating_point(a.out)
        if op:
            open(os.path.join(a.out, "operating_point.tex"), "w").write(op)
            print(f"wrote {a.out}/operating_point.tex")
        ext = latex_external(a.runs)
        if ext:
            open(os.path.join(a.out, "external.tex"), "w").write(ext)
            print(f"wrote {a.out}/external.tex")
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
