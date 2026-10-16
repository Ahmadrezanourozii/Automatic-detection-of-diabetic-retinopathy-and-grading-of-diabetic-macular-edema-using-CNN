"""
paper_supp_tex.py — Supplementary Information tables, generated from JSON (no hand-typed
numbers). Writes paper/supp/table_s*.tex; paper/supplementary.tex inputs them. A table whose
source has not been produced yet is written as a one-line placeholder so the document still
compiles and the gap is visible.
"""
from __future__ import annotations
import json, os

GP = "docs/generated/paper"
OUT = "paper/supp"
RULES = ("R1", "R2", "R3", "R4", "R4m")

# one factor changed per configuration (Methods); dev QWK is read from the predictions
CONFIGS = [
    ("E08", "--", "DenseNet121", "cumulative", "448", "ImageNet + EyePACS (5 ep)", "40", "reference"),
    ("E09", "E08", "DenseNet121", "cumulative", "224", "ImageNet + EyePACS", "40", "resolution"),
    ("E10", "E08", "DenseNet121", "cumulative", "640 (560 effective)", "ImageNet + EyePACS", "40", "resolution"),
    ("E17NAT", "E10", "DenseNet121", "cumulative", "640", "ImageNet + EyePACS", "40", "native-resolution source files"),
    ("E11FULL", "E10", "EfficientNet-B3", "cumulative", "448", "ImageNet + EyePACS", "30", "backbone (selected model)"),
    ("E14MAC", "E08", "DenseNet121", "cumulative", "448 + macula crop", "ImageNet + EyePACS", "40", "fovea-centred crop for DME head"),
    ("E15LPFT", "E08", "DenseNet121", "cumulative", "448", "ImageNet + EyePACS", "40 (LP 5)", "linear probe then fine-tune"),
    ("E20CORAL", "E08", "DenseNet121", "CORAL", "448", "ImageNet + EyePACS", "40", "ordinal head"),
    ("E22CORN", "E08", "DenseNet121", "CORN", "448", "ImageNet + EyePACS", "40", "ordinal head"),
    ("E21CNXA", "E08", "ConvNeXt-tiny", "cumulative", "448", "ImageNet + EyePACS", "40", "backbone (2 folds)"),
    ("I24FULL", "E09", "RETFound ViT-L/16", "cumulative", "224", "RETFound (colour fundus)", "30 (LP 5)", "foundation encoder"),
    ("S1E08--S4E08", "E08", "DenseNet121", "cumulative", "448", "ImageNet + EyePACS", "40", "seed only"),
    ("R1APTOS", "E08", "DenseNet121", "cumulative", "448", "ImageNet + EyePACS", "40", "trained on APTOS (reverse)"),
]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def sg(v, d=4):
    return f"{v:+.{d}f}".replace("-", "$-$")


def placeholder(name, caption, label):
    return ["\\begin{table}[h]", f"\\caption{{{caption}}}\\label{{{label}}}", "\\centering",
            f"\\emph{{[{name.replace('_', '\\_')}: analysis output not yet available.]}}", "\\end{table}"]


def write(fname, lines):
    open(os.path.join(OUT, fname), "w").write("\n".join(lines) + "\n")


def s2_configs():
    reg = load("results/predictions/registry.json") or {}
    dev = load(f"{GP}/dev_summary.json") or {}
    t = ["\\begin{table}[h]",
         "\\caption{\\textbf{Configurations.} Each configuration changes one factor relative to its",
         "control. Dev QWK: development out-of-fold DR QWK at matched calibration (fold-wise",
         "cross-fitted cut-points); folds: development folds trained.}\\label{tab:s2}",
         "\\centering\\scriptsize", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{@{}lllllllll@{}}", "\\toprule",
         "Model & Control & Backbone & DR head & Pixels & Pretraining & Epochs & Factor & Dev QWK \\\\",
         "\\midrule"]
    for m, c, bb, hd, px, pre, ep, fac in CONFIGS:
        q = dev.get(m, {}).get("qwk_matched")
        folds = reg.get(m, {}).get("oof_folds")
        qs = "--" if q is None else f"{q:.4f}" + ("" if not folds or len(folds) == 5 else f" ({len(folds)} f)")
        t.append(" & ".join([m, c, bb, hd, px, pre, ep, fac, qs]) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    write("table_s2.tex", t)


def s3_interventions():
    rows = load(f"{GP}/interventions.json")
    t = ["\\begin{table}[h]",
         "\\caption{\\textbf{Pre-registered interventions, shipped and matched calibration.}",
         "Difference (intervention minus control) in development DR QWK with 95\\% paired-bootstrap",
         "intervals (DME QWK for the macula crop; APTOS DR QWK for the ensemble).}\\label{tab:s3}",
         "\\centering\\scriptsize", "\\begin{tabular}{@{}lllll@{}}", "\\toprule",
         "Intervention & Control & Folds & Matched difference [95\\% CI] & Verdict \\\\", "\\midrule"]
    for r in rows or []:
        t.append(f"{r['label']} & {r['control']} & {r['folds']} & {sg(r['diff'])} "
                 f"[{sg(r['lo'])}, {sg(r['hi'])}] & {r['verdict']} \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_s3.tex", t)


def rule_table(corpus, label):
    d = load(f"{GP}/phase0_{corpus.lower()}.json")
    cap = (f"\\textbf{{Full decision-rule matrix, {corpus}.}} QWK, accuracy, per-grade recall"
           " (\\%) and referable-DR sensitivity/specificity (\\%) of every model under every rule."
           " R4/R4m are fitted on the external labels by two-fold cross-fitting and are"
           " diagnostic bounds, not performance.")
    if not d:
        return placeholder(f"phase0_{corpus.lower()}.json", cap, label)
    t = ["\\begin{table}[h]", f"\\caption{{{cap}}}\\label{{{label}}}", "\\centering\\scriptsize",
         "\\begin{tabular}{@{}llrrrrrrrrr@{}}", "\\toprule",
         "Model & Rule & QWK & Acc. & No DR & Mild & Mod. & Sev. & PDR & Sens. & Spec. \\\\",
         "\\midrule"]
    for i, m in enumerate(d["models"]):
        for j, r in enumerate(RULES):
            b = d["rule_matrix"][m][r]
            rec = " & ".join(f"{100*x:.1f}" for x in b["recall"])
            t.append(f"{m if j == 0 else ''} & {r} & {b['qwk']:.3f} & {100*b['accuracy']:.1f} & "
                     f"{rec} & {100*b['ref_sens']:.1f} & {100*b['ref_spec']:.1f} \\\\")
        if i < len(d["models"]) - 1:
            t.append("\\addlinespace")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return t


def s4_s5_rules():
    write("table_s4.tex", rule_table("APTOS", "tab:s4"))
    write("table_s5.tex", rule_table("DDR", "tab:s5"))


def s6_seeds():
    d = load(f"{GP}/seeds.json")
    cap = ("\\textbf{Seed variation of the reference configuration (O1--O3).} Five independent"
           " training seeds of E08 (identical data, split and recipe).")
    if not d:
        write("table_s6.tex", placeholder("seeds.json", cap, "tab:s6"))
        return
    t = ["\\begin{table}[h]", f"\\caption{{{cap}}}\\label{{tab:s6}}", "\\centering\\scriptsize",
         "\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
         "Seed & Dev QWK (matched) & APTOS QWK (R1) & APTOS mild R1 (\\%) & APTOS mild R4m (\\%) & DDR QWK (R1) \\\\",
         "\\midrule"]
    f = lambda v, sc=1, k=3: "--" if v is None else f"{sc*v:.{k}f}"
    for m in d["seeds"]:
        dv = d["dev"].get(m, {})
        ap = d.get("APTOS", {}).get("per_seed", {}).get(m, {})
        dd = d.get("DDR", {}).get("per_seed", {}).get(m, {})
        g = lambda c, r, key: (c[r][key] if r in c else None)
        t.append(f"{m} & {f(dv.get('qwk_matched'), 1, 4)} & {f(g(ap, 'R1', 'qwk'))} & "
                 f"{f(ap['R1']['recall'][1] if ap else None, 100, 1)} & "
                 f"{f(ap['R4m']['recall'][1] if ap else None, 100, 1)} & {f(g(dd, 'R1', 'qwk'))} \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_s6.tex", t)


def s7_reverse():
    d = load(f"{GP}/reverse.json")
    cap = ("\\textbf{Reverse direction (O4--O6).} The reference recipe trained on APTOS 2019"
           " and evaluated on Messidor-2 and IDRiD.")
    if not d:
        write("table_s7.tex", placeholder("reverse.json", cap, "tab:s7"))
        return
    t = ["\\begin{table}[h]", f"\\caption{{{cap}}}\\label{{tab:s7}}", "\\centering\\scriptsize",
         "\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
         "Corpus & Rule & QWK & Mild recall (\\%) & Mild F1 & Sens. (\\%) & Spec. (\\%) & Pred.$-$true grade \\\\",
         "\\midrule"]
    for c in ("Messidor-2", "IDRiD"):
        if c not in d:
            continue
        for j, r in enumerate(RULES):
            b = d[c]["rules"][r]
            t.append(f"{c if j == 0 else ''} & {r} & {b['qwk']:.3f} & {100*b['recall'][1]:.1f} & "
                     f"{b['mild_f1']:.3f} & {100*b['ref_sens']:.1f} & {100*b['ref_spec']:.1f} & "
                     f"{sg(b['mean_pred_minus_true'], 2)} \\\\")
        t.append("\\addlinespace")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_s7.tex", t)


def s8_overlap():
    d = load(f"{GP}/overlap_audit.json")
    cap = ("\\textbf{Content-overlap audit.} Pairs of images across corpora whose 256-bit"
           " difference hashes lie within the stated Hamming distance.")
    if not d:
        write("table_s8.tex", placeholder("overlap_audit.json", cap, "tab:s8"))
        return
    t = ["\\begin{table}[h]", f"\\caption{{{cap}}}\\label{{tab:s8}}", "\\centering\\scriptsize",
         "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
         "Corpus pair & Images & Pairs $\\leq$16 bits & $\\leq$8 bits & 0 bits \\\\", "\\midrule"]
    names = {"dev": "Development pool", "APTOS": "APTOS", "DDR": "DDR", "EyePACS": "EyePACS"}
    for tag, v in d["pairs"].items():
        a, _, b = tag.partition("_vs_")
        if b:
            # EyePACS_vs_X compares X against EyePACS
            n = f"{d['n'][a]:,} / {d['n'][b]:,}"
            lab = f"{names[a]} vs {names[b]}"
        else:
            a = tag.replace("_within", "")
            n = f"{d['n'][a]:,}"
            lab = f"within {names[a]}"
        t.append(f"{lab} & {n} & {v['n_pairs']:,} & {v['n_le8']:,} & "
                 f"{v['hist'][0] if 'hist' in v else '--'} \\\\")
    t.append(f"\\multicolumn{{5}}{{@{{}}l}}{{Threshold {d['threshold_bits']} bits; development pairs "
             f"split across folds: {len([p for p in d.get('dev_within_cross_fold_examples', []) if p[2] <= 8]):,}}} \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_s8.tex", t)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (s2_configs, s3_interventions, s4_s5_rules, s6_seeds, s7_reverse, s8_overlap):
        f()
    print("supplementary tables written to", OUT)
