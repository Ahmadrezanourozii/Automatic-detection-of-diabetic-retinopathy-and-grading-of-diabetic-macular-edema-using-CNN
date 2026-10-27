"""
paper_supp_tex.py — Supplementary Information tables, generated from JSON (no hand-typed
numbers). Writes paper/supp/table_s*.tex; paper/supplementary.tex inputs them. A table whose
source has not been produced yet is written as a one-line placeholder so the document still
compiles and the gap is visible.
"""
from __future__ import annotations
import json, os
import numpy as np

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
         "control. Dev QWK: development out-of-fold DR QWK under the matched decision rule (fold-wise",
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
    write("table_configs.tex", t)


def s3_interventions():
    rows = load(f"{GP}/interventions.json")
    t = ["\\begin{table}[h]",
         "\\caption{\\textbf{Pre-specified interventions, shipped and matched decision rule.}",
         "Difference (intervention minus control) in development DR QWK with 95\\% paired-bootstrap",
         "intervals (DME QWK for the macula crop; APTOS DR QWK for the ensemble).}\\label{tab:s3}",
         "\\centering\\scriptsize", "\\begin{tabular}{@{}lllll@{}}", "\\toprule",
         "Intervention & Control & Folds & Matched difference [95\\% CI] & Verdict \\\\", "\\midrule"]
    for r in rows or []:
        t.append(f"{r['label']} & {r['control']} & {r['folds']} & {sg(r['diff'])} "
                 f"[{sg(r['lo'])}, {sg(r['hi'])}] & {r['verdict']} \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_interv_supp.tex", t)


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
    for c in ("APTOS", "DDR", "DeepDRiD", "SUSTech"):
        write(f"table_rules_{c.lower()}.tex", rule_table(c, f"tab:rules_{c.lower()}"))


def s6_seeds():
    d = load(f"{GP}/seeds.json")
    cap = ("\\textbf{Seed variation of the reference configuration (O1--O3).} Five independent"
           " training seeds of E08 (identical data, split and recipe).")
    if not d:
        write("table_seeds.tex", placeholder("seeds.json", cap, "tab:s6"))
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
    write("table_seeds.tex", t)


def s7_reverse():
    d = load(f"{GP}/reverse.json")
    cap = ("\\textbf{Reverse direction (O4--O6).} The reference recipe trained on APTOS 2019"
           " and evaluated on Messidor-2 and IDRiD.")
    if not d:
        write("table_reverse.tex", placeholder("reverse.json", cap, "tab:s7"))
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
    write("table_reverse.tex", t)


def s8_overlap():
    d = load(f"{GP}/overlap_audit.json")
    cap = ("\\textbf{Content-overlap audit.} Pairs of images across corpora whose 256-bit"
           " difference hashes lie within the stated Hamming distance.")
    if not d:
        write("table_overlap.tex", placeholder("overlap_audit.json", cap, "tab:s8"))
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
    write("table_overlap.tex", t)


def s_provenance():
    d = load(f"{GP}/provenance.json")
    cap = ("\\textbf{Analysis plans and the runs they cover.} Commit identifiers and times (UTC) are from "
           "the public repository history; launch times are from the Kaggle kernel listing archived in "
           "docs/generated/paper/kaggle\\_runs.csv. Every run listed was launched after the commit of its "
           "plan (for the post-hoc checks, of its code) and pinned the code commit it ran. The two outcome plans are the analysis-plan files in docs/ "
           "of the repository; the second-stage plan was written after an internal pre-submission review "
           "of an earlier draft.")
    if not d:
        write("table_provenance.tex", placeholder("provenance.json", cap, "tab:provenance"))
        return
    t = ["\\begin{table}[h]", f"\\caption{{{cap}}}\\label{{tab:provenance}}", "\\centering\\scriptsize",
         "\\begin{tabular}{@{}p{5.6cm}lll@{}}", "\\toprule",
         "Plan & Commit & Committed (UTC) & First run launched (UTC) \\\\", "\\midrule"]
    for r in d:
        plan = r["plan"].replace("_", "\\_").replace("O1-O9", "O1--O9").replace("A1-E2", "A1--E2")
        t.append(f"{plan} & \\texttt{{{r['commit']}}} & {r['commit_utc']} & {r['first_launch_utc'] or '--'} \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_provenance.tex", t)


def s_clinical():
    d = load(f"{GP}/revision.json")
    cap = ("\\textbf{What the operating point means for a screening programme.} Referable patients missed "
           "and false referrals per 10\\,000 people screened, and predictive values, for the selected model "
           "at its shipped rule (R1) and with cut-points refitted on local labels (R4), at three referable "
           "prevalences. Sensitivity and specificity are those measured on each external set.")
    if not d:
        write("table_clinical.tex", placeholder("revision.json", cap, "tab:clinical"))
        return
    t = ["\\begin{table}[h]", f"\\caption{{{cap}}}\\label{{tab:clinical}}", "\\centering\\scriptsize",
         "\\begin{tabular}{@{}llrrrrrrr@{}}", "\\toprule",
         "Corpus & Rule & Sens. & Spec. & Prevalence & Missed & False referrals & PPV & NPV \\\\",
         " & & (\\%) & (\\%) & (\\%) & /10\\,000 & /10\\,000 & (\\%) & (\\%) \\\\", "\\midrule"]
    for c, C in d["corpora"].items():
        r = C["models"].get("E11FULL")
        if not r:
            continue
        for rule in ("R1", "R4"):
            se, sp = r["rules"][rule]["ref_sens"], r["rules"][rule]["ref_spec"]
            for k, prev in enumerate((0.05, 0.10, 0.20)):
                ppv = se * prev / (se * prev + (1 - sp) * (1 - prev))
                npv = sp * (1 - prev) / (sp * (1 - prev) + (1 - se) * prev)
                t.append(f"{c if rule == 'R1' and k == 0 else ''} & {rule if k == 0 else ''} & "
                         f"{100*se:.1f} & {100*sp:.1f} & {100*prev:.0f} & {(1-se)*prev*1e4:,.0f} & "
                         f"{(1-sp)*(1-prev)*1e4:,.0f} & {100*ppv:.1f} & {100*npv:.1f} \\\\")
        t.append("\\addlinespace")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_clinical.tex", t)


def s_eyepacs():
    d = load(f"{GP}/eyepacs.json")
    cap = ("\\textbf{Models fine-tuned on EyePACS (outcome E2).} Trained on four folds of a patient-grouped "
           "EyePACS split (\\pn{eye.eb3.trainn} images); the fifth fold (\\pn{eye.eb3.devn} single-graded images) "
           "selected the checkpoint and is the development set. Referral area under the curve, mean "
           "predicted-minus-true grade and referable sensitivity and specificity at the shipped rule (R1), "
           "sensitivity with cut-points refitted on external labels (R4), and with a threshold fitted on the "
           "held-out EyePACS fold for 85\\% sensitivity. The development row shows the referral area under the "
           "curve on the held-out fold itself.")
    if not d:
        write("table_eyepacs.tex", placeholder("eyepacs.json", cap, "tab:eyepacs"))
        return
    t = ["\\begin{table}[h]", f"\\caption{{{cap}}}\\label{{tab:eyepacs}}", "\\centering\\scriptsize",
         "\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
         "Model (EyePACS-trained) & Corpus & AUC & Pred.$-$true & Sens.\\ R1 & Spec.\\ R1 & Sens.\\ R4 & Sens.\\ thr.\\ 85 \\\\",
         "\\midrule"]
    for m, v in d["models"].items():
        t.append(f"{v['label'].split(',')[0]} & EyePACS fold 0 (dev.) & {v['dev']['auroc']:.3f} & -- & -- & -- & -- & -- \\\\")
        for j, (c, r) in enumerate(v["external"].items()):
            t.append(f" & {c} & {r['auroc']:.3f} & {sg(r['mean_R1'], 2)} & "
                     f"{100*r['sens_R1']:.1f} & {100*r['spec_R1']:.1f} & {100*r['sens_R4']:.1f} & {100*r['sens_tr85']:.1f} \\\\")
        t.append("\\addlinespace")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_eyepacs.tex", t)


def s_mechanism():
    d = load(f"{GP}/mechanism.json")
    cap = ("\\textbf{Image degradation and image statistics (outcomes A2, A3).} Top: change relative to the "
           "undegraded re-scoring of the development pool (out of fold) in mean predicted-minus-true grade "
           "and in referable sensitivity at the shipped rule. Bottom: median image statistics per corpus "
           "(interquartile range in the repository).")
    if not d:
        write("table_mechanism.tex", placeholder("mechanism.json", cap, "tab:mechanism"))
        return
    t = ["\\begin{table}[h]", f"\\caption{{{cap}}}\\label{{tab:mechanism}}", "\\centering\\scriptsize",
         "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
         "Degradation & \\multicolumn{2}{c}{Reference model} & \\multicolumn{2}{c}{Selected model} \\\\",
         " & $\\Delta$ pred.$-$true & $\\Delta$ sens.\\ (pts) & $\\Delta$ pred.$-$true & $\\Delta$ sens.\\ (pts) \\\\",
         "\\midrule"]
    for cond in d["conditions"]:
        a = d["models"].get("E08", {}).get(cond, {}); b = d["models"].get("E11FULL", {}).get(cond, {})
        f = lambda r, k, sc: "--" if k not in r else sg(sc * r[k], 2 if sc == 1 else 1)
        name = d['labels'][cond].replace('%', r'\%')
        t.append(f"{name} & {f(a, 'dmean', 1)} & {f(a, 'dsens', 100)} & {f(b, 'dmean', 1)} & {f(b, 'dsens', 100)} \\\\")
    t += ["\\midrule", "Corpus & Long side (px) & JPEG quality & Brightness & Laplacian var. \\\\", "\\midrule"]
    for c, v in d.get("stats", {}).items():
        g = lambda k, fmt: "--" if not v.get(k) else fmt.format(v[k]["median"])
        t.append(f"{c} & {g('resolution', '{:,.0f}')} & {g('jpeg_quality', '{:.0f}')} & {g('brightness', '{:.0f}')} & {g('laplacian_var', '{:.0f}')} \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_mechanism.tex", t)


def s_samplesize():
    d = load(f"{GP}/posthoc.json")
    cap = ("\\textbf{How many locally labelled cases a referral threshold needs (post hoc).} The threshold "
           "is set to the largest value giving at least 85\\% sensitivity on the referable cases of a "
           "labelled sample; harm is a population sensitivity below 80\\%. Top: by the number $k$ of "
           "referable cases in the sample. The closed form is distribution-free (with continuous scores the "
           "population sensitivity of the cut at the $j$-th largest of $k$ referable scores, "
           "$j=\\lceil0.85k\\rceil$, follows a Beta$(j,k-j+1)$ distribution); the simulated range covers the "
           "selected and reference models on the four external sets (2\\,000 draws each, sensitivity measured "
           "on the corpus's referable images not drawn; $k$ at most a quarter of them). The last two columns "
           "give the sample sensitivity at which the cut must be set for the risk to stay at or below 5\\% or "
           "10\\%. Bottom: by the number of consecutive images labelled at a given referable prevalence "
           "(closed form, binomial number of referable cases; no referable case counts as harm).")
    if not d:
        write("table_samplesize.tex", placeholder("posthoc.json", cap, "tab:samplesize"))
        return
    ex = d["exact"]
    sims = {}
    for C in d["corpora"].values():
        for v in C["models"].values():
            for k, r in v.get("by_cases", {}).items():
                sims.setdefault(k, []).append(r["p_harm"])
    pc = lambda v: "--" if v is None else f"{100 * v:.1f}"
    t = ["\\begin{table}[h]", f"\\caption{{{cap}}}\\label{{tab:samplesize}}", "\\centering\\scriptsize",
         "\\begin{tabular}{@{}rrrrr@{}}", "\\toprule",
         "Referable cases $k$ & Harm, closed form (\\%) & Harm, simulated (\\%) & "
         "Sample sens.\\ for $\\leq$5\\% & for $\\leq$10\\% \\\\", "\\midrule"]
    for k in (20, 40, 60, 80, 100, 120, 160, 200, 300):
        sk = sims.get(str(k))
        rq = ex["required_sample_sens"].get(str(k), {})
        t.append(f"{k} & {pc(ex['harm_by_k'].get(str(k)))} & "
                 + ("--" if not sk else f"{100 * min(sk):.1f}--{100 * max(sk):.1f}")
                 + f" & {pc(rq.get('0.05'))} & {pc(rq.get('0.10'))} \\\\")
    t += ["\\midrule", "Images labelled & \\multicolumn{4}{l}{Harm (\\%) at a referable prevalence of 5\\% / 10\\% / 20\\%} \\\\",
          "\\midrule"]
    for n in (100, 200, 500, 1000, 2000):
        h = [ex["harm_by_prevalence"][f"{p:.2f}_{n}"] for p in (0.05, 0.10, 0.20)]
        t.append(f"{n:,} & \\multicolumn{{4}}{{l}}{{{' / '.join(pc(x) for x in h)}}} \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_samplesize.tex", t)


def s_labelfree():
    d = load(f"{GP}/posthoc.json")
    cap = ("\\textbf{Label-free prior adaptation against the decoding it uses (post hoc).} R5 (EM prior "
           "adaptation) and R5b (black-box shift estimation) decode the adapted class posteriors by argmax, so "
           "the like-for-like reference is R0a, the argmax of the unadapted posteriors, not the 0.5 rule R1. "
           "Median (range) over the ten development models. Mild recall: median over models.")
    if not d:
        write("table_labelfree.tex", placeholder("posthoc.json", cap, "tab:labelfree"))
        return
    t = ["\\begin{table}[h]", f"\\caption{{{cap}}}\\label{{tab:labelfree}}", "\\centering\\scriptsize",
         "\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
         "Corpus & Quantity & R1 & R0a & R5 & R5b & R4 (diagnostic) \\\\", "\\midrule"]
    names = {"APTOS": "APTOS", "DDR": "DDR", "DeepDRiD": "DeepDRiD", "SUSTech": "SUSTech-SYSU"}
    for c, C in d["corpora"].items():
        r = [v["R5_vs_R0a"] for v in C["models"].values()]
        rules = ("R1", "R0a", "R5", "R5b", "R4")
        med = lambda key, rule, sc=100, dd=1: f"{sc * np.median([x[key][rule] for x in r]):.{dd}f}"
        rng = lambda key, rule: f"{100 * min(x[key][rule] for x in r):.1f}--{100 * max(x[key][rule] for x in r):.1f}"
        t.append(f"{names[c]} & Referable sensitivity (\\%) & " + " & ".join(f"{med('sens', q)} ({rng('sens', q)})" for q in rules) + " \\\\")
        t.append(" & QWK & " + " & ".join(med("qwk", q, 1, 3) for q in rules) + " \\\\")
        t.append(" & Mild recall (\\%) & " + " & ".join(med("mild_recall", q) for q in rules) + " \\\\")
        t.append("\\addlinespace")
    t[-1:] = []
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_labelfree.tex", ["\\begin{table}[h]"] + t[1:3] + ["\\resizebox{\\textwidth}{!}{%"] + t[3:-1] + ["}", "\\end{table}"])


def s_spectrum():
    d = load(f"{GP}/posthoc.json")
    acq = load(f"{GP}/acquisition.json")
    cap = ("\\textbf{Case mix and discrimination of each population (post hoc).} Areas under the ROC "
           "curve, range over the ten development models. Referral AUC standardised to the development grade "
           "mix weights each image by the development share of its grade over the corpus share, within "
           "referable and within non-referable images. Adjacent-grade AUCs use $P(y>0)$ for no DR against mild "
           "and $P(y>1)$ for the other pairs."
           + (" Acquisition (post hoc, run X6ACQ): the largest area under the curve, taken in whichever direction "
              "separates better, with which any single acquisition statistic of the image (resolution, file "
              "size, estimated JPEG quality, share of the frame inside the retinal mask, brightness, contrast, "
              "sharpness) separates referable from non-referable images, and any DR from no DR, without a "
              "model; 0.5 is no separation. Contrast and sharpness also rise with lesions." if acq else ""))
    if not d:
        write("table_spectrum.tex", placeholder("posthoc.json", cap, "tab:spectrum"))
        return
    rngf = lambda v: f"{min(v):.3f}--{max(v):.3f}"
    cols = "@{}lrrrrrr" + ("rr" if acq else "") + "@{}"
    head = ("Population & Mild (\\%) & Referral AUC & Standardised & No DR vs mild & Mild vs moderate & "
            "No DR vs moderate" + (" & Acq.\\ referable & Acq.\\ any DR" if acq else "") + " \\\\")
    t = ["\\begin{table}[h]", f"\\caption{{{cap}}}\\label{{tab:spectrum}}", "\\centering\\scriptsize",
         "\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{{cols}}}", "\\toprule", head, "\\midrule"]
    names = {"dev": "Development", "APTOS": "APTOS", "DDR": "DDR", "DeepDRiD": "DeepDRiD", "SUSTech": "SUSTech-SYSU"}
    acq_key = {"dev": None, "APTOS": "APTOS", "DDR": "DDR", "DeepDRiD": "DeepDRiD", "SUSTech": "SUSTech"}
    dev_y = None
    rows = [("dev", list(d["dev"].values()), None)] + [(c, list(C["models"].values()), C) for c, C in d["corpora"].items()]
    for key, ms, C in rows:
        sp = lambda k: [m["spectrum"][k] for m in ms]
        if C is None:
            g = (load(f"{GP}/revision.json") or {}).get("dev", {}).get("E08", {}).get("grade_dist")
            pm = "--" if not g else f"{100 * g[1] / sum(g):.1f}"
        else:
            pm = f"{100 * C['grade_dist'][1] / C['n']:.1f}"
        line = (f"{names[key]} & {pm} & {rngf(sp('auc_ref'))} & "
                + ("--" if key == "dev" else rngf(sp("auc_ref_devmix")))
                + f" & {rngf(sp('auc_0v1'))} & {rngf(sp('auc_1v2'))} & {rngf(sp('auc_0v2'))}")
        if acq:
            a = acq.get(acq_key[key] or "", None) if key != "dev" else acq.get("dev")
            line += (" & --- & ---" if not a else f" & {a['max_ref']:.2f} & {a['max_any']:.2f}")
        t.append(line + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    write("table_spectrum.tex", t)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (s_provenance, s2_configs, s3_interventions, s_eyepacs, s_mechanism, s_clinical,
              s4_s5_rules, s6_seeds, s7_reverse, s8_overlap, s_samplesize, s_labelfree, s_spectrum):
        f()
    print("supplementary tables written to", OUT)
