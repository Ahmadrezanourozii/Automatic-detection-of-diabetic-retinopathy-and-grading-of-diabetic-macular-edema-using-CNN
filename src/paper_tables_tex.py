"""
paper_tables_tex.py — LaTeX tables for the manuscript, generated from JSON (no hand-typed
numbers). Writes paper/sections/table_*.tex.
"""
from __future__ import annotations
import json, os

GP = "docs/generated/paper"
OUT = "paper/sections"


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def mm(v):
    return f"{v:.4f}"


def sg(v, d=4):
    s = f"{v:+.{d}f}"
    return s.replace("-", "$-$")


def ci(lo, hi, d=4):
    return f"[{sg(lo, d)}, {sg(hi, d)}]"


def cohorts():
    rows = [
        ("Development", "IDRiD", "India", "516", "3 graders (published grading)"),
        ("", "Messidor-2", "France", "1,744", "adjudicated by 3 specialists"),
        ("Pretraining; large-data models", "EyePACS 2015", "USA", "35,126", "single grader"),
        ("External", "APTOS 2019", "India", "3,662", "single grader"),
        ("", "DDR", "China, 147 hospitals", "12,522", "7 graders"),
        ("", "DeepDRiD (regular)", "China, Shanghai", "2,000 (500 patients)", "challenge release"),
        ("", "SUSTech-SYSU", "China, Shenzhen", "1,151", "international scale, released"),
        ("Reverse direction", "APTOS $\\rightarrow$ Messidor-2, IDRiD", "", "3,662 $\\rightarrow$ 2,260", "as above"),
    ]
    t = ["\\begin{table}[t]", "\\caption{\\textbf{Cohorts.} DR is graded on the international 5-grade scale",
         "in every corpus. No external image was used for training, model selection or",
         "threshold fitting.}\\label{tab:cohorts}", "\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{@{}p{2.3cm}p{3.1cm}p{2.4cm}p{2.3cm}p{2.7cm}@{}}", "\\toprule",
         "Role & Corpus & Origin & Images & Grading \\\\", "\\midrule"]
    for r in rows:
        t.append(" & ".join(r) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(OUT, "table_cohorts.tex"), "w").write("\n".join(t) + "\n")


MODELS = [
    ("E08", "DenseNet121", "448", "reference"),
    ("E09", "DenseNet121", "224", "resolution"),
    ("E10", "DenseNet121", "640 (560 effective)", "resolution"),
    ("E17NAT", "DenseNet121", "640", "native-resolution source images"),
    ("E11FULL", "EfficientNet-B3", "448", "backbone (selected model)"),
    ("E14MAC", "DenseNet121", "448", "macula-centred crop for the DME head"),
    ("E15LPFT", "DenseNet121", "448", "linear probe, then fine-tune"),
    ("E20CORAL", "DenseNet121", "448", "CORAL ordinal head"),
    ("E22CORN", "DenseNet121", "448", "CORN ordinal head"),
    ("E21CNXA", "ConvNeXt-tiny", "448", "backbone (2 of 5 folds)"),
    ("I24FULL", "RETFound ViT-L/16", "224", "foundation encoder (not in the O7 set)"),
]


def models():
    dev = load(f"{GP}/dev_summary.json") or {}
    eye = load(f"{GP}/eyepacs.json") or {}
    sys_path = os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.path.insert(0, sys_path)
    from paper_figures import MODEL_LABELS
    t = ["\\begin{table}[t]", "\\caption{\\textbf{Models.} Each development model changes one factor relative",
         "to the reference. Development QWK: out-of-fold DR QWK under the matched decision rule",
         "(fold-wise cross-fitted cut-points). Repository codes are given for traceability.}\\label{tab:models}",
         "\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}p{4.3cm}p{2.3cm}p{1.6cm}p{3.4cm}r@{}}", "\\toprule",
         "Model (code) & Backbone & Pixels & Factor changed & Dev.\\ QWK \\\\", "\\midrule"]
    for code, bb, px, fac in MODELS:
        q = dev.get(code, {}).get("qwk_matched")
        t.append(f"{MODEL_LABELS.get(code, code)} ({code}) & {bb} & {px} & {fac} & "
                 + ("--" if q is None else f"{q:.4f}") + " \\\\")
    for code, bb in (("EYEDN", "DenseNet121"), ("EYEEB3", "EfficientNet-B3")):
        q = eye.get("models", {}).get(code, {}).get("dev", {}).get("qwk_matched")
        t.append(f"{MODEL_LABELS[code]} ({code}) & {bb} & 448 & trained on EyePACS & "
                 + ("--" if q is None else f"{q:.4f}") + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(OUT, "table_models.tex"), "w").write("\n".join(t) + "\n")


def record():
    r = load(f"{GP}/record.json")
    iv = {x["label"]: x for x in load(f"{GP}/interventions.json")}
    if not r:
        return
    bb = r["backbone_aptos"]
    rows = []
    q = lambda rule: bb[rule]["qwk"]
    rows.append(("EfficientNet-B3 vs DenseNet121, APTOS QWK", sg(q("R1")["diff"]) + " " + ci(q("R1")["lo"], q("R1")["hi"]),
                 sg(q("R4")["diff"]) + " " + ci(q("R4")["lo"], q("R4")["hi"]), "advantage dissolves"))
    for k, name in ((2, "moderate"), (3, "severe"), (4, "proliferative")):
        a, b = bb["R1"]["recall"][k], bb["R4"]["recall"][k]
        rows.append((f"\\quad recall, {name} (points)",
                     f"{sg(100*a['diff'],1)}{'$^\\ast$' if a['significant'] else ''}",
                     f"{sg(100*b['diff'],1)}{'$^\\ast$' if b['significant'] else ''}",
                     "sign changes" if (a['diff'] * b['diff'] < 0 and (a['significant'] or b['significant']))
                     else "persists" if (a['significant'] and b['significant'])
                     else "significance changes" if (a['significant'] != b['significant'])
                     else "n.s. at both"))
    lp = r["lpft_dev"]
    rows.append(("LP-FT vs fine-tuning, development accuracy (points)",
                 sg(100*lp["shipped"]["accuracy"]["diff"], 2) + " " + ci(100*lp["shipped"]["accuracy"]["lo"], 100*lp["shipped"]["accuracy"]["hi"], 2),
                 sg(100*lp["matched"]["accuracy"]["diff"], 2) + " " + ci(100*lp["matched"]["accuracy"]["lo"], 100*lp["matched"]["accuracy"]["hi"], 2),
                 "reverses"))
    e = iv["EfficientNet-B3 backbone"]
    rows.append(("EfficientNet-B3 vs DenseNet121, development QWK",
                 sg(e["dr_default"]["diff"]) + " " + ci(e["dr_default"]["lo"], e["dr_default"]["hi"]),
                 sg(e["dr_matched"]["diff"]) + " " + ci(e["dr_matched"]["lo"], e["dr_matched"]["hi"]),
                 "survives"))
    t = ["\\begin{table}[t]",
         "\\caption{\\textbf{The matched decision rule changes model comparisons.} Differences (first minus",
         "second) with 95\\% paired-bootstrap intervals at the shipped rule and with both models'",
         "cut-points refitted by the same procedure (external rows: R4, two-fold cross-fitted on",
         "APTOS labels; development rows: fold-wise cross-fitted). $^\\ast$Interval excludes zero.",
         "A per-grade verdict that depends on the cross-fitting procedure is reported in Supplementary",
         "Note~3.}\\label{tab:record}",
         "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}p{4.0cm}p{2.7cm}p{2.7cm}l@{}}", "\\toprule",
         "Comparison & Shipped rule & Matched decision rule & Verdict \\\\", "\\midrule"]
    for x in rows:
        t.append(" & ".join(x) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(OUT, "table_record.tex"), "w").write("\n".join(t) + "\n")


def orderdr():
    d = load(f"{GP}/orderdr_reanalysis.json")
    if not d:
        return
    agg = d["seed_mean_sd"]
    rules = [("R1", "R1 argmax"), ("R2", "R2 val.-fitted"), ("R4", "R4 ext.\\ QWK"),
             ("R4m", "R4m ext.\\ recall")]
    t = ["\\begin{table}[t]",
         "\\caption{\\textbf{Decision-rule reanalysis of a published model family (ORDER-DR,",
         "APTOS$\\rightarrow$Messidor-2).} Mean $\\pm$ s.d. over the authors' three seeds, computed",
         f"from their released image-level probabilities on {d['n_external']:,} Messidor-2 images",
         f"({d['n_mild']} mild). For ORDER-DR the R2 row uses the authors' own fusion and cut-points",
         "and reproduces their reported result. R4/R4m use external labels (two-fold cross-fitting)",
         "and are diagnostic bounds, not performance claims.}\\label{tab:orderdr}",
         "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}llccccc@{}}", "\\toprule",
         "Model & Rule & QWK & Mild rec.\\ (\\%) & Mild F1 & Sens.\\ (\\%) & Spec.\\ (\\%) \\\\",
         "\\midrule"]
    for name in ("CB-384", "LORS-384", "Swin-T", "ORDER-DR"):
        for i, (r, lab) in enumerate(rules):
            v = agg[name][r]
            t.append(f"{name if i == 0 else ''} & {lab} & {v['qwk'][0]:.3f} $\\pm$ {v['qwk'][1]:.3f} & "
                     f"{100*v['mild_recall'][0]:.1f} & {v['mild_f1'][0]:.3f} & "
                     f"{100*v['ref_sens'][0]:.1f} & {100*v['ref_spec'][0]:.1f} \\\\")
        if name != "ORDER-DR":
            t.append("\\addlinespace")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(OUT, "table_orderdr.tex"), "w").write("\n".join(t) + "\n")


def interventions():
    rows = load(f"{GP}/interventions.json")
    t = ["\\begin{table}[h]",
         "\\caption{\\textbf{Pre-registered interventions on the model.} Development DR QWK",
         "difference (intervention minus control) at shipped and matched calibration with 95\\%",
         "paired-bootstrap intervals; the macula crop was registered on the DME head and the",
         "ensemble on the external set.}\\label{tab:interv}", "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{@{}lllll@{}}", "\\toprule",
         "Intervention & Control & Folds & Matched difference & Verdict \\\\", "\\midrule"]
    for r in rows:
        t.append(f"{r['label']} & {r['control']} & {r['folds']} & {sg(r['diff'])} {ci(r['lo'], r['hi'])} & {r['verdict']} \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(OUT, "table_interventions.tex"), "w").write("\n".join(t) + "\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (cohorts, models, record, orderdr, interventions):
        f()
    print("tables written")
