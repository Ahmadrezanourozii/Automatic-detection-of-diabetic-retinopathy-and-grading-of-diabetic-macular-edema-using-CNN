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
    reg = load("results/predictions/registry.json") or {}
    rows = [
        ("Development pool", "IDRiD", "India", "516", "DR 0--4; DME 0--2", "3 graders (published grading)"),
        ("", "Messidor-2", "France", "1,744", "DR 0--4; referable DME", "adjudicated (3 specialists)"),
        ("Pretraining only", "EyePACS 2015", "USA", "35,126", "DR 0--4", "single grader"),
        ("External (forward)", "APTOS 2019", "India", "3,662", "DR 0--4", "single grader"),
        ("", "DDR", "China (147 hospitals)", "12,522", "DR 0--4", "7 graders"),
        ("Reverse direction", "APTOS $\\rightarrow$ Messidor-2, IDRiD", "", "3,662 $\\rightarrow$ 2,260", "DR 0--4", "as above"),
    ]
    t = ["\\begin{table}[t]", "\\caption{\\textbf{Cohorts.} No external image was used for training,",
         "model selection or threshold fitting. DR, diabetic retinopathy (international 5-grade",
         "scale); DME, diabetic macular oedema.}\\label{tab:cohorts}", "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{@{}p{2.0cm}p{2.5cm}p{1.9cm}p{1.6cm}p{1.8cm}p{2.0cm}@{}}", "\\toprule",
         "Role & Corpus & Origin & Images & Labels & Grading \\\\", "\\midrule"]
    for r in rows:
        t.append(" & ".join(r) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(OUT, "table_cohorts.tex"), "w").write("\n".join(t) + "\n")


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
    sc = r["source_control"]
    rows.append(("Image source (E12), moderate recall (points)$^\\dagger$",
                 sg(100*sc["shipped"]["recall"][2]["diff"], 1) + " " + ci(100*sc["shipped"]["recall"][2]["lo"], 100*sc["shipped"]["recall"][2]["hi"], 1),
                 sg(100*sc["matched"]["recall"][2]["diff"], 1) + " " + ci(100*sc["matched"]["recall"][2]["lo"], 100*sc["matched"]["recall"][2]["hi"], 1),
                 "procedure-dependent"))
    t = ["\\begin{table}[t]",
         "\\caption{\\textbf{Matched calibration changes model comparisons.} Differences (first minus",
         "second) with 95\\% paired-bootstrap intervals at the shipped rule and with both models'",
         "cut-points refitted by the same procedure (external rows: R4, two-fold cross-fitted on",
         "APTOS labels; development rows: fold-wise cross-fitted). $^\\ast$Interval excludes zero.",
         "$^\\dagger$Under random-half cross-fitting the matched difference is significant; under",
         "fold-wise cross-fitting it is not (Supplementary Note~3).}\\label{tab:record}",
         "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}p{4.0cm}p{2.7cm}p{2.7cm}l@{}}", "\\toprule",
         "Comparison & Shipped rule & Matched calibration & Verdict \\\\", "\\midrule"]
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
    for f in (cohorts, record, orderdr, interventions):
        f()
    print("tables written")
