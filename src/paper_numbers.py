"""
paper_numbers.py — every number quoted in the journal manuscript, generated from artefacts.

Writes paper/numbers.tex as \\pndef{key}{value} lines. The manuscript prints numbers only
through \\pn{key}; an undefined key stops the LaTeX build. After writing, the script scans
paper/**/*.tex for \\pn{...} keys and fails if any is undefined here, so the ledger and the
text cannot drift apart. A parallel JSON (docs/generated/paper/numbers.json) records the raw
value and the source file of every key.
"""
from __future__ import annotations
import glob, json, os, re, sys
import numpy as np

G, GP = "docs/generated", "docs/generated/paper"
LEDGER = {}


def put(key, value, fmt, source):
    if value is None:
        return
    if isinstance(fmt, str):
        txt = fmt.format(value)
    else:
        txt = fmt(value)
    txt = txt.replace("-", "\\textminus{}") if txt.startswith("-") else txt
    LEDGER[key] = {"tex": txt, "value": value, "source": source}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


q4 = "{:.4f}"
q3 = "{:.3f}"
p1 = lambda v: f"{100 * v:.1f}"
p2 = lambda v: f"{100 * v:.2f}"
i0 = lambda v: f"{int(round(v)):,}"
sg4 = lambda v: f"{v:+.4f}".replace("-", "\\textminus{}")


def ci4(lo, hi):
    return f"[{sg4(lo)}, {sg4(hi)}]"


def thesis_ledger():
    t = load(f"{G}/thesis_numbers.json")
    src = f"{G}/thesis_numbers.json"
    v = lambda k: t[k]["value"]
    put("n.idrid", v("pool.n_idrid"), i0, src)
    put("n.messidor2", v("pool.n_messidor2"), i0, src)
    put("n.dev", v("pool.n"), i0, src)
    put("n.aptos", v("external.n"), i0, src)
    put("dme.n1", v("dme.n_grade1"), i0, src)
    put("dme.floor", v("dme.floor_ungated"), "{:.1f}", src)
    put("sel.dev.qwk", v("champion.dr.qwk"), q4, src)
    put("sel.dev.acc", v("champion.dr.accuracy"), "{:.1f}", src)
    put("sel.dme.qwk", v("champion.dme_ungated.qwk"), q4, src)
    put("sel.dme.acc", v("champion.dme_ungated.accuracy"), "{:.1f}", src)
    put("sel.dme.mid", v("champion.dme_ungated.recall.grade1"), "{:.1f}", src)
    put("sel.aptos.qwk", v("external.qwk"), q4, src)
    put("sel.aptos.lo", v("external.qwk.lo"), q4, src)
    put("sel.aptos.hi", v("external.qwk.hi"), q4, src)
    put("sel.aptos.acc", v("external.accuracy"), "{:.1f}", src)
    put("sel.aptos.floor", v("external.floor"), "{:.1f}", src)
    put("sel.aptos.sens", v("external.referable_sensitivity"), "{:.2f}", src)
    put("sel.aptos.spec", v("external.referable_specificity"), "{:.2f}", src)
    put("sel.aptos.mild", v("external.recall.dr1"), "{:.1f}", src)
    put("probe.imagenet", v("probe.dr.b"), q4, src)
    put("probe.retfound", v("probe.dr.a"), q4, src)
    put("probe.diff", v("probe.dr.diff"), sg4, src)
    put("probe.ci", (v("probe.dr.lo"), v("probe.dr.hi")), lambda x: ci4(*x), src)
    put("pipeline.worth", v("pipeline_worth.qwk"), "{:+.3f}", src)
    put("parta.n", v("parta.n_usable"), i0, src)
    put("parta.informative", v("parta.n_informative"), i0, src)
    put("parta.exact", v("parta.major_axis.exact"), "{:.1f}", src)
    put("parta.constant", v("parta.constant_floor"), "{:.1f}", src)
    put("lit.p12.dme.sens", v("lit.p12.dme_sensitivity"), "{:.1f}", src)
    put("lit.p12.dme.kappa", v("lit.p12.dme_kappa"), "{:.2f}", src)
    put("lit.p12.dr.kappa", v("lit.p12.dr_kappa"), "{:.2f}", src)
    put("lit.p6.mid", v("lit.p6.dme_middle_pct"), "{:.1f}", src)
    put("dme.mid.pct", 100 * v("dme.n_grade1") / v("dme.n_3class"), "{:.1f}", src)
    put("ensemble.diff", v("ensemble.diff"), sg4, src)
    put("ensemble.ci", (v("ensemble.lo"), v("ensemble.hi")), lambda x: ci4(*x), src)


def interventions():
    rows = load(f"{GP}/interventions.json")
    if not rows:
        return
    key = {"EfficientNet-B3 backbone": "effnet", "Macula-centred crop (DME QWK)": "macula",
           "Linear probe then fine-tune": "lpft", "Native 640-px resolution": "native",
           "CORAL ordinal head": "coral", "CORN ordinal head": "corn",
           "ConvNeXt-tiny backbone": "convnext", "RETFound ViT-L/16 (fine-tuned)": "retfound",
           "Six-model ensemble (APTOS, external)": "ensemble"}
    src = f"{GP}/interventions.json"
    nf = 0
    for r in rows:
        k = key[r["label"]]
        put(f"iv.{k}.diff", r["diff"], sg4, src)
        put(f"iv.{k}.ci", (r["lo"], r["hi"]), lambda x: ci4(*x), src)
        if "dr_default" in r:
            put(f"iv.{k}.ship", r["dr_default"]["diff"], sg4, src)
            put(f"iv.{k}.shipci", (r["dr_default"]["lo"], r["dr_default"]["hi"]),
                lambda x: ci4(*x), src)
        nf += r["verdict"] != "confirmed"
    put("iv.n", len(rows), i0, src)
    put("iv.nfalsified", nf, i0, src)


def phase0(corpus):
    d = load(f"{GP}/phase0_{corpus.lower()}.json")
    if not d:
        return
    src = f"{GP}/phase0_{corpus.lower()}.json"
    c = corpus.lower()
    rm = d["rule_matrix"]
    ms = d["models"]
    put(f"{c}.nmodels", len(ms), i0, src)
    for rule in ("R1", "R2", "R3", "R4", "R4m"):
        mild = [rm[m][rule]["recall"][1] for m in ms]
        sens = [rm[m][rule]["ref_sens"] for m in ms]
        qwk = [rm[m][rule]["qwk"] for m in ms]
        put(f"{c}.{rule}.mild.min", min(mild), p1, src)
        put(f"{c}.{rule}.mild.max", max(mild), p1, src)
        put(f"{c}.{rule}.sens.min", min(sens), p1, src)
        put(f"{c}.{rule}.sens.max", max(sens), p1, src)
        put(f"{c}.{rule}.qwk.min", min(qwk), q4, src)
        put(f"{c}.{rule}.qwk.max", max(qwk), q4, src)
    r2_worse = sum(rm[m]["R2"]["qwk"] < rm[m]["R1"]["qwk"] for m in ms)
    put(f"{c}.r2worse", r2_worse, i0, src)
    for metric in ("mild_recall", "ref_sens", "qwk"):
        s = d["spread"][metric]
        f = q4 if metric == "qwk" else p1
        put(f"{c}.o7.{metric}.within", s["within_median"], f, src)
        put(f"{c}.o7.{metric}.between", s["between_model_range"]["R1"], f, src)
        put(f"{c}.o7.{metric}.holds", "holds" if s["o7_holds"] else "does not hold",
            "{}", src)
    for k, v in d["spearman"].items():
        put(f"{c}.rho.{k.replace('_', '')}", v, "{:.2f}", src)
    if "E11FULL" in rm:
        b = rm["E11FULL"]
        for rule in ("R1", "R2", "R3", "R4", "R4m"):
            put(f"{c}.sel.{rule}.qwk", b[rule]["qwk"], q4, src)
            put(f"{c}.sel.{rule}.mild", b[rule]["recall"][1], p1, src)
            put(f"{c}.sel.{rule}.sens", b[rule]["ref_sens"], p2, src)
            put(f"{c}.sel.{rule}.spec", b[rule]["ref_spec"], p2, src)
            put(f"{c}.sel.{rule}.mean", b[rule]["mean_pred_minus_true"], "{:+.2f}", src)
        tf = b["_threshold_free"]
        put(f"{c}.sel.auroc", tf["ref_auroc"], q3, src)
        put(f"{c}.sel.brier", tf["ref_brier"], q3, src)
        put(f"{c}.sel.ece", tf["ref_ece"], q3, src)


def transfer(corpus):
    d = load(f"{GP}/transfer_{corpus.lower()}.json")
    if not d:
        return
    src = f"{GP}/transfer_{corpus.lower()}.json"
    c = corpus.lower()
    for m, v in d["models"].items():
        mm = m.lower()
        for r in v["transfer"]:
            t = r["tag"].replace("*", "s")
            put(f"{c}.tr.{mm}.{t}.sens", r["ext_sens"], p2, src)
            put(f"{c}.tr.{mm}.{t}.spec", r["ext_spec"], p2, src)
            put(f"{c}.tr.{mm}.{t}.gap", 100 * r["gap"], "{:+.1f}", src)
            put(f"{c}.tr.{mm}.{t}.target", 100 * r["target"], "{:.1f}", src)
        s = v["frontier_summary"]
        put(f"{c}.band.{mm}.sens.lo", s["band_sens"][0], p1, src)
        put(f"{c}.band.{mm}.sens.hi", s["band_sens"][1], p1, src)
        put(f"{c}.band.{mm}.spec.lo", s["band_spec"][0], p1, src)
        put(f"{c}.band.{mm}.spec.hi", s["band_spec"][1], p1, src)
        put(f"{c}.band.{mm}.miss.lo", s["band_missed_count"][0], i0, src)
        put(f"{c}.band.{mm}.miss.hi", s["band_missed_count"][1], i0, src)
        put(f"{c}.band.{mm}.nref", s["n_referable"], i0, src)
        put(f"{c}.band.{mm}.qwkmax", s["qwk_max"], q4, src)


def orderdr():
    d = load(f"{GP}/orderdr_reanalysis.json")
    if not d:
        return
    src = f"{GP}/orderdr_reanalysis.json"
    put("odr.n", d["n_external"], i0, src)
    put("odr.nmild", d["n_mild"], i0, src)
    names = {"CB-384": "cb", "LORS-384": "lors", "Swin-T": "swin", "ORDER-DR": "odr"}
    for name, k in names.items():
        for r, v in d["seed_mean_sd"][name].items():
            put(f"odr.{k}.{r}.qwk", v["qwk"][0], q4, src)
            put(f"odr.{k}.{r}.qwksd", v["qwk"][1], q4, src)
            put(f"odr.{k}.{r}.mild", v["mild_recall"][0], p1, src)
            put(f"odr.{k}.{r}.f1", v["mild_f1"][0], q3, src)
            put(f"odr.{k}.{r}.sens", v["ref_sens"][0], p1, src)
            put(f"odr.{k}.{r}.spec", v["ref_spec"][0], p1, src)
            put(f"odr.{k}.{r}.pred1", v["mild_pred"][0], "{:.1f}", src)
            put(f"odr.{k}.{r}.prec", v["mild_precision"][0], q3, src)
    agg = d["seed_mean_sd"]
    r4 = [agg[n]["R4"]["qwk"][0] for n in names]
    r1 = [agg[n]["R1"]["qwk"][0] for n in names]
    put("odr.R4.qwk.min", min(r4), q3, src); put("odr.R4.qwk.max", max(r4), q3, src)
    put("odr.R1.qwk.min", min(r1), q3, src); put("odr.R1.qwk.max", max(r1), q3, src)
    put("odr.R4.range", max(r4) - min(r4), q3, src)
    m1 = [agg[n]["R1"]["mild_recall"][0] for n in names]
    m4 = [agg[n]["R4m"]["mild_recall"][0] for n in names]
    s1 = [agg[n]["R1"]["ref_sens"][0] for n in names]
    s4 = [agg[n]["R4m"]["ref_sens"][0] for n in names]
    put("odr.R1.mild.min", min(m1), p1, src); put("odr.R1.mild.max", max(m1), p1, src)
    put("odr.R4m.mild.min", min(m4), p1, src); put("odr.R4m.mild.max", max(m4), p1, src)
    put("odr.R1.sens.min", min(s1), p1, src); put("odr.R1.sens.max", max(s1), p1, src)
    put("odr.R4m.sens.min", min(s4), p1, src); put("odr.R4m.sens.max", max(s4), p1, src)
    f4 = [agg[n]["R4m"]["mild_f1"][0] for n in names]
    put("odr.R4m.f1.max", max(f4), q3, src)


def recalibration():
    d = load(f"{G}/recalibration_curve.json")
    src = f"{G}/recalibration_curve.json"
    r = d["results"]["E08X2"]
    for n in d["sizes"]:
        g = r[str(n)]
        put(f"recal.{n}.gain", g["macro_recall_gain"][0], "{:+.2f}", src)
        put(f"recal.{n}.ci", g["macro_recall_gain"][1:], lambda x: f"[{x[0]:+.2f}, {x[1]:+.2f}]", src)
        put(f"recal.{n}.harm", 100 * g["p_harmful"], "{:.1f}", src)
        put(f"recal.{n}.mild", g["mild_recall_gain"][0], "{:+.1f}", src)
    put("recal.draws", d["draws"], i0, src)


def optimism():
    d = load(f"{GP}/selection_optimism.json")
    if not d:
        return
    src = f"{GP}/selection_optimism.json"
    v = [x["mean_optimism_dr_qwk"] for x in d.values()]
    put("optimism.median", float(np.median(v)), "{:+.4f}", src)
    put("optimism.min", min(v), "{:+.4f}", src)
    put("optimism.max", max(v), "{:+.4f}", src)


def misc():
    put("grid.effect", 0.0033, "{:.4f}", "src/paper_rules.py: E08 matched QWK 0.8646 (grid 60) vs 0.8613 (grid 80)")
    put("fovea.median", 0.196, "{:.3f}", "runs/E13gate/fovea_gate.json")
    put("n.eyepacs", 35126, i0, "runs/E08/results.json pretrain.n_images")
    put("n.ddr", 12522, i0, "DDR DR_grading.csv (grades 0-4)")


def write():
    os.makedirs("paper", exist_ok=True)
    with open("paper/numbers.tex", "w") as f:
        f.write("% GENERATED by src/paper_numbers.py -- do not edit by hand.\n")
        for k in sorted(LEDGER):
            f.write(f"\\pndef{{{k}}}{{{LEDGER[k]['tex']}}}\n")
    json.dump({k: {"tex": v["tex"], "value": v["value"], "source": v["source"]}
               for k, v in LEDGER.items()}, open(f"{GP}/numbers.json", "w"), indent=1,
              default=float)
    used = set()
    for p in glob.glob("paper/**/*.tex", recursive=True):
        if p.endswith("numbers.tex"):
            continue
        text = "\n".join(re.sub(r"(?<!\\)%.*", "", l) for l in open(p).read().splitlines())
        used |= set(re.findall(r"\\pn\{([^}]+)\}", text))
    missing = sorted(used - set(LEDGER))
    print(f"{len(LEDGER)} numbers defined, {len(used)} used in the manuscript")
    if missing:
        print("UNDEFINED KEYS USED IN THE MANUSCRIPT:", missing)
        sys.exit(1)


if __name__ == "__main__":
    for f in (thesis_ledger, interventions, lambda: phase0("APTOS"), lambda: phase0("DDR"),
              lambda: transfer("APTOS"), lambda: transfer("DDR"), orderdr, recalibration,
              optimism, misc):
        f()
    write()
