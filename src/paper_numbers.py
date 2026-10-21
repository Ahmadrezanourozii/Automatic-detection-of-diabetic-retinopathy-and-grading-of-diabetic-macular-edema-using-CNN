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
    for rule in ("R1", "R4"):
        spec = [rm[m][rule]["ref_spec"] for m in ms]
        mean = [rm[m][rule]["mean_pred_minus_true"] for m in ms]
        put(f"{c}.{rule}.spec.min", min(spec), p1, src)
        put(f"{c}.{rule}.spec.max", max(spec), p1, src)
        put(f"{c}.{rule}.mean.min", min(mean), lambda v: f"{v:+.2f}".replace("-", "\\textminus{}"), src)
        put(f"{c}.{rule}.mean.max", max(mean), lambda v: f"{v:+.2f}".replace("-", "\\textminus{}"), src)
    dm = [d["dev"][m]["mild_shipped"] for m in ms]
    put(f"{c}.devmild.min", min(dm), p1, src)
    put(f"{c}.devmild.max", max(dm), p1, src)
    au = [rm[m]["_threshold_free"]["ref_auroc"] for m in ms]
    put(f"{c}.auroc.min", min(au), q3, src)
    put(f"{c}.auroc.max", max(au), q3, src)
    put(f"{c}.o7.nfail", sum(not d["spread"][k]["o7_holds"] for k in ("mild_recall", "ref_sens")), i0, src)
    r2_worse = sum(rm[m]["R2"]["qwk"] < rm[m]["R1"]["qwk"] for m in ms)
    put(f"{c}.r2worse", r2_worse, i0, src)
    for metric in ("mild_recall", "ref_sens", "qwk"):
        s = d["spread"][metric]
        f = q4 if metric == "qwk" else p1
        put(f"{c}.o7.{metric.replace('_', '')}.within", s["within_median"], f, src)
        put(f"{c}.o7.{metric.replace('_', '')}.between", s["between_model_range"]["R1"], f, src)
        put(f"{c}.o7.{metric.replace('_', '')}.holds", "held" if s["o7_holds"] else "was falsified",
            "{}", src)
    for k, v in d["spearman"].items():
        put(f"{c}.rho.{k.replace('_', '')}", v, "{:.2f}", src)
    for mname in ("E08", "E11FULL"):
        if mname in rm:
            for rule in ("R1", "R2", "R3", "R4", "R4m"):
                b = rm[mname][rule]
                put(f"{c}.{mname.lower()}.{rule}.sens", b["ref_sens"], p2, src)
                put(f"{c}.{mname.lower()}.{rule}.spec", b["ref_spec"], p2, src)
                put(f"{c}.{mname.lower()}.{rule}.mild", b["recall"][1], p1, src)
                put(f"{c}.{mname.lower()}.{rule}.qwk", b["qwk"], q4, src)
    if c == "aptos" and "E11FULL" in d["dev"]:
        put("sel.dev.mild", d["dev"]["E11FULL"]["mild_shipped"], p1, src)
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


def secondary(corpus):
    """RETFound (five folds) is not in the pre-registered O7 model set; its rule matrix is
    computed separately (src/paper_phase0.py --out-dir docs/generated/paper/secondary)."""
    src = f"{GP}/secondary/phase0_{corpus.lower()}.json"
    d = load(src)
    if not d or "I24FULL" not in d["rule_matrix"]:
        return
    c = corpus.lower()
    r = d["rule_matrix"]["I24FULL"]
    for rule in ("R1", "R4", "R4m"):
        put(f"rf.{c}.{rule}.qwk", r[rule]["qwk"], q3, src)
        put(f"rf.{c}.{rule}.mild", r[rule]["recall"][1], p1, src)
        put(f"rf.{c}.{rule}.sens", r[rule]["ref_sens"], p1, src)
    put(f"rf.{c}.auroc", r["_threshold_free"]["ref_auroc"], q3, src)


def revision():
    """The no-training revision analyses (src/paper_revision.py)."""
    src = f"{GP}/revision.json"
    d = load(src)
    if not d:
        return
    sm = lambda v: f"{v:+.2f}".replace("-", "\\textminus{}")
    for m in ("E08", "E11FULL"):
        dv = d["dev"][m]
        put(f"r2.dev.{m.lower()}.mildf1", dv["summary_R1"]["mild"]["f1"], q3, src)
        put(f"r2.dev.{m.lower()}.cindex", dv["cindex"], q3, src)
        for k in range(5):
            put(f"r2.dev.{m.lower()}.es{k}", dv["es_by_grade"][k], "{:.2f}", src)
    for c, C in d["corpora"].items():
        k = c.lower()
        ms = list(C["models"])
        put(f"r2.{k}.n", C["n"], i0, src)
        put(f"r2.{k}.nmodels", len(ms), i0, src)
        gd = C["grade_dist"]
        for g in range(5):
            put(f"r2.{k}.pct{g}", gd[g] / sum(gd), p1, src)
        put(f"r2.{k}.prevref", sum(gd[2:]) / sum(gd), p1, src)
        for t in range(4):
            v = [C["models"][m]["auc_k"][t] for m in ms]
            put(f"r2.{k}.auc{t}.min", min(v), q3, src); put(f"r2.{k}.auc{t}.max", max(v), q3, src)
        allauc = [x for m in ms for x in C["models"][m]["auc_k"]]
        put(f"r2.{k}.aucall.min", min(allauc), q3, src)
        v = [C["models"][m]["cindex"] for m in ms]
        put(f"r2.{k}.cindex.min", min(v), q3, src); put(f"r2.{k}.cindex.max", max(v), q3, src)
        put(f"r2.{k}.A1above", C["A1_above"], i0, src)
        put(f"r2.{k}.A1below", C["A1_below"], i0, src)
        put(f"r2.{k}.B25", C["B_n_models_R5_closes_25pct"], i0, src)
        kt = C["kendall_qwkR1_vs_sens_at_spec90"]
        put(f"r2.{k}.tau", kt["tau"], sm, src)
        for m in ms:
            if m not in ("E08", "E11FULL"):
                continue
            r = C["models"][m]
            mm = m.lower()
            for k2 in range(5):
                put(f"r2.{k}.{mm}.es{k2}", r["es_by_grade"][k2], "{:.2f}", src)
            for rule in ("R1", "R4", "R4m", "R5", "R5b", "R0a"):
                b = r["rules"][rule]
                put(f"r2.{k}.{mm}.{rule}.sens", b["ref_sens"], p1, src)
                put(f"r2.{k}.{mm}.{rule}.spec", b["ref_spec"], p1, src)
                put(f"r2.{k}.{mm}.{rule}.mildp", b["mild"]["precision"] if b["mild"]["precision"] == b["mild"]["precision"] else 0.0, q3, src)
                put(f"r2.{k}.{mm}.{rule}.mildf1", b["mild"]["f1"], q3, src)
                put(f"r2.{k}.{mm}.{rule}.mildr", b["mild"]["recall"], p1, src)
                put(f"r2.{k}.{mm}.{rule}.modtomild", b["confusion"][2][1], i0, src)
            for n, v in r.get("sample_size", {}).items():
                put(f"r2.{k}.{mm}.ss{n}.sens", v["sens_mean"], p1, src)
                put(f"r2.{k}.{mm}.ss{n}.lo", v["sens_lo"], p1, src)
                put(f"r2.{k}.{mm}.ss{n}.harm", v["p_harm"], p1, src)
                put(f"r2.{k}.{mm}.ss{n}.spec", v["spec_mean"], p1, src)
            # what the operating point means for a screening programme
            for prev in (5, 10, 20):
                pv = prev / 100
                for rule in ("R1", "R4"):
                    se, sp = r["rules"][rule]["ref_sens"], r["rules"][rule]["ref_spec"]
                    put(f"r2.{k}.{mm}.{rule}.miss{prev}", (1 - se) * pv * 10000, i0, src)
                    put(f"r2.{k}.{mm}.{rule}.fref{prev}", (1 - sp) * (1 - pv) * 10000, i0, src)
                    ppv = se * pv / (se * pv + (1 - sp) * (1 - pv))
                    npv = sp * (1 - pv) / (sp * (1 - pv) + (1 - se) * pv)
                    put(f"r2.{k}.{mm}.{rule}.ppv{prev}", ppv, p1, src)
                    put(f"r2.{k}.{mm}.{rule}.npv{prev}", npv, p1, src)
        f1r4m = [C["models"][m]["rules"]["R4m"]["mild"]["f1"] for m in ms]
        f1r1 = [C["models"][m]["rules"]["R1"]["mild"]["f1"] for m in ms]
        put(f"r2.{k}.R4m.mildf1.min", min(f1r4m), q3, src); put(f"r2.{k}.R4m.mildf1.max", max(f1r4m), q3, src)
        put(f"r2.{k}.R1.mildf1.min", min(f1r1), q3, src); put(f"r2.{k}.R1.mildf1.max", max(f1r1), q3, src)
        gaps = [C["models"][m]["gap_closed_ref_sens"]["R5"] for m in ms
                if C["models"][m]["gap_closed_ref_sens"]["R5"] is not None]
        if gaps:
            put(f"r2.{k}.R5gap.min", min(gaps), "{:.2f}", src); put(f"r2.{k}.R5gap.max", max(gaps), "{:.2f}", src)
            put(f"r2.{k}.R5gap.median", float(np.median(gaps)), "{:.2f}", src)
    a1 = [x for C in d["corpora"].values() for x in (C["models"][m]["auc_k"][1] for m in C["models"])]
    put("r2.auc1.allmin", min(a1), q3, src)
    for t in range(4):
        v = [d["dev"][m]["auc_k"][t] for m in d["models"]]
        put(f"r2.dev.auc{t}.min", min(v), q3, src); put(f"r2.dev.auc{t}.max", max(v), q3, src)
    ak = [x for C in d["corpora"].values() for m in C["models"] for x in C["models"][m]["auc_k"]]
    put("r2.aucall.allmin", min(ak), q3, src)
    # outcome verdicts of the revision plan
    if "DDR" in d["corpora"]:
        C = d["corpora"]["DDR"]
        put("r2.B.verdict", "held" if C["B_n_models_R5_closes_25pct"] >= 6 else "was falsified", "{}", src)
    a1 = all(d["corpora"][c]["A1_above"] >= 8 for c in ("APTOS",) if c in d["corpora"]) and \
        all(d["corpora"][c]["A1_below"] >= 8 for c in ("DDR",) if c in d["corpora"])
    put("r2.A1.verdict", "held" if a1 else "was falsified", "{}", src)
    harm = [d["corpora"][c]["models"][m]["sample_size"]["200"]["p_harm"]
            for c in d["corpora"] for m in ("E08", "E11FULL")
            if m in d["corpora"][c]["models"] and "200" in d["corpora"][c]["models"][m].get("sample_size", {})]
    if harm:
        put("r2.C.verdict", "held" if max(harm) < 0.20 else "was falsified", "{}", src)
        put("r2.C.harm200.max", max(harm), p1, src); put("r2.C.harm200.min", min(harm), p1, src)


def eyepacs():
    src = f"{GP}/eyepacs.json"
    d = load(src)
    if not d or not d.get("models"):
        return
    ms = d["models"]
    short = {"EYEDN": "dn", "EYEEB3": "eb3"}
    aucs, tr = [], []
    for m, v in ms.items():
        k = short[m]
        put(f"eye.{k}.devn", v["dev"]["n"], i0, src)
        if v.get("train_n"):
            put(f"eye.{k}.trainn", v["train_n"], i0, src)
        put(f"eye.{k}.devqwk", v["dev"]["qwk_matched"], q4, src)
        put(f"eye.{k}.devauc", v["dev"]["auroc"], q3, src)
        put(f"eye.{k}.ngap", v["n_gap_gt5"], i0, src)
        for c, r in v["external"].items():
            cc = c.lower().replace("-", "")
            put(f"eye.{k}.{cc}.auc", r["auroc"], q3, src)
            put(f"eye.{k}.{cc}.mean", r["mean_R1"], lambda x: f"{x:+.2f}".replace("-", "\\textminus{}"), src)
            put(f"eye.{k}.{cc}.tr85", r["sens_tr85"], p1, src)
            put(f"eye.{k}.{cc}.r1sens", r["sens_R1"], p1, src)
            put(f"eye.{k}.{cc}.r4sens", r["sens_R4"], p1, src)
            aucs.append(r["auroc"]); tr.append(r["sens_tr85"])
    if aucs:
        put("eye.auc.min", min(aucs), q3, src); put("eye.auc.max", max(aucs), q3, src)
        put("eye.tr85.min", min(tr), p1, src); put("eye.tr85.max", max(tr), p1, src)
    held = [v["E2_holds"] for v in ms.values() if v.get("E2_holds") is not None]
    if held:
        put("eye.E2.verdict", "held" if all(held) else "was falsified", "{}", src)


def mechanism():
    src = f"{GP}/mechanism.json"
    d = load(src)
    if not d:
        return
    sm = lambda v: f"{v:+.2f}".replace("-", "\\textminus{}")
    sp = lambda v: f"{100 * v:+.1f}".replace("-", "\\textminus{}")
    for m, res in d["models"].items():
        k = m.lower()
        for cond, short in d["short"].items():
            if cond in res:
                put(f"mech.{k}.{short}.dmean", res[cond]["dmean"], sm, src)
                put(f"mech.{k}.{short}.dsens", res[cond]["dsens"], sp, src)
        put(f"mech.{k}.orig.mean", res["orig"]["mean"], sm, src)
        put(f"mech.{k}.orig.qwk", res["orig"]["qwk"], q3, src)
        if "_archived" in res:
            put(f"mech.{k}.arch.qwk", res["_archived"]["qwk"], q3, src)
            put(f"mech.{k}.arch.sens", res["_archived"]["sens"], p1, src)
        put(f"mech.{k}.orig.sens", res["orig"]["sens"], p1, src)
        b = res["_most_negative"]
        put(f"mech.{k}.best.name", d["labels"][b].lower(), "{}", src)
        put(f"mech.{k}.best.dmean", res[b]["dmean"], sm, src)
        put(f"mech.{k}.best.dsens", res[b]["dsens"], sp, src)
        put(f"mech.{k}.nhits", len(res["_hits"]), i0, src)
    put("mech.A2.verdict", "held" if d.get("A2_holds") else "was falsified", "{}", src)
    for c, v in d.get("stats", {}).items():
        k = c.lower().replace("-", "")
        for key, name, fmt in (("laplacian_var", "sharp", "{:.0f}"), ("jpeg_quality", "jpegq", "{:.0f}"),
                               ("brightness", "bright", "{:.0f}"), ("rms_contrast", "contrast", "{:.0f}"),
                               ("resolution", "res", i0)):
            if v.get(key):
                put(f"stats.{k}.{name}", v[key]["median"], fmt, src)


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
        put(f"recal.{n}.ci", g["macro_recall_gain"][1:], lambda x: f"[{x[0]:+.2f}, {x[1]:+.2f}]".replace("-", "\\textminus{}"), src)
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


def record():
    d = load(f"{GP}/record.json")
    if not d:
        return
    src = f"{GP}/record.json"
    bb = d["backbone_aptos"]
    for rule in ("R1", "R4", "R4m"):
        q = bb[rule]["qwk"]
        put(f"rec.bb.{rule}.qwk", q["diff"], sg4, src)
        put(f"rec.bb.{rule}.qwkci", (q["lo"], q["hi"]), lambda x: ci4(*x), src)
        for k, nm in ((1, "mild"), (2, "mod"), (3, "sev"), (4, "pdr")):
            put(f"rec.bb.{rule}.{nm}", 100 * bb[rule]["recall"][k]["diff"], "{:+.1f}", src)
    lp = d["lpft_dev"]
    for mode, nm in (("shipped", "ship"), ("matched", "match")):
        a = lp[mode]["accuracy"]
        put(f"rec.lpft.{nm}", 100 * a["diff"], "{:+.2f}", src)
        put(f"rec.lpft.{nm}ci", (100 * a["lo"], 100 * a["hi"]),
            lambda x: f"[{x[0]:+.2f}, {x[1]:+.2f}]".replace("-", "\\textminus{}"), src)
    for mode, v in d["source_control"].items():
        r = v["recall"][2]
        put(f"rec.e12.{mode}", 100 * r["diff"], lambda x: f"{x:+.1f}".replace("-", "\\textminus{}"), src)
        put(f"rec.e12.{mode}ci", (100 * r["lo"], 100 * r["hi"]),
            lambda x: f"[{x[0]:+.1f}, {x[1]:+.1f}]".replace("-", "\\textminus{}"), src)


def seeds():
    d = load(f"{GP}/seeds.json")
    if not d:
        return
    src = f"{GP}/seeds.json"
    put("seeds.n", len(d["seeds"]), i0, src)
    put("seeds.dev.sd", d["dev_matched_qwk_sd"], q4, src)
    put("seeds.dev.lo", d["dev_matched_qwk_range"][0], q4, src)
    put("seeds.dev.hi", d["dev_matched_qwk_range"][1], q4, src)
    put("seeds.yardstick", 2 * np.sqrt(2) * d["dev_matched_qwk_sd"], q4, src)
    for c in ("APTOS", "DDR"):
        if c not in d:
            continue
        v = d[c]; k = c.lower()
        put(f"seeds.{k}.sd", v["qwk_R1_sd"], q4, src)
        put(f"seeds.{k}.qwklo", v["qwk_R1_range"][0], q4, src)
        put(f"seeds.{k}.qwkhi", v["qwk_R1_range"][1], q4, src)
        put(f"seeds.{k}.mildR1.lo", v["mild_R1"][0], p1, src)
        put(f"seeds.{k}.mildR1.hi", v["mild_R1"][1], p1, src)
        put(f"seeds.{k}.mildR4m.lo", v["mild_R4m"][0], p1, src)
        put(f"seeds.{k}.mildR4m.hi", v["mild_R4m"][1], p1, src)
        put(f"seeds.{k}.sigR1", v["pairwise_R1"]["n_significant"], i0, src)
        put(f"seeds.{k}.sigR4", v["pairwise_R4"]["n_significant"], i0, src)
        put(f"seeds.{k}.ntests", v["pairwise_R1"]["n_tests"], i0, src)
        put(f"seeds.{k}.maxR1", 100 * v["pairwise_R1"]["max_abs_diff"], "{:.1f}", src)
        if c == "APTOS":
            put("seeds.o2", "held" if v["O2_prediction_holds"] else "was falsified", "{}", src)
            put("seeds.o3", "held" if v["O3_holds"] else "was falsified", "{}", src)


def reverse():
    d = load(f"{GP}/reverse.json")
    if not d:
        return
    src = f"{GP}/reverse.json"
    for c, k in (("Messidor-2", "m2"), ("IDRiD", "idrid")):
        if c not in d:
            continue
        for rule in ("R1", "R4", "R4m"):
            b = d[c]["rules"][rule]
            put(f"rev.{k}.{rule}.mild", b["recall"][1], p1, src)
            put(f"rev.{k}.{rule}.qwk", b["qwk"], q4, src)
            put(f"rev.{k}.{rule}.sens", b["ref_sens"], p1, src)
            put(f"rev.{k}.{rule}.spec", b["ref_spec"], p1, src)
            put(f"rev.{k}.{rule}.mean", b["mean_pred_minus_true"], "{:+.2f}", src)
            put(f"rev.{k}.{rule}.f1", b.get("mild_f1", float("nan")), q3, src)
    for o in ("o4", "o5", "o6"):
        if o in d:
            put(f"rev.{o}", "held" if d[o] else "was falsified", "{}", src)
    for c, k in (("Messidor-2", "m2"), ("IDRiD", "idrid")):
        if c in d:
            dest = d[c]["mild_destination_R1"]
            put(f"rev.{k}.nmild", d[c]["n_mild"], i0, src)
            put(f"rev.{k}.mild.to0", dest[0], i0, src)
            put(f"rev.{k}.mild.to2", dest[2], i0, src)
            put(f"rev.{k}.auroc", d[c]["threshold_free"]["ref_auroc"], q3, src)
    if "dev" in d:
        put("rev.dev.qwk", d["dev"]["qwk_matched"], q4, src)


def outcomes():
    t = load(f"{GP}/transfer_ddr.json")
    if t and "E08" in t["models"]:
        b = [r for r in t["models"]["E08"]["transfer"] if r["tag"] == "B"][0]
        put("ddr.o8", "held" if abs(b["gap"]) > 0.05 else "was falsified", "{}",
            f"{GP}/transfer_ddr.json")
    rows = load(f"{GP}/interventions.json") or []
    for r in rows:
        if r["label"].startswith("RETFound"):
            put("iv.retfound.folds", r["folds"], "{}", f"{GP}/interventions.json")
            if str(r["folds"]) == "5":
                put("retfound.o9", "held" if r["hi"] < 0 else "was falsified", "{}",
                    f"{GP}/interventions.json")


def overlap():
    dsrc = f"{GP}/crossfold_dupes.json"
    dd = load(dsrc)
    if dd:
        put("dup.npairs", dd["n_pairs"], i0, dsrc)
        put("dup.nimg", dd["n_images"], i0, dsrc)
        put("dup.maxdiff", dd["max_abs_diff"], q4, dsrc)
    src = f"{GP}/overlap_audit.json"
    d = load(src)
    if not d:
        return
    put("ov.bits", d["threshold_bits"], i0, src)
    for tag, v in d["pairs"].items():
        put(f"ov.{tag.replace('_', '')}", v["n_pairs"], i0, src)
    put("ov.crossfold", len([p for p in d.get("dev_within_cross_fold_examples", []) if p[2] <= 8]), i0, src)
    for tag, v in d["pairs"].items():
        if "n_le8" in v:
            put(f"ov.{tag.replace('_', '')}.le8", v["n_le8"], i0, src)
            put(f"ov.{tag.replace('_', '')}.exact", v["hist"][0], i0, src)
            nz = [i for i, h in enumerate(v["hist"]) if h]
            if nz:
                put(f"ov.{tag.replace('_', '')}.mind", nz[0], i0, src)
    ssrc = f"{GP}/overlap_sensitivity.json"
    sd = load(ssrc)
    if sd:
        for c, v in sd["corpora"].items():
            put(f"ov.{c.lower()}.flagged", v["n_flagged"], i0, ssrc)
            put(f"ov.{c.lower()}.maxchange", 100 * v["max_abs_change"], "{:.2f}", ssrc)


def posthoc():
    """Analyses added after an internal editorial assessment (src/paper_posthoc.py); post hoc."""
    src = f"{GP}/posthoc.json"
    d = load(src)
    if not d:
        return
    pct1 = lambda v: f"{100 * v:.1f}"
    pts = lambda v: f"{100 * v:+.1f}".replace("-", "\\textminus{}")
    sg2 = lambda v: f"{v:+.2f}".replace("-", "\\textminus{}")
    ex = d["exact"]
    for k, v in ex["harm_by_k"].items():
        put(f"ph.harm.k{k}", v, pct1, src)
    for key, v in ex["harm_by_prevalence"].items():
        p_, n_ = key.split("_")
        put(f"ph.harm.n{n_}.p{int(round(100 * float(p_)))}", v, pct1, src)
    for k, v in ex["required_sample_sens"].items():
        put(f"ph.req.k{k}.r05", v["0.05"], pct1, src)
        put(f"ph.req.k{k}.r10", v["0.10"], pct1, src)
    put("ph.sim.maxdev", 100 * ex["max_abs_dev_sim"], "{:.1f}", src)
    sims = {}
    for c, C in d["corpora"].items():
        for m, v in C["models"].items():
            for k, r in v.get("by_cases", {}).items():
                sims.setdefault(k, []).append(r["p_harm"])
    for k, v in sims.items():
        put(f"ph.sim.k{k}.min", min(v), pct1, src); put(f"ph.sim.k{k}.max", max(v), pct1, src)
    dv = d["dev"]
    mr = [v["mild_mr"]["f1"] for v in dv.values()]
    put("ph.dev.mildmr.min", min(mr), q3, src); put("ph.dev.mildmr.max", max(mr), q3, src)
    put("ph.dev.e11full.mildmr", dv["E11FULL"]["mild_mr"]["f1"], q3, src)
    for key in ("auc_ref", "auc_0v1", "auc_1v2", "auc_0v2"):
        v = [x["spectrum"][key] for x in dv.values()]
        kk = key.replace("auc_", "")
        put(f"ph.dev.auc{kk}.min", min(v), q3, src); put(f"ph.dev.auc{kk}.max", max(v), q3, src)
    for c, C in d["corpora"].items():
        k = c.lower()
        ms = C["models"]
        put(f"ph.{k}.nmild", C["grade_dist"][1], i0, src)
        put(f"ph.{k}.pctmild", C["grade_dist"][1] / C["n"], pct1, src)
        for key in ("auc_ref", "auc_ref_devmix", "auc_0v1", "auc_1v2", "auc_0v2"):
            v = [x["spectrum"][key] for x in ms.values()]
            kk = key.replace("auc_", "").replace("_", "")
            put(f"ph.{k}.auc{kk}.min", min(v), q3, src); put(f"ph.{k}.auc{kk}.max", max(v), q3, src)
        r = [x["R5_vs_R0a"] for x in ms.values()]
        ds = [x["sens"]["R5"] - x["sens"]["R0a"] for x in r]
        put(f"ph.{k}.r5.dsens.med", float(np.median(ds)), pts, src)
        put(f"ph.{k}.r5.dsens.min", min(ds), pts, src); put(f"ph.{k}.r5.dsens.max", max(ds), pts, src)
        dq = [x["qwk"]["R5"] - x["qwk"]["R0a"] for x in r]
        put(f"ph.{k}.r5.dqwk.med", float(np.median(dq)), lambda v: f"{v:+.3f}".replace("-", "\\textminus{}"), src)
        put(f"ph.{k}.r0a.mild.med", float(np.median([x["mild_recall"]["R0a"] for x in r])), pct1, src)
        put(f"ph.{k}.r5.mild.med", float(np.median([x["mild_recall"]["R5"] for x in r])), pct1, src)
        put(f"ph.{k}.r5.nmildmodels", sum(x["mild_npred"]["R5"] > 0 for x in r), i0, src)
        put(f"ph.{k}.r5.closed25", sum(1 for x in r if x["closed_R5"] is not None and x["closed_R5"] >= 0.25), i0, src)
        t = C["tau"]
        put(f"ph.{k}.tau", t["tau"], sg2, src); put(f"ph.{k}.tau.lo", t["lo"], sg2, src)
        put(f"ph.{k}.tau.hi", t["hi"], sg2, src); put(f"ph.{k}.tau.p", t["p_perm"], "{:.2f}", src)
        for m in ("E08", "E11FULL"):
            sa = ms[m].get("spec_at_sens")
            if sa:
                for t_ in ("0.85", "0.9"):
                    put(f"ph.{k}.{m.lower()}.spec{int(round(100 * float(t_)))}", sa[t_], pct1, src)
    ps = [C["tau"]["p_perm"] for C in d["corpora"].values()]
    put("ph.tau.p.min", min(ps), "{:.2f}", src); put("ph.tau.p.max", max(ps), "{:.2f}", src)
    allr = [x["R5_vs_R0a"] for C in d["corpora"].values() for x in C["models"].values()]
    put("ph.r5.nmild.models", sum(x["mild_npred"]["R5"] > 0 for x in allr), i0, src)
    put("ph.r5.n", len(allr), i0, src)
    prev = [C["n_ref"] / C["n"] for C in d["corpora"].values()]
    put("ph.prev.min", min(prev), pct1, src); put("ph.prev.max", max(prev), pct1, src)
    put("ph.k200", 200 * float(np.mean(prev)), i0, src)       # referable cases in 200 enriched images


def verdicts():
    """One-word verdicts for the outcome table (Supplementary Table S2), read from the keys the
    analyses already wrote; 'held' or 'falsified'."""
    word = lambda k: None if k not in LEDGER else ("falsified" if "falsified" in LEDGER[k]["value"] else "held")
    pairs = {"O2": "seeds.o2", "O3": "seeds.o3", "O4": "rev.o4", "O5": "rev.o5", "O6": "rev.o6",
             "O8": "ddr.o8", "O9": "retfound.o9", "A1": "r2.A1.verdict", "A2": "mech.A2.verdict",
             "B": "r2.B.verdict", "C": "r2.C.verdict", "E2": "eye.E2.verdict"}
    for o, k in pairs.items():
        w = word(k)
        if w:
            put(f"v.{o}", w, "{}", f"ledger key {k}")
    o7 = [k for k in LEDGER if k.endswith(".holds") and ".o7." in k and ("mildrecall" in k or "refsens" in k)]
    if o7:
        n_f = sum("falsified" in LEDGER[k]["value"] for k in o7)
        put("v.O7", "falsified" if n_f == len(o7) else ("partly falsified" if n_f else "held"), "{}",
            "ledger keys " + ", ".join(sorted(o7)))
        put("v.O7.n", n_f, i0, "ledger"); put("v.O7.of", len(o7), i0, "ledger")


def misc():
    put("grid.effect", 0.0033, "{:.4f}", "src/paper_rules.py: E08 matched QWK 0.8646 (grid 60) vs 0.8613 (grid 80)")
    put("fovea.median", 0.196, "{:.3f}", "runs/E13gate/fovea_gate.json")
    put("n.eyepacs", 35126, i0, "runs/E08/results.json pretrain.n_images")
    put("n.ddr", 12522, i0, "DDR DR_grading.csv (grades 0-4)")
    # the archived (erroneous) CORN comparison, quoted in Supplementary Note 4
    src = "docs/generated/matched_corn_e08.json"
    arc = load(src) or []
    for r in arc:
        if r["head"] == "dr" and r["metric"] == "QWK" and r["mode"] == "matched":
            put("corn.archived.diff", r["diff"], sg4, src)
            put("corn.archived.ci", (r["lo"], r["hi"]), lambda x: ci4(*x), src)
            put("corn.archived.qwk", r["a"], q4, src)
    d = load(f"{GP}/dev_summary.json") or {}
    for m, v in d.items():
        put(f"dev.{m.lower()}.matched", v["qwk_matched"], q4, f"{GP}/dev_summary.json")
        put(f"dev.{m.lower()}.shipped", v["qwk_shipped"], q4, f"{GP}/dev_summary.json")


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
              lambda: phase0("DeepDRiD"), lambda: phase0("SUSTech"),
              lambda: transfer("APTOS"), lambda: transfer("DDR"),
              lambda: transfer("DeepDRiD"), lambda: transfer("SUSTech"), eyepacs, mechanism,
              lambda: secondary("APTOS"), lambda: secondary("DDR"), orderdr, recalibration,
              optimism, record, seeds, reverse, outcomes, overlap, revision, posthoc, misc, verdicts):
        f()
    write()
