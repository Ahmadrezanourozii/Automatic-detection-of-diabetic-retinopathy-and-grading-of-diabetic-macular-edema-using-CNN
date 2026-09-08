"""
thesis_numbers.py — the single source of every number the thesis quotes.

Chapter 3 opens with the claim that no number in it was typed by hand. That claim was not
true and nothing was in a position to notice: three decimals in it are wrong, and one of
them says the default sigmoid threshold is 5.0. This script turns the claim into a check.

It reads the archived artefacts — `runs/*/results.json`, `runs/*/external_aptos.json`, the
comparison JSONs under `docs/generated/` — and writes `docs/generated/thesis_numbers.json`:
one entry per quotable number, carrying the value, the Persian rendering the prose must use,
and the file it came from. `tools/farsi_lint.py --numbers` then refuses any Persian numeral in
a chapter that is not in that ledger.

WHAT THE CHECK DOES AND DOES NOT PROVE. It proves a numeral in the prose exists in an
archived artefact and is rendered in the right digit order. It does NOT prove the numeral is
the right one for the sentence around it — no lint can. It removes hand-typing, stale values
and digit reversal; it does not remove the need to read the sentence.

Usage:  python3 src/thesis_numbers.py --datasets <root>
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GEN = "docs/generated"
LEDGER = f"{GEN}/thesis_numbers.json"

FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fa(value, decimals=None, percent=False):
    """Render a number the way Persian prose in this thesis must carry it.

    The integer part comes FIRST, then the separator, then the fraction — the logical order,
    which is what the bidi algorithm renders correctly because Persian digits (U+06F0..U+06F9)
    are strong left-to-right. Typing the two groups the other way round, as chapter 3 did,
    renders 0.5 as 5.0. That is the whole reason this helper exists rather than a str.format
    at each call site.

    The separator is U+066B ARABIC DECIMAL SEPARATOR — the same one `src/report.py` already
    emits into the generated tables. The hand-written chapters used "/" instead, which is
    also read as a decimal point in Persian practice but collides with the fraction and date
    forms, and is what the reversed numbers were hiding inside.
    """
    if decimals is None:
        decimals = 0 if float(value).is_integer() and not percent else (1 if percent else 4)
    s = f"{value:.{decimals}f}"
    intpart, _, frac = s.partition(".")
    out = intpart if not frac else f"{intpart}\u066b{frac}"
    return out.translate(FA_DIGITS)


class Ledger:
    def __init__(self):
        self.rows = {}

    def add(self, key, value, source, decimals=None, percent=False, note=""):
        assert key not in self.rows, f"duplicate ledger key {key}"
        self.rows[key] = {"value": float(value), "fa": fa(value, decimals, percent),
                          "source": source, "note": note}
        return self.rows[key]

    def add_interval(self, key, lo, hi, source, decimals=4, note=""):
        self.add(f"{key}.lo", lo, source, decimals, note=note)
        self.add(f"{key}.hi", hi, source, decimals, note=note)


def add_matched(led, tag, path):
    """Every matched-calibration comparison: both runs' scores, the difference, the interval."""
    if not os.path.exists(path):
        return
    for r in json.load(open(path)):
        if r["mode"] != "matched":
            continue          # the shipped-cut-point rows are not what the thesis quotes
        stem = f"{tag}.{r['head']}.{r['metric'].lower()}"
        pct = r["metric"] == "accuracy"
        d = 1 if pct else 4
        scale = 100 if pct else 1
        led.add(f"{stem}.a", r["a"] * scale, path, d, pct)
        led.add(f"{stem}.b", r["b"] * scale, path, d, pct)
        led.add(f"{stem}.diff", r["diff"] * scale, path, d, pct)
        led.add_interval(f"{stem}", r["lo"] * scale, r["hi"] * scale, path, d)
        led.add(f"{stem}.n", r["n"], path, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="dataset roots; needed only for the label-distribution rows")
    a = ap.parse_args()
    led = Ledger()

    # ---- the declared model, read from the declaration and never inferred (PROTOCOL.md §3)
    sel = json.load(open("data/selected_model.json"))
    selected = sel.get("run") or sel.get("run_id") or sel.get("selected")

    # ---- external validation of the DECLARED model, not of whichever run scored best
    ext_path = f"runs/EXT{selected}/external_aptos.json"
    ext = json.load(open(ext_path))["metrics"]["dr"]
    led.add("external.n", ext["n"], ext_path, 0)
    led.add("external.qwk", ext["qwk"], ext_path, 4)
    led.add_interval("external.qwk", *ext["qwk_ci95"], ext_path, 4)
    led.add("external.accuracy", ext["accuracy"] * 100, ext_path, 1, True)
    led.add_interval("external.accuracy", ext["accuracy_ci95"][0] * 100,
                     ext["accuracy_ci95"][1] * 100, ext_path, 1)
    led.add("external.floor", ext["majority_floor"] * 100, ext_path, 1, True)
    led.add("external.macro_f1", ext["macro_f1"], ext_path, 3)
    led.add("external.referable_sensitivity", ext["referable_sensitivity"] * 100, ext_path, 1, True)
    led.add("external.referable_specificity", ext["referable_specificity"] * 100, ext_path, 1, True)
    for i, (rec, sup) in enumerate(zip(ext["per_class_recall"], ext["support"])):
        led.add(f"external.recall.dr{i}", rec * 100, ext_path, 1, True)
        led.add(f"external.support.dr{i}", sup, ext_path, 0)

    # ---- every matched-calibration comparison the thesis cites
    for tag, path in (("champion_vs_e10", f"{GEN}/matched_e11full_e10.json"),
                      ("lpft", f"{GEN}/matched_comparison.json"),
                      ("macula", f"{GEN}/matched_e08_e14mac.json"),
                      ("native_res", f"{GEN}/matched_e10_e17nat.json"),
                      ("coral", f"{GEN}/matched_coral_e08.json"),
                      ("corn", f"{GEN}/matched_corn_e08.json"),
                      ("convnext", f"{GEN}/matched_cnx_e08.json"),
                      # deliberately cross-source: RETFound replaces the EyePACS stage, so the
                      # mounted datasets differ BY CONSTRUCTION and the comparison is run with
                      # --acknowledge-consumption-diff, which stamps that into its document.
                      ("retfound_ft", f"{GEN}/matched_retfound_e09.json")):
        add_matched(led, tag, path)

    # ---- the ensemble, judged on the held-out set rather than on the pool that chose it
    ens = f"{GEN}/ensemble_external.json"
    if os.path.exists(ens):
        e = json.load(open(ens))["vs_validation_selected"]
        led.add("ensemble.diff", e["diff"], ens, 4)
        led.add_interval("ensemble", e["lo"], e["hi"], ens, 4)

    # ---- frozen-probe comparison (F8): representation quality with the backbone frozen
    pp = f"{GEN}/probe_vs_probe.json"
    if os.path.exists(pp):
        for head, r in json.load(open(pp)).items():
            if isinstance(r, dict) and "diff" in r:
                led.add(f"probe.{head}.a", r["a"], pp, 4)
                led.add(f"probe.{head}.b", r["b"], pp, 4)
                led.add(f"probe.{head}.diff", r["diff"], pp, 4)
                led.add_interval(f"probe.{head}", r["lo"], r["hi"], pp, 4)

    # ---- Part A, the IDRiD derivation gate
    gate = f"{GEN}/idrid_derivation_gate.json"
    if os.path.exists(gate):
        g = json.load(open(gate))
        led.add("parta.n_segmentation", g["n_segmentation"], gate, 0)
        led.add("parta.n_crosswalk", g["n_crosswalk"], gate, 0)
        led.add("parta.n_usable", g["n_usable"], gate, 0)
        led.add("parta.corr_min", g["corr_min"], gate, 4)
        led.add("parta.od_max_px", g["od_max_px"], gate, 0)
        for name, d in g["by_definition"].items():
            key = name.replace(" ", "_").replace("-", "_")
            led.add(f"parta.{key}.exact", d["exact_match"] * 100, gate, 1, True)
            for grade, rec in enumerate(d["per_class_recall"]):
                led.add(f"parta.{key}.recall.grade{grade}", rec * 100, gate, 1, True)

    # ---- Part A: the constant-predictor floor the derivation has to beat. It is NOT in
    # the gate JSON as its own field, and that is the number the verdict turns on, so it is
    # recomputed here from the support vector rather than quoted from the session that
    # first printed it.
    if os.path.exists(gate):
        g = json.load(open(gate))
        any_def = next(iter(g["by_definition"].values()))
        support = any_def["support"]
        led.add("parta.constant_floor", 100 * max(support) / sum(support), gate, 1, True)
        for grade, n in enumerate(support):
            led.add(f"parta.support.grade{grade}", n, gate, 0)
        led.add("parta.n_informative", sum(support) - max(support), gate, 0,
                note="images whose grade is not the majority grade -- the only ones on which "
                     "the derivation can be told apart from the constant predictor")

    # ---- Part A per-image geometry. The two contradicting images and the two agreeing ones
    # are quoted image by image in the report, so their distances and disc diameters belong in
    # the ledger rather than being retyped from a table.
    if os.path.exists(gate):
        # The ranges the report quotes are over the FOUR INFORMATIVE images -- the ones whose
        # expert grade is not the majority grade -- not over all 54. Quoting the range over
        # all 54 would describe a set on which the test cannot discriminate, which is the one
        # thing the report is at pains to say it did not do.
        majority = max(range(3), key=lambda gr: sum(x["true"] == gr for x in g["per_image"]))
        informative = [x for x in g["per_image"] if x["true"] != majority]
        led.add("parta.informative.min_dist.smallest",
                min(x["min_dist_px"] for x in informative), gate, 0)
        led.add("parta.informative.min_dist.largest",
                max(x["min_dist_px"] for x in informative), gate, 0)
        dds = [d for x in informative for d in x["dd"].values()]
        led.add("parta.informative.dd.smallest", min(dds), gate, 0)
        led.add("parta.informative.dd.largest", max(dds), gate, 0)
        # The disc diameter quoted per image is the equivalent-area definition, which is the
        # one the write-up uses; all three agree on every verdict, and that is stated there.
        for x in g["per_image"]:
            key = x["seg"].replace("IDRiD_", "seg")
            if x["true"] != x["equivalent_area"] or x in informative:
                led.add(f"parta.{key}.min_dist", x["min_dist_px"], gate, 0)
                led.add(f"parta.{key}.dd", x["dd"]["equivalent_area"], gate, 0)
                led.add(f"parta.{key}.expert", x["true"], gate, 0)
                led.add(f"parta.{key}.derived", x["equivalent_area"], gate, 0)

    # ---- the crosswalk overlay check, regenerated by src/verify_crosswalk_pairs.py. The
    # report previously quoted 0.78--0.96 from a session that archived nothing; those figures
    # could not be reproduced and these replace them. The verdict they support is unchanged.
    cwp = f"{GEN}/crosswalk_pairs.json"
    if os.path.exists(cwp):
        c = json.load(open(cwp))
        led.add("crosswalk.diff_min", c["min_mean_abs_diff"], cwp, 2)
        led.add("crosswalk.diff_max", c["max_mean_abs_diff"], cwp, 2)
        led.add("crosswalk.n_informative", c["n_informative"], cwp, 0)

    # ---- epoch wall-clock, read out of the training logs rather than remembered. The
    # ConvNeXt decision in chapter 4 is priced in these seconds, so they are archived like any
    # other quoted number instead of being typed from a session that is gone (FINDINGS.md F11).
    import glob as _glob, re as _re
    for tag, run in (("convnext", "E21CNXA"), ("densenet", "E08")):
        secs = []
        for lg in _glob.glob(f"runs/{run}/train*.log"):
            secs += [int(m) for m in _re.findall(r"score\s+[\d.]+\s+(\d+)s", open(lg).read())]
        if secs:
            secs.sort()
            led.add(f"epoch_seconds.{tag}", secs[len(secs) // 2],
                    f"runs/{run}/train*.log ({len(secs)} epochs, median)", 0)

    # ---- numbers quoted FROM THE LITERATURE. Each was read against the source and is
    # recorded in results/VERIFICATION-*.md; the key names the paper so a reader of the thesis
    # can go from a numeral in the prose to the paper and the check in two steps. Nothing goes
    # in here that was not confirmed at source -- an unverified figure is not citable at all,
    # so it must not be renderable either.
    LIT = [
        # key, value, decimals, percent, source
        ("lit.p1.underestimate_default", 84.7, 1, True, "P1 Table 12"),
        ("lit.p1.underestimate_calibrated", 49.7, 1, True, "P1 Table 12"),
        ("lit.p1.grade1_predicted", 712.7, 1, False, "P1 Table 12"),
        ("lit.p1.grade1_true", 270, 0, False, "P1 Table 12"),
        ("lit.p1.grade1_precision", 0.193, 3, False, "P1 Table 11"),
        ("lit.p1.ece_internal", 0.049, 3, False, "P1 Fig. 2"),
        ("lit.p1.ece_external", 0.160, 3, False, "P1 Fig. 2"),
        ("lit.p1.qwk_external", 0.6423, 4, False, "P1 Table 6"),
        ("lit.p3.aptos_frozen", 0.93, 2, False, "P3 results"),
        ("lit.p3.aptos_finetuned", 0.94, 2, False, "P3 results"),
        ("lit.p3.odir_frozen", 0.78, 2, False, "P3 results"),
        ("lit.p3.odir_finetuned", 0.80, 2, False, "P3 results"),
        ("lit.p3.head_params", 5125, 0, False, "P3 methods"),
        ("lit.p3.retfound_params_m", 303.3, 1, False, "P3 methods"),
        ("lit.p5.accuracy_aptos", 87.98, 2, True, "P5 abstract"),
        ("lit.p5.qwk_aptos", 0.9370, 4, False, "P5 abstract"),
        ("lit.p5.auc_mild", 0.84, 2, False, "P5 §4.8"),
        ("lit.p6.resnet_accuracy", 0.823, 3, False, "P6 Table 3"),
        ("lit.p6.retfound_accuracy", 0.822, 3, False, "P6 Table 3"),
        ("lit.p6.oct_images", 2938, 0, False, "P6 methods"),
        ("lit.p6.dme_middle", 280, 0, False, "P6 Table 2"),
        ("lit.p6.dme_middle_pct", 100 * 280 / 2938, 1, True, "P6 Table 2, derived"),
        ("lit.p7.fda_devices", 3, 0, False, "P7 challenges"),
        ("lit.p7.retfound_pretrain_m", 1.6, 1, False, "P7 foundation models"),
        ("lit.p9.eyes", 320, 0, False, "P9 abstract"),
        ("lit.p9.patients", 160, 0, False, "P9 abstract"),
        ("lit.p9.kappa_clinical", 0.86, 2, False, "P9 Table 4"),
        ("lit.p9.kappa_standard", 0.78, 2, False, "P9 Table 4"),
        ("lit.p9.discrepant_clinical", 22, 0, False, "P9 Table 2"),
        ("lit.p9.discrepant_standard", 34, 0, False, "P9 Table 2"),
        ("lit.p10.studies", 38, 0, False, "P10 results"),
        ("lit.p10.external_validated", 12, 0, False, "P10 results"),
        ("lit.p10.external_pct", 32, 0, True, "P10 results"),
        ("lit.p10.missing_data", 4, 0, False, "P10 TRIPOD"),
        ("lit.p10.missing_data_pct", 11, 0, True, "P10 TRIPOD"),
        ("lit.p11.pooled_sensitivity", 0.95, 2, False, "P11 meta-analysis"),
        ("lit.p11.fn_derived_sensitivity", 0.06, 2, False, "P11, 676/(676+9969)"),
        ("lit.p12.specificity_low", 14.25, 2, True, "P12 Table"),
        ("lit.p12.specificity_high", 96.01, 2, True, "P12 Table"),
        ("lit.p12.kappa_before", 0.65, 2, False, "P12"),
        ("lit.p12.kappa_after", 0.72, 2, False, "P12"),
        ("lit.p12.dme_sensitivity", 26.5, 1, True, "P12"),
        ("lit.p12.dme_kappa", 0.38, 2, False, "P12"),
        ("lit.p12.dr_kappa", 0.81, 2, False, "P12"),
    ]
    for key, val, dec, pct, src_ in LIT:
        led.add(key, val, f"literature, verified at source: {src_}", dec, pct)

    # ---- our own provenance failure, F11: the numbers the thesis quotes about itself
    for key, val, dec, pct, why in (
            ("f11.reversed_decimals", 8, 0, False, "decimals written with the groups swapped"),
            ("f11.floor_written", 69.6, 1, True, "the gated DME floor as first written"),
            ("f11.parta_written", 96.6, 1, True, "Part A exact match as first written"),
            ("f11.overlay_written_lo", 0.78, 2, False, "unreproducible overlay figure, low"),
            ("f11.overlay_written_hi", 0.96, 2, False, "unreproducible overlay figure, high")):
        led.add(key, val, f"FINDINGS.md F11 -- {why}", dec, pct)

    # ---- method constants: values chosen rather than measured. They are in the ledger so a
    # chapter cannot introduce a number from nowhere, and each carries what it is.
    led.add("const.retfound_pretrain_images_m", 1.6,
            "RETFound (Nature 2023) -- unlabelled retinal images in its pretraining corpus, "
            "verified in results/VERIFICATION-P8-RETFound.md", 1)
    led.add("const.sigmoid_default", 0.5, "the untuned decode threshold on a sigmoid output", 1)
    led.add("const.corr_runner_up_margin", 0.01,
            "src/idrid_derivation_gate.py: the band within which the runner-up correlation "
            "sits, which is why pixel correlation alone cannot confirm a crosswalk match", 2)

    # ---- constants of the sources, not measurements of ours. They are in the ledger so a
    # chapter cannot introduce a number from nowhere, and each carries where it comes from.
    for key, val, why in (
            ("const.idrid_width", 4288, "IDRiD fundus frame width in pixels"),
            ("const.idrid_height", 2848, "IDRiD fundus frame height in pixels"),
            ("const.byte_max", 255, "8-bit intensity range, used for the overlay difference"),
            ("const.messidor1_named", 687, "data/DATASETS.md: files in our Messidor-2 mirror "
                                           "carrying Messidor-1 naming"),
            ("const.messidor2_named", 1057, "data/DATASETS.md: files carrying the Messidor-2 "
                                            "convention")):
        led.add(key, val, why, 0)

    # ---- F9: what the trained pipeline is worth against frozen features and a linear head.
    # Both sides are read from the ledger's own rows so the subtraction cannot drift.
    if "probe.dr.b" in led.rows and "champion_vs_e10.dr.qwk.a" in led.rows:
        champ = led.rows["champion_vs_e10.dr.qwk.a"]["value"]
        frozen = led.rows["probe.dr.b"]["value"]
        led.add("pipeline_worth.qwk", champ - frozen, "derived: champion - frozen probe", 4,
                note="F9 -- same images, folds, decoding and calibration on both sides")

    # ---- label distributions, computed from the corpora rather than remembered
    if a.datasets:
        from tune_thresholds import load
        rows, _, _, _ = load(f"runs/{selected}", a.datasets)
        three = [r for r in rows if r.get("dme") is not None
                 and r.get("dme_label_space") == "3class"]
        gated = [r for r in three if r["dr"] >= 1]
        src = "runs/%s + corpora" % selected
        led.add("pool.n", len(rows), src, 0)
        for corpus in sorted({r["corpus"] for r in rows}):
            led.add(f"pool.n_{corpus.replace('-', '').lower()}",
                    sum(r["corpus"] == corpus for r in rows), src, 0)
        led.add("dme.n_3class", len(three), src, 0)
        led.add("dme.n_gated", len(gated), src, 0)
        for grade in (0, 1, 2):
            led.add(f"dme.n_grade{grade}", sum(r["dme"] == grade for r in three), src, 0)
        for name, subset in (("ungated", three), ("gated", gated)):
            counts = [sum(r["dme"] == g for r in subset) for g in (0, 1, 2)]
            led.add(f"dme.floor_{name}", 100 * max(counts) / len(subset), src, 1, True)
        # the gate's own premise: no DME-positive image carries DR grade 0
        dr0 = [r for r in three if r["dr"] == 0]
        led.add("dme.n_dr0", len(dr0), src, 0)
        led.add("dme.n_dr0_positive", sum(r["dme"] > 0 for r in dr0), src, 0)

        # ---- CORN's conditional subsets: the third threshold trains only on rows whose
        # grade already exceeds 2, and that count is why the head collapses under this class
        # imbalance. Computed here rather than quoted, so it moves if the pool moves.
        led.add("corn.threshold3_rows", sum(r["dr"] >= 3 for r in rows if r["dr"] is not None),
                src, 0, note="rows available to CORN's third conditional threshold")

        # ---- the champion's own per-class recall at matched calibration. Chapter 4 quotes
        # these next to the headline because a model can hold its accuracy while going blind
        # to a rare class -- which is exactly what grade 1 does here (F1, F5).
        import numpy as np
        from compare_matched import expected_grade, crossfit_grades, head_arrays, N_DR, N_DME
        import metrics as M
        rows2, folds2, dr_l, dme_l = load(f"runs/{selected}", a.datasets)
        for head, logits, k in (("dr", dr_l, N_DR), ("dme_ungated", dme_l, N_DME)):
            score, y, keep = head_arrays(rows2, logits, head)
            pred = crossfit_grades(score, y, folds2[keep], k, 0)
            led.add(f"champion.{head}.qwk", M.quadratic_weighted_kappa(y, pred, k), src, 4)
            led.add(f"champion.{head}.accuracy", 100 * M.accuracy(y, pred), src, 1, True)
            led.add(f"champion.{head}.n", len(y), src, 0)
            for g in range(k):
                m = y == g
                led.add(f"champion.{head}.support.grade{g}", int(m.sum()), src, 0)
                led.add(f"champion.{head}.recall.grade{g}",
                        100 * float((pred[m] == g).mean()) if m.any() else 0.0, src, 1, True)

    os.makedirs(GEN, exist_ok=True)
    json.dump(led.rows, open(LEDGER, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {LEDGER}  ({len(led.rows)} numbers)")
    for k in sorted(led.rows):
        print(f"  {k:44s} {led.rows[k]['fa']:>12s}   {led.rows[k]['value']}")


if __name__ == "__main__":
    main()
