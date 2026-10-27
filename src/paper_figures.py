"""
paper_figures.py — every figure in the journal manuscript, from generated JSON only.

Inputs are the files under docs/generated/ and docs/generated/paper/; nothing is typed by
hand. Vector PDF output to paper/figures/. Colour: the first three slots of the validated
categorical palette (blue, orange, aqua) plus neutral grey, so no figure relies on a pair
that fails the colour-vision checks; every series is also distinguished by marker shape.
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import FancyBboxPatch

G = "docs/generated"
GP = "docs/generated/paper"
OUT = "paper/figures"
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
YELLOW = "#eda100"
# corpora: colour AND marker, because two of the four hues sit below 3:1 on white
CORPUS_STYLE = {"Development": (GREY, "o"), "APTOS": (BLUE, "s"), "DDR": (ORANGE, "^"),
                "DeepDRiD": (AQUA, "D"), "SUSTech": (YELLOW, "v")}
# readable names for the configuration codes used in the repository
MODEL_LABELS = {"E08": "DenseNet121-448 (reference)", "E09": "DenseNet121-224",
                "E10": "DenseNet121-560", "E17NAT": "DenseNet121-640 native",
                "E11FULL": "EfficientNet-B3 (selected)", "E14MAC": "DenseNet121 + macula crop",
                "E15LPFT": "DenseNet121 LP-FT", "E20CORAL": "DenseNet121 CORAL head",
                "E22CORN": "DenseNet121 CORN head", "E21CNXA": "ConvNeXt-T (2 folds)",
                "I24FULL": "RETFound ViT-L/16", "EYEDN": "DenseNet121, EyePACS-trained",
                "EYEEB3": "EfficientNet-B3, EyePACS-trained"}
RULE_STYLE = {"R1": (GREY, "o", "R1 shipped (0.5)"),
              "R2": (BLUE, "s", "R2 dev-tuned (QWK)"),
              "R3": (AQUA, "D", "R3 dev-tuned (sens ≥ 0.90)"),
              "R4m": (ORANGE, "^", "R4m local labels (diagnostic)")}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5,
})
MM = 1 / 25.4


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def label_panel(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right", color=INK)


# ── Figure 1: study design ────────────────────────────────────────────────────
def fig_design():
    fig, ax = plt.subplots(figsize=(183 * MM, 62 * MM))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=MUTED, lw=0.6))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=7,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 4.6, body, ha="center", va="top", fontsize=6,
                color=INK, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color=MUTED))

    box(1, 19, 20, 14, "EyePACS 2015", "35,126 images\npretraining; and two models\ntrained on 4 of 5 folds\n(about 28,000 images)",
        "#f1f0ec")
    box(25, 19, 24, 14, "Development pool",
        "IDRiD 516 + Messidor-2 1,744\n5-fold CV, frozen split\n10 models + 5 seeds",
        "#e8f0fb")
    box(53, 19, 23, 14, "External (forward)", "APTOS 2019  3,662\nDDR  12,522\nDeepDRiD  2,000\nSUSTech-SYSU  1,151",
        "#fdeee6")
    box(80, 19, 19, 14, "Reverse direction", "train APTOS 3,662\n→ Messidor-2 1,744\n→ IDRiD 516",
        "#e6f6ef")
    arrow(21.5, 26, 24.5, 26); arrow(49.5, 26, 52.5, 26)

    ax.text(1, 15.5, "Decision rules applied to every model's ordinal scores "
            "(expected grade s = Σ P(y > k))", fontsize=6.5, color=INK, fontweight="bold")
    rules = [("R1", "shipped:\nP(y>k) > 0.5"),
             ("R2 / R3", "cut-points fitted on\ndevelopment predictions"),
             ("R4 / R4m", "cut-points fitted on external\nlabels (2-fold, diagnostic)"),
             ("R5 / R5b", "label-free prior adaptation\n(EM; black-box shift)")]
    for i, (k, v) in enumerate(rules):
        x = 1 + i * 24.8
        col = [GREY, BLUE, ORANGE, AQUA][i]
        ax.add_patch(FancyBboxPatch((x, 2), 23, 10.5, boxstyle="round,pad=0.3,rounding_size=1",
                                    fc="white", ec=col, lw=1.0))
        ax.text(x + 1.2, 10.8, k, fontsize=7, fontweight="bold", color=INK, va="top")
        ax.text(x + 1.2, 7.6, v, fontsize=6, color=INK, va="top", linespacing=1.3)
    fig.savefig(os.path.join(OUT, "fig1_design.pdf"))
    plt.close(fig)


# ── Figure 2: decision rule versus model, per external corpus ────────────────
def fig_rules():
    corpora = [c for c in ("APTOS", "DDR") if load(f"{GP}/phase0_{c.lower()}.json")]
    if not corpora:
        return
    fig, axes = plt.subplots(len(corpora), 2, figsize=(183 * MM, (52 + 4) * MM * len(corpora)),
                             squeeze=False)
    for r, c in enumerate(corpora):
        d = load(f"{GP}/phase0_{c.lower()}.json")
        models = d["models"]
        order = sorted(models, key=lambda m: d["dev"][m]["qwk_matched"])
        for col, (metric, title, f) in enumerate((
                ("mild", "Mild DR recall (%)", lambda b: 100 * b["recall"][1]),
                ("sens", "Referable-DR sensitivity (%)", lambda b: 100 * b["ref_sens"]))):
            ax = axes[r, col]
            for yi, m in enumerate(order):
                vals = [f(d["rule_matrix"][m][x]) for x in ("R1", "R2", "R3", "R4m")]
                ax.plot([min(vals), max(vals)], [yi, yi], color="#d6d5d0", lw=1.2, zorder=1)
                for x, v in zip(("R1", "R2", "R3", "R4m"), vals):
                    cc, mk, _ = RULE_STYLE[x]
                    ax.scatter(v, yi, s=16, marker=mk, color=cc, zorder=3,
                               edgecolors="white", linewidths=0.4)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([MODEL_LABELS.get(m, m) if col == 0 else "" for m in order])
            ax.set_xlabel(title)
            ax.set_title(f"{c} (n = {d['rule_matrix'][order[0]]['R1']['support'] and sum(d['rule_matrix'][order[0]]['R1']['support']):,})",
                         loc="left", color=INK)
            ax.grid(axis="y", visible=False)
            label_panel(ax, "abcd"[r * 2 + col])
    handles = [plt.Line2D([], [], marker=mk, color=cc, ls="", markersize=4.5, label=lab)
               for cc, mk, lab in RULE_STYLE.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "fig2_rules.pdf"))
    plt.close(fig)


# ── Figure 3: the grading metric cannot see the referral operating point ─────
def fig_frontier():
    panels = []
    for c in ("APTOS", "DDR"):
        d = load(f"{GP}/transfer_{c.lower()}.json")
        if d and "E11FULL" in d["models"]:
            panels.append((c, d["models"]["E11FULL"]))
    if not panels:
        return
    fig, axes = plt.subplots(1, 2 * len(panels), figsize=(183 * MM, 58 * MM))
    for i, (c, m) in enumerate(panels):
        fr = m["frontier"]; s = m["frontier_summary"]
        sens = np.array([r["sens"] for r in fr]) * 100
        ax = axes[2 * i]
        ax.axvspan(s["band_sens"][0] * 100, s["band_sens"][1] * 100, color="#eef3fb", zorder=0)
        ax.plot(sens, [r["qwk"] for r in fr], color=BLUE, lw=1.4, label="QWK (5 grades)")
        ax.plot(sens, [r["kappa"] for r in fr], color=AQUA, lw=1.4, ls="--",
                label="Cohen's κ (5 grades)")
        ax.set_xlim(70, 100.5)
        ax.set_ylim(0.3, 1.0)
        ax.set_xlabel("Referable-DR sensitivity (%)")
        ax.set_ylabel("Agreement with reference grade")
        ax.set_title(f"{c}: agreement", loc="left", color=INK, pad=8)
        ax.legend(frameon=False, loc="lower left")
        label_panel(ax, "abcd"[2 * i])
        ax2 = axes[2 * i + 1]
        ax2.axvspan(s["band_sens"][0] * 100, s["band_sens"][1] * 100, color="#eef3fb", zorder=0)
        ax2.plot(sens, [r["spec"] * 100 for r in fr], color=ORANGE, lw=1.4)
        ax2.set_xlim(70, 100.5)
        ax2.set_xlabel("Referable-DR sensitivity (%)")
        ax2.set_ylabel("Specificity (%)")
        ax2.set_title(f"{c}: specificity traded", loc="left", color=INK, pad=8)
        label_panel(ax2, "abcd"[2 * i + 1])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_frontier.pdf"))
    plt.close(fig)


# ── Figure 3: transfer gap of development-fitted referral thresholds ─────────
def fig_transfer():
    fig, ax = plt.subplots(figsize=(120 * MM, 72 * MM))
    ax.plot([0, 100.5], [0, 100.5], color=GREY, lw=0.8, ls="--", zorder=1)
    cs = [c for c in ("APTOS", "DDR", "DeepDRiD", "SUSTech") if load(f"{GP}/transfer_{c.lower()}.json")]
    for j, c in enumerate(cs):
        col, mk = CORPUS_STYLE[c]
        dx = (j - (len(cs) - 1) / 2) * 0.3
        d = load(f"{GP}/transfer_{c.lower()}.json")
        if not d or "E08" not in d["models"]:
            continue
        rows = d["models"]["E08"]["transfer"]
        x = np.array([100 * r["target"] for r in rows]) + dx
        y = np.array([100 * r["ext_sens"] for r in rows])
        lo = np.array([100 * r["ext_sens_ci"][0] for r in rows]); hi = np.array([100 * r["ext_sens_ci"][1] for r in rows])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt=mk, color=col, ms=4.5, lw=1.0, capsize=2,
                    label=f"{c}", zorder=3, markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xlabel("Target sensitivity, fitted on development data (%)")
    ax.set_ylabel("Achieved external sensitivity (%)")
    ax.set_xlim(78, 97.5)
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(40, 101)
    ax.text(79, 77.0, "perfect transfer", color=MUTED, fontsize=6, rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_transfer.pdf"))
    plt.close(fig)


# ── Figure S1: matched decision rule and the seed noise floor ───────────────────
def fig_forest():
    rows = load(f"{GP}/interventions.json")
    if not rows:
        return
    seeds = load(f"{GP}/seeds.json")
    fig, ax = plt.subplots(figsize=(120 * MM, (14 + 6.5 * len(rows)) * MM))
    if seeds and seeds.get("dev_matched_qwk_sd") is not None:
        band = 2 * np.sqrt(2) * seeds["dev_matched_qwk_sd"]
        ax.axvspan(-band, band, color="#f1f0ec", zorder=0, label="±2√2·SD across seeds")
    ax.axvline(0, color=MUTED, lw=0.6)
    for i, r in enumerate(rows[::-1]):
        col = BLUE if r["verdict"] == "confirmed" else (ORANGE if r["lo"] > 0 or r["hi"] < 0 else GREY)
        ax.plot([r["lo"], r["hi"]], [i, i], color=col, lw=1.4)
        ax.scatter(r["diff"], i, color=col, s=14, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows[::-1]])
    ax.set_xlabel("Δ development DR QWK, matched decision rule (intervention − control)")
    ax.grid(axis="y", visible=False)
    if seeds:
        ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_forest.pdf"))
    plt.close(fig)


# ── Figure 5: how many local labels recalibration needs ───────────────────────
def fig_recal():
    d = load(f"{G}/recalibration_curve.json")
    if not d:
        return
    r = d["results"]["E08X2"]
    ns = [int(n) for n in d["sizes"]]
    gain = np.array([r[str(n)]["macro_recall_gain"] for n in ns])
    harm = np.array([r[str(n)]["p_harmful"] for n in ns]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(150 * MM, 52 * MM))
    ax = axes[0]
    ax.fill_between(ns, gain[:, 1], gain[:, 2], color="#e8f0fb", lw=0)
    ax.plot(ns, gain[:, 0], color=BLUE, marker="o", ms=3.5, lw=1.4)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Locally labelled images"); ax.set_ylabel("Macro-recall gain (points)")
    ax.set_title("Mean gain and 95% range over 200 draws", loc="left", color=INK)
    label_panel(ax, "a")
    ax = axes[1]
    ax.bar(range(len(ns)), harm, color=ORANGE, width=0.6)
    for i, h in enumerate(harm):
        ax.text(i, h + 0.8, f"{h:.1f}%", ha="center", fontsize=6, color=INK)
    ax.set_xticks(range(len(ns))); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Locally labelled images"); ax.set_ylabel("Probability of harm (%)")
    ax.set_title("Draws in which recalibration made it worse", loc="left", color=INK)
    ax.grid(axis="x", visible=False)
    label_panel(ax, "b")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_recalibration.pdf"))
    plt.close(fig)



# ── Figure 2: ranking transfers, the score-to-grade mapping does not ─────────
def fig_shift():
    d = load(f"{GP}/revision.json")
    if not d:
        return
    corp = [c for c in ("APTOS", "DDR", "DeepDRiD", "SUSTech") if c in d["corpora"]]
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 62 * MM))
    ax = axes[0]
    series = [("Development", [d["dev"][m]["auc_k"] for m in d["models"]])] + \
             [(c, [d["corpora"][c]["models"][m]["auc_k"] for m in d["corpora"][c]["models"]]) for c in corp]
    for j, (name, rows) in enumerate(series):
        a = np.array(rows)
        col, mk = CORPUS_STYLE[name]
        x = np.arange(4) + (j - (len(series) - 1) / 2) * 0.13
        med = np.median(a, axis=0)
        ax.vlines(x, a.min(axis=0), a.max(axis=0), color=col, lw=1.0)
        ax.scatter(x, med, marker=mk, s=18, color=col, edgecolors="white", linewidths=0.4,
                   zorder=3, label=name)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["P(y>0)\nany DR", "P(y>1)\nreferable", "P(y>2)", "P(y>3)\nproliferative"])
    ax.set_ylabel("Area under the ROC curve")
    lo = min(min(min(r) for r in rows) for _, rows in series)
    ax.set_ylim(np.floor(lo * 20) / 20 - 0.01, 1.005)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    ax.set_title("Ranking at every ordinal threshold", loc="left", color=INK)
    ax.grid(axis="x", visible=False)
    label_panel(ax, "a")
    ax = axes[1]
    grades = np.arange(5)
    ax.plot([-0.3, 4.3], [-0.3, 4.3], color="#d6d5d0", lw=0.8, ls="--", zorder=0)
    series = [("Development", [d["dev"][m]["es_by_grade"] for m in d["models"]])] + \
             [(c, [d["corpora"][c]["models"][m]["es_by_grade"] for m in d["corpora"][c]["models"]]) for c in corp]
    for j, (name, rows) in enumerate(series):
        a = np.array(rows)
        col, mk = CORPUS_STYLE[name]
        x = grades + (j - (len(series) - 1) / 2) * 0.11
        ax.vlines(x, np.nanmin(a, axis=0), np.nanmax(a, axis=0), color=col, lw=1.0)
        ax.plot(x, np.nanmedian(a, axis=0), color=col, lw=0.8, marker=mk, ms=3.6,
                markeredgecolor="white", markeredgewidth=0.4, label=name)
    ax.set_xticks(grades)
    ax.set_xticklabels(["No DR", "Mild", "Moderate", "Severe", "PDR"])
    ax.set_xlabel("Reference grade")
    ax.set_ylabel("Mean expected grade (model score)")
    ax.set_title("Score given the true grade", loc="left", color=INK)
    ax.grid(axis="x", visible=False)
    label_panel(ax, "b")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_shift.pdf"))
    plt.close(fig)


# ── Figure 5: how many local labels a referral threshold needs ───────────────
def fig_samplesize():
    """a: risk of harm against the number of REFERABLE cases in the labelled sample -- closed
    form (distribution-free) and simulation on each corpus; b: against the number of images at
    screening prevalences, beside the pre-specified simulation on the enriched external sets."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paper_posthoc import harm_exact, harm_prevalence_exact
    ph, rv = load(f"{GP}/posthoc.json"), load(f"{GP}/revision.json")
    if not ph or not rv:
        return
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 66 * MM))
    ax = axes[0]
    ks = np.arange(20, 301, 20)
    ax.plot(ks, [100 * harm_exact(int(k)) for k in ks], color=INK, lw=1.0, zorder=2,
            label="Closed form (any model, any population)")
    for j, c in enumerate(("APTOS", "DDR", "DeepDRiD", "SUSTech")):
        bc = ph["corpora"][c]["models"]["E11FULL"]["by_cases"]
        kk = np.array(sorted(int(k) for k in bc), dtype=float)
        col, mk = CORPUS_STYLE[c]
        ax.plot(kk + (j - 1.5) * 2.2, [100 * bc[str(int(k))]["p_harm"] for k in kk], ls="none",
                marker=mk, ms=3.8, color=col, markeredgecolor="white", markeredgewidth=0.4,
                zorder=3, label=f"{c} (simulated)")
    for y in (10, 5):
        ax.axhline(y, color=MUTED, lw=0.6, ls=":")
    ax.set_xlim(0, 310); ax.set_ylim(0, 45)
    ax.set_xlabel("Referable cases in the labelled sample")
    ax.set_ylabel("Probability that sensitivity falls below 80% (%)")
    ax.set_title("Threshold set for 85% sensitivity on the sample", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="x", visible=False)
    label_panel(ax, "a")

    ax = axes[1]
    ns = np.unique(np.geomspace(50, 4000, 60).astype(int))
    blues = {0.05: "#8db6ea", 0.10: BLUE, 0.20: "#173f73"}
    for p_, col in blues.items():
        h = [100 * harm_prevalence_exact(int(n), p_) for n in ns]
        ax.plot(ns, h, color=col, lw=1.1, label=f"{int(100 * p_)}% referable")
    prev = np.mean([ph["corpora"][c]["n_ref"] / ph["corpora"][c]["n"] for c in ph["corpora"]])
    ax.plot(ns, [100 * harm_prevalence_exact(int(n), prev) for n in ns], color=GREY, lw=1.0, ls="--",
            label=f"{100 * prev:.0f}% referable (external sets as sampled)")
    for j, c in enumerate(("APTOS", "DDR", "DeepDRiD", "SUSTech")):
        ss = rv["corpora"][c]["models"]["E11FULL"].get("sample_size", {})
        nn = np.array(sorted(int(n) for n in ss), dtype=float)
        col, mk = CORPUS_STYLE[c]
        ax.plot(nn * (1 + (j - 1.5) * 0.04), [100 * ss[str(int(n))]["p_harm"] for n in nn], ls="none",
                marker=mk, ms=3.4, color=col, markeredgecolor="white", markeredgewidth=0.4, zorder=3)
    ax.legend(frameon=False, loc="upper right")
    for y in (10, 5):
        ax.axhline(y, color=MUTED, lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 200, 500, 1000, 2000, 4000])
    ax.set_xticklabels(["50", "100", "200", "500", "1000", "2000", "4000"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(45, 4400); ax.set_ylim(0, 50)
    ax.set_xlabel("Consecutive images labelled")
    ax.set_ylabel("Probability that sensitivity falls below 80% (%)")
    ax.set_title("At screening prevalence", loc="left", color=INK)
    ax.grid(axis="x", visible=False)
    label_panel(ax, "b")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_samplesize.pdf"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_design, fig_rules, fig_shift, fig_transfer, fig_frontier, fig_forest, fig_recal,
              fig_samplesize):
        f()
        print("ok", f.__name__)


if __name__ == "__main__":
    main()
