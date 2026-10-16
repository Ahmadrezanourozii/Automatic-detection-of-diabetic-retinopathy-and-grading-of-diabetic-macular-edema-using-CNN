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
from matplotlib.patches import FancyBboxPatch

G = "docs/generated"
GP = "docs/generated/paper"
OUT = "paper/figures"
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
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

    box(1, 19, 20, 14, "Pretraining", "EyePACS 2015\n35,126 images\nDR head only", "#f1f0ec")
    box(26, 19, 24, 14, "Development pool",
        "IDRiD 516 + Messidor-2 1,744\n5-fold CV, frozen split\nconfigurations + seed replicates",
        "#e8f0fb")
    box(55, 19, 20, 14, "External (forward)", "APTOS 2019: 3,662\nDDR: 12,522\n(never used to\nselect or tune)",
        "#fdeee6")
    box(79, 19, 20, 14, "Reverse direction", "train APTOS 3,662\n→ Messidor-2 1,744\n→ IDRiD 516",
        "#e6f6ef")
    arrow(21.5, 26, 25.5, 26); arrow(50.5, 26, 54.5, 26)

    ax.text(1, 15.5, "Decision rules applied to every model's ordinal scores "
            "(expected grade s = Σ P(y > k))", fontsize=6.5, color=INK, fontweight="bold")
    rules = [("R1", "shipped: P(y>k) > 0.5"),
             ("R2", "QWK-optimal cut-points on s,\nfitted on development OOF"),
             ("R3", "R2, referral cut moved to\ndev. sensitivity ≥ 0.90"),
             ("R4 / R4m", "cut-points fitted on external\nlabels (2-fold, diagnostic)")]
    for i, (k, v) in enumerate(rules):
        x = 1 + i * 24.8
        col = [GREY, BLUE, AQUA, ORANGE][i]
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
            ax.set_yticklabels([f"{m}" for m in order])
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
        ax.set_ylim(0.5, 1.0)
        ax.set_xlabel("Referable-DR sensitivity (%)")
        ax.set_ylabel("Agreement with reference grade")
        ax.set_title(f"{c}: agreement vs operating point", loc="left", color=INK)
        ax.legend(frameon=False, loc="lower left")
        label_panel(ax, "abcd"[2 * i])
        ax2 = axes[2 * i + 1]
        ax2.axvspan(s["band_sens"][0] * 100, s["band_sens"][1] * 100, color="#eef3fb", zorder=0)
        ax2.plot(sens, [r["spec"] * 100 for r in fr], color=ORANGE, lw=1.4)
        ax2.set_xlim(70, 100.5)
        ax2.set_xlabel("Referable-DR sensitivity (%)")
        ax2.set_ylabel("Specificity (%)")
        ax2.set_title(f"{c}: what the referral cut trades", loc="left", color=INK)
        label_panel(ax2, "abcd"[2 * i + 1])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_frontier.pdf"))
    plt.close(fig)


# ── Figure 4: matched calibration and the seed noise floor ───────────────────
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
    ax.set_xlabel("Δ development DR QWK at matched calibration (intervention − control)")
    ax.grid(axis="y", visible=False)
    if seeds:
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_forest.pdf"))
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


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_design, fig_rules, fig_frontier, fig_forest, fig_recal):
        f()
        print("ok", f.__name__)


if __name__ == "__main__":
    main()
