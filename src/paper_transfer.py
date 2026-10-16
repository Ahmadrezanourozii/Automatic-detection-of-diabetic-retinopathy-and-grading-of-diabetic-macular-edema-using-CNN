"""
paper_transfer.py — operating points: do they transfer, and can the grading metric see them?

Two analyses, both from archived predictions only.

1. TRANSFER GAP (T1 generalised to every model and external corpus; pre-registered O8 for DDR).
   A referral threshold on P(grade >= 2) is fitted on the development OOF predictions to reach
   a target sensitivity, then applied unchanged to the external corpus. The result is how far
   the achieved sensitivity lands from the target. Uses t1_transfer_gap.fit_threshold, so the
   APTOS numbers reproduce docs/generated/t1_transfer_gap.md exactly.

2. METRIC INVARIANCE (the frontier). With the other cut-points held at their QWK-optimal
   external values (diagnostic), sweep only the referral cut and record QWK, Cohen's kappa,
   accuracy, sensitivity and specificity. The "QWK band" is the set of referral cuts whose QWK
   is within 0.01 of the best: everything inside it is indistinguishable to the grading
   metric, and we report how much the referral decision varies inside it.

Writes docs/generated/paper/transfer_<corpus>.json.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_rules as P
from paper_phase0 import load_model, load_ext
# Verbatim copies of t1_transfer_gap.fit_threshold / sens_spec / TARGETS, which cannot be
# imported here because that module pulls in torch. The APTOS rows this script writes are
# checked against docs/generated/t1_transfer_gap.json in main(), so a divergence would fail.
TARGETS = [
    (0.80, "A", "British Diabetic Association 1997 (provenance unverified at primary source)"),
    (0.85, "B", "IDx-DR pivotal trial pre-specified superiority endpoint (> 85 %)"),
    (0.872, "B*", "IDx-DR pivotal trial achieved sensitivity"),
    (0.955, "C", "EyeArt pivotal trial reported sensitivity (not a standard)"),
]


def sens_spec(score, y_ref, thr):
    pred = score > thr
    Pn, Nn = y_ref.sum(), (~y_ref).sum()
    return (pred & y_ref).sum() / max(1, Pn), (~pred & ~y_ref).sum() / max(1, Nn)


def fit_threshold(score, y_ref, target):
    order = np.argsort(-score)
    tp = np.cumsum(y_ref[order])
    sens = tp / max(1, y_ref.sum())
    i = int(np.argmax(sens >= target)) if (sens >= target).any() else len(sens) - 1
    return float(score[order][i])

BAND = 0.01


def transfer(dev, ext, head, rng_seed=0, n_boot=2000):
    ds = P.p_greater(dev["dr"], head)[:, 1]
    dy = dev["y"] >= 2
    xs = P.p_greater(ext["dr"], head)[:, 1]
    xy = ext["y"] >= 2
    rng = np.random.default_rng(rng_seed)
    rows = []
    for target, tag, _ in TARGETS:
        thr = fit_threshold(ds, dy, target)
        d_sens, d_spec = sens_spec(ds, dy, thr)
        x_sens, x_spec = sens_spec(xs, xy, thr)
        boots = []
        n = len(xs)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            boots.append(sens_spec(xs[idx], xy[idx], thr)[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"target": target, "tag": tag, "threshold": thr,
                     "dev_sens": float(d_sens), "dev_spec": float(d_spec),
                     "ext_sens": float(x_sens), "ext_spec": float(x_spec),
                     "ext_sens_ci": [float(lo), float(hi)], "gap": float(x_sens - target)})
    return rows


def frontier(ext, head):
    s = P.expected_grade(ext["dr"], head)
    cuts, best = P.fit_cuts(s, ext["y"], "qwk", grid=P.GRID_EXT)
    rows = P.referral_frontier(ext["dr"], ext["y"], head, cuts, n=161)
    qmax = max(r["qwk"] for r in rows)
    band = [r for r in rows if r["qwk"] >= qmax - BAND]
    n_ref = int((ext["y"] >= 2).sum())
    summary = {
        "qwk_max": qmax, "band_width": BAND, "n_in_band": len(band),
        "band_sens": [min(r["sens"] for r in band), max(r["sens"] for r in band)],
        "band_spec": [min(r["spec"] for r in band), max(r["spec"] for r in band)],
        "band_kappa": [min(r["kappa"] for r in band), max(r["kappa"] for r in band)],
        "band_missed_per_1000_referable": [
            1000 * (1 - max(r["sens"] for r in band)), 1000 * (1 - min(r["sens"] for r in band))],
        "band_missed_count": [round(n_ref * (1 - max(r["sens"] for r in band))),
                              round(n_ref * (1 - min(r["sens"] for r in band)))],
        "n_referable": n_ref,
        "base_cuts": [round(c, 4) for c in cuts],
    }
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--splits", default="data/splits/dev_v1.json")
    ap.add_argument("--pred-root", default="results/predictions")
    ap.add_argument("--corpus", default="APTOS")
    ap.add_argument("--models", default="E08,E11FULL")
    ap.add_argument("--out-dir", default="docs/generated/paper")
    a = ap.parse_args()
    labels = P.dev_labels(a.labels, a.splits)
    reg = json.load(open(os.path.join(a.pred_root, "registry.json")))
    out = {"corpus": a.corpus, "models": {}}
    for m in a.models.split(","):
        if a.corpus not in reg.get(m, {}).get("external", {}):
            print(f"{m}: no {a.corpus} predictions, skipped")
            continue
        head = reg[m]["head"]
        dev = load_model(a.pred_root, m, labels)
        ext = load_ext(a.pred_root, m, a.corpus)
        tr = transfer(dev, ext, head)
        fr, summ = frontier(ext, head)
        out["models"][m] = {"transfer": tr, "frontier": fr, "frontier_summary": summ}
        print(f"\n{m} on {a.corpus}")
        for r in tr:
            print(f"  target {r['target']*100:5.1f}%  dev {r['dev_sens']*100:5.1f}%  ->  "
                  f"{a.corpus} {r['ext_sens']*100:5.2f}% [{r['ext_sens_ci'][0]*100:.2f}, "
                  f"{r['ext_sens_ci'][1]*100:.2f}]  spec {r['ext_spec']*100:5.2f}%  "
                  f"gap {r['gap']*100:+.2f}")
        print(f"  QWK band (within {BAND} of max {summ['qwk_max']:.4f}): sens "
              f"{summ['band_sens'][0]*100:.1f}-{summ['band_sens'][1]*100:.1f}%, spec "
              f"{summ['band_spec'][0]*100:.1f}-{summ['band_spec'][1]*100:.1f}%, kappa "
              f"{summ['band_kappa'][0]:.3f}-{summ['band_kappa'][1]:.3f}; missed referable "
              f"{summ['band_missed_count'][0]}-{summ['band_missed_count'][1]} of "
              f"{summ['n_referable']}")
    if a.corpus == "APTOS" and "E08" in out["models"]:
        # reproduction check against the archived T1 analysis
        ref = json.load(open("docs/generated/t1_transfer_gap.json"))
        for r0, r1 in zip(ref, out["models"]["E08"]["transfer"]):
            assert abs(r0["ext_sens"] - r1["ext_sens"]) < 1e-9 and \
                abs(r0["threshold"] - r1["threshold"]) < 1e-6, (r0, r1)
        print("\nE08 APTOS transfer rows reproduce docs/generated/t1_transfer_gap.json exactly")
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(a.out_dir, f"transfer_{a.corpus.lower()}.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
