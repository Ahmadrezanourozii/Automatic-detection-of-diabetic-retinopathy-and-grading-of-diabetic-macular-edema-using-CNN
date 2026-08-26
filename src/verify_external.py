"""
verify_external.py — re-derive the external claim from the archived artifact.

The APTOS number is the single most important claim in this project, and it came out of a
run whose output directory was also carrying another run's results.json (ISSUES.md §20).
"It looks right" is not a standard. This script re-derives every reported figure from the
archived confusion matrix, checks the evaluated cohort against the APTOS labels on disk, and
checks that cohort is disjoint from the development pool.

What it CAN verify: n, the per-class supports, accuracy, QWK, macro-F1, per-class recall,
the majority floor, referable sensitivity and specificity, and disjointness from the
development pool.

What it CANNOT verify, and says so: the bootstrap intervals. Those need per-image predictions
and groups, and `eval_external.py` archived only the aggregate confusion matrix. That is a
gap in the artifact, not a doubt about the point estimates — and it is fixed going forward.

Usage:
    python src/verify_external.py --run runs/E08X --datasets <root>
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpora
import metrics as M

OK, BAD = [], []


def check(cond, name, detail=""):
    (OK if cond else BAD).append(name)
    print(f"[{'  ok  ' if cond else ' FAIL '}] {name}" + (f"\n           {detail}" if detail else ""))


def close(a, b, tol=5e-4):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/E08X")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--corpus", default="APTOS")
    a = ap.parse_args()

    p = os.path.join(a.run, f"external_{a.corpus.lower()}.json")
    j = json.load(open(p))
    rep = j["metrics"]["dr"]
    cm = np.array(rep["confusion"], dtype=np.int64)
    print(f"verifying {p}\n  corpus={j['corpus']} folds_ensembled={j['n_folds_ensembled']} "
          f"tta={j['tta']}\n")

    # ── 1. the cohort is the whole external corpus, and nothing else ──────────
    ext = corpora.build(a.datasets, (a.corpus,))
    check(len(ext) > 0, f"{a.corpus} is readable from disk", f"{len(ext)} images found")
    if ext:
        check(int(cm.sum()) == len(ext),
              "evaluated cohort size equals the corpus size",
              f"confusion matrix totals {int(cm.sum())}, corpus holds {len(ext)}")
        true_dist = np.bincount([r["dr"] for r in ext], minlength=cm.shape[0])
        check(np.array_equal(cm.sum(axis=1), true_dist),
              "per-class supports match the corpus labels exactly",
              f"archived {cm.sum(axis=1).tolist()} vs labels {true_dist.tolist()}")

        # ── 2. disjoint from the development pool ─────────────────────────────
        dev = corpora.build(a.datasets, ("IDRiD", "Messidor-2"))
        clash = {r["uid"] for r in ext} & {r["uid"] for r in dev}
        check(not clash, "no development image appears in the external cohort",
              f"{len(dev)} development images, {len(ext)} external, "
              f"{len(clash)} shared" + (f" e.g. {sorted(clash)[:3]}" if clash else ""))

    # ── 3. every reported figure regenerates from the confusion matrix ────────
    n = int(cm.sum())
    k = cm.shape[0]
    y = np.repeat(np.arange(k), cm.sum(axis=1))
    pred = np.concatenate([np.repeat(np.arange(k), cm[i]) for i in range(k)])
    check(len(y) == n and len(pred) == n, "confusion matrix expands to n predictions")

    check(close(M.accuracy(y, pred), rep["accuracy"]),
          "accuracy regenerates", f"recomputed {M.accuracy(y, pred):.6f} vs archived {rep['accuracy']:.6f}")
    q = M.quadratic_weighted_kappa(y, pred, k)
    check(close(q, rep["qwk"]), "QWK regenerates",
          f"recomputed {q:.6f} vs archived {rep['qwk']:.6f}")
    check(close(M.macro_f1(y, pred, k), rep["macro_f1"]), "macro-F1 regenerates")
    check(close(M.majority_floor(y), rep["majority_floor"]), "majority floor regenerates",
          f"recomputed {M.majority_floor(y)*100:.2f}% vs archived {rep['majority_floor']*100:.2f}%")

    rec, sup = M.per_class_recall(y, pred, k)
    check(all(close(x, r) for x, r in zip(rec, rep["per_class_recall"])),
          "per-class recall regenerates",
          " ".join(f"{x*100:.1f}%" for x in rec if x is not None))
    check(sup == rep["support"], "support regenerates")

    if "referable_sensitivity" in rep:
        s, sp = M.binary_sens_spec(y, pred, 2)
        check(close(s, rep["referable_sensitivity"]) and close(sp, rep["referable_specificity"]),
              "referable-DR sensitivity and specificity regenerate",
              f"recomputed {s*100:.2f}% / {sp*100:.2f}% vs archived "
              f"{rep['referable_sensitivity']*100:.2f}% / {rep['referable_specificity']*100:.2f}%")

    # ── 4. the intervals, when every group is a single image ──────────────────
    #
    # A group bootstrap resamples groups i.i.d. with replacement. When every group holds
    # exactly one image -- true of APTOS, which publishes no patient ids -- that is exactly
    # resampling (y_true, y_pred) pairs i.i.d. from the empirical joint distribution. The
    # confusion matrix IS that distribution. So the interval is fully determined by the
    # archived matrix and needs no per-image file after all.
    #
    # This verifies the interval is CORRECT, not that it is bit-identical: the realized
    # draws depend on row order, which the matrix does not preserve. The comparison is
    # therefore against Monte-Carlo error, which is reported alongside.
    one_per_group = ext and len({r["uid"] for r in ext}) == len(ext)
    if not one_per_group:
        print("\n[ note ] groups are not one image each, so the intervals cannot be "
              "re-derived from the confusion matrix; per-image predictions are needed.")
    else:
        print(f"\n  every group in {a.corpus} is a single image "
              f"({len(ext)} images, {len(ext)} groups), so the group bootstrap is an image "
              f"bootstrap and the confusion matrix determines the interval.")
        g = np.arange(n)
        for key, fn, scale in (
                ("accuracy", M.accuracy, 100),
                ("qwk", lambda t, q: M.quadratic_weighted_kappa(t, q, k), 1)):
            arch = rep.get(key + "_ci95")
            if not arch or arch[0] is None:
                continue
            reps = [M.bootstrap_ci(y, pred, fn, groups=g, n_boot=8000, seed=sd)
                    for sd in (0, 1, 2)]
            los, his = [r[0] for r in reps], [r[1] for r in reps]
            mc = max(np.std(los), np.std(his))          # spread across independent seeds
            agree = (abs(np.mean(los) - arch[0]) <= 4 * mc + 1e-9 and
                     abs(np.mean(his) - arch[1]) <= 4 * mc + 1e-9)
            check(agree, f"{key} interval regenerates from the confusion matrix",
                  f"recomputed [{np.mean(los)*scale:.3f}, {np.mean(his)*scale:.3f}] "
                  f"vs archived [{arch[0]*scale:.3f}, {arch[1]*scale:.3f}]  "
                  f"(Monte-Carlo sd across seeds {mc*scale:.4f})")

    print(f"\n{len(OK)} checks passed, {len(BAD)} failed")
    sys.exit(1 if BAD else 0)


if __name__ == "__main__":
    main()
