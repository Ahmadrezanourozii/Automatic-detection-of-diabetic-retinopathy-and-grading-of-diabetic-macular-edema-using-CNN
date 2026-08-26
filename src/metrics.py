"""
metrics.py — every metric this project reports, in one place.

Numbers appearing in two places must be produced by the same code path. That rule is the
reason this module exists; import it, never re-derive a metric inline.
"""
from __future__ import annotations
import numpy as np


# ── point metrics ─────────────────────────────────────────────────────────────
def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def confusion(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[int(t), int(p)] += 1
    return cm


def quadratic_weighted_kappa(y_true, y_pred, k):
    """
    QWK — the primary metric for both ordinal heads (PROTOCOL.md §4).

    Charges (i-j)^2 for confusing grade i with grade j, so a two-grade error costs four
    times a one-grade error. This is the field standard for DR grading and it is what the
    APTOS/EyePACS leaderboards rank on.

    Returns 0.0 when the expected-disagreement denominator vanishes (a degenerate case that
    occurs when both raters are constant); that is the correct reading — a constant
    predictor carries no ordinal information.
    """
    O = confusion(y_true, y_pred, k).astype(np.float64)
    n = O.sum()
    if n == 0:
        return float("nan")
    W = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / (k - 1) ** 2
    hist_t = O.sum(axis=1)
    hist_p = O.sum(axis=0)
    E = np.outer(hist_t, hist_p) / n
    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return float(1.0 - (W * O).sum() / denom)


def per_class_recall(y_true, y_pred, k):
    """Recall per class. A model can gain accuracy while going blind to a rare class;
    this is how you notice. NaN for a class with no support."""
    cm = confusion(y_true, y_pred, k)
    sup = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diag(cm) / sup
    return [None if s == 0 else float(v) for v, s in zip(r, sup)], sup.tolist()


def macro_f1(y_true, y_pred, k):
    cm = confusion(y_true, y_pred, k)
    f1s = []
    for c in range(k):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def majority_floor(y_true):
    """What you score by always guessing the most common class. Every reported number is
    quoted against this (PROTOCOL.md §5.1)."""
    y = np.asarray(y_true)
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / len(y))


def binary_sens_spec(y_true, y_pred, positive_from):
    """Sensitivity/specificity for the screening decision — 'grade >= positive_from'."""
    t = np.asarray(y_true) >= positive_from
    p = np.asarray(y_pred) >= positive_from
    tp, fn = int((t & p).sum()), int((t & ~p).sum())
    tn, fp = int((~t & ~p).sum()), int((~t & p).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return float(sens), float(spec)


# ── uncertainty ───────────────────────────────────────────────────────────────
def _group_index(groups):
    """Precompute group -> row indices ONCE.

    The obvious implementation rebuilds this inside every bootstrap iteration, which is
    O(n_groups x n_rows) per draw. With 2 260 one-image groups and 2 000 draws that is
    ~10^10 operations and the comparison never finishes — it looks like the data is slow
    when it is the resampler. Hoisting it out makes each draw a single fancy-index.
    """
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(uniq)))
    ends = np.searchsorted(inv[order], np.arange(len(uniq)), side="right")
    return order, starts, ends, len(uniq)


def _resample_groups(index, rng):
    """Resample WHOLE GROUPS with replacement and return the row indices they select.

    This is the point of the module. Resampling rows would treat two eyes of one patient —
    or two fields of one eye — as independent evidence, which makes the interval roughly
    twice too narrow. PROTOCOL.md §4.
    """
    order, starts, ends, n_groups = index
    picked = rng.integers(0, n_groups, size=n_groups)
    counts = ends[picked] - starts[picked]
    if counts.max() == 1:                     # every group is one row: the common case here
        return order[starts[picked]]
    return np.concatenate([order[starts[g]:ends[g]] for g in picked])


def bootstrap_ci(y_true, y_pred, fn, groups=None, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap interval for any metric, resampled over groups."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    groups = np.arange(len(y_true)) if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    index = _group_index(groups)
    stats = []
    for _ in range(n_boot):
        idx = _resample_groups(index, rng)
        stats.append(fn(y_true[idx], y_pred[idx]))
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_diff(y_true, pred_a, pred_b, fn, groups=None,
                          n_boot=2000, alpha=0.05, seed=0):
    """
    Compare two models on the SAME units. Returns (mean_diff, lo, hi, significant).

    If the interval contains zero the models are indistinguishable and must be reported as
    such — not ranked anyway. PROTOCOL.md §4.
    """
    y_true = np.asarray(y_true)
    pred_a, pred_b = np.asarray(pred_a), np.asarray(pred_b)
    groups = np.arange(len(y_true)) if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    index = _group_index(groups)
    diffs = []
    for _ in range(n_boot):
        idx = _resample_groups(index, rng)
        diffs.append(fn(y_true[idx], pred_a[idx]) - fn(y_true[idx], pred_b[idx]))
    lo, hi = np.nanpercentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(diffs)), float(lo), float(hi), bool(lo > 0 or hi < 0)


def report(y_true, y_pred, k, groups=None, seed=0, n_boot=2000, referable_from=None):
    """The standard metric block for one head. Everything reported comes from here."""
    acc = accuracy(y_true, y_pred)
    qwk = quadratic_weighted_kappa(y_true, y_pred, k)
    rec, sup = per_class_recall(y_true, y_pred, k)
    acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, accuracy, groups, n_boot, seed=seed)
    qwk_lo, qwk_hi = bootstrap_ci(
        y_true, y_pred, lambda a, b: quadratic_weighted_kappa(a, b, k),
        groups, n_boot, seed=seed)
    out = {
        "n": int(len(y_true)),
        "accuracy": acc, "accuracy_ci95": [acc_lo, acc_hi],
        "qwk": qwk, "qwk_ci95": [qwk_lo, qwk_hi],
        "macro_f1": macro_f1(y_true, y_pred, k),
        "per_class_recall": rec,
        "support": sup,
        "majority_floor": majority_floor(y_true),
        "beats_floor": bool(acc_lo > majority_floor(y_true)),
        "confusion": confusion(y_true, y_pred, k).tolist(),
    }
    if referable_from is not None:
        s, sp = binary_sens_spec(y_true, y_pred, referable_from)
        out["referable_sensitivity"] = s
        out["referable_specificity"] = sp
    return out
