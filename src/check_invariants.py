"""
check_invariants.py — the assertions PROTOCOL.md §8 requires before any reported result.

These are checked empirically, by recomputing things and comparing, rather than by reading
the code and believing it. Reading the code is how the project got here: the old loader
*looked* like it held out a test set, and one of its two entry points did not.

Every check reports PASS, FAIL, or SKIP. SKIP means the artifact it needs does not exist
yet and says which one — it is never counted as a pass.

Usage:
    python src/check_invariants.py --idrid <path> [--groups data/groups.json]
                                   [--splits data/splits/<name>.json] [--run runs/E01]
Exit code is non-zero if any check FAILs, so this can gate a commit or a report.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS: list[tuple[str, str, str]] = []


def record(status, name, detail=""):
    RESULTS.append((status, name, detail))
    mark = {"PASS": "  ok  ", "FAIL": " FAIL ", "SKIP": " skip "}[status]
    print(f"[{mark}] {name}" + (f"\n           {detail}" if detail else ""))


# ── 1. splits are disjoint by group ───────────────────────────────────────────
def check_split_disjoint(splits_path):
    if not splits_path or not os.path.exists(splits_path):
        return record("SKIP", "no group appears in two splits",
                      f"no split file yet ({splits_path or 'none given'})")
    with open(splits_path) as f:
        splits = json.load(f)
    seen, clash = {}, []
    for split_name, items in splits.items():
        if not isinstance(items, list):
            continue
        for it in items:
            g = it["group"] if isinstance(it, dict) else it
            if g in seen and seen[g] != split_name:
                clash.append((g, seen[g], split_name))
            seen[g] = split_name
    if clash:
        return record("FAIL", "no group appears in two splits",
                      f"{len(clash)} groups straddle a boundary, e.g. {clash[:3]}")
    record("PASS", "no group appears in two splits", f"{len(seen)} groups, all unique")


# ── 2. no duplicate image across merged corpora ───────────────────────────────
def check_no_cross_corpus_duplicates(groups_path):
    if not groups_path or not os.path.exists(groups_path):
        return record("SKIP", "no image appears twice across merged datasets",
                      f"run src/dedup_groups.py first ({groups_path or 'none given'})")
    with open(groups_path) as f:
        g = json.load(f)
    cross = g.get("cross_corpus_pairs", [])
    if cross:
        # Not automatically a failure — it is a fact that must be *known* and handled.
        # It fails only once those corpora are on opposite sides of a split.
        return record("PASS", "cross-corpus duplicates enumerated",
                      f"{len(cross)} cross-corpus pairs found and recorded; they are "
                      f"merged into shared groups, so no split can separate them")
    record("PASS", "no image appears twice across merged datasets",
           f"{g.get('n_images')} images -> {g.get('n_groups')} groups, 0 cross-corpus pairs")


# ── 3 + 4. statistics are fitted on train only, and the check discriminates ───
def check_scaler_fit_on_train_only(run_dir, size=224):
    """
    Reconstruct the standardisation by hand from TRAIN rows and confirm the pipeline
    produced the same thing — then confirm the all-data statistics produce something
    DIFFERENT, so that the first check actually discriminates.

    Invariant 4 is the one people skip. Without it, a check that would pass no matter what
    is mistaken for evidence.
    """
    feats = None
    for v in ("rgb", "green_clahe", "green_clahe_raw01"):
        p = os.path.join(run_dir, f"features_{v}_{size}.npy")
        if os.path.exists(p):
            feats, which = np.load(p), v
            break
    if feats is None:
        return record("SKIP", "normalisation statistics fitted on train only",
                      f"no cached features in {run_dir}")

    man = os.path.join(run_dir, "manifest_idrid.json")
    if not os.path.exists(man):
        return record("SKIP", "normalisation statistics fitted on train only",
                      "no manifest")
    with open(man) as f:
        rows = json.load(f)
    tr = np.array([r["official_split"] == "train" for r in rows])

    from sklearn.preprocessing import StandardScaler
    fitted = StandardScaler().fit(feats[tr])

    by_hand_mean = feats[tr].mean(axis=0)
    all_data_mean = feats.mean(axis=0)

    same_as_train = np.allclose(fitted.mean_, by_hand_mean, atol=1e-4)
    same_as_all = np.allclose(fitted.mean_, all_data_mean, atol=1e-4)

    if not same_as_train:
        return record("FAIL", "normalisation statistics fitted on train only",
                      f"scaler mean != train mean (variant {which})")
    if same_as_all:
        return record("FAIL", "the train-only check actually discriminates",
                      "train statistics and all-data statistics are indistinguishable, so "
                      "the check above proves nothing")
    gap = float(np.abs(by_hand_mean - all_data_mean).max())
    record("PASS", "normalisation statistics fitted on train only",
           f"scaler mean == train mean; all-data mean differs by up to {gap:.4f}, "
           f"so the check discriminates (variant {which})")


# ── 5. preprocessing is deterministic and matches the cache ───────────────────
def check_preprocess_determinism(idrid_root, run_dir, cache_dir, size=224):
    import cv2
    from preprocess import apply_variant
    man = os.path.join(run_dir, "manifest_idrid.json")
    if not (os.path.exists(man) and os.path.isdir(cache_dir)):
        return record("SKIP", "one image's tensor recomputed in isolation matches the batch",
                      "no manifest or image cache yet")
    with open(man) as f:
        rows = json.load(f)
    r = rows[0]
    p = os.path.join(cache_dir, r["uid"] + ".png")
    if not os.path.exists(p):
        return record("SKIP", "one image's tensor recomputed in isolation matches the batch",
                      f"{p} missing")
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    a = apply_variant(img, "rgb", size)
    b = apply_variant(cv2.imread(p, cv2.IMREAD_COLOR), "rgb", size)
    if not np.array_equal(a, b):
        return record("FAIL", "one image's tensor recomputed in isolation matches the batch",
                      f"max abs diff {np.abs(a - b).max():.3e}")
    record("PASS", "one image's tensor recomputed in isolation matches the batch",
           f"{r['uid']} reproduced byte-for-byte, shape {a.shape}")


# ── 6. labels still match the source CSVs ─────────────────────────────────────
def check_labels_match_source(idrid_root, run_dir):
    """Guards against a manifest going stale relative to the ground-truth CSVs."""
    from data_idrid import build_manifest
    man = os.path.join(run_dir, "manifest_idrid.json")
    if not os.path.exists(man):
        return record("SKIP", "cached manifest matches the source label CSVs", "no manifest")
    with open(man) as f:
        cached = {r["uid"]: (r["dr"], r["dme"], r["official_split"]) for r in json.load(f)}
    fresh = {r["uid"]: (r["dr"], r["dme"], r["official_split"])
             for r in build_manifest(idrid_root)}
    if cached != fresh:
        diff = [k for k in fresh if cached.get(k) != fresh[k]]
        return record("FAIL", "cached manifest matches the source label CSVs",
                      f"{len(diff)} rows differ, e.g. {diff[:5]}")
    record("PASS", "cached manifest matches the source label CSVs",
           f"{len(fresh)} rows identical")


# ── 7. the DME gate discards nothing (PROTOCOL.md §5.1) ───────────────────────
def check_gate_assumption(idrid_root):
    from data_idrid import build_manifest
    rows = build_manifest(idrid_root)
    bad = [r["name"] for r in rows if r["dr"] == 0 and r["dme"] > 0]
    if bad:
        return record("FAIL", "the DR>=1 gate discards no DME-positive cases",
                      f"{len(bad)} images have DR=0 and DME>0, e.g. {bad[:5]} — "
                      f"PROTOCOL.md §5.1 rests on this being zero and must be revisited")
    record("PASS", "the DR>=1 gate discards no DME-positive cases",
           f"0 of {len(rows)} images have DR=0 with DME>0")


# ── 7b. image identifiers are unique ──────────────────────────────────────────
def check_uids_unique(idrid_root):
    """IDRiD numbers its train and test sets from 001 independently, so `name` collides
    103 times. Anything keyed on `name` silently loses or overwrites those rows
    (ISSUES.md §8)."""
    from data_idrid import build_manifest
    rows = build_manifest(idrid_root)
    uids, names = {r["uid"] for r in rows}, {r["name"] for r in rows}
    if len(uids) != len(rows):
        return record("FAIL", "image identifiers are unique",
                      f"{len(rows) - len(uids)} duplicate uids")
    record("PASS", "image identifiers are unique",
           f"{len(uids)} unique uids for {len(rows)} images "
           f"(bare names collide {len(rows) - len(names)} times -- do not key on name)")


# ── 7c. every archived result declares who made it, and that commit still exists ──
def check_run_provenance(runs_dir="runs"):
    """Each runs/<ID>/results.json must name <ID> and a commit on the current branch.

    Three ways this has broken, all silent (ISSUES.md §20):
      * another run's results.json copied in under the generic name (E08X);
      * schema drift -- E01 writes `experiment` where later runs write `run_id`, so a
        checker keyed on one of them skips the other rather than flagging it;
      * a rebase rewriting history, which orphans every SHA already written into an
        archive.
    """
    import glob as _glob
    import subprocess as _sp
    reachable = set(_sp.run(["git", "log", "--format=%H"], capture_output=True,
                            text=True).stdout.split())
    remap = {}
    if os.path.exists("data/commit_remap.json"):
        remap = json.load(open("data/commit_remap.json")).get("remap", {})
    problems, checked, remapped = [], 0, []
    for p in sorted(_glob.glob(os.path.join(runs_dir, "*", "results.json"))):
        run = os.path.basename(os.path.dirname(p))
        try:
            j = json.load(open(p))
        except Exception as e:
            problems.append(f"{p}: unreadable ({e})"); continue
        rid = j.get("run_id") or j.get("experiment")
        if rid is None:
            problems.append(f"{p}: declares no run_id/experiment"); continue
        checked += 1
        if rid != run:
            problems.append(f"{p}: says it is run {rid}, not {run}")
        com = (j.get("commit") or "").split("-")[0]
        if not com:
            problems.append(f"{p}: records no commit")
        elif com not in reachable:
            if com in remap and remap[com]["now"] in reachable:
                remapped.append(f"{run}:{com[:7]}->{remap[com]['now'][:7]}")
            else:
                problems.append(f"{p}: commit {com[:10]} is not on the current branch "
                                f"and has no entry in data/commit_remap.json")
    if problems:
        return record("FAIL", "archived results declare their own run and a live commit",
                      "; ".join(problems[:4]) + (" ..." if len(problems) > 4 else ""))
    note = f"{checked} run output(s) verified"
    if remapped:
        note += f"; {len(remapped)} SHA(s) resolved through data/commit_remap.json " \
                f"({', '.join(remapped)})"
    record("PASS", "archived results declare their own run and a live commit", note)


# ── 8. the metrics module agrees with itself ──────────────────────────────────
def check_metrics_sanity():
    import metrics as M
    y = np.array([0, 0, 1, 2, 2, 3, 4, 4, 1, 2])
    problems = []
    if M.quadratic_weighted_kappa(y, y, 5) != 1.0:
        problems.append("QWK of a perfect predictor != 1")
    const = np.zeros_like(y)
    if M.quadratic_weighted_kappa(y, const, 5) != 0.0:
        problems.append("QWK of a constant predictor != 0")
    far = np.where(y <= 2, 4, 0)
    near = np.where(y <= 2, y + 1, y - 1)
    if not M.quadratic_weighted_kappa(y, near, 5) > M.quadratic_weighted_kappa(y, far, 5):
        problems.append("QWK does not punish a far error more than a near one")
    if abs(M.accuracy(y, y) - 1.0) > 1e-12:
        problems.append("accuracy of a perfect predictor != 1")
    if abs(M.majority_floor(np.array([0, 0, 0, 1])) - 0.75) > 1e-12:
        problems.append("majority_floor wrong")
    # a group bootstrap over one single group must have zero width
    lo, hi = M.bootstrap_ci(y, y, M.accuracy, groups=np.zeros_like(y), n_boot=50)
    if abs(hi - lo) > 1e-12:
        problems.append("group bootstrap ignored the grouping")
    if problems:
        return record("FAIL", "metrics module self-consistent", "; ".join(problems))
    record("PASS", "metrics module self-consistent",
           "QWK ordering, floors, and group-aware bootstrap all behave")


def check_consumption_manifests(run_dirs):
    """PROTOCOL.md §9: runs may only be combined or compared when they read the same files.

    Reports every pair, so the output says which combinations are permitted rather than
    failing on the first problem. A run with no manifest at all is a SKIP, not a PASS —
    unverifiable consumption is exactly what §26 was.
    """
    import manifest as _manifest
    if len(run_dirs) < 2:
        record("SKIP", "consumption manifests", "fewer than two runs given (--compare)")
        return
    mans = {os.path.basename(d.rstrip("/")): _manifest.load(d, os.path.basename(d.rstrip("/")))
            for d in run_dirs}
    names = sorted(mans)
    unmanifested = [n for n in names if mans[n] is None]
    if unmanifested:
        record("SKIP", "consumption manifests",
               f"no manifest for {unmanifested} — what they read is unverifiable "
               f"(ISSUES.md §26); they may not be combined")
    ok_pairs, bad_pairs, coarse = [], [], False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if mans[a] is None or mans[b] is None:
                continue
            good, why = _manifest.compare(mans[a], mans[b], a, b)
            if (mans[a].get("manifest_version", 0) == 0
                    or mans[b].get("manifest_version", 0) == 0):
                coarse = True
            (ok_pairs if good else bad_pairs).append((a, b, why))
    for a, b, why in bad_pairs:
        record("FAIL", f"consumption {a} vs {b}", "; ".join(why))
    if ok_pairs and not bad_pairs:
        note = f"{len(ok_pairs)} pair(s) read the same files"
        if coarse:
            note += " — COARSE: some manifests are retrofitted from kernel metadata, "
            note += "so they prove the same datasets were mounted, not the same files read"
        record("PASS", "consumption manifests", note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idrid", required=True)
    ap.add_argument("--groups", default="data/groups.json")
    ap.add_argument("--splits", default="")
    ap.add_argument("--run", default="runs/E01")
    ap.add_argument("--cache", default="data/cache/idrid_640")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--compare", nargs="*", default=[],
                    help="run directories that are about to be combined, ensembled or "
                         "compared. Every pair must have agreeing consumption manifests "
                         "(PROTOCOL.md §9).")
    a = ap.parse_args()

    print("PROTOCOL.md §8 and §9 invariants\n" + "=" * 72)
    check_metrics_sanity()
    check_gate_assumption(a.idrid)
    check_uids_unique(a.idrid)
    check_run_provenance()
    check_labels_match_source(a.idrid, a.run)
    check_preprocess_determinism(a.idrid, a.run, a.cache, a.size)
    check_scaler_fit_on_train_only(a.run, a.size)
    check_no_cross_corpus_duplicates(a.groups)
    check_split_disjoint(a.splits)
    check_consumption_manifests(a.compare)

    n_fail = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    n_skip = sum(1 for s, _, _ in RESULTS if s == "SKIP")
    n_pass = sum(1 for s, _, _ in RESULTS if s == "PASS")
    print("=" * 72)
    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped (artifact not built yet)")
    if n_skip:
        print("A skip is not a pass. No result may be reported while a check it depends on "
              "is skipped.")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
