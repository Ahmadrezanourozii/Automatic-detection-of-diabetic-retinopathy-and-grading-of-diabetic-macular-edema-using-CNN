"""
fetch.py — pull a finished (or running) Kaggle run's log and results back, archive them,
and print an analysis.

Every run's artifacts land in runs/<RUN_ID>/ under version control, permanently. A run from
three weeks ago has to still be readable when we are on run thirty (the knowledge-base rule
in the project brief), so nothing here writes to a temporary directory.

Usage:
    python kaggle/fetch.py --run-id E05 [--slug dr-dme-e05] [--wait]
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, subprocess, sys, tempfile, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def _kaggle_bin():
    """Prefer the CLI in this repo's venv over whatever PATH happens to hold.

    Relying on PATH meant a plain invocation died with FileNotFoundError, and because the
    caller was filtering the output with grep, the traceback was swallowed and the empty
    result read as "the run produced no weights" (ISSUES.md §15).
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local = os.path.join(here, ".venv", "bin", "kaggle")
    return local if os.path.exists(local) else (shutil.which("kaggle") or "kaggle")


def kaggle(*args, check=False):
    try:
        r = subprocess.run([_kaggle_bin(), *args], capture_output=True, text=True)
    except FileNotFoundError as e:
        raise SystemExit(f"kaggle CLI not found: {e}. Install it, or run from the repo "
                         f"venv (.venv/bin/kaggle).")
    if check and r.returncode != 0:
        raise SystemExit(r.stderr or r.stdout)
    return (r.stdout or "") + (r.stderr or "")


def status(slug):
    out = kaggle("kernels", "status", f"ah22reza/{slug}")
    for token in ("RUNNING", "COMPLETE", "ERROR", "CANCEL", "QUEUED"):
        if token in out.upper():
            return token, out.strip()
    return "UNKNOWN", out.strip()


def fetch(slug, dest, keep_weights=False):
    tmp = tempfile.mkdtemp()
    out = kaggle("kernels", "output", f"ah22reza/{slug}", "-p", tmp)
    os.makedirs(dest, exist_ok=True)
    moved = []
    run_id = os.path.basename(dest)
    for src in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
        if os.path.isdir(src):
            continue
        rel = os.path.relpath(src, tmp)
        parts = rel.split(os.sep)
        # Take only the run's own outputs and the kernel log. The notebook clones the repo
        # into its working directory, so a naive glob copies the whole repo back over the
        # run directory -- including a results.json that is not this run's.
        is_run_file = run_id in parts
        is_kernel_log = len(parts) == 1 and rel.endswith(".log")
        if not (is_run_file or is_kernel_log):
            continue
        name = parts[-1] if is_run_file else "kernel.log"
        if name.startswith("ckpt_") and name.endswith(".pt"):
            continue        # resumable checkpoints carry optimiser state and are ~90 MB
        if name.endswith((".pt", ".pth")) and not keep_weights:
            continue        # pass --weights to pull the ~30 MB per-fold selected weights
        dst = os.path.join(dest, name)
        shutil.copy2(src, dst)
        moved.append((name, os.path.getsize(dst)))
    shutil.rmtree(tmp, ignore_errors=True)
    return moved, out


def check_log_matches(run_dir, results):
    """The log and the results must come from the same run.

    Kaggle carries /kaggle/working across notebook versions, so a stale log from a failed
    earlier attempt can be archived next to a successful run's results.json. An archive
    that silently pairs one run's log with another run's numbers is worse than no archive
    (ISSUES.md §13).
    """
    logs = sorted(glob.glob(os.path.join(run_dir, "train*.log")))
    if not logs:
        return
    want = results.get("commit", "").split("-")[0]
    for lp in logs:
        head = open(lp, errors="replace").read(4000)
        got = ""
        for line in head.splitlines():
            if line.startswith("COMMIT "):
                got = line.split()[1].split("-")[0]
                break
        if want and got and got != want:
            print(f"  !! {os.path.basename(lp)} is from commit {got[:10]}, but "
                  f"results.json is from {want[:10]} — STALE LOG, do not read it as "
                  f"this run's")
            os.rename(lp, lp + ".stale")
        elif want and not got:
            print(f"  !! {os.path.basename(lp)} has no COMMIT line — cannot verify it "
                  f"belongs to this run")


def report_code_commit(run_dir, expected=None):
    """Print which commit the kernel actually checked out, and compare it to the pin.

    Kaggle serves the most recent *completed* version's output, which is not necessarily
    the version just pushed. Twice now a result has been read as belonging to a fix that
    the running kernel did not contain (ISSUES.md §13, §16). The training script prints
    `CODE COMMIT <sha>` as its first line precisely so this can be checked mechanically.
    """
    import json as _json
    kl = os.path.join(run_dir, "kernel.log")
    if not os.path.exists(kl):
        return None
    try:
        events = _json.load(open(kl))
        texts = [e.get("data", "") for e in events]
    except Exception:
        texts = [open(kl, errors="replace").read()]
    got = None
    for t in texts:
        for line in t.splitlines():
            if line.startswith("CODE COMMIT "):
                got = line.split()[2].strip()
                break
        if got:
            break
    if not got:
        print("  !! kernel log has no CODE COMMIT line — cannot tell which code ran")
        return None
    if expected and not got.startswith(expected.rstrip("*")):
        print(f"  !! the kernel ran commit {got[:10]}, but {expected[:10]} was pinned — "
              f"this output is from an EARLIER version; do not read it as the new one")
    else:
        print(f"  kernel ran commit {got[:10]}")
    return got


def pinned_commit(slug):
    """The commit the local notebook for this kernel pins, if it is on disk."""
    import re as _re
    nb = os.path.join("kaggle", slug, f"{slug}.ipynb")
    if not os.path.exists(nb):
        return None
    src = open(nb).read()
    m = _re.search(r'COMMIT = \\"(\w+)\\"', src) or _re.search(r'COMMIT = "(\w+)"', src)
    return m.group(1) if m else None


def analyse(run_dir):
    rj = os.path.join(run_dir, "results.json")
    if not os.path.exists(rj):
        print("no results.json yet")
        return None
    r = json.load(open(rj))
    if "run_id" not in r:
        print(f"  {rj} has no run_id — it is not one of our run outputs; ignoring")
        return None
    check_log_matches(run_dir, r)
    print(f"\n{'='*78}")
    print(f"RUN {r['run_id']}   commit {r['commit'][:10]}   "
          f"split {r.get('split_fingerprint')}   {r.get('runtime_sec', 0)/60:.0f} min")
    if r.get("hypothesis"):
        print(f"hypothesis: {r['hypothesis']}")
    c = r.get("config", {})
    print(f"config: backbone={c.get('backbone')} head={c.get('head')} size={c.get('size')} "
          f"batch={c.get('batch')} epochs={c.get('epochs')} "
          f"alpha={c.get('alpha')} beta={c.get('beta')} lr_head={c.get('lr_head')}")
    print("=" * 78)

    for f in r.get("folds", []):
        h = f.get("history", [])
        if not h:
            continue
        best = max(h, key=lambda e: e["score"])
        print(f"\nfold {f['fold']}  train={f['n_train']} val={f['n_val']}  "
              f"{len(h)} epochs, best at epoch {best['epoch']}")
        print(f"   first epoch : loss {h[0]['loss']:.4f}  DR qwk {h[0]['dr_qwk']:.3f}")
        print(f"   best  epoch : loss {best['loss']:.4f}  DR qwk {best['dr_qwk']:.3f}  "
              f"DME qwk {(best['dme_qwk'] or 0):.3f}")
        print(f"   last  epoch : loss {h[-1]['loss']:.4f}  DR qwk {h[-1]['dr_qwk']:.3f}")
        # divergence check -- the failure mode that killed every archived run before this
        if h[-1]["loss"] > h[0]["loss"]:
            print("   WARNING: training loss ended higher than it started (diverging)")
        if best["epoch"] == 0 and len(h) > 3:
            print("   WARNING: best epoch is the first one -- training is not helping")

    pooled = r.get("pooled_oof")
    if pooled:
        print(f"\n{'='*78}\nPOOLED OUT-OF-FOLD  {pooled['n_images']} images, "
              f"folds {pooled['folds']}\n{'='*78}")
        for k, v in pooled["metrics"].items():
            lo, hi = v["accuracy_ci95"]
            qlo, qhi = v["qwk_ci95"]
            verdict = "BEATS floor" if v["beats_floor"] else "BELOW FLOOR"
            print(f"  {k:22s} n={v['n']:5d}  acc {v['accuracy']*100:5.1f}% "
                  f"[{lo*100:.1f},{hi*100:.1f}]  floor {v['majority_floor']*100:5.1f}%  "
                  f"-> {verdict}")
            print(f"  {'':22s}          QWK {v['qwk']:6.3f} [{qlo:.3f},{qhi:.3f}]   "
                  f"macro-F1 {v['macro_f1']:.3f}")
            rec = v.get("per_class_recall") or []
            print(f"  {'':22s}          recall " +
                  "  ".join("--" if x is None else f"{x*100:.0f}%" for x in rec) +
                  f"   support {v.get('support')}")
            if "referable_sensitivity" in v:
                print(f"  {'':22s}          referable sens "
                      f"{v['referable_sensitivity']*100:.1f}%  spec "
                      f"{v['referable_specificity']*100:.1f}%")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--slug", default=None)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--weights", action="store_true",
                    help="also archive best_<fold>.pt (~30 MB each); needed for external "
                         "evaluation, ensembling and Grad-CAM")
    a = ap.parse_args()
    slug = a.slug or f"dr-dme-{a.run_id.lower()}"
    run_dir = os.path.join("runs", a.run_id)

    st, raw = status(slug)
    print(f"[{slug}] {st}")
    while a.wait and st in ("RUNNING", "QUEUED"):
        time.sleep(a.poll)
        st, raw = status(slug)
        print(f"[{slug}] {st}  {time.strftime('%H:%M:%S')}", flush=True)

    moved, out = fetch(slug, run_dir, a.weights)
    report_code_commit(run_dir, pinned_commit(slug))
    if moved:
        print(f"\narchived into {run_dir}/:")
        for n, sz in sorted(moved):
            print(f"   {sz/1e6:8.2f} MB  {n}")
    else:
        print(f"no output yet ({out.strip()[:200]})")
    analyse(run_dir)


if __name__ == "__main__":
    main()
