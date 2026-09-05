"""
build_kernel.py — generate the Kaggle notebook and its metadata, then push it.

The notebook is deliberately thin: it clones the repo at a given commit and runs
src/train.py. All the science lives in the repo, so a Kaggle run is reproducible from a
SHA rather than from whatever happened to be pasted into a cell.

Usage:
    python kaggle/build_kernel.py --run-id E05 --args "--epochs 30 --size 448" [--push]
"""
from __future__ import annotations
import argparse, json, os, subprocess

OWNER = "ah22reza"
REPO = "https://github.com/Ahmadrezanourozii/Automatic-detection-of-diabetic-retinopathy-and-grading-of-diabetic-macular-edema-using-CNN.git"

DATASETS = [
    "aaryapatel98/indian-diabetic-retinopathy-image-dataset",
    "google-brain/messidor2-dr-grades",
    "mariaherrerot/messidor2preprocess",
]
# EyePACS 2015 (~35 k images, DR only). Pretraining corpus, never a test set -- its labels
# are single-grader and noisy. Attached only for runs that ask for it, because it adds
# 7.8 GB and ~10 minutes of cache building.
EYEPACS = "tanlikesmath/diabetic-retinopathy-resized"
# APTOS 2019, held out entirely since the protocol was frozen. Attached ONLY for runs that
# do the final external evaluation, so it cannot drift into a development run by accident.
APTOS = "mariaherrerot/aptos2019"
# Messidor-2 at native 2240x1488 rather than the 512x512 default mirror. Covers 1057 of
# the 1744 labelled images -- the IM*-named ones are absent, so the corpus is left at
# mixed resolution. Worth it because E09 showed Mild recall is resolution-bound.
MESSIDOR_HI = "borhan2003/messidor-diabetic-retinopathy-dataset-jpg-format"
# RETFound CFP ViT-L/16 encoder, stripped from the official gated checkpoint and re-uploaded
# PRIVATELY (redistributing CC-BY-NC gated weights publicly is not ours to do). The probe
# verifies its sha256 at load time, so the path is not trusted (docs/RETFound_provenance.md).
RETFOUND = "ah22reza/retfound-cfp-encoder"

# A run whose ID names a piece of work must actually launch that work. E13gate was pushed
# as the fovea-localiser gate and silently ran src/train.py for 643 s instead, producing a
# complete, well-formed DR/DME results.json under the gate's run-id -- indistinguishable
# from a real run without reading it (ISSUES.md §24). The cause was that --script was
# parsed into `a.script` and never passed to cells(), so the flag was dead: passing it
# correctly produced no error and no warning. A silent no-op is the failure mode this
# project keeps paying for, so the intent is now declared here and checked twice --
# once against the run-id, and once against the notebook that actually got generated.
# Each token maps to the SET of scripts that legitimately do that work. A token pinned to a
# single script was too coarse: "probe" pointed only at e01_linear_probe.py and so refused
# I24PROBE, which runs retfound_probe.py -- also a probe, just a different one. The guard's
# intent is "a run-id claiming to be a probe must run a probe", not "must run THAT probe".
RUN_ID_REQUIRES_SCRIPT = {
    "gate": {"src/fovea.py"},
    "fovea": {"src/fovea.py"},
    "recal": {"src/recalibration_curve.py"},
    "thresh": {"src/tune_thresholds.py"},
    "probe": {"src/e01_linear_probe.py", "src/retfound_probe.py"},
}


def cells(run_id, train_args, commit, external_only=False, from_run="", script="",
          ext_args=None):
    setup = f'''# {run_id} — pulls the code from GitHub so a run is reproducible from a commit SHA
import os, subprocess, sys, shutil, time
REPO = "{REPO}"
COMMIT = "{commit}"
WORK = "/kaggle/working/repo"
if os.path.isdir(WORK):
    shutil.rmtree(WORK)
subprocess.run(["git", "clone", "--quiet", REPO, WORK], check=True)
if COMMIT and COMMIT != "HEAD":
    subprocess.run(["git", "-C", WORK, "checkout", "--quiet", COMMIT], check=True)
sha = subprocess.check_output(["git", "-C", WORK, "rev-parse", "HEAD"]).decode().strip()
print("CODE COMMIT", sha)
print(subprocess.check_output(["git", "-C", WORK, "log", "-1", "--pretty=%s"]).decode().strip())
'''

    inputs = '''import os
for d in sorted(os.listdir("/kaggle/input")):
    n = sum(len(f) for _, _, f in os.walk(f"/kaggle/input/{d}"))
    print(f"{d:55s} {n:7d} files")
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
'''

    guard = '''# The pool does not always honour the pinned accelerator. A P100 is sm_60 and the
# preinstalled torch cu128 build ships sm_70+ kernels only, so every CUDA call fails.
# Rather than lose the run, install a torch that supports this device (ISSUES.md §9).
import subprocess, sys, torch
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
print(f"GPU {name}  sm_{cap[0]}{cap[1]}  torch {torch.__version__}")
ok = True
try:
    (torch.zeros(8, 8, device="cuda") @ torch.zeros(8, 8, device="cuda")).sum().item()
    print("kernels execute fine on this device")
except Exception as e:
    ok = False
    print("UNUSABLE:", e)
if not ok:
    print("installing a torch build that supports this GPU ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.5.1", "torchvision==0.20.1",
                    "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)
    print("installed -- src/train.py runs in a subprocess so it picks up the new build")
'''

    train = f'''import subprocess, sys, os, time
RUN_ID = "{run_id}"
OUT = f"/kaggle/working/{{RUN_ID}}"
os.makedirs(OUT, exist_ok=True)
# Kaggle carries /kaggle/working across notebook versions, so appending to a fixed
# filename can leave the previous version's log sitting next to this version's results
# (ISSUES.md §13). Fresh file, stamped, every run.
STAMP = time.strftime("%Y%m%d-%H%M%S")
LOG = f"{{OUT}}/train_{{STAMP}}.log"
for _stale in os.listdir(OUT) if os.path.isdir(OUT) else []:
    if _stale.startswith("train") and _stale.endswith(".log"):
        os.remove(os.path.join(OUT, _stale))
        print("removed stale log", _stale)

cmd = [sys.executable, "-u", "/kaggle/working/repo/src/train.py",
       "--datasets", "/kaggle/input",
       "--splits", "/kaggle/working/repo/data/splits/dev_v1.json",
       "--run-id", RUN_ID, "--out", OUT,
       "--cache", "/kaggle/temp/cache560",
       "--channels-last", "--resume",
       {train_args}]
print(" ".join(cmd), flush=True)

# tee to the log file AND to the notebook output, so a killed session still leaves a log
t0 = time.time()
with open(LOG, "w") as f:
    f.write(f"===== launched {{time.strftime('%Y-%m-%d %H:%M:%S')}} =====\\n")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         bufsize=1, cwd="/kaggle/working/repo")
    for line in p.stdout:
        print(line, end="", flush=True)
        f.write(line); f.flush()
    p.wait()
print(f"\\nexit={{p.returncode}}  {{time.time()-t0:.0f}}s")
'''

    collect = f'''# keep results.json, the log, the OOF predictions and the selected weights;
# drop only the resumable checkpoints, which carry optimiser state and are ~3x larger
import glob, os, shutil, json
RUN_ID = "{run_id}"
OUT = f"/kaggle/working/{{RUN_ID}}"
for p in glob.glob(f"{{OUT}}/ckpt_*.pt"):
    print("dropping", os.path.basename(p), f"{{os.path.getsize(p)/1e6:.0f}} MB")
    os.remove(p)
for p in sorted(glob.glob(f"{{OUT}}/*")):
    print(f"{{os.path.getsize(p)/1e6:8.2f}} MB  {{os.path.basename(p)}}")
rj = f"{{OUT}}/results.json"
if os.path.exists(rj):
    r = json.load(open(rj))
    print("\\nrun", r["run_id"], "commit", r["commit"][:10],
          "| folds", [f["fold"] for f in r["folds"]])
    if "pooled_oof" in r:
        for k, v in r["pooled_oof"]["metrics"].items():
            print(f"  {{k:22s}} n={{v['n']:5d}} acc {{v['accuracy']*100:5.1f}}% "
                  f"floor {{v['majority_floor']*100:5.1f}}% QWK {{v['qwk']:6.3f}}")
'''
    external = f'''# External validation on a corpus held out since the protocol was frozen.
# Runs only if the corpus is attached; it is the number that survives a defence.
import subprocess, sys, os, glob
RUN_ID = "{run_id}"
OUT = f"/kaggle/working/{{RUN_ID}}"
if not glob.glob(f"{{OUT}}/best_*.pt"):
    print("no fold weights were saved -- skipping external evaluation")
else:
    # No shallow os.listdir check here. Kaggle mounted all five datasets under a single
    # /kaggle/input/datasets/ directory in E08, so a one-level scan reported APTOS missing
    # when it was present and skipped the evaluation (ISSUES.md §15). eval_external.py
    # already exits loudly if the corpus matches zero images -- let it be the judge.
    cmd = [sys.executable, "-u", "/kaggle/working/repo/src/eval_external.py",
           "--run", OUT, "--datasets", "/kaggle/input", "--corpus", "APTOS",
           "--cache", "/kaggle/temp/cache_ext", "--tta"]
    print(" ".join(cmd), flush=True)
    with open(f"{{OUT}}/external.log", "w") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, cwd="/kaggle/working/repo")
        for line in p.stdout:
            print(line, end="", flush=True); f.write(line); f.flush()
        p.wait()
    print("external exit=", p.returncode)
'''
    if script:
        # a standalone analysis script, not the training loop
        only = f'''import subprocess, sys, os, time, shutil
RUN_ID = "{run_id}"
OUT = f"/kaggle/working/{{RUN_ID}}"
# Kaggle carries /kaggle/working across notebook versions (ISSUES.md §13), and this
# run-id already holds the output of a previous version that ran the WRONG script
# (ISSUES.md §24): a results.json, best_*.pt and oof_*.npz from src/train.py. If a later
# version of this notebook fails, Kaggle serves the last COMPLETED version's output, and
# fetch.py's commit check cannot tell the two apart because both pin the same SHA. So the
# directory is emptied before this script runs: whatever ends up here came from this run.
if os.path.isdir(OUT):
    for _stale in sorted(os.listdir(OUT)):
        _p = os.path.join(OUT, _stale)
        print("purging stale artefact from a previous version:", _stale)
        shutil.rmtree(_p) if os.path.isdir(_p) else os.remove(_p)
os.makedirs(OUT, exist_ok=True)
assert not os.listdir(OUT), f"{{OUT}} is not empty after the purge"
cmd = [sys.executable, "-u", "/kaggle/working/repo/{script}",
       "--datasets", "/kaggle/input",
       "--splits", "/kaggle/working/repo/data/splits/dev_v1.json",
       "--out", OUT, {train_args}]
print(" ".join(cmd), flush=True)
with open(f"{{OUT}}/run.log", "w") as f:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, cwd="/kaggle/working/repo")
    for line in p.stdout:
        print(line, end="", flush=True); f.write(line); f.flush()
    p.wait()
print("exit=", p.returncode)
'''
        return [setup, inputs, guard, only]

    if external_only:
        # No training. The finished run's output is attached as a kernel source, so its
        # per-fold weights arrive under /kaggle/input rather than being retrained.
        # Default recipe is the 5-fold TTA ensemble; ext_args overrides it (I22).
        ext_args = ", ".join(f'"{t}"' for t in (ext_args or "--tta").split())
        ext_only = f'''import subprocess, sys, os, glob, shutil
SRC = "{from_run}"
OUT = f"/kaggle/working/{run_id}"
os.makedirs(OUT, exist_ok=True)
found = []
for root, _, files in os.walk("/kaggle/input"):
    for fn in files:
        if fn.startswith("best_") and fn.endswith(".pt"):
            found.append(os.path.join(root, fn))
        elif fn == "results.json" and "best_0.pt" in files:
            # The SOURCE run's results.json, kept for reference. It must NOT be written as
            # "results.json" -- that is this run's own reserved name, and copying another
            # run's file there is what made E08X look like it had produced E08's metrics
            # (ISSUES.md §20). Name it after where it came from.
            shutil.copy2(os.path.join(root, fn), f"{{OUT}}/source_run_results.json")
print(f"found {{len(found)}} fold weights from {{SRC}}")
for p_ in sorted(found):
    shutil.copy2(p_, os.path.join(OUT, os.path.basename(p_)))
if not found:
    raise SystemExit(f"no best_*.pt found under /kaggle/input -- is {{SRC}} attached "
                     f"as a kernel source?")
assert not os.path.exists(f"{{OUT}}/results.json"), \
    "results.json already exists in this run's output before the run produced one"

cmd = [sys.executable, "-u", "/kaggle/working/repo/src/eval_external.py",
       "--run", OUT, "--datasets", "/kaggle/input", "--corpus", "APTOS",
       "--cache", "/kaggle/temp/cache_ext", {ext_args}]
print(" ".join(cmd), flush=True)
with open(f"{{OUT}}/external.log", "w") as f:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, cwd="/kaggle/working/repo")
    for line in p.stdout:
        print(line, end="", flush=True); f.write(line); f.flush()
    p.wait()
print("exit=", p.returncode)
for p_ in glob.glob(f"{{OUT}}/best_*.pt"):
    os.remove(p_)          # they came from the source run; no need to duplicate them
'''
        return [setup, inputs, guard, ext_only]
    return [setup, inputs, guard, train, external, collect]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--args", default="--folds 0,1,2,3,4 --epochs 30 --size 448")
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--slug", default=None)
    ap.add_argument("--gpu", default="GPU T4 x2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--script", default="",
                    help="run a standalone analysis script instead of src/train.py")
    ap.add_argument("--external-only", action="store_true",
                    help="no training: load a finished run's weights and evaluate "
                         "externally")
    ap.add_argument("--ext-args", default="",
                    help="arguments for src/eval_external.py in an --external-only run. "
                         "Defaults to '--tta' (the 5-fold TTA ensemble). Use e.g. "
                         "'--folds 0' to isolate the inference recipe (IDEAS.md I22).")
    ap.add_argument("--from-run", default="",
                    help="kernel slug whose output supplies the weights, "
                         "e.g. ah22reza/dr-dme-e08")
    ap.add_argument("--aptos", action="store_true",
                    help="attach APTOS and run the external evaluation after training")
    ap.add_argument("--messidor-hi", action="store_true",
                    help="attach the native-resolution Messidor-2 mirror (1057 of 1744)")
    ap.add_argument("--retfound", action="store_true",
                    help="attach the private RETFound CFP encoder dataset")
    ap.add_argument("--eyepacs", action="store_true",
                    help="attach the EyePACS 2015 corpus for pretraining")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    if a.commit == "HEAD":
        a.commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

    # The notebook clones from GitHub and checks out this SHA. A commit that exists only
    # locally cannot be checked out there, and the kernel dies two minutes in on a git
    # error that says nothing about the cause (ISSUES.md §23). Verify it is on the remote
    # BEFORE writing a notebook that depends on it.
    subprocess.run(["git", "fetch", "--quiet", "origin"], check=False)
    on_remote = subprocess.run(
        ["git", "merge-base", "--is-ancestor", a.commit, "origin/main"]).returncode == 0
    if not on_remote:
        raise SystemExit(
            f"commit {a.commit[:10]} is not on origin/main. The kernel clones from GitHub, "
            f"so it could not check this out. Push first:\n"
            f"    git push origin main\n"
            f"(If you meant to pin an older commit, pass --commit <sha> explicitly.)")
    # A run-id that names a piece of work must launch that work (ISSUES.md §24). This is
    # checked before anything is written, so the failure is a refusal to build rather than
    # a notebook that runs the wrong thing under the right name.
    rid = a.run_id.lower()
    for token, allowed in RUN_ID_REQUIRES_SCRIPT.items():
        if token in rid and a.script not in allowed:
            required = " or ".join(sorted(allowed))
            raise SystemExit(
                f"run-id {a.run_id!r} contains {token!r}, which names {required}, but this "
                f"launch would run {a.script or 'src/train.py'}.\n"
                f"Either pass --script with one of {sorted(allowed)}, or rename the run so "
                f"its ID does not claim work it does not do.\n"
                f"(E13gate ran src/train.py under a fovea-gate name and archived a "
                f"DR/DME results.json that looked entirely legitimate -- ISSUES.md §24.)")

    # Kaggle derives the slug from the TITLE, so they must agree or every
    # status/output call afterwards addresses a kernel that does not exist
    slug = a.slug or f"dr-dme-{a.run_id.lower()}"
    title = slug.replace("-", " ").upper().replace("DR DME", "DR/DME")
    out = a.out or f"kaggle/{slug}"
    os.makedirs(out, exist_ok=True)

    train_args = ", ".join(f'"{t}"' for t in a.args.split())
    nb = {"cells": [{"cell_type": "code", "source": c, "metadata": {},
                     "execution_count": None, "outputs": []}
                    for c in cells(a.run_id, train_args, a.commit, a.external_only,
                                   a.from_run, a.script, a.ext_args)],
          "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3",
                                      "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 4}
    # The registry above encodes intent; this checks the artefact. Whatever the flags
    # said, the notebook about to be pushed must invoke the script we think it does, and
    # must not also invoke the training loop. This is what would have caught §24: the
    # generated cells never mentioned src/fovea.py, and nothing looked.
    emitted = "\n".join(c["source"] for c in nb["cells"])
    # An --external-only run legitimately never touches src/train.py: it loads a finished
    # run's weights and evaluates them. Defaulting `intended` to src/train.py made this
    # check refuse every such run -- a false positive in the §24 guard, found the first
    # time an external-only kernel was built after the guard landed.
    intended = a.script or ("src/eval_external.py" if a.external_only else "src/train.py")
    invoked = [s_ for s_ in ("src/train.py", a.script) if s_ and f"repo/{s_}" in emitted]
    if f"repo/{intended}" not in emitted:
        raise SystemExit(
            f"generated notebook does not invoke {intended} -- refusing to push. "
            f"It invokes: {invoked or 'nothing recognisable'}")
    if (a.script or a.external_only) and "repo/src/train.py" in emitted:
        raise SystemExit(
            f"generated notebook invokes BOTH {a.script} and src/train.py -- refusing to "
            f"push an ambiguous run.")
    print(f"notebook invokes {intended}  (checked against run-id {a.run_id!r})")

    with open(f"{out}/{slug}.ipynb", "w") as f:
        json.dump(nb, f, indent=1)

    meta = {
        "id": f"{OWNER}/{slug}",
        "title": title,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        # The key Kaggle actually reads is machine_shape -- "accelerator" in the metadata
        # file is silently ignored, which cost two runs (ISSUES.md §9). It must be pinned:
        # some pools hand out a P100 (sm_60) and the preinstalled torch cu128 build ships
        # sm_70+ kernels only. The notebook still carries a runtime fallback in case the
        # pin is not honoured.
        "machine_shape": a.gpu,
        "enable_internet": True,
        "dataset_sources": (DATASETS + ([EYEPACS] if a.eyepacs else [])
                            + ([APTOS] if a.aptos else [])
                            + ([MESSIDOR_HI] if a.messidor_hi else [])
                            + ([RETFOUND] if a.retfound else [])),
        "competition_sources": [],
        "kernel_sources": [a.from_run] if a.from_run else [],
    }
    with open(f"{out}/kernel-metadata.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {out}/  (id {meta['id']}, commit {a.commit[:10]})")

    if a.push:
        r = subprocess.run(["kaggle", "kernels", "push", "-p", out,
                            "--accelerator", a.gpu],
                           capture_output=True, text=True)
        print(r.stdout or r.stderr)


if __name__ == "__main__":
    main()
