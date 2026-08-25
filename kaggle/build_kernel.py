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


def cells(run_id, train_args, commit):
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
LOG = f"{{OUT}}/train.log"

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
with open(LOG, "a") as f:
    f.write(f"\\n===== launched {{time.strftime('%Y-%m-%d %H:%M:%S')}} =====\\n")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         bufsize=1, cwd="/kaggle/working/repo")
    for line in p.stdout:
        print(line, end="", flush=True)
        f.write(line); f.flush()
    p.wait()
print(f"\\nexit={{p.returncode}}  {{time.time()-t0:.0f}}s")
'''

    collect = f'''# keep results.json, the log and the out-of-fold predictions; drop the heavy checkpoints
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
    return [setup, inputs, guard, train, collect]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--args", default="--folds 0,1,2,3,4 --epochs 30 --size 448")
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--slug", default=None)
    ap.add_argument("--gpu", default="GPU T4 x2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    if a.commit == "HEAD":
        a.commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    # Kaggle derives the slug from the TITLE, so they must agree or every
    # status/output call afterwards addresses a kernel that does not exist
    slug = a.slug or f"dr-dme-{a.run_id.lower()}"
    title = slug.replace("-", " ").upper().replace("DR DME", "DR/DME")
    out = a.out or f"kaggle/{slug}"
    os.makedirs(out, exist_ok=True)

    train_args = ", ".join(f'"{t}"' for t in a.args.split())
    nb = {"cells": [{"cell_type": "code", "source": c, "metadata": {},
                     "execution_count": None, "outputs": []}
                    for c in cells(a.run_id, train_args, a.commit)],
          "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3",
                                      "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 4}
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
        "dataset_sources": DATASETS,
        "competition_sources": [],
        "kernel_sources": [],
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
