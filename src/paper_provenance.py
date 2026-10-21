"""
paper_provenance.py — Supplementary Table S1: each analysis plan's commit (identifier and UTC time,
from git) beside the first Kaggle launch of the runs it covers (from the Kaggle kernel listing
archived in docs/generated/paper/kaggle_runs.csv). Writes docs/generated/paper/provenance.json.
"""
from __future__ import annotations
import csv, json, subprocess
from datetime import datetime, timezone

PLANS = [
    ("Repository history begins (evaluation protocol present)", "5846b1b", []),
    ("Part A: derivation test of the macular-oedema grade", "f79abc4", []),
    ("CORAL, CORN and ConvNeXt interventions (hypotheses in EXPERIMENTS.md)", "b723945",
     ["dr-dme-e20coral", "dr-dme-e22corn", "dr-dme-e21cnxa"]),
    ("Journal outcomes O1-O9 (journal experiments plan)", "46f1bdd",
     ["dr-dme-s1e08", "dr-dme-s2e08", "dr-dme-s3e08", "dr-dme-s4e08", "dr-dme-x2ddr",
      "dr-dme-r1aptos", "dr-dme-i24ft234"]),
    ("Revision outcomes A1-E2 (revision plan)", "1456815",
     ["dr-dme-eyedn", "dr-dme-eyeeb3", "dr-dme-x5new"]),
]


def commit_time(h):
    s = subprocess.check_output(["git", "show", "-s", "--format=%cI", h]).decode().strip()
    return datetime.fromisoformat(s).astimezone(timezone.utc)


runs = {}
for r in csv.DictReader(open("docs/generated/paper/kaggle_runs.csv")):
    if r.get("ref") and r.get("lastRunTime"):
        runs[r["ref"].split("/")[-1]] = datetime.fromisoformat(r["lastRunTime"]).replace(tzinfo=timezone.utc)
out = []
for name, h, kernels in PLANS:
    t = commit_time(h)
    launches = sorted(runs[k] for k in kernels if k in runs)
    out.append({"plan": name, "commit": h, "commit_utc": t.strftime("%Y-%m-%d %H:%M:%S"),
                "runs": kernels, "first_launch_utc": launches[0].strftime("%Y-%m-%d %H:%M:%S") if launches else None,
                "all_after_commit": all(x >= t for x in launches) if launches else None})
    print(out[-1])
json.dump(out, open("docs/generated/paper/provenance.json", "w"), indent=1)
