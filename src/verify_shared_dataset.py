"""
verify_shared_dataset.py — does a dataset shared between our two Kaggle accounts mount, and
is it byte-identical to the copy account 1 uploaded?

Account 2 (reza12123) was granted "can view" on account 1's private RETFound dataset. Sharing
is not the same as mounting: the kernel metadata still names `ah22reza/retfound-cfp-encoder`,
Kaggle may flatten every dataset under one directory (ISSUES.md §15), and a shared copy could
in principle differ from the original. None of that is assumed here — the file is located
recursively, hashed, and compared against the sha256 recorded when account 1 built it.

`PROTOCOL.md` §9: record what was consumed, not what was configured. The hash goes into the
run's manifest so any later result built on these weights can be traced to the bytes that
produced it.

Usage:
    python src/verify_shared_dataset.py --datasets /kaggle/input --out runs/ACCT2VERIFY
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, platform, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXPECTED_SHA256 = "847f9dd0e33bf8d450cc6121295d2919fc4bba3c185757a17ce6427bfa14ed37"
EXPECTED_SOURCE_SHA256 = "e1e4f66a1b792eeb6e2efaf158f33be35c8255f36b3d17ed67cd5129da246485"
EXPECTED_BYTES = 1213299887
WEIGHT_BASENAME = "retfound_cfp_encoder.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["/kaggle/input"])
    ap.add_argument("--out", default="runs/ACCT2VERIFY")
    # build_kernel passes --splits to every standalone script; accepted and unused here
    ap.add_argument("--splits", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    report = {"host": platform.platform(), "expected_sha256": EXPECTED_SHA256,
              "expected_bytes": EXPECTED_BYTES, "mounted": {}, "verdict": None}

    for root in a.datasets:
        if not os.path.isdir(root):
            print(f"[mount] {root} does not exist", flush=True)
            continue
        top = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        report["mounted"][root] = top
        print(f"[mount] {root} contains {len(top)} directories: {top}", flush=True)

    hits = []
    for root in a.datasets:
        hits += sorted(glob.glob(os.path.join(root, "**", WEIGHT_BASENAME), recursive=True))
    report["candidates"] = hits
    print(f"[find] {len(hits)} file(s) named {WEIGHT_BASENAME}", flush=True)
    for h in hits:
        print(f"   {h}", flush=True)

    if not hits:
        report["verdict"] = "NOT MOUNTED"
        seen = []
        for root in a.datasets:
            for dirpath, _, _ in os.walk(root):
                seen.append(dirpath)
                if len(seen) > 40:
                    break
        report["directories_seen"] = seen
        json.dump(report, open(os.path.join(a.out, "shared_dataset_check.json"), "w"), indent=1)
        raise SystemExit(f"{WEIGHT_BASENAME} is not mounted on this account — the share did "
                         f"not take effect, or the dataset was not attached to this kernel.")

    path = hits[0]
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    got = h.hexdigest()
    report.update({"path": path, "bytes": size, "sha256": got})
    print(f"\n[hash] {path}", flush=True)
    print(f"  bytes : {size}  (expected {EXPECTED_BYTES})", flush=True)
    print(f"  sha256: {got}", flush=True)
    print(f"  expect: {EXPECTED_SHA256}", flush=True)

    ok_size = size == EXPECTED_BYTES
    ok_hash = got == EXPECTED_SHA256
    report["verdict"] = "IDENTICAL" if (ok_size and ok_hash) else "DIFFERS"

    # the stripped encoder embeds its own origin; confirm that survived the share too
    try:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        src = ck.get("source_sha256")
        report["embedded_source_sha256"] = src
        report["embedded_source_matches"] = (src == EXPECTED_SOURCE_SHA256)
        report["n_tensors"] = len(ck.get("model", ck))
        print(f"  embedded source sha256: {src}", flush=True)
        print(f"  tensors: {report['n_tensors']}", flush=True)
    except Exception as e:                                  # torch may be absent on CPU-only
        report["embedded_source_sha256"] = f"not read: {e}"

    json.dump(report, open(os.path.join(a.out, "shared_dataset_check.json"), "w"), indent=1)
    print(f"\nVERDICT: {report['verdict']}", flush=True)
    print(f"wrote {a.out}/shared_dataset_check.json", flush=True)
    if report["verdict"] != "IDENTICAL":
        raise SystemExit("the shared copy is NOT byte-identical to account 1's upload — "
                         "refusing to treat it as the same artefact (PROTOCOL.md §9).")


if __name__ == "__main__":
    main()
