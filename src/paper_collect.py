"""
paper_collect.py — gather every model's per-image predictions into results/predictions/.

The runs' own directories keep logits as oof_<fold>.npz and external_<corpus>_predictions.npz,
which runs/.gitignore excludes as bulky. The paper needs them versioned: every number in it is
recomputed from these files by src/paper_rules.py, and releasing them lets a reader re-derive
every table without a GPU. They are small (float32 logits, ~0.1-0.6 MB per file).

Layout written:
    results/predictions/<MODEL>/oof.npz              uids, folds, dr_logits, dme_logits
    results/predictions/<MODEL>/external_<corpus>.npz uids, y_dr, dr_logits
    results/predictions/registry.json                 model -> head, sources, provenance

Usage:
    python src/paper_collect.py --search /tmp/kout /tmp/kout3 ...
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

# MODEL: (head, [oof run dirs], {corpus: external run dir}, note)
# A run dir is the directory a kernel wrote (its RUN_ID). Assembled models list several.
REGISTRY = {
    "E05": ("ordinal", ["E05"], {}, "densenet121 448px, no pretraining, 30 ep"),
    "E06": ("ordinal", ["E06"], {}, "densenet121 448px, EyePACS pretraining, 25 ep"),
    "E08": ("ordinal", ["E08"], {"APTOS": "E08X2"}, "densenet121 448px, EyePACS, 40 ep (control)"),
    "E09": ("ordinal", ["E09"], {"APTOS": "EXTE09"}, "densenet121 224px"),
    "E10": ("ordinal", ["E10"], {"APTOS": "EXTE10"}, "densenet121 560px effective"),
    "E11FULL": ("ordinal", ["E11", "E19E11C"], {"APTOS": "EXTE11FULL"},
                "efficientnet-b3 448px, EyePACS (selected model)"),
    "E14MAC": ("ordinal", ["E14MAC"], {"APTOS": "EXTE14MAC"}, "densenet121 + macula crop for DME"),
    "E15LPFT": ("ordinal", ["E15LPFT"], {"APTOS": "EXTE15LPFT"}, "densenet121, LP-FT"),
    "E17NAT": ("ordinal", ["E17NAT"], {"APTOS": "EXTE17NAT"}, "densenet121 640px native"),
    "E20CORAL": ("coral", ["E20CORAL"], {}, "densenet121, CORAL head"),
    "E22CORN": ("corn", ["E22CORN"], {}, "densenet121, CORN head"),
    "E21CNXA": ("ordinal", ["E21CNXA"], {}, "convnext-tiny, folds 0-1 only"),
    "I24FT01": ("ordinal", ["I24FT01"], {}, "RETFound ViT-L/16 LP-FT 224px, folds 0-1"),
    # journal-paper runs (docs/PAPER_EXPERIMENTS_preregistration.md)
    "S1E08": ("ordinal", ["S1E08"], {"APTOS": "S1E08", "DDR": "S1E08"}, "E08, seed 1"),
    "S2E08": ("ordinal", ["S2E08"], {"APTOS": "S2E08", "DDR": "S2E08"}, "E08, seed 2"),
    "S3E08": ("ordinal", ["S3E08"], {"APTOS": "S3E08", "DDR": "S3E08"}, "E08, seed 3"),
    "S4E08": ("ordinal", ["S4E08"], {"APTOS": "S4E08", "DDR": "S4E08"}, "E08, seed 4"),
    # folds 0-1 were scored inside X2DDR, folds 2-4 inside I24FT234; a list of external dirs
    # is combined as the fold-count-weighted mean of their logit means, which equals the mean
    # over all five fold models because eval_external averages logits linearly
    "I24FULL": ("ordinal", ["I24FT01", "I24FT234"],
                {"APTOS": ["I24FT01", "I24FT234"], "DDR": ["I24FT01", "I24FT234"]},
                "RETFound ViT-L/16 LP-FT 224px, folds 0-1 + 2-4"),
}
# DDR re-scoring of every archived model happened in one kernel (X2DDR), which writes each
# model's predictions into a directory named after the model; the models without archived
# APTOS predictions were re-scored on APTOS there too.
for _m in ("E08", "E09", "E10", "E11FULL", "E14MAC", "E15LPFT", "E17NAT", "E20CORAL",
           "E22CORN", "E21CNXA"):
    REGISTRY[_m][2].setdefault("DDR", _m)
for _m in ("E20CORAL", "E22CORN", "E21CNXA"):
    REGISTRY[_m][2].setdefault("APTOS", _m)
# the reverse direction: trained on APTOS (own split), external Messidor-2 and IDRiD
REVERSE = {"R1APTOS": ("ordinal", ["R1APTOS"], {"Messidor-2": "R1APTOS", "IDRiD": "R1APTOS"},
                       "E08 recipe trained on APTOS (reverse direction)")}


def find_dirs(roots, name):
    """Every directory called `name` under the search roots, excluding repo clones."""
    hits = []
    for root in roots:
        for d in glob.glob(os.path.join(root, "**", name), recursive=True):
            if os.path.isdir(d) and f"{os.sep}repo{os.sep}" not in d:
                hits.append(d)
    return sorted(set(hits))


def collect_oof(roots, run_dirs):
    uids, folds, dr, dme, src = [], [], [], [], {}
    for rd in run_dirs:
        dirs = find_dirs(roots, rd)
        files = []
        for d in dirs:
            files += glob.glob(os.path.join(d, "oof_*.npz"))
        by_name = {}
        for f in files:                     # the same kernel downloaded twice is not two sources
            by_name.setdefault(os.path.basename(f), f)
        for name, f in sorted(by_name.items()):
            k = int(name.split("_")[1].split(".")[0])
            if k in src:
                raise SystemExit(f"fold {k} supplied by {src[k]} and {f}")
            src[k] = f
            z = np.load(f, allow_pickle=True)
            n = len(z["uids"])
            uids += [str(u) for u in z["uids"]]; folds += [k] * n
            dr.append(z["dr_logits"].astype(np.float32)); dme.append(z["dme_logits"].astype(np.float32))
    if not uids:
        return None, {}
    return {"uids": np.array(uids), "folds": np.array(folds, dtype=np.int16),
            "dr_logits": np.concatenate(dr), "dme_logits": np.concatenate(dme)}, src


def collect_external(roots, run_dir, corpus):
    if isinstance(run_dir, (list, tuple)):
        parts = []
        for rd in run_dir:
            e, f = collect_external(roots, rd, corpus)
            if e is None:
                return None, None
            meta = os.path.join(os.path.dirname(f), f"external_{corpus.lower()}.json")
            n = json.load(open(meta))["n_folds_ensembled"]
            parts.append((e, f, n))
        e0 = parts[0][0]
        for e, f, _ in parts[1:]:
            assert (e["uids"] == e0["uids"]).all() and (e["y_dr"] == e0["y_dr"]).all(), f
        w = np.array([n for *_, n in parts], dtype=np.float64)
        dr = sum(wi * e["dr_logits"].astype(np.float64) for wi, (e, _, _) in zip(w, parts)) / w.sum()
        return {"uids": e0["uids"], "y_dr": e0["y_dr"], "dr_logits": dr.astype(np.float32)}, \
            "+".join(f"{f} (x{n})" for _, f, n in parts)
    pats = [f"external_{corpus.lower()}_predictions.npz"]
    for d in find_dirs(roots, run_dir):
        for p in pats:
            f = os.path.join(d, p)
            if os.path.exists(f):
                z = np.load(f, allow_pickle=True)
                return {"uids": np.array([str(u) for u in z["uids"]]),
                        "y_dr": np.asarray(z["y_dr"]).astype(np.int8),
                        "dr_logits": np.asarray(z["dr_logits"]).astype(np.float32)}, f
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", nargs="+", required=True)
    ap.add_argument("--out", default="results/predictions")
    ap.add_argument("--extra", default="",
                    help="JSON file adding/overriding registry entries "
                         "{MODEL: [head, [oof dirs], {corpus: dir}, note]}")
    a = ap.parse_args()
    reg = dict(REGISTRY)
    reg.update(REVERSE)
    if a.extra:
        reg.update({k: tuple(v) for k, v in json.load(open(a.extra)).items()})
    os.makedirs(a.out, exist_ok=True)
    registry = {}
    for model, (head, oof_dirs, ext, note) in reg.items():
        entry = {"head": head, "note": note, "oof_sources": oof_dirs, "external": {},
                 "reverse": model in REVERSE}
        dst = os.path.join(a.out, model)
        oof, src = collect_oof(a.search, oof_dirs)
        if oof is not None:
            os.makedirs(dst, exist_ok=True)
            np.savez_compressed(os.path.join(dst, "oof.npz"), **oof)
            entry["oof_folds"] = sorted(src)
            entry["oof_n"] = int(len(oof["uids"]))
        for corpus, rd in ext.items():
            e, f = collect_external(a.search, rd, corpus)
            if e is None:
                print(f"  {model}: no {corpus} predictions under {rd}")
                continue
            os.makedirs(dst, exist_ok=True)
            np.savez_compressed(os.path.join(dst, f"external_{corpus.lower()}.npz"), **e)
            entry["external"][corpus] = {"source_run": rd, "source": f, "n": int(len(e["uids"]))}
        registry[model] = entry
        print(f"{model:9s} oof={entry.get('oof_n')} folds={entry.get('oof_folds')} "
              f"external={ {k: v['n'] for k, v in entry['external'].items()} }")
    json.dump(registry, open(os.path.join(a.out, "registry.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
