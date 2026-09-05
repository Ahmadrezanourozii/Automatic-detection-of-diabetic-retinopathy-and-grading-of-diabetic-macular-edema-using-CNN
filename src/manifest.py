"""
manifest.py — record what a run actually CONSUMED, not what it was configured to consume.

ISSUES.md §26. Two runs were launched to complete each other's folds. Their `config` blocks
were identical, their `split_fingerprint` matched, the §24 script guard passed, and both
recorded `messidor_source: prefer-native` — a field that was literally true of both while
meaning different things, because one kernel had the native-resolution Messidor-2 mirror
mounted and the other did not. Four independent provenance mechanisms all passed on a
combination that would have produced a 5-fold number whose folds saw different images.

Every one of those mechanisms records **configuration**. None records **consumption**.

This module records consumption: for each corpus, how many images were resolved and a hash
over the resolved paths *relative to the dataset mount*, so it captures which dataset
directory supplied each file. Two runs that read the same files produce the same hash on any
machine; two runs that read different files cannot produce the same hash however identical
their configs look.

Written into `results.json` at runtime from inside the kernel, by whatever process actually
walked the disk.
"""
from __future__ import annotations
import hashlib, os


def _rel(path, roots):
    """Path relative to whichever dataset root contains it.

    Absolute paths differ between a Kaggle mount and a laptop, so hashing them would make
    every run look different for no reason. The portion *below* the mount is what identifies
    the file and, crucially, the dataset directory it came from -- which is exactly what
    differed in §26.
    """
    best = None
    for r in roots:
        r = os.path.abspath(r)
        p = os.path.abspath(path)
        if p.startswith(r + os.sep):
            cand = p[len(r) + 1:]
            if best is None or len(cand) < len(best):
                best = cand
    return best if best is not None else os.path.basename(path)


def build(rows, datasets, cache_dir=None, extra=None):
    """Manifest of what this run resolved. `rows` are corpora.build's output."""
    roots = [os.path.abspath(d) for d in datasets]
    per = {}
    for r in rows:
        p = r.get("path")
        if not p:
            continue
        per.setdefault(r.get("corpus", "?"), []).append(_rel(p, roots))

    corpora_out = {}
    for c, rels in sorted(per.items()):
        rels = sorted(rels)
        # the top-level directory under the mount = the dataset that supplied the file
        sources = sorted({r.split(os.sep)[0] for r in rels})
        corpora_out[c] = {
            "n": len(rels),
            "path_hash": hashlib.sha256("\n".join(rels).encode()).hexdigest()[:16],
            "source_dirs": sources,
        }

    mounted = []
    for r in roots:
        if os.path.isdir(r):
            mounted += sorted(d for d in os.listdir(r)
                              if os.path.isdir(os.path.join(r, d)))
    out = {
        "corpora": corpora_out,
        "mounted_dirs": sorted(set(mounted)),
        "cache_dir": cache_dir,
        "manifest_version": 1,
    }
    if extra:
        out.update(extra)
    return out


# Datasets that can never supply a development-pool image, so mounting them cannot change
# what a run trained on. APTOS is held out in its entirety (PROTOCOL.md §2, §6.1) and is only
# ever read by the external evaluation step, which runs after training finishes. Excluding it
# from the retrofit comparison is a consequence of the protocol, not a convenience: a run with
# APTOS attached and one without consumed the same development pool.
EVAL_ONLY_DATASETS = {"mariaherrerot/aptos2019"}


def retrofit_from_kernel_metadata(run_id, kaggle_dir="kaggle"):
    """Partial manifest for runs that predate this module.

    A finished run's notebook directory records which Kaggle datasets its kernel mounted.
    That is not the full consumption record — it does not say which files were resolved —
    but it is exactly the axis §26 turned on, so it gives the check real teeth on the
    archive without retraining anything.
    """
    import json
    slug = f"dr-dme-{run_id.lower()}"
    p = os.path.join(kaggle_dir, slug, "kernel-metadata.json")
    if not os.path.exists(p):
        return None
    meta = json.load(open(p))
    srcs = [d for d in meta.get("dataset_sources", []) if d not in EVAL_ONLY_DATASETS]
    return {"manifest_version": 0,
            "retrofitted": True,
            "dataset_sources": sorted(srcs),
            "excluded_eval_only": sorted(set(meta.get("dataset_sources", []))
                                         & EVAL_ONLY_DATASETS)}


def load(run_dir, run_id=None, kaggle_dir="kaggle"):
    """Manifest for an archived run: the runtime one if present, else a retrofit.

    The kernel's dataset_sources are attached to a runtime manifest as well, because a
    version-1 manifest and a version-0 retrofit have no common axis otherwise: `mounted_dirs`
    reads `['datasets']` whenever Kaggle flattens every dataset under one directory
    (ISSUES.md §15), which is not a dataset list and must never be compared against one.
    Without this, combining a new run with an older one is refused for a reason that is an
    artefact of the mount layout rather than a difference in what was read.
    """
    import json
    rid = run_id or os.path.basename(run_dir.rstrip("/"))
    retro = retrofit_from_kernel_metadata(rid, kaggle_dir)
    rj = os.path.join(run_dir, "results.json")
    if os.path.exists(rj):
        try:
            m = json.load(open(rj)).get("consumption")
            if m:
                if retro and retro.get("dataset_sources"):
                    m = dict(m, dataset_sources=retro["dataset_sources"])
                return m
        except Exception:
            pass
    return retro


def compare(a, b, name_a="A", name_b="B"):
    """Return (ok, reasons). Runs may only be combined or compared when this passes."""
    reasons = []
    if a is None or b is None:
        missing = [n for n, m in ((name_a, a), (name_b, b)) if m is None]
        return False, [f"no consumption manifest for {', '.join(missing)} — what it read "
                       f"is unverifiable, so it cannot be combined or compared "
                       f"(ISSUES.md §26)"]

    # retrofitted manifests only carry the mounted dataset list; compare on that axis
    if a.get("manifest_version", 0) == 0 or b.get("manifest_version", 0) == 0:
        # Compare on kernel dataset_sources — the only axis both versions share. Falling back
        # to mounted_dirs here would compare a flattened ['datasets'] against a real list.
        sa = set(a.get("dataset_sources") or [])
        sb = set(b.get("dataset_sources") or [])
        if not sa or not sb:
            return False, [f"no comparable dataset list for "
                           f"{name_a if not sa else name_b} — consumption unverifiable"]
        if sa and sb and sa != sb:
            reasons.append(f"mounted datasets differ: only in {name_a}: "
                           f"{sorted(sa - sb) or 'none'}; only in {name_b}: "
                           f"{sorted(sb - sa) or 'none'}")
        # A retrofit is COARSE and says so: it compares which datasets were mounted, not
        # which files were resolved. It catches §26 exactly, and it would miss a difference
        # in which files were read from the same mounted dataset. Runs written after
        # manifest_version 1 carry the fine-grained record.
        return (not reasons), reasons or ["retrofitted manifests (COARSE): mounted dataset "
                                          "sources match, but the resolved file list was "
                                          "not recorded — only version-1 manifests prove "
                                          "the same files were read"]

    ca, cb = a.get("corpora", {}), b.get("corpora", {})
    for c in sorted(set(ca) | set(cb)):
        if c not in ca or c not in cb:
            reasons.append(f"corpus {c!r} present in only one run")
            continue
        if ca[c]["n"] != cb[c]["n"]:
            reasons.append(f"{c}: {ca[c]['n']} images in {name_a} vs {cb[c]['n']} in {name_b}")
        if ca[c]["path_hash"] != cb[c]["path_hash"]:
            reasons.append(
                f"{c}: DIFFERENT FILES were read "
                f"({name_a} {ca[c]['path_hash']} from {ca[c]['source_dirs']}; "
                f"{name_b} {cb[c]['path_hash']} from {cb[c]['source_dirs']})")
    return (not reasons), reasons


def require_same(runs, kaggle_dir="kaggle", allow_unmanifested=False):
    """Raise unless every run in `runs` (dict name -> dir) consumed the same thing."""
    mans = {n: load(d, n, kaggle_dir) for n, d in runs.items()}
    names = sorted(mans)
    problems = []
    for i in range(len(names) - 1):
        ok, why = compare(mans[names[i]], mans[names[i + 1]], names[i], names[i + 1])
        if not ok:
            problems += [f"{names[i]} vs {names[i+1]}: {w}" for w in why]
    if problems:
        soft = all("no consumption manifest" in p for p in problems)
        if soft and allow_unmanifested:
            print("[manifest] WARNING — proceeding without verified consumption:")
            for p in problems:
                print("   ", p)
            return mans
        raise SystemExit(
            "refusing to combine or compare these runs — consumption differs or is "
            "unverifiable (PROTOCOL.md §9, ISSUES.md §26):\n  " + "\n  ".join(problems))
    return mans
