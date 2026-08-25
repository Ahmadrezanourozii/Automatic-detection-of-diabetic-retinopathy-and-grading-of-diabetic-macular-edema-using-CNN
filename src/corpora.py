"""
corpora.py — one manifest for every corpus, with the label spaces harmonised.

Runs unchanged locally and on Kaggle: paths are discovered by searching for the
ground-truth files rather than hard-coded, because the same corpus is laid out
differently in every mirror (and the macOS Google Drive mount URL-encodes folder names).

Label semantics are the ones justified in data/LABEL_MAPPING.md. In particular:

  dr   : 0..4 ICDR, identical across IDRiD / Messidor-2 / APTOS / EyePACS.
  dme  : 0..2 by hard-exudate distance to the macula centre
           0 = none, 1 = present but > 1 disc diameter out, 2 = within 1 DD (referable)
  dme_label_space : "3class" when the corpus grades all three DME levels (IDRiD),
           "binary" when it only separates referable from not (Messidor-2), None otherwise.
           The 3-class DME metric is computed on "3class" corpora ONLY -- see below.
  dme_candidates : the set of DME grades this row is known to be in.
           IDRiD gives an exact grade    -> a single-element set, e.g. {2}
           Messidor-2 gives referable/not -> {2} when referable, {0,1} when not
           APTOS/EyePACS give nothing     -> None, the DME loss is masked for that row

  WHY dme_label_space MATTERS FOR EVALUATION, NOT JUST TRAINING
  Messidor-2's referable rows have an exact grade (2) and its non-referable rows do not.
  Scoring the 3-class metric on "every row with an exact grade" therefore silently adds 151
  images that are ALL grade 2, turning a 47 %-majority evaluation set into a 59 %-majority
  one and inflating accuracy for free. Partial labels are legitimate supervision and
  illegitimate evaluation: the 3-class number is reported on IDRiD only, and Messidor-2 is
  scored on the binary referable task it can actually answer.

  The {0,1} case is why the DME head is trained with a marginal (partial-label) loss:
  Messidor-2 supervises the coarse distinction over 1 744 images, IDRiD supervises the
  fine one over 516. Flattening IDRiD to binary instead would discard the 3-class
  grading that is the thesis' contribution.
"""
from __future__ import annotations
import csv, os, re

DR_CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]
DME_CLASSES = ["No_DME", "Non_referable_DME", "Referable_DME"]
N_DR, N_DME = 5, 3

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


# ── path discovery ────────────────────────────────────────────────────────────
def find_file(roots, *name_fragments, must_end=None):
    """Depth-first search for the first file whose path contains all fragments."""
    frags = [f.lower() for f in name_fragments]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for fn in files:
                p = os.path.join(dirpath, fn)
                low = p.lower()
                if all(f in low for f in frags) and (must_end is None or low.endswith(must_end)):
                    return p
    return None


def dataset_root_of(path, roots):
    """The top-level corpus directory a file belongs to.

    Without this, load_eyepacs' `find_dir(roots, "train")` happily matched IDRiD's
    "a. Training Set" and indexed the wrong corpus's images. Each loader must stay inside
    the dataset that supplied its label file.
    """
    path = os.path.abspath(path)
    for root in roots:
        root = os.path.abspath(root)
        if not path.startswith(root + os.sep):
            continue
        rel = os.path.relpath(path, root).split(os.sep)
        return os.path.join(root, rel[0]) if len(rel) > 1 else root
    return os.path.dirname(path)


def find_dir(roots, *fragments):
    frags = [f.lower() for f in fragments]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirs, _ in os.walk(root):
            low = dirpath.lower()
            if all(f in low for f in frags):
                return dirpath
    return None


def index_images(root):
    """{basename without extension: full path} for every image under root."""
    out = {}
    if not root or not os.path.isdir(root):
        return out
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(IMG_EXT):
                out.setdefault(os.path.splitext(fn)[0], os.path.join(dirpath, fn))
                out.setdefault(fn, os.path.join(dirpath, fn))
    return out


def _rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f)]


# ── IDRiD ─────────────────────────────────────────────────────────────────────
def load_idrid(roots):
    """516 images, DR 0-4 and DME 0-2, plus fovea / optic-disc centres for all of them."""
    out = []
    fovea, od = {}, {}
    for sink, frag in ((fovea, "fovea"), (od, "od_center")):
        for split in ("training", "testing"):
            p = find_file(roots, "localization", frag, split, must_end=".csv") or \
                find_file(roots, frag, split, must_end=".csv")
            if not p:
                continue
            for r in _rows(p):
                name = (r.get("Image No") or "").strip()
                if not name:
                    continue
                xs = [v for k, v in r.items() if k and k.strip().lower().startswith("x")]
                ys = [v for k, v in r.items() if k and k.strip().lower().startswith("y")]
                try:
                    sink[f"{split[:4]}_{name}"] = (int(float(xs[0])), int(float(ys[0])))
                except (ValueError, IndexError):
                    pass

    for split, frag, imgfrag in (("train", "training labels", "training set"),
                                 ("test", "testing labels", "testing set")):
        csv_path = find_file(roots, "grading", frag, must_end=".csv")
        if not csv_path:
            continue
        img_dir = find_dir(roots, "grading", "original images", imgfrag)
        idx = index_images(img_dir)
        for r in _rows(csv_path):
            name = (r.get("Image name") or "").strip()
            if not name:
                continue
            dme_key = [c for c in r if c and "macular" in c.lower()][0]
            path = idx.get(name)
            if not path:
                continue
            dme = int(r[dme_key].strip())
            out.append({
                # IDRiD numbers train and test independently from 001 -- the bare name
                # collides 103 times (ISSUES.md §8). Never key on `name`.
                "uid": f"IDRiD_{split}_{name}",
                "corpus": "IDRiD",
                "idrid_split": split,
                "path": path,
                "dr": int(r["Retinopathy grade"].strip()),
                "dme": dme,
                "dme_candidates": (dme,),
                "dme_label_space": "3class",
                "fovea": fovea.get(f"{'trai' if split=='train' else 'test'}_{name}"),
                "optic_disc": od.get(f"{'trai' if split=='train' else 'test'}_{name}"),
            })
    return out


# ── Messidor-2 ────────────────────────────────────────────────────────────────
def load_messidor2(roots):
    """1 744 gradable images, DR 0-4 adjudicated, DME as a PARTIAL label."""
    csv_path = find_file(roots, "messidor_data", must_end=".csv")
    if not csv_path:
        return []
    img_dir = find_dir(roots, "messidor-2", "preprocess") or find_dir(roots, "messidor")
    idx = index_images(img_dir)
    rows = _rows(csv_path)
    if not rows:
        return []
    # Mirrors disagree on column names: the Google Brain original ships
    # image_id/adjudicated_dr_grade, other mirrors rename them id_code/diagnosis.
    cols = rows[0].keys()
    id_col = "image_id" if "image_id" in cols else "id_code"
    dr_col = "adjudicated_dr_grade" if "adjudicated_dr_grade" in cols else "diagnosis"
    out = []
    for r in rows:
        if str(r.get("adjudicated_gradable", "")).strip() != "1":
            continue          # ungradable rows carry no DR or DME grade at all
        fn = (r.get(id_col) or "").strip()
        if not fn or not str(r.get(dr_col, "")).strip():
            continue
        path = idx.get(fn) or idx.get(os.path.splitext(fn)[0])
        if not path:
            continue
        referable = int(r["adjudicated_dme"])
        out.append({
            "uid": f"MESSIDOR2_{os.path.splitext(fn)[0]}",
            "corpus": "Messidor-2",
            "path": path,
            "dr": int(r[dr_col]),
            # exact grade unknown when not referable -- see the module docstring
            "dme": 2 if referable else None,
            "dme_candidates": (2,) if referable else (0, 1),
            # graded referable/not-referable only -- never scored on the 3-class metric
            "dme_label_space": "binary",
            "fovea": None, "optic_disc": None,
        })
    return out


# ── APTOS 2019 ────────────────────────────────────────────────────────────────
def load_aptos(roots):
    """DR only. Reserved as the external DR test set (PROTOCOL.md §2/§6)."""
    out = []
    seen = set()
    for frag in ("train", "valid", "test"):
        p = find_file(roots, "aptos", frag, must_end=".csv") or find_file(roots, frag, must_end=".csv")
        if not p:
            continue
        try:
            rows = _rows(p)
        except Exception:
            continue
        if not rows or "id_code" not in rows[0] or "diagnosis" not in rows[0]:
            continue
        idx = index_images(dataset_root_of(p, roots))
        for r in rows:
            code = r["id_code"].strip()
            if code in seen:
                continue
            path = idx.get(code) or idx.get(os.path.splitext(code)[0])
            if not path:
                continue
            seen.add(code)
            out.append({
                "uid": f"APTOS_{os.path.splitext(code)[0]}",
                "corpus": "APTOS",
                "path": path,
                "dr": int(r["diagnosis"]),
                "dme": None, "dme_candidates": None, "dme_label_space": None,
                "fovea": None, "optic_disc": None,
            })
    return out


# ── EyePACS 2015 ──────────────────────────────────────────────────────────────
_EYEPACS_RE = re.compile(r"^(\d+)_(left|right)$", re.I)


def load_eyepacs(roots):
    """
    DR only, ~35 k images. Pretraining corpus, never a test set -- single-grader labels.

    The one corpus where patient grouping is free: filenames are `<patient>_<left|right>`,
    so the two eyes of a patient are explicitly linked and can be kept on the same side of
    any split (PROTOCOL.md §1).
    """
    p = find_file(roots, "trainlabels", must_end=".csv") or \
        find_file(roots, "eyepacs", "train", must_end=".csv")
    if not p:
        return []
    try:
        rows = _rows(p)
    except Exception:
        return []
    key = "image" if rows and "image" in rows[0] else "id_code"
    lvl = "level" if rows and "level" in rows[0] else "diagnosis"
    if not rows or key not in rows[0] or lvl not in rows[0]:
        return []
    idx = index_images(dataset_root_of(p, roots))
    out = []
    for r in rows:
        code = r[key].strip()
        path = idx.get(code) or idx.get(os.path.splitext(code)[0])
        if not path:
            continue
        m = _EYEPACS_RE.match(code)
        out.append({
            "uid": f"EYEPACS_{code}",
            "corpus": "EyePACS",
            "path": path,
            "dr": int(r[lvl]),
            "dme": None, "dme_candidates": None, "dme_label_space": None,
            "fovea": None, "optic_disc": None,
            "patient": f"EYEPACS_P{m.group(1)}" if m else None,
        })
    return out


# ── assembly ──────────────────────────────────────────────────────────────────
LOADERS = {"IDRiD": load_idrid, "Messidor-2": load_messidor2,
           "APTOS": load_aptos, "EyePACS": load_eyepacs}


def build(roots, corpora=("IDRiD", "Messidor-2")):
    """Manifest for the named corpora, with a `group` on every row.

    `group` is the unit that splits and bootstrap resampling respect. EyePACS gives real
    patient ids for free; the others have none published, so each image is its own group
    until the near-duplicate pass (src/dedup_groups.py) merges some of them. The known
    gap -- fellow-eye pairs in Messidor-2 and APTOS -- is recorded in PROTOCOL.md §1 and
    reported with every result rather than assumed away.
    """
    rows = []
    for name in corpora:
        got = LOADERS[name](roots)
        print(f"[corpora] {name:11s} {len(got):6d} images")
        rows += got
    for r in rows:
        r["group"] = r.get("patient") or r["uid"]
    seen = set()
    for r in rows:
        assert r["uid"] not in seen, f"duplicate uid {r['uid']}"
        seen.add(r["uid"])
    return rows


def summarise(rows):
    import collections
    by = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        by[r["corpus"]]["n"] += 1
        by[r["corpus"]][f"dr{r['dr']}"] += 1
        if r["dme_candidates"] is not None:
            by[r["corpus"]]["dme_supervised"] += 1
        if r["dme"] is not None:
            by[r["corpus"]]["dme_exact"] += 1
    lines = []
    for corpus, c in sorted(by.items()):
        dr = " ".join(f"DR{i}={c[f'dr{i}']}" for i in range(N_DR))
        lines.append(f"  {corpus:11s} n={c['n']:6d}  {dr}   "
                     f"dme_supervised={c['dme_supervised']:5d} "
                     f"(exact={c['dme_exact']})")
    return "\n".join(lines)
