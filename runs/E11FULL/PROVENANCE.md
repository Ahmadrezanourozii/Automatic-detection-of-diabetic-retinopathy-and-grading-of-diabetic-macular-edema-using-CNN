# E11FULL — assembled, not run

Folds 0–2 from **E11** (commit `ebe8a61`), folds 3–4 from **E19E11C** (commit `568ee7e4`).

## Checks performed before assembling

* **Consumption** (`ISSUES.md` §26, `PROTOCOL.md` §9): `manifest.require_same` **PASSES**.
  Both kernels mounted the identical five datasets, including
  `borhan2003/messidor-...-jpg-format`, the native-resolution mirror. This is the same check
  that **refused** the earlier attempt — E19E11B omitted `--messidor-hi` and would have
  combined folds trained on different pixels.
* **Config**: every training field identical. Difference set empty.
* **Split**: fingerprint `0cfbbfeb081999af` on both.

## The cross-commit risk — CLOSED BY INSPECTION, 2026-09-05

The two runs pin different commits, nine days apart. Rather than leave that as an argument,
every hunk between `ebe8a61` and `568ee7e4` in the files a training run imports was read, and
each is inert on the path a `--backbone tf_efficientnet_b3` run **without** `--macula` and
**without** `--lpft` executes:

| file | hunks | why it cannot change this run |
|---|---|---|
| `preprocess.py`, `splits.py` | 0 | unchanged |
| `model.py` | 1 | `forward(x)` became `forward(x, xm=None)`. With `xm=None`, `fm = f`, so the body reduces **exactly** to the previous `return self.dr_head(f), self.dme_head(f)`. |
| `corpora.py` | 2 | adds `MESSIDOR_SOURCE`, default `prefer-native`, whose branch is `idx = dict(hi_idx)` then `setdefault` from `lo_idx` — **semantically identical** to the previous `update(hi)` + `setdefault(lo)`. |
| `train.py` | 14 | `macula_crop()` is a new function never called when `fovea is None`; the crop block sits inside `if self.fovea is not None`; the LP-FT block inside `if args.lpft`; `FundusDataset.__init__` gains defaulted parameters; `predict()` pairs each view with `None` when `xm is None`, so `model(v, None)` ≡ the old `model(v)`; `build_cache` gains a `cache_size >= size` assertion that does not fire at 560 ≥ 448. |
| `metrics.py` | 3 | affects only the metrics written into `results.json`, not the archived OOF logits. Every number reported for E11FULL is recomputed from those logits with one current `metrics.py`, so both halves are scored identically regardless. |

**What this is and is not.** This is a line-by-line verification that no changed statement is
reachable on this configuration — considerably stronger than the earlier "the flags are off"
argument. It is **not** a runtime proof: only re-running folds 0–2 at `568ee7e4` and comparing
predictions would be that, and it would cost ~9.5 h of quota. Given every hunk is either a new
uncalled function, a guarded branch, or a provably equivalent refactor, that spend is not
justified — but the distinction is recorded rather than glossed.
