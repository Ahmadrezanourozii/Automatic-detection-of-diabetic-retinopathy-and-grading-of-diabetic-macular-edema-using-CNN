# FILE_MAP.md — repository structure

## Knowledge base (read in this order)
| file | role |
|---|---|
| `STATE.md` | where the project stands; **read first every session** |
| `HANDOFF.md` | session handover: what stopped mid-flight and what to do next |
| `PROTOCOL.md` | frozen evaluation protocol, §1–§10. Changes need a new dated section |
| `FINDINGS.md` | **authoritative** findings F1–F10 |
| `EXPERIMENTS.md` | ledger: what each run did, with its pre-registered hypothesis |
| `ISSUES.md` | 27 entries, mostly silent failures and the guards built against them |
| `IDEAS.md` | backlog with verdicts (I01–I24) |
| `LITERATURE.md` | to be reconciled with `CLAIMS (2).md` |
| `GLOSSARY.md` | locked Persian terminology and orthography |
| `data/DATASETS.md`, `data/LABEL_MAPPING.md` | corpora and label semantics |

## Campaign documents (supplied by the owner)
| file | role |
|---|---|
| `CLAIMS (2).md` | literature ledger P1–P12. **Has its own F-numbering — use the crosswalk at its top** |
| `dr-dme-campaign-brief.md` | five-phase experiment plan (Phase 0–4) |
| `FINAL-PROMPT.md` | master execution prompt, Parts A–D. Crosswalk inserted at the top |

## Source (`src/`)
| file | role |
|---|---|
| `train.py` | training loop. Flags: `--macula`, `--lpft`, `--head {ordinal,softmax,coral,corn}` |
| `model.py` | `MultiOutputNet`, `OrdinalHead`, `CoralHead`, CORN loss/decode, `retfound:` backbone |
| `corpora.py` | dataset loading; `MESSIDOR_SOURCE` selector |
| `manifest.py` | **consumption manifests (§9)** — build, retrofit, compare, `require_same` |
| `metrics.py` | QWK, per-class recall, group bootstrap, paired bootstrap |
| `compare_matched.py` | paired bootstrap at **matched calibration** (§4.1) |
| `compare_runs.py` | paired bootstrap at shipped cut-points |
| `ensemble_oof.py` / `ensemble_external.py` | ensembling on dev pool / on APTOS (I23) |
| `tune_thresholds.py` | cross-fitted cut-point tuning |
| `t1_transfer_gap.py` | T1: how far a dev-fitted threshold lands on APTOS |
| `fovea.py` | fovea localiser (E13gate) + `fit_predict` for I07 |
| `idrid_derivation_gate.py` | **Part A** geometric derivation test |
| `retfound_probe.py` | I24 Stage 1 cached linear probe (+ `--imagenet-baseline` control) |
| `compare_probes.py` | frozen probe vs frozen probe |
| `verify_shared_dataset.py` | account-2 shared-dataset hash check |
| `diagnose_thesis_config.py` | D01 tiny-batch diagnostic on the original Keras code |
| `eval_external.py` | external evaluation (`--folds`, recipe-named artefacts) |
| `report.py` | regenerates thesis tables; **reads `data/selected_model.json`** |
| `check_invariants.py` | §8 + §9 invariants; `--compare` for consumption |
| `lint.py` | unbound-name check — **run before every push** |

## Kaggle
| path | role |
|---|---|
| `kaggle/build_kernel.py` | notebook generator. `--account`, `--script`, `--code-dataset`, `--ext-args`, `--from-run` (comma-separated) |
| `kaggle/fetch.py` | fetch + archive; verifies `CODE COMMIT`; `KAGGLE_OWNER` env for account 2 |
| `kaggle/dr-dme-*/` | account 1 kernels (`ah22reza`) |
| `kaggle2/dr-dme-*/` | account 2 kernels (`reza12123`) |

## Outputs
| path | role |
|---|---|
| `runs/<ID>/` | `results.json`, logs, `oof_*.npz`, external predictions, `PROVENANCE.md` |
| `docs/generated/` | every generated table/figure — **no hand-typed numbers** |
| `docs/PARTA_preregistration.md` | Part A criterion, committed before any number |
| `docs/RETFound_provenance.md` | checkpoint hashes, licence, card-mismatch resolution |
| `docs/T1_referral_threshold_candidates.md` | T1 candidates with verified provenance |
| `results/VERIFICATION-*.md` | paper verification records |
| `results/VERIFICATION-STATUS.md` | what is verified vs UNVERIFIED |
| `thesis/parta_report.tex` | Part A write-up (Persian) — **done** |
| `thesis/chapter3.tex` | روش کار — **done** |
| `thesis/chapter4.tex` | نتایج — **NOT WRITTEN** |
| `tools/farsi_lint.py` | forbidden-word counter; must reach 0 |
| `data/selected_model.json` | **the declared reported model** |
| `data/splits/dev_v1.json` | frozen split, fingerprint `0cfbbfeb081999af` |

## Outside the repo
- Datasets, prior code, checkpoints: `~/Library/CloudStorage/GoogleDrive-…/My Drive/Alireza/`
- LaTeX thesis: `~/Desktop/Alireza Thesis-25/thesis-chegeni/` (results in `tex/chapter4.tex`)
- `.env` (gitignored): `GITHUB_TOKEN`, `KAGGLE_API_TOKEN`, `KAGGLE_API_TOKEN_2`, `HF_TOKEN`
