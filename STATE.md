# STATE.md — read this first, every session

**Project:** Automatic detection of diabetic retinopathy (DR) and grading of diabetic macular
edema (DME) from a single retinal fundus photograph.
**Owner:** Alireza Chegeni (810102111), MSc, ECE, University of Tehran.
**Last updated:** 2026-08-30, end of session 2.

**Read in this order:** this file → `PROTOCOL.md` (frozen, §4.1/§4.2 are recent and load-bearing)
→ `FINDINGS.md` (what the project learned) → `EXPERIMENTS.md` (what each run did) →
`ISSUES.md` (23 entries, mostly silent failures) → `IDEAS.md` (backlog with verdicts).

---

## Where the project stands

The thesis this restarted from reported 91.6 % DR / 87.6 % DME with **no run behind either
figure**; the only archived evaluation scored 27.2 % and 15.9 %, both below the majority-class
floor (`ISSUES.md` §1). Everything below was computed, with intervals, from runs that are
archived and reproducible from a commit SHA.

## Current best result

| head | n | floor | accuracy | QWK | run |
|---|---|---|---|---|---|
| DR, 5-class (folds 0–2) | 1 362 | 52.3 % | **78.1 %** | **0.894** | E11 |
| DR, 5-class (5 folds) | 2 260 | 52.4 % | 74.4 % | 0.868 | E10 |
| DME, 3-class ungated (5 folds) | 516 | 47.1 % | **87.2 %** | **0.899** | E10 |
| Referable DME, binary | 2 260 | 82.6 % | 94.9 % | 0.819 | E10 |

**E11 (EfficientNet-B3 + EyePACS pretraining) is the best DR configuration internally**
(+0.029 QWK over E08, fold-matched, significant) but ran **3 folds only**, so its headline is
not on the same basis as the others until folds 3–4 are added (~3.2 h, queued).

### External validation — the number that matters at a defence

On **3 662 APTOS images never seen in training**, verified end-to-end by
`src/verify_external.py` (14 checks, 0 failures, intervals included):

> DR accuracy **73.5 %** (95 % CI 72.0–74.9), **QWK 0.897** (95 % CI 0.889–0.905), against a
> 49.3 % majority floor. Referable-DR sensitivity **99.53 %** at 84.28 % specificity.

E11's weights give QWK 0.903; the paired difference is +0.0063 [+0.0009, +0.0116] as shipped
and **indistinguishable once both models sit at matched operating points** (F3).

On IDRiD's official 103-image test split — the set the old thesis quoted 91.6 % on — the E05
pipeline scores **61.2 %**.

---

## What is running right now

**E13gate** — the fovea-localiser gate (`src/fovea.py`), on Kaggle. Acceptance was **fixed
before the numbers**: median out-of-fold error < 0.5 disc diameters, 90th percentile < 1.0 DD.
Stated in DD because that is the unit the DME grade uses, and 1 DD is the size of the whole
decision region.

**Its outcome decides the next experiment's scope**, not whether it happens:
* **Pass** → the macula-pooled DME head extends to Messidor-2 (2 260 images of DME
  supervision with position information), carrying the stated, unvalidatable assumption that
  the localiser transfers to a corpus with no fovea ground truth.
* **Fail** → confined to IDRiD's 516 — still the entire 3-class DME evaluation set, so the
  experiment is still worth running, with a narrower claim and no unvalidated transfer.

Report the number either way.

## Next three planned experiments, and why

| # | experiment | est. | hypothesis / falsifying outcome |
|---|---|---|---|
| 1 | **Macula-pooled DME head** (scope set by E13gate) | ~2 h | DME is stuck because global average pooling dilutes the decisive region ~16× — 1 DD is ~55 px in a 448 px image. Pooling the DME head's features at the fovea should move DME QWK. **Falsified if** DME QWK moves less than its ±0.03 interval, which would mean the DME ceiling is data, not architecture. |
| 2 | **T1 — deliberate referral threshold** | 0 h GPU | The 99.53 % sensitivity came from an unchosen `sigmoid > 0.5`. Choose the target **first**, justify it against screening practice, fit cross-fitted on the development pool (**never** APTOS), report the achieved sensitivity, the specificity cost, and the **transfer gap** — which F3 predicts will be non-zero and is itself a result. **Falsified if** no threshold transfers within a useful band, in which case the recommendation becomes "recalibrate locally" with F4's 200-image figure attached. |
| 3 | **I20 — native-resolution test** | ~4 h | Gated on I19, which **passed in aggregate only**. Must report aggregate QWK as attributable to resolution and **per-class results explicitly under the source caveat** (see `IDEAS.md` I20). |

Then, in order: E11 folds 3–4 (3.2 h), multi-seed ensemble (~6 h), re-implementing the two
comparison baselines on our split (~4 h — closes the oldest gap in the thesis, since the
comparison table has been indirect since the 91.6 % failed to reproduce).

**Budget:** 30 h/week, not rolling over — about 360 h over three months. No single run over
~10 h. Checkpoint every epoch. Deferring work on cost grounds was a mistake made three times
and has been corrected.

---

## The six findings (`FINDINGS.md`)

* **F1** — the external Mild collapse is **calibration, not capacity**, and QWK conceals it.
  Mild recall 5–6 % on APTOS vs 36–45 % internally; recalibration alone recovers it to 77.8 %.
  Recalibrating *for QWK* reaches only 19.5 %, and recovering Mild **costs** QWK (0.903 →
  0.887) — the primary metric prices against the fix.
* **F3** — **per-class comparisons between models are dominated by cut-point placement.** Every
  E11-vs-E08 per-class difference changes under recalibration; one reverses sign. The aggregate
  external advantage becomes indistinguishable at matched operating points. **Recalibration is
  worth +7.3 macro-recall points; the whole backbone change is worth +1.5, n.s.**
* **F4** — recalibration needs **about 200 labelled local images**: +5.09 [+0.32, +7.28] points
  with a 1.5 % risk of harm. **Below 100, roughly one attempt in four makes the model worse.**
* **F5 (settled)** — **macro-recall must not be primary.** Tuning for it costs **19.2 points of
  referable sensitivity** (7 missed referable patients → 292), because Mild is not a referable
  grade. Adopted: **referable sens/spec as the operationally primary pair, QWK as the primary
  grading metric, macro-recall as a diagnostic never optimised.**
* **F6** — a deployment recommendation evaluated only at its **mean** can be harmful; the
  probability of harm must be reported.
* **F2** — Messidor-1 cannot be the external DME test set: 1 057 of 1 200 images are already in
  the development pool, the 143 survivors are 78 % one site, site predicts DME grade at
  χ² p = 1.5 × 10⁻¹⁴, and the interval on 143 images would be ±0.073 QWK. **"No external DME
  validation" is a declared limitation with a measured reason.**

## The standing negative result

**No intervention has significantly improved 3-class DME on QWK, the primary metric** — across
pretraining, schedule, backbone, and architecture-plus-data combined. The one exception is
effective resolution (E10), which moved DME **accuracy** +3.04 pts while leaving QWK
indistinguishable. Every other gain in this project is on the DR head.

## The two protocol rules added this session

**§4.1 — no difference is attributed to representation until it survives matched calibration.**
The rule has now overturned three predictions **in both directions** (Mild: capacity → calibration;
backbone: representation → boundary; Moderate: artefact → **real difference**). The third row is
what licenses the other two — without a case where the check comes back positive, the rule is
indistinguishable from motivated reasoning. Regenerated into the thesis by `report.py` as
`docs/generated/calibration_record.tex`.

**§4.2 — every per-class claim carries two qualifications**: no multiple-comparison correction is
applied, so credibility comes from **replication across conditions** rather than a p-value in
isolation; and a per-class claim inherits every confound the aggregate does.

---

## Infrastructure

| piece | where |
|---|---|
| code | `github.com/Ahmadrezanourozii/Automatic-detection-...-using-CNN`, branch `main` |
| training | Kaggle notebook per run, clones the repo at a **pinned commit** |
| launch | `python kaggle/build_kernel.py --run-id EXX [--eyepacs] [--aptos] [--messidor-hi] [--external-only --from-run <slug>] [--script src/foo.py] --args "..."` then `kaggle kernels push -p kaggle/dr-dme-exx --accelerator "GPU T4 x2"` |
| fetch | `python kaggle/fetch.py --run-id EXX [--weights]` — verifies the log's `CODE COMMIT` against the pinned commit |
| archive | `runs/<ID>/` — `results.json`, `train_<stamp>.log`, `oof_*.npz`, `best_*.pt`, external predictions |
| thesis | `report.py` regenerates every table and figure into `docs/generated/`; `check_thesis_numbers.py` fails on any result-shaped number in prose |

**Before every push:** `python src/lint.py` (unbound names — has caught this class twice) and
`python src/check_invariants.py --idrid <path>`.

**Chapter 4 is rewritten** from computed numbers, with **0 result-shaped literals in prose**.

## Blocking / needed from the owner

1. **Rotate the credentials.** Two GitHub PATs and two Kaggle keys are in chat transcripts.
   They work and are in `.env` (gitignored, verified absent from git history).
2. **Defence date** — still unknown; it determines how much of the backlog is reachable.
3. **Messidor-1 errata are unverified** (`ISSUES.md` §22) — the download shipped no erratum
   document, so our table's provenance is a web page. Closing it means diffing against the
   current ADCIS page.
4. **The T1 sensitivity target** needs a clinical anchor, and every candidate standard must be
   checked against its source document before citation — a standard quoted from memory is
   exactly the unverified provenance this project keeps catching.
