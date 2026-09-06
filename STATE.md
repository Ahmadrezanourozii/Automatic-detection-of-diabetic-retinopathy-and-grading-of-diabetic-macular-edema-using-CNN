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

**The project had no established baseline.** The 91.6 % / 87.6 % figures in the earlier draft
were **placeholder numbers in a template** — never presented anywhere, never claimed as a
result, and never produced by any run (owner, 2026-08-31). There was nothing to reproduce and
nothing to beat. What did exist was one archived evaluation that **collapsed**: 27.2 % DR and
15.9 % DME, both below the majority-class floor, from a training run whose validation loss
rose 1.86 → 3.40. That collapse is real and still unexplained (`ISSUES.md` §1).

Everything below is therefore **the first real measurement this project has produced**,
computed with intervals from runs archived and reproducible from a commit SHA.

## Current best result

**The selected model is E10.** Selection is by development-pool ranking at matched
calibration, never by test score (`PROTOCOL.md` §3). Among the nine runs with a complete
5-fold out-of-fold prediction, E10 has the highest DR QWK; every number below follows from
that one choice.

| head | n | floor | accuracy | QWK (matched cuts) | QWK (shipped cuts) |
|---|---|---|---|---|---|
| DR, 5-class | 2 260 | 52.4 % | 74.8 % | **0.8749** | 0.868 |
| DME, 3-class ungated | 516 | 47.1 % | 86.6 % | **0.8948** | 0.899 |
| Referable DME, binary | 2 260 | 82.6 % | 94.9 % | — | 0.819 |

**Externally, on 3 662 held-out APTOS images: DR QWK 0.9026 [0.8941, 0.9108]**, accuracy
74.93 %, referable sensitivity 99.26 % at 84.32 % specificity. **This is the thesis headline.**
It is the selected model's score on data it has never seen, and the selection was made on the
development pool, never on APTOS (`PROTOCOL.md` §3).

**Superseded, recorded rather than deleted:**

| run | external DR QWK | why it is not the headline |
|---|---|---|
| E08 | 0.8972 | quoted as the headline until 2026-09-05; E08 is no longer the validation-selected model |
| E10 | 0.8867 | selected model until E11FULL completed its five folds |
| **E11FULL** | **0.9026** | **current — highest development-pool QWK, hence selected** |

Note that E08 scores highest on APTOS among several members of the archive. **That is not a
reason to report it**: choosing a model by its test score is selection on the test set, which
§3 forbids. The headline follows the validation ranking wherever it lands.

*Why matched cuts lead here:* T1 showed the shipped `sigmoid > 0.5` operating point is an
arbitrary default that means different things on different corpora, so a number decoded at it
is not a property of the model (§4.1, `docs/generated/t1_transfer_gap.md`).

**Two caveats on E10, both measured rather than assumed.** Its `--size 640` label is inflated:
the cache capped images at 560, so it is a 560 px run (`ISSUES.md` §18). E17NAT paid 7.2 h to
run a genuinely-640 px version and the difference was +0.0086 [−0.0025, +0.0199] — nothing. So
the label is wrong and the number is not affected.

**E11 (EfficientNet-B3) scored DR QWK 0.894 on folds 0–2 and may well be better**, but three
folds is not the same basis as five, and completing it is **blocked on GPU quota**. Until then
it is not the selected model and its figure is not quoted as a headline.

### Ensembling — investigated, gained on development, **falsified externally** (I23)

Recorded in full rather than tidied, because the sequence is the point.

| stage | DR QWK | what changed |
|---|---|---|
| first reported | 0.8933 | logit-average of all 9 five-fold runs, dev pool |
| corrected | **0.8828** | consumption manifests showed three of four ensembles averaged over **different renderings of the same eyes**; the homogeneous set is the honest one (`PROTOCOL.md` §9) |
| **falsified** | **—** | on APTOS the gain is **+0.0011 [−0.0033, +0.0053]** against the validation-selected single model — indistinguishable. The development gain was optimism. |

**Ensembling is closed as a route to a higher number.** What survives is the *asymmetry*:
on the development pool it moved DR and not DME, which is the variance-versus-supervision
distinction F7 rests on — now labelled as a development-pool statement.

### External validation — the number that matters at a defence

On **3 662 APTOS images never seen in training**, verified end-to-end by
`src/verify_external.py` (14 checks, 0 failures, intervals included):

> DR accuracy **73.5 %** (95 % CI 72.0–74.9), **QWK 0.897** (95 % CI 0.889–0.905), against a
> 49.3 % majority floor.

⚠️ **The referable-DR operating point is NOT a model property — corrected 2026-08-31.** It was
previously quoted here as "sensitivity 99.53 % at 84.28 % specificity" without qualification.
That pair is what an **arbitrary, unchosen `sigmoid > 0.5`** happens to do *on APTOS*. The
same model at the same cut-point sits at the opposite end of its own curve on the development
pool:

| corpus | referable-DR sensitivity | specificity |
|---|---|---|
| development pool (n = 2 260) | **86.28 %** | **96.42 %** |
| APTOS external (n = 3 662) | **99.53 %** | **84.28 %** |

On dev that default lands near the IDx-DR regulatory floor; on APTOS it lands past EyeArt's
reported operating point. **Neither number may be quoted as the model's sensitivity.** Quote
the operating curve, or a threshold that was deliberately chosen — which is what T1 exists to
produce. See `docs/T1_referral_threshold_candidates.md`, and `PROTOCOL.md` §4.1: an untuned
default is a hyper-parameter that has been chosen, not one that has been avoided.

E11's weights give QWK 0.903; the paired difference is +0.0063 [+0.0009, +0.0116] as shipped
and **indistinguishable once both models sit at matched operating points** (F3).

On IDRiD's official 103-image test split the E05 pipeline scores **61.2 %** — reported for
completeness and read against `PROTOCOL.md` §7: that split carries a ±6.3 pt interval and is
**not headline material**.

---

## What is running right now

## GPU quota — EXHAUSTED for the week

**30.00 h weekly Kaggle quota reached 2026-09-01.** Blocked until it resets: EXTE17NAT (the
sixth I23 member, cannot change a gain already measured at zero), **E19E11C** (E11 folds 3–4,
which decides whether EfficientNet-B3 becomes the selected model), and **RETFound Stage 1**
(staged and ready, `src/retfound_probe.py`, weights verified and uploaded).

**~7.9 h of the 30 was E19E11B** — the run wasted by omitting `--messidor-hi`, which mounted
a different image source and could not be combined with E11's folds (`ISSUES.md` §26). A
quarter of the week's budget on one avoidable error.

**E13gate — PASSED, 2026-08-31.** The fovea localiser (`src/fovea.py`), out-of-fold over all
516 IDRiD images, against a threshold fixed before any number was seen:

| metric | result | threshold |
|---|---|---|
| median error | **0.196 DD** | < 0.5 DD ✅ |
| 90th percentile | **0.433 DD** | < 1.0 DD ✅ |
| 99th percentile | 0.857 DD | — |
| within 0.5 DD / 1 DD | 92.8 % / 99.0 % | — |

Per-fold medians 0.191 / 0.188 / 0.260 / 0.189 / 0.180 — stable, no fold carrying it.
Archived at `runs/E13gate/fovea_gate.json`, commit `89f8918`, **no `results.json`** (see
`ISSUES.md` §24 for why that absence is the point).

**Consequence: the macula-pooled DME head extends to Messidor-2** — 2 260 images of DME
supervision with position information rather than IDRiD's 516. The caveat travels with every
number it produces: the localiser is validated on IDRiD only, and Messidor-2 has no fovea
ground truth, so the transfer is an assumption **no experiment available to this project can
check**. `IDEAS.md` I07 carries this as a binding condition, along with the requirement to
report the IDRiD-only result alongside the pooled one.

**D01 — the tiny-batch diagnostic (CPU, no quota).** Partially reported; suspects 2 and 3
still running. See "the LP-FT re-ranking" below.

## Next three planned experiments, and why

**Re-ranked 2026-08-31 on evidence.** LP-FT was priority 1 because it promised to lift the
floor under every number; D01 withdrew that promise. I07 moved the other way — the gate
widened it from 516 to 2 260 images. Ordering follows current expected value, not yesterday's.

| # | experiment | est. | hypothesis / falsifying outcome |
|---|---|---|---|
| 1 | **I07 — macula-pooled DME head**, at 2 260 images | ~2 h | GAP dilutes the decisive region ~16× — 1 DD is ~55 px in a 448 px image. Pooling the DME head's features at the fovea should move DME QWK. **Falsified if** DME QWK moves less than its ±0.03 interval, meaning the DME ceiling is data, not architecture — **a real possibility given the standing negative result, named in advance.** Three binding conditions in `IDEAS.md` I07: the transfer caveat travels with every number; the IDRiD-only 516 result is reported alongside the 2 260 one; §4.1 matched calibration before any per-class attribution. |
| 1b | **I21 — LP-FT**, as an independent question | ~3 h | Part one only: `src/train.py` has no linear-probe phase and trains the backbone from step 0 at 1e-4. **Part two withdrawn on D01 evidence** — the floor-lift argument is dead. Expected value **uncertain, not likely**. **Falsified if** it does not beat current at matched calibration. |
| 2 | **T1 — deliberate referral threshold** | 0 h GPU | The 99.53 % sensitivity came from an unchosen `sigmoid > 0.5`. Choose the target **first**, justify it against screening practice, fit cross-fitted on the development pool (**never** APTOS), report the achieved sensitivity, the specificity cost, and the **transfer gap** — which F3 predicts will be non-zero and is itself a result. **Falsified if** no threshold transfers within a useful band, in which case the recommendation becomes "recalibrate locally" with F4's 200-image figure attached. |
| 3 | **I20 — native-resolution test** | ~4 h | Gated on I19, which **passed in aggregate only**. Must report aggregate QWK as attributable to resolution and **per-class results explicitly under the source caveat** (see `IDEAS.md` I20). |

Then: E11 folds 3–4 (running as E19E11B), multi-seed ensemble (~6 h), and **re-implementing
the two literature baselines on our split (~4 h)**. That last item is now **the single most
important thing in the queue for interpreting our own numbers**: with no internal baseline of
any kind, a fair like-for-like comparison on our exact split is the *only* thing that will
say whether DR QWK 0.865 is good.

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

## The standing negative result — now upgraded to a positive finding (F7)

**No intervention has significantly improved 3-class DME on QWK, the primary metric** — across
pretraining, schedule, backbone, and architecture-plus-data combined.

> ⚠️ **This is a statement about the DME head only, and must not be generalised to DR.** On DR
> the backbone demonstrably matters: E11FULL vs E10, both EyePACS-pretrained at 448 with the
> backbone the only difference, is **+0.0207 [+0.0105, +0.0320], significant**. The same
> comparison on DME is indistinguishable (−0.0067 [−0.0313, +0.0172]). A summary of the form
> "backbone swaps did not work here" is **false for DR and true for DME**, and the two heads
> have behaved differently under almost every intervention this project has run. The one exception is
effective resolution (E10), which moved DME **accuracy** +3.04 pts while leaving QWK
indistinguishable. Every other gain in this project is on the DR head.

**E14MAC closed this out on 2026-08-31.** The macula-centred crop — an architecture taken
directly from the DME label's own clinical definition, using a localiser validated at median
0.196 DD against a threshold fixed before the numbers, with the fovea inside the crop window
in over 99 % of images — **did not move DME QWK**: 0.8764 vs 0.8538 at matched calibration,
+0.0237 [−0.0094, +0.0566], inside the ±0.03 band declared in advance.

**So the DME limit is supervision, not modelling (`FINDINGS.md` F7).** 516 images carry a
3-class label and **51 of them are the middle grade**; Messidor-2's 1 744 supply only a binary
label and cannot populate it. Further architectural work on this head is not the lever — I15
(auxiliary hard-exudate segmentation from IDRiD's 81 masks) is promoted as the one remaining
idea that adds information rather than rearranging it.

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

1. ~~**Rotate the credentials.**~~ **Deferred by owner decision on 2026-08-30 — not
   blocking, do not stall a session on it.** The tokens are live and working, and the
   exposure was re-verified this session: `.env` is gitignored, appears in **0 commits**,
   and neither value occurs in any blob across all refs. The only exposure is the chat
   transcripts, accepted as a risk. Separately, `.env` now uses **`KAGGLE_API_TOKEN`**:
   CLI 2.2.4 ignores the old `KAGGLE_USERNAME`/`KAGGLE_KEY` pair and fails with a bare
   "Authentication required" that never names the cause. That fix stays regardless.
2. ~~**Defence date**~~ — **answered 2026-08-30: horizon ~3 months, ~360 GPU-hours.**
   Quota is explicitly *not* the constraint. More quota is permission to stop deferring
   experiments that settle something — it is not permission to search blindly.
3. **Messidor-1 errata are unverified** (`ISSUES.md` §22) — the download shipped no erratum
   document, so our table's provenance is a web page. Closing it means diffing against the
   current ADCIS page.
4. **The T1 sensitivity target** needs a clinical anchor, and every candidate standard must be
   checked against its source document before citation — a standard quoted from memory is
   exactly the unverified provenance this project keeps catching.
