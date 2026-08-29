# STATE.md — read this first, every session

**Project:** Automatic detection of diabetic retinopathy (DR) and grading of diabetic
macular edema (DME) from a single retinal fundus photograph.
**Owner:** Alireza Chegeni (810102111), MSc, ECE, University of Tehran.
**Last updated:** 2026-08-26 — session 2. GPU quota exhausted; E10/E11 running, CPU-only work until the reset (~59 h).

---

## Where the project stands

Session 1 was an audit of the work done up to July 2026. It did **not** train anything.
The audit's conclusion is the single most important fact in this repository:

> **The results reported in the existing thesis (chapter 4) cannot be reproduced from
> anything in this project, and the evidence indicates they were never computed.**
> The only evaluation output ever saved to disk shows a model that scores **27.2 % on
> 5-class DR** and **15.9 % on 3-class DME** — both *below* the majority-class floor.

Everything downstream of that follows: there is no working baseline, so there is nothing
to "improve on". The project restarts from a trivial baseline and builds up.

Run `python3 src/audit_baseline.py --drive "<Google Drive Alireza folder>"` to reproduce
every number in this file. See `ISSUES.md` §1 for the full diagnosis.

## Current best result

**E11 — EfficientNet-B3 with EyePACS pretraining.** Significantly the best DR configuration:
+0.029 QWK over E08 [+0.012, +0.046], fold-matched.

| head | n | floor | accuracy | QWK |
|---|---|---|---|---|
| DR, 5-class *(folds 0–2)* | 1 362 | 52.3 % | **78.1 %** | **0.894** |
| DME, 3-class ungated *(folds 0–2)* | 314 | 48.1 % | **87.6 %** | **0.902** |

**Caveat: three folds, not five.** E11 was sized to fit the wall clock and finished in 9.49 h.
Its headline is not on the same basis as the other runs until folds 3–4 are added (~3.2 h,
queued first when quota returns). The paired comparisons above *are* fold-matched and valid.

Five-fold reference points, for comparison: E10 DR 74.4 % / QWK 0.868, DME 87.2 % / 0.899;
E08 DR 74.2 % / 0.860.

**External validation (E08X, verified):** on 3 662 APTOS images never seen in training, DR
accuracy **73.5 %** (95 % CI 72.0–74.9), **QWK 0.897** (95 % CI 0.889–0.905), floor 49.3 %;
referable-DR sensitivity **99.5 %** at 84.3 % specificity. The model over-grades APTOS by
about one step, so ranking transfers and cut-points do not. **This used E08's weights — it is
an open question whether E11 generalises as well, and must be measured rather than assumed.**

On IDRiD's official 103-image test split — the set the old thesis quoted 91.6 % on — the E05
pipeline scores **61.2 %**.

## Chapter 4 — rewritten from computed numbers

`thesis-chegeni/tex/chapter4.tex` is rewritten. It contains **no result numbers in prose**;
every table and figure is `\input`-ed from `docs/generated/`, which `src/report.py` writes
from archived `results.json` files. Re-running that script after a new run updates the
chapter without touching its text.

`src/check_thesis_numbers.py` enforces this structurally: **0 result-shaped literals in
prose**. Its self-test (`tests/thesis_numbers_selftest.tex`) still goes red on both a
fabricated figure and a true one, because being correct does not exempt a result from
belonging in a table rather than a sentence.

The chapter reports what was actually found, including the negative result below, and states
the three caveats attached to the external number.

## Read also: FINDINGS.md

Results that are about the problem rather than about a run. Currently two:

* **F1 — the external Mild collapse is calibration, not capacity, and QWK conceals it.**
  Mild recall is 5–6 % on APTOS against 36–45 % internally, while every other class transfers
  within ~25 %. Moving cut-points alone recovers it to **77.8 %**. But recalibrating for QWK
  reaches only 19.5 %, and recovering Mild *costs* QWK (0.903 → 0.887) — so the primary metric
  prices against the fix. Per-class recall must be reported beside every headline.
* **F2 — Messidor-1 cannot be the external DME test set**, with the overlap, site-bias and
  power numbers that show why.
* **T1 (queued, zero GPU) — select the referral threshold deliberately.** The 99.53 %
  sensitivity came from an unchosen `sigmoid > 0.5`. Pick the target first, justify it against
  screening practice, fit cross-fitted on the development pool (never APTOS), and report the
  transfer gap. Third appearance of the unchosen-default pattern, this time on the number the
  clinical claim rests on.
* **F5 — macro-recall must not be the primary metric (settled).** Tuning for it costs **19.2 points of
  referable-DR sensitivity** (99.53 % → 80.36 %; 7 missed referable patients → 292), because
  Mild is not a referable grade and macro-recall prices a Mild error like a Proliferative one.
  Recommended: referable sens/spec as the operationally primary *pair*, QWK as the primary
  grading metric, macro-recall as a diagnostic only. Adopted; the original counter-position
  and the number that overturned it are recorded in F5.
* **F6 — a deployment recommendation evaluated at its mean can be harmful.** Generalises F4.
* **F4 — recalibration needs about 200 labelled local images.** At n = 200 it recovers ~70 %
  of the achievable gain with a 1.5 % risk of harm; below 100 images roughly one attempt in
  four makes the model *worse*, so the advice there is not to recalibrate at all.
* **F3 — per-class comparisons between models are dominated by cut-point placement.** Every
  per-class E11-vs-E08 difference on APTOS changes under recalibration; one reverses sign. The
  aggregate external advantage (+0.0063 QWK) becomes indistinguishable once both models sit at
  matched operating points. Recalibration is worth **+7.3 macro-recall points**; the whole
  backbone change is worth **+1.5, n.s.** Before attributing any difference to representation,
  check it survives matched calibration.

## The standing negative result

**No intervention has significantly improved 3-class DME on QWK, the primary metric.** That
now spans pretraining, schedule, backbone, and architecture-plus-data combined. The single
exception is effective resolution (E10), which moved DME **accuracy** +3.04 pts while leaving
QWK indistinguishable. Every other gain in this project is on the DR head. Either the 516-image DME evaluation set is too small to
resolve the differences (interval ≈ ±0.03), or the interventions tried do not address what
that head lacks — which is exudate position, not capacity. Messidor-1 would settle the first;
the macula-centred crop would test the second.

## Operating mode

**Autonomous, since 2026-08-25.** The owner has asked for the loop to run without
per-step approval: push to GitHub → run on Kaggle → fetch logs → analyse → change
something → run again. Protocol and honest reporting stay as they are; what is dropped is
waiting for sign-off between iterations.

## Infrastructure — live

| piece | where |
|---|---|
| code | `github.com/Ahmadrezanourozii/Automatic-detection-...-using-CNN` (branch `main`) |
| training | Kaggle notebook `ah22reza/dr-dme-<runid>`, GPU, clones the repo at a pinned SHA |
| data | attached Kaggle datasets — nothing uploaded, nothing downloaded locally |
| archive | `runs/<RUN_ID>/results.json` + `train.log` + `oof_*.npz`, committed permanently |
| launch | `python kaggle/build_kernel.py --run-id EXX --args "..." --push` |
| fetch | `python kaggle/fetch.py --run-id EXX --wait` |

Kaggle datasets attached: `aaryapatel98/indian-diabetic-retinopathy-image-dataset`,
`google-brain/messidor2-dr-grades`, `mariaherrerot/messidor2preprocess`.

## GPU budget — binding constraint

**30 h/week, currently 0 h left, resets in ~59 h.** Rules adopted after E07 was killed at the
12-hour wall having produced 4 of 5 folds:

* **No single run over ~10 h wall clock.** Split long configurations across sessions by fold.
* Every run checkpoints **every epoch** and writes `results.json` **after every fold**, so a
  kill costs one epoch, not the run (`ISSUES.md` §17 — verified: E07's full working directory
  was published despite being cancelled).
* An empty Output tab on a **running** kernel means "not published yet", never "nothing saved".

## Provenance discipline (added after §20)

`src/check_invariants.py` verifies that every `runs/<ID>/results.json` names `<ID>` and a
commit reachable on the current branch, resolving documented rewrites through
`data/commit_remap.json`. Currently: **6 outputs verified, 1 remap resolved**. Do not rewrite
git history while archives reference it.

`src/verify_external.py` re-derives an external claim from its archived artifact. The APTOS
result passes **14 of 14** checks, **intervals included**: every APTOS group is a single
image, so the group bootstrap is an image bootstrap and the confusion matrix determines the
interval. No GPU re-run was needed.

> **Defence claim, fully verified:** on 3 662 unseen APTOS images, DR accuracy **73.5 %**
> (95 % CI 72.0–74.9), **QWK 0.897** (95 % CI 0.889–0.905), floor 49.3 %.

That shortcut expires the moment groups stop being one image each — when Messidor-1 arrives,
or patient ids are recovered. `eval_external.py` archives per-image predictions and groups
from now on, and `verify_external.py` checks the precondition before relying on it.

## What is running right now

| run | what it tests | status |
|---|---|---|
| **E10** | native-resolution Messidor-2 at 640 px | running — **CONFOUNDED, see below** |
| **E11** | EfficientNet-B3 **+** EyePACS pretraining, 3 folds, 30 ep | running, ~10.8 h estimated |

**E10's numbers must not be read as a resolution result.** `build_cache()` caps images at
560 px and E10 trains at 640, so every image is upsampled and the native 2240 × 1488 mirror is
discarded before training sees it (`ISSUES.md` §18). It is a 560 px run paying 640 px compute.
Flagged in `EXPERIMENTS.md` ahead of its numbers landing.

**I cannot stop a Kaggle session.** The CLI offers no `cancel` — only `delete`, which destroys
the kernel and its history rather than stopping it cleanly. The watch alerts at 10 h and 11 h
elapsed on E11; stopping has to be done from the Kaggle UI. A stop is safe: per-epoch
checkpoints and per-fold `results.json` are already written, and E07 proved a cancelled run
still publishes its whole working directory (`ISSUES.md` §17).

## Blocking

1. **Credentials must be rotated when convenient.** Two GitHub PATs and two Kaggle keys have
   now been pasted into chat transcripts. They work and are in `.env` (gitignored, verified
   absent from git history), but they are exposed.
2. **Unknown:** defence date, weekly GPU budget. Both affect planning, neither blocks work.
3. **To acquire:** Messidor-1 (external DME validation), after a duplicate check against the
   Messidor-2 mirror (`ISSUES.md` §3).

## Next three planned experiments

Ordered by information gained per GPU-hour. None may start before the protocol is agreed.

| # | Hypothesis | Falsified if | Cost |
|---|---|---|---|
| E01 | The prior training collapse is an optimisation failure, not a data failure: a frozen-backbone linear probe on ImageNet features separates DR grades well above the 33 % floor. | Linear probe ≤ floor + its CI → the signal is not in the representation, and preprocessing/resolution is the problem, not the training loop. | **running** — local MPS, no quota |
| E02 | The five-step preprocessing chain (green→CLAHE→blur) is not better than plain RGB. Same split, same backbone, chain on vs chain off. | Chain wins by more than the paired-bootstrap interval → the chain is load-bearing and the thesis' ablation claim survives. | ~1 GPU-h |
| E03 | Splitting on images rather than on eyes/patients inflates the score. Same pipeline, group-wise split vs naive random-image split, difference reported. | No gap → the corpora have no near-duplicate structure; record the number and move on. | ~1 GPU-h |

E01 is the gate. If a linear probe cannot beat the floor, no amount of architecture work will.

## Files a new session must read, in order

1. `STATE.md` (this file)
2. `PROTOCOL.md` — the frozen evaluation protocol
3. `ISSUES.md` §1 — why the prior results are void
4. `data/DATASETS.md`, `data/LABEL_MAPPING.md` — what data exists and what the labels mean
5. `EXPERIMENTS.md` — the run ledger
