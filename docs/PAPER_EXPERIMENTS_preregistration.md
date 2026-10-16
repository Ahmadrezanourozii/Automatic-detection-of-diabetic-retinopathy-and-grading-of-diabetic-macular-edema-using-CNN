# Journal-paper experiments — pre-registration

**Committed 2026-10-16, before any of the runs below was launched and before any number they
produce was seen.** Same discipline as `docs/PARTA_preregistration.md` and every `IDEAS.md`
entry: the hypothesis, the control and the falsifying outcome are fixed here, and the
results document that follows quotes this file rather than restating it.

## Why these runs, and why now

The modelling phase was closed on 2026-09-08 for the thesis. The journal submission
(target: *npj Digital Medicine*) raises a different bar, and three reviewer objections can be
predicted from the current evidence:

1. **One external corpus, one direction.** Every external claim (F1, F3, T1) rests on APTOS,
   with the development pool → APTOS direction only. A reader can say the Mild collapse and
   the over-grading are an APTOS labelling quirk.
2. **One seed per configuration.** Every architecture delta in the archive is a single
   training run. A reader can say the deltas — and the per-class "reversals" under matched
   calibration — are within seed noise.
3. **The published failures the paper argues against were measured APTOS → Messidor-2.**
   Our own external direction is the opposite one, so the comparison with Poyrazer et al.
   (grade-1 F1 = 0.000 on Messidor-2) is indirect.

None of these needs a new architecture. Each needs either more seeds of a configuration we
already have, the same models scored on a second external corpus, or the same recipe run in
the other direction. The owner provided two Kaggle accounts (~60 GPU-hours) for exactly this.

## Standing rules that apply

* `PROTOCOL.md` §3: nothing here changes which model is the selected one (E11FULL). Seeds
  of E08 are a **noise measurement**, not candidates.
* `PROTOCOL.md` §4.1: per-class and architecture claims are made at matched calibration.
* `PROTOCOL.md` §9: every run writes a runtime consumption manifest; seeds must consume the
  same development pool as E08 (same mounted development datasets, no native Messidor-2
  mirror), which the manifest checks.
* DDR and APTOS are **evaluation only**. DDR is added to `manifest.EVAL_ONLY_DATASETS`.
* In the reverse-direction run APTOS is the **training** corpus. That run is a separate
  experiment; nothing trained on APTOS is ever scored on APTOS as "external", and no
  development-pool model is affected.

## Decision rules used throughout (fixed now)

For every model with out-of-fold (OOF) development predictions and external predictions:

| rule | cut-points | fitted on | role |
|---|---|---|---|
| **R1** | shipped, `sigmoid > 0.5` on every ordinal threshold | nothing | the default every published comparison uses |
| **R2** | 4 cut-points on the expected grade, maximising QWK | development OOF (all folds pooled) | a deployable, validation-tuned rule |
| **R3** | as R2, then the referral cut moved to the lowest value giving development referable sensitivity ≥ 0.90 | development OOF | a deployable, sensitivity-constrained rule |
| **R4** | 4 cut-points maximising QWK, 2-fold cross-fitted **on the external set itself** | external labels | **oracle / diagnostic only** — an upper bound on what recalibration with local labels recovers; never a result |
| **R4m** | as R4, maximising macro-recall | external labels | diagnostic only (F1's recovery analysis) |

Metrics at every rule: QWK, accuracy, macro-F1, per-class recall, referable (grade ≥ 2)
sensitivity and specificity, Cohen's unweighted κ. Threshold-free alongside: referable AUROC
on P(grade ≥ 2). Calibration of P(grade ≥ 2): ECE (10 equal-width bins) **always beside**
Brier score and AUROC (Poyrazer et al.'s reporting standard).

---

## G1 — Seed replicates of E08 (account 1: S1E08, S2E08; account 2: S3E08, S4E08)

**Configuration.** E08 exactly: densenet121, ordinal heads, 448 px, cache 560, batch 16,
40 epochs, EyePACS pretraining 5 epochs, TTA, folds 0–4, frozen split `0cfbbfeb081999af`,
the 512 px Messidor-2 mirror (no `--messidor-hi`). **Only `--seed` differs: 1, 2, 3, 4.**
E08 is seed 0. External evaluation on APTOS and DDR after training.

**What it measures.** (a) The seed noise floor of every quantity the paper reports;
(b) whether matched-calibration verdicts are stable across seeds; (c) whether per-class
"differences" between models appear even between models that differ only by seed.

**Pre-registered outcomes.**

* **O1 — noise floor (descriptive, used as a yardstick).** Standard deviation across the 5
  seeds of: development DR QWK at matched (cross-fitted) cuts; APTOS DR QWK under R1; DDR DR
  QWK under R1. An architecture delta is called "beyond seed noise" only if it exceeds
  `2·√2·SD_seed`.
* **O2 — spurious per-class significance at shipped cuts.** Over the 10 seed pairs × 5
  classes = 50 paired-bootstrap per-class recall comparisons on APTOS under R1, **we predict
  more than 5 (10 %) are "significant" at 95 %**, i.e. at least double the 2.5 expected by
  chance — models identical in everything but the seed produce the kind of per-class
  "effects" that the literature attributes to architecture. **Falsified if ≤ 5 of 50.**
* **O3 — the Mild collapse replicates across seeds.** For **every** seed, APTOS Mild recall
  under R1 is < 15 %, and under R4m (external cross-fitted, macro-recall) it is > 50 %.
  **Falsified if any seed violates either bound.**

## G2 — Reverse direction: APTOS → Messidor-2 and IDRiD (account 2: R1APTOS)

**Configuration.** E08's recipe (densenet121, 448 px, 40 epochs, EyePACS pretraining, TTA,
seed 0), trained with `--corpora APTOS` on a 5-fold split of all 3 662 APTOS images built
inside the kernel by `src/splits.py --corpora APTOS --dedup-hamming 16 --seed 0` before
training (near-duplicates merged into one group). External: Messidor-2 (1 744, 512 px mirror)
and IDRiD (516), DR only. This is the direction in which Poyrazer et al. and ORDER-DR report
their external failures.

**Pre-registered outcomes.**

* **O4 — the published collapse appears with a fine-tuned CNN.** Under R1, Messidor-2 Mild
  (grade 1, n = 270) recall < 20 %. **Falsified if ≥ 20 %.**
* **O5 — it is recoverable by the decision rule alone.** Under R4m, Messidor-2 Mild recall at
  least 3× its R1 value and ≥ 40 % absolute, with no retraining. **Falsified if either fails.**
* **O6 — the miscalibration reverses direction with the direction of transfer.** Forward
  (development pool → APTOS) the model over-grades. Reverse, we predict it **under-grades**:
  on Messidor-2 under R1 the mean predicted grade is below the mean true grade, and the
  dominant Mild error is Mild → No DR rather than Mild → Moderate. **Falsified if it
  over-grades.**

## G3 — DDR as a second external set for the archived models (account 1: X2DDR)

**Models.** Every archived 5-fold development model with weights: E08, E09, E10, E11FULL
(E11 + E19E11C), E14MAC, E15LPFT, E17NAT, E20CORAL, E22CORN; and the 2-fold E21CNXA.
Inference only (5-fold logit average, flip TTA), same recipe as every APTOS number.

**Pre-registered outcomes.**

* **O7 — decision rule vs model (the paper's Phase 0 gate).** On each external set (APTOS,
  DDR), for Mild recall and for referable sensitivity, the **median within-model range across
  the deployable rules R1–R3 exceeds the range across models under R1.** **Falsified on a
  corpus × metric if the between-model range is larger.** (If falsified on both corpora for
  both metrics, the paper's central claim is weakened and is reported as such.)
* **O8 — the operating point does not transport to DDR either.** The referral threshold
  fitted on E08's development OOF for 85 % sensitivity (T1 candidate B) lands more than 5
  points away from 85 % on DDR. **Falsified if within ±5 points** — in which case APTOS, not
  DDR, is the exception, and the paper says so.

## G4 — RETFound folds 2–4 (account 2: I24FT234)

**Configuration.** I24FT01 exactly (RETFound CFP ViT-L/16, hash-verified, 224 px, LP-FT:
5 probe epochs then backbone at 1e-5 with 2 warm-up epochs, 30 epochs, TTA), folds 2, 3, 4.
I24FT01 + I24FT234 = I24FULL, assembled only if their consumption manifests agree.
The previous estimate of ~14 h was wrong: I24FT01 took 2.0 h for two folds on a P100.

**Pre-registered outcome.**

* **O9 — F8 at five folds.** Matched-calibration development DR QWK of I24FULL minus E09
  (densenet121 @224, EyePACS-pretrained) has a 95 % interval excluding zero on the negative
  side, as it did on folds 0–1 (−0.0399 [−0.0669, −0.0124]). **Falsified if the interval
  contains zero or the difference is positive.**

---

## Budget

| run | account | est. GPU-h |
|---|---|---|
| S1E08, S2E08 | 1 | 2 × ~4.3 |
| S3E08, S4E08 | 2 | 2 × ~4.3 |
| R1APTOS | 2 | ~7 |
| X2DDR | 1 | ~4–6 |
| I24FT234 (+ its external evals) | 2 | ~3.5 |
| **total** | | **~32–35 of ~60** |

The remainder is held back for failures and reruns, not pre-spent.

## What would change the paper

* O2 falsified → the "per-class reversals" argument rests on calibration differences between
  *architectures* only; the seed-noise framing is dropped.
* O3/O4/O5 falsified → the Mild collapse is corpus-specific; the paper narrows the claim.
* O7 falsified on both corpora → the central claim is reported as not supported at the
  level of model ranking, and the paper is reframed around the per-class and operating-point
  results that survive.
* O9 falsified → F8 is reported as a two-fold result that did not replicate at five folds.

---

## Outcomes — recorded 2026-10-16, after every run above had finished

Appended after the fact; nothing above this line was edited. Numbers are generated by
`src/paper_seeds.py`, `src/paper_reverse.py`, `src/paper_phase0.py`, `src/paper_transfer.py` and
`src/paper_matched.py` into `docs/generated/paper/`.

| outcome | verdict | evidence |
|---|---|---|
| O1 noise floor | measured | matched dev QWK across 5 seeds 0.8646–0.8764, SD 0.0043; APTOS R1 QWK SD 0.0026 |
| O2 >5 of 50 seed-pair per-grade differences "significant" on APTOS (R1) | **held** | 15 of 50 (29 of 50 under R4); DDR 17 of 30 |
| O3 every seed APTOS mild R1 < 15 % and R4m > 50 % | **held** | R1 5.4–6.2 %, R4m 63.8–72.7 % |
| O4 reverse: Messidor-2 mild R1 < 20 % | **falsified** | 34.4 % |
| O5 reverse: R4m ≥ 3× R1 and ≥ 40 % | **falsified** | 44.4 % (1.3×) |
| O6 reverse under-grades | **held** | mean pred − true −0.26; mild → No DR 175, → Moderate 2 |
| O7 within-rule range > between-model range | **falsified on both corpora, both metrics** | APTOS mild 5.7 vs 12.7, sens 0.9 vs 1.3; DDR mild 21.0 vs 24.8, sens 7.1 vs 11.4 (10-model set as pre-registered) |
| O8 DDR transfer of the 85 % threshold off by > 5 points | **held** | 63.4 % (gap −21.6); APTOS 99.3 % (+14.3) |
| O9 5-fold RETFound − E09 interval < 0 | **held** | −0.0414 [−0.0583, −0.0238] |

Consequences applied as pre-registered: with O4/O5 falsified the mild-grade collapse is reported
as corpus-specific; with O7 falsified on both corpora the claim that the decision rule outweighs
the model is reported as not supported, and the paper is reframed around the results that
survive (transfer of ranking but not of the score-to-grade mapping, rule-recoverable failures,
threshold transfer, metric blindness, seed noise, matched calibration).

Unplanned observations, labelled as such in the paper: the development models under-grade DDR
(mean −0.17 to −0.35) as they over-grade APTOS; the reverse model refers 43.5 % of referable
Messidor-2 eyes at R1 and 77.9 % at R4; RETFound's APTOS deficit is mostly rule (QWK 0.703 at R1,
0.834 at R4) while its DDR deficit is representational (AUROC 0.855).
