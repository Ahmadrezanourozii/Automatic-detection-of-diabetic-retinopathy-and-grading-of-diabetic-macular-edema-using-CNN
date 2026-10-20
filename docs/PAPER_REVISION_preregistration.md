# Revision analysis plan — committed 2026-10-20, before any of these analyses was run

This plan answers an internal pre-submission review of the manuscript. Like
`docs/PAPER_EXPERIMENTS_preregistration.md`, it is an analysis plan committed to version
control before the corresponding code was executed; it is not a registry entry. Every outcome
below is stated with the result that would falsify it, and falsified outcomes will be reported
with the same weight as confirmed ones.

Terminology change for the revision: the comparison rule previously called "matched
calibration" is renamed **matched decision rule** (both models' cut-points refitted by the same
cross-fitted procedure). Nothing about the computation changes.

## R-A. Mechanism of the population shift (no training)

* **A1 — conditional, not prior, shift.** Under pure label (prior) shift the distribution of a
  model's score given the true grade is the same in every population. For every pre-registered
  development model we compare the mean expected grade E[s | y = k] on APTOS, DDR, DeepDRiD and
  SUSTech-SYSU with its development out-of-fold value, grade by grade.
  Disclosure: the overall mean predicted-minus-true grade per corpus is already reported in the
  manuscript; the grade-by-grade breakdown has not been computed.
  *Prediction:* on APTOS E[s | y = k] exceeds the development value for at least 3 of the 5
  grades in at least 8 of the 10 models, and on DDR it falls below it for at least 3 of 5 grades
  in at least 8 of 10 models. *Falsified* if either fails (the shift would then be compatible with
  prior shift alone).
* **A2 — image degradation.** The development pool is re-scored out of fold (each image by the
  fold model that did not train on it; E08 and E11FULL) after five fixed degradations of the
  cached image: JPEG at quality 30 after resizing to 512 px; downsampling to 256 px; Gaussian blur
  (sigma 2 px); gamma 1.8 darkening; contrast reduced to 60 %. *Prediction:* at least one
  degradation lowers the mean predicted-minus-true grade by at least 0.10 relative to the
  undegraded re-scoring and lowers referable sensitivity at the shipped rule by at least 10
  points, i.e. image quality alone can produce DDR-like under-grading. *Falsified* if no
  degradation does both.
* **A3 — image statistics.** Per corpus (development IDRiD and Messidor-2, APTOS, DDR, DeepDRiD,
  SUSTech-SYSU; up to 1 000 images each): source resolution, mean brightness and RMS contrast
  inside the retinal mask, Laplacian variance (sharpness) at a common 512 px scale, and the
  estimated JPEG quality of the source file. Descriptive; no prediction.
* **A4 — grade distributions.** The DR grade distribution of each corpus. Descriptive.

## R-B. A label-free decision rule (no training)

* **B1 — prior adaptation by EM (Saerens et al., 2002).** Class probabilities are taken from the
  ordinal head (p_k = P(y > k-1) - P(y > k), clipped and renormalised); the source prior is the
  mean development out-of-fold class probability; the target prior is estimated by EM on the
  unlabelled external images; posteriors are reweighted and decoded by argmax (rule **R5**).
* **B2 — black-box shift estimation (Lipton et al., 2018).** The development out-of-fold
  confusion matrix of the shipped rule and the external predicted-grade distribution give the
  importance weights w = C^-1 mu (negative weights clipped at 0, renormalised); posteriors are
  reweighted as in B1 and decoded by argmax (rule **R5b**).
* *Prediction:* on DDR, R5 closes at least 25 % of the gap in referable sensitivity between the
  shipped rule R1 and the external-label rule R4 (gap closed = (sens_R5 - sens_R1) /
  (sens_R4 - sens_R1)) for at least 6 of the 10 models. *Falsified* otherwise. Reported for every
  external set and both rules; no external label is used to fit either rule.

## R-C. Recalibration sample size with the operational objective (no training)

The earlier analysis fitted cut-points to maximise macro-averaged recall. It is repeated with the
clinically primary objective: on n randomly drawn labelled external images (n = 25, 50, 100,
200, 400, 800; 500 draws each) the referral cut on P(grade >= 2) is set to the largest value that
gives at least 85 % sensitivity on those n images; sensitivity and specificity are measured on
the remaining images. *Harm* is a held-out sensitivity below 80 %. Reported for the selected
model and the reference model on APTOS, DDR, DeepDRiD and SUSTech-SYSU. *Prediction:* with
n = 200 the probability of harm is below 20 % on every external set. *Falsified* otherwise.

## R-D. Descriptive additions (no training, no predictions)

* Per-threshold areas under the ROC curve (P(y > 0), ..., P(y > 3)) and Harrell's C-index of
  the expected grade, development and external.
* Mild-grade precision, F1 and the full confusion matrix under every rule, for every model.
* Ranking of models by external QWK versus by referable sensitivity at a fixed specificity of
  90 % (Kendall tau).
* Positive and negative predictive values, and referable patients missed per 10 000 screened,
  at referable prevalences of 5 %, 10 % and 20 %.

## R-E. New data (GPU)

* **E1 — two further external sets** for every pre-registered development model (inference
  only, same recipe): DeepDRiD regular fundus images (2 000 images, 500 patients) and
  SUSTech-SYSU (1 151 images).
* **E2 — a model trained on 28 000 images.** DenseNet121 (account 1) and EfficientNet-B3
  (account 2) trained from ImageNet on EyePACS 2015 (patient-grouped 5-fold split, folds 1-4 for
  training, fold 0 held out for epoch selection and for fitting R2/R3), 448 px, 15 epochs,
  evaluated on APTOS, DDR, Messidor-2, IDRiD, DeepDRiD and SUSTech-SYSU.
  *Prediction:* on at least two of the six external sets the threshold fitted on fold 0 for 85 %
  sensitivity misses its target by more than 5 points, while the referral AUROC stays at or above
  0.90 — the decomposition seen with 2 260 training images is not an artefact of small
  development data. *Falsified* if no external set shows a transfer gap above 5 points.

## Consequences fixed in advance

* A1 falsified on a corpus: the paper says that shift is compatible with prior shift there and
  gives the label-free rule (R-B) more weight.
* A2 falsified: image quality is not offered as an explanation of under-grading; grading
  convention remains a candidate and recalibration on local labels is described as fitting the
  local grading standard.
* B falsified: reported as a negative result; local labels remain necessary.
* C falsified: the 200-image recommendation is withdrawn and replaced by what the data support.
* E2 falsified: the paper states that the decomposition was not reproduced with a large training
  set.

---

## Outcomes — recorded 2026-10-20, after every analysis and run above had finished

Appended after the fact; nothing above this line was edited. Numbers are generated by
`src/paper_revision.py`, `src/paper_mechanism.py` and `src/paper_eyepacs.py` into
`docs/generated/paper/` (`revision.json`, `mechanism.json`, `eyepacs.json`).

| outcome | verdict | evidence |
|---|---|---|
| A1 conditional shift | **held** | E[s \| y] above development for ≥ 3 of 5 grades in 10 of 10 models on APTOS, below in 10 of 10 on DDR; also measured: below in 10 of 10 on DeepDRiD, above in 10 of 10 on SUSTech-SYSU |
| A2 a degradation lowers mean by ≥ 0.10 and sensitivity by ≥ 10 points | **held** | E08: JPEG q30 −0.45 / −36.8, blur −0.30 / −15.4; E11FULL: JPEG q30 −0.53 / −45.6, blur −0.24 / −15.9 |
| A3 image statistics | descriptive | DDR is not degraded: 512 px, estimated JPEG quality 95, Laplacian variance 70 (Messidor-2 39) |
| B R5 closes ≥ 25 % of the DDR R1→R4 sensitivity gap in ≥ 6 of 10 models | **falsified** | 2 of 10 on DDR (APTOS 0, DeepDRiD 3, SUSTech-SYSU 0) |
| C harm at n = 200 below 20 % on every external set | **held** | 9.4–15.8 % across corpora and the two models |
| E1 two further external sets | done | DeepDRiD (2 000 images) and SUSTech-SYSU (1 151) scored by every development model |
| E2 transfer gap > 5 points on ≥ 2 of 6 sets with AUROC ≥ 0.90 | **held for both models** | DenseNet121 and EfficientNet-B3: gap > 5 points on 6 of 6 sets; AUROC 0.938–0.999; fold-0 threshold reached 90.8–100 % |

Consequences applied as fixed in advance: with B falsified the label-free rule is reported as a
negative result and local labels remain necessary. With A2 held but A3 showing that DDR images
are not degraded, image quality is reported as able to produce under-grading in principle but is
not offered as the explanation for DDR; a difference in grading convention remains a candidate.

Observation, labelled as such in the paper: the direction of the shift reproduced with EyePACS
training (APTOS +0.20 and +0.14, DDR −0.18 and −0.23 mean predicted-minus-true grade), whereas
the sign of the DDR threshold gap did not (63.4 % from the development pool, 90.9 % and 90.8 %
from EyePACS).
