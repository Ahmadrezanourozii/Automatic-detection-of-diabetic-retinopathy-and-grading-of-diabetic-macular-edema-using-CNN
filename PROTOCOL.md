# PROTOCOL.md — the evaluation protocol

## §0. Status: AGREED 2026-08-25 — frozen

Signed off by the owner on 2026-08-25 (sections 5.1, 6.2 and 6.3 decided; 6.1 settled by
the maintainer as noted). This file is now **frozen**:
changes require a new dated section at the bottom saying what changed and why, and every
result computed under the old protocol keeps its old label. Silently changing the protocol
to make a number look better is the specific failure this file exists to prevent.

---

## §1. Unit of independence

The unit is **the patient**, and where no patient identifier exists, **the eye**.
Both eyes of one person, and both fields of one eye, must never straddle a split.

What we actually have:

| Corpus | Patient IDs published? | Eye pairing recoverable? | Decision |
|---|---|---|---|
| IDRiD (516) | No | No — one image per eye, no pairing file | Treat each image as its own group, **after** a near-duplicate check |
| Messidor-2 (1744) | No | Upstream release pairs 2 images per examination; **our mirror's CSV does not carry the pairing** | Must recover the pairing, or cluster by image similarity. Blocking for any Messidor-2 result. |
| APTOS 2019 (3662) | No | Known to contain left/right pairs of the same patient | Cluster by image similarity |

**Action before the first split is written:** compute a perceptual hash (pHash) and a
vessel-pattern descriptor for every image in the pooled corpus; cluster; treat each cluster
as one group. This simultaneously (a) builds the grouping, (b) finds cross-dataset
duplicates — Messidor-1 and Messidor-2 overlap, and 687 of the 1744 files in our
"Messidor-2" mirror carry Messidor-**1** style filenames (`IM######.JPG`), so overlap is
not hypothetical. See `ISSUES.md` §3.

The resulting split assignment is written to `data/splits/` , committed, and loaded from
there by every model in the project. Once written it is frozen.

**Leakage quantification (do this once, keep the number):** run the same pipeline under a
group-wise split and under a naive random-image split, and report both. That single
comparison is expected to explain part of the gap between our numbers and the headline
numbers in the literature, and it belongs in the thesis.

---

## §2. What is held out

Two levels, because a single 15 % split of IDRiD cannot measure what we are hunting.

**Development pool** (training + model selection, via repeated stratified group 5-fold CV):
IDRiD 516 + Messidor-2 1 744 = **2 260 images**, with EyePACS (~35 k) used for pretraining
only and never for selection.

**External test sets, never seen until the final model is frozen:**
- **DR: APTOS 2019, held out in its entirety (3 662 images).** Already in hand, so it costs
  nothing to reserve; large enough for a ± 1.1 pt interval; and a genuine distribution shift
  (different population, different cameras) rather than a re-slice of the training corpus.
  *Caveat to state in the thesis:* APTOS labels are single-grader and noisier than IDRiD's
  or Messidor-2's adjudicated ones, so some of the internal-to-external drop will be label
  noise rather than generalisation failure. Do not attribute the whole drop to the model.
  DDR is a stretch goal for a second external DR set if time allows.
- **DME (3-class): Messidor-1** (1 200 images) — it carries a "risk of macular edema" label
  with *the same clinical definition* IDRiD uses (exudate distance to the macula centre, in
  disc diameters). The most credible single addition available to this thesis. Acquire
  early, and pHash it against our Messidor-2 mirror **before** using it (`ISSUES.md` §3).

If external validation proves impossible, that is reported as a limitation, not papered over.

---

## §3. Model selection

Configurations are ranked by **validation** performance, never by test. This includes the
choice *between* experiments, not only hyper-parameters within one. The winner of the
validation ranking is the number that gets reported.

The cost of getting this wrong is itself measured and reported once: rank the final
candidate set by test score and by validation score, and state the gap. That gap is pure
optimism and belongs in the thesis' limitations section.

---

## §4. Metrics

**Primary, DR (5-class ordinal, heavily imbalanced):**
- **Quadratic weighted kappa (QWK)** — the field standard for DR grading; it charges more
  for a two-grade error than a one-grade error, which is the exact failure mode we are
  trying to fix. This leads.
- Accuracy, reported alongside, never alone.
- **Per-class recall.** A model can gain accuracy while going blind to the proliferative
  class; this is how you notice.
- **Referable-DR (grade ≥ 2) sensitivity and specificity** — this is the actual screening
  decision the system supports, and it is what a clinician will ask about.

**Primary, DME (3-class ordinal):** QWK, accuracy, per-class recall, and
referable-DME (grade 2) sensitivity/specificity.

Every number carries a bootstrap interval. Because rows are correlated (two eyes per
patient, two fields per eye), **resample whole groups, not images**. Resampling images
treats near-duplicates as independent evidence and makes the interval roughly twice too
narrow.

To compare two models: **paired bootstrap over the same groups**, report the interval of
the difference. If it contains zero, the models are indistinguishable and are reported as
such. They are not ranked anyway.

### 4.1 No difference is attributed to representation until it survives matched calibration

**A difference between two models measured at their shipped decision thresholds is not
evidence about their representations.** Before any such difference is described as one model
learning better features than another, both must be recalibrated the same way — cut-points
cross-fitted under an identical objective — and the comparison repeated. Only what survives
that is a statement about representation; what does not survive is a statement about where
two arbitrary thresholds happened to fall.

This rule exists because it has already overturned two claims this project was about to make,
both at zero compute cost (`FINDINGS.md` F1 and F3):

* A Mild-recall collapse read as a capacity limit was calibration: recalibration moved it
  5.4 % → 77.8 %.
* An entire backbone comparison — including a "significant" external QWK advantage and a
  per-class pattern that looked like a real representational trade — dissolved at matched
  operating points. One per-class difference **reversed sign**; three changed significance;
  the aggregate advantage became indistinguishable.

**The corollary, which is the more general trap.** Both models shipped with the default
threshold of 0.5 on every ordinal cut. That default is arbitrary — it is what you get from
`sigmoid > 0.5`, not from anything about the grading scale — and it was silently determining
our per-class results and very nearly our architecture conclusion. **An untuned default is a
hyper-parameter that has been chosen, not one that has been avoided**, and any comparison that
holds it fixed across models is confounded by it.

---

## §5. The two decisions that must not be quietly revisited

### 5.1 What the DME branch is evaluated on — **ungated is primary**

The existing work gates the DME branch on DR ≥ 1. Proposal: **evaluate ungated (all images,
3-class where 0 = no DME) as the primary definition**, and report the gated number as
secondary, always labelled.

Three reasons:
1. **The gate discards nothing.** Verified on IDRiD: of 516 images, every one of the 168
   DR=0 images has DME grade 0. There are zero DME-positive cases with DR=0. The gate is
   therefore not protecting against anything — it only changes the denominator.
2. **A gated metric is conditional on the DR head being right**, so it is not comparable
   across models with different DR heads. Comparing a gated model to an ungated one
   compares two different quantities.
3. **The floors differ enormously**, and this is the crux:

   | Evaluation set | n | Majority-class floor |
   |---|---|---|
   | Gated, DR ≥ 1, IDRiD test | 69 | **69.6 %** |
   | Gated, DR ≥ 1, all IDRiD | 348 | **69.8 %** |
   | Ungated, all IDRiD | 516 | **47.1 %** |

   The existing thesis reports 87.6 % on the gated set. Against a 69.6 % floor, that claim
   is worth about 18 points, not 87. Reported ungated against a 47 % floor it would mean
   something quite different. **Any DME number in this project must be quoted next to its
   floor.**

### 5.2 Which test set the headline number comes from

The headline DR number comes from repeated group CV over the development pool, reported as
mean ± interval; the external-set number is reported separately and is the one that
survives a defence. Single-split numbers on IDRiD's 103-image test set are **not** headline
material — see §7.

---

## §6. Decisions — settled 2026-08-25

1. **External DR test corpus: APTOS 2019, held out entirely.** Settled by the maintainer as
   a routine call, not put to the owner: it is already in hand, it is the largest clean
   corpus available to reserve, and reserving it costs no acquisition work. Revisable if
   DDR or DeepDRiD is acquired later.
2. **Acquire Messidor-1 for external DME validation.** Agreed. Duplicate-check against the
   Messidor-2 mirror first.
3. **Messidor-2 stays in development**, supplying coarse (partial-label) DME supervision per
   `data/LABEL_MAPPING.md`. It is therefore *not* an external set and no number computed on
   it may ever be described as external validation.
4. **DME evaluated ungated as primary**, gated DR≥1 as secondary and always labelled (§5.1).
5. **Thesis chapter 4 will be rewritten from computed numbers**, generated by script from
   archived `results.json` files. No hand-typed numbers. (`ISSUES.md` §1.)

---

## §7. Resolution of the experiment — the number that governs everything

95 % confidence half-widths for a single accuracy estimate:

| Test set | n | acc ≈ 0.88 → ± |
|---|---|---|
| IDRiD official test, DR | 103 | **± 6.3 pts** |
| IDRiD test gated DR≥1, DME | 69 | **± 7.7 pts** |
| Pooled dev CV, one fold of 2 260 | 452 | ± 3.0 pts |
| Pooled dev, 5×5 repeated CV (all 2 260 predicted) | 2 260 | ± 1.3 pts |
| APTOS held out entirely | 3 662 | ± 1.1 pts |

**Consequence:** on IDRiD's official test split, a three- or four-point improvement is
unmeasurable no matter how many seeds are averaged. Any experiment whose expected effect is
smaller than ~6 points must be evaluated by **repeated cross-validation over the whole
2 260-image development pool** — where every image is predicted once per repeat, giving
± 1.3 pts — rather than on any single held-out slice. It is not worth the GPU time
otherwise. This is the constraint that should decide the experiment queue.

---

## §8. Invariants — asserted by script before every reported result

`src/check_invariants.py` (to be written) must pass:
1. No group appears in two splits.
2. No image hash appears twice across merged datasets.
3. Normalisation statistics were computed on training data only — verified by
   reconstructing one test image by hand from the training statistics and comparing it
   byte-for-byte against what the pipeline produced.
4. The same result is **not** reproducible from all-data statistics (i.e. the check in 3
   actually discriminates).
5. One image's preprocessed tensor, recomputed in isolation, matches the cached batch.
6. Class weights, CLAHE parameters and decision thresholds were fitted on train only.
