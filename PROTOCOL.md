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

### 4.2 Every per-class claim carries the same two qualifications

Per-class recall is reported beside every headline (§4.1 and `FINDINGS.md` F1 exist because it
was not). Two qualifications travel with **every** such claim in this project, not only the
ones where they happen to be inconvenient:

1. **No correction for multiple comparisons is applied.** Five classes times the conditions
   compared means roughly one significant cell is expected by chance in a typical table.
   **Credibility therefore comes from a difference replicating across conditions, not from its
   p-value in isolation.** The Moderate source effect is credible because it held at −6.8
   before recalibration and −6.5 after; a cell significant in one condition only is a lead,
   not a finding.
2. **Where the comparison varies more than one thing, the per-class claim inherits every
   confound the aggregate does.** The source control established that two image sources give
   models that are equal in aggregate but distribute per-class errors differently by several
   points. So a per-class result from any experiment that also changes the image source —
   the native-resolution test above all — is **not cleanly attributable**, even where the
   aggregate is.

**A difference between two models measured at their shipped decision thresholds is not
evidence about their representations.** Before any such difference is described as one model
learning better features than another, both must be recalibrated the same way — cut-points
cross-fitted under an identical objective — and the comparison repeated. Only what survives
that is a statement about representation; what does not survive is a statement about where
two arbitrary thresholds happened to fall.

**The rule has now overturned claims in both directions, and that is what makes it evidence
rather than a filter.** A test that only ever dissolved inconvenient differences would be a
way of explaining away results, and a reader would be right to distrust every use of it. Its
record so far:

| claim | initial reading | what matched calibration showed |
|---|---|---|
| Mild recall collapse (F1) | a capacity limit | **calibration** — 5.4 % → 77.8 % by moving cut-points |
| EfficientNet beats DenseNet (F3) | a better representation | **boundary placement** — the advantage vanished, one cell reversed sign |
| Moderate differs by source (E12) | another boundary artefact | **a real difference** — −6.8 → −6.5, essentially unchanged |
| E08 beats LP-FT on DR accuracy (E15) | LP-FT costs accuracy | **boundary placement, and it REVERSES** — +5.57 pts [+3.54, +7.52] for E08 at shipped cuts becomes **−1.92 pts [−3.67, −0.13] for LP-FT** at matched cuts |

Four predictions, four overturned, in **two different directions**, with **two outright sign
reversals**. The Moderate row is the one that licenses the others: without a case where the
check came back positive, the rule would be indistinguishable from motivated reasoning.

The E15 row is the starkest instance yet of the corollary below. LP-FT's *representation* was
never worse — its default cut-points were simply badly placed, costing it 5.4 points of
accuracy that recalibration handed straight back. Reported at shipped thresholds, the same
experiment would have produced a confident and completely wrong conclusion about what LP-FT
does to a network.

This rule exists because it has already overturned claims this project was about to make,
at zero compute cost (`FINDINGS.md` F1 and F3, and the E12 source control):

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

   A gated number of 87.6 % against a 69.6 % floor is worth about 18 points, not 87; the same
   number reported ungated against a 47 % floor would mean something quite different. **Any
   DME number in this project must be quoted next to its floor.** (This paragraph originally
   cited 87.6 % as a claim from the earlier draft. It was a template placeholder, not a claim
   — corrected 2026-08-31, `ISSUES.md` §1. The arithmetic and the rule are unaffected.)

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

## §10. A check that only ever fires one way is not a check

**Added 2026-09-01. This is the criterion by which every other rule in this file earns its
standing, and it belongs in the methods chapter as such.**

**The rule.** A methodological check is only evidence if it is *capable of returning either
answer*, and it is only *credible* once it has been observed to return both. A rule that has
only ever dissolved inconvenient results is indistinguishable from a way of explaining them
away; a rule that has only ever confirmed them is indistinguishable from a rubber stamp.
Either way the reader is right to distrust every use of it, because nothing in the record
separates the rule from the conclusion it happens to support.

**So each check in this protocol carries its record, and the record must contain
disconfirmations of the maintainer's own position in both directions.** Two rules currently
meet that bar.

**§4.1 — matched calibration before attribution.** Four claims overturned, in two directions,
with two outright sign reversals:

| claim | initial reading | what matched calibration showed |
|---|---|---|
| Mild recall collapse (F1) | a capacity limit | **calibration** — 5.4 % → 77.8 % |
| EfficientNet beats DenseNet (F3) | a better representation | **boundary placement** — advantage vanished |
| Moderate differs by source (E12) | another boundary artefact | **a real difference** — −6.8 → −6.5 |
| E08 beats LP-FT on accuracy (E15) | LP-FT costs accuracy | **reversed** — +5.57 pts became −1.92 pts *for* LP-FT |

**§9 — consumption manifests.** Three findings, in both directions, on the day the rule
landed:

| case | what the manifest did |
|---|---|
| E11 vs E19E11B (§26) | **refused** a combination that would have corrupted the best DR result |
| I20 per-class caveat | **overturned a caveat in the optimistic direction** — E10 and E17NAT read the same 1 744 files, so I20 changed one variable, not two, and the caveat was withdrawn |
| the OOF ensemble | **corrected the maintainer's own reported number downward**, 0.8933 → 0.8828, by finding that three of four ensembles averaged over different renderings of the same eyes |

**What this licenses, and what it does not.** A check with a two-directional record may be
cited as evidence about the thing it checks. A check without one may only be described as a
filter that has not yet been tested. **No result in this project may lean on a check that has
never been observed to contradict the maintainer.**

**The failure mode this prevents.** Every rule here was written by someone with a stake in
the outcome. The defence against that is not good intentions; it is a record showing the rule
has cost its author something. §4.1 cost an architecture conclusion the project was about to
publish. §9 cost a headline number, twenty-four hours after it was reported.

---

## §9. Configuration is not consumption — provenance must record what a run READ

**Added 2026-09-01 after `ISSUES.md` §26. This is a new dated section; nothing above is
changed by it.**

**The rule.** Every run must record, at runtime and from inside the process that walked the
disk, a **consumption manifest**: for each corpus, how many images were resolved and a hash
over the resolved paths relative to the dataset mount. It is written to `results.json` as
`consumption`. **Two runs may not be combined, ensembled, or compared unless their
consumption manifests agree.** `src/manifest.py` enforces it; `compare_matched.py` and
`ensemble_oof.py` refuse without it.

**Why a new rule was needed, rather than a fix to an old one.** Two runs launched to complete
each other's folds passed **four** independent provenance mechanisms while having trained on
different images:

| mechanism | why it passed |
|---|---|
| identical `config` blocks | the image source is not a training argument — it is which datasets the kernel mounted |
| matching `split_fingerprint` | the split is over uids and does not know which file backs a uid |
| the §24 generated-notebook guard | the correct script did run |
| `messidor_source: prefer-native` in both | **literally true of both, and meaning different things** — one kernel had the native mirror mounted, the other did not |

Every one of those records what a run was **configured** to do. None records what it
**consumed**. A field that is true of two runs while meaning different things is not
provenance.

**The deliberate-difference case.** Some comparisons are meant to cross sources — I20
compared two runs precisely to measure a change of inputs. That is legitimate, so an override
exists (`--acknowledge-consumption-diff`), but it **does not silence the difference: it stamps
it into the output document**. A caveat the reader cannot see is not a caveat.

**Retrofit, and its declared limit.** Runs predating this rule get a manifest reconstructed
from their kernel's `dataset_sources`. That is **coarse**: it catches a difference in which
datasets were mounted — exactly the §26 axis — and would miss a difference in which files
were read from the same mounted dataset. Retrofitted comparisons say so in their output.
Datasets that cannot supply a development-pool image (APTOS, held out entirely under §2 and
§6.1) are excluded from the retrofit comparison, because mounting one cannot change what a
run trained on.

**It has already worked in both directions.** It refuses the §26 pairing that would have
corrupted the project's best DR result. It also **overturned a caveat that was too
conservative**: I20's per-class results were reported as confounded by a source change, and
the manifests showed E10 and E17NAT read the same 1 744 files with identical per-grade counts,
differing only in `cache_size`. A provenance check that only ever added doubt would be no more
trustworthy than one that only ever removed it.

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
