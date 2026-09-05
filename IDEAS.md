# IDEAS.md — backlog with verdicts

Status ∈ {untested, queued, running, confirmed, rejected, inconclusive}.
An idea that is neither converted into an experiment nor explicitly rejected is not finished.
Rejected ideas keep the evidence that rejected them.

Re-ranked 2026-08-25 against what the audit found. The ranking changed a lot from the
backlog in the project prompt, for one reason: **there is no working baseline**, so ideas
that improve a working model are worth nothing until one exists.

---

## Implemented and in flight

| ID | Idea | Status | Note |
|---|---|---|---|
| I02 | **Ordinal threshold heads** for both targets (P(y>k) rather than K-way softmax) | **implemented**, running in E05/E06 | Puts the grade ordering into the loss, which is what QWK measures; makes decoding rank-consistent by construction; and — the reason it is Tier 0 rather than Tier 1 — it is what makes Messidor-2's coarse DME label usable. `src/model.py`. |
| I04a | **Messidor-2 partial-label DME supervision** | **implemented**, running in E05/E06 | "Referable" is exactly IDRiD grade 2, so in threshold form it supervises P(y>1) exactly and leaves P(y>0) masked. DME supervision 348 → 2 260 images with no approximation. |
| I04b | **EyePACS pretraining** (35 126 images, 17 563 patients) | **running in E06** | Done once, reused by all folds; refuses to run if the corpus matches zero images (`ISSUES.md` §10). |
| I03 | Augmentation: free rotation, flips, mild colour/gamma jitter | **implemented** | Rotation is unrestricted because a retina has no canonical up and every grading criterion is rotation-invariant. |
| I13 | Test-time augmentation over the four dihedral flips | **implemented** | `--tta`. Fold ensembling by logit averaging is in `src/eval_external.py`. |
| I01 | Group-wise frozen splits, stratified by (corpus, DR) | **done** | `data/splits/dev_v1.json`, fingerprint `0cfbbfeb081999af`. Training refuses to start on an image missing from it. |
| I12 | External validation | **infrastructure ready** | `src/eval_external.py`, APTOS held out. Blocked for DME until Messidor-1 is acquired. |

## Tier 0 — must happen before anything else

| ID | Idea | Status | Cheapest falsifying test | Prediction |
|---|---|---|---|---|
| E01 | The representation carries DR signal; the prior collapse was an optimisation failure. | **queued** | Frozen DenseNet121/ImageNet features at 224 px on a 2 000-image subset → logistic regression. CPU, minutes. | QWK ≫ 0 and accuracy well above the 33 % floor. If not, the problem is preprocessing or resolution, not the training loop, and the whole plan changes. |
| I00 | Trivial baselines, computed and recorded once. | **queued** | Majority class per head; logistic regression on frozen features; plain RGB DenseNet121 with no preprocessing chain. | Already partly known: floors are 33.0 % (DR, IDRiD test), 69.6 % (DME gated), 47.1 % (DME ungated). |
| I01b | Quantify the cost of a naive image-wise split | **queued** | Re-run one fold with a random image split and report the difference. Cheap, and the number belongs in the thesis. | A real gap, smaller than the 25 points seen in the prior Parkinson's project but non-zero. |

## Tier 1 — largest expected gain per GPU-hour

| ID | Idea | Status | Notes |
|---|---|---|---|
| I07 | **Macula-centred crop / macula-pooled DME head** | **PRIORITY 1 — unblocked at 2 260 images by E13gate, 2026-08-30** | The DME grade *is* defined by exudate distance to the macula centre, and we have that centre for all 516 IDRiD images. GAP over a 512×512 whole-fundus feature map discards exactly the positional information the label depends on. Strongest architectural idea available, and it is clinically motivated rather than a hyper-parameter. **See the design block below — it carries three conditions that are part of the experiment, not commentary.** |
| I03 | Augmentation: rotation, flips, scale, brightness/contrast jitter, mixup/CutMix. | untested | Cheapest known lever given the DME data volume. Note the prior notebook explicitly turned augmentation **off** to help memorisation. |
| I06 | **The green-channel question.** green×3 vs [green, CLAHE(green), grey] vs full RGB vs RGB with CLAHE on LAB's L channel. | untested | The thesis' 10-point ablation claim for the green channel has no run behind it (`ISSUES.md` §1). This tests a load-bearing assumption. Feeding one replicated channel into a 3-channel ImageNet-pretrained network discards two thirds of the pretrained first-layer filters. |

### I07 — **RUN AS E14MAC, 2026-08-31. FALSIFIED. See `FINDINGS.md` F7.**

DME QWK at matched calibration 0.8764 (E08) vs 0.8538 (E14MAC), difference +0.0237
[−0.0094, +0.0566] — indistinguishable, crop nominally worse, inside the ±0.03 band named in
advance. **The DME ceiling is data, not architecture.** The null is not a localiser artefact:
the window is 3 DD wide and the localiser's 99th-percentile error is 0.857 DD, so the fovea is
inside the crop in over 99 % of images. Condition 2 discharged — the 3-class DME evaluation set
*is* IDRiD's 516, so the headline carries no unvalidated fovea transfer at evaluation time.

**Consequence for the queue: further architectural work on the DME head is not the lever.**
I15 (auxiliary hard-exudate segmentation from IDRiD's 81 masks) is promoted — it is the one
remaining DME idea that adds information rather than rearranging what is already there.

### I07 design — fixed before the run (2026-08-31)

**Why it is first.** Re-ranked ahead of LP-FT on evidence, not preference. Both items changed
status on the same day and in opposite directions: E13gate widened I07 from 516 to 2 260
images, while D01 withdrew LP-FT's floor-lift argument (see I21). I07 is also the only queued
idea that addresses the DME head, which carries **the project's standing negative result** —
no intervention has ever significantly improved 3-class DME on QWK.

**Hypothesis.** Global average pooling over a 448 px feature map dilutes the decisive region
by roughly 16×: one disc diameter is ~55 px, so the decision region is a ~110 px disc in a
448 px image. Pooling the DME head's features at the fovea should move DME QWK.

**Falsifying outcome, stated before the run.** *If DME QWK moves less than its ±0.03
interval, the DME ceiling is data, not architecture.* Given the standing negative result on
this head — pretraining, schedule, backbone, and architecture-plus-data have all failed to
move DME QWK, with effective resolution moving only accuracy — **this is a real possibility
and is named here in advance rather than discovered afterwards.** A null result is a
publishable finding about where the ceiling lives, not a failed experiment.

**Condition 1 — the transfer caveat travels with every number.** The localiser was validated
on IDRiD only (median 0.196 DD, 90th 0.433 DD, out-of-fold on 516). Messidor-2 has **no fovea
ground truth**, so applying it there is an assumption that cannot be checked by any experiment
available to this project. A gain measured at 2 260 images therefore rests partly on an
unvalidated assumption and **must not be reported as though it did not**.

**Condition 2 — report the IDRiD-only result alongside the pooled one.** Every I07 headline
carries two numbers: the 2 260-image result, and the result on IDRiD's 516 where every fovea
coordinate is ground truth. The second is what survives if the transfer assumption fails, and
a reader must be able to see it without asking. If the two disagree, that disagreement is
itself evidence about the transfer and is reported as such.

**Condition 3 — §4.1 still binds.** Any per-class DME difference must survive matched
calibration before it is described as the crop learning better features. This head is exactly
where cut-point placement has been most misleading (F1, F3).

### T1 — the reported result is the TRANSFER GAP, not a threshold (fixed 2026-08-31)

`sigmoid > 0.5` gives **86.28 % sensitivity / 96.42 % specificity** on the development pool
and **99.53 % / 84.28 %** on APTOS — same model, same cut-point, opposite ends of the curve.
So a single achieved sensitivity is not a reportable result.

**T1's deliverable is therefore: fit a cut-point cross-fitted on the development pool to hit
the chosen target sensitivity, then report how far it lands from that target on APTOS.** That
distance, with its interval and its sign, is the finding. **Falsified if** no threshold
transfers within a useful band — in which case the recommendation becomes "recalibrate
locally", with F4's 200-image figure attached.

Candidate targets with verified provenance and specificity costs are in
`docs/T1_referral_threshold_candidates.md`. **The target itself is still unchosen — an open
question for the owner, not a blocker**, since T1 sits behind I07 and I21 in the queue.

### I21 — LP-FT — **RUN AS E15LPFT, 2026-08-31. FALSIFIED on its own pre-registered criterion.**

**Verdict.** The falsifying outcome was fixed before the run: *if it does not beat current at
matched calibration, it is not the ceiling and we stop pursuing it.* It does not.
**E15LPFT vs E08, paired bootstrap over groups, both recalibrated identically
(cut-points cross-fitted on the other folds), `docs/generated/matched_comparison.md`:**

| head | cut-points | metric | E08 | E15LPFT | E08 − LPFT | 95 % interval | verdict |
|---|---|---|---|---|---|---|---|
| DR | shipped | QWK | 0.8599 | 0.8466 | +0.0136 | [−0.0004, +0.0275] | indistinguishable |
| DR | shipped | accuracy | 74.20 % | 68.67 % | +5.57 pts | [+3.54, +7.52] | **significant** |
| **DR** | **matched** | **QWK** | **0.8646** | **0.8655** | **−0.0008** | **[−0.0126, +0.0105]** | **indistinguishable** |
| DR | matched | accuracy | 72.12 % | 74.07 % | −1.92 pts | [−3.67, −0.13] | **significant, for LP-FT** |
| DME | shipped | QWK | 0.8845 | 0.8551 | +0.0303 | [−0.0017, +0.0618] | indistinguishable |
| DME | matched | QWK | 0.8764 | 0.8577 | +0.0193 | [−0.0099, +0.0481] | indistinguishable |

**On the primary metric it is a dead heat** — DR QWK differs by 0.0008, an interval tight
around zero. DME is indistinguishable at both operating points. **LP-FT does not beat current
full fine-tuning, so by the stated criterion we stop pursuing it.** Cost: 3.4 h.

**But the run earned its keep as a §4.1 case, and this is the part worth keeping.** At shipped
cut-points E08 looked **significantly better on DR accuracy, +5.57 points**. At matched
cut-points that **reverses**: −1.92 points, significant, *in LP-FT's favour*. LP-FT's
representation was never worse. Its default `sigmoid > 0.5` cut-points were simply badly
placed, costing it 5.4 points of accuracy that recalibration handed straight back
(68.67 % → 74.07 %, the largest recalibration gain any run in this project has shown, versus
E08's 74.20 % → 72.12 % *loss*). Reported at shipped thresholds — as almost every paper does —
this experiment would have concluded confidently and wrongly that LP-FT damages a network.

**Recorded as the fourth row of `PROTOCOL.md` §4.1** and its second outright sign reversal.

**Status.** Closed. Do not revisit without a new reason that is not "maybe with different
hyper-parameters".

---

### I21 — original design, kept for the record — hypothesis narrowed by D01

**Status.** Queued, after I07. Independent question.

**What it still claims (part one — stands).** `src/train.py` has **no linear-probe phase**:
the backbone trains from step 0 at `lr_backbone=1e-4` with 5 % warmup and cosine decay
(`src/train.py:360-363`). E01 established that frozen ImageNet features already beat the
majority floor (47.6 %). LP-FT is the standard method for keeping fine-tuning from distorting
a representation that is already linearly separable, and distortion would show up as exactly
the calibration shift F3 documented twice. That is a coherent reason to try it.

**What it no longer claims (part two — WITHDRAWN on evidence, 2026-08-31).** The original
argument was that the void baseline's 27.2 % collapse, against E01's 47.6 % on the same
features, proved fine-tuning was destroying the representation — so **every** current model,
including the ones giving 74 % and 86 %, was trained through the same distortion and sits
below its real ceiling. LP-FT would then lift the floor under every downstream number.

**D01 killed this, and then went further.** The archived recipe, run on the original Keras
code with regularisation and augmentation off, **memorises 20 images: DR 100 %, DME 100 %,
final loss 0.0005.** Fine-tuning at 1e-5 did not destroy the frozen-phase representation, it
completed it (95 % → 100 %). The archived collapse showed the opposite signature — validation
loss rising 1.86 → 3.40.

**All three named suspects are now eliminated** (`ISSUES.md` §1, D01): fine-tuning distortion;
the class-weight broadcast (reproduced through the archived `tf.data` path with genuinely
non-uniform weights — still memorises); and the unmasked DME loss, which is **vacuous** —
`_zero_dme_for_healthy` changes 0 of 516 IDRiD labels, because all 168 DR=0 images already
have DME 0.

So the collapse tells us **nothing** about fine-tuning, in either direction. It is not
evidence for LP-FT and not evidence against it. LP-FT has to stand or fall on its own
measurement.

**Revised expected value: uncertain, not likely.** "LP-FT beats current full fine-tuning at
matched calibration" is now a genuinely open question rather than a favoured prediction. The
falsifying outcome is unchanged and still binds: **if it does not beat current at matched
calibration, it is not the ceiling and we stop pursuing it.**

**Why this is worth recording as a pattern.** A hypothesis was narrowed by a free CPU
diagnostic before it burned a GPU run. The floor-lift argument was the reason LP-FT was
priority 1; it was withdrawn on evidence that cost nothing, and the queue was reordered as a
result. This is the same shape as §4.1 — a cheap check overturning a claim the project was
about to act on.

## Tier 2

| ID | Idea | Status | Notes |
|---|---|---|---|
| I05 | Backbone comparison: EfficientNet, ConvNeXt, Swin, retinal foundation model (RETFound). | untested | Do not start before E01. |
| I08 | Loss weighting: α/β sweep, then uncertainty-based task weighting. | untested | Low expected value until the heads work at all. |
| I09 | Hard vs soft gating of the DME branch; cost of DR-head errors propagating. | untested | See `PROTOCOL.md` §5.1 — the gate discards nothing, so this is about metric definition more than architecture. |
| I10c | Test the resolution hypothesis before acquiring anything | **CONFIRMED (E09)** — 448 beats 224 by +0.040 QWK, and Mild recall by +18.6 points, while DME is unaffected. Resolution binds for DR only. | Train the E06 config at 224 px and compare Mild recall against 448 px. If halving the resolution does not hurt Mild, resolution is not the binding constraint and I10b is not worth doing. This is the cheap falsifying test for a data-acquisition decision, and it costs one short run instead of a partial re-download. |
| I10b | Acquire a full-resolution Messidor-2 mirror | **UNBLOCKED — do it next** | `borhan2003/messidor-diabetic-retinopathy-dataset-jpg-format` holds Messidor-2 at **2240×1488** rather than our mirror's 512×512. But it has 1 200 files covering only **1 057 of our 1 744** labelled images; the 687 `IM*`-named ones are absent, so the upgrade is partial and would leave the corpus at mixed resolution. Worth doing only if I10c shows resolution binds. |
| I10 | Resolution: 512 vs 640 vs 768, traded against batch size and quota. | untested | **Confounded on Messidor-2**, which is already 512 px (`ISSUES.md` §4). Run on IDRiD + APTOS only. |
| I12b | **Acquire Messidor-1** — the only realistic external test for 3-class DME | **queued, not started** | Same clinical definition as IDRiD. Needs a duplicate check against our Messidor-2 mirror first (`ISSUES.md` §3). Until this exists there is no external DME number, and that is a stated limitation. |
| I14 | Grad-CAM on correct and incorrect cases. | untested | Addresses the interpretability gap *and* works as a bug detector: if the model attends to the optic disc rather than to lesions, something is wrong. Cheap. |
| I11 | Repeated cross-validation on the final candidates, mean ± interval. | untested | Mandatory before any headline number, given §7 of the protocol. |
| I15 | Auxiliary hard-exudate segmentation head from IDRiD's 81 masks. | untested | Small n, but the masks are exactly the lesion the DME label is about. Pairs naturally with I07. |
| I16 | Re-implement the two comparison baselines (SVM, single-output CNN) on our exact split. | untested | Makes the thesis' comparison table a fair one. Currently it compares numbers from different datasets and different protocols. |

| I17 | Cross-fitted decision cut-point tuning | **inconclusive → demoted** | Worth +0.047 QWK on E05, worth +0.008 (n.s.) on E06. It repairs a miscalibrated output layer; pretraining repairs the same thing better. Kept as a diagnostic, not part of the reported pipeline. |

| I18 | Re-run the external APTOS evaluation on GPU to get verified intervals | **not needed — resolved analytically** | Every APTOS group is one image, so the confusion matrix determines the bootstrap interval; recomputed and matched to Monte-Carlo error. Zero GPU spent. Removed from the queue. |

| I19 | **Source control for the resolution test** | **queued — must run BEFORE the resolution test** | The full-resolution images come from Messidor-1 TIFFs, so a naive resolution run changes resolution *and* file source, codec and compression history at once. Control: take the 1 057 overlapping images, downsample the Messidor-1 originals to exactly the 512 px our pipeline already uses, and train a model on those. If it matches the model trained on our existing Messidor-2 512 px copies (paired bootstrap, same folds), the source is not a confound and the resolution result is interpretable. If it differs, the resolution result is uninterpretable and the difference itself becomes the finding. ~1.5 h, and it protects a ~4 h run. |
| I20 | Native-resolution test, gated on I19 | **queued, with a licence attached** | I19 passed **in aggregate only**. So I20 must report: **aggregate QWK/accuracy attributable to resolution**; **per-class results explicitly under the source caveat**, since the source alone shifts per-class error by several points (E12, Moderate −6.5 surviving recalibration); and the PROTOCOL §4.2 note that no multiple-comparison correction is applied and credibility comes from replication across conditions. A per-class resolution gain must not be allowed to read as clean. Requires `cache_size >= size`, now enforced. |

### I20 — **RUN AS E17NAT, 2026-08-31. FALSIFIED. I10 / I10b / I10c closed.**

DR QWK at matched calibration 0.8749 (E10) vs 0.8662 (E17NAT), difference +0.0086
[−0.0025, +0.0199] — inside the ±0.015 band named in advance, interval containing zero,
nominally favouring E10. **Effective resolution above ~560 does not bind for DR.** ISSUES §18
remains a correct diagnosis with a null consequence: the bug was real, fixing it cost 7.2 h
and bought no QWK. **Correction 2026-09-01:** the I19 source caveat originally attached to
I20's per-class results does not apply — the consumption manifests show E10 and E17NAT read
the same files, differing only in `cache_size`. I20 changed one variable, not two. The −6.1
Mild change is still only a lead, on §4.2 grounds (one condition, no correction applied).

### I20 design — fixed before the run, launched as E17NAT 2026-08-31

**Hypothesis.** ISSUES §18 established that E10 was a **560 px run wearing a 640 px label**:
`build_cache` capped every image at 560 and `FundusDataset` then *upsampled* to 640, throwing
away the native-resolution Messidor-2 mirror before training saw it. E09 showed resolution
binds for DR (224 → 448 was worth +0.040 QWK and +18.6 points of Mild recall). **If
resolution still binds above 448, giving the pipeline genuinely more pixels — cache 768,
train 640, native mirror attached — should move DR QWK above E10's.**

**Control: E10.** Same `--size 640`, same `--batch 8`, same 40 epochs, EyePACS pretraining,
TTA, folds 0–4. The differences are `--cache-size 768` (so 640 is real rather than upsampled
from 560) and `--messidor-hi` (the 2240×1488 mirror, covering 1 057 of 1 744 Messidor-2
images). Batch is held at 8 deliberately: it is E10's, so it is not a second thing changing.

**Falsifying outcome, stated before the numbers.** *If DR QWK moves less than its interval
(≈ ±0.015 on 2 260 images), effective resolution above ~560 does not bind for DR, and §18's
correction — while still a real bug — costs us nothing in accuracy.* That is a publishable
result: it bounds how much of the literature's resolution advantage is reachable here, and it
closes I10/I10b/I10c as a line of work.

**Reporting licence, carried from I19 — this is binding, not commentary.** I19 (E12def vs
E12nat) passed **in aggregate only**. So E17NAT must report:
1. **aggregate QWK/accuracy as attributable to resolution**;
2. **per-class results explicitly under the source caveat** — the file source alone shifts
   per-class error by several points (E12: Moderate −6.8 before recalibration, −6.5 after),
   so a per-class resolution gain is **not cleanly attributable** and must not read as clean;
3. the §4.2 note that no multiple-comparison correction is applied, and that credibility comes
   from replication across conditions rather than a p-value in isolation.

### I22 — **RUN AS E18RECIPE, 2026-08-31. CLOSED — the gap is distribution shift.**

Re-scored APTOS with E08 **fold 0 only, no TTA**, matching the development pool's inference
recipe. The transfer gap is unchanged to within 0.5 points at every target (+17.58 / +14.26 /
+12.19 / +4.37 versus +18.12 / +14.33 / +12.33 / +4.43). Aggregate barely moves: QWK 0.8872
vs 0.8968, referable sens 99.26 % / spec 84.55 % vs 99.53 % / 84.28 %.

**The inference recipe explains essentially none of the gap.** T1's transfer gap is
distribution shift, and the caveat is now measured rather than argued. 20 minutes of GPU
converted a stated limitation into a closed one.

### I22 — original design, kept for the record

T1's transfer gap (`docs/generated/t1_transfer_gap.md`) compares **single-fold, no-TTA**
development predictions against a **5-fold logit ensemble with TTA** on APTOS. Both ensembling
and TTA change calibration, so the measured gap mixes distribution shift with the inference
recipe. Score APTOS with one fold's weights and no TTA (`--external-only --from-run`, plus a
single-fold / no-TTA option in `eval_external.py`) and recompute the table. **Falsified if**
the gap is materially unchanged, which would establish it as distribution shift outright.
Cheap, and it converts a stated caveat into a measured one.

### I23 — **RUN 2026-09-01. FALSIFIED. Ensembling is closed as a lever.**

| | ensemble | validation-selected single (E10) | gain |
|---|---|---|---|
| development pool | 0.8903 | 0.8749 | **+0.0154** |
| **APTOS (held out)** | 0.8878 | 0.8867 | **+0.0011 [−0.0033, +0.0053] — indistinguishable** |

**Completed with all six members, 2026-09-05.** E17NAT's external eval landed and the
conclusion is unchanged: ensemble of 6 scores APTOS 0.8892 against E10's 0.8867, a difference
of **+0.0025 [−0.0016, +0.0066] — indistinguishable**, versus +0.0136 on the development pool.
The five-member figure was +0.0011; adding the sixth moved it to +0.0025 and it remains inside
its interval.

**The development-pool gain was optimism.** By the criterion fixed before the run — *falsified
if the ensemble's APTOS QWK does not exceed the best member's by more than its interval* —
ensembling does not produce a better predictor on data no member has seen.

**A selection error of mine, caught before reporting.** The first version of
`src/ensemble_external.py` compared the ensemble against the member scoring highest **on
APTOS** (E08, 0.8972), giving −0.0094 [−0.0128, −0.0061], "significant" — i.e. the ensemble
looked actively harmful. That is model selection on the test set (`PROTOCOL.md` §3). The legal
comparison is against **E10, selected on the development pool**, and against that the ensemble
is merely indistinguishable. Both are printed in the output; only the validation-selected one
is the result. Note the direction: applying §3 correctly made the result *better*, not worse.

**What survives.** The dev-pool observation that ensembling helps DR and not DME still stands
as a statement about *variance versus supervision* (F7), but it is a development-pool
statement and is now labelled as one. **The three-cycle sequence 0.8933 → 0.8828 → not a
headline is the honest record** and is kept in full rather than tidied.

### I23 — original design, kept for the record

The archived-prediction ensemble reaches **DR QWK 0.8933** on the development pool, +0.0186
[+0.0089, +0.0283] over the best single run — significant, at zero GPU cost
(`docs/generated/ensemble_oof.md`). But every member was selected on that pool, so the number
is not held-out. **Score APTOS with each member (external-only, no retraining) and average the
logits the same way.** Only that number can be a headline. **Falsified if** the ensemble's
APTOS QWK does not exceed the best single member's by more than its interval — which would
mean the gain is dev-pool optimism, not a better predictor.

### I24 Stage 1 — RETFound cached linear probe — **pre-registered 2026-09-05, before launch**

**Backbone.** RETFound CFP ViT-L/16, MAE self-supervised on ~1.6 M retinal images, frozen.
Checkpoint sha256 `847f9dd0…` (stripped encoder of source `e1e4f66a…`), pinned in
`src/retfound_probe.py` and verified at load time; provenance in
`docs/RETFound_provenance.md`. Licence CC-BY-NC 4.0.

**Method.** One forward pass per image, features cached, then a cross-fitted linear ordinal
head (fold *f*'s head trained on the other folds only). This is the LP half of LP-FT, so it
answers the staging question by measurement rather than argument, and it costs ~15 min instead
of the ~19 h a full ViT-L fine-tune would need.

**Control: E09, not E08.** RETFound is 224-native. Interpolating its 197-token position
embedding to 448 would quadruple the token count and the compute, so the clean one-change
comparison is against **E09 (densenet121 @224, 5 folds, EyePACS-pretrained): DR QWK 0.8389 at
matched calibration.** Comparing against E08 @448 would confound backbone with resolution.

**The handicap, stated in advance.** Our own 224 → 448 jump is worth about **+0.04 QWK**
(E09 0.8389 → E08 0.8646, matched). RETFound is locked to 224, so it must overcome that before
it can beat our best. **Clearing E09 but not E08 is therefore informative about
representations, not a loss** — it would say a foundation backbone is worth more than doubling
resolution, while still not being the best available pipeline.

**Falsifying outcome, fixed before the numbers.** *A frozen linear probe on RETFound features
must beat E09's 0.8389 DR QWK by more than its interval (≈ ±0.015).* If it does not, a
representation trained on 1.6 M retinal images does not carry more DR signal than our own
224 px supervised backbone, and Stage 2 (fine-tuning) is not worth 19 h of quota.

**Note on the comparison's fairness, in RETFound's favour.** A *frozen linear probe* is being
compared against a *fully fine-tuned* network. That is deliberately hard on RETFound: if the
probe clears E09 anyway, the representation claim is strong. If it lands below E09 but well
above the floor, that is not decisive against fine-tuning it, and Stage 2 remains arguable —
which is exactly the fork to bring back to the owner rather than resolve unilaterally.

**The DME test, reported separately (F7).** F7 says the 3-class DME ceiling is supervision,
not architecture, on the strength of five falsified architectural interventions. A
representation trained on 1.6 M retinal images is the cleanest available challenge to it.
**If DME QWK moves past its ±0.03 band, F7 needs revising. If it does not, F7 becomes much
harder to argue with** — a foundation model failing where architecture failed points squarely
at the 516 labels, 51 of them the middle grade.

### Queue order 2026-09-05 — CORAL/CORN promoted above the backbone swaps

**Agreed with the owner's instinct, but for a different reason than the one offered.** The
proposed reason was that CORAL/CORN targets the adjacent-grade error mode F1 documented. That
argument is weaker than it looks: F1's finding was that the Mild collapse is **calibration**,
and §4.1 plus T1 established that cut-point placement dominates adjacent-grade behaviour —
which cross-fitted threshold tuning already handles. A better ordinal *loss* would have to
improve the underlying **ranking**, not the cut placement.

**The reason to promote it anyway is the backbone evidence, which is bad.** At matched
calibration on the 5-fold development pool:

| run | backbone | size | DR QWK |
|---|---|---|---|
| E08 | densenet121 | 448 | **0.8646** |
| E07 | tf_efficientnet_b3 | 448 | **0.8441** |

**The one clean 5-fold backbone swap this project has run made things worse.** E11's apparent
EfficientNet advantage was measured on 3 folds, and F3 showed its external advantage dissolved
at matched operating points. So the prior on another backbone swap producing a durable gain is
low, and backbone comparisons are precisely the class §4.1 keeps dissolving. CORAL/CORN
changes the training objective instead of the feature extractor — a different, untried axis.

**Expected effect, stated honestly: small.** Our head is *already* ordinal (thresholds on
P(y > k)) with rank-consistent decoding via `cummin`. CORAL/CORN is an increment on an
existing ordinal treatment, not a new capability. Order: **CORAL/CORN, then ConvNeXt, then
Swin**, and if CORAL/CORN is null I would spend the remaining quota on I16 rather than on the
second backbone.

### I15 — auxiliary exudate segmentation head — **SKIPPED, and the reason is the point**

The owner's condition was: run it only if a falsifying outcome exists that **81 masks could
plausibly clear**. It does not, and saying so is more useful than running it for completeness.

* The masks cover **81 of 2 260 images (3.6 %)**. As an auxiliary loss their gradient
  contribution is small by construction.
* The metric they would have to move is **3-class DME QWK, whose interval is ±0.03 on 516
  images** with 51 in the middle grade.
* **E14MAC is the decisive precedent.** It handed the DME head the exact region the label is
  defined by — a macula crop, with the fovea inside the window in over 99 % of images — and
  moved DME QWK by nothing (+0.0237 [−0.0094, +0.0566], the crop nominally worse). An
  auxiliary signal on 3.6 % of the pool is a far weaker intervention than that.

**So the honest pre-registration would be a hypothesis I do not believe**, and running it would
consume ~4 h to produce a sixth null that F7 already predicts. Skipped.

**What the 81 masks are still worth, and it is not performance.** Use them as a *diagnostic*
(`IDEAS.md` I14): measure whether the model's saliency overlaps annotated exudate regions.
That tests whether the network attends to the lesion the DME grade is defined by, which is an
interpretability result the thesis can use and a genuine bug detector — if attention sits on
the optic disc instead, something is wrong. Cheap, CPU-feasible on 81 images, and it does not
pretend to raise a number.

### CORAL/CORN — hypothesis restated 2026-09-05 (owner withdrew the original rationale)

The original rationale — *it targets the adjacent-grade error mode F1 documented* — was
withdrawn by the owner after the counter-argument: adjacent-grade behaviour is dominated by
**cut-point placement**, which cross-fitted threshold tuning already handles (§4.1, T1).

**So the hypothesis is now stated as the thing that actually has to be true for it to matter:**
*an ordinal loss with a rank-monotonicity constraint improves the underlying **ranking** of
images, not merely where the cuts fall.* Since QWK at matched calibration is computed with
cut-points already optimised, any gain must come from a better ordering.

**Falsifying outcome:** DR QWK at matched calibration does not exceed E08's 0.8646 by more
than its interval (≈ ±0.015). **Expected effect: small** — our head is already ordinal
(thresholds on P(y > k)) with rank-consistent `cummin` decoding, so CORAL/CORN is an increment
on an existing ordinal treatment rather than a new capability.

### I24 Stage 1 — result, and why the headline number is NOT a RETFound verdict

**DR QWK 0.7008 [0.6758, 0.7240] at matched calibration, against E09's 0.8389.** The
pre-registered criterion (beat 0.8389 by more than ≈±0.015) is not met, by 0.138.

**That number must not be reported as a verdict on RETFound**, and the pre-registration said
so in advance: it compares a **frozen linear probe** against a **fully fine-tuned network**,
so it measures probing versus fine-tuning. The comparison that isolates the representation is
`I24BASEPROBE` — the identical probe on frozen ImageNet DenseNet121 features
(`src/compare_probes.py`, `docs/generated/probe_vs_probe.md`).

**Two errors of mine on the way, recorded because both were mine and not the model's:**

1. The run's own `results.json` reported DR QWK 0.5141 at **23.2 % accuracy against a 52.4 %
   floor**, No-DR recall 0.005 — apparently catastrophic. It was a **decoding failure**: the
   probe decoded with a fixed `floor(score + 0.5)` while being compared against E09's
   *matched-calibration* figure. That is exactly the asymmetry §4.1 exists to prevent, built
   into my own script. Recalibrated, 0.5141 → **0.7008**.
2. The first launch died instantly on `ISSUES.md` §15 — Kaggle mounted every dataset under a
   single `datasets/` directory, so `/kaggle/input/retfound-cfp-encoder` did not exist. The
   same one-level-scan assumption that once cost E08 an external evaluation.

**Decision rule, pre-committed by the owner before the control landed:** RETFound clearly
beats ImageNet beyond the paired interval → Stage 2, launched without asking, split under the
10 h cap. Indistinguishable → Stage 2 closed and the null goes in `FINDINGS.md` as a measured
statement about what foundation-model pretraining buys at this data scale. ImageNet wins →
same close, stated plainly.

### I24 Stage 2 — **CLOSED 2026-09-05 after folds 0–1. See `FINDINGS.md` F8.**

I24FT01, folds 0–1, fold-matched at matched calibration: **RETFound 0.7896 vs E09 0.8298,
−0.0399 [−0.0669, −0.0124], significant** — the pre-registered criterion (exceed 0.8389 by
more than ~0.015) is missed in the wrong direction. Folds 2–4 not run: ~14 h to complete a
result already significantly negative on 40 % of the data. **Reported as a 2-of-5-fold
result**, since the pre-registration specified five.

Recorded as **F8**: frozen RETFound features beat frozen ImageNet features (+0.0401,
significant) while fine-tuned RETFound loses to DenseNet+EyePACS (−0.0399, significant). Two
candidate mechanisms — labelled in-domain data beating unlabelled at this scale, and 303 M
parameters against 1 808 images per fold — are confounded by construction and stated as
hypotheses.

### I24 Stage 2 — original pre-registration, kept for the record

**What triggered it.** Probe versus probe, same images, same cross-fitted head, same matched
calibration, only the frozen backbone differing (`docs/generated/probe_vs_probe.md`):

| head | n | RETFound CFP ViT-L/16 | ImageNet DenseNet121 | A − B | 95 % interval | verdict |
|---|---|---|---|---|---|---|
| DR, 5-class | 2 260 | **0.7008** | **0.6611** | **+0.0401** | [+0.0122, +0.0692] | **significant** |
| DME, 3-class | 516 | 0.7074 | 0.6645 | +0.0434 | [−0.0173, +0.1018] | indistinguishable |

**RETFound's representation carries significantly more DR signal than ImageNet's**, which is
the condition the owner pre-committed to. Stage 2 launches without asking.

**Design.** `--backbone retfound:/kaggle/input` (hash-verified at load, §9), 224 px, LP-FT
staging: 5 probe epochs on a frozen backbone, then unfreeze at 1e-5 with 2 warmup epochs. The
probe result *is* the empirical justification for that staging — the LP half has already been
measured rather than argued for.

**Split across three runs** to stay under the 10 h cap: ViT-L/16 @224 is ~5.3× E08's per-image
compute, so 5 folds × 30 epochs is ~19 h. Folds 0–1, folds 2–3, fold 4.

**Control: E09** (densenet121 @224, EyePACS-pretrained, 5 folds) at matched calibration,
**DR QWK 0.8389**. **Falsifying outcome:** DR QWK at matched calibration does not exceed
0.8389 by more than its interval (≈ ±0.015).

**The confound, stated rather than hidden.** RETFound replaces *both* the backbone and the
EyePACS pretraining stage — E09 is ImageNet→EyePACS→fine-tune, this is retinal-SSL→fine-tune.
That is the treatment rather than an accident (the question is what retinal self-supervised
pretraining buys), but a per-class difference inherits it and **must not be attributed to
architecture alone** (§4.2).

**The handicap still applies.** Our own 224 → 448 jump is worth ~+0.04 QWK, and RETFound is
224-locked. Clearing E09 but not E08's 0.8646 would say a foundation backbone is worth about
as much as doubling resolution — informative, and still not the best available pipeline.

## Rejected

| ID | Idea | Verdict | Evidence |
|---|---|---|---|
| R01 | `load_idrid_all()` / `--no-holdout` / `memorize=True` — train and evaluate on the same 516 images. | **rejected** | Produces training accuracy quoted as test accuracy. Its own docstring states the goal is "to show the highest possible accuracy numbers rather than to measure generalization". Delete the code path. `ISSUES.md` "Things not to redo". |
| R02 | Messidor-1 for the **DR** head. | **rejected** | 4-level lesion-count scale, not ICDR; no clean mapping to 5 classes. Use it for DME only, where the definition is identical. `data/LABEL_MAPPING.md`. |
| R03 | Flattening IDRiD's 3-class DME to binary to match Messidor-2. | **rejected** | Discards the 3-class grading that is the thesis' stated contribution. Use partial labels instead. |
