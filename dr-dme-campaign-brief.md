# DR/DME Grading — Experiment Campaign Brief

## The claim the evidence must support or refute

**The DR grading literature systematically reports decision-rule failures as representation failures.** Published external minority-grade collapse — grade-1 F1 = 0.000, grade-2 recall = 0.189 — is attributed to what the encoder cannot see. We have direct evidence that a substantial fraction of it is recoverable by changing the decision rule alone, with the model untouched.

Two corollaries:

- Under matched calibration, reported architecture gains in cross-dataset DR grading shrink or reverse; the QWK-optimal decision rule is systematically misaligned with clinically usable operating points.
- Where supervision is genuinely the binding constraint (DME on IDRiD), no architectural intervention moves the metric — which is what a real ceiling looks like, in contrast to the apparent ceilings above.

**Keep three things separate throughout. The literature conflates them:**

1. **Probability calibration** — temperature scaling, ECE, Brier. What Poyrazer et al. study.
2. **Decision-rule selection** — ordinal cutpoints, referral thresholds. What ORDER-DR studies.
3. **Representation quality** — what the frozen or fine-tuned encoder encodes. What both *claim* to study.

Nobody separates them. Our factorial design does. Every experiment below exists to test one of these claims. Do not add architectural variants.

---

## Reference target: Sheng et al., ORDER-DR (Front Endocrinol 2026, DOI 10.3389/fendo.2026.1923216)

Read the paper and their repo (`github.com/Hajimi-Sudo/ORDER-DR`) before starting. Their setup:

- Development: APTOS 2019 (3,662 images), 70/15/15 stratified, 3 seeds. External: Messidor-2 (1,744 gradable).
- Ordinal thresholds τ₁..τ₄ fit on the expected-grade score `g(x) = Σ k·p(k)` to maximize **validation QWK**.
- ORDER-DR = fixed 0.5/0.5 probability fusion of a class-balanced EfficientNet-B0-384 branch and an ordinal-risk (LORS) EfficientNet-B0-384 branch.
- Headline: external QWK 0.6423 ± 0.0364 vs 0.5725 ± 0.0184 for the best single branch.

**Exploitable weaknesses.** Each maps to a section of our paper:

1. Their winning model has the **lowest external accuracy of all ten** models evaluated (0.5214 vs 0.6414 for LORS-384).
2. Their calibrated rule assigns 40.9% of external images to grade 1 when true grade-1 prevalence is 15.5% — 713 predicted vs 270 actual, grade-1 precision 0.193. QWK rises anyway.
3. External referable-DR sensitivity at the locked threshold is 0.415 (267 of 457 referable eyes missed). At a threshold targeting validation sensitivity ≥ 0.99, external sensitivity is still 0.679.
4. **No matched-calibration comparison anywhere.** The 0.07 QWK "architecture gain" is never tested against the calibration null.
5. Thresholds are fit to maximize validation QWK; QWK is then the primary reported metric. Circular.
6. Three seeds, external SDs 0.018–0.066. The authors themselves label their bootstrap contrasts descriptive only.
7. No preprocessing — no border removal, retinal-field extraction, or illumination normalization — a plausible driver of the domain gap, framed as a reproducibility choice.
8. One external set, one direction (APTOS → Messidor-2). No reverse validation.
9. Image-level analysis only, despite Messidor-2's two-images-per-examination structure. No eye-pair clustering in CIs.
10. DME excluded entirely.

We win on **4, 5, 6, 8, 9, 10**. We do not try to beat their QWK.

---

## Reference target 2: Poyrazer et al. (Front Med 2026, DOI 10.3389/fmed.2026.1815982)

Repo: `github.com/opisthion06/diabetic_retinopathy`, archived at Zenodo DOI 10.5281/zenodo.19210682. Frozen-encoder benchmark of MedSigLIP / RETFound / EfficientNet-B0, APTOS → Messidor-2, calibration as co-primary endpoint.

**What they got right — adopt all of it:**

- Patient-level cluster bootstrap (2,000 iterations) on the external set. Messidor-2 within-patient grade correlation is r = 0.84, binary concordance 90.9%; design effect ≈ 1.84. Image-level resampling inflates significance.
- Contamination audit table mapping every dataset to every encoder's documented pretraining sources. They excluded EyePACS (in MedSigLIP and RETFound pretraining) and FLAIR (pretrained on APTOS, **IDRiD**, and OIA-DDR). They make no claim of contamination freedom, only transparency.
- DeLong for AUC, McNemar for correctness, paired bootstrap for the rest, Benjamini–Hochberg FDR correction within each dataset–task–statistic group.
- 5-fold CV × 5 seeds, with seed-level variance reported separately to show it is negligible.
- ECE interpreted alongside Brier and discrimination, never alone: RETFound has the *lowest* external ECE (0.086) while having the *worst* AUC (0.697), because its probability mass compresses near the boundary. A low ECE can be an artifact of an uninformative predictor.

**What we exploit:**

1. **Grade-1 external F1 = 0.000 for RETFound and EfficientNet-B0** — not one of 270 mild cases identified. MedSigLIP only 0.153. They attribute this to representational blindness to microaneurysms. **They never adjusted the decision threshold.** This is the same phenomenon we already partially reversed by recalibration alone.
2. **Resolution confound they concede.** MedSigLIP runs at 448×448, the other two at 224×224 — four times the pixels. Grade 1 is defined by scattered microaneurysms, which are plausibly destroyed at 224 px. Their headline conclusion about vision–language pretraining cannot be separated from resolution.
3. **Temperature scaling fails under shift** — dev ECE 0.014–0.022, external 0.086–0.149; TS moved external ECE by 0.004 for MedSigLIP and *increased* it by 0.002 for RETFound. Scalar post-hoc calibration is not the lever. Decision-rule selection is a different lever they never test.
4. **Fold ensembling did not rescue external performance** for any encoder — they conclude the error is representational rather than stochastic. This independently matches our own finding that the ensemble's development gain did not survive external validation. Cite it.

---

## Phase 0 — Matched-calibration re-ranking (no training; run this first)

For every checkpoint already in the archive, evaluate on the internal test split and on the external set under four decision rules:

- **R1** — default argmax.
- **R2** — ordinal thresholds fit on internal validation to maximize QWK (the ORDER-DR rule).
- **R3** — ordinal thresholds fit on internal validation subject to referable sensitivity ≥ 0.90.
- **R4** — oracle: thresholds fit on the external set itself. Report as an upper bound only, never as a result.

**Deliverables:**
- Table: model × rule × {QWK, macro-F1, accuracy, referable sensitivity, referable specificity, ECE}.
- Spearman rank correlation between model rankings under R1, R2, R3.
- Direct comparison: QWK spread *across rules within a model* vs QWK spread *across models under a fixed rule*.

**Decision gate.** If between-rule spread is comparable to or larger than between-model spread, the central claim holds — proceed. If not, stop and report back before Phase 1. Do not proceed on a weak Phase 0.

---

## Phase 1 — QWK / sensitivity Pareto frontier

Sweep the referable threshold across its full range on the external set. For each point record QWK, macro-F1, accuracy, sensitivity, specificity, PPV, NPV.

Plot the frontier. Mark three points: the QWK-optimal threshold, the threshold meeting the supervisor-agreed clinical sensitivity target, and the argmax default. Quantify the QWK cost of moving from the first to the second, and the sensitivity cost of the reverse.

This figure is the paper's centrepiece. Also report the predicted-vs-true class distribution at each marked point — the distortion is the mechanism, not a side effect.

---

## Phase 2 — Reproduce two published failures and test whether they are threshold artifacts

This is the paper's strongest possible experiment. Both target papers released code.

**2a — Reproduce.** Clone both repos, rebuild their pipelines, confirm we can recover their reported external numbers within their stated seed variance. Report the reproduction fidelity honestly, including anything that does not reproduce.

**2b — Apply matched calibration to their predictions.** Take their external prediction files as-is. Without touching a single model weight, refit the decision rule under R3 (sensitivity-constrained) and R4 (oracle). Report what happens to:

- Poyrazer et al.'s grade-1 F1 = 0.000 (RETFound, EfficientNet-B0) on 270 Messidor-2 images.
- ORDER-DR's grade-2 recall of 0.189.

If a meaningful fraction of these "catastrophic representation failures" recovers under a threshold change alone, that is the paper. If it does not recover, that is also publishable — it would establish the failure as genuinely representational and would be the first study to actually test it. **Either outcome is a result. Do not stop if the first one is negative.**

**2c — Decompose.** For each recovered case, partition the gap into: threshold selection, probability calibration (temperature scaling), and residual representation deficit. This decomposition is the methodological contribution.

**2d — Technique collapse (secondary).** If time permits, take two or three techniques the literature reports as gains (ordinal regression head, class-balanced loss, probability-level fusion) on identical backbone and splits, and report each delta under R1 versus matched calibration, with cluster-bootstrap CIs.

---

## Phase 3 — DME supervision ceiling

**3a — Scaling curve.** Train the DME head on 12.5%, 25%, 50%, 100% of available DME labels; 5 seeds each. Plot QWK vs n with CIs. Fit a curve and extrapolate the label count needed for a target QWK. This converts six failed interventions into a measured ceiling.

**3b — Label transfer probe.** Messidor-1 carries a risk-of-macular-edema grade defined by shortest hard-exudate–to-macula distance, which is close to IDRiD's DME definition. We already established ~88% perceptual-hash overlap between Messidor-1 and the development pool. Attempt to transfer ME grades onto matched development images.

Before using them: verify label-definition compatibility on a manually inspected sample, and quantify agreement on any images carrying both an IDRiD and a Messidor-1 label. Report all DME results twice — IDRiD-only and pooled — never pooled alone.

---

## Phase 4 — RETFound as a ceiling probe

RETFound's role has changed. We are not looking for a gain from it; we are looking for **evidence about what kind of ceiling we hit**. If a ViT pretrained on ~1.6M retinal images still plateaus at the same DME QWK, the ceiling is supervision, not representation.

**Two design constraints, both mandatory — the probe is worthless without them:**

1. **Fine-tune. Do not freeze.** Poyrazer et al. report frozen RETFound at external AUC 0.697, *below* the ImageNet baseline (ΔAUC = −0.051, p = 0.016, cluster-robust). But the same encoder reaches AUROC 0.822 on the same dataset pair under full fine-tuning. The frozen result reflects a known property of MAE-pretrained embeddings — they are not linearly separable without non-linear adaptation. A frozen failure here would confound the supervision ceiling with linear separability and prove nothing. Fine-tune, or at minimum use LoRA, and state which.

2. **Control resolution.** RETFound's native input is 224×224. DME grading depends on hard exudate position relative to the macula, and DR grade 1 depends on scattered microaneurysms — lesion classes plausibly destroyed at that resolution. Run at RETFound's native resolution *and* at an interpolated-positional-embedding higher resolution matched to our CNN branches. Report both. If the ceiling moves with resolution, it was never a supervision ceiling.

Position explicitly against Poyrazer et al.: our claim is about fine-tuned adaptation under matched decision rules, theirs is about frozen transfer under a fixed one. Different questions, and the distinction is a contribution.

---

## Reporting standards (non-negotiable, all phases)

- **Runtime manifest.** Every run writes a manifest of files actually opened, with per-fold hashes. Config files record intent, not consumption.
- **Matched calibration before attribution.** No performance delta is attributed to an architectural or training change until it has survived a matched-calibration check.
- **Uncertainty.** Minimum 5 folds × 5 seeds. Report seed-level variance separately from fold-level. Bootstrap CIs on every headline number. Report whether typical literature-scale deltas fall inside our seed noise.
- **Clustering.** Patient/examination-level cluster bootstrap (2,000 iterations) for all external comparisons, with the design effect reported. Never image-level resampling on a dataset with two eyes per subject.
- **Statistical tests.** DeLong for AUC, McNemar for classification correctness, paired bootstrap for everything else. Benjamini–Hochberg FDR correction within each dataset–task–statistic group. Report effect sizes with CIs, not p-values alone.
- **Calibration metrics are never reported alone.** ECE always appears alongside Brier and a discrimination metric — a compressed, uninformative predictor can post a deceptively good ECE.
- **Contamination audit.** Build a dataset × encoder table mapping every dataset we touch to every pretrained model's documented pretraining corpus. Exclude FLAIR outright — its pretraining includes IDRiD, our primary dataset. Check RETFound's Moorfields/UK Biobank corpus and any EyePACS-derived weights against our development pool. Claim transparency, not contamination freedom.
- **Bidirectional external validation.** Run both directions of the development/external split. Report both.
- **Provenance of comparisons.** Any cross-model comparison names which model was selected on which split. Never compare against a member selected on the test set.
- **Negative results are results.** Report interventions that did not work, with the same rigour as those that did.

---

## Sequencing

Phase 0 this week — it is free and it gates everything else. **Phase 2a–2c next: it is now the headline experiment and it needs no training, only their released prediction pipelines.** Phase 1 alongside it. Phase 3 in parallel if compute allows. Phase 4 last, since it is the most expensive and the least load-bearing.

Report after Phase 0 before committing further compute.
