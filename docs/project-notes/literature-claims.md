# CLAIMS LEDGER

Every claim extracted from every paper we ingest, with its evidence, how we can test it, and what it costs.

**Status vocabulary**
- `UNTESTED` — nobody, including the authors, has tested it under the conditions where it matters
- `CHEAP` — testable with no GPU (prediction files, re-analysis, audit)
- `MODERATE` — testable with our existing checkpoints plus modest compute
- `EXPENSIVE` — needs new training runs
- `NOT-TESTABLE` — outside our scope or resources; record and move on
- `CONTESTED` — two or more papers disagree; resolving it is a contribution in itself

**Rule:** a claim that does not become an experiment or an explicit "rejected because X" is not finished.

---

## Citation structure

ORDER-DR (P1) cites **all four** of the other papers: Poyrazer (its ref 13), Dual-SwinOrd (ref 18), CLAIM (ref 21), Nielsen (ref 28). It is the hub of this citation cluster. Build our related-work section around it.

**Above the cluster sits P8 (RETFound, Nature 2023).** P2 (R1/R2), P3 (N1), P6 (M2/M4) and P7 (V3) are all downstream of it, and each disputes or reinterprets it in a cell P8 never tested. The related-work structure is therefore two-level: P8 as the claim under audit, P1–P7 as five independent partial re-tests of it.

**P9–P12 form a second, separate cluster: deployment and screening evidence**, not model-development papers. P10 cites P9 (its ref 81) and P8 (its ref 35). P12 cites Lee et al. 2021 (its ref 36) — the head-to-head study we have been chasing. This cluster is where the decision-rule argument stops being a methodological point and becomes a clinical one.

---

## CLAIM PROVENANCE INDEX

Every claim ID in this ledger, and exactly which paper it came from. **Nothing in this file is our own finding unless it is labelled F1–F7.**

| Prefix | Paper | Full citation | Role in our paper |
|---|---|---|---|
| **S1–S7** | P1 | Sheng et al., ORDER-DR, *Front Endocrinol* 2026 | Related-work hub; circular threshold tuning |
| **R1–R8** | P2 | Poyrazer et al., *Front Med* 2026 | Methods to adopt; grade-1 collapse (R6) |
| **N1–N5** | P3 | Nielsen et al., *J Med Imaging* 2025 | Frozen-vs-fine-tuned cell of Contradiction 1 |
| — | P4 | Mongan et al., CLAIM, *Radiol AI* 2020 | Audit instrument only, not a claim source |
| **D1–D6** | P5 | Yu et al., Dual-SwinOrd, *Bioengineering* 2026 | Contradiction 2; the AUC/calibration error (D6) |
| **M1–M6** | P6 | Tang et al., MMRDR, *Sci Data* 2026 | Contradiction 1; FLAIR contamination failure |
| **V1–V5** | P7 | Ruamviboonsuk et al., *Taiwan J Ophthalmol* 2024 | Introduction (V4, V5) |
| **Z1–Z11** | P8 | Zhou et al., RETFound, *Nature* 2023 | Source paper under audit; Z3 is the find |
| **T1–T6** | P9 | Tomić et al., *Biomedicines* 2024;12:34 | Audit case; mild-NPDR boundary instance |
| **Y1–Y7** | P10 | Yang et al., *eClinicalMedicine* 2025;81:103089 | Introduction; audit case; two leads |
| **H1–H5** | P11 | Tahir et al., *Front Med* 2025;12:1519768 | Contradiction 2 at meta-analysis scale |
| **G1–G8** | P12 | Duggal et al., *JMIR Med Inform* 2025;13:e67529 | **Strongest external support we have**; Contradictions 1b and 2b |
| **F1–F7** | — | **Ours** | See "Our findings → evidence map" at the end |

---

## P1 — Sheng et al., ORDER-DR
*Front Endocrinol 2026, DOI 10.3389/fendo.2026.1923216. Code: `github.com/Hajimi-Sudo/ORDER-DR`*
APTOS 2019 dev (70/15/15, 3 seeds) → Messidor-2 external (1,744 gradable).

| ID | Claim | Their evidence | Status | Our test |
|---|---|---|---|---|
| S1 | Dual-branch 0.5/0.5 fusion beats either branch alone externally | Ext QWK 0.6423±0.0364 vs 0.5725±0.0184 (LORS-384) and 0.5534±0.0387 (CB-384) | CHEAP | Phase 0/2b: re-rank under matched calibration. Their 0.07 gain is never tested against the calibration null |
| S2 | Validation-derived ordinal thresholds transfer to an external set without external labels | Thresholds fit on APTOS val, applied unchanged | CHEAP | They transfer *the rule*, not *the performance*: ext referable sensitivity is 0.415. Test whether "transfers" is the right word |
| S3 | ORDER-DR beats Swin-Tiny externally despite losing internally | Ext QWK 0.6423 vs 0.5628±0.0255 | CHEAP | ~2 SD gap on 3 seeds. Re-test with 5×5 and cluster bootstrap |
| S4 | Ordinal agreement and referable-risk ranking are complementary and can diverge | ORDER-DR best QWK; LORS-384 best AUPRC/Brier | CHEAP | **Agrees with our finding.** Cite as independent support, then go further: they observe the divergence, we explain the mechanism |
| S5 | Calibrated thresholds reduce severe-grade underestimation | 84.7±8.5 → 49.7±6.5 underestimated grade-3/4 | CHEAP | True but incomplete: the same rule pushes grade-1 predictions to 713 when only 270 exist (precision 0.193) |
| S6 | Grad-CAM attention on retinal pathology supports validity | Two hand-picked true-positive cases | NOT-TESTABLE | Two cherry-picked examples are not evidence. Note in critique, do not spend compute |
| S7 | External probability calibration degrades and needs local recalibration | ECE 0.049±0.001 (APTOS) → 0.160±0.008 (Messidor-2) | CHEAP | **Agrees with us.** But connect to our deployment recalibration harm curve — below ~100 local labels, recalibration hurts |

**Unstated weaknesses to exploit:** lowest external accuracy of all ten models (0.5214 vs 0.6414); thresholds fit to maximize validation QWK then QWK reported as primary (circular); 3 seeds only; no preprocessing; one direction only; image-level analysis on a two-eyes-per-exam dataset (violates CLAIM item 21, which they cite); DME excluded.

---

## P2 — Poyrazer et al., frozen-encoder calibration benchmark
*Front Med 2026, DOI 10.3389/fmed.2026.1815982. Code: `github.com/opisthion06/diabetic_retinopathy`, Zenodo 10.5281/zenodo.19210682*
APTOS 2019 dev (5-fold × 5 seeds) → Messidor-2 external. MedSigLIP / RETFound / EfficientNet-B0, all frozen, identical MLP head.

| ID | Claim | Their evidence | Status | Our test |
|---|---|---|---|---|
| R1 | Under frozen transfer, MedSigLIP generalizes far better than RETFound and EfficientNet-B0 | Ext AUC 0.915 vs 0.697 vs 0.745; ΔAUC +0.219 (95% CI 0.180–0.258) | MODERATE | Confounded with resolution (448 vs 224). Test at matched resolution |
| R2 | Domain-specific pretraining does not guarantee domain-general frozen representations; RETFound < ImageNet externally | ΔAUC −0.051, p=0.016 cluster-robust | CONTESTED | **Directly contradicts Nielsen N1 and our own frozen result.** See "Contradiction 1" below |
| R3 | Development-set discrimination cannot differentiate encoders | All three at AUC 0.980–0.985 internally, wildly divergent externally | CHEAP | **Agrees with us.** Strong plank; we extend it from encoders to decision rules |
| R4 | Temperature scaling works in-distribution, fails under domain shift | Dev ECE 0.014–0.022; ext 0.086–0.149. TS moved ext ECE by 0.004 (MedSigLIP), **+0.002** (RETFound) | CHEAP | **Agrees with us and is load-bearing.** Establishes that scalar probability calibration is not the lever — leaving decision-rule selection, which they never test |
| R5 | Brier should be a reporting standard; ECE alone misleads | RETFound has lowest ext ECE (0.086) and worst AUC (0.697) — probability mass compressed at boundary | CHEAP | Adopt outright as a reporting standard. Also use against Dual-SwinOrd's D6 |
| R6 | All encoders catastrophically fail on grade 1 externally — a representation limitation | F1 = 0.000 for RETFound and EffNet-B0 on 270 images; MedSigLIP 0.153 | **CHEAP — HEADLINE** | They never adjusted the threshold. We recovered minority-class recall ~5% → ~78% by cut-point shift alone. **This is Phase 2b** |
| R7 | Fold ensembling does not rescue external performance → error is representational, not stochastic | Supplementary Table S3 | CHEAP | **Independently matches our finding** that ensemble dev gain did not survive external validation. Cite |
| R8 | MAE embeddings need non-linear fine-tuning; contrastive VL embeddings are linearly separable | Cited from representation-learning literature, not tested here | MODERATE | Asserted, not demonstrated. Testable via linear vs MLP probe on both — they ran this ablation (S8) but only for ranking, not for this mechanism |

**What to adopt wholesale:** patient-level cluster bootstrap (2,000 iters, design effect ≈1.84, Messidor-2 within-patient r=0.84); contamination audit table; DeLong + McNemar + paired bootstrap + Benjamini–Hochberg; 5 folds × 5 seeds with seed variance reported separately.

⚠️ **They excluded FLAIR because its pretraining corpus includes APTOS, IDRiD, and OIA-DDR.** IDRiD is our primary dataset. FLAIR stays hard-excluded.

---

## P3 — Nielsen et al., homomorphic-encryption federated learning
*J Med Imaging 2025;12(3):034504, DOI 10.1117/1.JMI.12.3.034504. Code: `github.com/chrisnielsen/homomorphic-encryption-fl`*
APTOS-2019 + ODIR-5K (6,457 images). Frozen RETFound → 1,024-d features → 5,125-parameter multiclass logistic regression head (OvR). **No external validation.**

| ID | Claim | Their evidence | Status | Our test |
|---|---|---|---|---|
| N1 | A 5,125-parameter linear head on frozen RETFound features matches fully fine-tuned RETFound (303.3M params) | APTOS AUROC 0.94±0.01 vs 0.94±0.02, p=0.50; ODIR-5K 0.80±0.02 vs 0.79±0.01, p=0.53. ANOVA, 5 seeds | **CONTESTED — HIGH VALUE** | See "Contradiction 1". We have both frozen and fine-tuned RETFound |
| N2 | Federated training matches centralized | APTOS 0.93±0.01 (fed) vs 0.94±0.02 (central) | NOT-TESTABLE | Out of scope. Record only |
| N3 | HE preserves accuracy while differential privacy destroys it | HE: AUROC 0.78±0.02, same as no defense. DP at σ₀=100: 0.78 → 0.47±0.05 | NOT-TESTABLE | Out of scope |
| N4 | Frozen retinal features leak identity and clinical attributes under gradient inversion | Reconstruction MSE 3.07×10⁻¹⁸ undefended; recovered myopia AUROC 0.82, glaucoma 0.77, cataract 0.78 | NOT-TESTABLE | Out of scope, but **cite in the discussion**: it is a strong argument for why frozen-feature pipelines are not automatically privacy-safe |
| N5 | Efficiency: 95.9× less compute, 63.0× less data transfer than full RETFound FL | 11.51 vs 0.12 TFLOPs; 80.72 vs 1.28 GB | NOT-TESTABLE | Record only |

**Weakness that makes N1 fragile:** they never test out of distribution, and they say so in their limitations. Their entire N1 claim is an in-distribution claim being read by the field as a general one.

**Discrepancy to get right in our own writing:** Nielsen describes RETFound as pretrained on 0.9M unlabeled fundus images; Poyrazer says ~1.6M fundus and OCT images. Both are defensible readings of the RETFound paper (fundus-only vs fundus+OCT). State our number precisely and say which we mean.

---

## P4 — Mongan et al., CLAIM checklist
*Radiology: Artificial Intelligence 2020;2(2):e200029*
Not a research paper. A 42-item reporting checklist, **formally adopted by Radiology: AI as part of its manuscript review standard.**

**Use: audit instrument, not a claim source.** Score every target paper against CLAIM and report the compliance matrix. This converts "these papers are sloppy" into "here is a structured audit against the field's own adopted standard." Zero GPU cost.

Items where our findings show non-compliance actually changes conclusions:

| Item | Requirement | Why it bites |
|---|---|---|
| 21 | Partitions disjoint at patient level or higher | ORDER-DR does image-level analysis on Messidor-2 (two eyes per exam) **while citing CLAIM**. Poyrazer does it correctly |
| 26 | Method of selecting the final model | Threshold-and-checkpoint selection rules are where the calibration confound hides |
| 27 | Ensembling techniques, if applicable | ORDER-DR's fixed 0.5/0.5 fusion; our own ensemble finding |
| 29 | Statistical measures of significance and uncertainty | 3 seeds with SD 0.02–0.06 and no inferential test |
| 30 | Robustness or sensitivity analysis | Nobody varies the decision rule — this is the missing sensitivity analysis, and it is our paper |
| 32 | Validation or testing on external data | Nielsen: none. Dual-SwinOrd: none |
| 36 | Diagnostic accuracy estimates with precision; calibration curves | Dual-SwinOrd reports no calibration analysis at all |
| 37 | Failure analysis of incorrectly classified cases | Grade-1 F1 = 0.000 is reported but never diagnosed |

**Planned deliverable:** a CLAIM compliance table across all target papers, as a paper figure. Include ourselves in it, scored honestly.

---

## P5 — Yu et al., Dual-SwinOrd
*Bioengineering 2026;13:374, DOI 10.3390/bioengineering13040374. No code released.*
Swin Transformer + PubMedCLIP semantic prior modulation (SPM) + progressive lesion-aware kernel attention (PLKA) + dual head (classification + ordinal). APTOS 2019 and DDR, **internal splits only, no external validation.**

| ID | Claim | Their evidence | Status | Our test |
|---|---|---|---|---|
| D1 | A dual-head design resolves the accuracy–ordinality trade-off, achieving SOTA on both simultaneously | APTOS Acc 0.880±0.0128, QWK 0.937±0.0146; DDR Acc 0.865±0.0134, QWK 0.904±0.0178 | **CONTESTED — DIRECTLY OPPOSED TO US** | Their own Table 2 refutes it: CRA-Net accuracy 0.891 > their 0.880. They win only on kappa. **Third independent instance of our tension, presented as its solution** |
| D2 | PubMedCLIP semantic priors bridge the semantic gap and boost performance | APTOS ablation: 81.69→86.61% Acc, 0.8820→0.9124 QWK | EXPENSIVE | Ablation seed count unstated. Low priority — do not chase |
| D3 | PLKA captures multi-scale lesions the Swin backbone misses | APTOS: QWK 0.8820→0.9251 | EXPENSIVE | Note that PLKA moves QWK more than accuracy while SPM does the reverse — consistent with a decision-boundary effect, not a representation effect. Interesting but expensive |
| D4 | Errors are confined to adjacent grades, eliminating catastrophic misclassification | Confusion matrices; zero PDR→Normal | CHEAP | Under argmax on an internal split. Untested under shift |
| D5 | AUC 1.00 on the Normal class shows superior screening rule-out capability | APTOS OvR ROC | CHEAP | Grade-1 AUC is only 0.84 and their own matrix shows Mild↔Normal confusion. Over-claimed |
| D6 | High AUC values confirm the model assigns well-calibrated probabilities | Stated in §4.8 | **FLAT TECHNICAL ERROR** | **AUC is invariant to any monotone transform of the probabilities and says nothing about calibration.** P2/R5 demonstrates the opposite in the same year. Cite both together — this is a clean, citable illustration of exactly the conflation our paper is about |

**Concrete inconsistencies (verify before citing, then cite precisely):**
- APTOS confusion matrix sums to 366 images; a stated 80/20 split of 3,662 should give ~732.
- DDR class counts sum to 12,522 (Table 1) but the text states 13,673 images. Difference is plausibly the excluded ungradable class, but it is unexplained.
- PLKA is called "Progressive Lesion-aware Kernel Attention" in the text and "Parallel Large-Kernel Attention" in the Figure 1 caption.
- Baselines in Tables 2–3 are quoted from other papers, not reproduced under matched splits — so the comparison is invalid on its face, and the quoted ResNet-50 kappa (0.915) exceeds their own baseline (0.8820).
- "PyTorch 3.8.10" is a Python version.
- Inference uses only the classification head, so the ordinal head is a training regularizer. Their trade-off claim is therefore about regularization, not about the decision rule — worth stating precisely, because it is the crux.

---

## P6 — Tang et al., MMRDR multimodal dataset
*Scientific Data 2026;13:639, DOI 10.1038/s41597-026-07005-9. Data: figshare 29423747. Code: `github.com/Vladimirovich2019/MMRDR_Evaluation`*
24,460 images: 11,118 CFP, 10,404 UWF, 2,938 OCT. 5-grade DR on CFP/UWF, 7 lesion types, **3-class DME on OCT**. Internal train/test splits only.

**Verdict on the DME question: not usable for our DME head.** Three independent blockers:
1. DME labels are on **OCT**, not fundus. Our model is fundus-only.
2. Different label ontology: no DME / non-center-involving / center-involving, defined from retinal thickening and fluid on OCT. IDRiD's grade is hard-exudate-to-macula distance on fundus. Related concepts, not interchangeable.
3. The CFP and OCT subsets are **unpaired** — CFP comes from public OIA-DDR, OCT independently from the Qingdao Eye Hospital PACS. No paired cross-modal supervision is possible.

⚠️ **MMRDR-CFP is OIA-DDR in its entirety** (13,673 → 11,118 after quality control), split at the image level because OIA-DDR has no patient identifiers. If we touch DDR anywhere, MMRDR-CFP is not independent data.

**Useful anyway:** their DME distribution is 1,017 / 280 / 1,641 — the middle grade is 9.5%. Ours is 51/516 = 9.9%. **Middle-grade DME scarcity is a property of the disease, not a defect of IDRiD.** This strengthens the supervision-ceiling framing considerably. Also: inter-rater Cohen's κ ≥ 0.75 for DR and ≥ 0.80 for DME on a 130-image reference subset — a rare published human-agreement figure, and ORDER-DR explicitly laments not having one.

| ID | Claim | Their evidence | Status | Our test |
|---|---|---|---|---|
| M1 | MMRDR is unprecedented in scale and modality diversity for DR/DME | Table 1 comparison | NOT-TESTABLE | Accept. But note CFP portion is not new data |
| M2 | Fine-tuned RETFound is the strongest model on MMRDR-CFP | DR grading Acc 0.822 / F1 0.714; DME Acc 0.897 / F1 0.759 | **CONTESTED — see Contradiction 1** | ResNet-50 gets Acc 0.823 — a **tie on accuracy**, in-distribution, both fully fine-tuned |
| M3 | Ophthalmic foundation models generalize across modality far better than LVLMs | ΔAcc ≤ 5.2% (CFP→UWF) vs 29.3–36.5% for LVLMs | NOT-TESTABLE | Out of scope, record only |
| M4 | ImageNet ResNet-50 transfers to UWF slightly better than ophthalmic foundation models — described by the authors as "counter-intuitive" | Table 3 | **CONTESTED — HIGH VALUE** | Second independent instance of the Poyrazer direction, under modality shift rather than dataset shift |
| M5 | UWF is the hardest modality (>7% accuracy drop vs CFP across all model categories) | Table 3 | NOT-TESTABLE | Record only |
| M6 | Fine-tuning LVLMs on MMRDR improves them substantially over zero-shot | InternVL3-38B: 0.615 ZS → 0.781 FT | NOT-TESTABLE | Out of scope |

**Weaknesses to exploit:**
- **FLAIR contamination, unmentioned.** They benchmark FLAIR (Acc_G 0.806) on MMRDR-CFP. FLAIR's pretraining corpus includes OIA-DDR and IDRiD, and MMRDR-CFP *is* OIA-DDR. Poyrazer excluded FLAIR for exactly this reason. MMRDR reports the number with no caveat. **This is a concrete, citable contamination failure in a Nature-family data descriptor.**
- **Adaptation methods are not matched.** RETFound and ResNet-50 use full fine-tuning; FLAIR and KeepFIT use linear probing. The two groups are then compared side by side. Attribution without matching — the exact error class our paper is about.
- **AUC excluded for every model**, because extracting logits from LVLM outputs is expensive. So a foundation-model benchmark reports only accuracy and F1 — both decision-rule-dependent. No threshold-independent metric, no calibration, no confidence intervals, no seed variance.
- No external validation; no reported number of training runs.

---

## P7 — Ruamviboonsuk et al., narrative review
*Taiwan J Ophthalmol 2024;14:473–485, DOI 10.4103/tjo.TJO-D-24-00064*
Narrative review of discriminative AI, generative AI, and foundation models in retinal imaging. PubMed + Scopus, 156 records.

**Primary use: our introduction.** A peer-reviewed ophthalmology review stating that the field lacks head-to-head comparisons and that the development–deployment gap is widening is the citation that motivates a methodology paper.

| ID | Claim | Their evidence | Status | Our use |
|---|---|---|---|---|
| V1 | ViT generally outperforms CNN for retinal discriminative tasks including DR screening | Multiple cited studies | **SELF-CONTRADICTED — see V2** | |
| V2 | A head-to-head of 8 CNN and 9 ViT models for referable AMD found **all** CNNs beat **all** ViTs (CNN sensitivity and specificity ≥90%; ViT 63–94% sensitivity, 24–48% specificity) | Cited head-to-head study | CHEAP | Fourth independent instance of the pattern: an architecture claim that reverses under different evaluation conditions. The review states V1 and V2 without reconciling them |
| V3 | RETFound (pretrained on 1.6M retinal images, CFP + OCT) outperforms supervised and ImageNet-pretrained self-supervised models internally and externally | RETFound paper | CONTESTED | Feeds Contradiction 1 |
| V4 | The development–deployment gap is widening: only three ophthalmic AI devices (IDx-DR, EyeArt, AEYE-DS) had FDA approval as of April 2024, roughly one per year | Regulatory record | NOT-TESTABLE | **Cite in the introduction** |
| V5 | Four barriers to deployment: lack of head-to-head model comparisons, absent cost-effectiveness evidence, equity and bias, medicolegal liability | Review synthesis | NOT-TESTABLE | **Cite in the introduction — the first barrier is our paper's justification** |

**Resolves an earlier discrepancy:** RETFound was pretrained on 1.6M retinal images total (fundus + OCT), of which roughly 0.9M are fundus. Nielsen quoted the fundus figure, Poyrazer the total. Both are defensible; state ours precisely.

**Lead to chase:** they cite Lee et al., *Diabetes Care* 2021;44:1168–75 — a multicenter head-to-head real-world validation of seven automated DR screening systems. That is the closest existing precedent to what we are doing. Obtain and read it before writing related work.

**Second lead, from their ref 112:** Zhang et al., *NPJ Digit Med* 2024;7:108 — RETFound-enhanced community-based fundus screening with **decision curve analysis**. A decision-curve paper on RETFound is the closest published thing to our decision-rule argument. Obtain.

---

## P8 — Zhou et al., RETFound
*Nature 2023;622:156–163, DOI 10.1038/s41586-023-06555-x. Open access (CC BY). Code: `github.com/rmaphoh/RETFound_MAE` (PyTorch, authoritative) and `github.com/uw-biomedical-ml/RETFound_MAE` (Keras). Weights gated on HuggingFace — we already have access via `HF_TOKEN`.*

The primary source behind V3, and the paper that R2, N1, M2 and M4 are all arguing with. Two separate models (CFP, OCT). MAE pretraining: ImageNet-1k SSL weights → 904,170 CFPs + 736,442 OCTs (1.64M images). Downstream: full fine-tuning of the ViT-large encoder plus an MLP head.

**Design, stated precisely, because every downstream dispute turns on it:**
- **Comparators:** SL-ImageNet (supervised, ImageNet-21k, 14M), SSL-ImageNet (MAE, ImageNet-1k, 1.4M), SSL-Retinal (MAE, retinal from scratch). Authors' words: *"All models use differing pretraining strategies but have the same model architecture as well as fine-tuning processes."* Adaptation and architecture are matched. **This is the correct design and P6 should have copied it.**
- **Metrics for DR:** AUROC and AUPR only, computed per class then macro-averaged. No accuracy, no kappa/QWK, no sensitivity/specificity, no per-grade breakdown, no confusion matrix for any DR task.
- **Uncertainty:** 5 seeds varying **training-data shuffling only** (not initialisation, not the split). CI = 1.96 × SD/√5 over the 5 runs, on a fixed test set. Two-sided *t*-test between RETFound and the single most competitive comparator.
- **Model selection:** checkpoint with the highest **validation AUROC**; AUROC is also the headline reported metric.
- **Preprocessing / fine-tuning recipe:** AutoMorph background removal, resize 256 → random crop 224, horizontal flip, normalise. 50 epochs, batch 16, lr 0 → 5×10⁻⁴ over 10 warmup epochs then cosine to 1×10⁻⁶, label smoothing. Pretraining mask ratio 0.75 CFP / 0.85 OCT, batch 1,792, 800 epochs, 8×A100, ~14 days.

| ID | Claim | Their evidence | Status | Our test |
|---|---|---|---|---|
| Z1 | RETFound beats SL-ImageNet, SSL-ImageNet and SSL-Retinal on internal DR classification | AUROC 0.943 (95% CI 0.941, 0.944) APTOS-2019; **0.822 (0.815, 0.829) IDRiD**; 0.884 (0.880, 0.887) MESSIDOR-2. All *P* < 0.001 vs SL-ImageNet | **CONTESTED — scope, not correctness** | The claim is true *within its cell*: ViT-large vs ViT-large, fully fine-tuned, macro-averaged threshold-free metrics. **There is no CNN anywhere in this paper.** Every contradicting result we hold sits outside that cell. See rewritten Contradiction 1 |
| Z2 | The advantage survives cross-dataset external evaluation in all six directions among APTOS/IDRiD/MESSIDOR-2 | Fine-tuned on APTOS → IDRiD 0.822 (0.815, 0.829), MESSIDOR-2 0.738 (0.729, 0.747). *P* < 0.001 in five panels; **IDRiD → APTOS only *P* = 0.026** | CHEAP | **Adopt the six-direction design.** Ours and P1's and P2's external claims are all one-directional; theirs is not, and it costs us nothing but evaluation passes over existing checkpoints |
| Z3 | (implicit) The superiority is a general representation advantage | Contradicted by their own supplement: **internal AUPR on IDRiD is *P* = 0.81** vs SL-ImageNet while AUROC is *P* < 0.001. JSIEC AUPR *P* = 0.246. External AUPR advantages collapse to *P* = 0.019 (APTOS→IDRiD) and *P* = 0.048 (IDRiD→APTOS) | **CHEAP — HIGH VALUE** | On **our primary dataset**, RETFound's advantage is significant on overall ranking and null on minority-class ranking. That is the AUROC/AUPR dissociation, in the source paper, unremarked. Cite exactly |
| Z4 | Adapted models fall substantially against new cohorts | APTOS internal 0.943 → 0.822 (IDRiD) and 0.738 (MESSIDOR-2). Ischaemic stroke AUROC drops 0.16 (CFP) / 0.19 (OCT) UK Biobank. Authors state this plainly | CHEAP | **Agrees with R3 and with us**, and it is the strongest version because it comes from the paper the field cites *for* generalisability. Lead the introduction with it |
| Z5 | RETFound generates reliable, not overconfident, probabilities | Lowest ECE on four CFP oculomic tasks: 0.015 heart failure, 0.017 MI, 0.033 Parkinson's, 0.020 stroke. *"This verifies that RETFound generates reliable predicted probabilities"* | **CONTESTED** | ECE with **no Brier, no refinement decomposition, no discrimination metric alongside**; internal cohorts only; **never computed for any DR task**. P2/R5 uses RETFound itself as the counterexample (lowest external ECE 0.086, worst external AUC 0.697). Feeds Contradiction 3 |
| Z6 | RETFound achieves the best operating point on 3-year MI prediction | Sensitivity 0.70, specificity 0.67 (ED Table 1; TP 293, FN 123, TN 280, FP 136, *n* = 832 balanced) | CHEAP | The only place in the paper where a decision rule produces a number — and **the threshold is never stated** (presumably argmax at 0.5) and never varied. One sentence in our critique |
| Z7 | Label efficiency: RETFound reaches comparator-level performance with far fewer labels | Comparable to best comparator with **45% of data (MESSIDOR-2)** and **50% (IDRiD)**; beats all comparators at **10%** for heart failure and MI | MODERATE | Relevant to our DME ceiling but not evidence about it: this is 5-class DR and binary systemic, never a 3-class task at *n* = 516 with 51 middle-grade cases. **Do not let a reviewer use Z7 to argue our DME ceiling is a label-efficiency failure** — prepare the rebuttal now |
| Z8 | Adaptation efficiency: ~80% training-time saving for MI, 46% for MESSIDOR-2 DR | ED Fig 4, epochs to checkpoint | NOT-TESTABLE | Record only |
| Z9 | Generative SSL (MAE) beats contrastive SSL (SwAV, SimCLR, MoCo-v3, DINO) in most tasks | DINO: 0.866 (0.864, 0.869) wet-AMD, 0.728 (0.725, 0.731) stroke; MAE higher, *P* < 0.001 on DR APTOS and IDRiD | NOT-TESTABLE | **Cite as the good-practice counterexample.** They explicitly warn that *"asserting the superiority of the masked autoencoder requires caution, given the presence of several variables across all models, such as network architectures (ResNet-50 for SwAV and SimCLR, Transformers for the others) and hyperparameters."* They name the confound our whole paper is about. P6 commits the same confound silently |
| Z10 | RELPROP salience shows RETFound attends to established pathology (hard exudates, haemorrhage, parapapillary atrophy) | ED Fig 6b, selected cases | NOT-TESTABLE | Same class as S6, better executed. Do not spend compute |
| Z11 | CFP and OCT encode different information for oculomics; CFP should be retained in standard retinal assessment | Stroke better on CFP, Parkinson's better on OCT, matched splits by patient ID | NOT-TESTABLE | Out of scope. Useful one-line citation if a reviewer asks why we are fundus-only |

**The four absences that dissolve Contradiction 1.** None of these are errors; they are the boundaries of the evidence, and the field has read past all four:
1. **No CNN baseline exists in this paper.** Not ResNet, not DenseNet, not EfficientNet. RETFound's superiority is established exclusively against ViT-large under three other pretraining schemes.
2. **Frozen and linear-probed RETFound are never evaluated.** The encoder is always fully fine-tuned. N1 (frozen linear head), R1/R2 (frozen MLP head) and our own frozen result all test a regime the source paper never reported.
3. **No per-class DR results of any kind.** Macro-averaging over five grades makes R6 — grade-1 F1 = 0.000 — structurally undetectable in this paper's reporting scheme. RETFound could have had a zero-recall minority class on every DR dataset and nothing in Figs 2 or ED 2 would show it.
4. **Calibration is never reported under distribution shift** (ECE is internal-cohort only) and never for DR at all.

**Numbers audit.**
- **Verify before citing:** internal IDRiD AUROC is given as 0.822 (95% CI 0.815, 0.829) and external APTOS→IDRiD as 0.822 (95% CI 0.815, 0.829) — identical to three decimals *including the interval*. Either a main-text duplication or a genuine coincidence. Check Supplementary Table 3 before either citing the internal number or alleging an error. If it is a duplication, it is a clean seventh instance of our silent-provenance-bug class, in *Nature*.
- Discussion says the model reconstructs OCT *"despite 75% of the retinal image being masked"*; Methods give the OCT mask ratio as **0.85**. Minor, but real.
- *"Consistently outperforms"* (abstract) vs the external systemic panels: ischaemic stroke *P* = 0.202 (CFP) and *P* = 0.451 (OCT), Parkinson's OCT *P* = 0.085; external AUPR ischaemic stroke OCT *P* = 0.958, MI OCT *P* = 0.198. Three to five of eight external panels are null depending on metric.
- Pretraining arithmetic is clean (815,468 + 88,702 = 904,170; 627,133 + 109,309 = 736,442; percentages, demographics and ethnicity all sum correctly). Compute figures are internally consistent (70 min/1,000 images ≈ 1.2 h; 14 days ≈ 2 weeks). **This paper is arithmetically much cleaner than P5 or P6.** Say so.
- The 55:15:30 split ratio is stated only for the prognosis/oculomics cohorts. **The split protocol for IDRiD, APTOS and MESSIDOR-2 is not in the main text**, and patient-level splitting is asserted only for AlzEye/UK Biobank. MESSIDOR-2 is two eyes per patient. Whether CLAIM item 21 is met for the DR datasets must be resolved from Supplementary Table 1 or the repository — do not assume either way.

**The uncertainty problem, which we must handle in our own writing.** Their 95% CI is the spread of five training runs on a *fixed* test set. It contains no test-set sampling uncertainty and no patient clustering. That is why IDRiD — roughly 100 test images — carries a CI of ±0.007. **Our cluster-bootstrap CIs and theirs are not the same quantity and must never be tabulated in the same column without a footnote.** Add to PROTOCOL.md.

**Item 10 — the pattern, in inverted form.** No kappa/accuracy divergence appears here, because neither metric is reported. What appears instead is the same failure one level up: a model declared superior on macro-averaged threshold-free metrics, whose complete failure under an actual decision rule on the *same external dataset* was only discovered three years later by P2/R6. **P2 is the decision-rule audit of P8, and P8's reporting scheme could not have detected its own failure.** This is the cleanest single illustration of our thesis in the entire ledger, and it involves the field's most-cited retinal foundation model in its highest-profile venue.

**Reproducibility: reproducible downstream, not reproducible at pretraining.** Weights, code (two independent implementations), fine-tuning hyperparameters and all eight public evaluation datasets are available. MEH-MIDAS and AlzEye are controlled-access; UK Biobank requires application. For our purposes this is the good case — we can reproduce every DR claim.

**Adopt immediately (this is the operationally valuable part):**
1. **Their exact fine-tuning recipe for Phase 4.** Our "fine-tuned RETFound lost to DenseNet+EyePACS" result is currently exposed to one obvious reviewer objection — that we fine-tuned it badly. Using the authors' published recipe verbatim removes that objection permanently. Highest-value single line in this paper for us.
2. **Their pretraining-comparison design:** same architecture, same adaptation, vary only the thing under test. Cite it as the standard, then score P6 against it.
3. **Six-direction cross-dataset external evaluation** instead of one direction.
4. **Their explicit confound caveat (Z9)** as the model for how we phrase our own limitations.

**Resolution note that changes Phase 4.** RETFound's downstream input is 224×224 by construction. R1's MedSigLIP-vs-RETFound comparison at 448 vs 224 is therefore *structural*, not a Poyrazer oversight — matching resolution means running MedSigLIP **down** at 224, not RETFound up at 448 (which needs positional-embedding interpolation and is a different model). Decide this before Phase 4 launches, and state which direction we matched.

---

## P9 — Tomić et al., hand-held camera + DeepDR
*Biomedicines 2024;12:34, DOI 10.3390/biomedicines12010034. No code. Data "on request". Claim prefix: **T***
Cross-sectional instrument validation, IDF DR Screening Project, Zagreb. 160 T2DM patients / **320 eyes**. Slit-lamp fundoscopy → VISUCAM Zeiss 45° two-field → TANG hand-held two-field. Standard-camera images graded by two retina specialists; hand-held images graded by **DeepDR AI *and* an independent IDF ophthalmologist**. No external validation, no calibration, no thresholds reported.

| ID | Claim | Their evidence | Status | Our test / use |
|---|---|---|---|---|
| T1 | Hand-held camera + AI grading matches clinical examination for DR detection | AUC 0.921 (SE 0.026, 95% CI 0.870–0.973); sens 89.1% (81.3–94.4); spec 100% (93.9–100); PPV 100%; NPV 91.4%; LR− 0.11; κ 0.86±0.04; DOR 936.48; accuracy 94.9% | CHEAP | **The AI is not separable from the human grader anywhere in this paper.** See "fatal design flaw" below. The title says AI; the numbers are AI+ophthalmologist |
| T2 | Hand-held + AI matches the standard fundus camera | AUC 0.883 (0.824–0.942); sens 83.2% (74.4–89.9); spec 100%; NPV 87.3%; LR− 0.17; κ 0.78±0.05; DOR 574.6; accuracy 92.2% | CHEAP | Same flaw. Also confounds device, operator (nurse vs retina specialist) and grader in a single comparison |
| T3 | The three methods do not differ at all in detecting moderate/severe NPDR and PDR | 60/60 severe NPDR/PDR detected by every method; zero discordance | CHEAP | **True, and it is the whole point.** At the referral cut-point the system is perfect. At the "any DR" cut-point it has 89.1% sensitivity. Same model, same images, two cut-points, two verdicts |
| T4 | The entire disagreement is confined to mild NPDR (single microaneurysms) | 22/320 (6.9%) eyes vs clinical exam and 34/320 (10.6%) vs standard camera were mild NPDR called no-DR. Zero errors at any higher grade | **CHEAP — HIGH VALUE** | **Fifth independent instance of grade-1 collapse** (with R6, S5, our own, and G4 below). Every one of these papers reports it; none adjusts a threshold |
| T5 | 100% of hand-held images were gradable (77.5% good, 22.5% medium, 0% unreadable) despite an inexperienced nurse operator | AI + IDF ophthalmologist image-quality assessment | NOT-TESTABLE | Contrast sharply with G6 below (8% gradability specificity in an Indian CHC). Record for the deployment-context discussion |
| T6 | Inter-grader agreement between the two retina specialists was perfect | *"there was no case where the experts assigned different grades, there was no need for a third grader"* | **IMPLAUSIBLE — DO NOT CITE AS AGREEMENT DATA** | 320 eyes, five-grade scale, zero disagreement, no κ reported. Compare M6's published κ ≥ 0.75. Either the grades were collapsed before comparison, or the graders were not independent |

**Fatal design flaw, stated precisely so we can cite it fairly.** The hand-held arm was graded by "the AI-based automated software (DeepDR) **and** an independent IDF ophthalmologist", and no result anywhere in the paper separates the two. The paper therefore contains **no measurement of AI performance at all**, despite AI appearing in the title, abstract, keywords and conclusions. This is a clean, uncontroversial example for our critique section: a paper the field will cite as AI validation evidence (it already is — it is P10's ref 81) that measures a human-plus-AI pipeline.

**Second flaw: the reference standard is not independent.** The first author M.T. selected the patients, performed the slit-lamp examination (comparator 1), operated the standard camera (comparator 2), and was one of the two graders producing the reference standard. No masking is described. QUADAS-2 domains 2 and 3 both fail.

**Third: eye-level analysis on 320 eyes from 160 patients**, exact binomial CIs, no clustering adjustment. Same CLAIM item 21 failure as P1. With P2's Messidor-2 design effect ≈ 1.84 as a guide, the reported CIs are roughly √1.84 ≈ 1.36× too narrow.

**Fourth, and this one is worth a worked paragraph: Table 4 and Table 2 describe different samples.** From Table 2 (vs clinical exam): 140 positives, 180 negatives, TP = 118, FN = 22, FP = 0 → sensitivity **84.3%**, not the reported 89.1%. Against the standard camera: 152 positives, TP = 118 → **77.6%**, not the reported 83.2%. The confidence intervals confirm it: specificity 100% with CI 93.9–100 implies (Clopper–Pearson, zero-event) about **48 negatives**, not 180; PPV 100% with CI 95.9–100 implies about **72 predicted positives**, not 118. So the accuracy analysis ran on roughly a third of the stated sample. Meanwhile the AUC **reproduces exactly** from Table 2 — treating the hand-held grade as a 3-level ordinal score, AUC = (118×180 + 0.5×22×180)/25,200 = **0.9214**. The AUC and the sensitivity in the same paper were computed from different inputs. (The standard-camera AUC recomputes to 0.888 against a reported 0.883.) **This is instance #3 in the provenance-bug register.**

**And the specificity is a zero-cell artefact, not a result.** The rule produced literally zero false positives, so sensitivity was the only free parameter. LR+ is therefore reported as "Infinity" with a CI of "NaN–Infinity", and DOR of 936.48 requires an unstated continuity correction and carries a CI of 54.2–16,194.6 — a 300-fold range printed beside a two-decimal point estimate.

**Minor:** the discussion asserts "no discrepancy in detecting eyes with moderate NPDR", but Table 2 collapses mild and moderate into one row, so the paper contains no evidence for that specific claim. Text reports 87 men / 95 women "of 250" for a 182-patient subgroup.

**Verdict: a weak paper, and one row's worth of ledger — except that T3 + T4 together are an accidental, clean demonstration of our thesis in a clinical dataset.** Cite it for that, not for its performance numbers.

---

## P10 — Yang et al., systematic review of AI + retinal imaging for diabetes complications
*eClinicalMedicine 2025;81:103089, DOI 10.1016/j.eclinm.2025.103089. PROSPERO CRD42023493512. Open access. Claim prefix: **Y***
38 studies, AI on retinal images for **systemic** diabetes complications (DR itself explicitly excluded). Quality by Newcastle–Ottawa + AXIS; adherence scored against TRIPOD.

**Scope note: this is not a DR-grading paper.** It bears on our work in three ways only — as an introduction citation, as a documented reporting-failure exhibit, and as a source of two leads. It is not a source of comparable performance numbers.

| ID | Claim | Their evidence | Status | Our test / use |
|---|---|---|---|---|
| Y1 | Only a minority of AI-retina studies perform external validation | **12/38 (32%)** per their TRIPOD analysis | **CHEAP — cite in introduction** | But see the counting failure below. Use the TRIPOD figure (32%), which is the one they derived systematically |
| Y2 | Models validated only internally report higher performance than those validated externally, indicating overfitting and training bias | Narrative synthesis across 38 studies | **CHEAP — agrees with R3, Z4 and us** | A third independent statement of the internal→external collapse, this time as a field-level generalisation in a Lancet-family journal. **Strong introduction citation** |
| Y3 | No randomised controlled trials exist in this literature | Review finding | NOT-TESTABLE | Introduction, alongside V4/V5 |
| Y4 | Reporting of missing-data handling is nearly absent | 4/38 (11%) reported it | NOT-TESTABLE | Add to our CLAIM/TRIPOD compliance figure as a field baseline |
| Y5 | AI performance across diabetes complications spans AUC 0.676–0.971 | Table 1 synthesis | NOT-TESTABLE | Out of scope. Record only |
| Y6 | Systemic risk-factor prediction from fundus spans **AUC 0.24 to 0.97** | Narrative synthesis | **REPORTING ERROR** | An AUC of 0.24 is worse than chance and would be 0.76 with the label convention reversed. Reported without comment. Cite as an audit example, never as a performance range |
| Y7 | Incident CVD prediction AUC "as high as 0.991" | Narrative synthesis | **IMPLAUSIBLE** | Sits in the same paragraph as MACE 0.686 and C-statistics 0.75–0.77. Do not propagate this number |

**Counting failures — three incompatible external-validation counts in one paper:**
- Results: *"26 had internal validation, and **122 studies** included external validation"* — out of 38.
- TRIPOD section: *"only **12 studies (32%)** conducted external validation"*.
- Discussion: *"only **6 of them** performed the external validation"*.

Also: the abstract reports screening "from a total of **337 abstracts**", a figure that appears nowhere in the flow diagram (1536 → 593 → 360 → 127 → 38). Category counts are 4 + 10 + 8 + 17 = 39 (abstract) or 4 + 10 + 9 + 17 = 40 (Fig. 1); **neither sums to 38**. OCT-using studies are 6 in the abstract and results, 5 in the discussion. Figure 2 displays 24 + 13 = 37 studies. NOS star ratings are assigned to all 38 despite AXIS being the stated instrument for cross-sectional designs.

⚠️ **They mis-describe P8.** Their Table 1 entry for Zhou 2023 (RETFound) records the dataset as "UK Biobank and EyePACS, exact sample size not explicitly mentioned" and the validation as "**trained on EyePACS**, validated on UK Biobank and other datasets". RETFound was self-supervised on MEH-MIDAS + EyePACS, fine-tuned on MEH-AlzEye, and externally validated on UK Biobank, with every sample size stated explicitly in its Methods. **A peer-reviewed systematic review in a Lancet-family journal has the flagship paper's design wrong.** This is direct evidence for the Contradiction 1 thesis: P8's evidence boundaries are not being read, even by systematic reviewers.

**Two leads, both worth chasing:**
1. ⭐ **Lam C, Wong YL, Tang Z, et al. "Performance of artificial intelligence in detecting diabetic macular edema from fundus photography and optical coherence tomography images: a systematic review and meta-analysis." *Diabetes Care* 2024;47:304–319** (their ref 80). This is a pooled DME-from-fundus meta-analysis. **It is the single most relevant paper we have not read.** It either bounds or scoops our DME ceiling argument. Obtain first.
2. Ghenciu et al., *Biomedicines* 2024;12:2150 — oculomics review reporting non-diabetic AUC 0.70–0.95 (their ref 85). Low priority.

---

## P11 — Tahir et al., AI vs manual DR screening meta-analysis
*Front Med 2025;12:1519768, DOI 10.3389/fmed.2025.1519768. PROSPERO CRD42024596611. Claim prefix: **H***
25 observational studies, 613,690 images, 2015–2024. QUADAS-2 / AXIS / NOS. Pooled sensitivity and specificity for AI vs manual grading, split by dilated / un-dilated.

| ID | Claim | Their evidence | Status | Our test / use |
|---|---|---|---|---|
| H1 | AI screening is comparable to manual screening, with higher sensitivity | Un-dilated: AI sens 0.90 (0.85–0.94), spec 0.94 (0.91–0.96); manual sens 0.79 (0.60–0.91), spec 0.99 (0.98–0.99). Dilated: AI sens 0.95 (0.91–0.97), spec 0.87 (0.79–0.92); manual sens 0.90 (0.87–0.92), spec 0.99 (0.99–1.00) | **NOT USABLE AS EVIDENCE** | The extracted 2×2 tables are corrupted and the pooled manual estimates rest on one or two studies. See below |
| H2 | AI trades sensitivity for specificity relative to human graders across every stratum | Their own four pooled pairs: AI is higher on sensitivity and lower on specificity in both dilated and un-dilated eyes | **CHEAP — the one usable finding** | This is a **cut-point difference between AI and humans, restated four times, and never named as one.** The consistency of the direction across strata is the tell. Useful even though the magnitudes are unreliable |
| H3 | Pooled estimates are valid despite heterogeneity | I² = 95.2%, 98.1%, 99.1%, 99.6%, 98.9%, **99.9%**; they still report single pooled proportions and describe the result as "no statistically significant difference" | **METHODOLOGICAL FAILURE** | Pooling at I² ≈ 99% is not a summary, it is an average over incompatible operating points |
| H4 | AI performance is variable across studies | Their Table 2 spans **sensitivity 0.06 to 1.00** and specificity 0.50 to 1.00 within the AI arm alone | **CHEAP — HEADLINE SUPPORT** | Ting 2017 AI referable DR at sens 0.25/spec 1.00; Hansen 2015 at 0.09/0.70; Zhang 2019 at 0.99/0.99. **This is a threshold-effect spread being pooled as if it were sampling noise.** See below |
| H5 | (implicit) All pooled rows measure the same quantity | The same pooled estimate mixes **"any DR", "referable DR", "moderate and beyond", "mild DR"** and vision-threatening DR as target conditions | **CHEAP — the mechanism behind H4** | Ting 2017 and Piatti 2024 each appear twice, same system, different target condition, wildly different sensitivity — Ting 0.25 (referable) vs 0.06 (moderate+); Piatti 0.41 (mild) vs 1.00 (moderate+). **Most of the 0.06–1.00 spread is the target condition and the cut-point, not the model.** Pooling accuracy across target conditions is a category error and it is exactly a threshold confound |

**Why this paper cannot be cited for numbers — verify each of these before writing, they are all checkable from the printed tables:**

1. **The 2×2 extraction is corrupted.** Table 2's column headers read `TP FP FP TN`; the third column should be FN. In **every AI row** the two middle columns are numerically identical (e.g. Piatti `70, 102, 102, 399`), which cannot occur across thirty-odd studies. Only the manual rows have distinct FP and FN.
2. **Table 2 and the forest plots disagree on the same studies.** Ting 2017 AI referable DR: sensitivity **0.25 in Table 2**, **0.90 in Figure 7** (3057/3379). Sosale 2020 AI referable DR: specificity **0.99 in Table 2**, **0.87 in Figure 6** (153/176). Piatti mild DR: specificity **0.93 in Table 2**, **0.80 in Figure 6** (399/501). The abstract's pooled numbers come from the forest plots; Table 2 is unusable.
3. **The "pooled" manual estimate for dilated eyes comes from a single study.** Figure 6's Dataset 2 stratum contains only Ting 2017, contributing two outcome rows. The un-dilated manual stratum contains two studies contributing four rows.
4. **Non-independent outcomes are pooled as independent observations.** Ting 2017, Zhang 2019, Zhang 2022, Piatti, Sosale and Limwattanayingyong each contribute two rows to the same pooled estimate.
5. **Sensitivity and specificity are pooled separately as simple proportions**, not jointly via a bivariate or HSROC model. For diagnostic test accuracy this is the standard error, and it is *our* error: pooling the two margins independently assumes there is no cut-point trade-off between them. Their own H2 result — AI higher on sensitivity, lower on specificity, in every stratum — is the trade-off asserting itself through a model that cannot represent it.
6. Abstract un-dilated AI sensitivity is 0.90; the results text says 0.92; the multi-test section says 0.90. Un-dilated AI specificity is 0.94 in two places and 0.95 in a third. Figure cross-references in the sensitivity/specificity sections point to the wrong figures. Soto-Pedre 2015 and Li 2019 are both recorded with exactly 5,278 participants.

**What P11 is actually worth to us — and it is worth a lot, as an exhibit rather than a source.** This is the field's most recent pooled statement that AI DR screening is clinically comparable to human grading. It is built on corrupted 2×2 tables, single-study "pooled" arms, I² near 100%, and a method that structurally cannot represent the threshold effect — while its own extracted data span sensitivity 0.06 to 1.00. **This answers the question our paper implicitly raises: why does the field behave as if the decision rule does not matter? Because the meta-analytic apparatus averages it away and reports the average as a property of "AI".** That is a paragraph in our discussion, and it is well evidenced.

---

## P12 — Duggal et al., real-world validation and implementation of commercial DR AI
*JMIR Med Inform 2025;13:e67529, DOI 10.2196/67529. CTRI/2022/10/046185. STARD + iCHECK-DH. Open access. No code, no model access (proprietary cloud services). Claim prefix: **G***
Prospective validation of **three commercial cloud AI algorithms** (Leben Care, Retinal AI Diagnostic Software, SigTuple; masked as AI-1/2/3) on the same 500 eyes / 250 patients, same camera (3Netra Classic, nonmydriatic two-field 45°), same reference standard, in Indian public health settings. Then the best algorithm was integrated and deployed at a community health centre: 686 eyes / 343 patients.

**This is the most directly useful paper in the ledger for our central argument.** Read the next two rows carefully.

| ID | Claim | Their evidence | Status | Our test / use |
|---|---|---|---|---|
| G1 | Commercial DR algorithms show "variable diagnostic performance" | Across three algorithms on **identical images with an identical reference standard**: sensitivity **59.7–97.7%**, specificity **14.25–96.01%**, PPV 30.16–86.67%, NPV 85–94.34%, accuracy 37.19–88.43% | **CHEAP — HEADLINE** | AI-1 flagged **446/500 eyes (89.2%)** as DR when true prevalence was 27.5%; AI-3 flagged 106 (21.2%). Same images. **This is an 82-point specificity spread that is a cut-point spread, not a representation spread**, and the authors call it "variable performance" |
| G2 | AI-3 improved after vendor retraining between the validation and implementation phases | Sensitivity **68.4% → 99.6%**; specificity **96.0% → 64.7%**. Authors: *"The sensitivity likely improved post validation due to algorithm training"* | **CHEAP — THE HEADLINE** | **+31.2 points of sensitivity for −31.3 points of specificity.** A near-exactly compensatory move along the ROC, in a deployed clinical system, attributed to retraining. This is our claim happening in the field and being misattributed exactly as we predict. See caveats below |
| G3 | AI-3 was the best performer and was selected for deployment | Validation: spec 96.01% (93.24–97.72), sens 68.42% (59.71–76.05), PPV 86.67% (78.31–92.26), accuracy 88.43%, κ 0.65 | CHEAP | **Model selection was performed on accuracy/specificity at each vendor's own fixed cut-point.** Under matched calibration the ranking of AI-1/2/3 is untested and may reverse — the same null our Phase 0 tests |
| G4 | Deployed RDR detection is acceptable; DME detection is not | RDR: sens 78.9%, spec 98.1%, PPV 89.6%, NPV 95.7%, κ 0.81. **DME: sens 26.5%, spec 99.7%, PPV 81.8%, NPV 96%, κ 0.38** | **CHEAP — HIGH VALUE FOR THE DME ARM** | A commercially deployed fundus-based DME detector misses **73.5%** of DME. Their DME definition — *hard exudates with or without foveal involvement* — is the **same construct as IDRiD's**, unlike MMRDR's OCT ontology. **This is the first genuinely comparable external DME datapoint in the ledger.** Verified internally consistent: RS 34 DME, AI 11, TP 9 → sens 0.265, PPV 0.818, FP 2/602 → spec 0.997 ✓ |
| G5 | Referral adherence caps the clinical value of any sensitivity gain | 64 referred → 28 (43.8%) reachable by phone → **9 (14%) attended an ophthalmologist** | **NOT-TESTABLE — cite in discussion** | Whatever we recover by cut-point shifting is multiplied by 0.14 downstream. State this honestly in our own clinical-utility framing; it strengthens rather than weakens the paper |
| G6 | Image-gradability AI is unusable as a filter in this setting | Gradability sens 100%, **spec 8%**, κ 0.69. The algorithm classified **21/50 (42%) cataract eyes as gradable** that human graders rejected | CHEAP | A gradability head with 8% specificity is a constant. Relevant to our preprocessing/QC discussion |
| G7 | Human-operator learning, not model change, drove part of the improvement | κ for image quality rose 0 → 0.74 and for DR grade 0 → 0.71 over 4.5 months as the optometrist gained skill | **CHEAP — the caveat to G2** | Honest of them to report. It means G2's shift is confounded with operator skill and with a **changed reference-standard construction** between phases (see below). Our own controlled cut-point sweep is what disambiguates this — which is precisely why our experiment is worth running |

**One more row, and it may be the most important line in this ledger:**

| ID | Claim | Their evidence | Status | Our test / use |
|---|---|---|---|---|
| G8 | Agreement with the reference standard was good and improved between phases | κ for DR grade **0.65 (validation) → 0.72 (implementation)** | **CHEAP — BUILD A FIGURE AROUND THIS** | Sensitivity moved **31 points** and specificity moved **31 points** (G2), and **κ moved 0.07**. An agreement statistic was almost completely invariant to the largest decision-rule change in the paper — the change that determines who gets referred. **This is the clinical analogue of our QWK result, and it is theirs, not ours.** Nobody has to accept our framing to accept G8; it is arithmetic on their own two tables |

**Why G2 needs care, and why that helps us.** Three things changed between the two phases: the vendor retrained the model, the operator's image quality improved (G7), and the **reference standard was constructed differently** — validation used HG2 alone as RS (chosen because HG2 agreed with a senior specialist at κ = 0.85 on the disputed subset), while implementation used consensus arbitration by a third grader. So the sens/spec swap is not cleanly attributable. **State this precisely in our writing.** The argument is not "G2 proves our claim"; it is: *a near-perfectly compensatory 31-point sensitivity-for-specificity trade occurred in a deployed system, and the paper attributes it to retraining without ever considering the decision rule or reporting a threshold — and the study design cannot distinguish the two. Nobody in this literature runs the experiment that could. We do.*

**Numbers audit — one row must not be cited, the rest are sound:**
- ❌ **The DR-grade row of Table 2 is irreconcilable with Figure 3.** Table 2 gives DR sens 99.6% / spec 64.7%; Figure 3 shows RS DR-positive 184 and AI-3 DR-positive 124. Sensitivity of 99.6% on 184 positives requires ≈183 true positives, so AI-3 would have to flag at least 183 — not 124. Cross-check with RDR, which *is* consistent (sens 0.789 × 109 ≈ 86 TP, plus 0.019 × 527 ≈ 10 FP ≈ 96, against a reported 99 AI referrals ✓) and with DME, which is consistent ✓. **Use G4's RDR and DME numbers; do not cite the 99.6% DR sensitivity without resolution.**
- Text states RS gradability "336 (92.71%)"; 92.71% of 686 is 636, and Figure 3 shows 636. The 336 is a typo.
- Text states RS detected DR in "189 (28.9%)"; 28.9% of 636 is 184, and Figure 3 shows 184.
- Implementation gender: text 140 men, Table 1 141.
- Validation gender: text subgroups give 125 men / 125 women; Table 1 gives 121 / 129.
- "224 images from 56 participants (75%)" — 56/250 is 22.4%; the 75% is unexplained.
- **Eye-level analysis throughout** ("All analyses were conducted on an eye-wise basis") on 500 and 686 eyes from 250 and 343 patients, with exact binomial CIs and no clustering adjustment. Third instance of the CLAIM item 21 failure, after P1 and P9.

**Reproducibility: neither reproducible nor reimplementable.** All three algorithms are proprietary cloud services; no weights, no architectures, no training corpora, no code. We can cite G1–G7 but can never re-run any of it. That asymmetry is itself an argument for our design — **our checkpoints and decision rules are auditable and theirs are not**, and G1's 82-point specificity spread is what unauditable cut-points look like from the outside.

**Confirms the priority of a lead we already had.** They quote Lee et al., *Diabetes Care* 2021;44:1168–75: seven AI DRS systems, NPV 82.72–93.69% but **sensitivity 50.98–85.90%** in real-world head-to-head validation. That is a second independent ~35-point sensitivity spread across commercial DR algorithms, consistent with G1. **Lee et al. moves from "obtain before writing related work" to "obtain now" — G1 plus Lee is a two-source empirical base for the deployment half of our argument.**

---


---

# ⚠️ FINDING-NUMBER CROSSWALK — READ BEFORE CITING ANY "F" NUMBER

**`FINDINGS.md` is the base.** This file's `F1–F7` were assigned independently and **do not
match** the repository's numbering. Nothing is renumbered — the repository is full of
cross-references and renumbering it would be an unnecessary risk. Instead the mapping is made
explicit here, and **all new writing uses the `FINDINGS.md` numbers.**

| this file | means | actual home in the repository |
|---|---|---|
| `CLAIMS F1` | cut-point shifting recovered external minority-class recall ~5 % → ~78 % | **`FINDINGS.md` F1** (same finding, same number — the only coincidence) |
| `CLAIMS F2` | matched calibration reversed four prior attributions, both directions | **`PROTOCOL.md` §4.1** (the record table) and **`FINDINGS.md` F3** (per-class comparisons dominated by cut-point placement) |
| `CLAIMS F3` | the DME head is supervision-limited, not architecture-limited | **`FINDINGS.md` F7** |
| `CLAIMS F4` | frozen RETFound beat ImageNet; fine-tuned RETFound lost to DenseNet+EyePACS | **`FINDINGS.md` F8** |
| `CLAIMS F5` | the ensemble's development gain did not survive external validation | **`IDEAS.md` I23** and `docs/generated/ensemble_external.md` — never given an F number |
| `CLAIMS F6` | Messidor-1 has ~88 % overlap with the development pool | **`FINDINGS.md` F2** |
| `CLAIMS F7` | provenance bugs produced no runtime errors but wrong numbers | **`ISSUES.md` §24, §26, §27** (and §16, §20, §23) — never given an F number |

**Findings with no counterpart in this file**, because they postdate it:

| | |
|---|---|
| `FINDINGS.md` F4 | recalibration needs ~200 labelled local images; below 100, one attempt in four makes the model worse |
| `FINDINGS.md` F5 | macro-recall must not be primary — tuning for it costs 19.2 points of referable sensitivity |
| `FINDINGS.md` F6 | a deployment recommendation evaluated only at its mean can be harmful |
| `FINDINGS.md` F9 | the pipeline is worth **+0.234 QWK** over frozen features and a linear model |
| `FINDINGS.md` F10 | the DME grade is not recoverable from published annotations (Part A verdict) |

**Rule from 2026-09-07 onward: cite `FINDINGS.md` numbers.** When quoting this file's text,
translate the F number through the table above.

---

# OPEN CONTRADICTIONS

Resolving these is the highest-value work available. Each is a paper section.

## Contradiction 1 — What is RETFound actually worth?

**Now anchored to the source paper (P8), which makes the structure much clearer.** Five sources, apparently incompatible, each sampling a different cell:

| Source | Adaptation | Evaluation | Comparator | Metric class | Finding |
|---|---|---|---|---|---|
| **Z1/Z2** RETFound | Fully fine-tuned | Internal + cross-dataset external | **ViT-large only, no CNN** | Macro AUROC/AUPR | RETFound wins everywhere |
| **Z3** RETFound (own supplement) | Fully fine-tuned | Internal, IDRiD | SL-ImageNet | AUPR | **No advantage, *P* = 0.81** |
| **N1** Nielsen | Frozen + linear head | In-distribution, no external | Fine-tuned RETFound | AUROC | Frozen ≈ fine-tuned (*p* = 0.50 / 0.53) |
| **M2/M4** MMRDR | Fully fine-tuned | Internal (DDR), then modality shift to UWF | **ResNet-50** | Accuracy/F1 | Tie in-distribution (0.822 vs 0.823); ImageNet ResNet-50 transfers *better* under modality shift |
| **R2** Poyrazer | Frozen + MLP head | External (Messidor-2) | EfficientNet-B0 | AUC | RETFound 0.984 → 0.697, **below ImageNet** (ΔAUC −0.051) |
| **Ours** | Both | Both | DenseNet+EyePACS | Mixed | Frozen RETFound beat ImageNet features; fine-tuned RETFound lost to DenseNet+EyePACS |

**These are not in conflict at all once the axes are named — and P8 shows why.** Every contradicting result sits in a cell RETFound never tested: no CNN comparator, no frozen or linear-probed condition, no per-class metric, no calibration under shift. The reading that covers all six rows:

> RETFound's advantage is established only for full fine-tuning of ViT-large against ViT-large baselines on macro-averaged threshold-free metrics. It shrinks to nothing on minority-class ranking **in the source paper's own supplement**, ties or loses against CNNs, and reverses under frozen transfer and under modality shift.

That is a defensible, counter-intuitive, now six-source claim, and no single paper states it because each sampled one cell and generalised. **Z3 is the strongest evidence for it, because it is theirs.**

**Resolution design: a 2×2 factorial — {frozen, fine-tuned} × {internal, external}** — at matched resolution and under matched decision rules, with ImageNet **and a CNN baseline (DenseNet+EyePACS) in every cell**. Cluster-bootstrap CIs throughout. Fine-tuning follows P8's published recipe verbatim. Report macro AUROC **and** per-class AUPR/recall in every cell, so the Z3 dissociation is visible rather than averaged away.

The CNN column is not redundant with the literature: **P8 has no CNN in it at all**, and P6's CNN comparison is unmatched on adaptation. Ours would be the first matched CNN-vs-RETFound comparison across all four cells.

We are positioned to run all four cells. This is now the core of Phase 4, and it has grown from a side probe into a second headline result.

## Contradiction 1b — Is the middle DME grade scarce, or is IDRiD just small?

MMRDR built a purpose-designed 2,938-image OCT DME set and still got 9.5% in the middle grade (280/2,938). IDRiD gives us 9.9% (51/516). **Scarcity of the intermediate grade appears to be a property of the disease and of clinical sampling, not of any one dataset.** This reframes our supervision ceiling from "IDRiD is too small" to "the intermediate DME grade is intrinsically rare, so any fundus-only DME head faces this floor."

**Partially resolved by G4 (P12), which is the second source we asked for — and it is the right kind.** Unlike MMRDR, P12's DME labels are on **fundus images** under the **ICDR definition** (hard exudates with or without foveal involvement), the same conceptual basis as IDRiD, in a real screening population. Results: DME prevalence 34/638 eyes (5.3%), and a deployed commercial system achieved **sensitivity 26.5%, κ = 0.38** on DME while achieving **κ = 0.81 on referable DR in the same run, on the same images, from the same vendor**. 

That within-run contrast is the strongest form of the argument available: it is not our architecture, our dataset, or our training budget — a production system with a large proprietary corpus fails on fundus DME by the same margin, on the same eyes where it succeeds on DR. **We can now assert the reframing rather than hedging it.**

Still open: nobody has published a 3-class fundus DME dataset larger than IDRiD. **Chase Lam et al., *Diabetes Care* 2024;47:304–319 (from P10 ref 80) before writing this section** — it is a meta-analysis of AI for DME from fundus vs OCT and may already contain the modality-split evidence we are assembling by hand.

## Contradiction 2 — Is the accuracy/ordinality trade-off architectural or decision-rule?

- **D1 (Dual-SwinOrd):** architectural — a dual-head design resolves it.
- **S4 (ORDER-DR):** observes the divergence as an inherent property of two complementary metrics.
- **Ours:** it is a decision-rule phenomenon; the cut-point moves along a frontier and both papers are reading points on that frontier as properties of their models.

Note that all three papers exhibit the same pattern — the kappa-winning model is the accuracy-loser — and each explains it differently. **Three independent instances, three incompatible explanations, zero threshold sweeps.**

## Contradiction 2b — The same question, at the level of deployed systems (new, opened by P11 and P12)

Contradiction 2 is about model comparisons on benchmarks. P11 and P12 show the identical confusion happening in clinical evaluations, where it decides what gets deployed.

| Source | Comparison | What they observed | What they concluded |
|---|---|---|---|
| **H1** Tahir (P11) | AI vs human graders, 25 studies | AI more sensitive (0.90–0.95), less specific (0.87–0.94); humans less sensitive (0.79–0.90), more specific (0.99) | AI and humans have "comparable" accuracy, AI "better in sensitivity" — framed as a property of the method |
| **G1** Duggal (P12) | Three vendors, identical eyes and reference standard | Specificity 14.25%–96.01%, sensitivity 59.7%–97.74% | Vendor quality differs; select the one with the best specificity and accuracy |
| **G2/G8** Duggal (P12) | One vendor, before vs after changes | Sensitivity 68.4% → 99.6%, specificity 96% → 64.7%, **κ 0.65 → 0.72** | "Sensitivity likely improved due to algorithm training" |
| **T4** Tomić (P9) | Hand-held AI arm vs clinical exam | Specificity 100% by construction (zero false positives), all errors at the mild-NPDR boundary | Excellent test performance; LR+ = infinity |
| **Ours (F1)** | One model, cut-point moved | External minority-class recall ~5% → ~78%, model untouched; QWK penalised the fix | It is the decision rule |

**Four independent clinical evaluations, four different explanations, and not one threshold sweep among them.** G2 is the decisive row: a 31-point sensitivity gain bought with a 31-point specificity loss, in one system, described as training. G8 is the mechanism our paper explains — the agreement statistic that everyone reports barely moved (0.07) while the thing clinicians actually experience moved 31 points in each direction.

**This is now our second headline, and it is stronger than the first**, because Contradiction 2 lives on benchmarks and Contradiction 2b decides which algorithm a public health system installs.

**Cheap experiment this suggests (add to Phase 1):** take our own Pareto frontier and, at each cut-point, report QWK, κ, accuracy, sensitivity and specificity together. If κ is as flat across our frontier as it is across G2/G8, we have reproduced their result mechanistically on a model we control, with their deployed system as the external instance. **No GPU cost — this is re-analysis of prediction files we already have.**

## Contradiction 3 — Does AUC tell you anything about calibration?

Now a three-point spectrum rather than a single error, which makes it a better introduction:

- **D6 (Dual-SwinOrd, 2026):** high AUC confirms good calibration. **Flatly wrong** — AUC is invariant to any monotone transform of the scores.
- **Z5 (RETFound, 2023):** low ECE verifies reliable probabilities. **Not wrong, but insufficient** — ECE alone, no Brier, no refinement decomposition, no external cohort, and never computed for DR.
- **R5 (Poyrazer, 2026):** the direct counterexample, **using RETFound itself**: lowest external ECE (0.086) with the worst external AUC (0.697), because the probability mass is compressed at the boundary.

The arc is citable as a single paragraph: the field's flagship paper makes the weak version of the claim in *Nature*, a 2026 paper makes the invalid version, and a third 2026 paper falsifies both using the flagship's own model. Nobody reports Brier alongside ECE. **Adopt R5's standard and be explicit that we are doing so because of this arc.**

## Contradiction 4 — The two literatures share no metric (new, opened by P8)

Across P1–P8, no paper reports an ordinal-agreement metric, a threshold-free ranking metric, a calibration metric, and per-class recall on the same external evaluation.

| Paper | Reports | Never reports |
|---|---|---|
| P1 ORDER-DR | QWK, accuracy, AUPRC, Brier, ECE, referable sensitivity | AUROC |
| P2 Poyrazer | AUC, ECE, per-class F1 | QWK, accuracy |
| P5 Dual-SwinOrd | Accuracy, QWK, AUC | calibration, external anything |
| P6 MMRDR | Accuracy, F1 | AUC, calibration, CIs |
| P8 RETFound | Macro AUROC, AUPR, ECE (oculomics only) | accuracy, kappa, per-class, sensitivity for DR |

Consequence: the foundation-model literature and the architecture literature **cannot be compared to each other at all**, and neither can be compared to a clinical operating point. This is not a rhetorical point — it is why five papers can produce six incompatible verdicts on the same encoder without any of them being wrong.

**This is a free contribution.** Reporting all four families on one external evaluation costs us no additional training, and it makes our results the only commensurable point in the cluster. Add the four-family reporting requirement to PROTOCOL.md and make the cross-paper metric-coverage matrix a paper figure alongside the CLAIM compliance table.

---

# CONTAMINATION REGISTER

Running record of pretraining and dataset overlaps. Every entry must be reflected in our final contamination audit table.

| Encoder / dataset | Overlaps with | Consequence |
|---|---|---|
| **FLAIR** | Pretraining includes APTOS, **IDRiD**, OIA-DDR | Hard-excluded from all our work. IDRiD is our primary dataset. Poyrazer excluded it; **MMRDR benchmarked it on OIA-DDR anyway, unmentioned** |
| **MMRDR-CFP** | Is OIA-DDR entirely (13,673 → 11,118 after QC) | Not independent from DDR. Image-level split, no patient IDs |
| **EyePACS** | **CONFIRMED in RETFound CFP pretraining: 88,702 images, 9.8% of the 904,170-image CFP corpus** (P8 Methods). Also in MedSigLIP pipelines | Poyrazer excluded it. **Our DenseNet+EyePACS baseline and RETFound have overlapping pretraining exposure**: a "RETFound vs DenseNet+EyePACS" framing must say so. Not test contamination unless EyePACS-derived images enter any evaluation set — audit that explicitly |
| **Messidor-1** | ~88% perceptual-hash overlap with our development pool | Unusable as independent external DME validation (our own finding) |
| **RETFound — CFP** | MEH-MIDAS 815,468 (90.2%, 37,401 diabetic patients, Moorfields 2000–2022) + Kaggle EyePACS 88,702 (9.8%) | Private corpus, no overlap with our pool. Note the corpus is **diabetes-enriched**, which is a distributional advantage on DR tasks and should be stated when we discuss why RETFound might be expected to win |
| **RETFound — OCT** | MEH-MIDAS 627,133 (85.2%) + Kermany et al. 109,309 (14.8%) | Kermany includes DME-labelled OCT. Irrelevant while we stay fundus-only; blocks any future OCT DME arm |
| ✅ **IDRiD / APTOS-2019 / MESSIDOR-2 vs RETFound** | **CLEARED.** P8 Methods list all three as downstream evaluation datasets only; neither appears in the 1.64M pretraining corpus | **RETFound is contamination-clear for our primary dataset**, unlike FLAIR. This is what makes Phase 4 legitimate. Cite the Methods section explicitly in our audit table |
| **MedSigLIP** | Partly proprietary corpus; APTOS and Messidor-2 not documented but not excludable | "Contamination-aware" framing only, per Poyrazer |
| **DeepDR** (P9) | Training corpus entirely undocumented; Shanghai Jiao Tong provenance. Applied to a Croatian cohort with no overlap statement | Closed commercial system. **Not usable as a comparator or a benchmark number.** Its only value to us is as an audit exhibit |
| **Leben Care / Retinal AI / SigTuple** (P12) | Training corpora undocumented. P12 does state that *"validation images were not used for training or testing the AI"* — a rare and creditable explicit statement | Good practice on their side; still means the G1 spread cannot be attributed to any documented data difference, which is itself the point |
| **P11 study pool** (25 studies, 613,690 images) | Includes APTOS-adjacent and EyePACS-derived public sets alongside proprietary cohorts, with no overlap audit at all across the pooled corpus | **A meta-analysis with no contamination audit pooling 613,690 images.** Note as a field-level gap in our audit table |

---

# INGEST LOG

| # | Paper | Venue | Code | Read | Claims |
|---|---|---|---|---|---|
| P1 | Sheng et al., ORDER-DR | Front Endocrinol 2026 | ✅ | ✅ | S1–S7 |
| P2 | Poyrazer et al., frozen encoder benchmark | Front Med 2026 | ✅ | ✅ | R1–R8 |
| P3 | Nielsen et al., HE federated learning | J Med Imaging 2025 | ✅ | ✅ | N1–N5 |
| P4 | Mongan et al., CLAIM | Radiol AI 2020 | — | ✅ | audit instrument |
| P5 | Yu et al., Dual-SwinOrd | Bioengineering 2026 | ❌ | ✅ | D1–D6 |
| P6 | Tang et al., MMRDR | Sci Data 2026 | ✅ | ✅ | M1–M6 |
| P7 | Ruamviboonsuk et al., narrative review | Taiwan J Ophthalmol 2024 | — | ✅ | V1–V5 |
| P8 | Zhou et al., RETFound | Nature 2023 | ✅ | ✅ | Z1–Z11 |
| P9 | Tomić et al., hand-held camera + DeepDR | Biomedicines 2024 | ❌ | ✅ | T1–T6 |
| P10 | Yang et al., oculomics systematic review | eClinicalMedicine 2025 | — | ✅ | Y1–Y7 |
| P11 | Tahir et al., AI vs manual meta-analysis | Front Med 2025 | ❌ | ✅ | H1–H5 |
| P12 | **Duggal et al., real-world validation + implementation** | **JMIR Med Inform 2025** | ❌ | ✅ | **G1–G8** |

⚠️ **P7 was re-submitted in this batch and is already ingested.** Re-read against the primary source: V3 is an accurate secondary summary of P8, but its scope must be narrowed in our writing to "fully fine-tuned ViT-large, threshold-free metrics." No new claims; no new row.

**To obtain, in priority order:**
1. **P8 Supplementary Tables 1 and 3.** Needed for three separate things: the IDRiD/APTOS/MESSIDOR-2 split protocol and whether it is patient-level (CLAIM item 21); confirmation or refutation of the duplicated IDRiD 0.822 CI; and the full quantitative results behind Figs 2 and ED 2. Open access — no barrier, do this first.
2. Lee et al., *Diabetes Care* 2021;44:1168–75 — multicenter head-to-head validation of seven automated DR screening systems. Closest precedent to our design.
3. Zhang et al., *NPJ Digit Med* 2024;7:108 — RETFound real-world screening with decision curve analysis. Decision-curve analysis is decision-rule-aware; this may be the nearest published relative of our argument, and we need to know whether it scoops any part of Phase 2b.

**Missing input:** `dr-dme-campaign-brief.md` has now been referenced twice but never attached. Brief-level changes implied by P8 and P12 are listed in the P8 section, in Contradiction 1, in Contradiction 2b, and below; they are **not** yet reflected in the brief itself.

---

# OUR FINDINGS → EVIDENCE MAP

**This is the section to write the paper from.** For each of our seven findings: which claim IDs support it, which contest it, which paper each came from, and what is still missing. Nothing here is ours unless it says F.

## F1 — Cut-point shifting recovered external minority-class recall from ~5% to ~78%, model untouched. QWK actively penalised it.
- **Independent support, model-level:** **R6** (P2, Poyrazer) — grade-1 F1 = 0.000 for RETFound and EfficientNet-B0 on 270 external images, threshold never adjusted. **S5** (P1, ORDER-DR) — their threshold rule cuts severe-grade underestimation from 84.7 to 49.7 while pushing grade-1 predictions to 713 against 270 true cases.
- **Independent support, deployment-level:** **G1** (P12) — three vendors, identical eyes and reference standard, specificity 14.25%–96.01%. **G2** (P12) — one vendor, sensitivity 68.4%→99.6% against specificity 96%→64.7%. **H4** (P11) — pooled AI sensitivity 0.95 over an individual-study range of 0.06–1.00, with **H5** showing the spread is target-condition and cut-point driven.
- **Independent support, metric-invariance:** **G8** (P12) — κ moved 0.65→0.72 while sensitivity and specificity each moved 31 points. **This is the single best external citation we have**, because it is the clinical version of the QWK insensitivity we found.
- **Boundary-of-failure support:** **T4** (P9) — every disagreement in a 320-eye study sits at the no-DR / mild-NPDR boundary. **Z3** (P8) — RETFound's AUPR advantage on IDRiD is *P* = 0.81 while its AUROC advantage is *P* < 0.001.
- **Contested by:** nothing. **Six independent instances across benchmark, meta-analysis and deployment literature, and no paper reports a threshold sweep.**
- **Still missing:** our own κ-across-the-frontier curve. Add to Phase 1 (re-analysis only, no GPU).

## F2 — Matched-calibration checks reversed four prior attributions, in both directions.
- **Support:** **R4** (P2) — temperature scaling works in-distribution and fails under shift (moved external ECE by 0.004, and by **+0.002** for RETFound). **R3** (P2) — development discrimination cannot differentiate encoders (all at AUC 0.980–0.985 internally, wildly divergent externally). **Y2** (P10) — internally validated models outperform externally validated ones, a Lancet-family statement of the same thing.
- **The error we correct, in the wild:** **D6** (P5) — high AUC "confirms" calibration, flatly wrong. **Z5** (P8) — low ECE "verifies reliable probabilities," with no Brier, no external cohort, and never computed for DR. **R5** (P2) — the counterexample, using RETFound itself: lowest external ECE (0.086) with the worst external AUC (0.697).
- **Unmatched adaptation in the wild:** **M2/M4** (P6) — RETFound and ResNet-50 fully fine-tuned, FLAIR and KeepFIT linear-probed, then tabulated side by side. **Z9** (P8) — the same confound, but the authors *name it*; cite as the good-practice contrast.
- **Still missing:** nothing. This finding is well-supported. Write it.

## F3 — The DME head is supervision-limited, not architecture-limited. Six interventions failed. IDRiD gives 516 images, 51 in the middle grade.
- **Support:** **M6-adjacent** (P6, MMRDR) — a purpose-built 2,938-image DME set still yields 9.5% middle grade. **G4** (P12) — a deployed commercial system reaches **26.5% sensitivity and κ 0.38 on fundus DME** while reaching **κ 0.81 on referable DR in the same run on the same images**.
- **Why G7 matters most:** it removes every alternative explanation available to a reviewer. Not our architecture, not our dataset size, not our training budget, not IDRiD. Same vendor, same images, same run, DR works and DME does not.
- **Contested by:** **Z7** (P8) — RETFound's label efficiency (matches comparators with 45–50% of labels; beats them at 10% for systemic tasks). **Prepare the rebuttal now:** Z7 is 5-class DR and binary systemic prediction, never a 3-class task at *n* = 516 with 51 middle-grade cases, and never on DME.
- **Still missing:** **Lam et al., *Diabetes Care* 2024;47:304–319** — a fundus-vs-OCT DME meta-analysis. Obtain before writing this section; it may already answer the modality question or, worse, scoop part of it.

## F4 — Frozen RETFound features beat ImageNet features; fine-tuned RETFound lost to DenseNet+EyePACS.
- **Support:** **R2** (P2) — frozen RETFound falls below the ImageNet baseline externally (ΔAUC −0.051, *p* = 0.016). **M4** (P6) — ImageNet ResNet-50 transfers better than ophthalmic foundation models under modality shift. **M2** (P6) — RETFound 0.822 vs ResNet-50 0.823, a tie in-distribution.
- **Contested by:** **Z1/Z2** (P8) and **V3** (P7), the source claim. **N1** (P3) — frozen linear head ≈ fine-tuned, in-distribution only.
- **How the conflict dissolves (from the P8 audit):** RETFound has **no CNN baseline anywhere in the paper**, never evaluates a frozen or linear-probed encoder, reports no per-class DR metric, and never calibrates under shift. Every contradicting result sits in a cell the source paper did not test. See Contradiction 1.
- **Contamination status: clear.** P8's Methods list IDRiD, APTOS-2019 and MESSIDOR-2 as downstream evaluation only. EyePACS *is* in RETFound's CFP pretraining (88,702 images, 9.8%) — state this when framing a DenseNet+EyePACS comparison.
- **Still missing:** the four-cell 2×2 with a CNN in every cell, at matched resolution, using P8's published fine-tuning recipe verbatim. That recipe is our inoculation against the "you fine-tuned it badly" objection. Phase 4.

## F5 — An ensemble's development gain did not survive external validation.
- **Support:** **R7** (P2) — fold ensembling does not rescue external performance. **Z4** (P8) — APTOS internal 0.943 → 0.822 (IDRiD) and 0.738 (MESSIDOR-2); ischaemic stroke AUROC drops 0.16 (CFP) / 0.19 (OCT) on external cohorts, stated by the authors. **Y2** (P10). **R3** (P2).
- **Contested by:** nothing.
- **Still missing:** nothing. This is a two-sentence result with four citations.

## F6 — Messidor-1 has ~88% overlap with our development pool.
- **Support for the practice, not the number:** **P2's** contamination audit table and their hard exclusion of FLAIR (pretraining includes APTOS, **IDRiD**, OIA-DDR). **M-weakness** (P6) — MMRDR benchmarks FLAIR on MMRDR-CFP, which *is* OIA-DDR, with no caveat, in a Nature-family data descriptor.
- **New field-level evidence:** the P11 pool aggregates 613,690 images across 25 studies **with no contamination audit of any kind**.
- **Still missing:** nothing. Our audit table is the contribution; these are the exhibits.

## F7 — Six provenance bugs produced no runtime errors but silently generated wrong numbers.
This finding was previously unsupported from outside. **It now has four published exhibits, all worked out in this ledger with the recomputation shown.**

| # | Source | The bug | Why it is silent |
|---|---|---|---|
| 1 | **P5** (Dual-SwinOrd) | APTOS confusion matrix sums to 366; an 80/20 split of 3,662 gives ~732. DDR class counts sum to 12,522 against a stated 13,673. PLKA named two different things in text and figure | Every individual number is plausible |
| 2 | **P8** (RETFound, *Nature*) | Internal IDRiD AUROC and external APTOS→IDRiD AUROC are identical to three decimals **including the confidence interval** (0.822, CI 0.815–0.829) | A coincidence is possible; verify against Supplementary Table 3 before asserting |
| 3 | **P9** (Tomić) | Table 4's accuracy statistics cannot be derived from Table 2's confusion matrix. Table 2 gives sensitivity 84.3%; they report 89.1%. Reported CIs imply ~48 negatives and ~72 predicted positives against 180 and 118 in the table. **The AUC reproduces exactly (0.9214); the sensitivity does not** | Two tables in one paper, computed from different inputs, both internally plausible |
| 4 | **P11** (Tahir) | **The false-negative column of Table 2 is a copy of the false-positive column** — header reads `TP FP FP TN`, values identical in all 37 rows. Every sensitivity in the table is wrong; the forest plots use different, correct values (Ting: table 0.25 vs plot 0.90; Sosale: 0.84 vs 0.99; Abràmoff: 0.67 vs 0.97) | No value is out of range; specificity reconciles perfectly, so the table looks self-consistent |

**Write this as its own section.** The argument is no longer "we made mistakes and caught them" — it is "this failure mode is endemic, here are four published instances across *Nature*, *Frontiers* and MDPI, here is the recomputation for each, and here is the consumption-manifest system that catches it." That reframes F7 from a confession into a contribution.

---

# WHAT CHANGES IN THE CAMPAIGN BRIEF

Consolidated from P8 and P12. None of this is in `dr-dme-campaign-brief.md` yet.

1. **Phase 1 gains one free experiment (G8).** At every point on our Pareto frontier, report QWK **and** Cohen's κ **and** accuracy **and** sensitivity **and** specificity together. If κ is as flat across our frontier as it was across P12's 31-point swing (G2 vs G8), we reproduce their deployment result on a model we control. Re-analysis of existing prediction files; zero GPU.
2. **Phase 2b gets a written form (G1).** Report the full spread across systems at fixed inputs and a fixed reference standard, not the winner. That is P12's design and it is the right presentation for our threshold-recovery result.
3. **Phase 4 uses P8's published fine-tuning recipe verbatim** — 50 epochs, batch 16, lr 5×10⁻⁴ cosine to 1×10⁻⁶, 10 warmup epochs, label smoothing, AutoMorph preprocessing, 224×224. This removes the only serious objection to F4.
4. **Phase 4 adds a CNN comparator in every cell**, now demonstrably novel: P8 contains no CNN at all, and P6's CNN comparison is unmatched on adaptation.
5. **Phase 4 adds six-direction cross-dataset external evaluation** (APTOS ↔ IDRiD ↔ MESSIDOR-2), as P8 did. Evaluation passes only.
6. **Resolution matching is decided downward.** RETFound's downstream input is 224 by construction, so matching R1's MedSigLIP comparison means running MedSigLIP at 224, not RETFound at 448. State which direction we matched.
7. **PROTOCOL.md additions:** (a) report all four metric families — ordinal agreement, threshold-free ranking, calibration, per-class recall — on every external evaluation, per Contradiction 4; (b) never tabulate our cluster-bootstrap CIs beside P8-style seed-variance CIs without a footnote, they are different quantities; (c) adopt iCHECK-DH alongside CLAIM for any deployment-facing claim.

---

**Acquisition queue, in priority order:**
1. **Lam et al., *Diabetes Care* 2024;47(2):304–319** — AI for DME from fundus vs OCT, systematic review and meta-analysis. Bears directly on F3 and Contradiction 1b. **Get this first.**
2. **P8 Supplementary Tables 1 and 3** — split protocol and patient-level status for IDRiD/APTOS/MESSIDOR-2, and the duplicated-CI check. Open access.
3. **Lee et al., *Diabetes Care* 2021;44:1168–1175** — seven systems head to head; we now know it reports NPV 82.72–93.69% against sensitivity 50.98–85.90%, which is the G1 signature. Get the primary.
4. Zhang et al., *NPJ Digit Med* 2024;7:108 — RETFound real-world screening with decision curve analysis.

*Awaiting further papers.*
