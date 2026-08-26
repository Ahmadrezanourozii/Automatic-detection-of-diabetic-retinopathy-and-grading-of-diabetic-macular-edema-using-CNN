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
| I07 | **Macula-centred crop for the DME branch**, using IDRiD's fovea coordinates. | untested | The DME grade *is* defined by exudate distance to the macula centre, and we have that centre for all 516 IDRiD images. GAP over a 512×512 whole-fundus feature map discards exactly the positional information the label depends on. Strongest architectural idea available, and it is clinically motivated rather than a hyper-parameter. |
| I03 | Augmentation: rotation, flips, scale, brightness/contrast jitter, mixup/CutMix. | untested | Cheapest known lever given the DME data volume. Note the prior notebook explicitly turned augmentation **off** to help memorisation. |
| I06 | **The green-channel question.** green×3 vs [green, CLAHE(green), grey] vs full RGB vs RGB with CLAHE on LAB's L channel. | untested | The thesis' 10-point ablation claim for the green channel has no run behind it (`ISSUES.md` §1). This tests a load-bearing assumption. Feeding one replicated channel into a 3-channel ImageNet-pretrained network discards two thirds of the pretrained first-layer filters. |

## Tier 2

| ID | Idea | Status | Notes |
|---|---|---|---|
| I05 | Backbone comparison: EfficientNet, ConvNeXt, Swin, retinal foundation model (RETFound). | untested | Do not start before E01. |
| I08 | Loss weighting: α/β sweep, then uncertainty-based task weighting. | untested | Low expected value until the heads work at all. |
| I09 | Hard vs soft gating of the DME branch; cost of DR-head errors propagating. | untested | See `PROTOCOL.md` §5.1 — the gate discards nothing, so this is about metric definition more than architecture. |
| I10b | **Acquire a full-resolution Messidor-2 mirror** | **queued, high value** | Our mirror is 512 px and holds 270 of the 295 Mild images, the class that carries most of the error. Microaneurysms are 1–2 px at 448. Until this exists, the Mild class is capped by the data, not the model. |
| I10 | Resolution: 512 vs 640 vs 768, traded against batch size and quota. | untested | **Confounded on Messidor-2**, which is already 512 px (`ISSUES.md` §4). Run on IDRiD + APTOS only. |
| I12b | **Acquire Messidor-1** — the only realistic external test for 3-class DME | **queued, not started** | Same clinical definition as IDRiD. Needs a duplicate check against our Messidor-2 mirror first (`ISSUES.md` §3). Until this exists there is no external DME number, and that is a stated limitation. |
| I14 | Grad-CAM on correct and incorrect cases. | untested | Addresses the interpretability gap *and* works as a bug detector: if the model attends to the optic disc rather than to lesions, something is wrong. Cheap. |
| I11 | Repeated cross-validation on the final candidates, mean ± interval. | untested | Mandatory before any headline number, given §7 of the protocol. |
| I15 | Auxiliary hard-exudate segmentation head from IDRiD's 81 masks. | untested | Small n, but the masks are exactly the lesion the DME label is about. Pairs naturally with I07. |
| I16 | Re-implement the two comparison baselines (SVM, single-output CNN) on our exact split. | untested | Makes the thesis' comparison table a fair one. Currently it compares numbers from different datasets and different protocols. |

| I17 | Cross-fitted decision cut-point tuning | **inconclusive → demoted** | Worth +0.047 QWK on E05, worth +0.008 (n.s.) on E06. It repairs a miscalibrated output layer; pretraining repairs the same thing better. Kept as a diagnostic, not part of the reported pipeline. |

## Rejected

| ID | Idea | Verdict | Evidence |
|---|---|---|---|
| R01 | `load_idrid_all()` / `--no-holdout` / `memorize=True` — train and evaluate on the same 516 images. | **rejected** | Produces training accuracy quoted as test accuracy. Its own docstring states the goal is "to show the highest possible accuracy numbers rather than to measure generalization". Delete the code path. `ISSUES.md` "Things not to redo". |
| R02 | Messidor-1 for the **DR** head. | **rejected** | 4-level lesion-count scale, not ICDR; no clean mapping to 5 classes. Use it for DME only, where the definition is identical. `data/LABEL_MAPPING.md`. |
| R03 | Flattening IDRiD's 3-class DME to binary to match Messidor-2. | **rejected** | Discards the 3-class grading that is the thesis' stated contribution. Use partial labels instead. |
