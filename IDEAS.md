# IDEAS.md — backlog with verdicts

Status ∈ {untested, queued, running, confirmed, rejected, inconclusive}.
An idea that is neither converted into an experiment nor explicitly rejected is not finished.
Rejected ideas keep the evidence that rejected them.

Re-ranked 2026-08-25 against what the audit found. The ranking changed a lot from the
backlog in the project prompt, for one reason: **there is no working baseline**, so ideas
that improve a working model are worth nothing until one exists.

---

## Tier 0 — must happen before anything else

| ID | Idea | Status | Cheapest falsifying test | Prediction |
|---|---|---|---|---|
| E01 | The representation carries DR signal; the prior collapse was an optimisation failure. | **queued** | Frozen DenseNet121/ImageNet features at 224 px on a 2 000-image subset → logistic regression. CPU, minutes. | QWK ≫ 0 and accuracy well above the 33 % floor. If not, the problem is preprocessing or resolution, not the training loop, and the whole plan changes. |
| I00 | Trivial baselines, computed and recorded once. | **queued** | Majority class per head; logistic regression on frozen features; plain RGB DenseNet121 with no preprocessing chain. | Already partly known: floors are 33.0 % (DR, IDRiD test), 69.6 % (DME gated), 47.1 % (DME ungated). |
| I01 | The split unit matters. | **queued** | pHash + vessel-descriptor clustering across all corpora; then the same pipeline under group split vs naive image split. | A real gap. Expect it to be smaller than the 25-point gap seen in the prior Parkinson's project, but non-zero — and it explains part of the distance to published headline numbers. |

## Tier 1 — largest expected gain per GPU-hour

| ID | Idea | Status | Notes |
|---|---|---|---|
| I04 | **More data.** EyePACS pretraining → fine-tune for DR. Messidor-2 partial labels for DME (`data/LABEL_MAPPING.md`). | untested | Almost certainly the biggest single lever. Takes the DME training set from 348 to 2 092 images. Attach EyePACS on Kaggle rather than downloading. |
| I02 | **Ordinal treatment of DR.** CORAL/CORN head, or label smoothing over neighbouring grades. | untested | Aimed exactly at the adjacent-grade error mode. Also the right thing to do given QWK is the primary metric. |
| I07 | **Macula-centred crop for the DME branch**, using IDRiD's fovea coordinates. | untested | The DME grade *is* defined by exudate distance to the macula centre, and we have that centre for all 516 IDRiD images. GAP over a 512×512 whole-fundus feature map discards exactly the positional information the label depends on. Strongest architectural idea available, and it is clinically motivated rather than a hyper-parameter. |
| I03 | Augmentation: rotation, flips, scale, brightness/contrast jitter, mixup/CutMix. | untested | Cheapest known lever given the DME data volume. Note the prior notebook explicitly turned augmentation **off** to help memorisation. |
| I06 | **The green-channel question.** green×3 vs [green, CLAHE(green), grey] vs full RGB vs RGB with CLAHE on LAB's L channel. | untested | The thesis' 10-point ablation claim for the green channel has no run behind it (`ISSUES.md` §1). This tests a load-bearing assumption. Feeding one replicated channel into a 3-channel ImageNet-pretrained network discards two thirds of the pretrained first-layer filters. |

## Tier 2

| ID | Idea | Status | Notes |
|---|---|---|---|
| I05 | Backbone comparison: EfficientNet, ConvNeXt, Swin, retinal foundation model (RETFound). | untested | Do not start before E01. |
| I08 | Loss weighting: α/β sweep, then uncertainty-based task weighting. | untested | Low expected value until the heads work at all. |
| I09 | Hard vs soft gating of the DME branch; cost of DR-head errors propagating. | untested | See `PROTOCOL.md` §5.1 — the gate discards nothing, so this is about metric definition more than architecture. |
| I10 | Resolution: 512 vs 640 vs 768, traded against batch size and quota. | untested | **Confounded on Messidor-2**, which is already 512 px (`ISSUES.md` §4). Run on IDRiD + APTOS only. |
| I12 | **External validation on a corpus never seen in training.** | untested | The single most credible thing that can be added to this thesis. Messidor-1 for DME, DDR/DeepDRiD for DR. |
| I14 | Grad-CAM on correct and incorrect cases. | untested | Addresses the interpretability gap *and* works as a bug detector: if the model attends to the optic disc rather than to lesions, something is wrong. Cheap. |
| I11 | Repeated cross-validation on the final candidates, mean ± interval. | untested | Mandatory before any headline number, given §7 of the protocol. |
| I13 | Test-time augmentation and ensembling. | untested | Last, and only if there is a real model to ensemble. |
| I15 | Auxiliary hard-exudate segmentation head from IDRiD's 81 masks. | untested | Small n, but the masks are exactly the lesion the DME label is about. Pairs naturally with I07. |
| I16 | Re-implement the two comparison baselines (SVM, single-output CNN) on our exact split. | untested | Makes the thesis' comparison table a fair one. Currently it compares numbers from different datasets and different protocols. |

## Rejected

| ID | Idea | Verdict | Evidence |
|---|---|---|---|
| R01 | `load_idrid_all()` / `--no-holdout` / `memorize=True` — train and evaluate on the same 516 images. | **rejected** | Produces training accuracy quoted as test accuracy. Its own docstring states the goal is "to show the highest possible accuracy numbers rather than to measure generalization". Delete the code path. `ISSUES.md` "Things not to redo". |
| R02 | Messidor-1 for the **DR** head. | **rejected** | 4-level lesion-count scale, not ICDR; no clean mapping to 5 classes. Use it for DME only, where the definition is identical. `data/LABEL_MAPPING.md`. |
| R03 | Flattening IDRiD's 3-class DME to binary to match Messidor-2. | **rejected** | Discards the 3-class grading that is the thesis' stated contribution. Use partial labels instead. |
