# P8 — Zhou et al., RETFound, *Nature* 2023 — verified against the full text

DOI [10.1038/s41586-023-06555-x](https://doi.org/10.1038/s41586-023-06555-x). Full text
retrieved from **PubMed** Central (PMC10550819); PMID 37704728;
*Nature* 2023;622(7981):156–163.

## Verified

| ID | claim as recorded | paper says | verdict |
|---|---|---|---|
| **Z4** | APTOS internal 0.943 → 0.822 (IDRiD) and 0.738 (MESSIDOR-2) | internal: *"AUROC of 0.943 (95% CI 0.941, 0.944), 0.822 … and 0.884 … respectively, on Kaggle APTOS-2019, IDRID and MESSIDOR-2"*; cross-evaluation: *"when fine-tuned on Kaggle APTOS-2019, RETFound achieved AUROC of 0.822 (95% CI 0.815, 0.829) and 0.738 (95% CI 0.729, 0.747) … on IDRID and MESSIDOR-2"* | **VERIFIED exactly** |
| **contamination** | EyePACS is in RETFound's CFP pretraining, 88,702 images, 9.8 % | *"88,702 (9.8%) CFPs are Kaggle EyePACS"* of 904,170 total | **VERIFIED exactly** |
| **contamination** | IDRiD, APTOS-2019, MESSIDOR-2 are downstream evaluation only | pretraining is MEH-MIDAS + EyePACS; the three appear only under "Data for ocular disease diagnosis" | **VERIFIED** |
| **no CNN baseline** | the paper has no CNN comparator | comparators are SL-ImageNet, SSL-ImageNet, SSL-Retinal, and *"All models … have the same model architecture"* (ViT-Large) | **VERIFIED, with a nuance** — ResNet-50 does appear, but only as the backbone of the SwAV/SimCLR **SSL-strategy** variants, never as a CNN transfer-learning baseline |
| **no frozen / linear-probe condition** | never evaluated | adaptation is described only as fine-tuning the encoder plus an MLP head; no frozen or linear-probe arm | **VERIFIED (absence)** |
| **Z5**, in part | ECE reported without Brier, and never computed for DR | calibration analysis is described **only for the oculomic tasks**: *"we also conducted calibration analyses for prediction models in oculomic tasks"*, reporting *"the lowest expected calibration error in the reliability diagram"*. No Brier score anywhere; no calibration for diabetic retinopathy | **VERIFIED for "no Brier" and "never for DR"** |
| **fine-tuning recipe** (needed for any fair re-test) | 50 epochs, batch 16, 10 warm-up epochs, cosine schedule, label smoothing, AutoMorph, 224×224 | *"The batch size is 16. The total training epoch is 50 and the first ten epochs are for learning rate warming up … followed by a cosine annealing schedule"*; label smoothing; AutoMorph preprocessing; resize 256×256 then crop 224×224 | **VERIFIED** (the exponents on the learning rates were lost in text extraction; take them from the PDF before quoting numerically) |

## ⚠️ The *Nature* anomaly — the observation is VERIFIED, the diagnosis is NOT

The ledger's fourth provenance exhibit claims: *"Internal IDRiD AUROC and external APTOS→IDRiD
AUROC are identical to three decimals including the confidence interval (0.822, CI
0.815–0.829)"*, and adds — correctly — *"a coincidence is possible; verify against
Supplementary Table 3 before asserting."*

**The identity is confirmed from the published main text, without needing the supplementary
table.** Both figures appear in the same paragraph pair:

* internal, model fine-tuned and evaluated on IDRiD: **0.822 (95 % CI 0.815, 0.829)**
* external, model fine-tuned on APTOS-2019 and evaluated on IDRiD: **0.822 (95 % CI 0.815, 0.829)**

Two different models — one trained on IDRiD, one trained on APTOS — evaluated on IDRiD, are
reported with the same point estimate *and* the same interval to three decimal places.

**What is established and what is not.** The identity is a fact about the published text and
can be stated. **Whether it is a transcription error or a genuine coincidence cannot be
determined from the paper**, and this record does not claim it is an error. Each figure is the
mean of five seeds with a CI derived as 1.96 × standard error, so an exact three-decimal
collision across two independent training regimes is improbable but not impossible.

**How the thesis must phrase it:** *"the internal IDRiD and the APTOS→IDRiD external AUROC are
reported identically, 0.822 (95 % CI 0.815–0.829); we could not determine from the published
material whether this reflects a coincidence or a transcription error."* **Do not assert an
error in a *Nature* paper on this evidence.** Resolving it needs Supplementary Table 3.

## Still UNVERIFIED — needs Supplementary Table 3

| ID | claim | why |
|---|---|---|
| **Z3** | IDRiD AUPR advantage *P* = 0.81 against AUROC *P* < 0.001 | not in the main text. **And the main text points the other way**: *"The AUPR results of RETFound were also significantly higher than the compared groups"*. Until the supplementary table is read, **Z3 must not be cited** — the main text appears to contradict it |
| **Z7**, in part | *"matches comparators with 45–50 % of labels"* | the main text supports only the 10 % claim for heart failure: *"RETFound outperformed the other pretraining strategies using only 10% of labelled training data"*. The 45–50 % figure is not in the main text |
| **per-class DR metrics** | *"reports no per-class DR metric"* | the main text says per-class AUROC/AUPR **are computed then averaged**: *"we calculate the AUROC and AUPR for each disease category and then average them"*. So they exist internally; whether they are tabulated in the supplement is unknown. **The claim should be narrowed to "the main text reports only macro-averaged figures"** |

## Consequence for our own F8

`FINDINGS.md` F8 contrasts frozen RETFound (better than ImageNet) with fine-tuned RETFound
(worse than DenseNet+EyePACS). This verification strengthens the framing in two ways:

1. **The EyePACS overlap is confirmed at 9.8 % of RETFound's CFP pretraining.** Our comparator
   is DenseNet pretrained on EyePACS, so the two share pretraining data. F8 already states the
   confound; it can now cite the exact figure.
2. **RETFound's paper contains no frozen condition and no CNN baseline**, so both of our
   comparisons sit in cells the source paper never tested. That is not a contradiction of the
   source; it is a different question, and saying so precisely is stronger than claiming to
   have refuted anything.
