# Verification of P1, P3, P5, P6, P7, P9, P10 — against source, 2026-09-08

Scope as instructed: DOI and bibliographic reference for all seven; **only the claims relied on
in the ledger's "OUR FINDINGS → EVIDENCE MAP"**. Anything not confirmed is tagged UNVERIFIED
and must not be cited while tagged. Nothing is deleted.

**Sourcing.** We hold no PDFs. All seven were read against **PubMed Central full text** or the
publisher's open-access full text, which for these journals is the same document as the PDF.
MDPI and Nature returned HTTP 403 / an auth redirect to the fetcher; both were read through PMC
instead. Article metadata and full text retrieved **according to PubMed**.

---

## Existence and citation — all seven confirmed

Resolved by DOI content negotiation, first author / journal / volume / year matching.

| | paper | citation as confirmed at source | DOI |
|---|---|---|---|
| P1 | Sheng, Dong, Wu, Liang | *Front Endocrinol* 2026;**17**:1923216 | [10.3389/fendo.2026.1923216](https://doi.org/10.3389/fendo.2026.1923216) |
| P3 | Nielsen, Wilms, Forkert | *J Med Imaging* 2025;**12**(3):034504 | [10.1117/1.JMI.12.3.034504](https://doi.org/10.1117/1.JMI.12.3.034504) |
| P5 | Yu, Si, Zhong | *Bioengineering* 2026;**13**(4):374 | [10.3390/bioengineering13040374](https://doi.org/10.3390/bioengineering13040374) |
| P6 | Tang, Wang, Guo *et al.* (10 authors) | *Sci Data* 2026;**13**(1) | [10.1038/s41597-026-07005-9](https://doi.org/10.1038/s41597-026-07005-9) |
| P7 | Ruamviboonsuk *et al.* | *Taiwan J Ophthalmol* 2024;**14**(4):473–485 | [10.4103/tjo.TJO-D-24-00064](https://doi.org/10.4103/tjo.TJO-D-24-00064) |
| P9 | Tomić *et al.* | *Biomedicines* 2024;**12**(1):34 | [10.3390/biomedicines12010034](https://doi.org/10.3390/biomedicines12010034) |
| P10 | Yang *et al.* (14 authors) | *eClinicalMedicine* 2025;**81**:103089 | [10.1016/j.eclinm.2025.103089](https://doi.org/10.1016/j.eclinm.2025.103089) |

**⚠️ The structural question is answered: no 2026 paper is missing.** P1, P2, P5 and P6 all
exist, are indexed, and their load-bearing numbers are readable. **The literature-review
structure does not have to change.**

**Citation corrections to carry:**
* **P9** is stamped **2023-12-22** online in *Biomedicines* **12**(1):34 — volume 12 is the 2024
  volume. Cite as the publisher lists it (2024, vol 12, issue 1); the 2023 date is online-first.
* **P5** has **three** authors (Yu, Si, Zhong), **P6 ten**, **P10 fourteen**. Get these right in
  the bibliography; the ledger does not list them.

---

## P1 — Sheng *et al.*, ORDER-DR — **VERIFIED**

Confirmed: dual-branch ordinal framework, APTOS-2019 development, Messidor-2 external
(1,744 gradable images).

| claim | ledger | at source | verdict |
|---|---|---|---|
| **S5** severe-grade underestimation | 84.7 ± 8.5 → 49.7 ± 6.5 | Table 12: **84.7 ± 8.5** (default argmax) → **49.7 ± 6.5** (validation-calibrated) | **VERIFIED exactly** |
| **S5** the other side of the same rule | grade-1 predictions 713 against 270 true | Table 12: **712.7 ± 164.1** predicted; support **270** | **VERIFIED** (713 is the rounded mean; quote 712.7 ± 164.1) |
| **S5** grade-1 precision | 0.193 | Table 11: **0.193** (95 % CI 0.169–0.217) | **VERIFIED exactly** |
| **S7** external calibration decay | ECE 0.049 ± 0.001 → 0.160 ± 0.008 | **0.049 ± 0.001** (APTOS) → **0.160 ± 0.008** (Messidor-2) | **VERIFIED exactly** |
| **S1/S3** external QWK | ORDER-DR 0.6423 ± 0.0364; LORS-384 0.5725 ± 0.0184; Swin-Tiny 0.5628 ± 0.0255 | Table 6: **0.6423 ± 0.0364**, **0.5725 ± 0.0184**, **0.5628 ± 0.0255** | **VERIFIED exactly** |
| **S1** the CB-384 comparator | 0.5534 ± 0.0387 | the retrieved Table 6 row reads *EfficientNet-B0 balanced CE* **0.5047 ± 0.0131** | ⚠️ **UNCONFIRMED — do not cite this one number** |

**S5 is the strongest single external support for F1** and it verifies exactly, in both
directions: the same threshold rule that halves severe-grade underestimation inflates grade-1
predictions to 2.6× the true count at precision 0.193. That is our finding stated by another
group without their drawing our conclusion from it.

---

## P3 — Nielsen *et al.* — **VERIFIED, with the ledger's numbers CORRECTED**

Confirmed: 6,457 images from APTOS-2019 + ODIR-5K; frozen RETFound features → a **5,125**-parameter
multiclass logistic-regression head; full RETFound **303.3 M** trainable parameters; **five**
random seeds; **one-way ANOVA** at 0.05.

**⚠️ N1's numbers in the ledger are wrong, and wrong in a direction that matters.**

| | ledger says | at source | |
|---|---|---|---|
| APTOS, frozen head | 0.94 ± 0.01 | **0.93 ± 0.01** | corrected |
| APTOS, fully fine-tuned | 0.94 ± 0.02 | **0.94 ± 0.01** | corrected |
| ODIR-5K, frozen head | 0.80 ± 0.02 | **0.78 ± 0.02** | corrected |
| ODIR-5K, fully fine-tuned | 0.79 ± 0.01 | **0.80 ± 0.02** | corrected |
| *p* values | 0.50, 0.53 | **0.50**, **0.53** | verified |

The ledger has the frozen head *equal on APTOS and ahead on ODIR-5K*. **At source the frozen
head is nominally behind on both.** The *p* values still say indistinguishable, so N1's substance
— a 5,125-parameter head on frozen features matches a 303.3 M-parameter fine-tune — survives.
But the direction the ledger implies does not, and our F4 discussion must not lean on it.

**N1 is in-distribution only, and the authors say so.** No external validation is performed. From
their limitations: *"evaluating the out-of-distribution performance on datasets external to the
training set would offer valuable insights into the robustness of our approach."* **VERIFIED.**
This is the sentence that dissolves the apparent conflict with our F8, and it is theirs.

---

## P5 — Yu *et al.*, Dual-SwinOrd — **D6 VERIFIED VERBATIM**

The claim was that this paper treats a high AUC as evidence of calibration. It does, in one
sentence, §4.8:

> *"Overall, the high AUC values across both datasets confirm that Dual-SwinOrd is not only
> accurate in prediction but also assigns well-calibrated probabilities, making it highly
> reliable for clinical decision support."*

**The paper reports no calibration metric of any kind** — no ECE, no Brier score, no reliability
diagram. Its stated metrics are Accuracy, AUC (one-vs-rest) and QWK. **AUC is invariant to any
monotone transformation of the scores and therefore carries no information about calibration**;
a model can hold that AUC and be arbitrarily miscalibrated. **VERIFIED as a published instance of
the exact error, in a 2026 peer-reviewed paper.**

**No external validation either.** Both benchmarks are trained on: APTOS-2019 under an 80/20
random split, DDR under its official split. Headline results: APTOS accuracy **87.98 %**, QWK
**0.9370**; DDR accuracy **86.54 %**, QWK **0.9040**. **VERIFIED.**

Two further exhibits, both at source:
* Mild DR (grade 1) is the acknowledged failure class — AUC **0.84** on APTOS, and the discussion
  concedes *"the differentiation of Mild DR (Grade 1) from Normal (Grade 0) remains a bottleneck
  (AUC ≈ 0.81 on DDR)"*. **Fifth independent instance of the mild-grade collapse.**
* The APTOS split is *"a randomized split of 80% training and 20% testing"* with **no patient or
  eye grouping**, on a dataset with repeat imaging.

---

## P6 — Tang *et al.*, MMRDR — **VERIFIED, with the metric corrected**

**⚠️ M2's numbers are ACCURACY, not AUC, and the distinction is the paper's own.** Table 3 reports
ResNet-50 **0.823** and RETFound **0.822** on CFP DR grading, both fully fine-tuned. The paper
states: *"AUC metrics are excluded due to prohibitive computational costs in extracting logits
from sequential LVLM outputs."* **Citing 0.822/0.823 as AUC would misattribute a metric the paper
explicitly declined to compute.** As accuracy, **VERIFIED exactly**, and M2's substance — a tie
in-distribution between a retinal foundation model and an ImageNet CNN — stands.

**M4 — VERIFIED VERBATIM:**
> *"The ImageNet-pretrained ResNet-50 generalizes slightly better to UWF than ophthalmic
> foundation models do. This counter-intuitive result is likely attributable to limitations in
> the scale and annotation granularity of currently available UWF datasets for pretraining
> ophthalmic foundation models."*

**M6-adjacent — VERIFIED.** MMRDR-OCT holds **2,938** images with 3-class DME grading. Table 2:
no DME **1,017**, non-centre-involving **280**, centre-involving **1,641**. The middle grade is
**280 / 2,938 = 9.5 %**. **A purpose-built DME set an order of magnitude larger than IDRiD still
yields a middle grade under 10 %** — this is the strongest external support F7 has, because it
removes "your dataset is too small" as an explanation.

**Unmatched adaptation — VERIFIED.** The paper tabulates side by side models adapted differently:
ophthalmic foundation models are *"evaluated under recommended configurations (fine-tuning the
full model or with linear probing)"*, and FLAIR and KeepFIT are the ones run with *"minimal
tuning"*, against fully fine-tuned ResNet-50, ViT and RETFound. **The confound is in the paper's
own description of its protocol.**

**Contamination — VERIFIED, and worse than the ledger states.** *"the OIA-DDR dataset … was
selected as the source of CFP images for the MMRDR dataset."* MMRDR-CFP **is** OIA-DDR, FLAIR's
pretraining includes OIA-DDR, and **the paper nowhere acknowledges the overlap.**

**A fourth exhibit, found during verification and not previously in the ledger.** The CFP subset
*"was split at the image level due to the unavailability of patient identifiers in the source
OIA-DDR dataset"*, while OCT and UWF *"were split at the patient level to prevent data leakage."*
**The paper knows the rule, states it, applies it to two modalities, and cannot apply it to the
third — and reports one benchmark table across all three.** This is precisely our F6 problem, in
a Nature-family data descriptor, and it belongs in the provenance chapter.

---

## P7 — Ruamviboonsuk *et al.* — **VERIFIED VERBATIM (V2, V3, V4, V5)**

**V4:** *"As of the end of April 2024, there have been only three ophthalmic AI devices, IDx-DR,
EyeArt, AEYE-DS, approved by the U.S. Food and Drug Administration (FDA)."* **VERIFIED.**

⚠️ **But the rate is theirs and it is wrong.** They continue: *"Considering the first AI model
approved was in 2021, this means there is an approval rate of a model a year."* **IDx-DR was
approved in April 2018**, so their arithmetic rests on a wrong start year. **Cite the three
devices; do not repeat "roughly one per year."**

**V5 — VERIFIED VERBATIM:** *"there are at least four challenging areas in deployment of AI:
(1) the lack of more head-to-head comparisons of the available models, (2) no clear evidence of
cost-effectiveness of AI …, (3) equity and bias issues, and (4) medicolegal considerations."*
The first barrier is this thesis's justification, in a peer-reviewed ophthalmology review.

**The gap sentence — VERIFIED VERBATIM**, and it is the better introduction citation:
> *"Whereas the number of studies on new techniques of AI in retinal imaging keeps increasing
> exponentially in recent years, the adoption of AI in ophthalmic care … increases at a much
> lower rate in comparison. This means the gap between development and deployment is widening."*

**V3 — VERIFIED:** *"RETFound, was pretrained with 1.6 million retinal images of CFPs and OCT …
This self-supervised model was found to outperform traditional supervised and self-supervised
models pretrained on ImageNet datasets for the same tasks in both internal and external
validations."* This settles the pretraining-count discrepancy: **1.6 M total (CFP + OCT)**, of
which ~0.9 M are fundus. Nielsen quoted the fundus figure. State ours as 1.6 M and say so.

**V1 versus V2 — the self-contradiction is real and unreconciled.** The review asserts *"Many
studies found better performance of ViT, compared to CNN"* and then reports, without comment:
> *"a head-to-head comparison between 8 CNN models and 9 ViT models to classify referrable and
> non-referrable AMD in CFPs found all the CNN models performed better than all the ViT models,
> with sensitivity and specificity of the CNN models reaching 90% or more, whereas the ViT models
> could achieve 63%–94% sensitivity and 24%–48% specificity."* **VERIFIED exactly.**

**UNVERIFIED detail:** the "156 records" screening count is in the PRISMA figure, not the text;
not confirmed. Do not cite the number — cite the search (PubMed + Scopus, 1 Nov 2023 – 30 Apr 2024).

---

## P9 — Tomić *et al.* — **T4 VERIFIED VERBATIM**

Confirmed: 160 patients, **320 eyes**, type 2 diabetes, hand-held TANG camera graded by the
**DeepDR** system against slit-lamp examination and a standard VISUCAM camera.

**T4 — VERIFIED, in two sentences that together make the claim exact:**
> *"The three screening methods did not differ in detecting moderate/severe nonproliferative and
> proliferative DR."*
> *"the substantial distinction in DR degree assessment among methods was found in the eyes with
> no retinopathy and those with signs of mild NPDR (single microaneurysms)."*

**Every disagreement in the study sits at the no-DR / mild-NPDR boundary.** Discrepancies: **22
eyes** graded mild NPDR clinically but no DR by the hand-held camera; **34 eyes** on the same
pattern against the standard camera. Agreement κ **0.86 ± 0.04** (vs clinical examination) and
**0.78 ± 0.05** (vs standard camera) — **VERIFIED**.

**Eye-level analysis, no clustering adjustment — VERIFIED.** 320 eyes from 160 patients analysed
eye-wise, with no adjustment for the two-eyes-per-patient correlation. **Third instance of the
CLAIM item 21 failure**, after P1 and P12.

---

## P10 — Yang *et al.* — **VERIFIED**

38 studies included from 337 abstracts; PROSPERO CRD42023493512; AUC range **0.676–0.971**.

**Y2 — VERIFIED VERBATIM**, from the Discussion:
> *"Models trained only in internal dataset demonstrated higher performance compared to those
> trained externally, suggesting a potential risk of overfittings and training bias."*

⚠️ **Quote it as written.** The paper says *trained* only internally; the ledger paraphrases it as
*validated* only internally. Use their wording, or the claim is ours rather than theirs.

**Y1 — VERIFIED:** *"only 12 studies (32%) conducted external validation"* (internal validation:
26 studies, 68 %). **Y4 — VERIFIED:** *"only 4 studies (11%) reported missing data handling, with
the remaining 34 studies (89%) lacking this information."*

**Y3 — VERIFIED, and cite the Discussion, not the abstract.** The Discussion states *"There were
no randomized clinical trials identified in our review evaluating the performance of these
algorithms"*; the abstract softens this to *"a paucity of randomized control trials."* The strong
form is defensible only as the Discussion phrases it.

---

## What changed as a result of this verification

1. **Nothing in the literature structure has to change** — every 2026 paper exists and is readable.
2. **Three numbers in the ledger are corrected**: P3's four AUROC values (direction reversed),
   P6's metric (accuracy, not AUC), P1's CB-384 comparator (unconfirmed, not citable).
3. **Two claims must be re-worded to their source**: P10's Y2 (*trained*, not *validated*), and
   P7's V4 (cite the three devices, not the one-per-year rate, which rests on a wrong start year).
4. **Two new exhibits found at source**, neither previously in the ledger: P6 splits its CFP
   subset at image level because the source dataset has no patient identifiers, and P5 splits
   APTOS 80/20 at random with no grouping at all. Both belong in the provenance chapter.
