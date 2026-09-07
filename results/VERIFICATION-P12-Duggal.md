# P12 — Duggal et al., JMIR Med Inform 2025;13:e67529 — **ALL CHECKED CLAIMS VERIFIED**

DOI [10.2196/67529](https://doi.org/10.2196/67529) resolves. Full text retrieved from PubMed
Central (PMC12419978). Article metadata and full text retrieved from **PubMed**.

Citation confirmed: Duggal M, Chauhan A, Gupta V, et al. *Real-World Evaluation of AI-Driven
Diabetic Retinopathy Screening in Public Health Settings: Validation and Implementation
Study.* JMIR Med Inform. 2025;13:e67529. PMID 40925861.

| ID | claim as recorded in the ledger | paper says | verdict |
|---|---|---|---|
| **G1** | three vendors, specificity 14.25 %–96.01 % | *"specificity (14.25%‐96.01%)"*, sensitivity 59.7 %–97.74 % | **VERIFIED exactly** |
| **G2** | one vendor, sensitivity 68.4 % → 99.6 %, specificity 96 % → 64.7 % | *"the AI-3 algorithm's sensitivity increased significantly from 68.4% to 99.6%, while specificity decreased from 96% to 64.7% between the validation and implementation phases"* | **VERIFIED exactly** |
| **G4** | DME sensitivity 26.5 %, κ 0.38, against referable-DR κ 0.81 in the same run | Table 2: DME grade sensitivity **26·5**, κ **0.38**; RDR sensitivity 78·9, κ **0.81** | **VERIFIED exactly** |
| **G5** | of 64 referred patients, 9 (14 %) attended | *"Of 64 referred participants, 28 (43.8%) were contacted; only 9 (14%) adhered to referral advice"* | **VERIFIED exactly** |
| **G8** | κ moved 0.65 → 0.72 while sensitivity and specificity each moved 31 points | validation AI-3 *"agreement with the RS (κ=0·65)"*; Table 2 DR grade κ **0.72**. Sensitivity 68.4→99.6 = **+31.2**; specificity 96→64.7 = **−31.3** | **VERIFIED — arithmetic exact** |

## ⚠️ One qualification on G8 that the ledger does not carry, and the thesis must

The ledger calls G8 *"the clinical version of the QWK insensitivity we found"* and *"the single
best external citation we have"*. The numbers are right, but **the mechanism is not a
decision-rule change.**

The κ = 0.65 figure comes from the **validation phase** (250 participants, PGIMER and PHC
Khizrabad, March–June 2021). The κ = 0.72 figure comes from the **implementation phase** (343
participants, CHC Badhani Kalan, February–June 2022). Between them the vendor **retrained the
algorithm** — the paper says so explicitly: *"The sensitivity likely improved post validation
due to algorithm training"* — and the paper attributes the specificity fall partly to
cataracts affecting media opacity in the second cohort.

So G8 is a genuine instance of **κ barely moving while sensitivity and specificity swing 31
points**, which is the phenomenon we care about. But it is **not** one model swept across
cut-points on fixed data. It is a different model on a different cohort. **Cite it as evidence
that summary agreement statistics are insensitive to operating-point changes that matter
clinically — not as a threshold sweep.** Our own frontier is the threshold sweep; this is the
independent corroboration that the insensitivity has clinical consequences.

## A bonus finding, useful for T1

The paper independently states the IDx-DR pivotal benchmark: *"the FDA's benchmark for
superiority was set at 85% sensitivity and 82.5% specificity"* — matching the figures our own
T1 work verified from Abràmoff et al. Two independent sources now agree.

It also reports the Lee et al. seven-system comparison second-hand: NPV 82.72 %–93.69 % against
sensitivity 50.98 %–85.90 %.
