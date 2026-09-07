# P2 — Poyrazer et al., Front Med 2026 — **ALL CHECKED CLAIMS VERIFIED**

DOI [10.3389/fmed.2026.1815982](https://doi.org/10.3389/fmed.2026.1815982) resolves; author,
journal and year match the ledger. Full text read from the publisher (open access).

| ID | claim as recorded | paper says | verdict |
|---|---|---|---|
| **R6** | grade-1 external F1 = 0.000 for RETFound and EfficientNet-B0 on 270 images; MedSigLIP 0.153 | *"RETFound and EfficientNet-B0 both returned an F1 of exactly 0.000 for this grade, failing to correctly identify a single mild case in the 270 MESSIDOR-2 images carrying that label. MedSigLIP fared somewhat better, but its grade-1 F1 of 0.153…"* | **VERIFIED exactly** |
| **R5** | RETFound has the lowest external ECE (0.086) with the worst external AUC (0.697) | AUC: MedSigLIP 0.915 ± 0.005, RETFound **0.697 ± 0.009** (worst), EfficientNet-B0 0.745 ± 0.012. Post-TS external ECE: MedSigLIP 0.109 ± 0.011, RETFound **0.086 ± 0.020** (lowest), EfficientNet-B0 0.149 ± 0.023 | **VERIFIED exactly** |
| **R2** | frozen RETFound below the ImageNet baseline externally, ΔAUC −0.051, *p* = 0.016 | *"Retina-specific RETFound performed below the ImageNet baseline (ΔAUC = −0.051; p = 0.016, cluster-robust bootstrap)"* | **VERIFIED exactly** |
| **R3** | development discrimination cannot separate the encoders, all AUC 0.980–0.985 | MedSigLIP 0.985 ± 0.005, RETFound 0.984 ± 0.005, EfficientNet-B0 0.980 ± 0.007 | **VERIFIED exactly** |
| **R4** | temperature scaling moved external ECE by 0.004 for MedSigLIP and *increased* it by 0.002 for RETFound | *"ECE reductions of 0.004 for MedSigLIP, an increase of 0.002 for RETFound, and a modest reduction of 0.014 for EfficientNet-B0"* | **VERIFIED**, with a note below |
| **R7** | fold ensembling did not rescue external performance for any encoder | *"averaging predictions from five folds did not meaningfully rescue external performance for any encoder, suggesting that the dominant source of error is representational … rather than stochastic"* | **VERIFIED exactly** |

**Minor note on R4.** The deltas quoted in the paper's text may not reconcile exactly with the
pre/post values in its Table 3 Panel A. The direction and magnitude are as claimed and the
qualitative point stands, but if the thesis quotes the deltas numerically it should quote the
text and say so, rather than recomputing them from the table.

**R7 directly corroborates our own I23 result** — the ensemble's development gain did not
survive external validation (`IDEAS.md` I23; +0.0136 on the development pool, +0.0025
[−0.0016, +0.0066] on APTOS). Two independent studies, different encoders, same conclusion.
