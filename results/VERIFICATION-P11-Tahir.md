# P11 — Tahir et al., Front Med 2025;12:1519768 — **core claim VERIFIED, one sub-claim CONTRADICTED**

DOI [10.3389/fmed.2025.1519768](https://doi.org/10.3389/fmed.2025.1519768) resolves; author,
journal and year match. Full text read from the publisher (open access).

| ID | claim as recorded | paper says | verdict |
|---|---|---|---|
| **H4** | pooled AI sensitivity 0.95 | *"0.95 [95% CI: 0.91–0.97]"* (dilated); 0.90 [0.85–0.94] undilated | **VERIFIED** |
| **H5** | individual study sensitivities span 0.06–1.00 | spans *"0.06 [0.06, 0.07]"* to *"1.00 [0.90, 1.00]"* | **VERIFIED** |
| **F7-exhibit 4** | Table 2's false-negative column is a copy of the false-positive column; header reads `TP FP FP TN`; 37 rows | header reads **TP, FP, FP, TN**; **37 rows**; e.g. Ting `TP=676, FP=9969, FP=9969, TN=102003` | **VERIFIED** |

## The arithmetic that makes the exhibit conclusive

For Ting et al., the paper displays sensitivity **0.06**. Computing from the table as printed,
treating the third column as the false-negative count:

    676 / (676 + 9969) = 0.0635 → 0.06

**The displayed sensitivity is exactly what you get from the duplicated column.** So the error
is not confined to a header: the wrong values propagated into the computed sensitivities. A
sensitivity of 0.06 for Ting et al. — one of the largest and best-known DR screening studies —
is not a plausible result, and it sits in a published meta-analysis pooling 613,690 images.

## ⚠️ One ledger sub-claim is CONTRADICTED and must not be repeated

The ledger states: *"the forest plots use different, correct values (Ting: table 0.25 vs plot
0.90; Sosale: 0.84 vs 0.99; Abràmoff: 0.67 vs 0.97)"*.

**Not what the source shows.** The sensitivities displayed for those studies are Ting **0.06**,
Sosale **0.93**, Abràmoff **0.67** — i.e. Ting's displayed value is the *corrupted* one, not a
corrected 0.90, and the ledger's "table 0.25" for Ting does not appear at all.

**Consequence for the thesis: state the exhibit as "the false-negative column is duplicated
from the false-positive column and the reported sensitivities are computed from it", which is
verified. Do NOT claim the forest plots carry corrected values — that specific assertion is
contradicted by the source.** Marked `UNVERIFIED-CONTRADICTED` in `CLAIMS.md` rather than
deleted, per the standing instruction.

This is itself an instance of the phenomenon the section is about: a specific, confident,
checkable number that does not survive checking.
