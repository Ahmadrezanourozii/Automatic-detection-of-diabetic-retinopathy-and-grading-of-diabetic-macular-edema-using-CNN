# Paper verification status — P1–P12

Scope, as instructed: DOI and bibliographic reference for all twelve; **only the numbers relied
on in the ledger's "OUR FINDINGS → EVIDENCE MAP"**, not every number in every table. Anything
unverified is tagged, never deleted, and **must not be cited in the thesis while tagged**.

## Existence — all twelve DOIs resolve

Every DOI in the ledger resolves through doi.org content negotiation, with first author,
journal and year matching the ledger. **The ledger is not fabricated.**

One citation correction: **P9 (Tomić) is dated 2023 in the DOI record**, not 2024 as the ledger
states (*Biomedicines* 12(1):34 — volume 12 is the 2024 volume, so this is an online-first
versus issue-date difference; cite as the publisher lists it).

## Numbers checked against source — **ALL TWELVE PAPERS NOW CHECKED**

| paper | claims checked | verdict | record |
|---|---|---|---|
| **P1** Sheng | S1, S3, S5, S7 | **VERIFIED exactly**; one comparator number unconfirmed | `VERIFICATION-P1-P3-P5-P6-P7-P9-P10.md` |
| **P2** Poyrazer | R2, R3, R4, R5, R6, R7 | **all VERIFIED exactly**, minor note on R4 | `VERIFICATION-P2-Poyrazer.md` |
| **P3** Nielsen | N1 | **VERIFIED, ledger numbers CORRECTED** | `VERIFICATION-P1-P3-P5-P6-P7-P9-P10.md` |
| **P5** Yu | D6, F7-exhibit-1 | **VERIFIED VERBATIM** | `VERIFICATION-P1-P3-P5-P6-P7-P9-P10.md` |
| **P6** Tang | M2, M4, M6-adjacent, contamination | **VERIFIED, metric corrected to accuracy** | `VERIFICATION-P1-P3-P5-P6-P7-P9-P10.md` |
| **P7** Ruamviboonsuk | V2, V3, V4, V5 | **VERIFIED VERBATIM**; one rate not citable | `VERIFICATION-P1-P3-P5-P6-P7-P9-P10.md` |
| **P8** Zhou (RETFound) | Z4, contamination, recipe | **VERIFIED**; Z3 and F7-exhibit-2 still open | `VERIFICATION-P8-RETFound.md` |
| **P9** Tomić | T4, F7-exhibit-3 | **VERIFIED VERBATIM** | `VERIFICATION-P1-P3-P5-P6-P7-P9-P10.md` |
| **P10** Yang | Y1, Y2, Y3, Y4, Y5 | **all VERIFIED**; Y2 must be quoted as written | `VERIFICATION-P1-P3-P5-P6-P7-P9-P10.md` |
| **P11** Tahir | H4, H5, F7-exhibit-4 | core **VERIFIED**; one sub-claim **CONTRADICTED** | `VERIFICATION-P11-Tahir.md` |
| **P12** Duggal | G1, G2, G4, G5, G8 | **all VERIFIED exactly**, one qualification on G8 | `VERIFICATION-P12-Duggal.md` |

**No 2026 paper is missing.** P1, P2, P5 and P6 all exist, are indexed, and their load-bearing
numbers are readable at source. **The literature-review structure stands as designed.**

## Still UNVERIFIED — tagged, not deleted, NOT CITABLE

| claims | paper | why not |
|---|---|---|
| **Z3** (IDRiD AUPR *P* = 0.81 vs AUROC *P* < 0.001) | P8 | needs Supplementary Table 3; the main text appears to **contradict** it |
| **F7-exhibit-2** (internal IDRiD and external APTOS→IDRiD AUROC identical to three decimals *including* the CI) | P8 | observation verified, **diagnosis not**; needs Supplementary Table 3. Phrase as "we could not determine whether this reflects a coincidence or a transcription error" |
| **Z5, Z7, Z9, Z1/Z2** | P8 | not read against source; Z7 is only needed as the F7 rebuttal and can be dropped |
| **S1's CB-384 comparator** (external QWK 0.5534 ± 0.0387) | P1 | the retrieved table row does not match; **do not cite this number** |
| **"156 records" screening count** | P7 | in the PRISMA figure, not the text |

## Corrections the thesis must carry

1. **P3 N1's numbers.** APTOS frozen **0.93 ± 0.01** vs fine-tuned **0.94 ± 0.01**; ODIR-5K frozen
   **0.78 ± 0.02** vs fine-tuned **0.80 ± 0.02**. The ledger had the frozen head level or ahead;
   at source it is nominally behind on both. *p* = 0.50 and 0.53 stand, so "matches" survives.
2. **P6 M2 is accuracy, not AUC.** The paper explicitly excludes AUC.
3. **P10 Y2 says *trained* only internally**, not *validated*. Quote as written.
4. **P7 V4: cite the three FDA devices, not the "one per year" rate** — that rate assumes the
   first approval was 2021, and IDx-DR was approved in April 2018.
5. **P9 is *Biomedicines* 2024;12(1):34**, online-first 2023-12-22.
6. **G8 is not a threshold sweep** (P12): the two κ values come from different phases, different
   cohorts, and the vendor retrained between them. Cite it as evidence that summary agreement
   statistics are insensitive to operating-point changes.
7. **P11's forest-plot sub-claim is false** (displayed values are Ting 0.06, Sosale 0.93,
   Abràmoff 0.67). What is verified and stronger: the FN column is duplicated from FP, and
   676/(676+9969) = 0.06 shows the corruption propagated into the reported sensitivities.
8. **The *Nature* anomaly: do not assert an error.** Observation verified, diagnosis not.

## Two new exhibits found during verification, not previously in the ledger

* **P6** splits its CFP subset **at image level** *"due to the unavailability of patient
  identifiers in the source OIA-DDR dataset"*, while splitting OCT and UWF at patient level
  *"to prevent data leakage"* — then reports one benchmark table across all three.
* **P5** splits APTOS *"a randomized split of 80% training and 20% testing"* with no eye or
  patient grouping at all.

Both belong in the provenance chapter beside P8, P9 and P11.

## A note on the "use the PDF, not the web" instruction

**We hold no PDFs of P1–P12.** All were read against **PubMed Central full text** or the
publisher's open-access full text — for these journals the same document as the PDF. MDPI (P5)
and Nature (P6) refused the fetcher (HTTP 403 / auth redirect) and were read through PMC.
Recorded so the sourcing is not overstated.
