# I23 — does the ensembling gain survive on APTOS?

Members: **E08, E09, E10, E14MAC, E15LPFT** — whichever have archived APTOS predictions, a mechanical fact rather than a score. APTOS is held out in its entirety and no member has seen an image of it.

**Every member sees identical images here**, because APTOS has a single source. The development-pool ensemble could not say that (`PROTOCOL.md` §9), so this number is free of the mixed-input caveat by construction.

## APTOS (external, n = 3662)

| member | APTOS DR QWK |
|---|---|
| E08 | 0.8972 |
| E09 | 0.8681 |
| E10 | 0.8867 |
| E14MAC | 0.8733 |
| E15LPFT | 0.8586 |
| **ensemble of 5** | **0.8878** |

The single model to beat is **E10**, chosen because it scores highest on the **development pool** — never on APTOS (`PROTOCOL.md` §3).

**Ensemble − E10 on APTOS: +0.0011 [-0.0033, +0.0053] — indistinguishable**

> For completeness: the member scoring highest *on APTOS* is **E08** (0.8972). Comparing against that would be selection on the test set and is not the reported result.

## The same members on the development pool, for comparison

| | ensemble | best single (E10) | gain |
|---|---|---|---|
| development pool | 0.8903 | 0.8749 | +0.0154 |
| APTOS (held out) | 0.8878 | 0.8867 | +0.0011 |

