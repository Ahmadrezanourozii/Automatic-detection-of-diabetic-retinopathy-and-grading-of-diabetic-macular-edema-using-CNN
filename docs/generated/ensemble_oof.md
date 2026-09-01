# Ensembling archived out-of-fold predictions (no GPU)

Logit-averaged across runs, decoded with cut-points **cross-fitted on the other folds** so the ensemble and every member sit at matched operating points (`PROTOCOL.md` §4.1). Membership follows rules fixed before the numbers were seen (`PROTOCOL.md` §3) — no subset search.

## Members, scored individually at matched calibration

| run | backbone | size | DR QWK | DR acc | DME QWK | DME acc |
|---|---|---|---|---|---|---|
| E05 | densenet121 | 448 | 0.8306 | 67.96 % | 0.8840 | 84.11 % |
| E06 | densenet121 | 448 | 0.8554 | 71.68 % | 0.8763 | 85.85 % |
| E07 | tf_efficientnet_b3 | 448 | 0.8441 | 70.22 % | 0.8989 | 86.05 % |
| E08 | densenet121 | 448 | 0.8646 | 72.12 % | 0.8764 | 83.33 % |
| E09 | densenet121 | 224 | 0.8389 | 63.72 % | 0.8522 | 83.53 % |
| E10 | densenet121 | 640 | 0.8749 | 74.78 % | 0.8948 | 86.63 % |
| E14MAC | densenet121 | 448 | 0.8595 | 73.27 % | 0.8538 | 81.20 % |
| E15LPFT | densenet121 | 448 | 0.8655 | 74.07 % | 0.8577 | 83.91 % |
| E17NAT | densenet121 | 640 | 0.8662 | 72.35 % | 0.8748 | 84.30 % |

## Pre-specified ensembles

| rule | members | DR QWK | DR acc | DME QWK | DME acc | best single DR QWK | Δ vs best single |
|---|---|---|---|---|---|---|---|
| `all-5fold` ⚠️MIXED-INPUTS | 9: E05, E06, E07, E08, E09, E10, E14MAC, E15LPFT, E17NAT | **0.8933** | 77.26 % | 0.8939 | 87.40 % | 0.8749 | **+0.0184** |
| | *vs best member `E10`, paired bootstrap* | | | | | | +0.0186 [+0.0089, +0.0283] **significant** |
| `matched-448` ⚠️MIXED-INPUTS | 8: E05, E06, E07, E08, E10, E14MAC, E15LPFT, E17NAT | **0.8917** | 77.39 % | 0.9047 | 88.18 % | 0.8749 | **+0.0168** |
| | *vs best member `E10`, paired bootstrap* | | | | | | +0.0170 [+0.0077, +0.0264] **significant** |
| `externally-checkable` ⚠️MIXED-INPUTS | 6: E08, E09, E10, E14MAC, E15LPFT, E17NAT | **0.8885** | 76.11 % | 0.8862 | 86.24 % | 0.8749 | **+0.0136** |
| | *vs best member `E10`, paired bootstrap* | | | | | | +0.0137 [+0.0045, +0.0230] **significant** |
| `same-consumption` | 5: E06, E08, E09, E14MAC, E15LPFT | **0.8828** | 74.42 % | 0.8872 | 86.82 % | 0.8655 | **+0.0173** |
| | *vs best member `E15LPFT`, paired bootstrap* | | | | | | +0.0174 [+0.0081, +0.0270] **significant** |


## ⚠️ Mixed inputs, found by the consumption manifests (PROTOCOL.md §9)

Three of the four rules are marked **MIXED-INPUTS**. Their members did not all read the same
files: E10 and E17NAT mounted the native-resolution Messidor-2 mirror, E08, E09, E14MAC and
E15LPFT did not. Averaging their logits averages predictions about **different renderings of
the same eyes**, so such an ensemble cannot be reproduced by feeding one image through every
member — which is what a deployed ensemble would do.

This was not visible before `src/manifest.py` existed. It was found by the same check that
caught §26, applied to a result I had already reported as clean.

**The fourth rule settles it.** `same-consumption` is the largest group whose manifests
agree — five runs, chosen by a mechanical property and never by score:

| rule | inputs | DR QWK | gain over best member | verdict |
|---|---|---|---|---|
| `all-5fold` | mixed | 0.8933 | +0.0186 [+0.0089, +0.0283] | significant |
| `externally-checkable` | mixed | 0.8885 | +0.0137 [+0.0045, +0.0230] | significant |
| **`same-consumption`** | **identical** | **0.8828** | **+0.0174 [+0.0081, +0.0270]** | **significant** |

**The ensembling gain is robust: +0.0174 on a set where every member read the same files,
against +0.0186 on the mixed set.** Input diversity is worth roughly 0.01 of *absolute* QWK
and essentially nothing of the *gain*. So the finding — ensembling significantly improves DR
and does nothing for DME — stands, and the honest headline candidate is **0.8828**, not
0.8933, until I23 reproduces a number on APTOS where every member necessarily sees identical
images.
