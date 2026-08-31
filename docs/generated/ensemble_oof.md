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
| `all-5fold` | 9: E05, E06, E07, E08, E09, E10, E14MAC, E15LPFT, E17NAT | **0.8933** | 77.26 % | 0.8939 | 87.40 % | 0.8749 | **+0.0184** |
| | *vs best member `E10`, paired bootstrap* | | | | | | +0.0186 [+0.0089, +0.0283] **significant** |
| `matched-448` | 8: E05, E06, E07, E08, E10, E14MAC, E15LPFT, E17NAT | **0.8917** | 77.39 % | 0.9047 | 88.18 % | 0.8749 | **+0.0168** |
| | *vs best member `E10`, paired bootstrap* | | | | | | +0.0170 [+0.0077, +0.0264] **significant** |

## DME, tested separately against F7

| rule | DME ensemble QWK | best member | difference | verdict |
|---|---|---|---|---|
| `all-5fold` | 0.8939 | E07 0.8989 | −0.0049 [−0.0292, +0.0193] | indistinguishable |
| `matched-448` | 0.9047 | E07 0.8989 | +0.0059 [−0.0186, +0.0293] | indistinguishable |

**Ensembling moves DR significantly and DME not at all.** That is the asymmetry F7 predicts:
logit averaging reduces *variance*, so it helps where the limit is variance (DR) and does
nothing where the limit is supervision (DME). F7 is corroborated by a method that is not an
architectural intervention at all, which is a different kind of evidence from the five
falsified architecture experiments behind it.

## What this number is, and what it is not

**It is a development-pool estimate.** Every member was trained and selected on this pool, so
the ensemble's OOF score is not a held-out result and **must not be quoted as a headline
until it is reproduced on APTOS**. Doing that needs one external-only evaluation per member
(cheap, no retraining) and is queued as I23.

**It is not a subset search.** Both membership rules were written into `src/ensemble_oof.py`
before any ensemble number was computed. Trying combinations and keeping the best would be
selection on the evaluation set (`PROTOCOL.md` §3) and would inflate this figure.

**Falsified experiments still contributed.** E14MAC and E15LPFT are both closed as failures on
their own hypotheses, and both are in the winning ensemble. A run that does not beat the
baseline can still carry complementary errors.
