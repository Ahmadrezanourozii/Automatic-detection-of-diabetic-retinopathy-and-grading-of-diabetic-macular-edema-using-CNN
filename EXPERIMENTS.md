# EXPERIMENTS.md — the run ledger

One row per run. **Appended, never edited.** This table is what the thesis results chapter
gets built from.

Columns: ID · date · commit SHA · hypothesis · what changed vs. parent · DR acc · DR QWK ·
DME acc · DME QWK · macro-F1 · bootstrap CI · significant? · archived log.

Every row must link to a `runs/<ID>/results.json` carrying the exact configuration, the git
SHA it ran from, environment versions, runtime and seed, plus `runs/<ID>/train.log`.
A number without a row here does not exist.

---

## Runs

| ID | Date | SHA | Hypothesis | Δ vs parent | DR acc | DR QWK | DME acc | DME QWK | macro-F1 | 95 % CI | Sig? | Log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E01·rgb** | 2026-08-25 | `075d83c` | Frozen ImageNet features beat the majority floor on both heads | baseline (plain RGB, 224 px, linear probe) | 47.6 % | 0.584 | **71.8 %** | **0.678** | 0.427 / 0.582 | DR [37.9, 57.3] · DME [63.1, 80.6] | **yes vs floor** | `runs/E01/` |
| **E01·green_clahe** | 2026-08-25 | `075d83c` | *(same)* | + thesis chain (green→CLAHE→blur), ImageNet norm | **51.5 %** | **0.654** | 68.9 % | 0.602 | 0.391 / 0.536 | DR [41.7, 60.2] · DME [60.2, 77.7] | **yes vs floor**; **no vs rgb** | `runs/E01/` |
| **E01·green_clahe_raw01** | 2026-08-25 | `075d83c` | *(same)* | + thesis chain exactly as old code fed it ([0,1], no ImageNet norm) | 47.6 % | 0.650 | 65.0 % | 0.610 | 0.399 / 0.536 | DR [37.9, 57.3] · DME [56.3, 74.8] | **yes vs floor**; **no vs rgb** | `runs/E01/` |

DME columns are the **ungated primary** definition (n=103, floor 46.6 %). Gated DR≥1 numbers
are in `runs/E01/results.json` and are discussed in the note below — **no variant beats the
gated floor**. All pairwise variant comparisons: `runs/E01/comparisons.json`.

### E01 — verdict

**Hypothesis confirmed.** Frozen ImageNet DenseNet121 features plus a logistic regression
beat the majority-class floor on DR (47.6–51.5 % vs 33.0 %, lower CI bound 37.9 %) and on
ungated DME (65.0–71.8 % vs 46.6 %). QWK intervals exclude zero on both heads.

**Therefore the archived collapse was an optimisation failure, not a data failure.** A
linear probe on frozen features, at 224 px, with no fine-tuning at all, outscores the old
fine-tuned model by **20 accuracy points on DR** (47.6 % vs 27.2 %) and by **53 points on
gated DME** (63.8 % vs 15.9 %). The signal was always there; the training loop destroyed it.
`IDEAS.md` I05 and everything downstream of it are unblocked.

### E01 — the preprocessing factorial (cheap version of E02)

**0 of 18 pairwise comparisons are significant.** Not one — across three variants, two
metrics and three evaluation definitions. Largest point estimate of any difference: 3.9
accuracy points.

This does **not** reproduce the thesis' ablation claim (green channel worth 10 points on DR
and 7 on DME; CLAHE 6 and 8). But it does not refute it either, and saying so matters: at
n=103 the paired interval on an accuracy difference is roughly ±10 to ±14 points, so a
10-point effect sits right at the edge of what this test can see. The honest statement is
**"no evidence of an effect, at a resolution too coarse to rule one out"** — which is why
E02 must be re-run under repeated CV over the full development pool (± 1.3 pts) before the
ablation table can be written either way. `PROTOCOL.md` §7.

One suggestive per-class detail, recorded but not claimed: both green-channel variants score
**0 % recall on the Mild DR class** where plain RGB scores 40 %. With 5 mild images in the
test set that is 2 images versus 0, far too few to mean anything — but it is the exact class
the thesis identifies as its main weakness, so it is worth watching in the repeated-CV run.

### E01 — the gating question, settled empirically

| Definition | n | floor | best variant | beats floor? |
|---|---|---|---|---|
| DME **ungated** (primary) | 103 | 46.6 % | 71.8 % | **yes**, comfortably |
| DME **gated DR≥1** (secondary) | 69 | 69.6 % | 65.2 % | **no** — every variant is below it |

A model that genuinely carries DME signal — it beats the ungated floor by 25 points — still
scores *below* the gated floor, because gating throws away the easy negatives and leaves a
set that is 70 % one class. This is the concrete demonstration that the gated number is a
different quantity, and it is why `PROTOCOL.md` §5.1 makes ungated primary. It also puts the
existing thesis' gated 87.6 % claim in context: on that framing, nothing here clears 69.6 %.

---

## Reference floors — every row above must be read against these

| Quantity | Set | n | Value |
|---|---|---|---|
| DR majority class | IDRiD official test | 103 | 33.0 % |
| DME majority class, gated DR≥1 | IDRiD test | 69 | 69.6 % |
| DME majority class, gated DR≥1 | all IDRiD | 348 | 69.8 % |
| DME majority class, ungated | all IDRiD | 516 | 47.1 % |

## Prior work — NOT rows in this ledger

The figures in thesis chapter 4 (DR 91.6 %, DME 87.6 %) and in `CLAUDE.md` (91.7 / 87.4)
have no run behind them and are not reproducible; see `ISSUES.md` §1. They are recorded
there as a finding, not here as a baseline. The only evaluation ever archived by this
project scored **27.2 % DR** and **15.9 % DME**, both below the floors above.
