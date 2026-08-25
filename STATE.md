# STATE.md — read this first, every session

**Project:** Automatic detection of diabetic retinopathy (DR) and grading of diabetic
macular edema (DME) from a single retinal fundus photograph.
**Owner:** Alireza Chegeni (810102111), MSc, ECE, University of Tehran.
**Last updated:** 2026-08-25 — session 1 (audit of prior work; nothing trained).

---

## Where the project stands

Session 1 was an audit of the work done up to July 2026. It did **not** train anything.
The audit's conclusion is the single most important fact in this repository:

> **The results reported in the existing thesis (chapter 4) cannot be reproduced from
> anything in this project, and the evidence indicates they were never computed.**
> The only evaluation output ever saved to disk shows a model that scores **27.2 % on
> 5-class DR** and **15.9 % on 3-class DME** — both *below* the majority-class floor.

Everything downstream of that follows: there is no working baseline, so there is nothing
to "improve on". The project restarts from a trivial baseline and builds up.

Run `python3 src/audit_baseline.py --drive "<Google Drive Alireza folder>"` to reproduce
every number in this file. See `ISSUES.md` §1 for the full diagnosis.

## Current best result

**None.** No model in this project has ever beaten a majority-class predictor on a
held-out set. The honest current state of the art for this repository is:

| Head | Trivial floor (majority class) | Best measured | Status |
|---|---|---|---|
| DR, 5-class, IDRiD official test (n=103) | 33.0 % | 27.2 % | below floor |
| DME, 3-class, gated DR≥1, IDRiD test (n=69) | 69.6 % | 15.9 % | below floor |

## What is running right now

Nothing. No GPU has been spent by this repository.

## Blocking

1. **Protocol not yet agreed.** `PROTOCOL.md` is a *proposal*. Nothing may be trained
   until the owner signs off on it. This is deliberate — see `PROTOCOL.md` §0.
2. **Credentials leaked.** A GitHub PAT and a Kaggle API key were pasted into a chat
   transcript on 2026-08-25. Both must be revoked and reissued before any push.
3. **Unanswered:** defence date, weekly GPU budget, Kaggle username, and whether
   chapter 4 of the thesis is to be rewritten from computed numbers (see `ISSUES.md` §1).

## Next three planned experiments

Ordered by information gained per GPU-hour. None may start before the protocol is agreed.

| # | Hypothesis | Falsified if | Cost |
|---|---|---|---|
| E01 | The prior training collapse is an optimisation failure, not a data failure: a frozen-backbone linear probe on ImageNet features separates DR grades well above the 33 % floor. | Linear probe ≤ floor + its CI → the signal is not in the representation, and preprocessing/resolution is the problem, not the training loop. | ~0 GPU (CPU, 224 px, cached features) |
| E02 | The five-step preprocessing chain (green→CLAHE→blur) is not better than plain RGB. Same split, same backbone, chain on vs chain off. | Chain wins by more than the paired-bootstrap interval → the chain is load-bearing and the thesis' ablation claim survives. | ~1 GPU-h |
| E03 | Splitting on images rather than on eyes/patients inflates the score. Same pipeline, group-wise split vs naive random-image split, difference reported. | No gap → the corpora have no near-duplicate structure; record the number and move on. | ~1 GPU-h |

E01 is the gate. If a linear probe cannot beat the floor, no amount of architecture work will.

## Files a new session must read, in order

1. `STATE.md` (this file)
2. `PROTOCOL.md` — the frozen evaluation protocol
3. `ISSUES.md` §1 — why the prior results are void
4. `data/DATASETS.md`, `data/LABEL_MAPPING.md` — what data exists and what the labels mean
5. `EXPERIMENTS.md` — the run ledger
