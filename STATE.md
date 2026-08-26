# STATE.md — read this first, every session

**Project:** Automatic detection of diabetic retinopathy (DR) and grading of diabetic
macular edema (DME) from a single retinal fundus photograph.
**Owner:** Alireza Chegeni (810102111), MSc, ECE, University of Tehran.
**Last updated:** 2026-08-26 — session 2 (autonomous; E05/E06 done, E07/E08 running).

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

**E06** — DenseNet121, ordinal threshold heads, Messidor-2 partial-label DME supervision,
EyePACS pretraining. Pooled out-of-fold over all 2 260 development images:

| head | n | floor | accuracy | QWK |
|---|---|---|---|---|
| DR, 5-class | 2 260 | 52.4 % | 71.9 % | **0.847** |
| DME, 3-class ungated *(primary)* | 516 | 47.1 % | **86.0 %** | **0.879** |
| DME, 3-class gated DR≥1 *(secondary)* | 348 | 69.8 % | 80.5 % | 0.719 |
| Referable DME, binary | 2 260 | 82.6 % | 94.8 % | 0.819 |
| Referable DR (the screening decision) | 2 260 | — | sens **88.3 %** / spec 94.9 % | — |

On IDRiD's official 103-image test split — the set the old thesis quoted 91.6 % on — the
E05 pipeline scores **61.2 %**. Every number above is re-scored under the corrected DME
definition (`ISSUES.md` §12) from archived logits.

## Operating mode

**Autonomous, since 2026-08-25.** The owner has asked for the loop to run without
per-step approval: push to GitHub → run on Kaggle → fetch logs → analyse → change
something → run again. Protocol and honest reporting stay as they are; what is dropped is
waiting for sign-off between iterations.

## Infrastructure — live

| piece | where |
|---|---|
| code | `github.com/Ahmadrezanourozii/Automatic-detection-...-using-CNN` (branch `main`) |
| training | Kaggle notebook `ah22reza/dr-dme-<runid>`, GPU, clones the repo at a pinned SHA |
| data | attached Kaggle datasets — nothing uploaded, nothing downloaded locally |
| archive | `runs/<RUN_ID>/results.json` + `train.log` + `oof_*.npz`, committed permanently |
| launch | `python kaggle/build_kernel.py --run-id EXX --args "..." --push` |
| fetch | `python kaggle/fetch.py --run-id EXX --wait` |

Kaggle datasets attached: `aaryapatel98/indian-diabetic-retinopathy-image-dataset`,
`google-brain/messidor2-dr-grades`, `mariaherrerot/messidor2preprocess`.

## What is running right now

Two runs in parallel (Kaggle allows two concurrent GPU sessions):

| run | what it tests | data |
|---|---|---|
| **E05** | multi-output DenseNet121, ordinal threshold heads, Messidor-2 partial-label DME supervision | IDRiD + Messidor-2 (2 260) |
| **E06** | the same, **plus EyePACS pretraining** | + 35 126 EyePACS images |

Both: 5-fold grouped CV, 448 px, EMA, balanced sampling, TTA. E06's pretraining is done once
and reused by every fold.

**Note before reading their numbers:** both were launched before `ISSUES.md` §12 was fixed,
so their 3-class DME metric is computed on a biased 667-image set (59 % floor) instead of the
correct 516-image one (47 % floor). Re-score them with `src/recompute.py` from the archived
out-of-fold logits before believing any DME figure. No GPU needs to be re-spent.

## Blocking

1. **Credentials must be rotated when convenient.** Two GitHub PATs and two Kaggle keys have
   now been pasted into chat transcripts. They work and are in `.env` (gitignored, verified
   absent from git history), but they are exposed.
2. **Unknown:** defence date, weekly GPU budget. Both affect planning, neither blocks work.
3. **To acquire:** Messidor-1 (external DME validation), after a duplicate check against the
   Messidor-2 mirror (`ISSUES.md` §3).

## Next three planned experiments

Ordered by information gained per GPU-hour. None may start before the protocol is agreed.

| # | Hypothesis | Falsified if | Cost |
|---|---|---|---|
| E01 | The prior training collapse is an optimisation failure, not a data failure: a frozen-backbone linear probe on ImageNet features separates DR grades well above the 33 % floor. | Linear probe ≤ floor + its CI → the signal is not in the representation, and preprocessing/resolution is the problem, not the training loop. | **running** — local MPS, no quota |
| E02 | The five-step preprocessing chain (green→CLAHE→blur) is not better than plain RGB. Same split, same backbone, chain on vs chain off. | Chain wins by more than the paired-bootstrap interval → the chain is load-bearing and the thesis' ablation claim survives. | ~1 GPU-h |
| E03 | Splitting on images rather than on eyes/patients inflates the score. Same pipeline, group-wise split vs naive random-image split, difference reported. | No gap → the corpora have no near-duplicate structure; record the number and move on. | ~1 GPU-h |

E01 is the gate. If a linear probe cannot beat the floor, no amount of architecture work will.

## Files a new session must read, in order

1. `STATE.md` (this file)
2. `PROTOCOL.md` — the frozen evaluation protocol
3. `ISSUES.md` §1 — why the prior results are void
4. `data/DATASETS.md`, `data/LABEL_MAPPING.md` — what data exists and what the labels mean
5. `EXPERIMENTS.md` — the run ledger
