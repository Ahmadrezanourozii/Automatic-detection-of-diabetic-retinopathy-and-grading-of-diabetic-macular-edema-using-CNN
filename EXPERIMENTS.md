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
| — | — | — | *no runs yet — protocol not agreed* | — | — | — | — | — | — | — | — | — |

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
