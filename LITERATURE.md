# LITERATURE.md — what has been read

Per paper: citation · what they did · numbers, on which dataset, under which protocol ·
the one idea that transfers · the cost of trying it.

**External numbers live only in this file.** They never migrate into our results table as
though they were ours.

---

## Read

*(nothing yet — session 1 was an audit)*

## In the project folder, not yet read

Four PDFs sit in `My Drive/Alireza/papers to learn from/`:
- `41433_2021_Article_1552.pdf` — *Eye* (Nature)
- `41433_2022_Article_2190.pdf` — *Eye* (Nature)
- `s41598-019-47181-w.pdf` — *Scientific Reports* 2019
- `ai-06-00269-v3.pdf` — *AI* (MDPI)

Plus the two works cited in the thesis' comparison table, which currently supply the only
external numbers in the document:
- `hardas2022svm` — handcrafted-feature SVM, ~77.3 % DR, reported on **DIARETDB1**.
- `suedumrong2024cnn` — single-output CNN, ~90.60 % DR.

**Both need checking before they can stay in the comparison table.** The specific thing to
check, on every DR paper: whether a reported accuracy is on **five classes** or on a
**binary referable/non-referable collapse**. The two get quoted interchangeably in this
literature and differ by ten points or more. A paper reporting far higher numbers than ours
very often also uses a weaker split — check that before believing it.

## Directly comparable leaderboards
- IDRiD grand challenge (Disease Grading sub-challenge) — same corpus, same task, published
  protocol. The most directly comparable numbers that exist for this thesis.
- APTOS 2019 and EyePACS 2015 Kaggle leaderboards — note these are ranked on **QWK**, which
  is another reason to make QWK primary (`PROTOCOL.md` §4).

## Reproduce before acting
Any number from a paper, from the owner, or from a reviewing agent gets reproduced before a
GPU-hour is spent on it.
