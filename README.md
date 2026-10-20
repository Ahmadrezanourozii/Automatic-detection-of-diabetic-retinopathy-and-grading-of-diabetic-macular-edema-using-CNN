# Decision rules and representations in external validation of diabetic retinopathy grading

Code, frozen data splits, analysis plans and archived per-image predictions for the article

> **Separating decision-rule failures from representation failures in external validation of
> diabetic retinopathy grading models**
> Ahmadreza Nourozi (FAU Erlangen-Nürnberg) and Alireza Chegeni (University of Tehran).
> Manuscript in preparation; the LaTeX source is in [`paper/`](paper/).

The project began as automatic detection of diabetic retinopathy (DR) and grading of diabetic
macular oedema (DME) from fundus photographs with convolutional networks, and grew into a study of
why such models fail on new populations: how much of an external failure lies in the ordinal
decision rule that turns scores into grades, and how much in the learned representation.

## What is in the repository

| path | contents |
|---|---|
| `src/` | training (`train.py`, `model.py`), data loading (`corpora.py`, `splits.py`), external evaluation (`eval_external.py`) and every analysis behind the article (`paper_*.py`) |
| `data/splits/` | the frozen development split and the split used for the reverse-direction model |
| `results/predictions/` | per-image out-of-fold and external predictions of every model |
| `docs/` | evaluation protocol, analysis plans (`PAPER_EXPERIMENTS_preregistration.md`, `PAPER_REVISION_preregistration.md`) and generated tables (`docs/generated/`) |
| `kaggle/`, `kaggle2/` | the notebooks that ran each training and evaluation job, each pinned to a code commit |
| `paper/` | manuscript, supplementary material, cover letter and figures |
| `PROTOCOL.md`, `EXPERIMENTS.md`, `FINDINGS.md`, `ISSUES.md` | research record: protocol, run ledger, findings and the errors found along the way |

A fuller map is in [`FILE_MAP.md`](FILE_MAP.md).

## Data

All images are from public datasets, which must be obtained from their providers: IDRiD,
Messidor-2 (with the adjudicated grades of Krause et al.), APTOS 2019, EyePACS 2015, DDR, DeepDRiD
and SUSTech-SYSU. No image is redistributed here.

## Reproducing the numbers in the article

Every number in the manuscript is a key in `paper/numbers.tex`, generated from the archived
predictions; none is typed by hand.

```bash
# 1. collect kernel outputs into results/predictions/ (registry.json lists every model)
python src/paper_collect.py --search <directories with kernel outputs>
# 2. analyses (each writes JSON to docs/generated/paper/)
python src/paper_phase0.py   --labels <label dir> --corpus APTOS   # also DDR, DeepDRiD, SUSTech
python src/paper_transfer.py --labels <label dir> --corpus APTOS
python src/paper_revision.py --labels <label dir>
python src/paper_seeds.py    --labels <label dir>
python src/paper_reverse.py
# 3. numbers, figures and tables, then LaTeX
python src/paper_numbers.py && python src/paper_figures.py
python src/paper_tables_tex.py && python src/paper_supp_tex.py
cd paper && pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

`src/paper_numbers.py` stops with an error if the manuscript uses a number that is not in the ledger.

## Citation

See [`CITATION.cff`](CITATION.cff).
