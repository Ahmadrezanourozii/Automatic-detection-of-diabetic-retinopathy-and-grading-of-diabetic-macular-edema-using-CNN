# FINDINGS.md

Results that are *about the problem* rather than about a run. `EXPERIMENTS.md` records what
each experiment did; this file records what the project learned that belongs in the thesis as
an argument, with the numbers that support it.

---

## F1 — The external Mild collapse is calibration, not capacity, and QWK conceals it

### The observation

Mild-NPDR recall falls from 36–45 % internally to **5–6 %** on APTOS, while every other class
holds within about a quarter of its internal value. A larger backbone does nothing for it.

| external ÷ internal recall | No DR | **Mild** | Moderate | Severe | Proliferative |
|---|---|---|---|---|---|
| E08 (DenseNet121) | 1.14 | **0.13** | 0.87 | 0.96 | 1.15 |
| E11 (EfficientNet-B3) | 1.11 | **0.15** | 0.74 | 0.88 | 1.02 |

This is not general minority-class degradation under domain shift — Proliferative, the rarest
class of all, transfers at 1.02–1.15. It is one class, collapsing eight-fold.

### The test

Two explanations are distinguishable by experiment. **Capacity:** the model cannot separate
Mild from its neighbours in this corpus. **Calibration:** it ranks Mild correctly but the
grade boundaries sit in the wrong place, so Mild falls on the far side of a cut.

Ordinal cut-points were fitted on one half of APTOS and applied to the other, and swapped
(2-fold cross-fitting, so no image influences its own cut-points). Fitted twice, under two
objectives, because they answer different questions.

| APTOS, n = 3 662 | No DR | **Mild** | Moderate | Severe | Prolif | QWK | macro-recall |
|---|---|---|---|---|---|---|---|
| as shipped (0.5 cuts) | 98.2 % | **5.4 %** | 61.8 % | 65.3 % | 60.3 % | 0.903 | 0.582 |
| recalibrated for QWK | 97.5 % | **19.5 %** | 83.9 % | 39.9 % | 59.0 % | **0.908** | 0.599 |
| recalibrated for macro-recall | 97.9 % | **77.8 %** | 40.2 % | 46.6 % | 60.0 % | 0.887 | **0.645** |

### What it shows

**1. The information is there. Mild recall goes from 5.4 % to 77.8 % by moving cut-points
alone** — no retraining, no new data, no architectural change. The representation separates
Mild on this corpus perfectly well; the decision boundaries were wrong for it. This is the
sharpest available evidence for the project's standing claim that **ranking transfers across
corpora and calibration does not**.

**2. QWK conceals the failure, and optimising QWK does not fix it.** Recalibrating to maximise
QWK lifts Mild only to 19.5 %, because a Mild image called Moderate is a one-grade error and
QWK charges very little for it. Recovering Mild costs QWK: 0.903 → 0.887. So the headline
external number is *not merely silent* about this class — the metric it is built on actively
prices against fixing it.

This is a real limitation of the metric this project chose as primary, and it should be stated
as such rather than discovered by an examiner. The choice of QWK remains right for the reasons
in `PROTOCOL.md` §4 — it is what makes two-grade errors expensive, and it is what revealed the
pretraining gain that accuracy could not see. But **it must be reported alongside per-class
recall, and never alone.**

**3. Internally, Mild is capacity-limited; externally, it is calibration-limited.** The same
diagnostic on the internal out-of-fold predictions:

| internal (E11, n = 1 362) | No DR | **Mild** | Moderate | Severe | Prolif | QWK |
|---|---|---|---|---|---|---|
| as shipped | 88.5 % | **36.2 %** | 83.0 % | 74.5 % | 59.3 % | 0.894 |
| recalibrated for macro-recall | 85.0 % | **45.2 %** | 65.9 % | 76.5 % | 66.1 % | 0.891 |

Internally, recalibration buys Mild only 9 points and it stays under half. So the two settings
have *different* Mild problems: at home the model genuinely struggles to see microaneurysms
(consistent with E09, where resolution was worth +18.6 points on exactly this class); abroad it
sees them and mislabels them. Conflating the two would have led to the wrong fix in both.

### What follows for the thesis

* **Report per-class recall beside every headline.** "What fraction of Mild cases does it
  catch?" has the answer 5 %, and the number that must be quoted is that one, not the 0.903.
* **Deployment needs a small labelled local sample** to set cut-points. This was already the
  recommendation from the over-grading analysis; the Mild result quantifies what it buys —
  from 5 % to 78 % on the class most likely to be a screening programme's earliest catch.
* **The recalibrated figures are a diagnostic, not a result.** Cut-points were fitted on APTOS
  labels, so they measure the ceiling recoverable *given* labelled local data. They are not
  external validation and appear in no results table.

Reproduce with `python src/analyse_mild.py --run runs/E11X`.

---

## F2 — Messidor-1 cannot serve as the external DME test set, and the reason is measurable

Recorded in full in `data/DATASETS.md`. In short: 1 057 of its 1 200 images are already in the
development pool (exact by filename, confirmed by content at dHash distance 5–6 and NCC
1.0000); the 143 survivors are 78 % one site; site predicts DME grade at χ² p = 1.5 × 10⁻¹⁴;
and a simulation of our best model on those 143 images gives a QWK interval of **± 0.073** —
wider than the entire range of differences this project has been trying to resolve.

**"No external DME validation" is therefore a declared limitation with a measured reason**,
which is worth more at a defence than silence. The gap is a DME corpus graded by the IDRiD
criterion with no Messidor lineage; none has been identified.
