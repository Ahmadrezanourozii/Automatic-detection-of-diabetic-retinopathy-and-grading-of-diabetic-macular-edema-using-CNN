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

### A claim I made here and then had to withdraw

I originally wrote this section as "an independent confirmation": E11's larger backbone moves
Mild by −0.6 points, indistinguishable, so capacity cannot be the problem. **That was wrong,
and the test in F3 is what showed it.**

The −0.6 holds only *at the shipped operating point*. Once both models are recalibrated for
macro-recall, E11 is **+7.0 points on Mild [+2.7, +11.4], significant**. So the larger
backbone does help this class — the help is simply invisible until the cut-points are fixed.

What survives, and it is still the main claim: **calibration is the dominant term and capacity
is a real but second-order one.** Recalibration alone moves Mild from 5.4 % to 77.8 %, a
fourteen-fold change; the backbone adds 7 points on top of that. Reporting the second without
the first would have been the same error in a different class — which is exactly what this
file was written to prevent.

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

## F3 — Per-class comparisons between models are dominated by cut-point placement, not representation

### Why this test was run

E11 (EfficientNet-B3) beat E08 (DenseNet121) on APTOS by +8.1 points on Proliferative while
being **worse** by −2.4 on Moderate, a well-populated class. A better representation should
not make a well-populated class worse. The alternative explanation is that neither number is
about representation: E11 simply places its Moderate/Proliferative boundary further toward
Proliferative, and the two effects are one effect seen from both sides.

Distinguishable at zero GPU cost with machinery already built: apply the same cross-fitted
cut-point tuning to **both** models on APTOS, then compare per-class recall again.

### The result

| E11 − E08, per class | as shipped | both tuned for QWK | both tuned for macro-recall |
|---|---|---|---|
| No DR | +1.2 **SIG** | −1.3 **SIG** | +1.3 **SIG** |
| Mild | −0.5 n.s. | +1.4 n.s. | **+7.0 SIG** |
| **Moderate** | **−2.4 SIG** | **+6.0 SIG** | −8.8 **SIG** |
| Severe | +1.0 n.s. | **−10.0 SIG** | −0.6 n.s. |
| **Proliferative** | **+8.1 SIG** | **+11.2 SIG** | +3.7 n.s. |

**The Moderate regression reverses sign under tuning** — from −2.4 significantly against E11 to
+6.0 significantly for it. It was a boundary placement, not a representational deficit, exactly
as suspected. Severe swings by 11 points across operating points. Proliferative ranges from
+3.7 (n.s.) to +11.2 (significant) depending only on where the cuts sit.

**Not one per-class difference is stable across operating points.** Three of the five change
significance, and one changes sign.

### And the headline difference does not survive either

| E11 − E08, aggregate | QWK difference | verdict |
|---|---|---|
| as shipped | +0.0063 [+0.0009, +0.0116] | **significant** |
| both tuned for QWK | +0.0042 [−0.0013, +0.0096] | indistinguishable |
| both tuned for macro-recall | +0.0016 [−0.0054, +0.0088] | indistinguishable |

The one significant external advantage E11 had **disappears once both models are compared at
matched operating points**. It was largely a calibration difference between two models that
happened to ship with the same arbitrary 0.5 cut-points.

### The magnitude comparison that settles the priority

| intervention | effect on macro-recall (APTOS) |
|---|---|
| recalibrating E08's cut-points | **+7.27 pts [+5.53, +8.99]** |
| recalibrating E11's cut-points | **+6.34 pts [+4.63, +8.10]** |
| replacing DenseNet121 with EfficientNet-B3 | +1.48 pts [−0.06, +2.96], n.s. |

**Moving the decision boundaries is worth roughly five times as much as changing the backbone,
and costs nothing.** Nine and a half GPU-hours bought E11; the recalibration is a coordinate
ascent over four numbers.

### What this changes

1. **Do not report per-class recall differences between models without fixing the operating
   point first.** They are not measuring what they appear to measure.
2. **The E11-vs-E08 backbone conclusion is weakened.** The internal +0.029 QWK advantage stands
   — it was measured out-of-fold on matched folds — but externally, at matched operating
   points, the two models are indistinguishable. Finishing E11's folds 3 and 4 is now clearly
   not worth 3.2 h.
3. **This is the same error as F1, one class over, and it was nearly written into the thesis
   as a capacity finding.** F1 caught it for Mild by asking whether calibration explained the
   collapse. F3 caught it for Proliferative and Moderate by asking the same question of a
   comparison rather than of a single model. The general rule: **before attributing a
   difference to representation, check that it survives matched calibration.**

Reproduce: the analysis is in the commit that added this section; predictions are in
`runs/E08X2/` and `runs/E11X/`.

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
