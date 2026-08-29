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

### The default threshold is itself a finding

Both models shipped with **0.5** on every ordinal cut — the number you get from `sigmoid > 0.5`,
not from anything about the grading scale. That arbitrary default was silently determining our
per-class results and very nearly our architecture conclusion.

It deserves stating explicitly because of how invisible it was: no configuration file named it,
no experiment varied it, and it never appeared in a results table. **An untuned default is a
hyper-parameter that has been chosen, not one that has been avoided.** Holding it fixed across
two models does not control for it — it confounds them jointly, which is why the E11-vs-E08
comparison looked like a representational result until both were moved off it.

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

## F4 — How much local labelled data recalibration needs: about 200 images

### Why this was measured

F3 established that moving the decision cut-points is worth several times more than changing
the backbone and costs nothing. That is only a recommendation if we can say **how much
labelled local data it takes**. "Recalibrate before deployment" is advice; "label 200 images
and you recover most of it, below 100 you are as likely to make it worse" is something a
clinic can act on.

### Design

For each sample size *n*, draw *n* images at random as the local labelled sample, fit ordinal
cut-points on those *n* alone (maximising macro-recall), and evaluate on the images **not
drawn** — so every number is out of sample with respect to the cut-points, as a deployment
would be. Repeat over 200 random draws, because *which* images a clinic happens to label is
itself a source of variation and at small *n* it dominates. Compare against the shipped 0.5
thresholds on the same held-out images.

### The curve (E08, DenseNet121; E11 is within a point at every size)

| labelled images | macro-recall gain | Mild recall gain | ΔQWK | **P(recalibration harms)** |
|---|---|---|---|---|
| 25 | +1.58 [−7.84, +6.99] | +51.7 [+3.8, +85.0] | −0.026 | **29.5 %** |
| 50 | +1.96 [−5.92, +6.76] | +54.8 [+6.5, +85.0] | −0.028 | **29.0 %** |
| 100 | +3.49 [−3.09, +7.43] | +67.0 [+15.1, +84.8] | −0.029 | **16.0 %** |
| **200** | **+5.09 [+0.32, +7.28]** | +64.9 [+30.9, +83.7] | −0.021 | **1.5 %** |
| 400 | +6.17 [+3.49, +7.74] | +67.9 [+49.8, +82.6] | −0.018 | **0.0 %** |

Ceiling, from cut-points fitted on half the corpus (~1 800 images): +7.27.

### The sentence a clinician can act on

> **Label about 200 local images.** At that size recalibration recovers roughly **70 % of the
> achievable gain** (+5.1 of +7.3 macro-recall points), the interval clears zero, and the risk
> of making performance worse falls to **1.5 %**. Four hundred images gets ~85 % of the ceiling
> and the risk to zero. **Below 100 images, do not recalibrate**: the expected gain is small
> and roughly **one attempt in four makes the model worse**.

### Three things worth noting alongside it

**The risk column is the operative one at small n, not the mean.** At 25 images the average
gain is positive (+1.58) and would look like an endorsement — while 29.5 % of clinics doing it
would end up worse off than shipping the defaults. A recommendation built on the mean alone
would have been actively harmful.

**Mild recall improves enormously at every size** (+50 to +68 points) but with intervals so
wide at small *n* that the size of the gain is unknowable from a small sample. The direction
is reliable long before the magnitude is.

**Recalibrating for macro-recall costs QWK**, consistently, by 0.017–0.029. This is F1's
trade-off again: the two objectives genuinely disagree, and which one to deploy against is a
clinical decision about the cost of a missed mild case, not a modelling one. The thesis should
present both operating points rather than pick one silently.

Reproduce with `python src/recalibration_curve.py --draws 200`.



---

## F5 — Which metric leads, and why macro-recall must not

*Recommendation pending the supervisor's decision; the evidence below is settled either way.*

### The question

F1 and F4 showed that recalibrating for macro-recall recovers Mild recall from 5.4 % to
77.8 %. That invites making macro-recall the primary metric, on the reasoning that in a
screening application a missed Mild case is the clinically costly error.

### The measurement that answers it

The decision this system supports is binary: **refer if grade ≥ 2**. Measured on the same
3 662 external images:

| operating point | referable sensitivity | specificity | **missed referable** | false referrals |
|---|---|---|---|---|
| shipped (0.5 cuts) | **99.53 %** | 84.28 % | **7 of 1 487** | 342 of 2 175 |
| tuned for QWK | 96.64 % | 87.03 % | 50 of 1 487 | 282 of 2 175 |
| **tuned for macro-recall** | **80.36 %** | 95.36 % | **292 of 1 487** | 101 of 2 175 |

Paired against the shipped thresholds, macro-recall tuning moves referable sensitivity by
**−19.20 points [−21.22, −17.24]**, significant.

### Why this settles it

**Mild NPDR is not referable.** Under standard screening protocols grade 0 and grade 1 both
receive routine rescreen; referral begins at grade 2. So a Mild → No-DR error does not change
management, while a Moderate → Mild error does.

Macro-recall weights those two errors **equally**. That is the assumption that fails, and its
price here is **285 additional patients with referable disease told they do not need
referral** — bought by recovering recall on a class that does not trigger referral at all.

### The recommended position

1. **Operationally primary: referable-DR sensitivity and specificity, as a pair.** This is the
   decision the system supports. Neither may be quoted alone, since either is trivially
   maximised by ignoring the other.
2. **Primary grading metric: QWK**, unchanged. It encodes that a two-grade error costs more
   than a one-grade error, matching the clinical ordering. Macro-recall asserts they cost the
   same, which no screening protocol agrees with.
3. **Macro-recall: a diagnostic, never an objective.** It is what exposed the Mild collapse
   and earns its place for that. It is a lens, not a target.

### Two things that survive the disagreement

**The Mild collapse still matters — just not for referral.** It matters for progression
monitoring and for any use whose output is "this patient has early disease". The referral
numbers being excellent must not be allowed to imply the model is fine at grading.

**Our best referral operating point is an accident and should stop being one.** The 99.53 %
sensitivity comes from `sigmoid > 0.5` — a default nobody chose, which tuning for QWK actively
*degrades* to 96.64 %. The threshold should be selected deliberately on validation data
against a stated sensitivity target, and reported as having been selected. At present we are
benefiting from luck and calling it a result, which is the same category of error as §20's
provenance failures: a number whose origin nobody checked.

---

## F6 — A deployment recommendation evaluated only by its mean can be harmful

The methodological point behind F4, stated generally because it outlives this task.

At n = 25 labelled images, recalibration's mean effect is **+1.58 macro-recall points** — a
number that reads as an endorsement and would, on its own, justify the recommendation
"recalibrate on whatever you can label". The distribution says otherwise: **29.5 % of adopters
end up worse than shipping the defaults.**

**A mean effect is silent about the variance across adopters, and at small samples that
variance is the entire story.** The quantity a clinic needs is not "how much does this help on
average" but "how likely is this to hurt me" — and those two questions have opposite answers
here at every size below 100.

The general rule: **any recommendation about what a deployer should do must be evaluated over
the distribution of deployers, not at its mean, and must report the probability of harm.** A
study reporting only the average would have published advice that is actively damaging to
roughly a third of the clinics following it.

This is the same shape as the project's other recurring failure — reporting a point estimate
without the interval that governs whether it means anything — applied to a decision rather
than to a measurement.

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
