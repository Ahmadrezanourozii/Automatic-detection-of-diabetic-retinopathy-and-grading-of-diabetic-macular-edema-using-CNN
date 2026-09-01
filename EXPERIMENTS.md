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
| **E07** | 2026-08-26 | `c79dac6` | EfficientNet-B3 and a 60-epoch schedule beat DenseNet121 at 30 | E05 + efficientnet_b3, 60 epochs, no pretraining | 73.8 %¹ | 0.837¹ | — | — | — | folds 0–3 only | **killed at the 12 h wall clock after 4/5 folds** | `runs/E07/` |

¹ folds 0–3 only (n = 1 811); the run was cancelled during fold 4.
| **E12def / E12nat** | 2026-08-29 | `e3cb4d1` | Source control: the same eyes at the same size from two different files should give the same model | identical config, `--messidor-source default-only` vs `native-only`, folds 0–2 | 72.2 / 69.6 % | 0.858 / 0.856 | 87.3 / 87.3 % | 0.891 / 0.901 | — | see below | **all four differences indistinguishable** | `runs/E12def/`, `runs/E12nat/` |
| **E11** | 2026-08-26 | `ebe8a61` | EfficientNet-B3 **plus** EyePACS pretraining beats DenseNet121 plus the same | E08 + efficientnet_b3, 30 ep, **folds 0–2 only** | **78.1 %**¹ | **0.894**¹ | **87.6 %**¹ | **0.902**¹ | — | DR [75.8, 80.3] · DME [83.8, 91.1] | **best DR; +0.029 QWK vs E08, significant** | `runs/E11/` |

¹ folds 0–2 only (DR n = 1 362, DME n = 314). Sized to fit the 12 h window after E07 was
killed; finished in 9.49 h. The remaining two folds are queued so the headline can be quoted
on the same 5-fold basis as every other run.
| **E10** | 2026-08-26 | `4413fc7` | Native-resolution Messidor-2 at 640 px lifts the Mild class | E08 + native mirror, size 640 (**effectively 560 — see below**) | **74.4 %** | **0.868** | **87.2 %** | **0.899** | — | DR [72.6, 76.3] · DME [84.3, 89.9] | best DME so far | `runs/E10/` |
| **E09** | 2026-08-26 | `cebfc20` | Halving the resolution tests whether Mild recall is resolution-bound | E08 at 224 px instead of 448 | 69.0 % | 0.820 | 84.3 % | 0.866 | — | DR [67.2, 70.9] | resolution test | `runs/E09/` |
| **E08** | 2026-08-26 | `98b5ece` | EyePACS pretraining + a longer schedule; first external validation | E06 + 40 epochs (from 25), weights persisted | **74.2 %** | **0.860** | 84.1 % | **0.884** | 0.646 / 0.715 | DR [72.5, 76.0] · DME [81.0, 87.2] | best DR so far | `runs/E08/` |
| **E06** | 2026-08-26 | `52d5ef4` | EyePACS pretraining then fine-tune on the dev pool | E05 + 35 126-image EyePACS pretraining (4 ep), 25 ep fine-tune | **71.9 %** | **0.847** | **86.0 %** | **0.879** | 0.631 / 0.758 | DR [70.1, 73.8] · DME [83.1, 88.8] | **QWK +0.064 vs E05, significant** | `runs/E06/` |
| **E05** | 2026-08-26 | `219881c` | Ordinal heads + Messidor-2 partial labels beat the frozen probe | first real training run; 5-fold grouped CV, 448 px, 30 ep, EMA, TTA | **70.4 %** | **0.783** | **84.1 %** | **0.874** | 0.547 / 0.736 | DR [68.6, 72.3] · DME [81.0, 87.0] | **yes vs floor** | `runs/E05/` |
| **E01·rgb** | 2026-08-25 | `075d83c` | Frozen ImageNet features beat the majority floor on both heads | baseline (plain RGB, 224 px, linear probe) | 47.6 % | 0.584 | **71.8 %** | **0.678** | 0.427 / 0.582 | DR [37.9, 57.3] · DME [63.1, 80.6] | **yes vs floor** | `runs/E01/` |
| **E01·green_clahe** | 2026-08-25 | `075d83c` | *(same)* | + thesis chain (green→CLAHE→blur), ImageNet norm | **51.5 %** | **0.654** | 68.9 % | 0.602 | 0.391 / 0.536 | DR [41.7, 60.2] · DME [60.2, 77.7] | **yes vs floor**; **no vs rgb** | `runs/E01/` |
| **E01·green_clahe_raw01** | 2026-08-25 | `075d83c` | *(same)* | + thesis chain exactly as old code fed it ([0,1], no ImageNet norm) | 47.6 % | 0.650 | 65.0 % | 0.610 | 0.399 / 0.536 | DR [37.9, 57.3] · DME [56.3, 74.8] | **yes vs floor**; **no vs rgb** | `runs/E01/` |

DME columns are the **ungated primary** definition (n=103, floor 46.6 %). Gated DR≥1 numbers
are in `runs/E01/results.json` and are discussed in the note below — **no variant beats the
gated floor**. All pairwise variant comparisons: `runs/E01/comparisons.json`.








### All-pairs comparison, and the negative result that matters

`src/compare_runs.py` now generates every pairwise paired bootstrap into
`docs/generated/comparisons.md`, restricted to the folds both runs completed (so anything
involving E07 covers folds 0–3 only, and says so). 41 comparisons.

**DR — the ordering, with ties named as ties:**

| comparison | DR QWK difference | verdict |
|---|---|---|
| E08 − E06 | +0.012 [−0.002, +0.028] | **indistinguishable — E08 does not beat E06** |
| E06 − E09 (448 vs 224) | +0.027 [+0.010, +0.045] | significant |
| E08 − E09 (448 vs 224) | +0.040 [+0.024, +0.056] | significant |
| E08 − E07 | +0.020 [+0.001, +0.041] | significant |
| E08 − E05 | +0.077 [+0.056, +0.099] | significant |
| E06 − E05 | +0.064 [+0.042, +0.087] | significant |

So the current best is **E06 and E08, statistically tied**, and both clearly above E05, E07
and E09.

**DME — nothing has worked.** Of the ten DME comparisons, **eight are indistinguishable**,
and the two that are significant both involve E07 on folds 0–3 (n = 415), where E07 is the
*worse* run. Stated plainly:

> **No intervention in this programme has significantly improved 3-class DME on QWK, the
> primary metric.** EyePACS pretraining, the longer schedule and the bigger backbone all move
> DME QWK by an amount whose interval contains zero.
>
> *Amended 2026-08-26 after E10:* raising the effective resolution from 448 to 560 px did
> significantly improve DME **accuracy** (+3.04 pts [+0.39, +5.62]) while leaving **QWK**
> indistinguishable. The negative result therefore stands for the primary metric and is
> broken for the secondary one — a distinction worth keeping, because accuracy moving without
> QWK means one-grade corrections rather than the two-grade errors that matter clinically.

This is worth saying clearly rather than letting the DR gains carry the whole story. Two
readings, and they are distinguishable by experiment rather than argument:

1. **The DME task is already near the ceiling of what 516 exactly-labelled images can
   measure.** The 95 % interval on its QWK is roughly ±0.03, so an improvement smaller than
   that is invisible no matter what we do. This predicts that acquiring Messidor-1 — which
   would roughly triple the 3-class DME evaluation set — should shrink the interval enough
   for real differences to appear.
2. **The interventions genuinely do not help DME.** EyePACS carries no DME labels at all, so
   it could only ever have helped through shared features; resolution does not bind for DME
   (E09 confirmed this directly, unlike for DR); and neither backbone nor schedule addresses
   what the DME head lacks, which is exudate *position* relative to the macula.

Reading 2 has a cheap test already queued: the macula-centred crop (`IDEAS.md` I07), which
uses the fovea coordinates IDRiD ships for all 516 images and is the only queued idea that
addresses position rather than capacity.

### E07 — data beats architecture, and the run did not fit the budget

E07 was cancelled at Kaggle's 12-hour wall clock having finished 4 folds of 5. Comparing it
on those four folds only (n = 1 811), against runs restricted to the same folds:

| run | configuration | DR accuracy | DR QWK |
|---|---|---|---|
| E05 | DenseNet121, 30 ep, no pretraining | 70.3 % | 0.782 |
| **E07** | **EfficientNet-B3, 60 ep, no pretraining** | 73.8 % | 0.837 |
| **E08** | DenseNet121, 40 ep, **+ EyePACS pretraining** | 73.7 % | **0.857** |

| paired comparison | DR accuracy | DR QWK | verdict |
|---|---|---|---|
| E07 − E05 (backbone + schedule) | +3.5 [+1.5, +5.4] | +0.055 [+0.033, +0.076] | **both significant** |
| E07 − E08 (vs pretrained DenseNet) | +0.1 [−2.0, +2.3] n.s. | **−0.020 [−0.041, −0.001]** | **significant, against E07** |

**A bigger backbone helps — and 35 000 extra pretraining images on a smaller one helps more.**
*(Refined after E11: they are complementary. See the E11 section — the combination beats both,
and the phrase "data beats architecture" was too broad.)*
EfficientNet-B3 with ImageNet initialisation is significantly *worse* than DenseNet121 with
EyePACS pretraining, on the metric that leads. The two are indistinguishable on accuracy,
which is again the pattern from E06: extra data buys ordinal quality that accuracy cannot see.

**And the cost was lopsided.** E07 consumed the full 12-hour session and returned four folds;
E08 finished five in about four hours. Pretraining also converges faster — E08's best epochs
were 16–22 of 40, while E07's were 32–58 of 60 — so the cheaper configuration is also the one
that needs fewer epochs.

**Budget lesson, recorded because it cost a session.** A run must be *sized to fit the
window*, not merely made resumable. Resumability saved the four completed folds, but a
12-hour job in a 12-hour box will be cut off in the middle of something. Long
configurations should be split across sessions by fold from the start.

**What this makes the next experiment.** Not "try a bigger backbone" — that has been answered
in isolation. The open question is whether EfficientNet-B3 **plus** EyePACS pretraining beats
DenseNet121 plus the same pretraining, and because pretraining converges faster it can be run
at ~30 epochs and fit the window.





### E12def / E12nat — the source control passes, and I20 is interpretable

The native-resolution images available to this project come from Messidor-1 TIFFs while our
512 px copies come from a Messidor-2 mirror, so a resolution experiment would otherwise change
pixel count, file source and compression history together. The control holds everything fixed
except the source: the same 1 057 eyes, the same labels, the same frozen folds, the same
512 px training resolution — only the file the pixels came from differs.

Both arms loaded exactly 1 057 Messidor-2 images and evaluated on the same 953 (DR) and 314
(DME), verified by uid set equality before the comparison ran.

| native − default, paired over the same eyes | difference | 95 % interval | verdict |
|---|---|---|---|
| DR accuracy | −2.64 pts | [−5.56, +0.31] | indistinguishable |
| DR QWK | −0.0016 | [−0.0201, +0.0173] | indistinguishable |
| DME accuracy | −0.01 pts | [−3.18, +3.18] | indistinguishable |
| DME QWK | +0.0103 | [−0.0205, +0.0421] | indistinguishable |

**In aggregate the source is not a confound.** QWK is flat on both heads and accuracy's
interval contains zero, so **I20 may proceed for aggregate metrics** — a QWK difference it
finds can be attributed to resolution.

**But the per-class picture is not clean, and I first reported this too favourably.**

**1. The Moderate cell survives matched calibration — I predicted it would not.** I originally
wrote that the single significant cell (Moderate, −6.8 [−11.6, −2.2]) was "the likely reading
is F3's phenomenon again — a boundary shift, not a source effect", and flagged that the check
had not been run. **It has now been run, and the prediction was wrong.**

| Moderate recall, native − default | difference | verdict |
|---|---|---|
| at shipped cut-points | −6.8 [−11.6, −2.2] | significant |
| with both arms recalibrated for QWK | **−6.5 [−11.9, −0.7]** | **still significant** |

The effect is essentially unchanged by recalibration, which is what a genuine representational
difference looks like and what a boundary artefact does not — contrast F3, where the same test
made the equivalent cell *reverse sign*.

**2. Recalibration introduced two further significant cells** rather than removing any: No DR
−8.6 [−12.4, −4.7] and Mild +14.2 [+5.0, +23.4]. Aggregate QWK remained indistinguishable in
both conditions (−0.0016 shipped, −0.0035 tuned).

**3. Multiple comparisons apply and are not corrected for.** Ten per-class tests were run
across the two conditions; at α = 0.05 roughly one false positive is expected by chance. Three
were significant under tuning, which is more than chance, and the Moderate effect is the one
that replicates *across* conditions — that consistency, not its p-value in isolation, is what
makes it credible.

**The honest conclusion, replacing my first one.** The two sources give models that are
**equally good in aggregate but distribute their errors differently across classes**. So:

* **Aggregate claims from I20 (QWK, overall accuracy) are attributable to resolution.**
* **Per-class claims from I20 carry a source caveat** and cannot be cleanly attributed, because
  we now know the source alone shifts the per-class error distribution by several points on at
  least one class.

That is a weaker licence than "the source is not a confound", which is what I said before
running the check I had myself flagged as outstanding.

**2. The control validates the mirror only at 512 px.** It shows the two sources are equivalent
*after both are reduced to 512*. It does not show they are equivalent at native resolution —
which is exactly what I20 varies. If I20 finds a difference, we will not be able to separate
"resolution helps" from "a source effect that only appears above 512 px" without a third arm
at native resolution from a *second* native source, which we do not have. That limitation
belongs with any I20 result.

### Planned: the resolution test needs a source control first

The native-resolution images available to this project come from **Messidor-1 TIFFs**, while
our current 512 px copies come from a **Messidor-2 mirror**. A run that simply raises the
resolution therefore changes three things at once: the pixel count, the file source, and the
codec and compression history (uncompressed TIFF from ADCIS versus a re-encoded PNG mirror).
Any difference it produced could not be attributed.

**The control, which runs first.** Take the 1 057 images present in both, downsample the
Messidor-1 originals to **exactly the 512 px our pipeline already uses**, and train under the
existing configuration. Compare against the same configuration trained on our existing
Messidor-2 copies, paired over the same folds.

* **If they match** — the source is not a confound, and a subsequent native-resolution run
  measures resolution alone.
* **If they differ** — the resolution experiment is uninterpretable as designed, and the
  difference between two supposedly identical images of the same eye becomes the finding in
  its own right. That would also cast doubt on the Messidor-2 mirror as a data source.

~1.5 h to protect a ~4 h run, and more importantly to protect a result from being
unattributable after the fact. Recorded as `IDEAS.md` I19, gating I20.


### E11X — the external number, and what it does not settle

E11's three folds ensembled with TTA on the same 3 662 held-out APTOS images:

| | E08X (DenseNet, 5 folds) | E11X (EfficientNet-B3, 3 folds) |
|---|---|---|
| accuracy | 73.51 % [72.04, 74.93] | 74.11 % [72.67, 75.59] |
| QWK | 0.8968 [0.8886, 0.9048] | 0.9031 [0.8949, 0.9108] |
| referable-DR sensitivity / specificity | 99.53 % / 84.28 % | 99.26 % / 84.60 % |

**The internal advantage largely does not transfer.** E11 beat E08 internally by **+0.029**
QWK, fold-matched and significant. Externally the gap is **+0.006**, about a fifth the size,
with heavily overlapping intervals.

**Settled by E08X2, which re-ran E08's weights with per-image predictions archived.** It
reproduced E08X's confusion matrix **exactly** — a clean reproducibility check on the whole
inference path — and enabled the paired test:

| E11 − E08, paired over the same 3 662 images | difference | 95 % interval | verdict |
|---|---|---|---|
| accuracy | +0.60 pts | [−0.33, +1.48] | indistinguishable |
| **QWK** | **+0.0063** | **[+0.0009, +0.0116]** | **significant** |
| macro-recall | +1.48 pts | [−0.06, +2.96] | indistinguishable |

So the external advantage is **real but about one fifth of the internal one** (+0.0063 against
+0.029). The interval barely clears zero, and at n = 3 662 — larger than the whole development
pool — this is a well-powered "small but real", not an underpowered "maybe nothing".

**Where the gain comes from, paired per class:**

| class | n | E08 | E11 | difference | |
|---|---|---|---|---|---|
| No DR | 1 805 | 97.0 % | 98.2 % | +1.2 [+0.5, +1.9] | significant |
| **Mild** | 370 | 5.9 % | 5.4 % | −0.6 [−3.5, +2.2] | **indistinguishable** |
| Moderate | 999 | 64.2 % | 61.8 % | −2.4 [−4.8, −0.1] | significant, **against** E11 |
| Severe | 193 | 64.2 % | 65.3 % | +1.0 [−4.1, +6.2] | indistinguishable |
| **Proliferative** | 295 | 52.2 % | **60.3 %** | **+8.1 [+4.4, +12.2]** | significant |

The entire external benefit of the larger backbone is **Proliferative**, plus a little No DR,
partly given back on Moderate. And **Mild is identical and collapsed in both** — an independent
confirmation of `FINDINGS.md` F1: a significantly better backbone moves that class by
*nothing*, because the problem there is not capacity.

**Consequence for the queue.** A +0.006 QWK external gain concentrated in one rare class is a
thin justification for the ~3.2 h needed to finish E11's folds 3 and 4. Deprioritised.

**And then it got thinner.** `FINDINGS.md` F3 re-ran the comparison with both models at
*matched* operating points. The +0.0063 becomes **+0.0042 (n.s.)** when both are tuned for
QWK and **+0.0016 (n.s.)** when both are tuned for macro-recall. The Moderate regression
*reverses sign*. Externally, at comparable cut-points, **the two backbones are
indistinguishable** — the shipped difference was largely a calibration accident of two models
sharing the same arbitrary 0.5 thresholds. The internal +0.029 stands; the external claim
does not.

**The consequence for priorities** is already clear enough to act on: finishing E11's folds 3
and 4 (~3.2 h) makes the internal headline comparable across runs but is unlikely to move the
number that matters at a defence. It is therefore deprioritised behind the source control and
the macula-crop experiment.

**And the more important thing this run surfaced** is not the E11-vs-E08 comparison at all —
it is that Mild recall is 5.4 % on APTOS for *both* models. See `FINDINGS.md` F1.

### E11 — architecture and data are complementary, and my E07 conclusion was too broad

E11 is EfficientNet-B3 **with** EyePACS pretraining — the combination E07 and E08 each had
only half of. Paired bootstrap on the three folds all runs share (DR n = 1 362, DME n = 314):

| comparison | DR accuracy | DR QWK | DME |
|---|---|---|---|
| E11 − E08 (DenseNet + pretraining) | **+4.37 pts [+1.91, +6.83]** | **+0.029 [+0.012, +0.046]** | indistinguishable |
| E11 − E10 (DenseNet + pretraining, 560 px) | **+4.49 pts [+1.91, +6.98]** | **+0.029 [+0.013, +0.046]** | indistinguishable |

**Correction to the E07 verdict.** From E07 I concluded "data beats architecture", because
EfficientNet-B3 *without* pretraining lost to DenseNet121 *with* it. That was true but the
phrasing was too broad, and read as "the backbone does not matter". It does. The accurate
statement is:

> Given a choice between a bigger backbone and 35 000 extra pretraining images, take the
> images. Given both, take both — they are complementary, and the combination is
> significantly better than either alone.

The evidence is now complete on this point: pretraining alone (E06 − E05) is +0.064 QWK;
backbone alone (E07 − E05) is +0.055; and backbone **on top of** pretraining (E11 − E08) is a
further +0.029, all significant. Nothing here supports the idea that architecture is
irrelevant, and I should not have implied it from a single half-configuration.

**DME is unchanged again.** Indistinguishable against both comparators on both metrics, which
keeps the standing negative result intact for the primary metric: the DME head has not been
moved by pretraining, schedule, backbone, or the architecture-plus-data combination. The only
thing that has ever moved it is effective resolution, and only its accuracy (E10).

**Two caveats attached to this result.**

1. **Three folds, not five.** E11 was deliberately sized to fit the wall clock after E07 was
   killed at it, and finished in 9.49 h — inside the ~10 h budget rule. Its headline numbers
   are therefore on a different basis from every other run's. The comparisons above are
   fold-matched and valid; the *headline* is not directly comparable until folds 3 and 4 are
   run, which is ~3.2 h and is queued.
2. **The DR gain does not transfer automatically to the external number.** E08X's APTOS
   result used E08's weights. Whether E11's larger backbone also generalises to an unseen
   corpus is an open question, not an assumption — and worth checking before the headline is
   changed, given E08X already showed that ranking transfers while calibration does not.

### E10 — a 560 px run wearing a 640 px label, and the first DME movement

E10 was flagged as confounded before its numbers landed, and it was — but **not in the way
first described, and the correction matters.** `build_cache()` caps images at 560 px.
E08 trains at 448, so it *downsamples* and sees 448 px of information. E10 trains at 640, so
it *upsamples* and sees 560. E10 therefore carries **more** effective resolution than E08,
not the same: it is a genuine **448-vs-560** comparison wearing a 448-vs-640 label. The
earlier claim that it paid double for "the same information" was my error, corrected in
`ISSUES.md` §18.

Paired bootstrap against E08 over the same 2 260 groups:

| quantity | E08 → E10 | 95 % interval | verdict |
|---|---|---|---|
| DR accuracy | +0.15 pts | [−1.7, +1.9] | indistinguishable |
| DR QWK | +0.008 | [−0.006, +0.022] | indistinguishable |
| **DME accuracy** | **+3.04 pts** | **[+0.39, +5.62]** | **significant** |
| DME QWK | +0.014 | [−0.008, +0.038] | indistinguishable |

**Two readings, stated precisely.**

**DR resolution saturates.** E09 showed 224 → 448 is worth +0.040 QWK, significant. E10 shows
448 → 560 is worth +0.008, indistinguishable. So the benefit of resolution for DR is real but
**flattens somewhere between 224 and 448 px** rather than continuing upward. That is a more
useful finding than "higher is better", and it means the expensive 768 px runs on the backlog
are unlikely to pay.

**The DME negative result is now partly broken — on the secondary metric only.** This is the
first significant DME improvement in the programme. Being exact about what it does and does
not overturn: the standing negative result concerned **QWK**, the primary metric, and QWK
remains indistinguishable [−0.008, +0.038]. What moved is **accuracy**, the secondary metric,
by +3.04 points. Accuracy rising while QWK does not means the extra correct calls are
mostly one-grade corrections rather than the two-grade errors QWK charges for.

**What it is not evidence for.** The native-resolution Messidor-2 mirror was attached, and the
log confirms it was found and used — but every image from it was still downscaled to 560 in
the cache, so its 2240 × 1488 source contributed at most 560 px. The real question — does
native-resolution source data lift the Mild class — **remains open**, and needs a run with
`cache_size >= size`. `build_cache()` now refuses to start such a mislabelled run.

### E09 — resolution binds for DR and not for DME, and the gain scales with lesion size

A single-factor test: E08's exact configuration at **224 px instead of 448**. Paired
bootstrap over the same 2 260 groups:

| | difference (448 − 224) | 95 % interval | verdict |
|---|---|---|---|
| DR accuracy | **+5.2 pts** | [+3.2, +7.2] | **significant** |
| DR QWK | **+0.040** | [+0.024, +0.056] | **significant** |
| DME accuracy | −0.2 pts | [−3.1, +2.7] | indistinguishable |
| DME QWK | +0.019 | [−0.011, +0.051] | indistinguishable |

**The per-class breakdown is the finding.** The benefit of resolution grows monotonically as
the lesion that defines the grade gets smaller:

| grade | defining lesion | 448 px | 224 px | gain |
|---|---|---|---|---|
| **Mild** | microaneurysms (~30–100 µm) | 45.4 % | 26.8 % | **+18.6** |
| Moderate | haemorrhages, exudates | 73.4 % | 64.1 % | +9.3 |
| Severe | venous beading, IRMA | 67.3 % | 64.3 % | +3.0 |
| No DR | — | 85.1 % | 84.0 % | +1.1 |
| Proliferative | neovascularisation (large) | 45.4 % | 48.5 % | −3.1 (n=97) |

And DME is untouched, which fits: hard exudates are bright and comparatively large, so
halving the resolution does not remove them.

**This settles the data-acquisition question that was deliberately left open.** The Mild
class *is* resolution-bound, so acquiring a full-resolution Messidor-2 mirror
(`IDEAS.md` I10b) is worth doing — it holds 270 of the 295 Mild images, and our current
mirror caps them at 512 px. Testing 640 or 768 px *before* that acquisition would move
almost nothing, because only IDRiD's 25 Mild images could benefit.

**Why this is worth a paragraph in the thesis rather than a line in a hyper-parameter
table.** The resolution was not chosen by sweeping and keeping the best. It was chosen after
a prediction — "the errors concentrate in the grade defined by the smallest lesion, so
resolution should bind there and nowhere else" — was tested and held, with the effect sizes
ordered exactly as the clinical definitions predict. That is an explanation, not a tuning
result.


### E08X — the external claim, verified rather than asserted

The APTOS figure is the most load-bearing number in this project, and it came from a run
whose output directory was simultaneously carrying E08's `results.json` (`ISSUES.md` §20).
That is reason enough not to take it on trust. `src/verify_external.py` re-derives it from
the archived artifact; **12 checks, 0 failures**:

| check | result |
|---|---|
| cohort size equals the APTOS corpus | 3 662 = 3 662 |
| per-class supports match the corpus labels exactly | [1805, 370, 999, 193, 295] both |
| no development image appears in the external cohort | 0 shared of 2 260 dev / 3 662 external |
| accuracy regenerates from the confusion matrix | 0.735117 = 0.735117 |
| QWK regenerates | 0.896837 = 0.896837 |
| macro-F1, majority floor, per-class recall, support | all regenerate |
| referable-DR sensitivity / specificity regenerate | 99.53 % / 84.28 % both |

**The intervals are verified too — no re-run was needed.** A group bootstrap resamples groups
i.i.d. with replacement. APTOS publishes no patient identifiers, so **every group is exactly
one image** (3 662 images, 3 662 groups, verified). A group bootstrap is then an image
bootstrap, which is exactly resampling (true, predicted) pairs from the empirical joint
distribution — and the confusion matrix *is* that distribution. So the interval is fully
determined by what was already archived:

| | recomputed from the matrix | archived | Monte-Carlo sd |
|---|---|---|---|
| accuracy 95 % | [72.064, 74.932] | [72.037, 74.932] | ± 0.022 |
| **QWK 95 %** | **[0.889, 0.905]** | **[0.889, 0.905]** | ± 0.0001 |

This establishes the interval is **correct**, not that it is bit-identical: the realized draws
depend on row order, which a confusion matrix does not preserve. The comparison is therefore
against Monte-Carlo error, estimated from three independent seeds at 8 000 draws each and
reported beside the figure. **14 checks, 0 failures.**

**The quotable claim, now fully verified:**

> On 3 662 APTOS images never seen in training: accuracy **73.5 %** (95 % CI 72.0–74.9),
> **QWK 0.897** (95 % CI 0.889–0.905), against a 49.3 % majority-class floor.

**The shortcut has a precondition, and it will expire.** It works *only* because every group
is one image. The moment Messidor-1 arrives, or patient identifiers are recovered for any
corpus, groups become multi-image and the confusion matrix stops determining the interval.
`verify_external.py` tests that precondition and says so explicitly when it fails, and
`eval_external.py` now archives per-image predictions and group ids regardless.

### E08X — external validation on APTOS: the ranking transfers, the calibration does not

**The first external number this project has.** E08's five folds, ensembled by logit
averaging with flip TTA, scored on 3 662 APTOS images — a different population, different
cameras, different graders — held out since the protocol was frozen and never touched.

| | external (APTOS) | internal (pooled OOF) | difference |
|---|---|---|---|
| n | 3 662 | 2 260 | |
| majority floor | 49.3 % | 52.4 % | |
| DR accuracy | 73.5 % [72.0, 74.9] | 74.2 % | **−0.7 pts** |
| **DR QWK** | **0.897** | 0.860 | **+0.037** |
| referable-DR sensitivity | **99.5 %** | 87.2 % | |
| referable-DR specificity | 84.3 % | 95.5 % | |

**There is no generalisation drop.** Accuracy is within a point and QWK is *higher* on the
unseen corpus. For a thesis whose previous version had no external validation at all, this is
the single most defensible result in it.

**But the confusion matrix says something the summary hides.**

| true | n | correct | dominant error |
|---|---|---|---|
| No DR | 1 805 | 97 % | — |
| Mild | 370 | **6 %** | 314 → Moderate |
| Moderate | 999 | 64 % | 337 → Severe |
| Severe | 193 | 64 % | — |
| Proliferative | 295 | 52 % | 80 → Severe, 59 → Moderate |

The model **systematically over-grades APTOS by about one step** through the middle of the
scale. Note this is the *opposite* direction from the internal error: on IDRiD and Messidor-2,
Mild was mostly called No DR; on APTOS it is mostly called Moderate.

**The interpretation, and it is the useful one.** QWK measures ranking quality and it went
*up*; accuracy measures agreement with the cut-points and it stayed flat only because the
over-grading and the corpus's easier majority class cancelled. What transferred is the
model's ordering of severity. What did not transfer is where the boundaries between grades
sit — those were learned on IDRiD and Messidor-2 and are specific to how those corpora were
graded.

**Consequences to state in the thesis:**

1. **The 99.5 % referable sensitivity is real but must be quoted with its 84.3 %
   specificity.** Over-grading pushes borderline cases across the referral line, which is why
   almost nothing referable is missed — and why 342 of 2 175 non-referable patients would be
   referred unnecessarily. For a screening programme that is a defensible trade; presenting
   the sensitivity alone would not be.
2. **Deployment to a new population would need threshold recalibration on a small local
   labelled sample.** The ranking is transferable; the cut-points are not. This is a concrete,
   clinically meaningful recommendation rather than a hedge.
3. **Recalibrating on APTOS itself is not allowed** and has not been done. Fitting cut-points
   on the external corpus would make it no longer external. The number above is the
   uncalibrated one, which is the honest one.

**Caveats that stay attached to this result.** APTOS labels are single-grader and noisier than
IDRiD's or Messidor-2's adjudicated ones. APTOS carries DR grades only, so **there is still no
external number for 3-class DME** — that needs Messidor-1, which is not on Kaggle, and its
absence is a stated limitation of this thesis rather than an oversight.

### E08 — best run so far

DR QWK **0.860** [0.845, 0.874] and accuracy **74.2 %** against a 52.4 % floor, over all
2 260 development images out of fold. Per-class recall 85 / 45 / 73 / 67 / 45 — every grade
above the E05 baseline, and macro-F1 up from 0.547 to 0.646. Referable-DR sensitivity 87.2 %
at 95.5 % specificity, straight from the decode with no threshold tuning.

**Attribution, now tested.** Paired bootstrap over the same 2 260 groups:

| comparison | what changed | DR accuracy | DR QWK |
|---|---|---|---|
| E06 − E05 | EyePACS pretraining | +1.4 [−0.7, +3.5] n.s. | **+0.064 [+0.042, +0.087]** |
| E08 − E06 | 25 → 40 epochs | **+2.3 [+0.6, +4.2]** | +0.012 [−0.002, +0.028] n.s. |
| E08 − E05 | both together | **+3.8 [+1.8, +5.9]** | **+0.077 [+0.056, +0.099]** |

Two interventions, each significant on a *different* metric and neither on both. Pretraining
buys ordinal quality — it fixes the two-grade errors QWK charges for — and buys no measurable
accuracy. The longer schedule buys accuracy and no measurable QWK. Reporting either one as
"an improvement" without naming the metric would be a half-truth in both directions.

So **E08's headline QWK of 0.860 is not claimed to beat E06's 0.847**; that interval contains
zero. What E08 adds over E06 is accuracy and the persisted weights.

**The external number is still missing**, for infrastructure reasons rather than scientific
ones — see `ISSUES.md` §15. The weights exist; a follow-up kernel scores them on APTOS
without retraining.

### Threshold tuning: a fix that mattered on the weak model and vanished on the strong one

Cross-fitted cut-point tuning was applied to both runs, identically:

| | E05 (no pretraining) | E06 (EyePACS pretrained) |
|---|---|---|
| DR QWK, default cuts | 0.783 | 0.847 |
| DR QWK, tuned cuts | **0.831** | 0.855 |
| change | **+0.047 [+0.034, +0.061] significant** | +0.008 [−0.001, +0.017] **indistinguishable** |
| referable-DR at default decode | 73.7 % sens / 98.0 % spec | **88.3 % sens / 94.9 % spec** |
| referable-DR tuned for ≥90 % sens | 90.0 % / 85.8 % | 89.7 % / 93.4 % |

Pretraining fixed the same miscalibration that threshold tuning was compensating for. The
two interventions overlap almost completely, and on the better model the patch is worth
nothing.

**This is worth stating in the thesis as a methodological point.** Had only E05 been run,
threshold tuning would have been written up as a key contribution worth +0.047 QWK. It is
not a contribution — it is a repair for an under-trained, poorly calibrated output layer, and
it disappears the moment the output layer is trained properly. An improvement measured
against a weak baseline is a statement about the baseline as much as about the method.

The same caution applies in reverse: E06 needs no threshold tuning to reach a usable
screening operating point (88.3 % sensitivity at 94.9 % specificity straight out of the
decode), so the tuning machinery stays in the repository as a diagnostic rather than as part
of the reported pipeline.

### E06 — verdict: EyePACS pretraining works, and only on the head it can reach

**Confirmed for DR, not measurable for DME.** Paired bootstrap over the same 2 260 groups,
E06 minus E05:

| quantity | difference | 95 % interval | verdict |
|---|---|---|---|
| DR QWK | **+0.064** | [+0.042, +0.087] | **significant** |
| DR accuracy | +1.4 pts | [−0.7, +3.5] | indistinguishable |
| DME QWK | +0.005 | [−0.022, +0.030] | indistinguishable |
| DME accuracy | +1.9 pts | [−0.8, +4.7] | indistinguishable |

The DME result is what it should be: EyePACS carries no DME labels at all, so it could only
have helped that head through better shared features, and it did not do so measurably.
Reporting it as a win would have been reading the pooled improvement as if both heads earned
it.

**The per-class table is the interesting part, and it is why QWK leads.**

| class | n | E05 | E06 | change |
|---|---|---|---|---|
| No DR | 1 185 | 92.9 % | 80.1 % | **−12.8** |
| Mild | 295 | 25.8 % | 51.5 % | **+25.8** |
| Moderate | 515 | 60.2 % | 70.5 % | +10.3 |
| Severe | 168 | 44.6 % | 69.0 % | **+24.4** |
| Proliferative | 97 | 30.9 % | 46.4 % | +15.5 |

The model gave up some majority-class recall and bought a large improvement on every one of
the four minority grades. **Accuracy cannot see this trade** — it moved +1.4 points with an
interval spanning zero — while QWK charges for exactly the two-grade errors that got fixed
and moved by a clearly significant margin. If accuracy had been the primary metric, as it is
in the existing thesis, this experiment would have been recorded as a null result and the
single largest lever found so far would have been discarded.

Referable-DR sensitivity at the default decode also rose from 73.7 % to **88.3 %**, before
any threshold tuning.

### E05 — verdict

**Hypothesis confirmed, and the pipeline is real.** Every fold trained: loss 0.50 → 0.13, DR
QWK from ~0.00 at epoch 0 to 0.77–0.80 by the end, no divergence anywhere. Compare with every
run archived before this project restarted, all of which collapsed below the majority floor
(`ISSUES.md` §1).

Pooled out-of-fold over all 2 260 development images, re-scored under the corrected DME
definition (`ISSUES.md` §12) from the archived logits, no GPU re-spent:

| head | n | floor | accuracy | QWK | beats floor |
|---|---|---|---|---|---|
| DR, 5-class | 2 260 | 52.4 % | **70.4 %** [68.6, 72.3] | **0.783** | yes |
| DME, 3-class ungated *(primary)* | 516 | 47.1 % | **84.1 %** [81.0, 87.0] | **0.874** | yes |
| DME, 3-class gated DR≥1 *(secondary)* | 348 | 69.8 % | 77.3 % | 0.737 | yes |
| Referable DME, binary | 2 260 | 82.6 % | 94.6 % | 0.794 | yes |

Per corpus, DR: IDRiD 65.7 % (floor 32.6 %, QWK 0.799), Messidor-2 71.8 % (floor 58.3 %,
QWK 0.728). The corpora are not interchangeable — Messidor-2 scores higher on accuracy while
scoring *lower* on QWK, purely because 58 % of it is grade 0. This is exactly why accuracy
alone is not allowed to lead (`PROTOCOL.md` §4).

**On IDRiD's official 103-image test split:** **DR 61.2 %** and **DME 79.6 %** ungated.
Reported for completeness only — `PROTOCOL.md` §7 puts a ±6.3 pt interval on that split, so
it is not headline material. (This paragraph previously compared these against 91.6 / 87.6 as
though those were claims; they were template placeholders, corrected 2026-08-31 — see
`ISSUES.md` §1.)


### E05 — what the confusion matrix says about *where* to spend the next GPU-hour

Two things are visible in `docs/generated/confusion_dr.png` that the summary metrics hide.

**The ordinal structure works.** Errors sit almost entirely on the adjacent-grade band. The
model never once predicted Severe or Proliferative for a No-DR or Mild image — the entire
top-right of the matrix is zero. That is the threshold decomposition doing its job, and it is
the failure mode the previous version of this thesis identified as its main weakness.

**One cell holds most of the loss: Mild → No DR, 203 of 295 images (69 %).** Mild NPDR is
defined by the presence of a few microaneurysms, which are roughly 30–100 µm across. At
448 px over a 50° field they span barely one or two pixels. The error is therefore most
likely a **resolution** limit rather than a loss, backbone or sampling problem — the
information may simply not survive the downsampling.

That reading is testable and cheap, and it changes the queue: resolution moves ahead of
focal loss and aggressive resampling, both of which reweight a signal that may not be there.

**But the test is confounded, and the confound is decisive.** Messidor-2 in our mirror is
already downsampled to 512 px (`ISSUES.md` §4), and it holds 270 of the 295 Mild images.
Training at 640 px would upsample Messidor-2 and add nothing to precisely the corpus that
carries the problem. So:

* Resolution must be tested on IDRiD and reported per corpus, not pooled; **and**
* the higher-value move is to **acquire a full-resolution Messidor-2 mirror**. Without it,
  the Mild class is capped by the data rather than by the model, and no amount of training
  will fix it.

### E05 — three things to fix, in order of value

1. **The screening operating point is wrong.** Referable-DR sensitivity is 73.7 % at the
   default decision threshold, with 98.0 % specificity. Sweeping the expected-grade
   threshold trades that far better for a referral system:

   | threshold | sensitivity | specificity |
   |---|---|---|
   | 0.8 | **87.4 %** | 90.7 % |
   | 1.0 | 82.6 % | 95.0 % |
   | 1.4 *(default)* | 73.2 % | 98.0 % |

   Missing a referable patient is the expensive error in screening; an extra referral is
   cheap. The threshold must be chosen on validation, never on the reported set — the plan
   is cross-fitted selection, choosing fold *f*'s threshold from the other folds' out-of-fold
   predictions.

2. **The model is undertrained.** Best epoch was 29 of 30 in four folds out of five, and 24
   in the fifth. The schedule ended while the model was still improving.

3. **The rare grades are the weakness.** DR per-class recall: 93 % / 26 % / 60 % / 45 % /
   31 %. Mild and proliferative are where macro-F1 (0.547) is lost, and mild↔moderate is the
   error mode the thesis set out to fix in the first place.

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
different quantity, and it is why `PROTOCOL.md` §5.1 makes ungated primary. **This is also
why a gated DME number must never be quoted without its floor:** on the gated framing nothing
here clears 69.6 %, while the same models beat the ungated floor by 25 points.

---

### E15LPFT — LP-FT (linear probe, then fine-tune) — 2026-08-31, 3.4 h, **FALSIFIED**

**Pre-registered before the run** (`IDEAS.md` I21, commit `68563ef`): *LP-FT beats current
full fine-tuning on internal QWK at matched calibration. Falsified if it does not.*

**Config.** E08 with **one change**: `--lpft --lpft-epochs 5 --lpft-lr-backbone 1e-5
--lpft-warmup-epochs 2`. Everything else identical — densenet121, 448 px, batch 16, 40
epochs, EyePACS pretraining, TTA, folds 0–4, split `0cfbbfeb081999af`. Backbone parameters
are detached during the probe phase rather than merely given `lr = 0`, so the probe does not
backprop through what it is not training.

**Result — paired bootstrap over groups, both runs recalibrated identically with cut-points
cross-fitted on the other folds (`src/compare_matched.py`,
`docs/generated/matched_comparison.md`):**

| head | cut-points | metric | E08 | E15LPFT | E08 − LPFT | 95 % interval | verdict |
|---|---|---|---|---|---|---|---|
| DR | shipped | QWK | 0.8599 | 0.8466 | +0.0136 | [−0.0004, +0.0275] | indistinguishable |
| DR | shipped | accuracy | 74.20 % | 68.67 % | +5.57 pts | [+3.54, +7.52] | **significant** |
| **DR** | **matched** | **QWK** | **0.8646** | **0.8655** | **−0.0008** | **[−0.0126, +0.0105]** | **indistinguishable** |
| DR | matched | accuracy | 72.12 % | 74.07 % | −1.92 pts | [−3.67, −0.13] | **significant, for LP-FT** |
| DME (n=516) | shipped | QWK | 0.8845 | 0.8551 | +0.0303 | [−0.0017, +0.0618] | indistinguishable |
| DME (n=516) | matched | QWK | 0.8764 | 0.8577 | +0.0193 | [−0.0099, +0.0481] | indistinguishable |

**Verdict.** A dead heat on the primary metric — DR QWK differs by 0.0008 with an interval
tight around zero — and DME indistinguishable at both operating points. By the criterion
fixed in advance, **LP-FT does not beat current full fine-tuning and is closed.**

**What the run bought anyway, and why it was not wasted.** It is the fourth and starkest case
in the `PROTOCOL.md` §4.1 record, and its second outright sign reversal. At shipped
cut-points E08 looks **significantly better on DR accuracy by 5.57 points**; at matched
cut-points that reverses to **1.92 points in LP-FT's favour, also significant**. LP-FT's
representation was never worse — its `sigmoid > 0.5` cut-points were badly placed, costing
5.4 points of accuracy that recalibration returned in full (68.67 % → 74.07 %, the largest
recalibration gain any run in this project has produced, against E08's 74.20 % → 72.12 %
*loss*). Reported at shipped thresholds, this experiment would have concluded confidently
and wrongly that LP-FT damages a network.

**Provenance note on the analysis tooling.** `src/compare_matched.py` was written for this
comparison and verified against the archived numbers before being trusted: its `shipped`
column reproduces `results.json` exactly on both heads (DR QWK 0.8599, DME QWK 0.8845). Two
bugs were found and fixed by that check — a reimplemented decode that did not match
`model.decode`, and a DME evaluation set of 667 rows rather than 516, caused by including
Messidor-2 rows that carry only a binary DME label.

---

### E14MAC — macula-centred crop for the DME head (I07) — 2026-08-31, 4.4 h, **FALSIFIED**

**Pre-registered before the run** (`IDEAS.md` I07, commit `68563ef`): *global average pooling
dilutes the decisive region ~16× — 1 DD is ~55 px in a 448 px image — so pooling the DME
head's features at the fovea should move DME QWK.* **Falsifying outcome, also fixed in
advance: if DME QWK moves less than its ±0.03 interval, the DME ceiling is data, not
architecture.**

**Config.** E08 with **one change**: `--macula --macula-size 224 --macula-dd 3.0`. Shared
backbone, two forward passes — the DME head reads a 3 DD macula crop, the DR head keeps the
whole fundus. Backbone weights are shared, so any DME change is attributable to what the head
looks at rather than to added capacity. Fovea coordinates from the E13gate localiser, trained
in-process and archived as `runs/E14MAC/fovea_coords.json`.

**Result — paired bootstrap over groups, both runs recalibrated identically
(`docs/generated/matched_e08_e14mac.md`):**

| head | cut-points | metric | E08 | E14MAC | E08 − MAC | 95 % interval | verdict |
|---|---|---|---|---|---|---|---|
| DR | shipped | QWK | 0.8599 | 0.8507 | +0.0090 | [−0.0050, +0.0227] | indistinguishable |
| DR | matched | QWK | 0.8646 | 0.8595 | +0.0049 | [−0.0078, +0.0171] | indistinguishable |
| DR | matched | accuracy | 72.12 % | 73.27 % | −1.16 pts | [−2.92, +0.62] | indistinguishable |
| **DME (n=516)** | shipped | QWK | 0.8845 | 0.8693 | +0.0163 | [−0.0132, +0.0456] | indistinguishable |
| **DME (n=516)** | **matched** | **QWK** | **0.8764** | **0.8538** | **+0.0237** | **[−0.0094, +0.0566]** | **indistinguishable** |
| DME (n=516) | matched | accuracy | 83.33 % | 81.20 % | +2.26 pts | [−1.36, +5.81] | indistinguishable |

**Verdict: falsified, and cleanly.** DME QWK did not move — the matched difference is
**−0.0237 in the crop's favour-minus direction** (i.e. the crop is nominally *worse*), the
interval contains zero, and the magnitude is inside the ±0.03 band named in advance. By the
criterion fixed before the run, **the DME ceiling is data, not architecture.**

**The null is not an artefact of the localiser.** The window is 3 DD wide, half-width 1.5 DD.
E13gate's out-of-fold error is median 0.196 DD, 90th 0.433 DD, **99th 0.857 DD** — so in over
99 % of images the true fovea lies inside the crop, with room to spare. The DME head was
genuinely shown the macula; it did not help.

**Condition 2 discharged.** The binding requirement was to report the IDRiD-only result
alongside the pooled one. It is the same number: the 3-class DME evaluation set **is** IDRiD's
516 images, and Messidor-2 enters only as binary (referable / not) training supervision plus
the n=2 260 referable-binary metric. So the headline DME result carries **no** unvalidated
fovea transfer at evaluation time. The transfer assumption touches training supervision only —
and since the result is null, it cannot be a gain resting on an unchecked assumption.

**Why this run was worth 4.4 h despite the null.** It is the strongest architectural test
available for this head: an intervention derived directly from the label's own clinical
definition (exudate distance to the macula centre), using a localiser validated against a
threshold fixed before the numbers. It failing moves "DME is stuck" from an open architecture
question to evidence about where the ceiling actually is. See `FINDINGS.md` F7.

---

### E17NAT — native-resolution test (I20) — 2026-08-31, 7.2 h, **FALSIFIED**

**Pre-registered before the run** (`IDEAS.md` I20, commit `2b643eb`): *ISSUES §18 showed E10
was a 560 px run wearing a 640 px label — cached at 560, then upsampled to 640, discarding
the native mirror. E09 established resolution binds for DR. If it still binds above 448,
genuinely more pixels should move DR QWK above E10's.* **Falsifying outcome, fixed in
advance: if DR QWK moves less than its interval (≈ ±0.015 on 2 260 images), effective
resolution above ~560 does not bind for DR.**

**Config.** E10's, with `--cache-size 768` (so 640 px is real, not upsampled from 560) and
`--messidor-hi` (the 2240×1488 mirror, 1 057 of 1 744 Messidor-2 images). Batch held at E10's
8 deliberately, so batch is not a second variable. `results.json` confirms `size: 640,
cache_size: 768, messidor_source: prefer-native`.

**Result — paired bootstrap over groups, both recalibrated identically
(`docs/generated/matched_e10_e17nat.md`):**

| head | cut-points | metric | E10 | E17NAT | E10 − NAT | 95 % interval | verdict |
|---|---|---|---|---|---|---|---|
| **DR** | shipped | **QWK** | 0.8683 | 0.8691 | −0.0010 | [−0.0140, +0.0121] | indistinguishable |
| **DR** | **matched** | **QWK** | **0.8749** | **0.8662** | **+0.0086** | **[−0.0025, +0.0199]** | **indistinguishable** |
| DR | matched | accuracy | 74.78 % | 72.35 % | +2.45 pts | [+0.58, +4.29] | **significant, for E10** |
| DME | matched | QWK | 0.8948 | 0.8748 | +0.0195 | [−0.0052, +0.0452] | indistinguishable |

**Verdict: falsified.** DR QWK moved by +0.0086 at matched calibration — inside the ±0.015
band named in advance, with an interval containing zero, and **nominally in E10's favour**.
Genuinely more pixels did not help. **Effective resolution above ~560 does not bind for DR**,
and I10 / I10b / I10c are closed as a line of work.

**What this does and does not say about ISSUES §18.** §18 was a real bug — E10's resolution
label *was* inflated, and the native mirror *was* being discarded at 560. That diagnosis
stands. What E17NAT adds is that **the bug cost nothing measurable**: fixing it and paying
7.2 h for genuinely-640 px training bought no QWK. A correct diagnosis and a null consequence
are not in tension, and both belong in the record.

**Per-class — under the I19 source caveat, which is binding here.**

| | No DR | Mild | Moderate | Severe | Proliferative |
|---|---|---|---|---|---|
| E10 | 85.3 | 51.9 | 67.8 | 75.6 | 42.3 |
| E17NAT | 88.3 | 45.8 | 67.4 | 73.8 | 40.2 |
| Δ | **+3.0** | **−6.1** | −0.4 | −1.8 | −2.1 |

**CORRECTION, 2026-09-01 — the source caveat above was wrong, and over-cautious.** This
paragraph originally said the per-class numbers were not cleanly attributable because E17NAT
changed the image source as well as the resolution. **It did not.** The consumption manifests
(`src/manifest.py`, added after `ISSUES.md` §26) show E10 and E17NAT read **the same files**:
both mounted `borhan2003/messidor-...-jpg-format`, both logged the same 1 200-file native
mirror, and both resolved 1 744 Messidor-2 images with identical per-grade counts
(DR0=1017, DR1=270, DR2=347, DR3=75, DR4=35). E10's log says *"native-resolution mirror found,
1200 files"*; E17NAT's says the same under the newer wording. **The only difference between
them is `cache_size` — 560 versus 768 — which is exactly the effective-resolution change the
experiment intended.**

So I20 is a **cleaner** experiment than first reported: one variable, not two. The aggregate
conclusion is unchanged and now rests on a stronger footing.

**The per-class numbers are attributable to resolution**, but `PROTOCOL.md` §4.2 still binds
for a different reason: no multiple-comparison correction is applied, and credibility comes
from replication across conditions rather than a p-value in isolation. This is one condition.
**Treat the −6.1 Mild change as a lead, not a finding** — on those grounds alone.

---

## Reference floors — every row above must be read against these

| Quantity | Set | n | Value |
|---|---|---|---|
| DR majority class | IDRiD official test | 103 | 33.0 % |
| DME majority class, gated DR≥1 | IDRiD test | 69 | 69.6 % |
| DME majority class, gated DR≥1 | all IDRiD | 348 | 69.8 % |
| DME majority class, ungated | all IDRiD | 516 | 47.1 % |

## Prior work — NOT rows in this ledger

The figures in thesis chapter 4 (DR 91.6 %, DME 87.6 %) and in `CLAUDE.md` (91.7 / 87.4) are
**template placeholders** — never presented, never claimed, never produced by a run (owner,
2026-08-31). They are **not a baseline**, not a target, and not a result, and nothing in this
ledger should be compared against them.

**This project has no internal baseline at all.** The only evaluation ever archived before
this ledger began scored **27.2 % DR** and **15.9 % DME**, both below the floors above, from
a run that collapsed (`ISSUES.md` §1). So every row here is a first measurement rather than an
improvement over anything. **The consequence is that re-implementing the two literature
baselines on our exact split (`IDEAS.md` I16) is the only route to knowing whether these
numbers are good**, and it is prioritised accordingly.
