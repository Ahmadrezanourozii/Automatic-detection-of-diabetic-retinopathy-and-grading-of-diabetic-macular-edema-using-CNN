# T1 — transfer gap: a threshold fitted on the development pool, measured on APTOS

Model: `E08` weights. Dev pool n=2260 (referable prevalence 34.5 %); APTOS n=3662 (prevalence 40.6 %). Thresholds are fitted on the development pool only — never on APTOS (`PROTOCOL.md` §3, §6.1).

| target | source | threshold | dev sens (cross-fitted) | dev sens (shipped fit) | dev spec | **APTOS sens** | 95 % CI | APTOS spec | **transfer gap** |
|---|---|---|---|---|---|---|---|---|---|
| **80.0 %** | A | 0.8298 | 80.24 % | 79.87 % | 97.97 % | **98.12 %** | [97.39, 98.77] | 86.30 % | **+18.12 pts** |
| **85.0 %** | B | 0.5927 | 84.97 % | 84.87 % | 96.35 % | **99.33 %** | [98.88, 99.73] | 84.74 % | **+14.33 pts** |
| **87.2 %** | B* | 0.5104 | 86.90 % | 87.18 % | 95.20 % | **99.53 %** | [99.17, 99.86] | 84.23 % | **+12.33 pts** |
| **95.5 %** | C | 0.1109 | 95.24 % | 95.38 % | 81.89 % | **99.93 %** | [99.79, 100.00] | 79.86 % | **+4.43 pts** |

## What the gap is, and what it is not

**The fitting procedure is honest in-domain.** Cross-fitted dev sensitivity lands on target
every time (80.24 % for an 80 % target, 84.97 % for 85 %, 86.90 % for 87.2 %, 95.24 % for
95.5 %). Nothing is wrong with the threshold fitting. The gap is entirely a transfer effect.

**Every target overshoots on APTOS, and the overshoot is enormous at the low end.** A
threshold chosen to refer 80 % of referable cases refers **98.12 %** of them on APTOS —
**+18.12 points** — while specificity falls from 97.97 % to 86.30 %. The gap shrinks as the
target rises (+18.1, +14.3, +12.3, +4.4) simply because sensitivity saturates: there is less
room to overshoot from 95.5 % than from 80 %.

**Mechanism, from the score distributions.** The model scores APTOS images systematically
higher on P(DR > 1), in *both* classes:

| | non-referable median | non-referable p90 | referable median | referable p10 |
|---|---|---|---|---|
| dev pool | 0.0426 | 0.2380 | 0.9686 | **0.3625** |
| APTOS | 0.0352 | **0.9475** | 0.9732 | **0.9464** |

The referable 10th percentile moves from 0.36 to 0.95 — on APTOS even the *least confident*
referable cases score above 0.94, so any threshold below that catches essentially all of them.
But the non-referable 90th percentile moves from 0.24 to 0.95 as well, which is where the
specificity goes. The model is not separating better on APTOS; it is pushing **everything**
upward. That is the same mechanism as `FINDINGS.md` F1 — APTOS Mild cases being pushed into
Moderate — observed here from the operating-point side rather than the per-class side, which
is the kind of replication `PROTOCOL.md` §4.2 asks for.

**⚠️ The confound, stated because it is not fully separable from the archives.** The dev
predictions are **single-model** (each image scored by the one fold that held it out) with
**no TTA**. The APTOS predictions are a **5-fold logit ensemble with TTA**. Both ensembling and
TTA change calibration, so this gap mixes distribution shift with the inference recipe.

Two things bound the concern. Logit averaging shrinks scores *toward the mean*, and TTA
smooths — neither predicts the observed **upward polarisation of both classes**, so the
dominant term is unlikely to be the recipe. And the effect replicates F1, which was measured
on per-class recall under a different analysis. But it is not proven, and the claim here is
therefore "the transfer gap is large, positive, and consistent with F1's mechanism", not "the
transfer gap is entirely distribution shift".

**To close it** costs one cheap external-only run: score APTOS with a *single* fold's weights
and no TTA, and recompute this table. That isolates the recipe from the shift. Queued.

## Consequence for the recommendation

A threshold fitted on this development pool **does not transfer within a useful band** — an
18-point sensitivity overshoot and an 11-point specificity loss is not a usable guarantee.
By T1's pre-registered falsifying outcome, that means the recommendation becomes
**"recalibrate locally"**, with `FINDINGS.md` F4's figure attached: about **200 labelled local
images**, worth +5.09 points [+0.32, +7.28] of macro-recall with a 1.5 % risk of harm, and
below 100 images roughly one attempt in four makes the model worse.
