# T1 — candidate referral-sensitivity targets, with provenance

**Prepared 2026-08-31, for the owner to choose from.** The target is fixed and justified
**before** the operating curve decides it. Order of work, stated so it can be audited: the
published standards below were found and verified **first**; only then was the cost curve
computed. Every candidate comes from an external source, not from what flatters our model.

Article metadata retrieved from **PubMed**.

---

## 0. The verified negative — "the NHS standard" is not in the NHS standards

The **NHS Diabetic Eye Screening Programme pathway standards in force from 1 October 2024**
contain **no sensitivity or specificity standard at all.** All thirteen (DES-PS01 … DES-PS13)
are coverage, uptake, timeliness and image-quality measures: screening coverage, round length,
uptake, persistent non-attenders, ungradable-image rate (< 5 %), result-letter and referral
timeliness. Diagnostic accuracy is not among them.

Source: [NHS Diabetic eye screening pathway standards from 1 October 2024, GOV.UK](https://www.gov.uk/government/publications/diabetic-eye-screening-programme-standards/nhs-diabetic-eye-screening-pathway-standards-from-1st-october-2024-public-facing-guidance-information)

**Consequence.** Any sentence of the form "the NHS requires 80 % sensitivity" cannot cite the
current programme standards. The 80 / 95 figure is real, but it comes from elsewhere — see
candidate A, and read its provenance warning before using it.

A separate 80 % figure exists in DESP and is **not** this one: trainee graders must reach
≥ 80 % sensitivity **and** specificity across three monthly live test sets. That is a
**grader certification** standard about people, not a programme operating point for a device.
Do not conflate them.

---

## Candidate A — British Diabetic Association: ≥ 80 % sensitivity, ≥ 95 % specificity

| | |
|---|---|
| **Target** | minimum sensitivity **80 %**, minimum specificity **95 %** |
| **Disease target** | **referable DR** — the same quantity we report |
| **Status** | **minimum standard** (a consensus requirement, not a measured result) |
| **Attributed to** | British Diabetic Association. *Retinal Photography Screening for Diabetic Eye Disease. A British Diabetic Association Report.* London: British Diabetic Association; **1997** |
| **Where we saw it** | quoted in a peer-reviewed review: [The Evolution of Diabetic Retinopathy Screening Programmes, PMC7381763](https://pmc.ncbi.nlm.nih.gov/articles/PMC7381763/) — *"a minimum sensitivity of 80% and 95% of specificity for referable DR"* |

> ⚠️ **PROVENANCE WARNING — READ BEFORE CITING.** We have **not** seen the primary document.
> It is a printed 1997 BDA report with no online copy we could retrieve. What we verified is
> that a peer-reviewed article quotes it and gives the full reference. Widely-repeated
> secondary summaries of this figure are also **demonstrably unreliable**: one search summary
> we retrieved rendered it as *"a minimum specificity of 80% and a specificity of 95%"* —
> the sensitivity figure silently relabelled as specificity. This is exactly the failure mode
> this project keeps catching. If A is chosen, either obtain the 1997 report, or cite it
> **explicitly as reported in** the secondary source, never as if read directly.

**Advantage.** It is the only candidate whose disease target is *referable DR*, matching ours
exactly. **Disadvantage.** Nearly thirty years old, and unverifiable at source.

---

## Candidate B — IDx-DR pivotal trial, FDA pre-specified endpoints: > 85 % / > 82.5 %

| | |
|---|---|
| **Target** | pre-specified superiority endpoints: sensitivity **> 85 %**, specificity **> 82.5 %** |
| **Achieved** | sensitivity **87.2 %** (95 % CI 81.8–91.2), specificity **90.7 %** (95 % CI 88.3–92.7), imageability 96.1 % |
| **Disease target** | **mtmDR** — more-than-mild DR, ETDRS level ≥ 35 **and/or DME**, in at least one eye |
| **Status** | the thresholds are a **regulatory minimum agreed in advance**; the 87.2/90.7 are **achieved performance** |
| **Source** | Abràmoff MD, Lavin PT, Birch M, Shah N, Folk JC. *Pivotal trial of an autonomous AI-based diagnostic system for detection of diabetic retinopathy in primary care offices.* npj Digital Medicine 2018;1:39. PMID 31304320. [DOI](https://doi.org/10.1038/s41746-018-0040-6) |
| **Trial** | n = 900 enrolled, 23.8 % mtmDR prevalence, NCT02963441; reference standard = Wisconsin FPRC widefield stereo photography + macular OCT |

**Why this candidate is the strongest.** The > 85 % / > 82.5 % pair is a threshold **fixed
before the trial ran and agreed with a regulator** — structurally the same act T1 is
performing. It is the closest thing available to a defensible prior commitment, and it
supported the first FDA authorisation of an autonomous AI diagnostic in any field.

---

## Candidate C — EyeArt pivotal trial: 95.5 % sensitivity at 85.0 % specificity

| | |
|---|---|
| **Reported** | mtmDR: sensitivity **95.5 %** (95 % CI 92.4–98.5), specificity **85.0 %** (95 % CI 82.6–87.4), undilated |
| | vtDR: sensitivity **95.1 %** (95 % CI 90.1–100), specificity **89.0 %** (95 % CI 87.0–91.1) |
| | after enrichment correction: mtmDR specificity **87.8 %**; vtDR sensitivity 97.0 %, specificity 90.1 % |
| **Disease target** | mtmDR and vtDR |
| **Status** | **reported performance of a cleared device — NOT a standard.** Nothing requires 95.5 % |
| **Source** | Ipp E, Liljenquist D, Bode B, et al. *Pivotal Evaluation of an Artificial Intelligence System for Autonomous Detection of Referrable and Vision-Threatening Diabetic Retinopathy.* JAMA Netw Open 2021;4(11):e2134254. PMID 34779843. [DOI](https://doi.org/10.1001/jamanetworkopen.2021.34254) |
| **Trial** | n = 893 patients / 1 786 eyes, 15 sites, NCT03112005 |

Choosing C means choosing to **match a cleared competitor's operating point**, which is a
different and more aggressive act than meeting a minimum standard. It should be argued as
such if chosen.

---

## What each target costs us

Computed on the **development pool** — E10, pooled 5-fold out-of-fold, n = 2 260, referable
prevalence 34.5 % — because `PROTOCOL.md` forbids fitting any threshold on APTOS. Referable
DR is grade ≥ 2, read off the ordinal cut P(y > 1).

| target sensitivity | achieved | **specificity** | threshold |
|---|---|---|---|
| 80.0 % (candidate A) | 80.00 % | **98.72 %** | 0.7478 |
| 85.0 % (candidate B, minimum) | 85.00 % | **97.23 %** | 0.5786 |
| 87.2 % (candidate B, achieved) | 87.31 % | **96.08 %** | 0.4619 |
| 90.0 % | 90.00 % | **94.05 %** | 0.2947 |
| 95.5 % (candidate C) | 95.51 % | **82.16 %** | 0.1025 |
| 99.0 % | 99.10 % | **33.11 %** | 0.0315 |

**Read this curve carefully — it is not linear.** Between 80 % and 90 % sensitivity the
specificity cost is gentle: ten points of sensitivity cost under five points of specificity.
Past 95 % it collapses. Pushing to 99 % sensitivity costs **65 points of specificity** and
would refer two thirds of healthy patients — operationally useless.

**Candidates A and B are already met, with margin.** At 80 % sensitivity we hold 98.72 %
specificity against A's 95 % floor; at 85 % we hold 97.23 % against B's 82.5 % floor. Choosing
A or B is therefore choosing a *defensible published commitment we can demonstrably satisfy*,
not a stretch. Candidate C is the only one that would cost us: matching 95.5 % sensitivity
drops us to 82.16 % specificity, **below** EyeArt's own 85.0 %.

---

## ⚠️ A finding that surfaced while computing this, and that T1 must confront

**The shipped `sigmoid > 0.5` threshold does not mean the same thing on the two corpora.**

| corpus | sensitivity | specificity |
|---|---|---|
| development pool (n = 2 260) | **86.28 %** | **96.42 %** |
| APTOS external (n = 3 662) | **99.53 %** | **84.28 %** |

Same model, same cut-point, an operating point at opposite ends of the curve. On the
development pool that default sits near candidate B; on APTOS it sits past candidate C.

This is the transfer gap F3 predicted, and it is far larger than expected. It means the
99.53 % headline in `STATE.md` is **not** a property of the model — it is what an arbitrary
default happens to do on one particular corpus. It also means **T1 cannot be a single
number**: a threshold fitted to hit 85 % sensitivity on the development pool will not deliver
85 % on APTOS, and the size and direction of that miss is the actual result of the experiment.

Consistent with `PROTOCOL.md` §4.1 and `FINDINGS.md` F1/F3: an untuned default is a
hyper-parameter that has been chosen, not one that has been avoided.

---

## Caveat that applies to A, B and C alike

**The disease targets are not identical.** Our *referable DR* is ICDR grade ≥ 2 on the DR head
alone. **mtmDR** (B and C) is ETDRS ≥ 35 **and/or DME** — it fires on macular oedema even where
the DR grade is lower. So B and C are measured on a slightly **broader** condition than ours,
and a sensitivity target borrowed from them is not exactly like-for-like. Only candidate A is
stated for *referable DR*, and candidate A is the one we cannot verify at source. State
whichever mismatch applies in the thesis rather than letting the comparison read as exact.

---

## Recommendation

**Candidate B**, on three grounds: its provenance is verifiable at primary source; its
thresholds were fixed in advance and agreed with a regulator, which is the same epistemic act
T1 performs; and we meet them with real margin (97.23 % specificity against an 82.5 % floor).
Report candidate A alongside as the referable-DR-specific corroboration, cited honestly as
second-hand.

**If the owner prefers not to borrow at all**, the fallback stated in T1's design still
stands: fix the target as an explicit design choice with its rationale, and say so, rather
than dressing a round number as a standard.
