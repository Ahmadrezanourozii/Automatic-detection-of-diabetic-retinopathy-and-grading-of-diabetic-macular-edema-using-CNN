# MASTER EXECUTION PROMPT — IDRiD gate → thesis rewrite → claims testing

Read this whole file before doing anything. It has four parts that run **in order**.
Do not start Part B before Part A's verdict is written. Do not start Part D before
Part B is delivered.

**Priority order, non-negotiable:** the thesis comes first. Everything in Part D is
secondary and may be paused at any point if Part B needs more work.

---

## PART 0 — Orientation

### Read first
- `CLAIMS.md` — the literature ledger. 12 papers (P1–P12), claim IDs S/R/N/D/M/V/Z/T/Y/H/G,
  our own findings F1–F7, four open contradictions, contamination register.
- `dr-dme-campaign-brief.md` — the five-phase experiment plan (Phase 0–4).
- `PROTOCOL.md`, `STATE.md`, `FINDINGS.md`, `EXPERIMENTS.md`, `ISSUES.md`,
  `LITERATURE.md`, `IDEAS.md`, `data/DATASETS.md` — the knowledge base.
- The thesis manuscript itself and its figure-generation scripts.

### Standing rules that apply to every part of this file
1. **Consumption manifest on every run.** Each run writes a manifest of the files it
   actually opened, with per-fold hashes. Config records intent; the manifest records
   consumption. This exists because six provenance bugs produced no runtime errors and
   silently generated wrong numbers (F7).
2. **Matched calibration before attribution.** No performance delta is attributed to an
   architecture or training change until it has survived a matched-calibration check.
3. **Pre-register acceptance criteria.** Write the criterion and commit it *before*
   computing the number it judges. Never look at a result and then decide what counts
   as a pass.
4. **Negative results are results.** Report what did not work with the same rigour as
   what did.
5. **Patient/eye-level clustering.** Never image-level resampling on a dataset with two
   eyes per subject. Cluster bootstrap, 2,000 iterations, report the design effect.
6. **ECE never appears alone.** Always beside Brier and a discrimination metric.
7. When a number cannot be reproduced from its own source, say so in the report rather
   than quietly substituting a different one.

---

---

# ⚠️ FINDING-NUMBER CROSSWALK — READ BEFORE CITING ANY "F" NUMBER

**`FINDINGS.md` is the base.** This file's `F1–F7` were assigned independently and **do not
match** the repository's numbering. Nothing is renumbered — the repository is full of
cross-references and renumbering it would be an unnecessary risk. Instead the mapping is made
explicit here, and **all new writing uses the `FINDINGS.md` numbers.**

| this file | means | actual home in the repository |
|---|---|---|
| `CLAIMS F1` | cut-point shifting recovered external minority-class recall ~5 % → ~78 % | **`FINDINGS.md` F1** (same finding, same number — the only coincidence) |
| `CLAIMS F2` | matched calibration reversed four prior attributions, both directions | **`PROTOCOL.md` §4.1** (the record table) and **`FINDINGS.md` F3** (per-class comparisons dominated by cut-point placement) |
| `CLAIMS F3` | the DME head is supervision-limited, not architecture-limited | **`FINDINGS.md` F7** |
| `CLAIMS F4` | frozen RETFound beat ImageNet; fine-tuned RETFound lost to DenseNet+EyePACS | **`FINDINGS.md` F8** |
| `CLAIMS F5` | the ensemble's development gain did not survive external validation | **`IDEAS.md` I23** and `docs/generated/ensemble_external.md` — never given an F number |
| `CLAIMS F6` | Messidor-1 has ~88 % overlap with the development pool | **`FINDINGS.md` F2** |
| `CLAIMS F7` | provenance bugs produced no runtime errors but wrong numbers | **`ISSUES.md` §24, §26, §27** (and §16, §20, §23) — never given an F number |

**Findings with no counterpart in this file**, because they postdate it:

| | |
|---|---|
| `FINDINGS.md` F4 | recalibration needs ~200 labelled local images; below 100, one attempt in four makes the model worse |
| `FINDINGS.md` F5 | macro-recall must not be primary — tuning for it costs 19.2 points of referable sensitivity |
| `FINDINGS.md` F6 | a deployment recommendation evaluated only at its mean can be harmful |
| `FINDINGS.md` F9 | the pipeline is worth **+0.234 QWK** over frozen features and a linear model |
| `FINDINGS.md` F10 | the DME grade is not recoverable from published annotations (Part A verdict) |

**Rule from 2026-09-07 onward: cite `FINDINGS.md` numbers.** When quoting this file's text,
translate the F number through the table above.

---

# PART A — GATE: the IDRiD derivation test

CPU only. No GPU. Can run alongside any training job. **Nothing about additional DME
data proceeds without this result.**

## A.1 — First, record the dataset search findings

Write these into `data/DATASETS.md` as verified findings so nobody re-derives them:

- **DDR** was recommended as having hard-exudate masks *plus* fovea coordinates *plus*
  optic disc location. It does not. Its 757-image segmentation subset carries four lesion
  masks only (MA, haemorrhage, hard exudate, soft exudate) — no fovea, no optic disc.
  Verified against the official release and three independent sources. Unusable for
  geometric derivation.
- **BRSET** was recommended on the claim that its `DR_SDRG` column isolates "hard exudates
  within 1–2 disc diameters of the fovea", i.e. our grade 1. It does not. Verified against
  the PhysioNet data dictionary: `DR_SDRG` is a *retinopathy* severity scale (0 none,
  1 mild background, 2 moderate background, 3 severe/pre-proliferative, 4 proliferative).
  The Scottish scheme's separate maculopathy grade is not in the release. The only oedema
  label is binary `macular_edema`. Same for **mBRSET**.
- **HEI-MED, e-ophtha EX, DIARETDB1, FGADR** — all lack native fovea and/or optic disc
  annotations. Using them would require a landmark detector, importing cascaded coordinate
  error into what is meant to be ground truth.
- **OLIVES** — near-infrared SLO, not colour fundus. Cannot enter the training pool.
- **SUSTech-SYSU** is the only viable candidate: 1,219 images, Figshare, open access,
  Chinese cohort, no overlap with Messidor/IDRiD/EyePACS/APTOS. Verified from the *Nature
  Scientific Data* paper — provides exudate annotations, the optic disc **bounding box**,
  and fovea location. Two risks, to be checked from the actual download only if we get
  there: (a) exudates are bounding boxes, not pixel masks, so distance-to-box-edge
  underestimates distance and biases true grade 1 into grade 2 — the exact class we are
  short of; (b) the paper refers only to "exudates" and trained its box refiner on IDRiD
  labels "combining soft exudates and hard exudates together", so it may not separate hard
  from soft. Since the DME grade is defined on hard exudates only, a merged label would
  falsely produce grade 2 whenever a cotton-wool spot sits near the macula.

## A.2 — The test

The DME grade is a deterministic geometric function:
- **grade 0** — no hard exudates;
- **grade 2** — minimum distance from any hard-exudate pixel to the fovea centre ≤ one
  optic disc diameter;
- **grade 1** — otherwise.

IDRiD's 81-image segmentation subset has everything needed to test whether that derivation
reproduces expert grading: hard-exudate pixel masks, fovea coordinates, optic disc
boundary, and — critically — the true 3-class grade for those same images from the
516-image set.

## A.3 — Pre-register the acceptance criterion BEFORE computing anything, and commit it

Proposed criterion (adjust only if you have a better one, and say why):
- exact-match rate overall;
- the full 3×3 confusion matrix;
- per-class recall, **with grade 1 reported separately**. Grade 1 is the entire reason this
  exercise exists, so an overall match rate that hides a grade-1 failure is not a pass.

## A.4 — Report honestly on the mechanics

- How many of the 81 segmentation images have a 3-class grade in the 516 set. If fewer
  than 81, say so; the test is only as strong as the overlap.
- **How you computed one disc diameter from the IDRiD optic disc annotation** — the major
  axis, the equivalent-area diameter, or something else. The optic disc is an ellipse, not
  a circle, so "diameter" is ambiguous, and the threshold is *defined* in disc diameters.
  The wrong choice shifts the threshold and could unfairly reject the derivation. This is
  part of the method and must be stated explicitly in the report, not buried in code.
  Run the derivation under **all three** definitions and report sensitivity of the verdict
  to that choice.
- Whether IDRiD's masks separate hard from soft exudates. They should — confirm rather
  than assume, since that separation is what makes IDRiD the ideal-conditions test.

## A.5 — The verdict, in these terms

- **Reproduces** under ideal conditions (pixel masks, expert landmarks, hard/soft
  separated) → the method is valid, and SUSTech-SYSU becomes worth the engineering with
  its two caveats stacked on top. **Stop and report before starting it.**
- **Does not reproduce** → the whole derivation route dies here at zero GPU cost, and that
  is a reportable negative worth a findings entry: the clinical grading definition is not
  recoverable from the annotations the field actually publishes, which would explain why
  nobody has released a second 3-class DME corpus.
- **Partly reproduces** (e.g. grades 0 and 2 well, grade 1 not) — the most interesting of
  the three outcomes, and it needs its own analysis, because it would say the boundary case
  is exactly where the definition and the annotations diverge.

Write it up as a `FINDINGS.md` entry either way. **Do not proceed to SUSTech-SYSU or any
other dataset on the strength of a partial result. Report first.**

---

# PART B — Thesis rewrite (the priority deliverable)

Goal: a manuscript the supervisor can read end to end, in Persian that reads like a
competent Iranian graduate student wrote it, incorporating everything learned since the
last version.

## B.1 — نثر فارسی: استاندارد نگارشی الزامی

این بخش را کلمه‌به‌کلمه رعایت کن. متن پایان‌نامه نباید بوی ترجمهٔ ماشینی یا متن
تولیدشده با هوش مصنوعی بدهد.

### مرجع رسمی
مبنای رسم‌الخط، **دستور خطّ فارسی مصوّب فرهنگستان زبان و ادب فارسی (ویراست ۱۴۰۱)** است.
اگر دانشگاه (FAU / دانشگاه مبدأ) شیوه‌نامهٔ نگارش پایان‌نامهٔ خودش را دارد، **آن مقدم
است**؛ اول شیوه‌نامه را پیدا کن و اگر پیدا نشد، همین‌جا را ملاک بگیر و در گزارش بنویس
که کدام را ملاک گرفته‌ای.

### قواعد رسم‌الخط
- نیم‌فاصله در: پیشوند فعلی «می‌» (می‌شود، می‌کند)، جمع «‌ها» (داده‌ها، تصویرها)،
  «‌تر/‌ترین» (بیش‌تر، دقیق‌ترین)، «‌ای» (نمونه‌ای)، و ترکیب‌های وصفی رایج (هم‌بستگی،
  پیش‌پردازش، ریزتنظیم).
- «ک» و «ی» فارسی، نه عربی. هیچ «ك» و «ي» در متن نباشد. یک بار با اسکریپت کل فایل را
  پاک‌سازی کن.
- کسرهٔ اضافه پس از «ه»: یک شکل را انتخاب کن و تا آخر متن ثابت نگه دار — یا «مقایسهٔ
  مدل‌ها» یا «مقایسه‌ی مدل‌ها». فرهنگستان شکل نخست را توصیه می‌کند. **مخلوط‌کردن این دو
  در یک متن، اولین چیزی است که ویراستار می‌بیند.**
- تنوین فقط در واژه‌های عربی‌ای که تنوین دارند: «تقریباً»، «مستقیماً». «گاهاً» و
  «تلفناً» غلط است.
- اعداد در متن فارسی: ارقام فارسی (۱۲۳). در فرمول‌ها، جدول‌ها و نام متغیرها ارقام لاتین.
  در کل متن یکدست باش.
- علائم سجاوندی: بدون فاصله قبل، با یک فاصله بعد. «مدل، داده و آستانه» نه «مدل ، داده».
- هر اصطلاح انگلیسی در **اولین** کاربرد: معادل فارسی + شکل لاتین در پانویس. بعد از آن
  فقط معادل فارسی.

### چیزهایی که متن را «هوش‌مصنوعی‌زده» می‌کند — ممنوع
- عبارت‌های پرکنندهٔ بی‌محتوا: «لازم به ذکر است»، «شایان ذکر است»، «در دنیای امروز»،
  «با توجه به اهمیت روزافزون»، «نقش بسزایی ایفا می‌کند»، «به عنوان یک ابزار قدرتمند».
- «می‌گردد» به‌جای «می‌شود». همه را به «می‌شود» برگردان.
- مجهول‌سازی بی‌دلیل: «مورد بررسی قرار گرفت» → «بررسی شد». «مورد استفاده قرار می‌گیرد»
  → «به کار می‌رود». «انجام پذیرفت» → «انجام شد».
- «می‌باشد» به‌جای «است». در نثر علمی معاصر «است» درست‌تر و کوتاه‌تر است.
- جمله‌های بلند با سه «که» پشت سر هم. هر جمله یک فکر. اگر جمله‌ای بیش از دو خط شد،
  دو جمله‌اش کن.
- ساختارهای سه‌تایی موازی که پشت هم تکرار می‌شوند («دقیق، سریع و قابل اعتماد»،
  «کارآمد، مقیاس‌پذیر و مؤثر»). یکی‌دو بار در کل فصل، نه در هر پاراگراف.
- شروع پاراگراف‌های پیاپی با قید یکسان («همچنین»، «علاوه بر این»، «در نهایت»).
- بولت‌پوینت در جایی که باید پاراگراف باشد. فصل‌های ۱ و ۲ و ۵ باید نثر پیوسته باشند،
  نه فهرست. فهرست فقط برای چیزی که واقعاً فهرست است.
- تعریف و تمجید از کار خودت. «نتایج بسیار چشمگیر» و «دستاورد قابل توجه» را بردار و
  عدد بگذار.

### چیزهایی که متن را درست می‌کند
- فعل معلوم و فاعل مشخص: «ما آستانه را روی مجموعهٔ اعتبارسنجی تنظیم کردیم» یا «آستانه
  روی مجموعهٔ اعتبارسنجی تنظیم شد» — نه «تنظیم آستانه صورت پذیرفت».
- ادعا با عدد و ارجاع. هر جملهٔ ادعایی یا عدد خودمان را دارد یا شمارهٔ منبع در قلاب
  [۱۲].
- تفاوت را صریح بگو: «برخلاف [۱۲]، ما آستانه را ثابت نگه نداشتیم». نثر علمی فارسی
  خوب، محافظه‌کار نیست؛ مبهم نیست.
- محدودیت‌ها را با همان صراحت بنویس که نتایج را.

### واژه‌نامهٔ الزامی — یک معادل، یک بار انتخاب، تا آخر ثابت
`GLOSSARY.md` بساز و این‌ها را در آن قفل کن. اگر جای دیگری معادل دیگری به کار رفته،
اصلاحش کن:

| English | فارسی |
|---|---|
| diabetic retinopathy | رتینوپاتی دیابتی |
| diabetic macular edema | ادم ماکولای دیابتی |
| fundus image | تصویر ته‌چشم |
| grading | درجه‌بندی |
| decision rule | قاعدهٔ تصمیم |
| threshold / cut-point | آستانه / نقطهٔ برش |
| calibration | واسنجی (کالیبراسیون) |
| matched calibration | واسنجی همتاشده |
| representation | بازنمایی |
| external validation | اعتبارسنجی بیرونی |
| sensitivity / specificity | حساسیت / ویژگی |
| referable DR | رتینوپاتی نیازمند ارجاع |
| foundation model | مدل بنیادین |
| fine-tuning | ریزتنظیم |
| frozen encoder | رمزگذار منجمد |
| supervision ceiling | سقف نظارت |
| provenance | تبار داده |
| ensemble | ترکیب مدل‌ها |
| data leakage / contamination | نشت داده / آلودگی داده |

**QWK، AUROC، AUPR، ECE، Brier، F1** را ترجمه نکن؛ به همان شکل لاتین بیاور و در
فهرست نشانه‌ها تعریف کن.

### بازبینی نهایی نثر
بعد از نوشتن هر فصل، یک بار کل فصل را با این چک‌لیست اسکن کن و شمارش بده:
تعداد «می‌گردد»، «می‌باشد»، «مورد ... قرار گرفت»، «لازم به ذکر است»، جمله‌های بلندتر
از دو خط، و پاراگراف‌هایی که با قید تکراری شروع می‌شوند. عدد را در گزارش بنویس و
تا صفر نشدن، فصل را تمام‌شده اعلام نکن.

## B.2 — What actually changes in the manuscript

The thesis contribution is **methodological, not performance-based**. Keep that frame and
strengthen it with the new evidence. Concretely:

**Chapter 1 (مقدمه) — rewrite the motivation.** It now has real citations behind it:
- the development–deployment gap: V4 (three FDA-approved ophthalmic AI devices as of
  April 2024, roughly one per year), V5 (four named barriers, the first of which is the
  absence of head-to-head comparisons), Y4 (no RCTs in the oculomics literature).
- G5: of 64 patients referred by a deployed AI screening system, 9 (14%) attended.
  A sensitivity gain that reaches 14% of the people it identifies is the honest frame for
  why decision rules matter more than another point of QWK.

**Chapter 2 (مرور ادبیات) — restructure into two levels.**
- Level 1: P8 (RETFound) as the claim under audit. State precisely what it establishes and
  what it does not: no CNN baseline anywhere in the paper, no frozen or linear-probed
  condition, no per-class DR metric, no calibration under shift. Cite Z1, Z2, Z3, Z4, Z5.
- Level 2: P1–P7 and P9–P12 as partial re-tests, each sampling a different cell.
- Add the **metric-incommensurability point** (Contradiction 4): across P1–P12 no paper
  reports ordinal agreement *and* threshold-free ranking *and* calibration *and* per-class
  recall on the same external evaluation. Our paper is the first that does.

**Chapter 3/4 (روش و نتایج) — the two headline results.**
- **Headline 1 — F1, the decision rule.** Now supported by six independent instances:
  R6, S5 (benchmark), H4 + H5 (meta-analysis, 25 studies / 613,690 images, individual
  sensitivities spanning 0.06–1.00), G1 (three vendors, identical eyes, specificity
  14.25%–96.01%), G2 (one vendor, sensitivity 68.4%→99.6% against specificity 96%→64.7%).
- **Headline 2 — metric invariance, and this is new.** G8: in P12, κ moved 0.65 → 0.72
  while sensitivity and specificity each moved 31 points. That is the clinical analogue
  of our QWK finding, in someone else's deployed system, computable from their own two
  tables. Pair it with our own frontier.
- **F3 (DME ceiling)** can now be asserted rather than hedged. G4: a deployed commercial
  system reaches DME sensitivity 26.5% and κ 0.38 while reaching κ 0.81 on referable DR
  **in the same run, on the same images**. Not our architecture, not our dataset size.
- **F4 (RETFound)** — reframe using the P8 audit. The apparent contradiction dissolves
  once the axes are named. Also record the contamination status: IDRiD, APTOS-2019 and
  MESSIDOR-2 are **clear** by P8's own Methods; EyePACS **is** in RETFound's CFP
  pretraining (88,702 images, 9.8%) and must be stated when framing a DenseNet+EyePACS
  comparison.
- **F7 (provenance)** — promote from a confession to a contribution. Four published
  exhibits with the recomputation shown: P5, P8 (*Nature*), P9, P11. The P11 case is the
  strongest: the FN column of its only data table is a copy of the FP column, all 37 rows
  are wrong, and the forest plots use different values.

**Chapter 5 (بحث و نتیجه‌گیری)** — limitations honestly, including Part A's verdict,
and the referral-adherence ceiling (G5) as the boundary of what any of this buys.

## B.3 — Figures and tables

Regenerate everything from scripts. No hand-edited images. Every figure script writes a
consumption manifest.

**Persian text in matplotlib** does not work out of the box. Use `arabic_reshaper` +
`python-bidi`, register a Persian font (Vazirmatn or XB Niloofar), and set
`matplotlib.rcParams['font.family']`. Verify visually that every caption and axis label
renders joined and right-to-left — do not assume it worked. Vector output (PDF/SVG) for
everything that goes in the manuscript.

Figures to produce or update:
1. **The Pareto frontier** (Phase 1) — QWK vs referable sensitivity, with three marked
   points: QWK-optimal threshold, clinical-sensitivity target, argmax default. Centrepiece.
2. **Metric invariance** — new. QWK, κ, accuracy, sensitivity, specificity all plotted
   against the cut-point on one axis. The flat lines are the point.
3. **Model × rule re-ranking** (Phase 0) — heatmap plus the Spearman rank correlations
   between rankings under R1/R2/R3.
4. **Predicted vs true class distribution** at each marked threshold — the distortion is
   the mechanism, not a side effect.
5. **DME label-scaling curve** (Phase 3a) — QWK vs n with CIs, extrapolated.
6. **IDRiD derivation confusion matrix** — 3×3, from Part A.
7. **Contamination audit table** — dataset × encoder, per the register in `CLAIMS.md`.
8. **CLAIM compliance matrix** across P1–P12, with ourselves scored honestly in it.
9. **Metric-coverage matrix** across P1–P12 (Contradiction 4).
10. **Deployment recalibration harm curve** — the ~100-local-label threshold.
11. **Table: F1–F7 with supporting and contesting claim IDs**, sourced from the
    "OUR FINDINGS → EVIDENCE MAP" section of `CLAIMS.md`.

Every figure needs a Persian caption written in the style of B.1, and every number in a
caption must match the number in the text.

---

# PART C — Knowledge-base sync

Before delivering the manuscript, bring every project file up to date. Nothing should
contradict anything else.

- `FINDINGS.md` — add the Part A verdict; add the external support now attached to each of
  F1–F7 with claim IDs.
- `LITERATURE.md` — reconcile with `CLAIMS.md` (P1–P12). `CLAIMS.md` is authoritative.
- `PROTOCOL.md` — add: (a) report all four metric families on every external evaluation;
  (b) report κ alongside sensitivity/specificity across any decision-rule change (the G8
  pairing); (c) never tabulate our cluster-bootstrap CIs beside RETFound-style
  seed-variance CIs without a footnote — they are different quantities; (d) adopt
  iCHECK-DH alongside CLAIM for any deployment-facing claim.
- `dr-dme-campaign-brief.md` — apply the seven changes listed under "WHAT CHANGES IN THE
  CAMPAIGN BRIEF" in `CLAIMS.md`. In particular: Phase 4 uses P8's published fine-tuning
  recipe verbatim (50 epochs, batch 16, lr 5×10⁻⁴ cosine to 1×10⁻⁶, 10 warmup epochs,
  label smoothing, AutoMorph preprocessing, 224×224); Phase 4 adds a CNN comparator in
  every cell; Phase 4 adds six-direction cross-dataset external evaluation; resolution
  matching is decided **downward** (MedSigLIP at 224, not RETFound at 448).
- `data/DATASETS.md` — the Part A.1 entries.
- `STATE.md` — current status, what is gated on what.
- `ISSUES.md` — anything Part A or the rewrite surfaced.

---

# PART D — Claims testing (only after Part B is delivered)

Goal: turn `CLAIMS.md` rows from assertions into verdicts, and use what survives to
strengthen the thesis and the paper.

## D.1 — The matched-conditions rule

**A claim may only be marked CONFIRMED or REFUTED if it was tested under the source
paper's own conditions**: same dataset pair, same direction, same target condition, same
metric, same adaptation regime. If any of those differ, the verdict is
`CONDITIONS-DIFFER` and the report states exactly which axis moved.

This is not pedantry — it is the whole thesis. We accuse the field of comparing across
unmatched cells. We do not get to do it.

## D.2 — Result records

One file per tested claim: `results/RESULT-<claimID>.md`, containing:
- the claim verbatim and its source paper;
- the source's conditions, and ours, side by side;
- what we ran, with the consumption manifest hash;
- the number, with cluster-bootstrap CI;
- verdict: `CONFIRMED` / `REFUTED` / `PARTIAL` / `NOT-REPRODUCIBLE` / `CONDITIONS-DIFFER`;
- one sentence on what it changes for us.

Then update the Status column of that row in `CLAIMS.md` and cross-reference the result
file. **`CLAIMS.md` must never disagree with `results/`.**

## D.3 — Order of work

**Tier 1 — zero GPU, run these first.** All are re-analysis of existing prediction files
or published tables:
- **Phase 0** from the campaign brief (matched-calibration re-ranking under rules R1–R4).
  This gates everything else. If between-rule spread is comparable to or larger than
  between-model spread, the central claim holds; if not, stop and report.
- **G8 replication on our own frontier** — compute QWK, κ, accuracy, sensitivity,
  specificity at every cut-point. Does κ stay as flat across our frontier as it did across
  P12's 31-point swing? This is the cheapest new experiment in the project and it is a
  headline figure.
- **R6 / S5** — refit the decision rule on Poyrazer's and ORDER-DR's released external
  prediction files under R3 and R4. Grade-1 F1 = 0.000 on 270 Messidor-2 images;
  ORDER-DR grade-2 recall 0.189. Either outcome is publishable.
- **Z3 verification** — obtain P8 Supplementary Table 3 and confirm the IDRiD AUPR
  *P* = 0.81 against AUROC *P* < 0.001, and check whether the internal-IDRiD and
  APTOS→IDRiD AUROC really do share a confidence interval to three decimals.
- **H4 / H5 audit** — recompute P11's Table 2 sensitivities from the forest-plot
  denominators and document the FP/FN column duplication. Pure desk work; it is an exhibit
  for F7, not an experiment.
- **CLAIM and metric-coverage matrices** across P1–P12.

**Tier 2 — modest compute.**
- Phase 2a (reproduce both released pipelines), 2c (decompose the recovered gap into
  threshold / calibration / residual representation), 2d (technique collapse).
- Phase 3a (DME label-scaling curve), and 3b only if Part A's verdict allows it.

**Tier 3 — expensive, last.**
- Phase 4, the 2×2 factorial {frozen, fine-tuned} × {internal, external}, with ImageNet
  and DenseNet+EyePACS baselines in every cell, at matched resolution, under matched
  decision rules, using P8's recipe. Report macro AUROC **and** per-class AUPR/recall in
  every cell so the Z3 dissociation is visible rather than averaged away.

## D.4 — Papers to obtain, in priority order

1. **Lam et al., *Diabetes Care* 2024;47(2):304–319** — AI for DME from fundus vs OCT,
   systematic review and meta-analysis. Bears directly on F3 and Contradiction 1b.
   **Get this before writing the DME chapter section.**
2. **P8 Supplementary Tables 1 and 3** — split protocol and patient-level status for
   IDRiD/APTOS/MESSIDOR-2, plus the duplicated-CI check. Open access.
3. **Lee et al., *Diabetes Care* 2021;44:1168–1175** — seven systems head to head. We know
   second-hand it reports NPV 82.72–93.69% against sensitivity 50.98–85.90%, which is the
   G1 signature.
4. Zhang et al., *NPJ Digit Med* 2024;7:108 — RETFound real-world screening with decision
   curve analysis.

---

# SEQUENCING AND REPORTING GATES

1. **Part A** → write the verdict and the `FINDINGS.md` entry → **stop and report.**
2. **Part B + Part C** → deliver the manuscript and the synced knowledge base →
   **stop and report.** This is what goes to the supervisor.
3. **Part D Tier 1**, starting with Phase 0 → **stop and report before committing further
   compute.**
4. Tiers 2 and 3 only on explicit go-ahead.

Report in Persian for anything touching the thesis, in English for anything touching
`CLAIMS.md` and the experiment records.

---

# APPENDIX A — The IDRiD derivation task, as originally written

Part A above is the working version. This is the original task text, kept verbatim so
nothing is lost to paraphrase. Where the two differ, Part A governs (it adds the
three-definitions sensitivity check on disc diameter).

> Next task: the IDRiD derivation test. This is a gate — nothing about additional DME data
> proceeds without its result. CPU only, no GPU needed, and it can run alongside ConvNeXt.
>
> BACKGROUND — the dataset search came back and most of it does not survive verification.
> Record these in data/DATASETS.md as verified findings so nobody re-derives them:
>
> - DDR was recommended as having hard-exudate masks PLUS fovea coordinates PLUS optic disc
>   location. It does not. Its 757-image segmentation subset carries four lesion masks only
>   (MA, haemorrhage, hard exudate, soft exudate) — no fovea, no optic disc. Verified against
>   the official release and three independent sources. Unusable for geometric derivation.
> - BRSET was recommended on the claim that its DR_SDRG column isolates "hard exudates within
>   1-2 disc diameters of the fovea", i.e. our grade 1. It does not. Verified against the
>   PhysioNet data dictionary: DR_SDRG is a RETINOPATHY severity scale (0 none, 1 mild
>   background, 2 moderate background, 3 severe/pre-proliferative, 4 proliferative). The
>   Scottish scheme's separate maculopathy grade is not in the release. The only oedema label
>   is binary macular_edema. Same for mBRSET.
> - HEI-MED, e-ophtha EX, DIARETDB1, FGADR: all lack native fovea and/or optic disc
>   annotations. Using them would need a landmark detector, importing cascaded coordinate
>   error into what is meant to be ground truth.
> - OLIVES: near-infrared SLO, not colour fundus. Cannot enter the training pool.
> - The only viable candidate is SUSTech-SYSU: 1,219 images, Figshare, open access, Chinese
>   cohort, no overlap with Messidor/IDRiD/EyePACS/APTOS. Verified from the Nature Scientific
>   Data paper — it provides exudate annotations, the optic disc BOUNDING BOX, and fovea
>   location. Two risks, both to be checked from the actual download if we ever get there:
>   exudates are bounding boxes not pixel masks, so distance-to-box-edge underestimates
>   distance and biases true grade 1 into grade 2 — the exact class we are short of; and the
>   paper refers only to "exudates" and trained its box refiner on IDRiD labels "combining
>   soft exudates and hard exudates together", so it may not separate hard from soft. Since
>   the DME grade is defined on hard exudates only, a merged label would falsely produce
>   grade 2 whenever a cotton-wool spot sits near the macula.
>
> THE TEST
>
> The DME grade is a deterministic geometric function: grade 0 if no hard exudates; grade 2 if
> the minimum distance from any hard-exudate pixel to the fovea centre is <= one optic disc
> diameter; grade 1 otherwise. IDRiD's 81-image segmentation subset has everything needed to
> test whether that derivation reproduces expert grading: hard-exudate pixel masks, fovea
> coordinates, optic disc boundary, and — critically — the true 3-class grade for those same
> images from the 516-image set.
>
> Pre-register the acceptance criterion BEFORE computing anything, and commit it. Do not look
> at agreement and then decide what counts as good. My proposal, adjust if you have a better
> one and say why: exact-match rate overall, the full 3x3 confusion matrix, and per-class recall
> with grade 1 reported separately — grade 1 is the entire reason this exercise exists, so an
> overall match rate that hides a grade-1 failure is not a pass.
>
> Report honestly on the mechanics too:
> - How many of the 81 segmentation images have a 3-class grade in the 516 set. If it is fewer
>   than 81, say so; the test is only as strong as the overlap.
> - How you computed one disc diameter from the IDRiD optic disc annotation, and whether that
>   is the major axis, the equivalent diameter, or something else — the threshold is defined in
>   disc diameters, so this choice is part of the method and could itself explain a mismatch.
> - Whether IDRiD's masks separate hard from soft exudates. They should, but confirm it rather
>   than assume, since that separation is what makes IDRiD the ideal-conditions test.
>
> Then the verdict, in these terms:
> - If the derived grade reproduces expert grading under ideal conditions — pixel masks, expert
>   landmarks, hard/soft separated — the method is valid, and SUSTech-SYSU becomes worth the
>   engineering with its two caveats stacked on top. Come back to me before starting it.
> - If it does not reproduce under ideal conditions, the whole derivation route dies here at
>   zero GPU cost, and that is itself a reportable negative worth a findings entry: the clinical
>   grading definition is not recoverable from the annotations the field actually publishes,
>   which would explain why nobody has released a second 3-class DME corpus.
>
> Either way write it up as a finding. If the derivation partly works — say it reproduces
> grade 0 and 2 well but not grade 1 — that is the most interesting outcome of the three and
> needs its own analysis, because it would say the boundary case is where the definition and
> the annotations diverge.
>
> Do not proceed to SUSTech-SYSU or any other dataset on the strength of a partial result.
> Report first.
>
> نکته‌ای که در پرامپت اضافه کردم و در نسخه‌ی قبلی نبود: چطور یک قطر دیسک را از حاشیه‌نویسی
> IDRiD حساب می‌کند. دیسک نوری بیضی است نه دایره، پس «قطر» می‌تواند محور بزرگ باشد، یا قطر
> معادل مساحت، یا چیز دیگر. اگر انتخاب اشتباه باشد، ممکن است آستانه جابه‌جا شود و اشتقاق را
> به‌ناحق رد کند. این باید صریح گزارش شود، نه پنهان در کد.
