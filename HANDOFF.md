# HANDOFF.md — session handover, 2026-09-08

Written because the previous session ran out of context. **Read `STATE.md` first, then this
file, then `PROTOCOL.md`.** This document covers what was done, what is running, what stopped
mid-flight, and the exact next actions.

Repository HEAD at handover: **`86404c3`**, branch `main`, pushed.
One uncommitted file: `results/VERIFICATION-P8-RETFound.md` (complete, just needs committing).

---

## 1. The single most important thing to know

**The project's headline is `E11FULL`**, declared in `data/selected_model.json`:

| | DR QWK | basis |
|---|---|---|
| internal, matched calibration | **0.8954** | 2 260 images, 5-fold OOF |
| **external, APTOS held out** | **0.9026** [0.8941, 0.9108] | 3 662 unseen images |

Selection is by **development-pool ranking, never by test score**. `src/report.py` reads the
declared model from `data/selected_model.json` and **refuses to infer it from scores**; when
another run scores higher it prints a loud note and reports the declared model anyway.

Anchored by **F9**: the full pipeline is **+0.234 QWK** over frozen ImageNet features and a
linear ordinal head, on identical images, folds, decoding and calibration.

---

## 2. Findings ledger — `FINDINGS.md` is authoritative

⚠️ **`CLAIMS (2).md` uses a DIFFERENT F-numbering.** A crosswalk table was inserted into both
`CLAIMS (2).md` and `FINAL-PROMPT.md`. **Always cite `FINDINGS.md` numbers.**

| CLAIMS says | actually is |
|---|---|
| F1 | `FINDINGS.md` F1 (only coincidence) |
| F2 | `PROTOCOL.md` §4.1 record + `FINDINGS.md` F3 |
| F3 | `FINDINGS.md` **F7** |
| F4 | `FINDINGS.md` **F8** |
| F5 | `IDEAS.md` **I23** — no F number |
| F6 | `FINDINGS.md` **F2** |
| F7 | `ISSUES.md` **§24, §26, §27** — no F number |

Current `FINDINGS.md`: F1 (Mild collapse is calibration), F2 (Messidor-1 unusable as external
DME), F3 (per-class dominated by cut-points), F4 (~200 images to recalibrate), F5 (macro-recall
must not lead), F6 (mean-only recommendation can harm), F7 (DME ceiling is data), F8 (RETFound
better frozen / worse fine-tuned), F9 (+0.234 over trivial baseline), **F10 (DME grade not
recoverable from published annotations — Part A verdict)**.

---

## 3. Experiment results since the objective changed to "maximise accuracy"

**Seven falsifications, one confirmation.** Every run pre-registered a hypothesis and a
falsifying outcome before launch.

| run | intervention | result |
|---|---|---|
| E13gate | fovea localiser gate | **PASSED** — median 0.196 DD, 90th 0.433 DD |
| E14MAC | macula-centred crop (I07) | **falsified** — DME QWK +0.0237 [−0.0094, +0.0566] → F7 |
| E15LPFT | LP-FT (I21) | **falsified** — DR QWK −0.0008 [−0.0126, +0.0105] at matched cuts |
| E17NAT | native resolution (I20) | **falsified** — +0.0086 [−0.0025, +0.0199]; I10/I10b/I10c closed |
| I23 | ensembling | **falsified externally** — dev +0.0136 → APTOS +0.0025 [−0.0016, +0.0066] |
| I24 Stage 2 | RETFound fine-tune | **falsified** — −0.0399 [−0.0669, −0.0124] vs E09 (2 of 5 folds) |
| E20CORAL | CORAL ordinal head | **falsified** — −0.0076 [−0.0184, +0.0032] |
| E22CORN | CORN ordinal head | **falsified** — −0.0479 [−0.0656, −0.0325], significantly worse |
| **E19E11C** | **EfficientNet-B3, folds 3–4** | **CONFIRMED** — E11FULL beats E10 by **+0.0207 [+0.0105, +0.0320]** |

**Ordinal-loss line is closed.** CORN's conditional subsets collapse under class imbalance:
threshold 3 trains on only **265 of 2 260 rows (11.7 %)**.

**CORAL bias-ordering check** (pre-registered): all 10 tensors strictly ordered — the guarantee
genuinely held, so the null is about CORAL, not a broken implementation.

---

## 4. ~~WHAT STOPPED MID-FLIGHT~~ — **all three items DONE on 2026-09-08**

### 4.1 `E21CNXA` — **FETCHED AND CLOSED.** ConvNeXt-tiny falsified

Fold-matched against E08 on the same 910 out-of-fold images at matched calibration:
**DR QWK 0.8267 vs 0.8628, −0.0362 [−0.0593, −0.0153], significant.** DME indistinguishable.
Write-up in `IDEAS.md` under I05; document `docs/generated/matched_cnx_e08.md`.

Two caveats recorded rather than smoothed over: the run carries **`hypothesis: ""`**, so its
criterion was reconstructed after the number was known (`ISSUES.md` §28 — `build_kernel.py`
now refuses a training notebook without one); and **fold 1 was truncated** at its best epoch
39/40 while fold 0 plateaued at 23/40, with the significant deficit concentrated in the
truncated fold. **Folds 2–4 not launched** — E11FULL is at 0.8954 and ConvNeXt's converged
fold sits below E08's 0.8646, so nothing in this line reaches the headline. The ~13.6 h
rescue is priced in `IDEAS.md` so the choice stays the owner's.

<details><summary>original instruction, kept for the record</summary>

### 4.1 `E21CNXA` is COMPLETE and UNFETCHED

ConvNeXt-tiny, folds 0–1, account 1. **It finished and was never fetched or analysed.**

```bash
cd "/Users/ahmadrezanourozi/Desktop/Alireza Thesis-25/dr-dme"
set -a && source .env && set +a
python3 kaggle/fetch.py --run-id E21CNXA --slug dr-dme-e21cnxa
DRIVE="/Users/ahmadrezanourozi/Library/CloudStorage/GoogleDrive-ahmadrezanourozii@gmail.com/My Drive/Alireza"
.venv/bin/python src/compare_matched.py --a runs/E21CNXA --b runs/E08 \
    --datasets "$DRIVE/Datasets" --out docs/generated/matched_cnx_e08.md
```

**Fold-match the comparison**: E21CNXA has folds 0–1 only, E08 has 0–4. Restrict E08 to folds
0–1 (see the pattern used for I24FT01 vs E09 in the session log, or extend
`compare_matched.py`). Control is **E08 at matched calibration, DR QWK 0.8646**.

Background at the time it was cancelled once before (session wall): fold 0, epoch 27/40,
DR QWK 0.857 and still rising. **convnext_tiny is ~207 s/epoch vs densenet121's ~65 s**, so
5 folds ≈ 11.5 h — it must stay split. Folds 2–3 and fold 4 have **not** been launched.

</details>

### 4.2 Uncommitted file — **DONE**, committed in `f769bf2`.

### 4.3 Part B chapters 3 and 4 — chapter 4 not started

- `thesis/parta_report.tex` — **DONE**, lints clean
- `thesis/chapter3.tex` (روش کار) — **DONE**, lints clean
- `thesis/chapter4.tex` (نتایج) — **DONE 2026-09-08**, lints clean
- Chapters 1 and 2 — **deliberately held** until paper verification finishes

⚠️ **Chapter 3 and the Part A report were NOT as finished as this file said.** Eight decimals
across them were written with the integer and fractional groups swapped — chapter 3 told the
reader the default sigmoid decode threshold was **5.0**. Three real errors were underneath:
the gated DME floor is **69.8 %** not 69.6 %, Part A's exact match is **96.3 %** not 96.6 %,
and the crosswalk overlay figure (0.78–0.96) **had no artefact at all** and could not be
reproduced — `src/verify_crosswalk_pairs.py` regenerates it at **0.65–0.75**, same verdict.
`ISSUES.md` §29. **Every number now comes from `docs/generated/thesis_numbers.json`** (315
values, generated by `src/thesis_numbers.py`) and `farsi_lint.py --numbers` refuses any
Persian numeral that is not in it. Run it before calling any chapter finished:

```bash
python3 src/thesis_numbers.py --datasets "$DRIVE/Datasets"   # regenerate the ledger
python3 tools/farsi_lint.py --numbers thesis/*.tex           # must print 0
```

---

## 5. Part A — the IDRiD derivation gate (COMPLETE, verdict final)

**Verdict: DOES NOT REPRODUCE.** Pre-registered in `docs/PARTA_preregistration.md` *before* any
number. Script `src/idrid_derivation_gate.py`, output
`docs/generated/idrid_derivation_gate.json`, written up as **F10** and
`thesis/parta_report.tex`.

**Two limitations that are themselves findings:**

1. **IDRiD publishes no crosswalk** between segmentation numbering (`IDRiD_01..81`) and grading
   numbering (`IDRiD_001..413`/`103`). Reconstructed from two independent signals that both had
   to agree — pixel correlation > 0.9999 **and** optic-disc centroid within 100 px.
   **54 of 81 confirmed.** Neither signal works alone.
2. **The segmentation subset is selected for "images worth segmenting"**, so the grade
   distribution is **[1 grade-0, 3 grade-1, 50 grade-2]**. A constant "always grade 2" predictor
   scores **92.6 %**; the derivation scores 96.3 %. **The test has 4 informative images, not 54.**

**Both competing explanations were ruled out** (as the owner required):
- *Wrong crosswalk?* No — overlaying mask + fovea on the matched grading image gives mean
  absolute pixel difference **0.78–0.96 of 255** (JPEG noise). Same eye, visually confirmed.
- *Different published definition?* No — IDRiD's criteria are exactly what was implemented.

**Corrected diagnosis** (an earlier, stronger claim was withdrawn): a minimum-distance rule over
a pixel mask is decided by the **single smallest annotated speck**. The blobs driving both
failures are **74 px** and **349 px** in a 2848×4288 frame, just inside the threshold. Grade 0
is defined on *"no **apparent** hard exudate(s)"* — what a clinician sees — while a mask marks
every speck. **Generalises: deriving a categorical clinical label from a segmentation mask via a
distance threshold inherits the mask's annotation floor as its decision boundary.**

**Consequence: derivation route closed; SUSTech-SYSU not worth the engineering.** A size
threshold is the obvious repair and explicitly **cannot** be established here (two failures
among four images = fitting to two examples, validating on none).

---

## 6. Two Kaggle accounts — both working

| | account 1 | account 2 |
|---|---|---|
| user | `ah22reza` | `reza12123` |
| kernels dir | `kaggle/` | `kaggle2/` |
| token in `.env` | `KAGGLE_API_TOKEN` | `KAGGLE_API_TOKEN_2` |
| quota | 30 h/week | 30 h/week |

```bash
python3 kaggle/build_kernel.py --run-id XXX --account 2 ...
KAGGLE_API_TOKEN="$KAGGLE_API_TOKEN_2" .venv/bin/kaggle kernels push -p kaggle2/dr-dme-xxx --accelerator "GPU T4 x2"
KAGGLE_OWNER=reza12123 python3 kaggle/fetch.py --run-id XXX   # fetch from account 2
```

**Shared RETFound dataset verified IDENTICAL on account 2** (`runs/ACCT2VERIFY/`):
mounted at `/kaggle/input/datasets/ah22reza/retfound-cfp-encoder/`, 1 213 299 887 bytes,
sha256 `847f9dd0…`, embedded source hash `e1e4f66a…`, 294 tensors. The `ah22reza/` path is
kept unchanged as instructed. **The dataset must NEVER be made public — gated, CC-BY-NC.**

**Two gotchas confirmed:**
- Account 2 initially had **no kernel internet** (`Could not resolve host: github.com`). The
  owner has since verified the phone number and the git-clone path works. A
  `--code-dataset` fallback exists in `build_kernel.py` for any future account without internet
  (ships code as a private dataset with a `COMMIT` file the kernel refuses to run without).
- **`ISSUES.md` §15 recurred**: `/kaggle/input` contained exactly one directory, `datasets`, so
  every dataset was flattened underneath. **Always search recursively for mounted files.**

---

## 7. Paper verification (task 2, partially done)

Scope agreed with the owner: DOI + citation for all 12; **only the numbers relied on in the
Evidence Map**; tag `UNVERIFIED` rather than delete; use PDFs for P7–P12.

**⚠️ We hold NO PDFs of P1–P12.** The only PDFs on the Drive are four unrelated papers. P12 was
read from PubMed Central, P2/P8/P11 from open-access full text. Recorded so sourcing is not
overstated.

**All 12 DOIs resolve** with matching author/journal/year — **the ledger is not fabricated.**
One correction: **P9 (Tomić) is dated 2023** in the DOI record, not 2024.

| paper | status | file |
|---|---|---|
| **P12** Duggal | G1, G2, G4, G5, G8 **all exact** | `results/VERIFICATION-P12-Duggal.md` |
| **P2** Poyrazer | R2, R3, R4, R5, R6, R7 **all exact** | `results/VERIFICATION-P2-Poyrazer.md` |
| **P11** Tahir | H4, H5, table duplication **verified**; one sub-claim **CONTRADICTED** | `results/VERIFICATION-P11-Tahir.md` |
| **P8** RETFound | Z4, contamination, recipe **verified**; Z3 **unverified** | `results/VERIFICATION-P8-RETFound.md` |
| P1, P3, P5, P6, P7, P9, P10 | **not started** | — |

**Three corrections the thesis must carry:**

1. **G8 is not a threshold sweep.** κ 0.65 → 0.72 with sensitivity/specificity each moving
   ~31 points is real, but the two κ values come from **different phases, different cohorts,
   and the vendor retrained the algorithm between them** (the paper says so). Cite it as
   evidence that summary agreement statistics are insensitive to operating-point changes —
   **not** as a cut-point sweep on fixed data.
2. **P11 sub-claim contradicted.** The ledger says the forest plots carry corrected values
   (Ting: table 0.25 vs plot 0.90). **False** — displayed values are Ting **0.06**, Sosale 0.93,
   Abràmoff 0.67. What *is* verified and is stronger: the FN column is duplicated from FP, and
   `676/(676+9969) = 0.06` shows the corruption **propagated into the reported sensitivities**.
3. **The *Nature* anomaly: observation verified, diagnosis NOT.** Internal IDRiD AUROC and
   external APTOS→IDRiD AUROC are both reported as **0.822 (95 % CI 0.815, 0.829)** — identical
   to three decimals including the interval, from two different training regimes. **Do not
   assert an error in a *Nature* paper on this evidence.** Phrase as: *"we could not determine
   from the published material whether this reflects a coincidence or a transcription error."*
   Resolving it needs Supplementary Table 3.
4. **Z3 must not be cited.** The claim (IDRiD AUPR *P* = 0.81 vs AUROC *P* < 0.001) is not in
   the main text, and the main text appears to **contradict** it: *"The AUPR results of RETFound
   were also significantly higher."*

---

## 8. Persian writing infrastructure (task 3)

- **`GLOSSARY.md`** — locked terminology; ezafe is **«ـهٔ»** everywhere (never «ه‌ی»); Persian
  digits in prose, Latin digits in formulas/tables; QWK/AUROC/ECE/Brier/F1 untranslated.
- **`tools/farsi_lint.py`** — counts forbidden words («می‌باشد», «می‌گردد», «مورد … قرار گرفت»,
  «لازم به ذکر است», Arabic ك/ي, mixed ezafe, sentences > 260 chars). Tables/figures excluded
  from prose counting. **A chapter is not finished until the count is 0.**

```bash
python3 tools/farsi_lint.py thesis/chapter3.tex thesis/chapter4.tex
```

Both written chapters currently lint at **0 violations, ezafe consistent**.

**Still to do for figures:** `fix_farsi` helper with `arabic_reshaper` + `python-bidi`, a
registered Persian font (Vazirmatn / XB Niloofar), `matplotlib.rcParams['font.family']`, and
**visual confirmation** that captions render joined and right-to-left. Vector output (PDF/SVG).

---

## 9. Standing rules — these bind every future action

From `PROTOCOL.md`:

- **§3** — rank by validation, never test. Model selection is a *declared fact*
  (`data/selected_model.json`), not inferred from scores.
- **§4.1** — **matched calibration before attribution.** No delta is attributed to architecture
  until both models sit at cross-fitted cut-points. Has reversed **four** claims in **two**
  directions with two sign reversals.
- **§4.2** — no multiple-comparison correction, so per-class credibility comes from
  **replication across conditions**; per-class claims inherit every aggregate confound.
- **§5.1** — DME **ungated is primary**; every DME number quoted next to its floor.
- **§9** — **configuration is not consumption.** Every run writes a consumption manifest;
  runs may not be combined/compared unless manifests agree (`src/manifest.py`).
  `--acknowledge-consumption-diff` allows deliberate cross-source comparison but **stamps the
  difference into the output document**.
- **§10** — **a check that only ever fires one way is not a check.** Each rule carries its
  record of firing in both directions.

From `ISSUES.md`:
- **§24** — `--script` was a dead flag; the guard now checks the *generated notebook*.
- **§26** — completing a run's folds silently changed the image source. **Diff
  `kernel-metadata.json`'s `dataset_sources`, not just config.**
- **§27** — **a check that reports success without examining anything.** Assert something was
  inspected; never return an accumulator initialised to the passing value.

**Every run pre-registers hypothesis + falsifying outcome before launch. Budget ~30 h/week per
account, no single run over ~10 h, checkpoint every epoch.**

---

## 10. Recommended order for the next session

1. ~~Fetch and analyse `E21CNXA`~~ — **DONE**, falsified, closed.
2. ~~Commit `results/VERIFICATION-P8-RETFound.md`~~ — **DONE**.
3. ~~Write `thesis/chapter4.tex`~~ — **DONE**, lints clean against the numbers ledger.
4. ~~Launch ConvNeXt folds 2–3 and 4~~ — **deliberately not launched**, see §4.1. **Swin is
   the open question**: it is the last untried backbone, and the same argument that closed
   ConvNeXt (nothing in the backbone line reaches 0.8954) applies to it before it is launched.
   Recommend closing the backbone line unless the owner wants the completeness.
5. **Continue paper verification**: P1 (S5), P6 (M2/M4/M6), P9 (T4), P10 (Y2), P5 (D6), P3 (N1),
   P7 (V3/V4/V5).
6. **I16 component 2** — single-output CNN (DR head only). Never measured whether the shared
   DME head helps or hurts DR. One run, one change.
7. **Chapters 1 and 2** only after verification completes.

**Skipped deliberately, with reasons recorded:** I15 (auxiliary exudate head) — 81 masks are
3.6 % of the pool against a ±0.03 interval, and E14MAC already showed the decisive region does
nothing. Refusing to pre-register a hypothesis not believed, to buy a sixth null F7 predicts.
The masks stay useful as the **I14 saliency diagnostic** (interpretability, not performance).

---

## 11. Open questions for the owner

1. **T1 target unchosen** — candidates A/B/C with verified provenance and specificity costs in
   `docs/T1_referral_threshold_candidates.md`. Not blocking.
2. **Pursue Z3 / the *Nature* anomaly?** Both need Supplementary Table 3.
3. **Credentials** — rotation deferred by decision (`ISSUES.md` §25, not blocking). `.env` holds
   `GITHUB_TOKEN`, `KAGGLE_API_TOKEN`, `KAGGLE_API_TOKEN_2`, `HF_TOKEN`; gitignored, verified
   absent from history. All are in chat transcripts — accepted risk.
4. **DME data search** — the owner has another model searching for 3-class DME sources. **F10
   now says the derivation route is closed**, so anything found must carry native 3-class
   grades, not derivable ones. Run the pHash overlap + provenance check before training.
