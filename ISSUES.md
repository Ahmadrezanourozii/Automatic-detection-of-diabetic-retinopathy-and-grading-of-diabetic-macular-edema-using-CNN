# ISSUES.md — problems, diagnoses, fixes

---

## §1. The reported thesis results are not reproducible  — 2026-08-25, OPEN, critical

**Symptom.** Thesis chapter 4 (`thesis-chegeni/tex/chapter4.tex`) reports 91.6 % accuracy
on 5-class DR over a 262-image test set, and 87.6 % on 3-class DME over an 89-image test
set, plus a three-row preprocessing ablation (green −10/−7, CLAHE −6/−8, blur −3/−4).
`CLAUDE.md` reports slightly different figures (91.7 / 87.4) for the same work.

**How it was diagnosed.** Four independent checks, all from files on disk:

1. **No split of any dataset in this project produces n = 262 or n = 89.**
   IDRiD official test = 103. IDRiD DR≥1 test = 69. 15 % of Messidor-2 = 262 — but
   Messidor-2 has no 3-class DME label at all (see §2), so it cannot produce the 89.
   The two numbers do not come from one coherent protocol.

2. **The only evaluation output ever saved is `results/DR_confusion_matrix.png` and
   `results/DME_confusion_matrix.png` (30 Jun 2026).** Their supports match the IDRiD
   official test set exactly (34/5/32/19/13) and its DR≥1 subset (11/10/48). Reading the
   cells off those matrices:
   - DR: 28 correct of 103 → **27.2 %**, against a 33.0 % majority floor. The model
     predicted "Moderate" for 89 of 103 images and never once predicted No_DR or Severe.
   - DME: 11 correct of 69 → **15.9 %**, against a 69.6 % floor. The model predicted
     class 0 for 67 of 69 images.

3. **Every archived training run diverged.** Parsing the TensorBoard event files under
   `results/logs/`: DR validation accuracy runs 0.315 → 0.055 over 9 epochs while
   validation loss rises 1.86 → 3.40; DME validation accuracy reaches 0.000. The longest
   run is 9 epochs. No run in the archive ever exceeded 0.32 validation accuracy on either
   head.

4. **No `results.json`, no metrics file, no run log** exists anywhere in the project.
   The ablation study would have required at least four additional full training runs;
   there are three log directories, all of them diverged and none of them an ablation.

**Root cause (of the divergence).** Not yet isolated, but the training loop is the prime
suspect, not the data — see `IDEAS.md` E01. Contributing factors visible in the code:
`unfreeze_backbone()` sets every non-BN layer trainable at 1e-5 while `epoch_learning_rate`
in the first archived run logs as < 5e-5 and the loss climbs monotonically; class weights
are applied to a two-head model where Keras broadcasts them across both outputs; and the
DME head is trained on labels that include DR=0 images whose DME loss is supposed to be
masked. Any of these can produce exactly this collapse.

**Root cause (of the reported numbers).** The numbers do not correspond to any computation
that happened in this project. The most likely explanation is that they were written into
the progress report as targets and then carried into the thesis as results.

**Fix.**
1. Treat every number in chapter 4, `CLAUDE.md`, and both Persian progress reports as
   **unverified** until recomputed. They must not be cited, compared against, or used as a
   baseline to beat.
2. Rewrite chapter 4 from `results.json` files produced by actual runs, via the
   figure-generation script (`DELIVERABLES` in the project prompt). No hand-typed numbers.
3. Rewrite the comparison table (§ tab:comparison) and the ablation table
   (§ tab:ablation) or delete them until the runs exist.

**How to recognise it if it comes back.** Any reported number that (a) has no
`results.json` with a git SHA behind it, or (b) sits above the majority-class floor by more
than the bootstrap interval without a paired comparison. `src/audit_baseline.py`
regenerates this whole diagnosis in about two seconds.

**Open question for the owner.** Chapter 4 is written and typeset. Rewriting it from real
numbers is the only defensible path, and the real numbers will initially be lower. This is
a decision the owner must make explicitly, not one to be made by default.

---

## §2. The DME class names in the code are clinically wrong  — 2026-08-25, OPEN

**Symptom.** `config.py` defines `DME_CLASSES = ['Mild_DME','Moderate_DME','Severe_DME']`,
and chapter 4 describes the three levels as خفیف / متوسط / شدید (mild / moderate / severe).

**Root cause.** IDRiD's column is *"Risk of macular edema"*, not *"DME severity"*. Its
levels are defined by the distance from the nearest hard exudate to the macula centre:

| Grade | IDRiD definition | Correct name |
|---|---|---|
| 0 | no visible hard exudates | **No DME** |
| 1 | hard exudates present, > 1 disc diameter from the macula centre | **Non-referable DME** |
| 2 | hard exudates within 1 disc diameter of the macula centre | **Referable DME** |

Grade 0 is *not* "mild DME" — it is the absence of disease. Calling it "Mild_DME" means
every confusion matrix, every per-class metric and every sentence of chapter 4's DME
analysis is labelled with the wrong clinical concept.

**Fix.** Rename throughout: `['No_DME','Non_referable_DME','Referable_DME']`. Correct the
thesis text. This also fixes the reasoning about the DR≥1 gate — see `PROTOCOL.md` §5.1.

---

## §3. The "Messidor-2" mirror contains Messidor-1-format files  — 2026-08-25, OPEN

**Symptom.** Of the 1 744 images in `Datasets/Messidor-2/messidor-2/messidor-2/preprocess/`,
1 057 are named `########_#####_####_PP.png` (Messidor-2 convention) and **687 are named
`IM######.JPG`** (Messidor-1 convention).

**Why it matters.** Messidor-1 and Messidor-2 overlap: Messidor-2 was built partly from
Messidor-1 examinations. If Messidor-1 is later acquired for external DME validation
(`PROTOCOL.md` §2), the external set may contain images the model trained on. That would
invalidate exactly the result the thesis most needs.

**Fix.** Perceptual-hash every image in every corpus before any merge, and record the
duplicate clusters in `data/DATASETS.md`. Do this *before* acquiring Messidor-1, so the
overlap is known in advance.

---

## §4. Messidor-2 images are already downsampled to 512×512  — 2026-08-25, OPEN

**Symptom.** Every file in the Messidor-2 folder is exactly 512×512 (the folder is named
`preprocess/`). IDRiD is full resolution (4288×2848); APTOS is native and variable
(3216×2136 down to 819×614).

**Consequence.** Any resolution experiment above 512 px (`IDEAS.md` I10) is impossible on
Messidor-2 without re-downloading the originals, and microaneurysm-scale detail may already
be destroyed in that corpus. A resolution ablation run on the pooled corpus would therefore
be confounded by dataset of origin. Run resolution experiments on IDRiD + APTOS only, or
re-acquire Messidor-2 at native resolution.

---

## §5. `glob.glob` with nested wildcards is unreliable on the Google Drive mount — 2026-08-25, FIXED

**Symptom.** `glob.glob(".../results/logs/*/*/events*")` returned 0 results from inside a
script while returning 5 from an interactive one-liner against the same path.

**Fix.** Use `os.walk` for any recursive traversal of the Drive mount. Applied in
`src/audit_baseline.py`.

**Recognise it by:** a file-finding step that silently returns an empty list on the Drive
mount while `ls` shows the files.

---

## §6. `Summary.Value` field numbers  — 2026-08-25, FIXED

**Symptom.** A hand-written TensorFlow event-file parser returned zero scalars from valid
event files.

**Root cause.** In the `Summary.Value` protobuf, `tag` is **field 1** and `node_name` is
field 7 — the opposite of what was assumed.

**Fix.** Applied in `src/audit_baseline.py`. Kept here because the parser exists only to
avoid importing TensorFlow, which takes minutes over the Drive mount.

---

## Things not to redo

Ideas that were tried and failed, so neither of us tries them again in three months.

- **`load_idrid_all()` / `--no-holdout` / `memorize=True` in the old codebase.**
  This mode sets `test = train` (all 516 images) and its own docstring says the goal is
  "to show the highest possible accuracy numbers rather than to measure generalization".
  `DR_DME_Combined_GPU_Colab.ipynb` is built entirely around it. **Delete this code path.**
  It produces numbers that cannot go in a thesis, and its existence next to the honest
  loader is how a training-set number ends up quoted as a test-set number.
- **Fine-tuning the whole DenseNet121 at 1e-5 with batch 16 on 413 images, with class
  weights applied to a two-head model.** Diverged in every archived run. Do not simply
  retry it with a different learning rate before E01 has established that the
  representation carries signal at all.
