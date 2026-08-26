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

## §7. Standard perceptual hashing does not work on fundus images  — 2026-08-25, FIXED

**Symptom.** The first version of `src/dedup_groups.py` used the textbook recipe — 64-bit
dHash and pHash, near-duplicate if Hamming distance ≤ 6. Run on 40 IDRiD images, every one
of them a different patient, it reported **26 duplicate pairs**.

**How it was diagnosed.** Rather than tuning the threshold by eye, both populations were
measured on 60 IDRiD images (1 770 distinct pairs), with true duplicates simulated by
re-encoding and resizing the same image:

| Hash | Distinct pairs | Same image, degraded | Separated? |
|---|---|---|---|
| 64-bit dHash | min **2**, p1 4, median 15 | 0–2 | **no** |
| 64-bit pHash | min **2**, p1 6, median 22 | 2–6 | **no** |
| 64×64 NCC | max **0.9911** | 0.9921–1.0000 | barely |
| **256-bit dHash (16×16)** | **min 28**, p1 36, median 66 | **0–8** | **yes, wide gap** |

**Root cause.** Every fundus photograph is a bright ellipse on a black ground. Reduced to an
8×8 thumbnail they are all nearly the same picture, so a 64-bit hash carries almost no
patient-specific information. This is a property of the imaging modality, not a bug — the
recipe is simply wrong for this domain.

**Why it mattered.** A false-positive duplicate merges two unrelated patients into one
group. At the observed rate that would have merged a large fraction of the corpus into a few
giant groups, wrecking the stratification, starving the folds, and — worst — doing it
silently, because a grouped split that is too *coarse* produces conservative-looking numbers
that nobody thinks to question.

**Fix.** 256-bit dHash on a 16×16 grid, threshold 16 bits, which sits in the empty gap
between the two populations. NCC on a 64×64 contrast-normalised thumbnail is recorded
alongside as a second opinion but is deliberately **not** the criterion, since it does not
separate. Re-run on the same 60 images: 60 groups, zero duplicate pairs.

**How to recognise it if it comes back.** Any grouping pass that merges an implausible
fraction of a corpus. Before trusting a similarity threshold on a new corpus, measure both
populations — distinct pairs and simulated duplicates — and confirm there is a gap. Do not
pick the threshold from a blog post.

**Still open.** This catches re-encodings, mirrored copies and cross-corpus overlap. It does
**not** catch fellow-eye pairs, which are not visually near-identical. Messidor-2's
examination pairing still has to come from the upstream release or from an optic-disc
laterality pass, and until it does, every Messidor-2 result carries that caveat.

---

## §8. IDRiD image names are not unique across its official splits  — 2026-08-25, FIXED

**Symptom.** `src/check_invariants.py` reported "cached manifest matches the source label
CSVs — **413** rows identical" for a corpus that has **516** images.

**Root cause.** IDRiD numbers its training set and its testing set independently, both
starting at `IDRiD_001`. So `IDRiD_001` names two different images, of two different
patients, with different labels:

| name | split | DR | DME |
|---|---|---|---|
| IDRiD_001 | train | 3 | 2 |
| IDRiD_001 | test | 4 | 0 |

103 names collide this way — every name in the test set.

**What it had already broken, silently.** Three things, none of which would have raised an
error:

1. **The image cache in E01 was corrupt.** It wrote one file per `name`, so each of the 103
   test images *overwrote* the train image with the same number. The run in progress had
   written 398 files for what should have been 516, and 103 of those "training" images were
   actually test images.
2. **Features would have been read against the wrong labels.** Rows 1–103 of the training
   set would have received test images' pixels while keeping training labels — simultaneous
   label corruption and train/test contamination.
3. **The grouping was wrong.** `group = name` merged the train and test `IDRiD_001` into one
   group, so a group-wise split would have treated two unrelated patients as one unit.

Any accuracy that came out of that would have been meaningless, and nothing in the run would
have looked wrong.

**Fix.** Every row now carries `uid = f"IDRiD_{split}_{name}"`, unique across the corpus.
`uid` is used for the cache filename, the feature index and the group. `name` is kept for
display only. Applied in `data_idrid.py`, `e01_linear_probe.py`, `dedup_groups.py` and
`check_invariants.py`; the corrupt cache was deleted and E01 restarted from scratch.

A new invariant, "image identifiers are unique", now fails loudly if this recurs, and
`build_cache` asserts uid uniqueness before writing anything.

**How to recognise it if it comes back.** A count that is short by exactly the size of one
split. More generally: **never key on a filename that a dataset's own directory structure
disambiguates.** IDRiD, and datasets like it, rely on the folder to make the name unique.

**Worth noting.** This was found by an invariant check written for an unrelated purpose,
before it had produced a single number — which is the entire argument for writing
`check_invariants.py` before the first result rather than after the tenth.

---

## §9. Kaggle's P100 cannot run the preinstalled PyTorch  — 2026-08-25, FIXED

**Symptom.** Run E05 died two minutes in, immediately after spending those two minutes
rebuilding the 2 260-image cache:

    torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device

**Root cause.** Kaggle assigned a **Tesla P100**, which is compute capability **sm_60**. The
preinstalled `torch 2.10.0+cu128` ships kernels for **sm_70 and above only**. Torch happily
reports `cuda.is_available() == True` and names the device — it just cannot execute anything
on it. The failure surfaces at the first real CUDA call, which happened to be
`model.to(memory_format=channels_last)`, so the traceback points at a line that is not the
problem.

**Fix, two parts.**

1. **Pin the accelerator — with the right key.** The first attempt wrote
   `"accelerator": "nvidiaTeslaT4"` into `kernel-metadata.json`. **Kaggle silently ignores
   that key** and handed out a P100 again. The key it actually reads is **`machine_shape`**,
   and the CLI equivalent is `kaggle kernels push --accelerator "GPU T4 x2"`. Confirmed by
   reading `kaggle_api_extended.py`: `request.machine_shape = acc or meta["machine_shape"]`,
   with the comment "the allowed names are in an enum that is not currently included in
   kagglesdk" — so there is no validation and a wrong key fails silently. Valid strings are
   the ones the notebook UI shows, e.g. `"GPU T4 x2"`.
2. **Fail fast.** `src/train.py` runs a real 8×8 matmul on the GPU before touching the
   data, and exits with the device's compute capability and the fix in the message. Without
   it the run burns two minutes of quota rebuilding a cache it will never use, and the error
   arrives disguised as a `channels_last` problem. This worked on the second attempt: the
   run died in seconds with an accurate message instead of two minutes in with a misleading
   traceback.

3. **Runtime fallback.** Because the pin is not always honoured, the notebook now probes the
   device and, if its kernels cannot run, installs `torch==2.5.1+cu121` (which supports
   sm_50–sm_90) before launching training. Costs a few minutes; saves the run.

**How to recognise it if it comes back.** `cuda.is_available()` true, device name printed,
then a CUDA error on the first tensor operation. Check
`torch.cuda.get_device_capability()` against the build's supported list — `is_available()`
is not evidence that a kernel will run.

**Collateral, fixed at the same time.** `kaggle/fetch.py` archived everything the notebook
left in its working directory. The notebook clones the repo there, so the fetch copied the
whole repository back over `runs/E05/` — including a `results.json` that belonged to a
different run entirely. It now takes only the run's own directory plus the kernel log. A
fetch that silently overwrites one run's results with another's is precisely the kind of
bookkeeping error that makes an archive worthless.

---

## §10. A missing pretraining corpus would have looked like a negative result  — 2026-08-25, FIXED (pre-emptively)

**Not a bug that fired — one that was about to.** `load_eyepacs` returns `[]` when it cannot
find a labels CSV, and `main()` treated an empty pretraining set as "no pretraining
requested". So if the EyePACS corpus were mis-attached or laid out differently than
expected, run E06 would have quietly become an exact duplicate of the E05 baseline — and
the paired comparison between them would have reported *"EyePACS pretraining makes no
difference"*, with a perfectly tight confidence interval around zero.

That is the most dangerous class of failure in this project: not a crash, but a **null
result manufactured by a silent no-op**. It looks like evidence. It would have gone into the
thesis.

**Fix.** `--pretrain-corpora X` matching zero images is now a hard exit with a message
saying so. The general rule: *whenever a flag is supposed to change what a run does, the run
must fail if that change cannot be applied.* Never let a requested condition silently
degrade into the control condition.

**Verified separately** that `tanlikesmath/diabetic-retinopathy-resized` does ship
`trainLabels.csv` (35 126 images, 17 563 patients, columns `image,level`) by downloading
just that file — the CLI file listing does not reach it within 12 000 entries because it
sorts after the whole `resized_train/` tree.

**Recognise it by:** a run whose distinguishing feature produces exactly the baseline's
numbers. Check the log for the count the corpus loader printed before believing the
comparison.

---

## §11. A text patch landed in two functions at once  — 2026-08-25, FIXED

**Symptom.** E06 died twenty minutes in, right after building a 37 386-image cache and
starting the pretraining phase:

    NameError: name 'init_state' is not defined

**Root cause.** The `init_state` block belongs in `run_fold()`. It was added with a
string replacement whose anchor —

    model = MultiOutputNet(...).to(device)
    if args.channels_last:

— appears **identically in both `pretrain()` and `run_fold()`**, and `str.replace` replaces
every occurrence. So the block was inserted into `pretrain()` too, where no such variable
exists.

**Why it survived every check.** `ast.parse` passed, the module imported fine, and the
misplaced line only executes once `pretrain()` is actually called — which happens after the
corpora load, after the split check and after 37 k images are cached. Every cheap check
this project had was blind to it, and the expensive one caught it.

**Fix.** Removed the block from `pretrain()`, and added **`src/lint.py`**: a symbol-table
pass that reports any name referenced in a function but never bound there, at module scope,
or as a builtin. It needs no data, no GPU and no imports, runs in milliseconds, and is now
run before every push.

**How to recognise it if it comes back.** A `NameError` deep into a run, in a function you
did not think you were editing. When patching by string replacement, either anchor on
something unique to the target function or use an editor that reports how many matches it
replaced — an anchor that appears twice in a file is a landmine.

---

## §12. The 3-class DME metric was scored on a biased subset  — 2026-08-25, FIXED

**Symptom.** Found by reading the evaluation code rather than by a failure. `evaluate()`
selected rows for the 3-class DME metric with "does this row have exactly one candidate
grade?". That is true for all 516 IDRiD images — and also for Messidor-2's 151 *referable*
rows, because "referable" pins the grade to exactly 2. It is false for Messidor-2's 1 593
non-referable rows, whose grade is genuinely unknown between 0 and 1.

So the evaluation set was 667 images of which **394 were grade 2**:

| | n | grade 0 | grade 1 | grade 2 | majority floor |
|---|---|---|---|---|---|
| as scored (wrong) | 667 | 222 | 51 | 394 | **59.1 %** |
| IDRiD only (right) | 516 | 222 | 51 | 243 | **47.1 %** |

**Why it matters.** Every one of the 151 extra images belongs to the single class the model
finds easiest, so accuracy on that set is inflated *and* the floor it is compared against
rises by twelve points. The two errors do not cancel — they combine to make a mediocre model
look like it clears a hard bar.

**Root cause.** Confusing *supervision* with *evaluation*. A partial label is perfectly good
supervision: "not referable" genuinely constrains the threshold P(y>1), and that is the
whole reason Messidor-2 is in this project. But a corpus that can only answer one of the two
ordinal questions cannot be scored on the three-way answer. Legitimate training signal,
illegitimate test set.

**Fix.** Every row now carries `dme_label_space` — `"3class"` (IDRiD), `"binary"`
(Messidor-2), or `None`. The 3-class metric is computed on `"3class"` rows only; Messidor-2
is scored on `dme_referable_binary`, the question it can actually answer. Both numbers are
reported.

**No GPU was re-spent.** Runs archive `oof_<fold>.npz` with uids and raw logits, so
`src/recompute.py` re-scores a finished run under the corrected definition. This is the
reason to archive predictions and not just metrics.

**Recognise it by:** an evaluation subset whose class balance differs from the corpus it is
supposed to represent. Print the distribution of every evaluation set, not just its size.

---

## §13. The archived log belonged to a different run than the archived results  — 2026-08-26, FIXED

**Symptom.** `runs/E06/train.log` ended in the `NameError` of §11 and reported
`PRETRAIN ... (5 epochs)`, while `runs/E06/results.json` from the same fetch carried the
*fixed* commit `52d5ef4`, `pretrain_epochs: 4`, and complete metrics for all five folds. The
log said the run crashed; the results said it finished.

**Which one was true.** The results. Three independent confirmations: the commit in
`results.json` is the one that fixed the crash; `pretrain_epochs: 4` was only ever passed on
the relaunch; and the five `oof_*.npz` files exist at all, which a crashed run cannot
produce — re-scoring from those logits reproduces the same numbers.

**Root cause.** Kaggle carries `/kaggle/working` across versions of the same notebook, and
the notebook opened its log with mode `"a"`. The failed version's log survived into the
successful version's working directory and was archived alongside the new results.

**Why it matters more than it looks.** Nothing errored. The fetch succeeded, the files
landed, the metrics were correct. Only the *provenance* was wrong — and provenance is the
entire point of the archive. A future session reading `runs/E06/` would have concluded the
run crashed, or worse, would have paired E06's numbers with a different run's training
curves and drawn conclusions about convergence from them.

**Fix, two parts.**

1. Each run writes a fresh timestamped `train_<stamp>.log` and deletes any stale
   `train*.log` in its output directory first. No appending to a carried-over file.
2. `kaggle/fetch.py` now reads the `COMMIT` line from every archived log and compares it
   against `results.json`. A mismatch is reported loudly and the file is renamed to
   `.stale` rather than left where it will be believed. This is why the training script's
   first line has always been the commit SHA — it is what makes the check possible.

**Recognise it by:** a log and a results file that disagree about anything — the config, the
outcome, the number of epochs. Never reconcile them by choosing the more convenient one;
find out which run each came from.

---

## §14. The checker passed on exactly the numbers it was written to catch  — 2026-08-26, FIXED

**Symptom.** `src/check_thesis_numbers.py` was written to fail when the thesis states a
result no archived run produced. Run against the real chapter 4 — which contains ۹۱٫۶ and
۸۷٫۶, the two figures this whole project started from — it reported **0 problems**.

**Root cause.** The scanner recognised a percentage as `91.6\%` or `91.6%`. In a Persian
thesis the unit is a *word*: `۹۱٫۶ درصد`. Every claim in chapter 4 is written that way, so
none of them matched the "looks like a result" test and all were skipped.

**Why this one is worth its own entry.** A checker that silently passes is worse than no
checker, because it converts an unexamined document into an apparently verified one. The
output "0 numeric literals match no archived result" would have been quoted as evidence that
chapter 4 was clean.

**Fix.** The number pattern now accepts `\%`, `%`, and `درصد` as the unit, and a bare
decimal in [0,1] with two or more places counts as a result claim (a QWK, an F1, an AUC).
Re-run: **9 unexplained numbers**, comprising 91.6 % and 87.6 % in two places each, and the
entire preprocessing ablation table (10, 8, 4, 3 %).

**The general rule this cost me.** *Every checker needs a test that makes it fail.* This one
now ships with a self-test — a two-line file containing one known-bad number and one
legitimate one — and the fix was only trusted after that file produced exactly one warning.
A green result from a check that has never been seen to go red is not evidence.

---

## §15. Three separate breaks between a finished run and its external number  — 2026-08-26, FIXED

E08 finished as the best run so far (DR QWK 0.860) with five 30 MB fold-weight files sitting
in its Kaggle output, and produced **no external evaluation at all**. Three independent
faults, none of which raised an error.

**1. A one-level directory scan.** The notebook guarded the external cell with
`any("aptos" in d.lower() for d in os.listdir("/kaggle/input"))`. Kaggle mounted all five
attached datasets under a *single* `/kaggle/input/datasets/` directory holding 77 136 files,
so the scan saw one entry named `datasets` and reported APTOS missing. Training was
unaffected because `corpora.build` walks recursively — only the shallow guard was fooled.
Removed entirely: `eval_external.py` already exits loudly on an empty corpus, so it should
be the judge rather than a guess made one level up.

**2. `fetch.py` discarded every `.pt`.** That rule dates from when the only checkpoints were
the 90 MB resumable ones. The 30 MB per-fold selected weights — the thing external
validation, ensembling and Grad-CAM all need — were being deleted on arrival. Now only
`ckpt_*.pt` is dropped, and `--weights` pulls the rest.

**3. `fetch.py` called `kaggle` from `PATH`** and died with `FileNotFoundError` when it was
not there. **I then read the empty output as "the run produced no weights"**, because the
invocation was piped through `grep` and the traceback went with it. It now resolves the
repo's own `.venv/bin/kaggle` first and reports a usable message.

**The one that cost most was the third, and it was mine.** A command whose output I filter is
a command whose failure I have hidden from myself. Check the exit status, or look at the
unfiltered output, before concluding anything from an empty result.

**And a fourth, immediately after:** adding the external-only kernel mode, I wrote a function
referencing `a` — the argparse namespace — which it never received. That is precisely the bug
class `src/lint.py` exists for, and I ran `ast.parse` instead, which passes on it. Then the
flag itself failed to reach the call site and the first push shipped a full training
notebook, burning a GPU slot. **A tool only helps on the runs you actually point it at**, and
a flag is not in effect until something downstream of it has been observed to change.

---

## §16. Reading one kernel version's output as another's  — 2026-08-26, FIXED

**Symptom.** E08X was fixed, committed, pushed and relaunched. The fetched log showed the
*same* `KeyError: 'run_id'` at the *same* line number as before the fix — while the file in
the pinned commit demonstrably contained the fix.

**Root cause.** `kaggle kernels output` serves the most recent **completed** version, which
is not the version just pushed. Version 2 finished while version 3 was still queued, so the
fetch returned version 2's failure. The pinned commit was `5489f48`; the kernel that
produced that output had checked out `8dc6bb5`.

**Why it kept happening.** This is §13 again in a different costume. Twice now a result has
been attributed to code that did not run: first a stale log carried across versions in
`/kaggle/working`, now a stale *version* served by the output endpoint. In both cases nothing
errored and the numbers looked like an answer.

**Fix.** `kaggle/fetch.py` now reads the `CODE COMMIT` line the notebook prints as its very
first output, compares it against the commit pinned in the local notebook, and says plainly
when they disagree:

    !! the kernel ran commit 8dc6bb5673, but 5489f48198 was pinned
       — this output is from an EARLIER version; do not read it as the new one

**The general rule.** *Every artifact must carry the identity of the code that produced it,
and something must check it mechanically.* Printing the SHA was never the point; comparing it
is. An unchecked provenance line is decoration.

**Related pattern, three occurrences now.** Every time something was located by *guessing
from a path* it was located wrongly: `find_dir(roots, "train")` reaching into IDRiD (§—),
the one-level `/kaggle/input` scan for APTOS (§15), and a `results.json` matched by run name
appearing in its directory (§15). The fix each time was the same: take the thing from a
source intrinsically bound to it — the checkpoint carries its own config, the loader
resolves from its own label file — rather than from whatever happens to sit nearby.

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
