# data/DATASETS.md

Everything verified on disk 2026-08-25 against the Google Drive folder
`My Drive/Alireza/Datasets`. Counts are from the label CSVs, dimensions from image headers.
No images are committed to this repository; only the download and manifest scripts are.

---

## In hand

### IDRiD — Indian Diabetic Retinopathy Image Dataset
- **Role:** primary. The only corpus with **3-class DME** labels.
- **Counts:** 413 train + 103 test = **516** images. Disease Grading subset.
- **Labels:** `Retinopathy grade` 0–4 (ICDR); `Risk of macular edema` 0–2.
- **Resolution:** 4288 × 2848, 50° FOV. Full resolution, not downsampled.
- **Also present, and under-used:**
  - **Fovea centre (x, y) for all 516 images** — `C. Localization/2. Groundtruths/2. Fovea Center Location/`
  - **Optic disc centre (x, y) for all 516 images** — same folder, subdir 1.
  - **Pixel-level lesion masks for 81 images** (54 train + 27 test):
    microaneurysms, haemorrhages, **hard exudates**, soft exudates, optic disc.
- **Why the fovea coordinates matter:** the DME grade *is defined* as the distance from the
  nearest hard exudate to the macula centre, in disc diameters. We have that centre for
  every image. A macula-centred crop for the DME branch is therefore not a heuristic — it
  is the label definition made explicit. See `IDEAS.md` I07.
- **Licence:** CC BY 4.0 (IEEE DataPort). Verify before publication.
- **Class distribution:**

  | | DR0 | DR1 | DR2 | DR3 | DR4 |
  |---|---|---|---|---|---|
  | train | 134 | 20 | 136 | 74 | 49 |
  | test | 34 | 5 | 32 | 19 | 13 |

  | | DME0 | DME1 | DME2 |
  |---|---|---|---|
  | train | 177 | 41 | 195 |
  | test | 45 | 10 | 48 |

- **Joint DR × DME (all 516)** — this table settles the gating question:

  | DR \ DME | 0 | 1 | 2 |
  |---|---|---|---|
  | 0 | 168 | 0 | 0 |
  | 1 | 25 | 0 | 0 |
  | 2 | 16 | 36 | 116 |
  | 3 | 6 | 7 | 80 |
  | 4 | 7 | 8 | 47 |

  **Zero** DR=0 images carry DME > 0, so the DR≥1 gate discards no positive cases. It only
  changes the denominator, and with it the majority-class floor: 47.1 % ungated (n=516)
  versus 69.8 % gated (n=348). See `PROTOCOL.md` §5.1.

### Messidor-2
- **Role:** large DR corpus; **binary referable-DME** labels.
- **Counts:** 1 744 gradable images (all rows have `adjudicated_gradable = 1`).
- **Labels:** `diagnosis` = adjudicated 5-point ICDR grade 0–4;
  `adjudicated_dme` = **referable DME, binary**, defined as *hard exudates within 1 disc
  diameter of the macula centre*. Adjudicated per Krause et al., *Ophthalmology* 2018 —
  the highest-quality DR labels publicly available.
- **Resolution:** **already downsampled to 512 × 512.** See `ISSUES.md` §4.
- **Filename anomaly:** 1 057 files use the Messidor-2 convention, **687 use the
  Messidor-1 convention** (`IM######.JPG`). See `ISSUES.md` §3.
- **Eye pairing: NOT recoverable from our copy.** The upstream release pairs two images per
  examination (one per eye); our CSV has one row per image and no examination column. This
  blocks a patient-wise split on Messidor-2. See `PROTOCOL.md` §1.
- **Source:** obtained via a Kaggle mirror; original access is restricted.
- **Class distribution:** DR {0: 1017, 1: 270, 2: 347, 3: 75, 4: 35};
  referable DME {0: 1593, 1: 151}.

### APTOS 2019 Blindness Detection
- **Role:** DR only. Largest clean corpus in hand.
- **Counts:** 2 930 / 366 / 366 in `train_1.csv` / `valid.csv` / `test.csv` = **3 662**
  unique ids. **Verified: zero id overlap between the three files.**
- **Labels:** `diagnosis` 0–4.
- **Resolution:** native and highly variable — 3216×2136 down to 819×614. Aspect ratios and
  black-border geometry differ across images; this is a nuisance variable to watch.
- **Class distribution (all 3 662):** {0: 1805, 1: 370, 2: 999, 3: 193, 4: 295}.

### EyePACS / Kaggle DR 2015
- **Status:** `Datasets/diabetic-retinopathy-detection.zip`, **88.3 GB**, present in Google
  Drive as a stub. This is the "100 GB" download.
- **Do not download it locally.** It is a Kaggle competition dataset and can be attached
  directly to a Kaggle notebook, which is where training happens anyway. Downloading 88 GB
  through the Drive mount would take days and buy nothing.
- **Role:** DR pretraining. ~35 k train images, 5-class, with left/right eye pairs
  explicitly named — so it is the one corpus where patient grouping is trivial.

### DRIVE
- 102 files, vessel-segmentation corpus. **Not relevant** to DR/DME grading. Listed only so
  a future session does not spend time rediscovering that.

---

### Messidor-1 (ADCIS) — acquired 2026-08-29, and it does **not** work as an external test set

1 200 images, TIFF at 2240×1488, twelve bases from three sites, with a "risk of macular
edema" label graded by the same criterion IDRiD uses. It was acquired specifically to be the
external test set for the 3-class DME task, and the measurement says it cannot be.

**88 % of it is already in our development pool.** Messidor-1 and our Messidor-2 mirror share
a filename convention, so the overlap is exact rather than estimated — and filename identity
was confirmed to mean image identity by content: two sampled pairs gave 256-bit dHash
distances of 5 and 6 and NCC 1.0000, well inside the ≤8 true-duplicate band calibrated in
`ISSUES.md` §7.

| | images |
|---|---|
| Messidor-1 total | 1 200 |
| already in the development pool | **1 057** |
| genuinely held out | **143** |

**And the 143 survivors are not representative.** The overlap is wildly uneven by site, and
site strongly predicts class:

| site | dilation | total | overlap | survives |
|---|---|---|---|---|
| Lariboisière | dilated | 400 | 372 | 28 |
| St Etienne | dilated | 400 | 397 | **3** |
| Brest | undilated | 400 | 288 | **112** |

78 % of what survives is Brest — the site with the *lowest* rate of referable DME (5 %,
against Lariboisière's 21 %). A χ² test of DME grade against site over the full corpus gives
**p = 1.5 × 10⁻¹⁴**, so "which site" is a shortcut feature here exactly as "which dataset" is
in the pooled corpus. An external number computed on a set skewed toward the easiest site
would be misleading even if it were precise.

**Recorded as a per-image covariate** (`site`, `dilation`) rather than only as provenance, so
this can be tested rather than assumed in any future use.

**The unexpected value.** Those 1 057 overlapping images are the **native-resolution originals
of images we currently hold only at 512 × 512**. Messidor-1 is therefore the full-resolution
mirror the resolution question needs — better sourced than the Kaggle copy below.

### A mislabelled public mirror, and why it matters to someone else

`borhan2003/messidor-diabetic-retinopathy-dataset-jpg-format` is published on Kaggle as a
Messidor-2 dataset. It holds **1 200** images at 2240 × 1488 that match **1 057** of our
Messidor-2 labels — the same 1 200 and the same 1 057 as Messidor-1. **It is Messidor-1,
mislabelled as Messidor-2.**

This is worth stating because of what it does to anyone who uses both: attach that mirror as
"Messidor-2" alongside a genuine Messidor-2 copy and you have silently duplicated 1 057
images across what you believe are two independent corpora. If they land on opposite sides of
a split, that is direct leakage, and nothing about it looks wrong. We found it only because
the filename conventions were checked against the label CSV rather than trusted.

## To acquire (verify availability and licence before relying on any of these)

| Corpus | Why | Priority |
|---|---|---|
| ~~Messidor-1~~ | **Acquired, and ruled out as an external DME test set** — see above. 88 % overlaps the development pool; the 143 survivors are 78 % one site and too few to resolve anything. Retained as a native-resolution image source. | done |
| **A DME corpus with no Messidor lineage** | Still the open gap. Any external 3-class DME number needs a corpus graded by the IDRiD criterion that was not built from Messidor examinations. None identified yet; until one is, "no external DME validation" is a declared limitation of the thesis, with a measured reason. | **highest** |
| DDR (13 673) | External DR test; includes an "ungradable" class. | high |
| DeepDRiD (2 000) | External DR test; dual-field per eye, with explicit patient IDs. | medium |
| FGADR / e-ophtha / DIARETDB1 | Hard-exudate masks — auxiliary segmentation head (`IDEAS.md` I15). | medium |
| OCT collections | Only if the multimodal extension is attempted. Out of scope for now. | low |

## Recorded for each corpus when added
source URL · licence · download date · image count · class distribution · pHash duplicate
clusters against every corpus already in the pool.
