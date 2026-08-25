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

## To acquire (verify availability and licence before relying on any of these)

| Corpus | Why | Priority |
|---|---|---|
| **Messidor-1** (1 200) | Carries "risk of macular edema" 0–2 with **the same clinical definition as IDRiD**. The only realistic external validation set for the 3-class DME contribution, and it would more than triple the 3-class DME training data if used for development instead. Check overlap with our Messidor-2 mirror first (`ISSUES.md` §3). | **highest** |
| DDR (13 673) | External DR test; includes an "ungradable" class. | high |
| DeepDRiD (2 000) | External DR test; dual-field per eye, with explicit patient IDs. | medium |
| FGADR / e-ophtha / DIARETDB1 | Hard-exudate masks — auxiliary segmentation head (`IDEAS.md` I15). | medium |
| OCT collections | Only if the multimodal extension is attempted. Out of scope for now. | low |

## Recorded for each corpus when added
source URL · licence · download date · image count · class distribution · pHash duplicate
clusters against every corpus already in the pool.
