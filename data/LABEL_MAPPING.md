# data/LABEL_MAPPING.md

How the label spaces of different corpora are harmonised. One row per mapping, with a
justification. Nothing is merged until its row exists here.

---

## DR — 5-class ICDR

| Corpus | Native label | Mapping | Justification |
|---|---|---|---|
| IDRiD | `Retinopathy grade` 0–4 | identity | Already ICDR 5-point. |
| Messidor-2 | `diagnosis` 0–4 | identity | Adjudicated 5-point ICDR (Krause 2018, stated in `messidor_readme.txt`). |
| APTOS 2019 | `diagnosis` 0–4 | identity | ICDR 5-point, single grader. **Caveat:** single-grader labels are noisier than IDRiD's or Messidor-2's adjudicated ones; this is a source of label noise, not a defect to correct. |
| EyePACS 2015 | `level` 0–4 | identity | ICDR 5-point, single grader, known to be noisy. Use for pretraining only, never as a test set. |
| Messidor-1 *(if acquired)* | `Retinopathy grade` **0–3** | **NOT identity — needs an explicit mapping** | Messidor-1 uses a 4-level scale defined by microaneurysm and haemorrhage counts, not ICDR. It does not map cleanly onto 5 classes. **Recommendation: do not use Messidor-1 for the DR head at all.** Use it for the DME head only, where its label *is* directly comparable. |

## DME

Two different label spaces exist and they are **not** the same quantity.

| Corpus | Native label | Space |
|---|---|---|
| IDRiD | `Risk of macular edema` | **3-class ordinal** by exudate distance to the macula centre |
| Messidor-1 | `Risk of macular edema` | **3-class ordinal**, same definition |
| Messidor-2 | `adjudicated_dme` | **binary**, "referable DME = hard exudates within 1 DD" |

**The 3-class definition (IDRiD, Messidor-1):**

| Grade | Definition | Name to use |
|---|---|---|
| 0 | no visible hard exudates | `No_DME` |
| 1 | hard exudates present, **> 1 disc diameter** from the macula centre | `Non_referable_DME` |
| 2 | hard exudates **within 1 disc diameter** of the macula centre | `Referable_DME` |

**Mapping Messidor-2's binary label onto the 3-class space:**

| Messidor-2 | 3-class equivalent | Justification |
|---|---|---|
| `adjudicated_dme = 1` | **grade 2** | Both are defined by the identical criterion — hard exudates within 1 disc diameter of the macula centre. This is an exact match, not an approximation. |
| `adjudicated_dme = 0` | **grade 0 or 1 — unknown which** | "No referable DME" means no exudates *within* 1 DD. It does not distinguish "no exudates at all" from "exudates further out". |

**Consequence, and the opportunity.** Messidor-2's 1 744 images cannot be given a hard
3-class label, but they are not unusable either: each one carries a **partial label** —
either exactly grade 2, or the set {0, 1}. Trained with a partial-label / marginal loss
(sum the softmax over the candidate set), Messidor-2 supervises the DME head at the coarse
level while IDRiD supervises the fine distinction. That takes the DME training set from
348 gated images to 2 092, and it is a defensible methodological contribution rather than a
data-cleaning fudge.

**Do not** flatten IDRiD to binary to match Messidor-2. That throws away the 3-class
contribution, which is the thesis' stated subject.

---

## Never do

- Merge two corpora without a row in this table.
- Pool normalisation statistics across a merge and then use them to normalise a test split
  drawn from one side of it (`PROTOCOL.md` §8).
- Treat "dataset of origin" as harmless. Messidor-2, IDRiD and APTOS differ in camera, FOV,
  resolution and class distribution, so origin is a shortcut feature. Test for it directly:
  train a classifier to predict the source dataset from the image, and report how easy it is.
