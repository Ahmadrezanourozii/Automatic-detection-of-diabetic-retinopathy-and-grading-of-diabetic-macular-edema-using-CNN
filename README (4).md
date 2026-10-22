# Automatic Detection of Diabetic Retinopathy and Grading of Diabetic Macular Edema with CNNs

**Ahmadreza Nourozi, Alireza Chegeni** — journal article in preparation.
The project started as Alireza Chegeni's M.Sc. thesis (Electrical and Computer Engineering, University of Tehran).

One shared CNN backbone with two ordinal heads grades a single retinal fundus photograph for
**diabetic retinopathy** (five ICDR grades) and **diabetic macular edema** (three grades).
The work is less about a new architecture than about what the reported numbers actually
establish: frozen splits, model selection on the development pool only, external validation on
data never seen in training, and comparisons made at matched operating points.

## Results

| | Data | DR quadratic weighted kappa |
|---|---|---|
| Development pool, 5-fold out-of-fold | 2,260 images (IDRiD + Messidor-2) | 0.8954 |
| **External, held out until the model was frozen** | **3,662 APTOS 2019 images** | **0.9026 [0.8941, 0.9108]** |

- The full pipeline is worth **+0.234 kappa** over frozen ImageNet features with a linear ordinal head on identical images, folds, decoding and calibration.
- With the same EyePACS pretraining, the selected EfficientNet-B3 backbone beats DenseNet-121 by +0.021 kappa [+0.011, +0.032] on retinopathy; on macular edema the two are indistinguishable.
- Of nine pre-registered interventions, eight were falsified. Ensembling raised the development score but gave +0.001 [−0.003, +0.005] externally, so it is not used.
- The macular-edema head meets a ceiling that comes from the data rather than the architecture: six independent interventions, including a macula-centred crop from a fovea localiser (median error 0.20 disc diameters), did not move it.
- Referable-DR sensitivity and specificity at the default `sigmoid > 0.5` cut-point swap places between corpora (86 % / 96 % on development, 99.5 % / 84 % on APTOS), so the default threshold is reported as a finding, not as a property of the model.

## Method

- **Data.** IDRiD (516 images, the only corpus with 3-class DME labels) and Messidor-2 (1,744 gradable images) form the development pool; EyePACS (~35k images) is used for pretraining; APTOS 2019 (3,662 images) is the external test set. No images are stored in this repository — see [`data/DATASETS.md`](data/DATASETS.md).
- **Splits.** Five-fold grouped cross-validation, frozen before the first real run and built from perceptual clustering so that near-duplicate images across corpora cannot leak between folds.
- **Model.** Shared backbone (`timm`) with two ordinal threshold heads; cut-points are cross-fitted, and no difference between two models is attributed to the representation unless both are evaluated at matched cut-points.
- **Training.** Kaggle GPU kernels, one per experiment (`kaggle/`), each run archived and reproducible from a commit.

## Repository

```
src/        training, evaluation, external verification, threshold tuning, fovea localiser
kaggle/     one notebook + kernel metadata per experiment (E05 … E19)
data/       dataset notes, label mapping, frozen split manifests (no images)
docs/       pre-registrations and generated comparison tables
results/    archived metrics and predictions per run
tests/      consistency checks for the reported numbers
thesis/     LaTeX sources of the M.Sc. thesis
```

Project notes: [`PROTOCOL.md`](PROTOCOL.md) (evaluation rules), [`EXPERIMENTS.md`](EXPERIMENTS.md),
[`FINDINGS.md`](FINDINGS.md), [`ISSUES.md`](ISSUES.md), [`STATE.md`](STATE.md).

## Reproducing the external result

```bash
pip install torch timm pandas scikit-learn
python src/verify_external.py --run runs/<run> --datasets <dataset root>   # recomputes the APTOS numbers and checks them end to end
```
