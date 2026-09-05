# Frozen probe vs frozen probe — does RETFound carry more signal than ImageNet?

**RETFound CFP ViT-L/16** against **ImageNet DenseNet121**. Same 2 260 images, same cross-fitted linear ordinal head, same matched calibration, same folds — **only the frozen backbone differs.**

> The Stage 1 figure of DR QWK 0.7008 against E09's 0.8389 is **not** a verdict on RETFound: it compares a frozen probe against a fully fine-tuned network, so it measures probing versus fine-tuning. This table is the comparison that isolates the representation.

| head | n | RETFound CFP ViT-L/16 | ImageNet DenseNet121 | A − B | 95 % interval | verdict |
|---|---|---|---|---|---|---|
| DR, 5-class | 2260 | **0.7008** | **0.6611** | +0.0401 | [+0.0122, +0.0692] | **significant** |
| DME, 3-class ungated | 516 | **0.7074** | **0.6645** | +0.0434 | [-0.0173, +0.1018] | indistinguishable |

## What follows, by the decision rule fixed before the numbers

**RETFound beats ImageNet beyond the paired interval.** The representation is genuinely better, and Stage 2 (fine-tuning) is worth the quota — split across runs to stay under the 10 h cap.
