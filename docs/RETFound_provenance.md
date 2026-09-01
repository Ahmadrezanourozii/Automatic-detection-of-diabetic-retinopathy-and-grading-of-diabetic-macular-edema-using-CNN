# RETFound checkpoint — provenance record

Verified 2026-09-01, before any experiment was planned around it. Recorded because the
HuggingFace **model card on the `natureCFP` repo is titled "Model Card for RETFound_MAE_MEH"**,
which is the class of label mismatch this project has been caught by before.

## What was obtained

| | |
|---|---|
| repo | `YukunZhou/RETFound_mae_natureCFP` (HuggingFace, gated `auto`) |
| file | `RETFound_mae_natureCFP.pth` |
| size | **3 952 489 221 bytes** |
| sha256 | `e1e4f66a1b792eeb6e2efaf158f33be35c8255f36b3d17ed67cd5129da246485` |
| licence | **CC-BY-NC 4.0**, verified in the repo's `LICENSE` (Attribution-NonCommercial) |

The downloaded file's sha256 **matches the `x-linked-etag` HuggingFace served for it**, so the
bytes on disk are the bytes the registry intended to serve.

## What the tensors say

Read from the checkpoint's own `args`, not from any label:

```
model       : mae_vit_large_patch16
input_size  : 224
data_path   : /home/yzhou_mehresearch_org/cfp_image/cfp/cfp_256/
mask_ratio  : 0.85     epochs 801, finished at epoch 800
resumed from: ./mae_pretrain_vit_large_full.pth   (ImageNet MAE ViT-L)
```

Structure: `pos_embed` (1, 197, 1024) → 196 patches + CLS at 224/16; 24 transformer blocks;
`patch_embed.proj.weight` (1024, 3, 16, 16). **329.5 M parameters total, 303.3 M in the
encoder** — ViT-Large/16 as documented.

## Verdict on the card mismatch — stated precisely

**Confirmed:** the artefact is a **ViT-Large/16 MAE trained on colour fundus photographs
(`cfp`) at 224 px for 800 epochs**, initialised from ImageNet MAE. The **OCT variant is
definitively excluded** — this is CFP data, and the OCT release is a different checkpoint.

**Not confirmable from contents alone:** whether this is specifically the *natureCFP* release
rather than the *meh* release. Both are CFP ViT-L models, and the other three variants remain
gated on this account, so no hash comparison against `meh` is possible. Note the training host
is `yzhou_mehresearch_org`, which plausibly explains the card's "MEH" title as a copy-paste
without implying the weights are the MEH release.

**What is consistent with natureCFP:** the served filename is `RETFound_mae_natureCFP.pth`;
the gate accepted was for that repo; the MAE configuration (mask ratio 0.85, 800 epochs,
ViT-L/16 @224 from an ImageNet MAE init) matches the Nature paper's described pretraining.

**How this is handled downstream:** the sha256 above is **pinned in the launch code**, and the
kernel verifies the file it loads rather than trusting a path (`PROTOCOL.md` §9 — record what
was consumed, not what was configured). Any result from this backbone is reported as "a
RETFound CFP ViT-L checkpoint with sha256 `e1e4f66a…`", which is exactly true and does not
overclaim which release it is.

## Derived artefact actually used

The full file carries optimiser and scaler state (hence 3.95 GB). Downstream use needs only
the encoder, so a stripped copy was made and uploaded as a **private** Kaggle dataset —
private because redistributing CC-BY-NC gated weights publicly is not ours to do.

| | |
|---|---|
| file | `retfound_cfp_encoder.pth` |
| size | 1 213 299 887 bytes |
| sha256 | `847f9dd0e33bf8d450cc6121295d2919fc4bba3c185757a17ce6427bfa14ed37` |
| kept | 294 encoder tensors |
| dropped | all `decoder_*`, `mask_token`, and the optimiser/scaler state |

The stripped file embeds `source_sha256`, `source_bytes`, and the full `pretrain_args`, so the
derived artefact carries its own origin.
