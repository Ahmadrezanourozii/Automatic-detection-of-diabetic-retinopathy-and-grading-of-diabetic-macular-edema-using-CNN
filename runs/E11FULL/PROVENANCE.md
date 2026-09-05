# E11FULL — assembled, not run

Folds 0–2 from **E11** (commit `ebe8a61`), folds 3–4 from **E19E11C** (commit `568ee7e4`).

**Checks performed before assembling (`ISSUES.md` §26, `PROTOCOL.md` §9):**

* **Consumption**: `manifest.require_same` PASSES. Both kernels mounted the identical five
  datasets, including `borhan2003/messidor-...-jpg-format`, the native-resolution mirror.
  This is the check that refused the earlier attempt — E19E11B omitted `--messidor-hi` and
  would have combined folds trained on different pixels.
* **Config**: every training field identical (backbone, epochs, size, batch, head, hidden,
  dropout, α/β, learning rates, EMA, pretraining corpus and epochs, TTA, smoothing, weight
  decay, seed). Difference set is empty.
* **Split**: fingerprint `0cfbbfeb081999af` on both.

**What differs and cannot be made identical:** the two runs pin different commits, because
folds 3–4 were run nine days later. The intervening changes to `src/` add the macula branch,
LP-FT staging and the RETFound backbone — all inactive without their flags — plus analysis
scripts that training does not import. No change touches the code path a
`--backbone tf_efficientnet_b3` run without `--macula` or `--lpft` executes. **That is an
argument, not a proof**, and it is the residual risk of assembling folds across commits.
