| Run | Commit | Hypothesis | Backbone / head | n | DR acc | DR QWK | DME acc | DME QWK | vs floor |
|---|---|---|---|---|---|---|---|---|---|
| `E05` | `3bf6f51` | ordinal-heads-plus-messidor2-partial-labels | densenet121 / ordinal @448px | 2260 | 70.4% | 0.783 | 84.1% | 0.874 | DR yes, DME yes |
| `E05`+cuts | `3bf6f51` | cross-fitted decision cut-points | tuned on other folds only | 2260 | 68.0% | 0.831 | 84.1% | 0.884 | QWK +0.047 significant |
| `E06` | `52d5ef4` | eyepacs-pretraining-then-finetune-on-dev-pool | densenet121 / ordinal @448px | 2260 | 71.9% | 0.847 | 86.0% | 0.879 | DR yes, DME yes |
| `E06`+cuts | `52d5ef4` | cross-fitted decision cut-points | tuned on other folds only | 2260 | 71.7% | 0.855 | 85.9% | 0.876 | QWK +0.008 n.s. |
| `E07` | `c79dac6` | efficientnet-b3-and-longer-schedule-beat-densenet121-at-30-epochs | tf_efficientnet_b3 / ordinal @448px | 1811 | 74.8% | 0.835 | 88.2% | 0.914 | — |
| `E08` | `98b5ece` | eyepacs-pretrain-plus-longer-schedule-and-first-external-validation | densenet121 / ordinal @448px | 2260 | 74.2% | 0.860 | 84.1% | 0.884 | DR yes, DME yes |
| `E09` | `cebfc20` | halving-resolution-tests-whether-mild-recall-is-resolution-bound | densenet121 / ordinal @224px | 2260 | 69.0% | 0.820 | 84.3% | 0.866 | DR yes, DME yes |
| `E10` | `4413fc7` | native-resolution-messidor-plus-640px-lifts-the-mild-class | densenet121 / ordinal @640px | 2260 | 74.4% | 0.868 | 87.2% | 0.899 | DR yes, DME yes |
| `E11` | `ebe8a61` | efficientnet-b3-plus-eyepacs-pretraining-beats-densenet121-plus-the-same | tf_efficientnet_b3 / ordinal @448px | 1362 | 78.1% | 0.894 | 87.6% | 0.902 | DR yes, DME yes |
| `E12def` | `23d765e` | source-control-default-only | densenet121 / ordinal @448px | 953 | 72.2% | 0.858 | 87.3% | 0.891 | DR yes, DME yes |
| `E12nat` | `23d765e` | source-control-native-only | densenet121 / ordinal @448px | 953 | 69.6% | 0.856 | 87.3% | 0.901 | DR yes, DME yes |
