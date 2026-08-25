| Run | Commit | Hypothesis | Backbone / head | n | DR acc | DR QWK | DME acc | DME QWK | vs floor |
|---|---|---|---|---|---|---|---|---|---|
| `E05` | `3bf6f51` | ordinal-heads-plus-messidor2-partial-labels | densenet121 / ordinal @448px | 2260 | 70.4% | 0.783 | 84.1% | 0.874 | DR yes, DME yes |
| `E05`+cuts | `3bf6f51` | cross-fitted decision cut-points | tuned on other folds only | 2260 | 68.0% | 0.831 | 84.1% | 0.884 | QWK +0.047 significant |
