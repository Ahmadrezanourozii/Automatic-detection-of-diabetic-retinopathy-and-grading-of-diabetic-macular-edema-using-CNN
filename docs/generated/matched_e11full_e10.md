# E11FULL vs E10 — shipped cut-points and matched cut-points

`PROTOCOL.md` §4.1: a difference measured at two arbitrary thresholds is not evidence about representations. **matched** rows recalibrate both runs the same way — cut-points cross-fitted on the other folds under one objective — and repeat the paired bootstrap over groups.

| head | cut-points | metric | E11FULL | E10 | A − B | 95 % interval | verdict |
|---|---|---|---|---|---|---|---|
| dr | default | QWK | 0.8883 | 0.8683 | +0.0202 | [+0.0071, +0.0329] | **significant** |
| dr | default | accuracy | 77.70 % | 74.38 % | 3.34 pts | [1.42 pts, 5.27 pts] | **significant** |
| dr | matched | QWK | 0.8954 | 0.8749 | +0.0207 | [+0.0105, +0.0320] | **significant** |
| dr | matched | accuracy | 77.83 % | 74.78 % | 3.07 pts | [1.33 pts, 4.82 pts] | **significant** |
| dme_ungated | default | QWK | 0.8942 | 0.8991 | -0.0051 | [-0.0299, +0.0186] | indistinguishable |
| dme_ungated | default | accuracy | 87.60 % | 87.21 % | 0.38 pts | [-2.52 pts, 3.29 pts] | indistinguishable |
| dme_ungated | matched | QWK | 0.8882 | 0.8948 | -0.0067 | [-0.0313, +0.0172] | indistinguishable |
| dme_ungated | matched | accuracy | 86.05 % | 86.63 % | -0.55 pts | [-3.30 pts, 2.33 pts] | indistinguishable |
