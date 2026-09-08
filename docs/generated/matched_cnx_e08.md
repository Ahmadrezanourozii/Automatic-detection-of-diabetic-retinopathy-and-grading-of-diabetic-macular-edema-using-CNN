# E21CNXA vs E08 — shipped cut-points and matched cut-points

`PROTOCOL.md` §4.1: a difference measured at two arbitrary thresholds is not evidence about representations. **matched** rows recalibrate both runs the same way — cut-points cross-fitted on the other folds under one objective — and repeat the paired bootstrap over groups.

| head | cut-points | metric | E21CNXA | E08 | A − B | 95 % interval | verdict |
|---|---|---|---|---|---|---|---|
| dr | default | QWK | 0.8323 | 0.8600 | -0.0275 | [-0.0514, -0.0038] | **significant** |
| dr | default | accuracy | 66.48 % | 74.18 % | -7.67 pts | [-10.88 pts, -4.50 pts] | **significant** |
| dr | matched | QWK | 0.8267 | 0.8628 | -0.0362 | [-0.0593, -0.0153] | **significant** |
| dr | matched | accuracy | 65.60 % | 70.11 % | -4.45 pts | [-7.47 pts, -1.32 pts] | **significant** |
| dme_ungated | default | QWK | 0.8728 | 0.9092 | -0.0368 | [-0.0804, +0.0020] | indistinguishable |
| dme_ungated | default | accuracy | 85.71 % | 86.67 % | -1.04 pts | [-5.24 pts, 3.33 pts] | indistinguishable |
| dme_ungated | matched | QWK | 0.8776 | 0.8633 | +0.0149 | [-0.0412, +0.0695] | indistinguishable |
| dme_ungated | matched | accuracy | 84.29 % | 79.05 % | 5.19 pts | [-0.48 pts, 10.95 pts] | indistinguishable |
