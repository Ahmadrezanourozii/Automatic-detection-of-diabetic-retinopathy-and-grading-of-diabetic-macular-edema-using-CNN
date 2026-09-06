# E20CORAL vs E08 — shipped cut-points and matched cut-points

`PROTOCOL.md` §4.1: a difference measured at two arbitrary thresholds is not evidence about representations. **matched** rows recalibrate both runs the same way — cut-points cross-fitted on the other folds under one objective — and repeat the paired bootstrap over groups.

| head | cut-points | metric | E20CORAL | E08 | A − B | 95 % interval | verdict |
|---|---|---|---|---|---|---|---|
| dr | default | QWK | 0.8562 | 0.8599 | -0.0038 | [-0.0156, +0.0086] | indistinguishable |
| dr | default | accuracy | 69.42 % | 74.20 % | -4.80 pts | [-6.90 pts, -2.79 pts] | **significant** |
| dr | matched | QWK | 0.8569 | 0.8646 | -0.0076 | [-0.0184, +0.0032] | indistinguishable |
| dr | matched | accuracy | 70.27 % | 72.12 % | -1.87 pts | [-3.76 pts, -0.09 pts] | **significant** |
| dme_ungated | default | QWK | 0.8701 | 0.8845 | -0.0149 | [-0.0408, +0.0106] | indistinguishable |
| dme_ungated | default | accuracy | 85.47 % | 84.11 % | 1.30 pts | [-1.16 pts, 3.88 pts] | indistinguishable |
| dme_ungated | matched | QWK | 0.8611 | 0.8764 | -0.0158 | [-0.0408, +0.0094] | indistinguishable |
| dme_ungated | matched | accuracy | 82.95 % | 83.33 % | -0.44 pts | [-3.29 pts, 2.33 pts] | indistinguishable |
