# E22CORN vs E08 — shipped cut-points and matched cut-points

`PROTOCOL.md` §4.1: a difference measured at two arbitrary thresholds is not evidence about representations. **matched** rows recalibrate both runs the same way — cut-points cross-fitted on the other folds under one objective — and repeat the paired bootstrap over groups.

| head | cut-points | metric | E22CORN | E08 | A − B | 95 % interval | verdict |
|---|---|---|---|---|---|---|---|
| dr | default | QWK | 0.8463 | 0.8599 | -0.0139 | [-0.0304, +0.0019] | indistinguishable |
| dr | default | accuracy | 74.96 % | 74.20 % | 0.74 pts | [-1.02 pts, 2.48 pts] | indistinguishable |
| dr | matched | QWK | 0.8168 | 0.8646 | -0.0479 | [-0.0656, -0.0325] | **significant** |
| dr | matched | accuracy | 69.69 % | 72.12 % | -2.43 pts | [-4.42 pts, -0.44 pts] | **significant** |
| dme_ungated | default | QWK | 0.8492 | 0.8845 | -0.0358 | [-0.0709, -0.0018] | **significant** |
| dme_ungated | default | accuracy | 84.69 % | 84.11 % | 0.56 pts | [-2.13 pts, 3.30 pts] | indistinguishable |
| dme_ungated | matched | QWK | 0.8689 | 0.8764 | -0.0078 | [-0.0363, +0.0190] | indistinguishable |
| dme_ungated | matched | accuracy | 83.53 % | 83.33 % | 0.12 pts | [-2.91 pts, 3.29 pts] | indistinguishable |
