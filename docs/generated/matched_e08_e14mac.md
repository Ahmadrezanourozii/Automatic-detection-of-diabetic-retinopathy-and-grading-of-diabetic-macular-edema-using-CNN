# E08 vs E14MAC — shipped cut-points and matched cut-points

`PROTOCOL.md` §4.1: a difference measured at two arbitrary thresholds is not evidence about representations. **matched** rows recalibrate both runs the same way — cut-points cross-fitted on the other folds under one objective — and repeat the paired bootstrap over groups.

| head | cut-points | metric | E08 | E14MAC | A − B | 95 % interval | verdict |
|---|---|---|---|---|---|---|---|
| dr | default | QWK | 0.8599 | 0.8507 | +0.0090 | [-0.0050, +0.0227] | indistinguishable |
| dr | default | accuracy | 74.20 % | 70.97 % | 3.23 pts | [1.42 pts, 5.18 pts] | **significant** |
| dr | matched | QWK | 0.8646 | 0.8595 | +0.0049 | [-0.0078, +0.0171] | indistinguishable |
| dr | matched | accuracy | 72.12 % | 73.27 % | -1.16 pts | [-2.92 pts, 0.62 pts] | indistinguishable |
| dme_ungated | default | QWK | 0.8845 | 0.8693 | +0.0163 | [-0.0132, +0.0456] | indistinguishable |
| dme_ungated | default | accuracy | 84.11 % | 81.98 % | 2.23 pts | [-1.16 pts, 5.62 pts] | indistinguishable |
| dme_ungated | matched | QWK | 0.8764 | 0.8538 | +0.0237 | [-0.0094, +0.0566] | indistinguishable |
| dme_ungated | matched | accuracy | 83.33 % | 81.20 % | 2.26 pts | [-1.36 pts, 5.81 pts] | indistinguishable |
