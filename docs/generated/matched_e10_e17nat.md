# E10 vs E17NAT — shipped cut-points and matched cut-points

`PROTOCOL.md` §4.1: a difference measured at two arbitrary thresholds is not evidence about representations. **matched** rows recalibrate both runs the same way — cut-points cross-fitted on the other folds under one objective — and repeat the paired bootstrap over groups.

| head | cut-points | metric | E10 | E17NAT | A − B | 95 % interval | verdict |
|---|---|---|---|---|---|---|---|
| dr | default | QWK | 0.8683 | 0.8691 | -0.0010 | [-0.0140, +0.0121] | indistinguishable |
| dr | default | accuracy | 74.38 % | 74.82 % | -0.45 pts | [-2.17 pts, 1.33 pts] | indistinguishable |
| dr | matched | QWK | 0.8749 | 0.8662 | +0.0086 | [-0.0025, +0.0199] | indistinguishable |
| dr | matched | accuracy | 74.78 % | 72.35 % | 2.45 pts | [0.58 pts, 4.29 pts] | **significant** |
| dme_ungated | default | QWK | 0.8991 | 0.8734 | +0.0256 | [+0.0007, +0.0508] | **significant** |
| dme_ungated | default | accuracy | 87.21 % | 84.88 % | 2.33 pts | [-0.00 pts, 4.84 pts] | indistinguishable |
| dme_ungated | matched | QWK | 0.8948 | 0.8748 | +0.0195 | [-0.0052, +0.0452] | indistinguishable |
| dme_ungated | matched | accuracy | 86.63 % | 84.30 % | 2.31 pts | [-0.39 pts, 5.23 pts] | indistinguishable |
