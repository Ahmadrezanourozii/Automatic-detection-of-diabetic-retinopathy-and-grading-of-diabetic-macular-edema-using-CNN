# E08 vs E15LPFT — shipped cut-points and matched cut-points

`PROTOCOL.md` §4.1: a difference measured at two arbitrary thresholds is not evidence about representations. **matched** rows recalibrate both runs the same way — cut-points cross-fitted on the other folds under one objective — and repeat the paired bootstrap over groups.

| head | cut-points | metric | E08 | E15LPFT | A − B | 95 % interval | verdict |
|---|---|---|---|---|---|---|---|
| dr | default | QWK | 0.8599 | 0.8466 | +0.0136 | [-0.0004, +0.0275] | indistinguishable |
| dr | default | accuracy | 74.20 % | 68.67 % | 5.57 pts | [3.54 pts, 7.52 pts] | **significant** |
| dr | matched | QWK | 0.8646 | 0.8655 | -0.0008 | [-0.0126, +0.0105] | indistinguishable |
| dr | matched | accuracy | 72.12 % | 74.07 % | -1.92 pts | [-3.67 pts, -0.13 pts] | **significant** |
| dme_ungated | default | QWK | 0.8845 | 0.8551 | +0.0303 | [-0.0017, +0.0618] | indistinguishable |
| dme_ungated | default | accuracy | 84.11 % | 83.72 % | 0.47 pts | [-2.71 pts, 3.49 pts] | indistinguishable |
| dme_ungated | matched | QWK | 0.8764 | 0.8577 | +0.0193 | [-0.0099, +0.0481] | indistinguishable |
| dme_ungated | matched | accuracy | 83.33 % | 83.91 % | -0.51 pts | [-3.49 pts, 2.71 pts] | indistinguishable |
