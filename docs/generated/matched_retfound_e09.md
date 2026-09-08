# I24FT01 vs E09 — shipped cut-points and matched cut-points

> ## ⚠️ THESE TWO RUNS DID NOT READ THE SAME FILES
>
> This comparison was run with `--acknowledge-consumption-diff`. Any difference below confounds the intended change with a change of input data (`ISSUES.md` §26, `PROTOCOL.md` §9). Details:
>
> ```
> refusing to combine or compare these runs — consumption differs or is unverifiable (PROTOCOL.md §9, ISSUES.md §26):
>   E09 vs I24FT01: mounted datasets differ: only in E09: ['tanlikesmath/diabetic-retinopathy-resized']; only in I24FT01: ['ah22reza/retfound-cfp-encoder']
> ```


`PROTOCOL.md` §4.1: a difference measured at two arbitrary thresholds is not evidence about representations. **matched** rows recalibrate both runs the same way — cut-points cross-fitted on the other folds under one objective — and repeat the paired bootstrap over groups.

| head | cut-points | metric | I24FT01 | E09 | A − B | 95 % interval | verdict |
|---|---|---|---|---|---|---|---|
| dr | default | QWK | 0.7281 | 0.8226 | -0.0943 | [-0.1284, -0.0611] | **significant** |
| dr | default | accuracy | 54.18 % | 69.23 % | -15.05 pts | [-19.23 pts, -11.21 pts] | **significant** |
| dr | matched | QWK | 0.7896 | 0.8298 | -0.0399 | [-0.0669, -0.0124] | **significant** |
| dr | matched | accuracy | 60.22 % | 63.85 % | -3.62 pts | [-7.25 pts, 0.00 pts] | indistinguishable |
| dme_ungated | default | QWK | 0.8208 | 0.8862 | -0.0667 | [-0.1423, +0.0052] | indistinguishable |
| dme_ungated | default | accuracy | 78.10 % | 84.29 % | -6.19 pts | [-12.87 pts, 0.48 pts] | indistinguishable |
| dme_ungated | matched | QWK | 0.7551 | 0.8628 | -0.1091 | [-0.2052, -0.0185] | **significant** |
| dme_ungated | matched | accuracy | 78.10 % | 81.90 % | -3.89 pts | [-10.49 pts, 2.86 pts] | indistinguishable |
