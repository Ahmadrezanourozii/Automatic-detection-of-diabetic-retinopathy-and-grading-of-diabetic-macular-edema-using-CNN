# T1 — transfer gap: a threshold fitted on the development pool, measured on APTOS

Model: `E08` weights. Dev pool n=2260 (referable prevalence 34.5 %); APTOS n=3662 (prevalence 40.6 %). Thresholds are fitted on the development pool only — never on APTOS (`PROTOCOL.md` §3, §6.1).

| target | source | threshold | dev sens (cross-fitted) | dev sens (shipped fit) | dev spec | **APTOS sens** | 95 % CI | APTOS spec | **transfer gap** |
|---|---|---|---|---|---|---|---|---|---|
| **80.0 %** | A | 0.8298 | 80.24 % | 79.87 % | 97.97 % | **97.58 %** | [96.74, 98.38] | 86.53 % | **+17.58 pts** |
| **85.0 %** | B | 0.5927 | 84.97 % | 84.87 % | 96.35 % | **99.26 %** | [98.77, 99.66] | 84.92 % | **+14.26 pts** |
| **87.2 %** | B* | 0.5104 | 86.90 % | 87.18 % | 95.20 % | **99.39 %** | [98.98, 99.73] | 84.60 % | **+12.19 pts** |
| **95.5 %** | C | 0.1109 | 95.24 % | 95.38 % | 81.89 % | **99.87 %** | [99.66, 100.00] | 79.36 % | **+4.37 pts** |
