# Account 2 — shared RETFound dataset, verified IDENTICAL

Account 2 (`reza12123`) was granted "can view" on account 1's **private** RETFound dataset.
The kernel metadata keeps the original path `ah22reza/retfound-cfp-encoder` unchanged, as
instructed. **The dataset must never be made public** — the weights are gated and CC-BY-NC.

## Result

| | |
|---|---|
| mounted at | `/kaggle/input/datasets/ah22reza/retfound-cfp-encoder/retfound_cfp_encoder.pth` |
| bytes | **1 213 299 887** — matches account 1 |
| sha256 | **`847f9dd0e33bf8d450cc6121295d2919fc4bba3c185757a17ce6427bfa14ed37`** — matches |
| embedded source sha256 | `e1e4f66a1b792eeb6e2efaf158f33be35c8255f36b3d17ed67cd5129da246485` — matches the gated original |
| tensors | 294 |
| **verdict** | **IDENTICAL** |

The sharing did not alter the bytes, and the stripped encoder still carries its own origin
hash, so a result produced on account 2 traces to the same artefact as one produced on
account 1 (`PROTOCOL.md` §9).

## Two things this run confirmed incidentally

**`ISSUES.md` §15 recurred.** `/kaggle/input` contained exactly one directory — `datasets` —
so every attached dataset was flattened underneath it and the path
`/kaggle/input/retfound-cfp-encoder` did not exist. The recursive, filename-first search was
necessary rather than defensive.

**Account 2's kernels now have internet.** An earlier attempt failed with
`Could not resolve host: github.com`, the standard restriction on accounts without phone
verification; the owner verified the account and the standard git-clone path works. The
`--code-dataset` fallback built during that outage is retained for any future account whose
kernels cannot reach GitHub.
