# E17NAT — archived by hand, not by fetch.py

`kaggle kernels output` finished downloading every artefact at 19:40 and then **hung** for
25+ minutes paginating the cloned repo directory (hundreds of small files, one request each).
`fetch.py` therefore never reached its archiving step.

The files here were copied from the completed download at
`/var/folders/.../tmpw32e7tbp/E17NAT/`. **The provenance check fetch.py would have run was
performed manually and passed**, using a stronger check than fetch.py's log grep:

    results.json "commit" : 2b643eb948ab21df4de277103b7854e27bbe5632
    commit pinned in the kernel : 2b643eb948...
    MATCH

`results.json`'s commit field is written by `train.py` from the SHA actually checked out
inside the kernel, so it establishes what code ran — the exact property ISSUES.md §24 showed
a log grep alone does not guarantee.

`best_*.pt` and `pretrained.pt` (~150 MB) were left in the temp download and not archived
here; `.gitignore` excludes weights from the repo in any case.
