"""The deterministic art gate: does this image meet a written style spec?

**Provenance.** These modules were vendored from ``python_pkg/artgate`` in the
``testsAndMisc`` repository at commit ``95a1f62`` (2026-08-15), together with
their tests. Both repositories are owned by the same author, so the copy is
relicensed here under this project's MIT licence; ``testsAndMisc`` carries no
licence file of its own.

Vendoring was chosen over a subprocess call into the sibling repository so
that an installed wheel actually works: ``gate_icon`` and ``describe_icon``
need no model and no GPU, and a reviewer who runs ``pip install artgate-gen``
should get a server whose gate tools function. The cost is that this copy can
drift from the original. There is deliberately no sync mechanism -- those rot
silently -- so the vendored tests are the behavioural contract instead.

The gate never judges whether art *looks good*. It measures canvas size,
palette count, alpha binarity, scale invariance, silhouette contrast and
margin, and reports codes. Aesthetic judgement stays with the human.
"""
