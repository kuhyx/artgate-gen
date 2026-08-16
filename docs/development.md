# Development

## Setup

```bash
git clone https://github.com/kuhyx/artgate-gen
cd artgate-gen
python -m venv venv
venv/bin/pip install -e ".[dev]"
pre-commit install && pre-commit install --hook-type pre-push
```

Generation additionally needs `.[generate]` and a CUDA GPU. Everything else —
the gate, the finisher, the CLI, the whole test suite — runs without either.

## The gate

```bash
ruff check . && ruff format --check .
mypy --strict artgate_gen tools tests generate.py mcp_server.py
pytest
python tools/check_line_cap.py
shellcheck run.sh
```

Under two seconds end to end. The same commands run in CI, so a clean local
gate means a green build.

## Three rules that shape the code

**No file over 250 lines.** Every tracked text file, no exemptions, enforced by
`tools/check_line_cap.py` in both pre-commit and CI. There is deliberately no
allow-list: an allow-list is how a hard cap decays into a soft one.

**No lint suppressions.** `ruff` runs with `select = ["ALL"]` and there are no
`# noqa` comments and no per-file ignores anywhere. The five entries in the
global `ignore` list are all rules that cannot be satisfied in source, and each
carries its reason. When a rule fires, fix the code — the exits used so far
were `contextlib.suppress` for blind excepts, a `console.emit` helper for
prints, and `importlib` for deliberately deferred imports.

**100% branch coverage.** `pytest` fails below it. Warnings are errors, which
has already earned its keep: it caught a file-descriptor leak in the repair
path and a Pillow 14 deprecation in `describe_icon`.

## Tests without a GPU

`tests/stubs/` holds real stub packages for torch, diffusers, transformers and
huggingface_hub, installed into `sys.modules` by `tests/conftest.py` before
anything imports them. They are ordinary source: linted, type-checked and
diffable.

numpy and Pillow are **not** stubbed. The finisher and the gate do genuine
array work, and faking that would make the coverage meaningless.

Tests use the `check_*` helpers from `tests/assertions.py` rather than bare
`assert`, because `S101` applies here too. Each one carries an explanation,
which is what you read when it fails.

## The vendored gate

`artgate_gen/gate/` is vendored from `python_pkg/artgate` in the author's
`testsAndMisc` repository, with its tests. It is a copy and can drift; there is
no sync mechanism, because sync mechanisms rot silently. The vendored tests are
the behavioural contract.

When changing anything under `gate/`, check a real icon's `--json` verdict
before and after. Passing the tests is weaker evidence than it looks, because
the tests came from the same source as the code.

## Releasing

Tag `vX.Y.Z` and push. CI builds an sdist and a wheel, installs the wheel into
a clean venv **without** the `generate` extra, verifies the style specs shipped
inside it and that the server imports, then attaches both to a GitHub Release.

That smoke test exists because package data is the usual casualty of a
restructure: the style TOMLs live inside the wheel, and nothing in the test
suite would notice if they stopped shipping.

<!-- touch to measure a warm-cache CI run -->
