# Contributing

## Before you start

```bash
python -m venv venv
venv/bin/pip install -e ".[dev]"
pre-commit install && pre-commit install --hook-type pre-push
```

[docs/development.md](docs/development.md) has the full picture. The short
version is three rules that are enforced, not suggested:

1. **No file over 250 lines** — any tracked text file, no exemptions.
2. **No lint suppressions** — no `# noqa`, no per-file ignores. Fix the code.
3. **100% branch coverage** — `pytest` fails below it, and warnings are errors.

The whole gate runs in under two seconds:

```bash
ruff check . && ruff format --check . \
  && mypy --strict artgate_gen tools tests generate.py mcp_server.py \
  && pytest && python tools/check_line_cap.py && shellcheck run.sh
```

## Changing the gate

`artgate_gen/gate/` is vendored code with its own tests. Passing those tests is
weaker evidence than it looks, because they came from the same source as the
code. Capture a real icon's `--json` verdict before your change and diff it
after.

## Pull requests

Keep the diff to one concern. Explain *why* in the commit message — the code
already says what. If you are adding a gate, say what real art you calibrated
its threshold against; a guessed threshold rejects good art and sends the next
person hunting for a fault in the artwork that is actually in the gate.
