## Commands

- run: `./run.sh out "potion=red potion flask"` (GPU; `./run.sh --help` for options)
- test: `venv/bin/python -m pytest -q`
- test-changed: `scripts/test_changed.sh`
- lint: `venv/bin/ruff check .`
- coverage: `venv/bin/python -m pytest -q --cov-report=xml`
- coverage-gaps: `coverage-gaps coverage.xml`
