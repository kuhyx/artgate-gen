## Commands

- run: `./run.sh out "potion=red potion flask"` (GPU; `./run.sh --help` for options)
- test: `venv/bin/python -m pytest -q`
- test-changed: `scripts/test_changed.sh`
- lint: `venv/bin/ruff check .`
- coverage: `venv/bin/python -m pytest -q --cov-report=xml`
- coverage-gaps: `coverage-gaps coverage.xml`
- deploy: `git -C ~/src/artgate-gen pull --ff-only && test "$(git -C ~/src/artgate-gen rev-parse HEAD)" = "$(git -C ~/src/artgate-gen rev-parse origin/main)" && cd ~/src/artgate-gen && venv/bin/python -c 'import artgate_gen.server, artgate_gen.sd35, artgate_gen.sd_cli' && ./run.sh --help | grep -q sd35`
