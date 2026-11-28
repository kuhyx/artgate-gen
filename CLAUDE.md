## Commands

- run: `./run.sh out "potion=red potion flask"` (GPU; `./run.sh --help` for options)
- test: `venv/bin/python -m pytest -q`
- test-changed: `scripts/test_changed.sh`
- lint: n/a: dropped (kuhy, 2026-11-27)
- coverage: n/a: dropped (kuhy, 2026-11-27)
- coverage-gaps: n/a: dropped (kuhy, 2026-11-27)
- deploy: `git -C ~/src/artgate-gen pull --ff-only && test "$(git -C ~/src/artgate-gen rev-parse HEAD)" = "$(git -C ~/src/artgate-gen rev-parse origin/main)" && cd ~/src/artgate-gen && venv/bin/python -c 'import artgate_gen.server, artgate_gen.sd35, artgate_gen.sd_cli' && ./run.sh --help | grep -q sd35`
