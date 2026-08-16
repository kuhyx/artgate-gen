# Usage

## Command line

```bash
artgate-gen OUT_DIR "name=prompt" ["name=prompt" ...] [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `--size` | `64` | Icon edge in pixels. 64 is recommended; see below. |
| `--colours` | spec cap | Palette cap. Defaults to 32, which is calibrated, not chosen. |
| `--candidates` | `1` | Renders per subject, each with its own seed. |
| `--seed` | `11` | Base seed; candidate *n* uses `seed + n`. |
| `--keep-raw` | off | Also save the 1024px render. |
| `--profile` | off | Report where the time went, plus peak GPU/RAM/load. |
| `--profile-json` | — | Write the full timing trace to a file. |

The hit rate is not 1.0. When quality matters ask for `--candidates 3` and
pick; that is cheaper than three separate runs, because the model loads once.

`--keep-raw` is worth more than it sounds: the 1024px render is often better
than any downscale of it, and it is the input you want if you later decide to
finish at a different size.

### Why 32 colours, at every size

Both shipped specs cap the palette at 32, and that number comes from measuring
real art rather than from taste. 250 genuine 32×32 icons top out at 30 colours;
300 genuine 48×48 icons top out at 14. Hand-made art does not spend more
colours just because the canvas grew, so the cap does **not** scale with
`--size`. Deriving it from the canvas — the original bug here — produced
64-colour icons that failed `TOO_MANY_COLORS` on all 18 subjects.

## The full pipeline script

`run.sh` wraps setup, generation, gating and repair:

```bash
./run.sh --out ./icons --size 64 "lantern=a brass lantern with a lit wick"
./run.sh --setup-only     # just build the venv and fetch the model
```

It creates its own venv, installs the pinned requirements, fetches the model,
generates, gates every output, and attempts a repair on anything that fails a
repairable check.

## MCP registration

The portable form, once the package is installed:

```json
{
  "mcpServers": {
    "artgate": { "command": "artgate-gen-mcp" }
  }
}
```

If you are running from a checkout with its own venv, point at that
interpreter instead:

```json
{
  "mcpServers": {
    "artgate": {
      "type": "stdio",
      "command": "/path/to/artgate-gen/venv/bin/python",
      "args": ["/path/to/artgate-gen/mcp_server.py"],
      "env": { "PYTHONPATH": "/path/to/artgate-gen" }
    }
  }
}
```

## Model caching

Generation needs about 7 GB of weights across three repositories. Two
strategies ship, and they answer different questions:

```bash
python -c "from artgate_gen.fetch import fetch_thrifty; fetch_thrifty()"
python -c "from artgate_gen.fetch import fetch_complete; fetch_complete()"
```

**Thrifty** (~7 GB) pulls only the files a load actually touches. **Complete**
(~77 GB) mirrors whole snapshots, which is what `local_files_only=True`
requires: huggingface_hub rejects a *partial* snapshot outright when told to
stay offline.

The loader probes for this rather than assuming. A thrifty cache works fine —
it just revalidates against the Hub on each load, costing a round trip. If you
see an "unauthenticated requests" notice, that is the probe telling you the
offline path was not available; it is deliberately not silenced, because
hiding it would hide the regression.

## VRAM

The MCP server does not keep the model warm. SDXL holds roughly 7 GB and the
same GPU usually runs other work, so the pipeline is dropped after ten idle
minutes. `model_status` reports whether it is currently resident.
