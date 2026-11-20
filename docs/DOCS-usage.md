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

**Thrifty** (~7 GB) pulls only the files a render actually touches: the
single-file SDXL checkpoint `sd_xl_base_1.0.safetensors`, its CLIP tokenizer,
the fp16-fix VAE and the LoRA. **Complete** (~77 GB) mirrors whole snapshots.

The loader resolves each file local-first, so a thrifty cache renders with no
Hub round trip. A file missing from the cache is downloaded on first use. If
you see an "unauthenticated requests" notice, a file was not cached; that
notice is deliberately not silenced.

## The renderer

Rendering runs in `sd-cli` from stable-diffusion.cpp (MIT) on its Vulkan
backend, one process per render: SDXL base 1.0, the fp16-fix VAE, the
`nerijs/pixel-art-xl` LoRA (as `<lora:pixel-art-xl:1>`), 25 Euler steps at
CFG 7. The binary is found on `PATH` as `sd-cli`; set `ARTGATE_SD_CLI` to use
another one. A missing binary fails with a message naming
`./run.sh --setup-only`, and a render that exits non-zero, writes no PNG or
writes the wrong size raises with the tail of the sd-cli log.

Prompt weighting keeps the A1111 syntax the old diffusers `lpw` pipeline
used: `(word:1.3)` is parsed by sd.cpp the same way, and a prompt longer than
77 CLIP tokens is split into windows, not truncated.

Measured on an RTX 3090 for the same icon prompt: sd.cpp takes about 20 s per
1024px image including its model load (the denoise alone is 15 to 16 s). The
old diffusers/CUDA path took 9 s per image once warm, after a 27 s one-off
load.

## VRAM

Nothing stays resident: each `sd-cli` process loads SDXL (about 6.6 GB of
VRAM) and frees it on exit, so the GPU is free for other work between renders.
`model_status` reports whether the server holds resolved render inputs, which
cost no VRAM.
