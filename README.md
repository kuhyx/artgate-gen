# artgate-gen

SDXL + pixel-art LoRA icon generation — the winning route from the art bake-off.

Generates game item icons with `stabilityai/stable-diffusion-xl-base-1.0` plus
the `nerijs/pixel-art-xl` LoRA, strips the background, and finishes to a
gate-passing pixel icon using the finisher in `~/testsAndMisc`
(`python_pkg/artgate/finish.py`).

## Use

```bash
# One-shot: installs uv, the venv, the deps and the model, then generates + gates
./run.sh ./icons "lantern=a brass lantern with a lit wick"
./run.sh --setup-only          # just prepare the environment
./run.sh -c 3 ./icons "crown=a golden crown with red jewels"   # 3 seeds, pick one
./run.sh --profile ./icons "cat=a black cat"   # where the time went + peak GPU/RAM

# Or drive the generator directly once set up
./venv/bin/python generate.py <outdir> "lantern=a brass lantern with a lit wick"

# gate what came out (deterministic, no tokens)
cd ~/testsAndMisc && python -m python_pkg.artgate.cli <icon.png> \
  --spec python_pkg/artgate/styles/pixel_icon_64.toml --json   # {"exit": 0} passes
```

Also registered as the `artgate` MCP server (`mcp_server.py`), which generates
*and* gates in one call: `generate_icons`, `gate_icon`, `describe_icon`,
`model_status`.

Defaults are the bake-off-winning settings: 64px, 32 colours, seed 11, 25
steps, guidance 7.0.

## Model cache: thrifty vs. complete

Two supported states. `generate.py` probes which one you are in and adapts —
you do not pass a flag.

| | disk | load path |
|---|---|---|
| **thrifty** (default, `fetch_model.py`) | ~7 GB | checks the Hub on every load |
| **complete** (`fetch_full.py`) | ~77 GB | fully local, no network |

The thrifty cache fetches only the fp16 weights, configs and tokenizers. That
makes the snapshot *partial*, and huggingface_hub rejects a partial snapshot
whenever no `allow_patterns` are given — which is exactly what diffusers does
when it cannot reach the Hub. So offline loading requires byte completeness,
including ~70 GB of fp32/ONNX/Flax/OpenVINO duplicates this pipeline never
loads. That is the whole cost, and it buys ~1-2s per run plus no network
dependency.

`fetch_model.py` tells you which state you are in:

```
all 3 models already cached                                  # complete
all 3 models already cached (partial; loads still check the Hub)
```

Note the `local_dir_use_symlinks` deprecation is **not** a network artifact and
survives either mode — diffusers passes a deprecated kwarg to huggingface_hub,
so it is filtered in `quiet_hub()`. The "unauthenticated requests" notice is
deliberately *not* filtered: it is a header on the Hub's own response, so it
appears only when a request really happened, which makes it a useful signal that
the offline path silently fell back.

## Profiling

`--profile` breaks the run into phases and reports peak GPU/RSS/load, because
"it feels slower" is not actionable and one wall-clock total cannot say which
phase moved. `--profile-json <path>` (on `generate.py`) writes the full trace,
including the sample series, for comparing runs over time.

Typical one-icon run on a 3090 (~15s total):

| phase | ~time | bound by |
|---|---|---|
| `load:vae` / `load:pipeline` | 1.5s | disk, plus a Hub metadata check |
| `load:lora` | 2.2s | CPU — merging 6498 modules into the UNet |
| `load:to_cuda` | 1.5s | PCIe transfer |
| `render:<name>` | 9.1s | GPU — 25 denoising steps at 1024px |
| `finish:<name>` | 0.8s | CPU — the pure-Python border flood fill |

Model load is paid once per process, so it amortises: generating N subjects in
one invocation costs roughly `5s + N * 10s`, which is why batching subjects into
a single call beats calling `run.sh` once per icon.

## Why it lives here and not in testsAndMisc

`import torch` under `python_pkg/` would force torch into that repo's
`requirements.txt`, which its CI mirror installs into a clean venv on every
push. This keeps its own venv instead.

## Things that will bite you

- **The palette cap is 32 at every canvas size**, not derived from the size.
  Calibrated against real art: 250 genuine 32px icons max out at 30 colours,
  300 genuine 48px icons max out at 14. Deriving it from the canvas produced
  64-colour icons that failed `TOO_MANY_COLORS` on all 18 subjects.
- **Prefer 64px.** SDXL renders at 1024, so 32px is a 32x downscale and fine
  detail (a coin's emblem, a lantern's panes) is destroyed.
- **Keep prompts terse.** Negation does not work in a positive prompt ("no
  tree" pushes toward trees) and repeating the noun multiplies the object.
  `a red apple with a green leaf` gives one apple; `a single red apple fruit,
  one whole apple, no tree, no branches` gives a tile of dozens. Change the
  seed before you change the prompt.
- **Background removal floods from every border pixel**, keyed to the border's
  median colour, and requires connectivity to the border. A four-corner flood
  with a fixed tolerance eats dark objects on dark fields.
- **`mcp>=1.12,<2`.** mcp 2.x renames `FastMCP` to `MCPServer`.
- The model is not kept warm — it loads on first use (~40 s, peaks 15.9 GB) and
  unloads after 10 idle minutes, returning 14.9 GB. Batch subjects into one run.

## Recreating the venv

```bash
uv venv venv && VIRTUAL_ENV=venv uv pip install -r requirements.txt
```
