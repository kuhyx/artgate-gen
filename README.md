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
