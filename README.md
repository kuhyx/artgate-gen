# artgate-gen

[![ci](https://github.com/kuhyx/artgate-gen/actions/workflows/ci.yml/badge.svg)](https://github.com/kuhyx/artgate-gen/actions/workflows/ci.yml)

Generate game item icons with SDXL and a pixel-art LoRA, then check them
against a written style spec — as a CLI and as an MCP server.

The generator is the winning route from a bake-off against procedural pixel
drawing, SVG rasterisation and Blender renders on the same twelve subjects.
The gate is deterministic: it measures canvas size, palette count, alpha
binarity, scale invariance, silhouette contrast and margin, and it never
judges whether the art looks good. That judgement stays with you.

## Install

```bash
pip install artgate-gen              # gate + describe tools, no GPU needed
pip install "artgate-gen[generate]"  # adds SDXL generation (CUDA)
```

The base install is small on purpose, so the gate tools work anywhere. Only
`generate_icons` needs the model.

## Quickstart

```bash
# Generate two icons at 64px and gate them.
artgate-gen ./icons "lantern=a brass lantern with a lit wick" "axe=a stone axe"

# Check an icon you already have.
artgate-gate ./icons/lantern.png --spec pixel_icon_64.toml --json
```

As an MCP server, with `.mcp.json`:

```json
{
  "mcpServers": {
    "artgate": { "command": "artgate-gen-mcp" }
  }
}
```

Four tools: `generate_icons`, `gate_icon`, `describe_icon`, `model_status`.
See [docs/DOCS-tools.md](docs/DOCS-tools.md).

## Two things worth knowing before you start

**Generate at 64px, not 32.** SDXL renders at 1024, so a 32px icon is a 32×
downscale and fine detail — a coin's struck emblem, a shield's radial pattern —
turns to mush. The same source survives at 64.

**Batch your subjects into one call.** The model is not kept warm: SDXL holds
about 7 GB and is dropped after ten idle minutes, so the first call after an
unload pays a ~40 s load. One call with ten subjects costs that once.

**For a set of one character — expressions, poses, variants — render one base
image and inpaint the region that changes.** Do not re-run the whole prompt
with a different clause per variant. Re-rolling re-conditions the entire
generation, so every attribute the prompt does not pin is re-invented: measured
across six seeds, accessories changed colour, garments gained and lost detail
and hair changed colour outright. Pinning identity in the prompt was tried
extensively and did not hold. With a mask, every pixel outside it is copied
from the base, so identity is invariant *by construction*.

The loaded `lpw_stable_diffusion_xl` already exposes `inpaint`, `mask_image`
and `strength`. Locate the mask from the image rather than a fixed box,
feather it (~12px at 1024; a hard edge seams once downscaled), keep anything
that must not drift *outside* it, and repaint at strength ~0.9 — identity is
protected by the mask, not by a timid strength. See
`artgate_gen/set_difference.py` for the landmark maths that locates a face.

## Documentation

- [Usage](docs/DOCS-usage.md) — CLI flags, MCP registration, model caching
- [Tools](docs/DOCS-tools.md) — the four MCP tools and their gate codes
- [Development](docs/DOCS-development.md) — the gate, the test suite, contributing

## Licence

MIT. See [LICENSE](LICENSE). Model weights are downloaded from Hugging Face
and carry their own terms — SDXL is under CreativeML Open RAIL++-M.
