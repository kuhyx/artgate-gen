# MCP tools

Four tools. Two of them need no model and no GPU, so they work in a base
install.

## `generate_icons`

Generate icons with SDXL plus the pixel-art LoRA, then gate them.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `out_dir` | string | — | Directory to write into; created if absent. |
| `subjects` | object | — | Icon name to description, e.g. `{"lantern": "a brass lantern with a lit wick"}`. |
| `size` | integer | `64` | Icon edge. 64 or 32. |
| `candidates` | integer | `1` | Renders per subject, each with its own seed. |
| `seed` | integer | `11` | Base seed; candidate *n* uses `seed + n`. |
| `keep_raw` | boolean | `false` | Also save the 1024px render. |

Returns each icon's path, seed and gate verdict, plus a summary.

**Batch your subjects.** This is the only tool that loads the model, and the
model is not kept warm — one call with ten subjects pays the ~40 s load once,
ten calls pay it ten times.

## `gate_icon`

Run the deterministic gates against an existing PNG. Needs no model.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `path` | string | — | Path to a PNG. |
| `size` | integer | `64` | Which spec to check against: 64 or 32. |

Returns `passed` and any failure `codes`. A size with no spec returns
`gated: false` and a reason, rather than silently passing.

## `describe_icon`

Report an icon's measurable properties — dimensions, opaque coverage, distinct
opaque colour count — without loading anything. Useful for checking what a file
actually is before regenerating it.

## `model_status`

Report whether the pipeline is currently resident in VRAM, how long it has been
idle, and the unload window.

## Gate codes

| Code | Meaning | Repairable |
|---|---|---|
| `CANVAS_SIZE` | Wrong dimensions for the spec. | yes |
| `TOO_MANY_COLORS` | Opaque palette exceeds the cap. | yes |
| `ALPHA_NOT_BINARY` | Alpha is not exactly `{0, 255}`. | yes |
| `MARGIN_TOO_SMALL` | Subject sits too close to the edge. | yes |
| `SCALE_NOT_INVARIANT` | Not genuinely blocky at its declared scale. | no |
| `SILHOUETTE_LOW_CONTRAST` | Subject does not read against its background. | no |
| `OFF_PALETTE` | An opaque pixel is outside the allowed palette. | no |
| `NO_OPAQUE_PIXELS` | Nothing opaque to measure. | no |

"Repairable" means re-encoding can fix it: quantise harder, threshold the
alpha, re-inset the subject. The rest are facts about the art that no amount
of re-encoding changes.

### Two orderings that are load-bearing

`ALPHA_NOT_BINARY` is evaluated **before** `TOO_MANY_COLORS`. A soft-alpha
image has no fully-opaque pixels at all, so an opaque-only colour count returns
zero and passes — two individually correct gates composing into a hole. The
colour count also fails closed on an empty opaque set for the same reason.

In the finisher, alpha is binarised **before** quantisation. Transparent pixels
keep arbitrary RGB, and quantising first blends those into the palette; that
inflated a 16-colour icon to 18 and rejected correct art.

## What the gate does not do

It does not judge whether the art looks good. Every check is a measurement
against a written spec, and a passing verdict means conformance, not quality.
Look at the images before reporting done.
