# Evaluate long-context pixel-art backends (SD3.5-Large, Flux) against SDXL

Work in `~/artgate-gen` (published quality: ruff `select = ["ALL"]`, mypy
strict, 100% branch coverage, 250-line file cap, **no `noqa` and no
`type: ignore` anywhere**) and `~/roadside-assistance` (the art + its manifest).

## WHY THIS EXISTS

artgate-gen is pinned to SDXL + `nerijs/pixel-art-xl`. SDXL conditions only on
CLIP, whose context is **77 tokens**. The stock diffusers pipeline tokenizes
with `truncation=True`, so a longer prompt is silently cut — and that really
happened: a portrait set shipped where 4 of 7 prompts overflowed and two
expression clauses ("dead eyes", "staring") were never asked for at all. The
images looked plausible; the only evidence was a warning line in a render log.

That specific bug is fixed. `artgate_gen.pipeline.LONG_PROMPT_PIPELINE` loads
the `lpw_stable_diffusion_xl` community pipeline, which encodes successive
77-token windows and concatenates the embeddings, so nothing is dropped.

**But chunking is a workaround, not a raised limit.** A token attends only
within its own window, so text past the first ~77 influences the render less
than text before it. There is no prompt ordering that makes every token equal
on SDXL. This brief asks whether moving to a long-context backbone is worth it.

## WHAT IS ALREADY KNOWN — do not re-derive

Researched 2026-08-16. Verified against the HF API and the installed diffusers
source, not from recall.

### The candidate LoRAs

| Repo | Base | Licence | Weight file | Notes |
|---|---|---|---|---|
| `nerijs/pixel-art-3.5L` | SD3.5-Large | apache-2.0 | `pixel-art-3.5L-v2_000000300.safetensors` (146.8 MB) | **Top candidate.** Same author as the `pixel-art-xl` we already use. Card explicitly targets *"pixel perfect and grid aligned 128x128 outputs"* — no other candidate claims grid alignment. Trigger `pixel art` **at the start** of the prompt; strength 1.0 best, 0.8 acceptable. Repo ships 5 raw step checkpoints; the README names v2_000000300 as the winner. |
| `UmeAiRT/FLUX.1-dev-LoRA-Modern_Pixel_art` | FLUX.1-dev | MIT | `ume_modern_pixelart.safetensors` (343.8 MB) | Best Flux option. Trigger `umempart`. Trained at 1024px; samples are *soft/stylised* pixel aesthetic, **not** true pixel grids. Author: "I'm not sure my settings are optimal." |
| `prithivMLmods/Retro-Pixel-Flux-LoRA` | FLUX.1-dev | creativeml-openrail-m | `Retro-Pixel.safetensors` (612.7 MB) | Most downloaded (696) but **explicitly unfinished**: 16 training images, 15 epochs, card says "not the final version and may contain artifacts". |

Also exists: `glif-loradex-trainer/AP123_flux_dev_2DHD_pixel_art` (trigger
`totk style`), `sWizad/pokemon-trainer-sprites-pixelart-flux`,
`strangerzonehf/Flux-Pixel-Background-LoRA` (backdrop-specific, trigger
`Pixel Background`) — the last is interesting for the parallax strips.

**Excluded:** `nerijs/pixel-art-medium-128-v0.1` (SD3-Medium) is
**cc-by-nc-4.0, non-commercial** — disqualifying for a game.
**Do not go looking for** `nerijs/pixel-art-flux`, `glif/anime-pixel-art-flux`
or `AlekseyCalvin/pixel`: all three 404 (checked against a control repo, so
this is not an auth artefact).

### The context win is real but SMALLER than it sounds

Verified in the installed diffusers source, not assumed:

- `pipeline_flux.py`: `max_sequence_length: int = 512` — but CLIP-L still
  truncates at `tokenizer_max_length = 77` and warns on overflow.
- `pipeline_stable_diffusion_3.py`: T5 default is **256**, not 512 (raisable).

Both are **multi-encoder**: a long T5 channel *alongside* short CLIP channels.
So they fix "my long prompt gets truncated". They do **not** give uniform
attention across the whole prompt — the 77-token CLIP branch still contributes.
Anyone claiming "all tokens attended equally" is overselling it.

### Cost on the target machine (RTX 3090, 24 GB)

- FLUX.1-dev does **not** fit in bf16: transformer ~23.8 GB + T5-XXL ~9.5 GB.
- Practical route is GGUF (`city96/FLUX.1-dev-gguf`, ungated, 130k downloads):
  **Q8_0 12.71 GB**, Q6_K 9.86 GB, Q4_K_S 6.81 GB (F16 reference 23.80 GB).
  Q8_0 leaves headroom for T5 + LoRA.
- `enable_model_cpu_offload()` also fits, at ~2-3x speed cost.
- Expect **~3-5x slower than SDXL** at equal steps (~30 s/image at 1024x1024,
  20 steps). FLUX.1-schnell (4 steps, apache-2.0) closes much of that gap.

### Blocker to clear FIRST

**Every base model is `gated=auto`** (FLUX.1-dev, FLUX.1-schnell,
SD3.5-large/medium) and **there is no HF token on this machine**. The LoRAs are
ungated, but nothing downloads until there is an account with the terms
accepted and a token in place. This is a real setup step, not a formality.

Ask the user to run the login themselves — it is interactive:

```
! /home/kuhy/artgate-gen/venv/bin/huggingface-cli login
```

Do not attempt to work around a gate.

## GOAL

Answer one question with real images: **is a long-context backbone worth
switching to for this project's pixel art?** The deliverable is a decision with
evidence attached, not a migration.

## APPROACH

1. **Clear the gate.** Confirm a token exists and the terms are accepted for
   whichever base you evaluate. If the user declines, stop and say so — the
   rest is unreachable.
2. **Start with `nerijs/pixel-art-3.5L`,** not Flux. It is the only candidate
   claiming grid alignment, it shares an author with the current LoRA, and
   SD3.5-Large is a smaller lift than a quantised Flux stack.
3. **Render the same subjects as the SDXL set,** using the recipes in
   `~/roadside-assistance/art/MANIFEST.json` — the seven portrait expressions
   and a couple of backdrop strips. Same subjects, different backbone, is the
   only comparison that means anything.
4. **Use a genuinely long prompt** in at least one arm. The whole premise is
   that more can be pinned; a 40-token prompt tests nothing.
5. **Put every render in `~/roadside-assistance/art/`** under its own category
   (e.g. `sd35-trial/`), each with a full manifest entry — including the
   failures. Nothing lands in `/tmp`.
6. **Build a labelled side-by-side contact sheet** (`tools/artgen.py:label_sheet`)
   with the SDXL set and the candidate set on the same image, and **show the
   user**. The decision is theirs and it is visual.

## MUST

- Record every render in `art/MANIFEST.json` before showing anything.
- Report wall-clock seconds per image and peak VRAM for each backbone. "Slower"
  is not a finding; "31 s vs 7 s, 12.9 GB vs 9.1 GB" is.
- Keep pixel-art quality judgements **visual**. A consistency metric measures
  agreement, not correctness — a set can be uniformly wrong and score well.
  Check grid alignment by eye at 1:1, then zoomed.
- If artgate-gen gains a second backbone, it must be **selectable, not
  swapped**: SDXL stays the default and stays working. Run the full gate before
  committing: `ruff check . && ruff format --check . && mypy --strict
  artgate_gen tools tests generate.py mcp_server.py && pytest && python
  tools/check_line_cap.py`.

## MUST NOT

- Do not delete or overwrite the SDXL portraits or backdrops.
- Do not add a `noqa` or `type: ignore` to either repo.
- Do not download a base model without the user's go-ahead — these are 15-25 GB.
- Do not migrate the project on the strength of a single good render.

## DONE

A labelled side-by-side sheet in `~/roadside-assistance/art/contact-sheets/`,
shown to the user, plus a short written verdict in
`~/artgate-gen/docs/` covering: image quality (grid alignment especially),
seconds per image, peak VRAM, licence, and a recommendation to switch or stay.
"Stay on SDXL" is a perfectly good outcome if that is what the images say.

REMOVE ME AFTER FINISH
