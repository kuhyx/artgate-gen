# Long-context backbone verdict: PENDING

The first SD3.5 potion rendered on 2026-11-23 and passed the 64 px gate. The
status stays on SDXL until kuhy makes the visual call.

| | SDXL + pixel-art-xl | SD3.5-Large + pixel-art-3.5L |
|---|---|---|
| s/image (3090, `./run.sh`, 1 candidate) | 26 s | 51 s with a game on the GPU (sd-cli 44 s: 12 steps at ~3 s/step, ~9 s load); 39 s with the weights out of page cache |
| peak VRAM (MB) | not measured | ~15 200 (weights: 8 600 model, 6 400 encoders) |
| peak RAM | not measured | 3.7 GB, no swap |
| licence | OpenRAIL++ | Stability Community; LoRA apache-2.0 |
| download | cached | 27.9 GB, plus 14.1 GB of q8_0 copies (`python -m artgate_gen.sd35_fetch`) |
| quality | | potion reads as a red flask (12 steps) |

How SD3.5 fits in one minute, measured on 2026-11-23 while a game shared the GPU:

- **fp16 with `--offload-to-cpu`** used 15 GB of RAM plus 7 GB of swap, and
  produced nothing in 68 minutes.
- **fp16 T5 on the CPU** needs 10 GB of RAM. The board's 10 GiB card cap
  OOM-killed it.
- **q8_0** fits all weights in 15 GB of VRAM. Its per-step speed matches fp16
  (~3.2 s at 1024² on Vulkan), and an fp16 and a q8_0 render with the same
  seed look the same. sd35_fetch converts the model and T5 once, in about
  20 s. Converting at load time (`--type q8_0`) instead cost about 10 s per
  render.
- **Steps:** 28 steps would take about 85 s of sampling alone. At 12 steps
  the flask came out clean, while 6 steps left a pink haze. A 28-step
  reference was never rendered, because it can't finish under the 60 s cap.
- **Intermittent crash:** sd-cli aborted once in about 10 runs
  (`free(): corrupted unsorted chunks`), after it had already saved the PNG.
  The rerun passed.

Both backends render through `sd-cli` (stable-diffusion.cpp) in the repo's own
venv; there is no torch install step. sd.cpp encodes SD3.5's T5 prompt in
77-token windows, not one 512-token window (see DOCS-longctx-backends.md), so
this trial measures SD3.5 on sd.cpp, not SD3.5's full T5 context. The SD3.5
VRAM figure comes from sd-cli's own weight report, not from the trial rows.

Resume (steps 3-4 exceed the 60 s cap: run them as a `systemd-run` job):

1. `scripts/check_hf_gate.sh stabilityai/stable-diffusion-3.5-large` then accept terms
2. `scripts/check_hf_gate.sh stabilityai/stable-diffusion-3.5-large --verify-only`
3. `venv/bin/python -m artgate_gen.sd35_fetch --download` (also writes the q8_0 copies)
4. `venv/bin/python -m tools.longctx_trial --manifest ~/src/roadside-assistance/art/MANIFEST.json --out ~/src/roadside-assistance/art/sd35-trial --backend sd35` (then `--backend sdxl`)

Checked offline: the LoRA parses (1058 transformer keys); the long arm is 203 CLIP tokens vs 75 short. Verified on 2026-11-23: sd.cpp maps this diffusers-format LoRA onto the q8_0 model, and the render log shows 1058 / 1058 tensors applied.
