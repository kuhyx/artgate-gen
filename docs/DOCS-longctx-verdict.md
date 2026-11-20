# Long-context backbone verdict: PENDING

No SD3.5 render exists yet. Status: stay on SDXL until kuhy's visual call.

| | SDXL + pixel-art-xl | SD3.5-Large + pixel-art-3.5L |
|---|---|---|
| s/image (3090) | not measured | not measured |
| peak VRAM (MB) | not measured | not measured |
| licence | OpenRAIL++ | Stability Community; LoRA apache-2.0 |
| download | cached | 27.9 GB (`python -m artgate_gen.sd35_fetch`) |
| quality | | |

Blocked on: SD3.5 terms (403 on 2026-11-05) and download go-ahead.

Both backends render through `sd-cli` (stable-diffusion.cpp) in the repo's own
venv; there is no torch install step. sd.cpp encodes SD3.5's T5 prompt in
77-token windows, not one 512-token window (see DOCS-longctx-backends.md), so
this trial measures SD3.5 on sd.cpp, not SD3.5's full T5 context. Peak VRAM
comes from `./run.sh --profile`, not from the trial rows.

Resume (steps 3-4 exceed the 60 s cap: run them as a `systemd-run` job):

1. `scripts/check_hf_gate.sh stabilityai/stable-diffusion-3.5-large` then accept terms
2. `scripts/check_hf_gate.sh stabilityai/stable-diffusion-3.5-large --verify-only`
3. `venv/bin/python -m artgate_gen.sd35_fetch --download`
4. `venv/bin/python -m tools.longctx_trial --manifest ~/src/roadside-assistance/art/MANIFEST.json --out ~/src/roadside-assistance/art/sd35-trial --backend sd35` (then `--backend sdxl`)

Checked offline: the LoRA parses (1058 transformer keys); the long arm is 203 CLIP tokens vs 75 short. Unverified: that sd.cpp maps this diffusers-format LoRA onto the SD3.5 single-file model (the first render's log shows the applied tensor count).
