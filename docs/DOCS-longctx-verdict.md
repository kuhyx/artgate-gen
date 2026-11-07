# Long-context backbone verdict: PENDING

No SD3.5 render exists yet. Status: stay on SDXL until kuhy's visual call.

| | SDXL + pixel-art-xl | SD3.5-Large + pixel-art-3.5L |
|---|---|---|
| s/image (3090) | not measured | not measured |
| peak VRAM (MB) | not measured | not measured |
| licence | OpenRAIL++ | Stability Community; LoRA apache-2.0 |
| download | cached | 27.6 GB (`python -m artgate_gen.sd35_fetch`) |
| quality | | |

Blocked on: SD3.5 terms (403 on 2026-11-05) and download go-ahead.

Resume (steps 3-4 exceed the 60 s cap: start those turns with `LONG:`):

1. `scripts/check_hf_gate.sh stabilityai/stable-diffusion-3.5-large` then accept terms
2. `scripts/check_hf_gate.sh stabilityai/stable-diffusion-3.5-large --verify-only`
3. `python -m artgate_gen.sd35_fetch --download`
4. `python -m tools.longctx_trial --manifest ~/src/roadside-assistance/art/MANIFEST.json --out ~/src/roadside-assistance/art/sd35-trial --backend sd35` (then `--backend sdxl`)

Checked offline: the LoRA parses (1058 transformer keys); the long arm is 203 CLIP tokens vs 75 short. Unverified: local-only load of an `allow_patterns` snapshot.
