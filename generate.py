"""Generate game art with SDXL + the pixel-art LoRA, then finish it to icons.

This is the winning route from the art bake-off (row B / "b_pixel"): SDXL base
1.0 with `nerijs/pixel-art-xl` at 1024px, background-stripped, then downscaled
and quantised by the artgate finisher. It beat procedural pixel drawing, SVG
vector, and Blender 3D on the same twelve subjects.

Lives outside `python_pkg/` on purpose: importing torch there would force it
into `requirements.txt`, and the repo's CI mirror installs that list into a
clean venv on every push. This module keeps its own venv at `~/artgate-gen/venv`.

Two findings from the bake-off are baked into the defaults:

* **64px, not 32px.** SDXL renders at 1024, so a 32px icon is a 32x downscale
  and fine detail (a coin's emblem, a shield's radial pattern) mushes. The same
  source survives at 64px.
* **Border-ring background removal.** Flooding from the four corners with a
  fixed tolerance cannot distinguish a dark object from a dark field. Seeding
  every border pixel and requiring connectivity to the border means an interior
  region that merely matches the background colour is never eaten.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.utils import logging as diffusers_logging
from PIL import Image
from transformers import logging as transformers_logging

sys.path.insert(0, "/home/kuhy/testsAndMisc")
from python_pkg.artgate.finish import (
    binarize_alpha,
    downscale,
    quantize,
)

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA = "nerijs/pixel-art-xl"
# SDXL's stock VAE overflows in fp16, so it ships `force_upcast=True`: diffusers
# casts it to fp32 before every decode and back to fp16 afterwards, via a method
# it has itself deprecated. That is two `.to()` calls per image, each emitting a
# warning. This drop-in decoder is numerically safe in fp16 and sets
# force_upcast=False, so neither cast happens and both warnings go away.
VAE = "madebyollin/sdxl-vae-fp16-fix"

# Verbatim from the bake-off run that produced the accepted icons.
STYLE = "pixel art, game item icon, centered, plain black background, crisp"
NEGATIVE = (
    "photo, realistic, blurry, soft focus, gradient background, text, "
    "watermark, multiple objects, hands, drop shadow"
)
STEPS = 25
GUIDANCE = 7.0

# Both artgate icon specs cap the palette at 32, and that number is calibrated
# against real art, not chosen: 250 genuine 32x32 icons top out at 30 colours,
# and 300 genuine 48x48 icons top out at 14. Hand-made art does not spend more
# colours on a bigger canvas, so this cap does NOT scale with `size`. Deriving
# it from the canvas instead (the original bug here) produced 64-colour icons
# that failed TOO_MANY_COLORS on all 18 subjects.
MAX_COLOURS = 32


def strip_background(rgba: np.ndarray, tol: int = 60) -> np.ndarray:
    """Clear the background by flooding inward from every border pixel.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.
        tol: Max summed per-channel distance from the border's median colour
            for a pixel to count as background.

    Returns:
        A copy whose background pixels have alpha 0.
    """
    h, w = rgba.shape[:2]
    rgb = rgba[:, :, :3].astype(int)
    border = np.concatenate([rgb[0, :], rgb[-1, :], rgb[:, 0], rgb[:, -1]])
    ref = np.median(border, axis=0)
    is_bg = np.abs(rgb - ref).sum(2) < tol

    keep = np.ones((h, w), bool)
    seen = np.zeros((h, w), bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(w):
        for y in (0, h - 1):
            if not seen[y, x]:
                seen[y, x] = True
                queue.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not seen[y, x]:
                seen[y, x] = True
                queue.append((y, x))

    while queue:
        y, x = queue.popleft()
        if not is_bg[y, x]:
            continue
        keep[y, x] = False
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx]:
                seen[ny, nx] = True
                queue.append((ny, nx))

    out = rgba.copy()
    out[:, :, 3] = np.where(keep, 255, 0)
    return out


def load_pipeline() -> DiffusionPipeline:
    """Load SDXL with the fp16-safe VAE and the pixel-art LoRA onto the GPU."""
    # Must be disabled before from_pretrained: pipe.set_progress_bar_config()
    # below only silences the *inference* bar, so the component-loading bars are
    # already drawn by then. With no TTY they redraw into one unreadable line.
    # Both libraries need telling -- the per-component "Loading weights" bars
    # come from transformers, not diffusers.
    diffusers_logging.disable_progress_bar()
    transformers_logging.disable_progress_bar()
    pipe = DiffusionPipeline.from_pretrained(
        MODEL,
        vae=AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float16),
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    _load_unet_lora(pipe)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _load_unet_lora(pipe: DiffusionPipeline) -> None:
    """Apply the LoRA to the UNet only.

    Not a micro-optimisation and not a warning suppression: every one of this
    LoRA's 2166 tensors is a ``lora_unet_*`` key, so there is nothing for a text
    encoder to load. The convenience wrapper ``load_lora_weights`` nevertheless
    calls ``load_lora_into_text_encoder`` unconditionally for both SDXL text
    encoders, and each one logs that it found no matching keys. Doing the UNet
    load explicitly injects the identical 6498 LoRA modules and stays silent.
    """
    state_dict, alphas, metadata = type(pipe).lora_state_dict(
        LORA, unet_config=pipe.unet.config, return_lora_metadata=True
    )
    pipe.load_lora_into_unet(
        state_dict,
        network_alphas=alphas,
        unet=pipe.unet,
        metadata=metadata,
        _pipeline=pipe,
    )


def generate(pipe: DiffusionPipeline, prompt: str, seed: int) -> Image.Image:
    """Render one 1024px candidate for ``prompt``."""
    return pipe(
        prompt=f"{STYLE}, {prompt}",
        negative_prompt=NEGATIVE,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]


def finish(raw: Image.Image, size: int, colours: int | None = None) -> Image.Image:
    """Strip the background and reduce a 1024px render to a ``size`` icon."""
    arr = np.asarray(raw.convert("RGBA"), dtype=np.uint8)
    stripped = strip_background(arr)
    reduced = quantize(
        binarize_alpha(downscale(stripped, size)), colours or MAX_COLOURS
    )
    return Image.fromarray(reduced, "RGBA")


def main() -> None:
    """Generate every requested subject at the requested size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="output directory")
    parser.add_argument(
        "subjects",
        nargs="+",
        help="'name=prompt' pairs, e.g. lantern='a brass lantern with a lit wick'",
    )
    parser.add_argument("--size", type=int, default=64, help="icon size (default 64)")
    parser.add_argument("--colours", type=int, default=None, help="palette cap")
    parser.add_argument(
        "--candidates",
        type=int,
        default=1,
        help="renders per subject, each with its own seed",
    )
    parser.add_argument("--seed", type=int, default=11, help="base seed")
    parser.add_argument("--keep-raw", action="store_true", help="also save 1024px")
    args = parser.parse_args()

    pairs = []
    for item in args.subjects:
        if "=" not in item:
            parser.error(f"expected name=prompt, got {item!r}")
        name, _, prompt = item.partition("=")
        pairs.append((name, prompt))

    args.out.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out / "raw"
    if args.keep_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline()
    for name, prompt in pairs:
        for c in range(args.candidates):
            seed = args.seed + c
            image = generate(pipe, prompt, seed)
            suffix = "" if args.candidates == 1 else f"_{c}"
            if args.keep_raw:
                image.save(raw_dir / f"{name}{suffix}.png")
            finish(image, args.size, args.colours).save(
                args.out / f"{name}{suffix}.png"
            )
            print(f"generated {name}{suffix} (seed {seed})", flush=True)


if __name__ == "__main__":
    main()
