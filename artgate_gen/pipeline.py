"""Model resolution and rendering through stable-diffusion.cpp.

Rendering runs in the ``sd-cli`` binary from stable-diffusion.cpp (MIT) on its
Vulkan backend, as a subprocess. That replaced diffusers on CUDA: torch pulled
in NVIDIA's proprietary CUDA wheels, and torch on CPU took about 11 minutes per
image. Measured on the RTX 3090 for the same icon prompt and settings, sd.cpp
takes about 20 s per image including its model load, against 9 s warm for
diffusers plus a 27 s one-off load. The pixel-art LoRA applies to all 2166 UNet
tensors, and the output passes the same gates.

Nothing in this package imports an ML library. The rest of the package --
finishing, the gate, repair, the CLI -- never depended on one, and now this
module does not either.
"""

from __future__ import annotations

import contextlib
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import hf_hub_download

from artgate_gen import prompt_budget
from artgate_gen._types import Pipeline
from artgate_gen.sd_cli import find_sd_cli, run_sd_cli
from artgate_gen.targets import ICON_NEGATIVE, ICON_STYLE

if TYPE_CHECKING:
    from collections.abc import Iterator

    from PIL import Image

    from artgate_gen.profiling import RunProfile
    from artgate_gen.targets import Target

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
# The single-file checkpoint, not the diffusers folders: sd.cpp loads UNet and
# both text encoders from this one file.
CHECKPOINT = "sd_xl_base_1.0.safetensors"
LORA = "nerijs/pixel-art-xl"
LORA_WEIGHTS = "pixel-art-xl.safetensors"
# sd.cpp applies a LoRA named in the prompt as `<lora:NAME:WEIGHT>`, looking
# NAME up in --lora-model-dir. Weight 1 matches diffusers' default scale. This
# LoRA has only `lora_unet_*` keys, so it touches the UNet alone, as before.
LORA_TAG = f"<lora:{Path(LORA_WEIGHTS).stem}:1>"
# SDXL's stock VAE overflows in fp16; this drop-in decoder does not. sd.cpp
# reads its diffusers-format weights directly.
VAE = "madebyollin/sdxl-vae-fp16-fix"
VAE_WEIGHTS = "diffusion_pytorch_model.safetensors"

# Re-exported from `targets`, which owns them alongside the backdrop variants.
# Kept as module-level names here because callers already import them.
STYLE = ICON_STYLE
NEGATIVE = ICON_NEGATIVE

STEPS = 25
GUIDANCE = 7.0
# diffusers' EulerDiscreteScheduler, which the accepted art was rendered with.
SAMPLER = "euler"

# Both artgate icon specs cap the palette at 32, and that number is calibrated
# against real art, not chosen: 250 genuine 32x32 icons top out at 30 colours,
# and 300 genuine 48x48 icons top out at 14. Hand-made art does not spend more
# colours on a bigger canvas, so this cap does NOT scale with `size`.
MAX_COLOURS = 32


def resolve(repo: str, filename: str) -> Path:
    """Return the cached path of one file, downloading it only if absent.

    Local first, so a warm cache costs no Hub round-trip; any miss falls back
    to the network. huggingface_hub raises a wide, version-dependent family of
    errors for "not cached", so the catch is deliberately broad.

    Args:
        repo: A Hugging Face repo id.
        filename: The file within the repo.

    Returns:
        The local path.
    """
    with contextlib.suppress(Exception):
        return Path(hf_hub_download(repo, filename, local_files_only=True))
    return Path(hf_hub_download(repo, filename))


def load_pipeline(profile: RunProfile | None = None) -> Pipeline:
    """Find the binary and the weight files it renders with.

    Nothing is loaded into memory here: ``sd-cli`` loads the weights per render
    (about 1.5 s from a warm page cache) and frees the GPU when it exits.

    Args:
        profile: When given, the resolution step is timed.

    Returns:
        The resolved render inputs.
    """
    span = profile.phase if profile is not None else _null_span
    with span("load:resolve"):
        binary = find_sd_cli()
        return Pipeline(
            binary=binary,
            model=resolve(MODEL, CHECKPOINT),
            vae=resolve(VAE, VAE_WEIGHTS),
            lora_dir=resolve(LORA, LORA_WEIGHTS).parent,
            tokenizer_dir=resolve(MODEL, "tokenizer/vocab.json").parent,
        )


@contextmanager
def _null_span(_name: str, **_detail: object) -> Iterator[None]:
    """Stand in for ``RunProfile.phase`` when profiling is off."""
    yield


def build_argv(
    pipe: Pipeline, prompt: str, seed: int, target: Target, out: Path
) -> list[str]:
    """Return the ``sd-cli`` command line for one render.

    The prompt goes through untouched, so A1111 weighting such as
    ``(word:1.2)`` -- the syntax the old lpw pipeline parsed -- is parsed by
    sd.cpp the same way. Prompts past CLIP's 77 tokens are split into windows,
    not truncated.

    Args:
        pipe: The resolved render inputs.
        prompt: The subject, appended to the target's style stem.
        seed: Seed for the sampler.
        target: What is being rendered.
        out: Where ``sd-cli`` writes the PNG.

    Returns:
        The argv, binary first.
    """
    width, height = target.render
    return [
        pipe.binary,
        *("-m", str(pipe.model), "--vae", str(pipe.vae)),
        *("--lora-model-dir", str(pipe.lora_dir)),
        *("-p", full_prompt(prompt, target) + LORA_TAG, "-n", target.negative),
        *("--steps", str(STEPS), "--cfg-scale", str(GUIDANCE)),
        *("--sampling-method", SAMPLER, "-s", str(seed)),
        *("-W", str(width), "-H", str(height), "-o", str(out)),
    ]


def full_prompt(prompt: str, target: Target) -> str:
    """Join ``target``'s style stem and ``prompt`` as sent to the pipeline."""
    return f"{target.style}, {prompt}"


def token_count(pipe: Pipeline, text: str) -> int:
    """Return how many CLIP tokens ``text`` costs; see ``prompt_budget``."""
    return prompt_budget.token_count(pipe, text)


def generate(pipe: Pipeline, prompt: str, seed: int, target: Target) -> Image.Image:
    """Render one candidate for ``prompt`` at ``target``'s render size.

    Args:
        pipe: The resolved render inputs.
        prompt: The subject, appended to the target's style stem.
        seed: Seed for the sampler.
        target: What is being rendered.

    Returns:
        One render at ``target.render``.
    """
    return run_sd_cli(
        lambda out: build_argv(pipe, prompt, seed, target, out), target.render
    )
