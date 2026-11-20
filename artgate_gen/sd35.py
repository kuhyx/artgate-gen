"""SD3.5-Large + ``nerijs/pixel-art-3.5L``, opt-in; rationale in docs/DOCS-longctx-*.

Renders through stable-diffusion.cpp's ``sd-cli``, the same binary and runner
as the SDXL backend, so this backend needs no torch and no extra install.

sd.cpp reads the ComfyUI-style single files from the gated SD3.5 repo: the
model (which carries the VAE) plus the CLIP-L, CLIP-G and T5-XXL encoders. The
SD3.5 licence must be accepted on Hugging Face before they can be fetched, so
loading never downloads: ``python -m artgate_gen.sd35_fetch --download`` is
the only way in.

sd.cpp encodes the T5 prompt in separate 77-token windows, as it does CLIP, so
it does not give the single 512-token T5 window diffusers' ``max_sequence_length``
did. Long prompts are split, not truncated.
"""

from __future__ import annotations

import contextlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import hf_hub_download

from artgate_gen import prompt_budget
from artgate_gen.backends import BackendUnavailableError
from artgate_gen.sd_cli import find_sd_cli, run_sd_cli

if TYPE_CHECKING:
    from collections.abc import Iterator

    from PIL import Image

    from artgate_gen.profiling import RunProfile
    from artgate_gen.targets import Target

MODEL = "stabilityai/stable-diffusion-3.5-large"
CHECKPOINT = "sd3.5_large.safetensors"
CLIP_L = "text_encoders/clip_l.safetensors"
CLIP_G = "text_encoders/clip_g.safetensors"
T5XXL = "text_encoders/t5xxl_fp16.safetensors"
# For prompt_budget.token_count only; CLIPTokenizer reads the whole folder.
TOKENIZER = "tokenizer/vocab.json"
LORA = "nerijs/pixel-art-3.5L"
LORA_WEIGHTS = "pixel-art-3.5L-v2_000000300.safetensors"
LORA_SCALE = 1.0
# sd.cpp applies a LoRA named in the prompt as `<lora:NAME:WEIGHT>`, looking
# NAME up in --lora-model-dir.
LORA_TAG = f"<lora:{Path(LORA_WEIGHTS).stem}:{LORA_SCALE:g}>"
TRIGGER = "pixel art"
STEPS = 28
GUIDANCE = 3.5
SAMPLER = "euler"
# The fp16 model and T5 together exceed a 24 GB card; sd.cpp keeps weights in
# RAM and moves each stage into VRAM when it runs.
OFFLOAD = True
# What `sd35_fetch` downloads: exactly the files above, not the diffusers
# folders or the fp8 T5 that share the repo.
ALLOW_PATTERNS = (CHECKPOINT, CLIP_L, CLIP_G, T5XXL, "tokenizer/*")


class NotCachedError(BackendUnavailableError):
    """The SD3.5 weights are not on disk; loading never downloads them."""


@dataclass(frozen=True, slots=True)
class Sd35Pipeline:
    """Everything ``sd-cli`` needs for an SD3.5 render besides the prompt.

    Attributes:
        binary: Path to the ``sd-cli`` executable.
        model: The single-file SD3.5-Large model, VAE included.
        clip_l: The CLIP-L text encoder.
        clip_g: The CLIP-G text encoder.
        t5xxl: The fp16 T5-XXL text encoder.
        lora_dir: Directory holding the pixel-art LoRA.
        tokenizer_dir: The CLIP tokenizer, for ``token_count``.
    """

    binary: str
    model: Path
    clip_l: Path
    clip_g: Path
    t5xxl: Path
    lora_dir: Path
    tokenizer_dir: Path


def _cached(wanted: list[tuple[str, str]]) -> list[Path]:
    """Return the cached path of each ``(repo, file)``, never downloading.

    Args:
        wanted: The files to look up.

    Returns:
        Their local paths, in order.

    Raises:
        NotCachedError: Naming every missing file and the fetch command.
    """
    found: list[Path] = []
    missing: list[str] = []
    for repo, filename in wanted:
        path: str | None = None
        # huggingface_hub's "not cached" errors vary by version; see pipeline.resolve.
        with contextlib.suppress(Exception):
            path = hf_hub_download(repo, filename, local_files_only=True)
        if path is None:
            missing.append(f"{repo}/{filename}")
        else:
            found.append(Path(path))
    if missing:
        message = (
            f"not cached: {', '.join(missing)}. Accept the SD3.5 licence on "
            "Hugging Face, then run `python -m artgate_gen.sd35_fetch` for the "
            "size and add `--download`."
        )
        raise NotCachedError(message)
    return found


@contextmanager
def _null_span(_name: str, **_detail: object) -> Iterator[None]:
    """Stand in for ``RunProfile.phase`` when profiling is off."""
    yield


def load_pipeline(profile: RunProfile | None = None) -> Sd35Pipeline:
    """Find ``sd-cli`` and the cached SD3.5 weights; nothing loads into memory.

    Args:
        profile: When given, the resolution step is timed.

    Returns:
        The resolved render inputs.
    """
    span = profile.phase if profile is not None else _null_span
    with span("load:resolve"):
        binary = find_sd_cli()
        model, clip_l, clip_g, t5xxl, vocab, lora = _cached(
            [
                *((MODEL, f) for f in (CHECKPOINT, CLIP_L, CLIP_G, T5XXL, TOKENIZER)),
                (LORA, LORA_WEIGHTS),
            ]
        )
        return Sd35Pipeline(
            binary=binary,
            model=model,
            clip_l=clip_l,
            clip_g=clip_g,
            t5xxl=t5xxl,
            lora_dir=lora.parent,
            tokenizer_dir=vocab.parent,
        )


def full_prompt(prompt: str, target: Target) -> str:
    """Join ``target``'s stem and ``prompt`` with the trigger leading once."""
    text = f"{target.style}, {prompt}"
    return text if text.startswith(TRIGGER) else f"{TRIGGER}, {text}"


def token_count(pipe: Sd35Pipeline, text: str) -> int:
    """Return how many CLIP tokens ``text`` costs; see ``prompt_budget``."""
    return prompt_budget.token_count(pipe, text)


def build_argv(
    pipe: Sd35Pipeline, prompt: str, seed: int, target: Target, out: Path
) -> list[str]:
    """Return the ``sd-cli`` command line for one SD3.5 render.

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
        *("-m", str(pipe.model), "--clip_l", str(pipe.clip_l)),
        *("--clip_g", str(pipe.clip_g), "--t5xxl", str(pipe.t5xxl)),
        *("--lora-model-dir", str(pipe.lora_dir)),
        *("-p", full_prompt(prompt, target) + LORA_TAG, "-n", target.negative),
        *("--steps", str(STEPS), "--cfg-scale", str(GUIDANCE)),
        *("--sampling-method", SAMPLER, "-s", str(seed)),
        *("-W", str(width), "-H", str(height), "-o", str(out)),
        *(("--offload-to-cpu",) if OFFLOAD else ()),
    ]


def generate(pipe: Sd35Pipeline, prompt: str, seed: int, target: Target) -> Image.Image:
    """Render one image of ``prompt`` at ``target.render``.

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
