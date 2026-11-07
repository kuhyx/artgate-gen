"""SD3.5-Large + ``nerijs/pixel-art-3.5L``, opt-in; rationale in docs/DOCS-longctx-*."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import logging as diffusers_logging
from huggingface_hub import snapshot_download
from transformers import logging as transformers_logging

if TYPE_CHECKING:
    from collections.abc import Iterator

    from PIL import Image

    from artgate_gen._types import InferenceResult
    from artgate_gen.profiling import RunProfile
    from artgate_gen.targets import Target

MODEL = "stabilityai/stable-diffusion-3.5-large"
LORA = "nerijs/pixel-art-3.5L"
LORA_WEIGHTS = "pixel-art-3.5L-v2_000000300.safetensors"
LORA_SCALE = 1.0
TRIGGER = "pixel art"
T5_MAX_LEN = 512
STEPS = 28
GUIDANCE = 3.5
OFFLOAD = True
ALLOW_PATTERNS = (
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "tokenizer_3/*",
    "transformer/*",
    "vae/*",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.safetensors",
    "text_encoder_3/config.json",
    "text_encoder_3/model.safetensors.index.json",
    "text_encoder_3/model-0000?-of-00002.safetensors",
)


class NotCachedError(RuntimeError):
    """The SD3.5 weights are not on disk; loading never downloads them."""


class Sd35Pipeline(Protocol):
    """The subset of ``StableDiffusion3Pipeline`` this module uses."""

    tokenizer: Any
    load_lora_weights: Any
    enable_model_cpu_offload: Any
    set_progress_bar_config: Any

    def __call__(self, **kwargs: object) -> InferenceResult:
        """Run inference."""
        ...


def _require_cache() -> None:
    """Raise ``NotCachedError`` unless the model and LoRA are already cached."""
    try:
        snapshot_download(
            MODEL, allow_patterns=list(ALLOW_PATTERNS), local_files_only=True
        )
        snapshot_download(LORA, allow_patterns=[LORA_WEIGHTS], local_files_only=True)
    except Exception as exc:
        message = (
            f"{MODEL} or {LORA} is not cached; run `python -m "
            "artgate_gen.sd35_fetch` for the size, then add `--download`."
        )
        raise NotCachedError(message) from exc


@contextmanager
def _null_span(_name: str, **_detail: object) -> Iterator[None]:
    """Stand in for ``RunProfile.phase`` when profiling is off."""
    yield


def load_pipeline(profile: RunProfile | None = None) -> Sd35Pipeline:
    """Load SD3.5-Large in bf16 with the LoRA, CPU-offloaded to fit 24 GB."""
    diffusers_logging.disable_progress_bar()
    transformers_logging.disable_progress_bar()
    _require_cache()
    span = profile.phase if profile is not None else _null_span
    with span("load:pipeline"):
        pipe = cast(
            "Sd35Pipeline",
            DiffusionPipeline.from_pretrained(
                MODEL, torch_dtype=torch.bfloat16, local_files_only=True
            ),
        )
    with span("load:lora"):
        pipe.load_lora_weights(LORA, weight_name=LORA_WEIGHTS, local_files_only=True)
    with span("load:offload"):
        pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def full_prompt(prompt: str, target: Target) -> str:
    """Join ``target``'s stem and ``prompt`` with the trigger leading once."""
    text = f"{target.style}, {prompt}"
    return text if text.startswith(TRIGGER) else f"{TRIGGER}, {text}"


def generate(pipe: Sd35Pipeline, prompt: str, seed: int, target: Target) -> Image.Image:
    """Render one image of ``prompt`` at ``target.render`` with the long T5 window."""
    width, height = target.render
    result = pipe(
        prompt=full_prompt(prompt, target),
        negative_prompt=target.negative,
        width=width,
        height=height,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        max_sequence_length=T5_MAX_LEN,
        joint_attention_kwargs={"scale": LORA_SCALE},
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    return result.images[0]
