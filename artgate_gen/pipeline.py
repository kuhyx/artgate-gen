"""Model loading and rendering: everything in this package that needs torch.

Isolating the ML stack behind one module is what lets the rest of the package
-- finishing, the gate, repair, the CLI -- import and be tested without a GPU,
a model download, or the multi-gigabyte CUDA wheels. Nothing else here imports
torch or diffusers.
"""

from __future__ import annotations

import contextlib
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

import torch
from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.utils import logging as diffusers_logging
from huggingface_hub import snapshot_download
from transformers import logging as transformers_logging

if TYPE_CHECKING:
    from collections.abc import Iterator

    from PIL import Image

    from artgate_gen._types import Pipeline
    from artgate_gen.profiling import RunProfile

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA = "nerijs/pixel-art-xl"
# Named explicitly rather than left to diffusers' _best_guess_weight_name: that
# helper inspects the repo listing to pick a file, and refuses outright under
# local_files_only ("you must specify a `weight_name`"). Guessing works online
# and fails offline, so the name is pinned to keep both paths identical.
LORA_WEIGHTS = "pixel-art-xl.safetensors"
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


def quiet_hub() -> None:
    """Silence the ``local_dir_use_symlinks`` deprecation.

    Not our call site: diffusers passes that long-deprecated kwarg to
    huggingface_hub itself, defaulted to ``"auto"`` rather than ``None``, so the
    deprecation check fires on every load. Nothing in this repo supplies it and
    nothing here can stop it. Loading offline does *not* suppress it either --
    it is a local deprecation, unrelated to the network. Remove this filter once
    diffusers stops passing the kwarg.

    The companion "unauthenticated requests" message is deliberately NOT
    filtered here: it is a ``Warning:`` header on the Hub's own HTTP response,
    so it appears if and only if a request was actually made. That makes it a
    useful signal that the local-only path silently fell back to the network,
    and hiding it would hide the regression.
    """
    warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")


def local_only() -> bool:
    """Report whether every repo can be loaded without touching the network.

    ``from_pretrained`` revalidates against the Hub on each load unless told not
    to, which costs a round-trip and is the source of the "unauthenticated
    requests" notice. Passing ``local_files_only=True`` skips it -- but
    huggingface_hub then rejects a *partial* snapshot outright, and the SDXL repo
    is only partial if it was fetched with ``allow_patterns``. So the answer is
    not a constant: it depends on how the cache was populated.

    Probing beats assuming in both directions. Hardcoding local-only breaks the
    thrifty ~7 GB cache; hardcoding online gives up the saving for anyone who ran
    ``fetch_full.py``. The probe is a cheap filesystem check, done once.

    Returns:
        ``True`` when the full snapshot is present, so loads can skip the Hub.
    """
    # Every failure mode means the same thing here -- the snapshot is not
    # fully cached, so fall back to the network. huggingface_hub raises a wide
    # and version-dependent family of errors for that condition, so narrowing
    # the catch would turn a graceful fallback into a crash on the first type
    # not listed.
    with contextlib.suppress(Exception):
        for repo in (MODEL, VAE, LORA):
            snapshot_download(repo, local_files_only=True)
        return True
    return False


def load_pipeline(profile: RunProfile | None = None) -> Pipeline:
    """Load SDXL with the fp16-safe VAE and the pixel-art LoRA onto the GPU.

    Args:
        profile: When given, each load step is timed separately. The steps have
            unrelated bottlenecks -- weights come off disk, the LoRA merge is
            CPU, and ``.to("cuda")`` is a PCIe transfer -- so a single "load"
            total cannot say which one moved.
    """
    # Must be disabled before from_pretrained: pipe.set_progress_bar_config()
    # below only silences the *inference* bar, so the component-loading bars are
    # already drawn by then. With no TTY they redraw into one unreadable line.
    # Both libraries need telling -- the per-component "Loading weights" bars
    # come from transformers, not diffusers.
    diffusers_logging.disable_progress_bar()
    transformers_logging.disable_progress_bar()
    quiet_hub()

    # Skips a Hub round-trip per repo when the full snapshot is cached; falls
    # back to the network automatically when it is not, so a thrifty cache keeps
    # working. Probed once here rather than per call.
    offline = local_only()

    span = profile.phase if profile is not None else _null_span
    with span("load:vae"):
        vae = AutoencoderKL.from_pretrained(
            VAE, torch_dtype=torch.float16, local_files_only=offline
        )
    with span("load:pipeline"):
        pipe = DiffusionPipeline.from_pretrained(
            MODEL,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=offline,
        )
    with span("load:lora"):
        _load_unet_lora(pipe, local_files_only=offline)
    with span("load:to_cuda"):
        pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return cast("Pipeline", pipe)


@contextmanager
def _null_span(_name: str, **_detail: object) -> Iterator[None]:
    """Stand in for ``RunProfile.phase`` when profiling is off."""
    yield


def _load_unet_lora(pipe: Pipeline, *, local_files_only: bool) -> None:
    """Apply the LoRA to the UNet only.

    Not a micro-optimisation and not a warning suppression: every one of this
    LoRA's 2166 tensors is a ``lora_unet_*`` key, so there is nothing for a text
    encoder to load. The convenience wrapper ``load_lora_weights`` nevertheless
    calls ``load_lora_into_text_encoder`` unconditionally for both SDXL text
    encoders, and each one logs that it found no matching keys. Doing the UNet
    load explicitly injects the identical 6498 LoRA modules and stays silent.

    Args:
        pipe: The pipeline whose UNet receives the LoRA.
        local_files_only: Skip the Hub revalidation for the LoRA repo too.
    """
    state_dict, alphas, metadata = type(pipe).lora_state_dict(
        LORA,
        weight_name=LORA_WEIGHTS,
        unet_config=pipe.unet.config,
        return_lora_metadata=True,
        local_files_only=local_files_only,
    )
    pipe.load_lora_into_unet(
        state_dict,
        network_alphas=alphas,
        unet=pipe.unet,
        metadata=metadata,
        _pipeline=pipe,
    )


def generate(pipe: Pipeline, prompt: str, seed: int) -> Image.Image:
    """Render one 1024px candidate for ``prompt``."""
    result = pipe(
        prompt=f"{STYLE}, {prompt}",
        negative_prompt=NEGATIVE,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=torch.Generator("cuda").manual_seed(seed),
    )
    return result.images[0]
