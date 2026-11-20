"""How many CLIP tokens a prompt costs, and why that still matters.

Split from ``pipeline`` along a real seam: this asks a question about *text*,
not about rendering. It is also the one place that still uses transformers --
for its CLIP tokenizer only, which needs no torch.

**The 77-token limit is not lifted, and on SDXL it cannot be.** 77 is CLIP's
own context length, and SDXL conditions only on CLIP (ViT-L + OpenCLIP bigG).
stable-diffusion.cpp splits a long prompt into 77-token windows (as the old
``lpw_stable_diffusion_xl`` pipeline did) -- a workaround, and the distinction
is worth keeping straight:

*What it fixes.* The stock pipeline tokenizes with ``truncation=True``, so a
longer prompt is silently cut. A portrait set shipped where 4 of 7 prompts
overflowed and two expressions ("dead eyes", "staring") were never asked for at
all. The images looked plausible; the only evidence was one warning line in a
render log. With chunking, nothing is dropped.

*What it does not fix.* Token strength is not uniform. Each 77-token window is
encoded separately and the embeddings are concatenated, so a token attends only
within its own window -- text past the first window genuinely influences the
render less than text before it. There is no ordering of a long prompt that
makes every token equal on this architecture.

*What a real fix needs.* A model whose text encoder has long uniform context.
SD3 and Flux carry a T5 encoder with ``max_sequence_length`` up to 512, where
every token is attended equally. That is an architecture change, not a flag:
``nerijs/pixel-art-xl`` is tagged
``base_model:adapter:stable-diffusion-xl-base-1.0`` and does not load onto a
different backbone, so the pixel-art look would have to be re-solved there.

So ``token_count`` stays useful: it reports when a prompt has crossed into the
diluted region, which is a real property of the render and not an artefact of
the tokenizer settings. Shorter prompts remain better prompts.
"""

from __future__ import annotations

import importlib
from functools import cache
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    from artgate_gen._types import Pipeline

    Tokenizer = Callable[[str], Mapping[str, list[int]]]

# CLIP's context length. Kept here beside the only function that reads it.
CLIP_CONTEXT = 77


def token_count(pipe: Pipeline, text: str) -> int:
    """Count the CLIP tokens a prompt costs, including its start/end markers.

    Args:
        pipe: The loaded pipeline, whose tokenizer defines the count.
        text: The prompt to measure.

    Returns:
        The token count.
    """
    return len(_tokenizer(str(pipe.tokenizer_dir))(text)["input_ids"])


@cache
def _tokenizer(directory: str) -> Tokenizer:
    """Load the CLIP tokenizer once per directory.

    Imported here, not at module scope, so rendering never pays for importing
    transformers. transformers is untyped to mypy (see the override in
    ``pyproject.toml``), hence the cast.

    Args:
        directory: The SDXL snapshot's ``tokenizer`` folder.

    Returns:
        A callable CLIP tokenizer.
    """
    transformers = importlib.import_module("transformers")
    return cast("Tokenizer", transformers.CLIPTokenizer.from_pretrained(directory))
