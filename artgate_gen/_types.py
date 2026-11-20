"""The resolved inputs of one stable-diffusion.cpp render.

Generation runs in a separate ``sd-cli`` process, so there is no in-memory
model to hold: a "pipeline" is just the binary plus the weight files it is
pointed at. Keeping that as one frozen record preserves the
``load_pipeline`` / ``generate`` split the CLI, the MCP server and the session
drivers were already written against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class Pipeline:
    """Everything ``sd-cli`` needs besides the prompt.

    Attributes:
        binary: Path to the ``sd-cli`` executable.
        model: The single-file SDXL base checkpoint.
        vae: The fp16-fix VAE weights.
        lora_dir: Directory holding the pixel-art LoRA, for
            ``--lora-model-dir``.
        tokenizer_dir: The SDXL snapshot's CLIP tokenizer, for
            ``prompt_budget.token_count``.
    """

    binary: str
    model: Path
    vae: Path
    lora_dir: Path
    tokenizer_dir: Path
