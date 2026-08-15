"""Pre-fetch the SDXL base checkpoint and the pixel-art LoRA into the HF cache.

Split out of ``run.sh`` rather than inlined as ``python -c``: this is real
logic, so it belongs in a file the repo's linters can see.

``snapshot_download`` is idempotent -- once the weights are cached this is a
fast metadata check, which is what makes ``run.sh`` safe to re-run.
"""

from __future__ import annotations

from huggingface_hub import snapshot_download

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA = "nerijs/pixel-art-xl"

# Only the fp16 weights are used (generate.py loads with variant="fp16"), so
# skip the fp32 duplicates -- they roughly double the download for nothing.
ALLOW = ["*.json", "*.txt", "*fp16.safetensors", "*tokenizer*"]


def main() -> None:
    """Download both repos into the local Hugging Face cache."""
    snapshot_download(MODEL, allow_patterns=ALLOW)
    print(f"cached {MODEL}")
    snapshot_download(LORA)
    print(f"cached {LORA}")


if __name__ == "__main__":
    main()
