"""Fetch the COMPLETE SDXL snapshot so the pipeline can load fully offline.

Normally ``fetch_model.py`` fetches only what generation reads: the fp16
safetensors, the configs and the tokenizers. That is ~7 GB instead of ~77 GB.

The cost of that thrift is that the local snapshot is *partial*, and
huggingface_hub refuses a partial snapshot whenever no ``allow_patterns`` are
supplied. Diffusers supplies patterns only when it can reach the Hub -- it needs
``model_info`` to know which files a variant implies -- so on the offline path it
asks for everything and the load fails. Completing the snapshot is therefore the
price of never touching the network at load time.

Run this once. It is idempotent; already-present files are skipped.
"""

from __future__ import annotations

import time

from huggingface_hub import snapshot_download

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


def main() -> None:
    """Download every file in the SDXL repo, then report the elapsed time."""
    start = time.time()
    # No allow/ignore patterns on purpose: the point is byte-completeness, so
    # that a later local_files_only=True load has nothing left to object to.
    path = snapshot_download(MODEL)
    print(f"\ncomplete snapshot at {path}")
    print(f"took {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
