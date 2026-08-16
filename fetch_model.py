"""Pre-fetch the SDXL checkpoint, the fp16-safe VAE and the pixel-art LoRA.

Split out of ``run.sh`` rather than inlined as ``python -c``: this is real
logic, so it belongs in a file the repo's linters can see.

``snapshot_download`` is idempotent, but it is not *free* -- it still makes a
metadata round-trip per repo and draws progress bars. So the cached case is
detected up front with ``local_files_only=True`` and reported in one line,
which is what makes re-running ``run.sh`` quiet as well as safe.
"""

from __future__ import annotations

from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA = "nerijs/pixel-art-xl"
# Fixed-decoder VAE. SDXL's stock VAE overflows in fp16 and carries
# force_upcast=True, which makes diffusers cast it to fp32 and back on every
# decode -- two deprecated `.to()` calls per image. See generate.py.
VAE = "madebyollin/sdxl-vae-fp16-fix"

# Only the fp16 weights are used (generate.py loads with variant="fp16"), so
# skip the fp32 duplicates -- they roughly double the download for nothing.
SDXL_ALLOW = ["*.json", "*.txt", "*fp16.safetensors", "*tokenizer*"]

# Patterns are PER REPO, deliberately. The VAE repo's weights are named
# `diffusion_pytorch_model.safetensors`, which does NOT match `*fp16.safetensors`
# -- reusing SDXL_ALLOW here would download only config.json, and the cache
# probe below would then report the repo as complete while the weights are
# missing. A probe must use exactly the patterns its own download uses.
REPOS: tuple[tuple[str, list[str] | None], ...] = (
    (MODEL, SDXL_ALLOW),
    (VAE, None),
    (LORA, None),
)


def is_cached(repo: str, allow: list[str] | None) -> bool:
    """Report whether ``repo`` is already complete in the local HF cache.

    Args:
        repo: A Hugging Face repo id.
        allow: The same ``allow_patterns`` the real download uses, or ``None``
            for the whole repo. Passing a different set here would let a
            partial cache masquerade as a complete one.

    Returns:
        ``True`` when every matching file is present locally.
    """
    try:
        snapshot_download(repo, allow_patterns=allow, local_files_only=True)
    except Exception:
        # Any miss -- absent repo, absent revision, absent file -- means we
        # still have to go to the network, and the reason does not change that.
        return False
    return True


def main() -> None:
    """Download whatever is missing, and say so only when there is work."""
    disable_progress_bars()
    missing = [(repo, allow) for repo, allow in REPOS if not is_cached(repo, allow)]

    if not missing:
        print(f"all {len(REPOS)} models already cached")
        return

    print(f"fetching {len(missing)} of {len(REPOS)} models (this downloads several GB)")
    for repo, allow in missing:
        snapshot_download(repo, allow_patterns=allow)
        print(f"cached {repo}")


if __name__ == "__main__":
    main()
