"""Populate the local Hugging Face cache for the models this package uses.

Two strategies, deliberately kept side by side. The thrifty fetch pulls only
the files a load actually touches (~7 GB); the complete fetch mirrors whole
repository snapshots (~77 GB) so that ``local_files_only`` loads work, because
huggingface_hub rejects a *partial* snapshot outright when asked to stay
offline. Which one you want depends on whether you care more about disk space
or about never touching the network mid-run.
"""

from __future__ import annotations

import contextlib
import time

from huggingface_hub import disable_progress_bars, snapshot_download

from artgate_gen.console import emit
from artgate_gen.pipeline import LORA, MODEL, VAE

# Only the fp16 weights are used (the pipeline loads with variant="fp16"), so
# the fp32 duplicates are skipped -- they roughly double the download for
# nothing.
SDXL_ALLOW = ["*.json", "*.txt", "*fp16.safetensors", "*tokenizer*"]

# Patterns are PER REPO, deliberately. The VAE repo's weights are named
# `diffusion_pytorch_model.safetensors`, which does NOT match
# `*fp16.safetensors` -- reusing SDXL_ALLOW here would download only
# config.json, and the cache probe would then report the repo as complete while
# the weights were missing. A probe must use exactly the patterns its own
# download uses.
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
    # Any miss -- absent repo, absent revision, absent file -- means we still
    # have to go to the network, and the reason does not change that. The error
    # family is wide and version-dependent, so narrowing this would turn a
    # graceful re-fetch into a crash.
    with contextlib.suppress(Exception):
        snapshot_download(repo, allow_patterns=allow, local_files_only=True)
        return True
    return False


def is_complete(repo: str) -> bool:
    """Report whether the *whole* repo is cached, not just the parts we use.

    Distinct from :func:`is_cached`, which asks the narrower question. A load
    with ``local_files_only=True`` skips the Hub round-trip but demands byte
    completeness, so this is what decides whether generation can run offline.

    Args:
        repo: A Hugging Face repo id.

    Returns:
        ``True`` when no file in the repo is missing locally.
    """
    with contextlib.suppress(Exception):
        snapshot_download(repo, local_files_only=True)
        return True
    return False


def fetch_thrifty() -> None:
    """Download whatever is missing, and say so only when there is work."""
    disable_progress_bars()
    missing = [(repo, allow) for repo, allow in REPOS if not is_cached(repo, allow)]

    if not missing:
        offline = all(is_complete(repo) for repo, _ in REPOS)
        note = "" if offline else " (partial; loads still check the Hub)"
        emit(f"all {len(REPOS)} models already cached{note}")
        return

    emit(f"fetching {len(missing)} of {len(REPOS)} models (this downloads several GB)")
    for repo, allow in missing:
        snapshot_download(repo, allow_patterns=allow)
        emit(f"cached {repo}")


def fetch_complete() -> None:
    """Download every file in the SDXL repo, then report the elapsed time."""
    start = time.time()
    # No allow/ignore patterns on purpose: the point is byte-completeness, so
    # that a later local_files_only=True load has nothing left to object to.
    path = snapshot_download(MODEL)
    emit(f"\ncomplete snapshot at {path}")
    emit(f"took {time.time() - start:.0f}s")
