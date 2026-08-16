"""A minimal stand-in for :mod:`huggingface_hub`.

``snapshot_download`` is the interesting one: the cache probes branch on
whether it raises, so tests need to drive both outcomes. The module-level
``missing`` set names the repos that should be treated as absent locally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.stubs._inert import fallback

if TYPE_CHECKING:
    from collections.abc import Callable

# Repos that `snapshot_download` should report as not cached. Tests mutate
# this to exercise the cached and uncached branches of the probes.
missing: set[str] = set()

# Every repo id passed to `snapshot_download`, in call order.
calls: list[str] = []


def snapshot_download(repo_id: str, **kwargs: object) -> str:
    """Pretend to resolve a repository snapshot.

    Args:
        repo_id: The repository being requested.
        **kwargs: Ignored; accepts ``allow_patterns`` and ``local_files_only``.

    Returns:
        A fake local path.

    Raises:
        OSError: When ``repo_id`` is listed in ``missing``.

    """
    del kwargs
    calls.append(repo_id)
    if repo_id in missing:
        message = f"stub: {repo_id} is not cached locally"
        raise OSError(message)
    return f"/stub/cache/{repo_id}"


def disable_progress_bars() -> None:
    """Silence progress bars; a no-op here."""


def reset() -> None:
    """Clear recorded state between tests."""
    missing.clear()
    calls.clear()


def __getattr__(name: str) -> Callable[..., None]:
    """Return an inert placeholder for any unexercised attribute.

    Args:
        name: The attribute being looked up.

    Returns:
        A no-op callable.

    """
    return fallback(name)
