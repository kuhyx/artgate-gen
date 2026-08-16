"""Tests for the model cache fetchers.

Both probes branch on whether ``snapshot_download`` raises, and the two
questions they ask are deliberately different: ``is_cached`` uses the same
allow-patterns the real download uses, while ``is_complete`` demands byte
completeness. Confusing the two once let a partial cache pass as complete.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from artgate_gen.fetch import (
    REPOS,
    SDXL_ALLOW,
    fetch_complete,
    fetch_thrifty,
    is_cached,
    is_complete,
)
from artgate_gen.pipeline import LORA, MODEL, VAE
from tests.assertions import check, check_equal, check_length

if TYPE_CHECKING:
    import pytest

EXPECTED_REPO_COUNT = 3


def test_repos_covers_the_three_models() -> None:
    """All three repositories the pipeline loads are listed."""
    check_equal(
        actual={repo for repo, _ in REPOS},
        expected={MODEL, VAE, LORA},
        message="every loaded repo is fetched",
    )


def test_only_sdxl_restricts_its_patterns() -> None:
    """The VAE and LoRA are fetched whole; reusing SDXL's patterns broke them."""
    patterns = dict(REPOS)
    check_equal(actual=patterns[MODEL], expected=SDXL_ALLOW, message="sdxl patterns")
    check(condition=patterns[VAE] is None, message="the VAE is fetched whole")
    check(condition=patterns[LORA] is None, message="the LoRA is fetched whole")


def test_is_cached_is_true_when_the_probe_succeeds() -> None:
    """A resolvable snapshot means the files are present."""
    check(condition=is_cached(MODEL, SDXL_ALLOW), message="a cached repo probes true")


def test_is_cached_is_false_when_the_probe_raises() -> None:
    """A failing probe means a download is still required."""
    sys.modules["huggingface_hub"].missing.add(MODEL)
    check(
        condition=not is_cached(MODEL, SDXL_ALLOW),
        message="an uncached repo probes false",
    )


def test_is_complete_is_true_when_the_probe_succeeds() -> None:
    """A whole-repo probe that resolves means offline loads will work."""
    check(condition=is_complete(VAE), message="a complete repo probes true")


def test_is_complete_is_false_when_the_probe_raises() -> None:
    """A partial snapshot cannot satisfy an offline load."""
    sys.modules["huggingface_hub"].missing.add(VAE)
    check(condition=not is_complete(VAE), message="a partial repo probes false")


def test_thrifty_reports_a_complete_cache(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With everything cached and complete, nothing is downloaded."""
    fetch_thrifty()
    out = capsys.readouterr().out
    check(condition="already cached" in out, message="reports the cache is populated")
    check(condition="partial" not in out, message="a complete cache says so")


def test_thrifty_flags_a_partial_cache(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached-but-incomplete is called out, because loads still hit the Hub.

    The probes ask different questions, so a repo can satisfy ``is_cached``
    with its allow-patterns while failing the byte-completeness check.
    """
    original = sys.modules["huggingface_hub"].snapshot_download

    def _partial(repo_id: str, **kwargs: object) -> str:
        # `is_cached` passes allow_patterns for SDXL and None for the other
        # two, so keying on that alone would also fail the narrow probe. The
        # whole-repo question is the one with no allow_patterns key present.
        if "allow_patterns" not in kwargs:
            message = f"stub: {repo_id} is only partially cached"
            raise OSError(message)
        return str(original(repo_id, **kwargs))

    monkeypatch.setattr("artgate_gen.fetch.snapshot_download", _partial)
    fetch_thrifty()
    check(condition="partial" in capsys.readouterr().out, message="partial is flagged")


def test_thrifty_downloads_what_is_missing(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing repositories are fetched and reported individually."""
    hub = sys.modules["huggingface_hub"]
    hub.missing.add(LORA)
    original = hub.snapshot_download

    def _download(repo_id: str, **kwargs: object) -> str:
        if kwargs.get("local_files_only"):
            return str(original(repo_id, **kwargs))
        hub.missing.discard(repo_id)
        return f"/stub/cache/{repo_id}"

    monkeypatch.setattr("artgate_gen.fetch.snapshot_download", _download)
    fetch_thrifty()
    out = capsys.readouterr().out
    check(condition="fetching 1 of 3" in out, message="reports the work to be done")
    check(condition=f"cached {LORA}" in out, message="reports each fetch")


def test_complete_reports_the_snapshot_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The full fetch reports where the snapshot landed and how long it took."""
    fetch_complete()
    out = capsys.readouterr().out
    check(condition="complete snapshot at" in out, message="reports the path")
    check(condition="took" in out, message="reports the elapsed time")


def test_complete_requests_the_whole_repo() -> None:
    """No allow-patterns are passed, so nothing is left out."""
    hub = sys.modules["huggingface_hub"]
    fetch_complete()
    check_length(sized=hub.calls, expected=1, message="one download was requested")
    check_equal(actual=hub.calls[0], expected=MODEL, message="the SDXL repo")
