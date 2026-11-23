"""Tests for the explicit, opt-in SD3.5 download."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from artgate_gen import sd35, sd35_fetch, sd_cli
from tests.assertions import check, check_equal

if TYPE_CHECKING:
    from pathlib import Path

# The single-file model, the three fp16 text encoders and the tokenizer: what
# sd-cli and the token counter read. Not the fp8 T5, not the diffusers folders.
NEEDED_BYTES = 16_460_379_262 + 1_389_382_176 + 246_144_152 + 9_787_841_024 + 1_059_962


def test_plan_counts_only_what_the_loader_reads() -> None:
    """The diffusers folders and the fp8 T5 are not part of the download."""
    check_equal(
        actual=sd35_fetch.planned_bytes(),
        expected=NEEDED_BYTES,
        message="only allow-listed files are summed",
    )


@pytest.fixture
def lookups(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record each cache lookup's ``local_files_only``; fail on any download.

    Args:
        monkeypatch: Wraps the module's hub functions.

    Returns:
        The recorded flags, in call order.
    """
    flags: list[object] = []
    hub = sys.modules["huggingface_hub"]

    def spy(repo: str, filename: str, **kwargs: object) -> str:
        flags.append(kwargs.get("local_files_only"))
        return str(hub.hf_hub_download(repo, filename, **kwargs))

    def refuse(repo: str, **_kwargs: object) -> str:
        message = f"downloaded {repo}"
        raise AssertionError(message)

    monkeypatch.setattr(sd35_fetch, "hf_hub_download", spy)
    monkeypatch.setattr(sd35_fetch, "snapshot_download", refuse)
    return flags


def test_the_default_run_reports_a_full_cache(
    capsys: pytest.CaptureFixture[str], lookups: list[object]
) -> None:
    """Without ``--download`` nothing is fetched; a full cache says so."""
    code = sd35_fetch.main([])
    out = capsys.readouterr().out
    check_equal(actual=code, expected=0, message="reporting succeeds")
    check(condition="cached" in out, message=out)
    check(condition="free" in out, message=out)
    check(condition="to fetch" not in out, message=out)
    check_equal(actual=set(lookups), expected={True}, message="local lookups only")


def test_the_default_run_reports_the_missing_size(
    capsys: pytest.CaptureFixture[str], lookups: list[object]
) -> None:
    """An empty cache reports the whole allow-listed size still to fetch."""
    sys.modules["huggingface_hub"].missing.add(sd35.MODEL)
    sd35_fetch.main([])
    out = capsys.readouterr().out
    check(condition="27.9 GB to fetch" in out, message=out)
    check(condition="free" in out, message=out)
    check_equal(actual=set(lookups), expected={True}, message="local lookups only")


def _drop_quantised() -> None:
    """Remove the q8_0 copies the conftest makes, as on a fresh machine."""
    for filename in sd35.QUANTISED:
        sd35.quant_path(filename).unlink()


def test_download_fetches_the_model_and_the_lora() -> None:
    """``--download`` pulls exactly the base subset and the LoRA."""
    code = sd35_fetch.main(["--download"])
    check_equal(actual=code, expected=0, message="the download succeeds")
    check_equal(
        actual=sys.modules["huggingface_hub"].calls[:2],
        expected=[sd35.MODEL, sd35.LORA],
        message="model then LoRA",
    )


def test_download_converts_the_model_and_t5_to_q8_0(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The q8_0 copies are made once, through ``sd-cli -M convert``."""
    _drop_quantised()
    record = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_SD_ARGV", str(record))
    check_equal(actual=sd35_fetch.main(["--download"]), expected=0, message="ok")
    for filename in sd35.QUANTISED:
        check(condition=sd35.quant_path(filename).is_file(), message=filename)
    argv: list[str] = json.loads(record.read_text(encoding="utf-8"))
    check_equal(actual=argv[argv.index("--type") + 1], expected="q8_0", message="type")
    check(condition=argv[argv.index("-m") + 1].endswith(sd35.T5XXL), message="src")
    leftovers = sorted(p.name for p in sd35.quant_dir().iterdir())
    check_equal(
        actual=leftovers,
        expected=sorted(sd35.quant_path(f).name for f in sd35.QUANTISED),
        message="no partial files left",
    )


def test_download_keeps_existing_quantised_copies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A second ``--download`` converts nothing."""
    record = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_SD_ARGV", str(record))
    sd35_fetch.main(["--download"])
    check(condition=not record.exists(), message="sd-cli was not run")


def test_a_failed_conversion_names_sd_cli_and_leaves_no_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A conversion error surfaces with sd-cli's log; nothing half-written stays."""
    _drop_quantised()
    monkeypatch.setenv("FAKE_SD_MODE", "fail")
    with pytest.raises(sd_cli.SdCliError, match="Vulkan buffer"):
        sd35_fetch.main(["--download"])
    check_equal(actual=list(sd35.quant_dir().iterdir()), expected=[], message="no copy")


def test_the_report_names_missing_quantised_copies(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The size report says when the q8_0 copies still need ``--download``."""
    _drop_quantised()
    sd35_fetch.main([])
    out = capsys.readouterr().out
    check(condition="q8_0" in out, message=out)
    check(condition="--download" in out, message=out)
