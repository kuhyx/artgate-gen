"""Tests for the explicit, opt-in SD3.5 download."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from artgate_gen import sd35, sd35_fetch
from tests.assertions import check, check_equal

if TYPE_CHECKING:
    import pytest

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


def test_the_default_run_only_reports(capsys: pytest.CaptureFixture[str]) -> None:
    """Without ``--download`` nothing is fetched; size and free disk print."""
    code = sd35_fetch.main([])
    out = capsys.readouterr().out
    check_equal(actual=code, expected=0, message="reporting succeeds")
    check(condition="27.9 GB" in out, message=out)
    check(condition="free" in out, message=out)
    check_equal(
        actual=sys.modules["huggingface_hub"].calls,
        expected=[],
        message="nothing was downloaded",
    )


def test_download_fetches_the_model_and_the_lora() -> None:
    """``--download`` pulls exactly the base subset and the LoRA."""
    code = sd35_fetch.main(["--download"])
    check_equal(actual=code, expected=0, message="the download succeeds")
    check_equal(
        actual=sys.modules["huggingface_hub"].calls,
        expected=[sd35.MODEL, sd35.LORA],
        message="model then LoRA",
    )
