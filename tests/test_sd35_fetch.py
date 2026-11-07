"""Tests for the explicit, opt-in SD3.5 download."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from artgate_gen import sd35, sd35_fetch
from tests.assertions import check, check_equal

if TYPE_CHECKING:
    import pytest

# Sum of the stub listing minus the root single-file checkpoint, which the
# diffusers loader never reads.
NEEDED_BYTES = 706 + 9_985_185_992 + 167_666_902


def test_plan_counts_only_what_the_loader_reads() -> None:
    """The ComfyUI single-file checkpoint is not part of the download."""
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
    check(condition="10.2 GB" in out, message=out)
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
