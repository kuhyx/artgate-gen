"""sd-cli's exit-time abort after a saved render, against the fake binary.

The real sd-cli sometimes aborts (SIGABRT) inside ``exit()`` after it has
logged ``images saved``; ``tests/fake_sd_cli.py`` replays that in mode
``abort-after-save``. Failure modes that must still raise live in
``test_pipeline.py``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from artgate_gen import pipeline
from artgate_gen.targets import TARGETS
from tests.assertions import check, check_equal

if TYPE_CHECKING:
    import pytest

RENDER_EDGE = 64


def test_an_abort_after_the_image_is_saved_keeps_the_render(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """sd-cli's exit-time heap abort never throws away a saved PNG."""
    monkeypatch.setenv("FAKE_SD_MODE", "abort-after-save")
    target = replace(TARGETS[64, 64], render=(RENDER_EDGE, RENDER_EDGE))
    image = pipeline.generate(pipeline.load_pipeline(), "a potion", 7, target)
    check_equal(
        actual=image.size,
        expected=(RENDER_EDGE, RENDER_EDGE),
        message="the saved render comes back",
    )
    check(
        condition="aborted after saving" in capsys.readouterr().err,
        message="the abort is reported on stderr, not hidden",
    )
