"""Tests for the target table.

The prompt-stem assertions are the point of this file. ``ICON_STYLE`` is the
exact string from the bake-off that produced the 31 accepted icons in ``art/``,
so it is asserted character-for-character: an "improvement" to it silently
invalidates every accepted result, and a test that only checked "contains
'pixel art'" would not notice.
"""

from __future__ import annotations

from artgate_gen import pipeline
from artgate_gen.finish import finish_backdrop, finish_icon
from artgate_gen.targets import (
    BACKDROP_NEGATIVE,
    BACKDROP_STYLE,
    ICON_NEGATIVE,
    ICON_STYLE,
    TARGETS,
)
from tests.assertions import check, check_equal

BACKDROP = (256, 64)


def test_icon_style_is_verbatim() -> None:
    """The icon stem matches the bake-off string exactly."""
    check_equal(
        actual=ICON_STYLE,
        expected="pixel art, game item icon, centered, plain black background, crisp",
        message="the icon style stem",
    )


def test_icon_negative_is_verbatim() -> None:
    """The icon negative prompt matches the bake-off string exactly."""
    check_equal(
        actual=ICON_NEGATIVE,
        expected=(
            "photo, realistic, blurry, soft focus, gradient background, text, "
            "watermark, multiple objects, hands, drop shadow"
        ),
        message="the icon negative prompt",
    )


def test_pipeline_re_exports_the_icon_strings() -> None:
    """Existing ``pipeline.STYLE`` importers keep the same values."""
    check_equal(actual=pipeline.STYLE, expected=ICON_STYLE, message="re-exported style")
    check_equal(
        actual=pipeline.NEGATIVE,
        expected=ICON_NEGATIVE,
        message="re-exported negative",
    )


def test_backdrop_negative_drops_multiple_objects() -> None:
    """A landscape is nothing but multiple objects, so the phrase must go."""
    check(
        condition="multiple objects" not in BACKDROP_NEGATIVE,
        message="the backdrop negative drops 'multiple objects'",
    )


def test_backdrop_negative_attacks_letterboxing() -> None:
    """Letterboxing is fought at the prompt level as well as the gate level."""
    for phrase in ("black bars", "letterbox", "border", "frame"):
        check(
            condition=phrase in BACKDROP_NEGATIVE,
            message=f"the backdrop negative contains {phrase!r}",
        )


def test_backdrop_style_drops_icon_framing() -> None:
    """The backdrop stem must not describe a centred object on a void."""
    for phrase in ("game item icon", "centered", "plain black background"):
        check(
            condition=phrase not in BACKDROP_STYLE,
            message=f"the backdrop style drops {phrase!r}",
        )


def test_icon_targets_use_the_icon_finisher() -> None:
    """Square icon targets finish through the background-stripping path."""
    for key in ((32, 32), (64, 64)):
        check(
            condition=TARGETS[key].finisher is finish_icon,
            message=f"{key} uses finish_icon",
        )


def test_backdrop_target_uses_the_backdrop_finisher() -> None:
    """The strip must not go through the background-stripping path."""
    check(
        condition=TARGETS[BACKDROP].finisher is finish_backdrop,
        message="the strip uses finish_backdrop",
    )


def test_backdrop_renders_wider_than_it_finishes() -> None:
    """The render carries the wrap overlap that ``wrap_blend`` folds back in."""
    target = TARGETS[BACKDROP]
    check(
        condition=target.render[0] > target.canvas[0],
        message="the backdrop renders wider than its canvas",
    )


def test_backdrop_reduces_by_a_uniform_integer_factor() -> None:
    """Both axes reduce by the same integer, so pixels stay square."""
    target = TARGETS[BACKDROP]
    blended_width = target.render[0] - (target.render[0] - target.canvas[0] * 8)
    check_equal(
        actual=blended_width // target.canvas[0],
        expected=target.render[1] // target.canvas[1],
        message="the horizontal and vertical reduction factors",
    )


def test_every_target_is_keyed_by_its_own_canvas() -> None:
    """A row cannot claim a canvas that disagrees with its key."""
    for key, target in TARGETS.items():
        check_equal(actual=target.canvas, expected=key, message=f"{key} canvas")
