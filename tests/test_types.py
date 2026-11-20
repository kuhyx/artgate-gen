"""Tests for the record describing one stable-diffusion.cpp render's inputs."""

from __future__ import annotations

from artgate_gen import pipeline
from artgate_gen._types import Pipeline
from tests.assertions import check


def test_load_pipeline_returns_the_record() -> None:
    """The loader's result is the declared type, so annotations are honest."""
    check(
        condition=isinstance(pipeline.load_pipeline(), Pipeline),
        message="load_pipeline returns a Pipeline",
    )
