"""Backbone registry: ``--backend`` name -> module exposing ``generate``."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

MODULES = {"sdxl": "artgate_gen.pipeline", "sd35": "artgate_gen.sd35"}
DEFAULT = "sdxl"


class MissingExtraError(ImportError):
    """A backend needs an opt-in extra that this install does not have."""


def load(name: str) -> ModuleType:
    """Import ``name``'s module lazily, keeping its libraries out of ``--help``."""
    return importlib.import_module(MODULES[name])
