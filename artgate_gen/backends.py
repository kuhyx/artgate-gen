"""Backbone registry: ``--backend`` name -> module exposing ``generate``."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

MODULES = {"sdxl": "artgate_gen.pipeline", "sd35": "artgate_gen.sd35"}
DEFAULT = "sdxl"


def load(name: str) -> ModuleType:
    """Import ``name``'s module lazily, keeping torch out of ``--help``."""
    return importlib.import_module(MODULES[name])
