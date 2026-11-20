"""Backbone registry: ``--backend`` name -> module exposing ``generate``."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

MODULES = {"sdxl": "artgate_gen.pipeline", "sd35": "artgate_gen.sd35"}
DEFAULT = "sdxl"


class BackendUnavailableError(RuntimeError):
    """A backend cannot render on this machine yet, e.g. its weights are absent."""


def load(name: str) -> ModuleType:
    """Import ``name``'s module lazily, keeping its libraries out of ``--help``."""
    return importlib.import_module(MODULES[name])
