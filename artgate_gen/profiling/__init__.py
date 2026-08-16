"""Run profiling for generation: where the time and the memory went."""

from __future__ import annotations

from artgate_gen.profiling.model import Phase, Sample
from artgate_gen.profiling.run import RunProfile

__all__ = ["Phase", "RunProfile", "Sample"]
