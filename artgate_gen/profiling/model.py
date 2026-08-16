"""Data records and cheap resource probes used by the profiler.

Split from the collector so the plain records can be imported and tested
without pulling in the sampling thread.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Phase:
    """One timed span of the run.

    Attributes:
        name: Phase label, e.g. ``"load_pipeline"``.
        seconds: Wall-clock duration.
        detail: Free-form extras, e.g. the subject and seed for a render.
    """

    name: str
    seconds: float
    detail: dict[str, object] = field(default_factory=dict)


@dataclass
class Sample:
    """One resource reading taken during the run."""

    at: float
    gpu_alloc_mb: float
    gpu_reserved_mb: float
    host_rss_mb: float
    load1: float


def _host_rss_mb() -> float:
    """Return this process's resident set size in MiB, or 0.0 if unreadable."""
    try:
        with Path("/proc/self/statm").open(encoding="ascii") as handle:
            pages = int(handle.read().split()[1])
    except (OSError, IndexError, ValueError):
        return 0.0
    return pages * os.sysconf("SC_PAGE_SIZE") / 1024 / 1024


def _load1() -> float:
    """Return the 1-minute load average, or 0.0 if unreadable."""
    try:
        return os.getloadavg()[0]
    except OSError:
        return 0.0
