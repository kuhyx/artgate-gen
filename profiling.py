"""Phase timing and resource sampling for the generation run.

Exists because "it feels slower" is not actionable and neither is a single
wall-clock total. Generation has four phases with completely different
bottlenecks -- import is disk and CPU, model load is disk, the CUDA transfer is
PCIe, and denoising is GPU compute -- so a run that got slower needs to say
*which* phase moved before anyone can act on it.

Sampling runs on a background thread rather than around each phase: peak GPU
memory and peak CPU load happen *inside* a phase, and start/end snapshots miss
them entirely. The thread is a daemon and the sampler is best-effort, because
observability must never be the reason a generation fails.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch


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
        with open("/proc/self/statm", encoding="ascii") as handle:
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


class RunProfile:
    """Collects phase timings and a resource trace for one generation run.

    Use as a context manager; :meth:`phase` wraps each span. Nothing here
    raises: a profiler that breaks the run it is measuring is worse than no
    profiler at all.
    """

    def __init__(self, *, interval: float = 0.5) -> None:
        """Set up a profile.

        Args:
            interval: Seconds between resource samples.
        """
        self.interval = interval
        self.phases: list[Phase] = []
        self.samples: list[Sample] = []
        self.started = time.time()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> RunProfile:
        """Start background sampling."""
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop sampling."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _sample_loop(self) -> None:
        """Record a resource sample every ``interval`` seconds until stopped."""
        while not self._stop.wait(self.interval):
            cuda = torch.cuda.is_available()
            self.samples.append(
                Sample(
                    at=time.time() - self.started,
                    gpu_alloc_mb=(
                        torch.cuda.memory_allocated() / 1024**2 if cuda else 0.0
                    ),
                    gpu_reserved_mb=(
                        torch.cuda.memory_reserved() / 1024**2 if cuda else 0.0
                    ),
                    host_rss_mb=_host_rss_mb(),
                    load1=_load1(),
                )
            )

    class _Span:
        """Context manager returned by :meth:`RunProfile.phase`."""

        def __init__(self, profile: RunProfile, name: str, detail: dict) -> None:
            self.profile = profile
            self.name = name
            self.detail = detail
            self.start = 0.0

        def __enter__(self) -> RunProfile._Span:
            # Pending GPU work would otherwise be attributed to whichever phase
            # happens to synchronise next, which is exactly backwards.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start = time.time()
            return self

        def __exit__(self, *exc: object) -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.profile.phases.append(
                Phase(self.name, time.time() - self.start, self.detail)
            )

    def phase(self, name: str, **detail: object) -> RunProfile._Span:
        """Time a span of the run.

        Args:
            name: Phase label.
            **detail: Extras to record alongside the timing.

        Returns:
            A context manager that records the phase on exit.
        """
        return self._Span(self, name, detail)

    def peak_gpu_mb(self) -> float:
        """Return peak reserved GPU memory across the run, in MiB."""
        return max((s.gpu_reserved_mb for s in self.samples), default=0.0)

    def peak_rss_mb(self) -> float:
        """Return peak host RSS across the run, in MiB."""
        return max((s.host_rss_mb for s in self.samples), default=0.0)

    def peak_load(self) -> float:
        """Return peak 1-minute load average across the run."""
        return max((s.load1 for s in self.samples), default=0.0)

    def environment(self) -> list[str]:
        """Describe the machine and library versions the run used.

        Included because a timing number is meaningless without it: the same
        phases on a different GPU, driver or torch build are a different
        measurement, and "it got slower" is unanswerable without knowing whether
        any of these moved.
        """
        lines = [
            f"  torch {torch.__version__} | cuda {torch.version.cuda}"
            f" | cudnn {torch.backends.cudnn.version()}"
        ]
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            lines.append(
                f"  device {props.name} | {props.total_memory / 1024**3:.1f} GiB"
                f" | sm_{props.major}{props.minor} | {props.multi_processor_count} SMs"
            )
        lines.append(f"  host {os.cpu_count()} cpus | {_host_rss_mb():.0f} MiB rss now")
        return lines

    def report(self) -> str:
        """Render a human-readable summary of where the time went."""
        total = sum(p.seconds for p in self.phases)
        width = max((len(p.name) for p in self.phases), default=10)
        lines = ["", "=== run profile ===="]
        for phase in self.phases:
            share = 100 * phase.seconds / total if total else 0.0
            bar = "#" * max(1, round(share / 4))
            extra = (
                " " + " ".join(f"{k}={v}" for k, v in phase.detail.items())
                if phase.detail
                else ""
            )
            lines.append(
                f"  {phase.name:<{width}} {phase.seconds:7.2f}s {share:5.1f}% "
                f"{bar}{extra}"
            )
        lines.append(f"  {'TOTAL':<{width}} {total:7.2f}s")
        if torch.cuda.is_available():
            lines.append(
                f"  peak GPU {self.peak_gpu_mb():.0f} MiB reserved"
                f" | device {torch.cuda.get_device_name(0)}"
            )
        lines.append(
            f"  peak host RSS {self.peak_rss_mb():.0f} MiB"
            f" | peak load1 {self.peak_load():.2f}"
            f" | {len(self.samples)} samples"
        )
        lines.extend(self.environment())
        return "\n".join(lines)

    def write_json(self, path: Path) -> None:
        """Write the full trace as JSON for later comparison.

        Args:
            path: Destination file; parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "started": self.started,
            "phases": [
                {"name": p.name, "seconds": p.seconds, **p.detail} for p in self.phases
            ],
            "peak": {
                "gpu_reserved_mb": self.peak_gpu_mb(),
                "host_rss_mb": self.peak_rss_mb(),
                "load1": self.peak_load(),
            },
            "samples": [
                {
                    "at": s.at,
                    "gpu_alloc_mb": s.gpu_alloc_mb,
                    "gpu_reserved_mb": s.gpu_reserved_mb,
                    "host_rss_mb": s.host_rss_mb,
                    "load1": s.load1,
                }
                for s in self.samples
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
