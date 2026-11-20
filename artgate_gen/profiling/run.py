"""The profile collector: phase timing plus a background resource sampler.

Sampling runs on its own thread rather than snapshotting around each phase,
because peak CPU load happens *inside* a phase and start/end readings miss it
entirely. The thread is a daemon and every probe is best-effort: observability
must never be the reason a generation fails.

Rendering happens in a separate ``sd-cli`` process, so GPU memory is not
visible from here; ``sd-cli`` logs its own VRAM use. The report gives the peak
RSS of finished child processes instead, which is the render's host memory.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import threading
import time
from typing import TYPE_CHECKING, Self

from artgate_gen.profiling.model import Phase, Sample, _host_rss_mb, _load1

if TYPE_CHECKING:
    from pathlib import Path


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

    def __enter__(self) -> Self:
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
            self.samples.append(
                Sample(
                    at=time.time() - self.started,
                    host_rss_mb=_host_rss_mb(),
                    load1=_load1(),
                )
            )

    class _Span:
        """Context manager returned by :meth:`RunProfile.phase`."""

        def __init__(
            self,
            profile: RunProfile,
            name: str,
            detail: dict[str, object],
        ) -> None:
            self.profile = profile
            self.name = name
            self.detail = detail
            self.start = 0.0

        def __enter__(self) -> RunProfile._Span:
            # No GPU sync needed: sd-cli has exited, and its work with it, by
            # the time a render call returns.
            self.start = time.time()
            return self

        def __exit__(self, *exc: object) -> None:
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

    @staticmethod
    def peak_child_rss_mb() -> float:
        """Return the peak RSS of any finished child process, in MiB."""
        return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024

    def peak_rss_mb(self) -> float:
        """Return peak host RSS across the run, in MiB."""
        return max((s.host_rss_mb for s in self.samples), default=0.0)

    def peak_load(self) -> float:
        """Return peak 1-minute load average across the run."""
        return max((s.load1 for s in self.samples), default=0.0)

    def environment(self) -> list[str]:
        """Describe the machine and library versions the run used.

        Included because a timing number is meaningless without it: the same
        phases on a different GPU, driver or sd.cpp build are a different
        measurement, and "it got slower" is unanswerable without knowing whether
        any of these moved. The device itself is named in sd-cli's own log.
        """
        # Same seam as `sd_cli.SD_CLI_ENV`, spelled out rather than imported:
        # the pipeline imports huggingface_hub, and the CLI imports this module
        # in the base install, which does not have it.
        binary = os.environ.get("ARTGATE_SD_CLI") or "sd-cli"
        lines = [f"  sd-cli {shutil.which(binary) or 'not found'}"]
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
        lines.append(f"  peak sd-cli RSS {self.peak_child_rss_mb():.0f} MiB")
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
                "child_rss_mb": self.peak_child_rss_mb(),
                "host_rss_mb": self.peak_rss_mb(),
                "load1": self.peak_load(),
            },
            "samples": [
                {
                    "at": s.at,
                    "host_rss_mb": s.host_rss_mb,
                    "load1": s.load1,
                }
                for s in self.samples
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
