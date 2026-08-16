"""Idle unloading of the diffusion pipeline.

Unlike a dedicated render host, this server does not keep its model warm. SDXL
holds roughly 7 GB and the same GPU runs local LLM work, so a permanently
resident pipeline would squat on memory someone else needs. The pipeline loads
on first use and is dropped after an idle window.

The loop waits on an :class:`threading.Event` rather than sleeping, so a test
can stop it deterministically and both branches are reachable without a
``pragma``. Nothing starts at import time.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

# How long the pipeline may sit unused before it is dropped.
IDLE_UNLOAD_SECONDS = 600

# How often the watcher checks. Also the granularity of the idle window.
POLL_SECONDS = 60


@dataclass
class PipelineState:
    """The loaded pipeline and when it was last used.

    Attributes:
        pipe: The loaded pipeline, or None when nothing is resident.
        last_used: Monotonic-ish timestamp of the most recent use.
        lock: Guards both fields against the watcher thread.
    """

    pipe: Any = None
    last_used: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


def maybe_unload(
    state: PipelineState,
    now: float,
    idle_seconds: float = IDLE_UNLOAD_SECONDS,
) -> bool:
    """Drop the pipeline if it has been idle long enough.

    Args:
        state: The shared pipeline state.
        now: The current time, passed in so tests need no clock control.
        idle_seconds: How long the pipeline may sit unused.

    Returns:
        True if a pipeline was dropped by this call.
    """
    with state.lock:
        if state.pipe is None or now - state.last_used <= idle_seconds:
            return False
        state.pipe = None
    return True


def unload_loop(
    state: PipelineState,
    stop: threading.Event,
    interval: float = POLL_SECONDS,
    idle_seconds: float = IDLE_UNLOAD_SECONDS,
) -> None:
    """Drop the pipeline whenever it goes idle, until stopped.

    Args:
        state: The shared pipeline state.
        stop: Set this to end the loop; it also cuts the wait short.
        interval: Seconds between checks.
        idle_seconds: How long the pipeline may sit unused.
    """
    while not stop.wait(interval):
        maybe_unload(state, time.time(), idle_seconds)


def start_unloader(
    state: PipelineState,
    interval: float = POLL_SECONDS,
) -> tuple[threading.Thread, threading.Event]:
    """Start the idle watcher on a daemon thread.

    Called from the server's entry point, never at import time, so that
    importing the module has no side effects.

    Args:
        state: The shared pipeline state.
        interval: Seconds between checks.

    Returns:
        The running thread and the event that stops it.
    """
    stop = threading.Event()
    thread = threading.Thread(
        target=unload_loop,
        args=(state, stop, interval),
        daemon=True,
    )
    thread.start()
    return thread, stop
