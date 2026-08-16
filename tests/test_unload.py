"""Tests for idle unloading of the pipeline.

The loop waits on an Event rather than sleeping, which is what makes both of
its branches reachable here without a real 60-second wait.
"""

from __future__ import annotations

import threading

from artgate_gen.unload import (
    IDLE_UNLOAD_SECONDS,
    PipelineState,
    maybe_unload,
    start_unloader,
    unload_loop,
)
from tests.assertions import check, check_equal

SENTINEL = object()
NOW = 1000.0


def test_nothing_to_unload_when_no_pipeline_is_resident() -> None:
    """An empty state is left alone."""
    state = PipelineState()
    check(
        condition=not maybe_unload(state, NOW),
        message="no pipeline means nothing to drop",
    )


def test_recently_used_pipeline_is_kept() -> None:
    """A pipeline inside the idle window survives."""
    state = PipelineState(pipe=SENTINEL, last_used=NOW)
    check(
        condition=not maybe_unload(state, NOW + 1.0),
        message="a fresh pipeline is kept",
    )
    check(condition=state.pipe is SENTINEL, message="the pipeline is still held")


def test_idle_pipeline_is_dropped() -> None:
    """A pipeline past the idle window is released."""
    state = PipelineState(pipe=SENTINEL, last_used=NOW)
    check(
        condition=maybe_unload(state, NOW + IDLE_UNLOAD_SECONDS + 1.0),
        message="an idle pipeline is dropped",
    )
    check(condition=state.pipe is None, message="the reference is released")


def test_boundary_is_exclusive() -> None:
    """Exactly at the window the pipeline is still kept."""
    state = PipelineState(pipe=SENTINEL, last_used=NOW)
    check(
        condition=not maybe_unload(state, NOW + IDLE_UNLOAD_SECONDS),
        message="the window boundary itself does not unload",
    )


def test_loop_exits_when_stopped_before_the_first_wait() -> None:
    """A pre-set stop event ends the loop without touching the state."""
    state = PipelineState(pipe=SENTINEL, last_used=0.0)
    stop = threading.Event()
    stop.set()
    unload_loop(state, stop, interval=0.0)
    check(
        condition=state.pipe is SENTINEL,
        message="the loop never ran a check",
    )


def test_loop_unloads_then_exits() -> None:
    """One iteration drops an idle pipeline, then the stop event ends it."""
    state = PipelineState(pipe=SENTINEL, last_used=0.0)
    stop = threading.Event()

    class _OnceEvent(threading.Event):
        """Allows exactly one wait to return False, then stops the loop."""

        def __init__(self) -> None:
            super().__init__()
            self.waits = 0

        def wait(self, timeout: float | None = None) -> bool:
            """Return False once, then True.

            Args:
                timeout: Ignored.

            Returns:
                False on the first call so the body runs, True after.

            """
            del timeout
            self.waits += 1
            return self.waits > 1

    stop = _OnceEvent()
    unload_loop(state, stop, interval=0.0, idle_seconds=0.0)
    check(condition=state.pipe is None, message="the idle pipeline was dropped")


def test_start_unloader_runs_a_daemon_thread_and_stops() -> None:
    """The watcher starts on demand and is stoppable."""
    state = PipelineState()
    thread, stop = start_unloader(state, interval=0.01)
    check(condition=thread.is_alive(), message="the watcher is running")
    check(condition=thread.daemon, message="the watcher does not block exit")
    stop.set()
    thread.join(timeout=2.0)
    check_equal(actual=thread.is_alive(), expected=False, message="the watcher stopped")
