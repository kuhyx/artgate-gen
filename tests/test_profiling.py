"""Tests for run profiling.

The resource probes are best-effort by design -- observability must never be
the reason a generation fails -- so their failure paths are tested as
deliberately as their happy paths.
"""

from __future__ import annotations

import json
import sys
import time
from typing import TYPE_CHECKING

from artgate_gen.profiling import Phase, RunProfile, Sample
from artgate_gen.profiling.model import _host_rss_mb, _load1
from tests.assertions import check, check_equal, check_length

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_phase_carries_its_detail() -> None:
    """A phase records its label, duration and extras."""
    phase = Phase(name="render", seconds=1.5, detail={"seed": 7})
    check_equal(actual=phase.name, expected="render", message="name")
    check_equal(actual=phase.detail["seed"], expected=7, message="detail")


def test_sample_is_a_plain_record() -> None:
    """A sample holds one reading of each resource."""
    sample = Sample(
        at=1.0,
        gpu_alloc_mb=2.0,
        gpu_reserved_mb=3.0,
        host_rss_mb=4.0,
        load1=5.0,
    )
    check_equal(actual=sample.host_rss_mb, expected=4.0, message="rss")


def test_host_rss_is_positive_on_linux() -> None:
    """The real probe reads /proc and reports something plausible."""
    check(condition=_host_rss_mb() > 0, message="this process has resident memory")


def test_host_rss_returns_zero_when_proc_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable /proc yields 0.0 rather than raising."""

    def _boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        message = "no /proc here"
        raise OSError(message)

    monkeypatch.setattr("pathlib.Path.open", _boom)
    check_equal(actual=_host_rss_mb(), expected=0.0, message="probe degrades to zero")


def test_load1_returns_zero_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A platform without load averages yields 0.0."""

    def _boom() -> tuple[float, float, float]:
        message = "no load average"
        raise OSError(message)

    monkeypatch.setattr("os.getloadavg", _boom)
    check_equal(actual=_load1(), expected=0.0, message="probe degrades to zero")


def test_load1_reads_the_real_average() -> None:
    """The real probe returns a non-negative number."""
    check(condition=_load1() >= 0.0, message="load average is non-negative")


def test_profile_records_phases() -> None:
    """A timed span lands in the phase list."""
    with RunProfile(interval=0.01) as profile, profile.phase("work", seed=3):
        pass
    check_length(sized=profile.phases, expected=1, message="one phase recorded")
    check_equal(actual=profile.phases[0].name, expected="work", message="phase name")
    check_equal(actual=profile.phases[0].detail["seed"], expected=3, message="detail")


def test_report_on_an_empty_run_does_not_divide_by_zero() -> None:
    """A run with no phases still produces a report."""
    with RunProfile(interval=0.01) as profile:
        pass
    check(condition=len(profile.report()) > 0, message="an empty run still reports")


def test_report_lists_phases_with_shares() -> None:
    """A completed run reports each phase."""
    with RunProfile(interval=0.01) as profile:
        with profile.phase("alpha"):
            pass
        with profile.phase("beta"):
            pass
    report = profile.report()
    check(condition="alpha" in report, message="the first phase appears")
    check(condition="beta" in report, message="the second phase appears")


def test_report_includes_the_environment() -> None:
    """The report names the library versions the run used."""
    with RunProfile(interval=0.01) as profile:
        pass
    check(condition="torch" in profile.report(), message="torch version is reported")


def test_report_includes_device_details_when_cuda_is_available() -> None:
    """With a GPU present the report names the device."""
    torch = sys.modules["torch"]
    torch.cuda.available = True
    with RunProfile(interval=0.01) as profile, profile.phase("work"):
        pass
    report = profile.report()
    torch.cuda.available = False
    check(condition="StubGPU" in report, message="the device is named")


def test_write_json_round_trips(tmp_path: Path) -> None:
    """The JSON trace holds the phases and samples."""
    with RunProfile(interval=0.01) as profile, profile.phase("work", seed=1):
        pass
    target = tmp_path / "trace.json"
    profile.write_json(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    check(condition="phases" in payload, message="phases are written")
    check_equal(
        actual=payload["phases"][0]["name"],
        expected="work",
        message="phase name survives the round trip",
    )


def test_peaks_are_reported() -> None:
    """The peak accessors return numbers even with no samples taken."""
    with RunProfile(interval=0.01) as profile:
        pass
    check(condition=profile.peak_gpu_mb() >= 0, message="gpu peak")
    check(condition=profile.peak_rss_mb() >= 0, message="rss peak")
    check(condition=profile.peak_load() >= 0, message="load peak")


def test_sampler_records_readings_while_the_run_is_open() -> None:
    """The background sampler appends readings until the profile closes.

    The interval is tiny so the loop body actually runs; with the default it
    would never fire inside a test.
    """
    profile = RunProfile(interval=0.001)
    with profile:
        deadline = time.time() + 0.5
        while not profile.samples and time.time() < deadline:
            time.sleep(0.005)
    check(condition=len(profile.samples) > 0, message="the sampler took a reading")
    check(condition=profile.samples[0].at >= 0.0, message="the reading is timestamped")


def test_sampler_records_gpu_readings_when_cuda_is_available() -> None:
    """With a GPU present the sampler reads allocation and reservation."""
    torch = sys.modules["torch"]
    torch.cuda.available = True
    torch.cuda.allocated = 4 * 1024**2
    torch.cuda.reserved = 8 * 1024**2
    profile = RunProfile(interval=0.001)
    try:
        with profile:
            deadline = time.time() + 0.5
            while not profile.samples and time.time() < deadline:
                time.sleep(0.005)
    finally:
        torch.cuda.available = False
        torch.cuda.allocated = 0
        torch.cuda.reserved = 0
    check(condition=len(profile.samples) > 0, message="the sampler took a reading")
    check(
        condition=profile.peak_gpu_mb() > 0.0,
        message="the GPU reservation was recorded",
    )


def test_exit_without_enter_is_safe() -> None:
    """Closing a profile that never started has nothing to join."""
    profile = RunProfile(interval=0.01)
    profile.__exit__()
    check_length(sized=profile.samples, expected=0, message="no samples were taken")
