"""Tests for the MCP server surface.

Two things are load-bearing here. Importing the module must not start a
thread, because a server that spawns a watcher on import cannot be introspected
safely. And the gate must run in-process against the vendored implementation,
producing the same verdicts the CLI does.
"""

from __future__ import annotations

import importlib
import threading
import time
from typing import TYPE_CHECKING

import anyio
import numpy as np
from PIL import Image

from artgate_gen import generation, server
from tests.assertions import check, check_equal, check_length

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

EDGE_32 = 32
EDGE_64 = 64
EXPECTED_TOOLS = {"generate_icons", "gate_icon", "describe_icon", "model_status"}
FULL_ALPHA = 255


def _icon(tmp_path: Path, edge: int = EDGE_32, name: str = "icon.png") -> Path:
    """Write a gate-passing icon.

    Args:
        tmp_path: Pytest temporary directory.
        edge: Canvas edge length.
        name: File name.

    Returns:
        The written path.

    """
    arr = np.zeros((edge, edge, 4), dtype=np.uint8)
    inset = max(2, edge // 8)
    arr[inset:-inset, inset:-inset, :3] = (220, 60, 40)
    arr[inset:-inset, inset:-inset, 3] = FULL_ALPHA
    path = tmp_path / name
    Image.fromarray(arr, "RGBA").save(path)
    return path


def test_importing_the_server_starts_no_threads() -> None:
    """The idle watcher must not spawn at import time.

    A module-level thread made the server unsafe to import for introspection
    and left one running in every test session.
    """
    before = threading.active_count()
    importlib.reload(server)
    check_equal(
        actual=threading.active_count(),
        expected=before,
        message="importing the server spawns nothing",
    )


def test_all_four_tools_are_registered() -> None:
    """The published tool surface is unchanged."""
    tools = anyio.run(server.mcp.list_tools)
    check_equal(
        actual={tool.name for tool in tools},
        expected=EXPECTED_TOOLS,
        message="the published tools",
    )


def test_every_tool_has_a_description() -> None:
    """Tool descriptions are what an agent reads to choose between them."""
    tools = anyio.run(server.mcp.list_tools)
    for tool in tools:
        check(
            condition=bool(tool.description and tool.description.strip()),
            message=f"{tool.name} has a description",
        )


def test_gate_icon_passes_a_conforming_icon(tmp_path: Path) -> None:
    """A well-formed icon passes with no codes."""
    result = server.gate_icon(str(_icon(tmp_path)), EDGE_32)
    check(condition=result["gated"], message="the gate ran")
    check(condition=result["passed"], message="a conforming icon passes")
    check_length(sized=result["codes"], expected=0, message="no failure codes")


def test_gate_icon_reports_a_canvas_failure(tmp_path: Path) -> None:
    """A wrong-sized canvas is reported by code, matching the CLI."""
    result = server.gate_icon(str(_icon(tmp_path, EDGE_64)), EDGE_32)
    check(condition=not result["passed"], message="a wrong canvas fails")
    check(condition="CANVAS_SIZE" in result["codes"], message="the code is reported")


def test_gate_icon_rejects_an_unknown_size(tmp_path: Path) -> None:
    """A size with no spec is reported rather than silently passing."""
    result = server.gate_icon(str(_icon(tmp_path)), 17)
    check(condition=not result["gated"], message="no spec means no verdict")
    check(condition="no spec" in result["reason"], message="the reason is given")


def test_describe_icon_measures_without_a_model(tmp_path: Path) -> None:
    """Describing an icon needs no GPU and no pipeline."""
    result = server.describe_icon(str(_icon(tmp_path)))
    check_equal(actual=result["size"], expected=[EDGE_32, EDGE_32], message="size")
    check(condition=result["opaque_fraction"] > 0, message="the subject is opaque")
    check(condition=result["distinct_colours"] >= 1, message="colours are counted")


def test_model_status_reports_an_unloaded_pipeline() -> None:
    """With nothing resident the status says so."""
    generation.state.pipe = None
    generation.state.last_used = 0.0
    status = server.model_status()
    check(condition=not status["loaded"], message="nothing is resident")
    check(condition=status["idle_seconds"] is None, message="no idle time yet")


def test_model_status_reports_a_loaded_pipeline() -> None:
    """A resident pipeline reports its idle time."""
    generation.state.pipe = object()
    generation.state.last_used = time.time()
    try:
        status = server.model_status()
        check(condition=status["loaded"], message="the pipeline is resident")
        check(condition=status["idle_seconds"] is not None, message="idle is reported")
    finally:
        generation.state.pipe = None
        generation.state.last_used = 0.0


def test_generate_icons_writes_gates_and_summarises(tmp_path: Path) -> None:
    """A generation run returns per-icon verdicts and a summary."""
    result = server.generate_icons(
        str(tmp_path),
        {"axe": "a stone axe"},
        size=EDGE_32,
    )
    check_length(sized=result["icons"], expected=1, message="one icon was produced")
    check(condition="passed the" in result["summary"], message="a summary is returned")
    check(condition=(tmp_path / "axe.png").exists(), message="the icon was written")
    generation.state.pipe = None


def test_generate_icons_rejects_an_unsupported_size(tmp_path: Path) -> None:
    """An unsupported size returns an explanation, not a KeyError.

    This crosses the MCP boundary, where an exception is an opaque tool
    failure; naming the supported sizes lets the caller fix the call.
    """
    unsupported = 48
    result = server.generate_icons(
        str(tmp_path),
        {"axe": "a stone axe"},
        size=unsupported,
    )
    check_length(sized=result["icons"], expected=0, message="nothing was produced")
    check(
        condition="no target for size" in result["summary"],
        message="the summary explains the refusal",
    )


def test_generate_icons_keeps_raw_renders(tmp_path: Path) -> None:
    """The full-resolution render is kept when asked."""
    server.generate_icons(
        str(tmp_path),
        {"axe": "a stone axe"},
        size=EDGE_32,
        keep_raw=True,
    )
    check(
        condition=(tmp_path / "raw" / "axe.png").exists(),
        message="the raw render was kept",
    )
    generation.state.pipe = None


def test_generate_icons_suffixes_multiple_candidates(tmp_path: Path) -> None:
    """Each candidate gets its own seed and file name."""
    result = server.generate_icons(
        str(tmp_path),
        {"axe": "a stone axe"},
        size=EDGE_32,
        candidates=2,
    )
    check_length(sized=result["icons"], expected=2, message="two candidates")
    seeds = {icon["seed"] for icon in result["icons"]}
    check_length(sized=seeds, expected=2, message="each candidate has its own seed")
    generation.state.pipe = None


def test_pipeline_is_loaded_once_and_reused(tmp_path: Path) -> None:
    """The pipeline is cached between calls rather than reloaded per icon."""
    generation.state.pipe = None
    server.generate_icons(str(tmp_path), {"axe": "an axe"}, size=EDGE_32)
    first = generation.state.pipe
    server.generate_icons(str(tmp_path), {"bow": "a bow"}, size=EDGE_32)
    check(condition=generation.state.pipe is first, message="the pipeline was reused")
    generation.state.pipe = None


def test_main_starts_the_watcher_and_serves(monkeypatch: pytest.MonkeyPatch) -> None:
    """The entry point starts the idle watcher before serving."""
    served = []
    monkeypatch.setattr(server.mcp, "run", lambda: served.append(True))
    server.main()
    check_length(sized=served, expected=1, message="the server was started")
