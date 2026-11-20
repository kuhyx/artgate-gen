"""Test configuration: library stubs plus a fake ``sd-cli`` on every test.

The stubs install at module scope rather than in a fixture because pytest
imports the rootdir conftest before any test module, so a fixture would run
after the first ``import huggingface_hub`` had already happened.

The stubs are real packages under ``tests/stubs`` rather than hand-built
``ModuleType`` objects, so they are linted, type-checked and diffable like any
other source in the repo.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest

STUB_ROOT = Path(__file__).parent / "stubs"
FAKE_SD_CLI = Path(__file__).parent / "fake_sd_cli.py"

# Only the libraries the CI check job does not install. numpy and Pillow stay
# real: the finisher and the gate do genuine array work, and faking them would
# make the coverage meaningless.
STUBBED = (
    # torch and diffusers only back the opt-in SD3.5 backend (the `sd35`
    # extra); the repo venv never installs them.
    "torch",
    "diffusers",
    # A real submodule path, because sd35 does
    # `from diffusers.utils import logging`.
    "diffusers.utils",
    "transformers",
    "huggingface_hub",
    # Real submodule paths, because fetch.py does
    # `from huggingface_hub.utils.tqdm import disable_progress_bars`.
    "huggingface_hub.utils",
    "huggingface_hub.utils.tqdm",
)


def _install_stubs() -> None:
    """Import the stub packages under the names they shadow."""
    if str(STUB_ROOT) not in sys.path:
        sys.path.insert(0, str(STUB_ROOT))
    for name in STUBBED:
        if name not in sys.modules:
            sys.modules[name] = importlib.import_module(f"tests.stubs.{name}")


_install_stubs()


@pytest.fixture(scope="session")
def fake_sd_cli(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write an executable wrapper that runs the fake binary.

    A wrapper rather than a shebang on the fake itself, so the fake runs under
    this interpreter (which has Pillow) and never under whatever ``python3``
    is first on PATH.

    Returns:
        Path to the executable.
    """
    wrapper = tmp_path_factory.mktemp("bin") / "sd-cli"
    wrapper.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_SD_CLI}" "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


@pytest.fixture(autouse=True)
def _reset_stub_state(
    fake_sd_cli: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Point the pipeline at the fake binary and clear stub state.

    The stubs record calls and carry flags that steer their branches; leaking
    that between tests would make results depend on execution order.

    Yields:
        Control to the test.

    """
    # Imported here, not at module scope: the pipeline imports huggingface_hub,
    # which must resolve to the stub installed above, not the real library.
    pipeline = importlib.import_module("artgate_gen.pipeline")
    monkeypatch.setenv(pipeline.SD_CLI_ENV, str(fake_sd_cli))
    monkeypatch.delenv("FAKE_SD_MODE", raising=False)
    monkeypatch.delenv("FAKE_SD_ARGV", raising=False)
    hub = sys.modules["huggingface_hub"]
    hub.reset()
    yield
    hub.reset()
