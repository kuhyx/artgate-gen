"""Test configuration: install the library stubs before anything imports them.

This runs at module scope rather than in a fixture because pytest imports the
rootdir conftest before any test module, and ``artgate_gen.pipeline`` imports
torch at module scope. By the time a fixture ran it would already be too late.

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

# Only the libraries the CI check job does not install. numpy and Pillow stay
# real: the finisher and the gate do genuine array work, and faking them would
# make the coverage meaningless.
STUBBED = (
    "torch",
    "diffusers",
    # A real submodule path, because the loader does
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


@pytest.fixture(autouse=True)
def _reset_stub_state() -> Iterator[None]:
    """Clear mutable stub state between tests.

    The stubs record calls and carry flags that steer their branches; leaking
    that between tests would make results depend on execution order.

    Yields:
        Control to the test.

    """
    hub = sys.modules["huggingface_hub"]
    torch = sys.modules["torch"]
    hub.reset()
    torch.cuda.available = False
    yield
    hub.reset()
    torch.cuda.available = False
