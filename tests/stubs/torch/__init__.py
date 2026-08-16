"""A minimal stand-in for :mod:`torch`.

The real package is a multi-gigabyte install that the CI check job
deliberately does not carry, so the tests supply exactly the surface this
project touches. Everything here is inert: no tensors, no CUDA, no allocation.

``cuda.is_available`` is backed by a module-level flag rather than a constant
because both of its branches are live in the profiler, and a test needs to
drive each one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.stubs._inert import fallback

if TYPE_CHECKING:
    from collections.abc import Callable

__version__ = "2.11.0+stub"

float16 = "float16"


class _Version:
    """Stands in for ``torch.version``."""

    cuda = "12.8"


version = _Version()


class Generator:
    """A seedable RNG handle, as accepted by the diffusion pipeline."""

    def __init__(self, device: str = "cpu") -> None:
        """Record the requested device.

        Args:
            device: Torch device string.

        """
        self.device = device
        self.seed = 0

    def manual_seed(self, seed: int) -> Generator:
        """Set the seed.

        Args:
            seed: The seed value.

        Returns:
            This generator.

        """
        self.seed = seed
        return self


class _DeviceProperties:
    """The subset of device properties the profiler reports."""

    name = "StubGPU"
    total_memory = 16 * 1024**3
    major = 8
    minor = 6
    multi_processor_count = 84


class _Cuda:
    """Stands in for ``torch.cuda``; every call is inert."""

    available = False
    allocated = 0
    reserved = 0

    def is_available(self) -> bool:
        """Report whether a GPU is present.

        Returns:
            The current value of the module-level flag.

        """
        return self.available

    def synchronize(self) -> None:
        """Wait for pending work; a no-op here."""

    def memory_allocated(self) -> int:
        """Report allocated bytes.

        Returns:
            The stubbed allocation.

        """
        return self.allocated

    def memory_reserved(self) -> int:
        """Report reserved bytes.

        Returns:
            The stubbed reservation.

        """
        return self.reserved

    def empty_cache(self) -> None:
        """Release cached blocks; a no-op here."""

    def get_device_name(self, index: int = 0) -> str:
        """Report the device name.

        Args:
            index: Device ordinal, ignored.

        Returns:
            A fixed stub name.

        """
        del index
        return _DeviceProperties.name

    def get_device_properties(self, index: int) -> _DeviceProperties:
        """Report device properties.

        Args:
            index: Device ordinal, ignored.

        Returns:
            The stub properties.

        """
        del index
        return _DeviceProperties()


cuda = _Cuda()


class _Cudnn:
    """Stands in for ``torch.backends.cudnn``."""

    def version(self) -> int:
        """Report the cuDNN version.

        Returns:
            A fixed stub version.

        """
        return 90000


class _Backends:
    """Stands in for ``torch.backends``."""

    cudnn = _Cudnn()


backends = _Backends()


def __getattr__(name: str) -> Callable[..., None]:
    """Return an inert placeholder for any unexercised attribute.

    Args:
        name: The attribute being looked up.

    Returns:
        A no-op callable.

    """
    return fallback(name)
