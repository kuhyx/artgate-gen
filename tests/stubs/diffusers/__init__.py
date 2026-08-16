"""A minimal stand-in for :mod:`diffusers`.

Only the two loaders and the pipeline surface this project touches are
present. ``DiffusionPipeline`` returns an object that satisfies the
``Pipeline`` protocol and produces a real, tiny PIL image, so the finishing
path downstream runs for real rather than against another fake.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from PIL import Image

from tests.stubs._inert import fallback

# Size of the image the stub pipeline renders. Small on purpose: the finisher
# and the gate run for real on it, so every pixel costs test time.
RENDER_SIZE = 64


class _UnetConfig:
    """Stands in for ``pipe.unet.config``."""

    sample_size = 128


class _Unet:
    """Stands in for the pipeline's UNet."""

    config = _UnetConfig()


class _Result:
    """The object returned by calling the pipeline."""

    def __init__(self, images: list[Image.Image]) -> None:
        """Store the rendered images.

        Args:
            images: The rendered images.

        """
        self.images = images


class AutoencoderKL:
    """Stands in for the VAE loader."""

    @classmethod
    def from_pretrained(cls, *args: object, **kwargs: object) -> AutoencoderKL:
        """Return an inert VAE.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            A new stub instance.

        """
        del args, kwargs
        return cls()


class DiffusionPipeline:
    """Stands in for the SDXL pipeline."""

    def __init__(self) -> None:
        """Record call history so tests can assert on it."""
        self.unet = _Unet()
        self.device = "cpu"
        self.progress_disabled = False
        self.lora_loaded = False

    @classmethod
    def from_pretrained(cls, *args: object, **kwargs: object) -> DiffusionPipeline:
        """Return an inert pipeline.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            A new stub instance.

        """
        del args, kwargs
        return cls()

    @classmethod
    def lora_state_dict(
        cls,
        source: str,
        **kwargs: object,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        """Return an empty LoRA state dict triple.

        Args:
            source: Ignored.
            **kwargs: Ignored.

        Returns:
            Empty state dict, alphas and metadata.

        """
        del source, kwargs
        return {}, {}, {}

    def load_lora_into_unet(self, state_dict: object, **kwargs: object) -> None:
        """Record that the LoRA was applied.

        Args:
            state_dict: Ignored.
            **kwargs: Ignored.

        """
        del state_dict, kwargs
        self.lora_loaded = True

    def to(self, device: str) -> DiffusionPipeline:
        """Record the target device.

        Args:
            device: Torch device string.

        Returns:
            This pipeline.

        """
        self.device = device
        return self

    def set_progress_bar_config(self, *, disable: bool) -> None:
        """Record the progress bar setting.

        Args:
            disable: Whether the bar is silenced.

        """
        self.progress_disabled = disable

    def __call__(self, **kwargs: object) -> _Result:
        """Render one inert image at the requested size.

        ``width``/``height`` are honoured rather than discarded, so a
        non-square target is distinguishable from a square one in tests. Both
        default to ``RENDER_SIZE``, leaving callers that omit them unaffected.

        Args:
            **kwargs: Rendering arguments; only ``width``/``height`` are read.

        Returns:
            A result holding one small RGB image.

        """
        width = kwargs.get("width", RENDER_SIZE)
        height = kwargs.get("height", RENDER_SIZE)
        size = (
            width if isinstance(width, int) else RENDER_SIZE,
            height if isinstance(height, int) else RENDER_SIZE,
        )
        image = Image.new("RGB", size, (40, 60, 90))
        return _Result([image])


def __getattr__(name: str) -> Callable[..., None]:
    """Return an inert placeholder for any unexercised attribute.

    Args:
        name: The attribute being looked up.

    Returns:
        A no-op callable.

    """
    return fallback(name)
