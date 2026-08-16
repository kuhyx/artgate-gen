"""Structural types for the diffusion stack.

The ML libraries are excluded from the CI check job, so mypy sees them as
``Any`` (see the override in ``pyproject.toml``). Strict mode's
``warn_return_any`` would then reject every function that returns a pipeline or
an image. Rather than weaken the return types, the parts of the pipeline this
package actually touches are declared here as a Protocol: our own call sites
stay type-checked even though the library itself is opaque.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL import Image


@runtime_checkable
class InferenceResult(Protocol):
    """The subset of a diffusers pipeline result this package reads."""

    images: Sequence[Image.Image]


@runtime_checkable
class Pipeline(Protocol):
    """The subset of ``diffusers.DiffusionPipeline`` this package uses."""

    unet: Any
    # CLIP tokenizer. Typed loosely for the same reason as `unet`: transformers
    # leaves this callable surface untyped, so pinning it would buy nothing and
    # force a suppression at the call site.
    tokenizer: Any

    def to(self, device: str) -> Pipeline:
        """Move the pipeline onto a device.

        Args:
            device: Torch device string, e.g. ``"cuda"``.

        Returns:
            The pipeline, relocated.

        """
        ...

    def set_progress_bar_config(self, *, disable: bool) -> None:
        """Enable or disable the inference progress bar.

        Args:
            disable: True to silence the bar.

        """
        ...

    def __call__(self, **kwargs: object) -> InferenceResult:
        """Run inference.

        Args:
            **kwargs: Passed through to the underlying pipeline.

        Returns:
            An object exposing ``.images``.

        """
        ...

    @classmethod
    def lora_state_dict(
        cls,
        source: str,
        **kwargs: object,
    ) -> tuple[Any, Any, Any]:
        """Load LoRA weights, network alphas and metadata.

        Args:
            source: Hub repo id or local path holding the LoRA.
            **kwargs: Passed through, e.g. ``weight_name``.

        Returns:
            The state dict, the network alphas and the metadata.

        """
        ...

    def load_lora_into_unet(self, state_dict: object, **kwargs: object) -> None:
        """Inject LoRA weights into the UNet.

        Args:
            state_dict: The LoRA tensors.
            **kwargs: Passed through, e.g. ``network_alphas``.

        """
        ...
