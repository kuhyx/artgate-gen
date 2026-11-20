"""A minimal stand-in for :mod:`transformers`.

The CLIP tokenizer is touched by ``prompt_budget.token_count``; rendering
runs in stable-diffusion.cpp, which tokenizes on its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.stubs._inert import fallback

if TYPE_CHECKING:
    from collections.abc import Callable


# Every directory passed to `CLIPTokenizer.from_pretrained`, in call order.
loads: list[str] = []


class CLIPTokenizer:
    """Stands in for the CLIP tokenizer.

    Counts whitespace-delimited words plus the two sentinel markers real CLIP
    adds. That is not how BPE works, and it does not need to be: the only
    behaviour under test is that a longer prompt reports a larger number.
    """

    @classmethod
    def from_pretrained(cls, source: str) -> CLIPTokenizer:
        """Record the load and return a tokenizer.

        Args:
            source: The tokenizer directory.

        Returns:
            A new stub instance.

        """
        loads.append(source)
        return cls()

    def __call__(self, text: str) -> dict[str, list[int]]:
        """Tokenize one string.

        Args:
            text: The prompt to measure.

        Returns:
            A mapping with an ``input_ids`` list, as transformers returns.

        """
        return {"input_ids": list(range(len(text.split()) + 2))}


def __getattr__(name: str) -> Callable[..., None]:
    """Return an inert placeholder for any unexercised attribute.

    Args:
        name: The attribute being looked up.

    Returns:
        A no-op callable.

    """
    return fallback(name)
