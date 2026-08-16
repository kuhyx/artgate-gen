"""Reason-code vocabulary for art gates.

Single source of truth. Every gate failure, every rejects-store record and
every contact-sheet annotation keys off these constants, so that "feed
rejections back as negative examples" is a queryable fact next session rather
than a good intention.

Exit codes are deliberately three-valued: conflating a harness crash with an
art rejection would silently convert our own bugs into art failures and
poison both the rejects store and the bake-off metrics.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

# All applicable gates passed.
EXIT_PASS: Final = 0

# The asset is well-formed but failed at least one gate.
EXIT_GATE_FAILURE: Final = 1

# Spec malformed or image unreadable -- a harness fault, never an art verdict.
EXIT_HARNESS_ERROR: Final = 2


class Code(StrEnum):
    """Machine-readable gate failure codes.

    Values are SCREAMING_SNAKE so they survive a round trip through JSONL and
    remain greppable with ``jq`` without a lookup table.
    """

    OFF_PALETTE = "OFF_PALETTE"
    TOO_MANY_COLORS = "TOO_MANY_COLORS"
    NO_OPAQUE_PIXELS = "NO_OPAQUE_PIXELS"
    ALPHA_NOT_BINARY = "ALPHA_NOT_BINARY"
    CANVAS_SIZE = "CANVAS_SIZE"
    MARGIN_TOO_SMALL = "MARGIN_TOO_SMALL"
    EMPTY_CANVAS = "EMPTY_CANVAS"
    SCALE_NOT_INVARIANT = "SCALE_NOT_INVARIANT"
    OUTLINE_INCONSISTENT = "OUTLINE_INCONSISTENT"
    SILHOUETTE_LOW_CONTRAST = "SILHOUETTE_LOW_CONTRAST"
    SEAM_ENERGY = "SEAM_ENERGY"
    QUADRANT_WEIGHT = "QUADRANT_WEIGHT"
    LOOP_POPS = "LOOP_POPS"
    STATIC_ANIMATION = "STATIC_ANIMATION"
    LICENSE_NOT_PERMISSIVE = "LICENSE_NOT_PERMISSIVE"
    LICENSE_AMBIGUOUS = "LICENSE_AMBIGUOUS"

    def __str__(self) -> str:
        """Return the bare code, so f-strings do not leak ``Code.`` prefixes."""
        return self.value


# Codes a spec may DECLARE. A code outside this set is either a result-only
# code (emitted by a gate, never requested) or not yet implemented.
#
# This set exists because declaring a gate must never be enough to make it run:
# a spec that names an unimplemented gate would parse cleanly, run nothing, and
# report a pass -- indistinguishable from art that genuinely passed. Keeping the
# declarable set explicit forces `load_spec` to fail closed instead.
DECLARABLE: Final = frozenset(
    {
        Code.ALPHA_NOT_BINARY,
        Code.TOO_MANY_COLORS,
        Code.OFF_PALETTE,
        Code.CANVAS_SIZE,
        Code.SCALE_NOT_INVARIANT,
        Code.SILHOUETTE_LOW_CONTRAST,
        Code.MARGIN_TOO_SMALL,
    }
)

# Codes a gate may EMIT but a spec may not request. ``NO_OPAQUE_PIXELS`` is the
# fail-closed result of the colour and palette gates, not a gate in its own right.
RESULT_ONLY: Final = frozenset(
    {
        Code.NO_OPAQUE_PIXELS,
        Code.EMPTY_CANVAS,
    }
)
