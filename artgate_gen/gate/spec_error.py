"""`SpecError`, split out for both `spec.py` and `spec_fields.py`.

Both modules raise it; giving it its own file avoids either importing the
other just for this one exception type.
"""

from __future__ import annotations


class SpecError(ValueError):
    """Raised when a spec file is missing, malformed, or self-contradictory.

    Distinct from a gate failure: this maps to the harness-error exit code,
    never to an art verdict.
    """
