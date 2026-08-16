"""Compatibility shim: the server now lives in :mod:`artgate_gen.server`.

Kept at the repository root because the registered MCP entry invokes this path
directly. New callers should use the `artgate-gen-mcp` console script.
"""

from __future__ import annotations

from artgate_gen.server import main

if __name__ == "__main__":
    main()
