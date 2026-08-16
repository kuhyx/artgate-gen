"""Command-line entrypoint for :mod:`repair`, called by ``run.sh``.

A separate file rather than a ``python -c`` block in the shell script: inline
program logic in another language is invisible to this repo's linters, so it
lives here where ruff can see it.

Exits 0 when a repair was written, 1 when the failure is not repairable. Saying
nothing about whether the repair *worked* is intentional -- ``run.sh`` re-gates
afterwards, and the gate is the only thing allowed to declare a pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from artgate_gen.repair import repair


def parse_codes(raw: str) -> list[str]:
    """Extract the gate codes from the artgate CLI's JSON verdict.

    Args:
        raw: The CLI's stdout, which may carry warning lines around the JSON.

    Returns:
        The failed gate codes, or an empty list if the verdict is unparseable.
    """
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            return list(json.loads(line).get("codes", []))
        except json.JSONDecodeError:
            continue
    return []


def main(argv: list[str] | None = None) -> int:
    """Repair one icon in place.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``. Passed
            explicitly by the tests so the entry point is exercisable without
            a subprocess.

    Returns:
        ``0`` when a repair was written, ``1`` when none was attempted.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("icon", type=Path, help="the failing icon")
    parser.add_argument("--spec", type=Path, required=True, help="artgate spec")
    parser.add_argument("--codes", required=True, help="the artgate CLI's JSON verdict")
    args = parser.parse_args(argv)

    codes = parse_codes(args.codes)
    return 0 if codes and repair(args.icon, args.spec, codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
