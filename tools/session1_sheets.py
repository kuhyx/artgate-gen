"""Build the contact sheets the session-1 verdicts are read from.

Throwaway driver, not package code. Two sheets:

``portraits`` answers both portrait questions in one image. Rows are grouped by
SEED, not by expression, because identity is judged across expressions at a
fixed seed -- a fixed seed fixes the initial noise, so if the three faces in a
row are not the same person, nothing downstream will make them one. Columns are
64/128/256, each nearest-neighbour upscaled to a common cell so the comparison
is about what the render resolves rather than about how big the file is.

``strips`` answers the banding question: the finished 256x64 strips at 3x, where
a 32-colour sky gradient either shows bands or does not.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from artgate_gen.console import emit

ROOT = Path(__file__).resolve().parent.parent
PORTRAITS = ROOT / "art" / "portrait-probe"
FINISHED = ROOT / "art" / "backdrops-256x64"
SHEETS = ROOT / "art" / "contact-sheets"

CELL = 256
PAD = 8
SIZES = (64, 128, 256)
EXPRESSIONS = ("neutral", "smiling", "angry")
SEEDS = (0, 1, 2)


def _nearest(path: Path, cell: int) -> Image.Image:
    """Load a PNG and nearest-neighbour scale it to a square cell.

    Args:
        path: Image to load.
        cell: Edge length of the output square.

    Returns:
        The upscaled image.
    """
    img = Image.open(path).convert("RGB")
    return img.resize((cell, cell), Image.Resampling.NEAREST)


def build_portraits() -> Path:
    """Write the 9x3 portrait sheet, rows grouped by seed.

    Returns:
        The written path.
    """
    rows = [(seed, expr) for seed in SEEDS for expr in EXPRESSIONS]
    width = len(SIZES) * CELL + (len(SIZES) + 1) * PAD
    height = len(rows) * CELL + (len(rows) + 1) * PAD
    sheet = Image.new("RGB", (width, height), (24, 24, 28))
    for r, (seed, expr) in enumerate(rows):
        for c, size in enumerate(SIZES):
            src = PORTRAITS / f"{expr}_s{seed}_{size}.png"
            sheet.paste(
                _nearest(src, CELL),
                (PAD + c * (CELL + PAD), PAD + r * (CELL + PAD)),
            )
    out = SHEETS / "SESSION1_PORTRAITS.png"
    sheet.save(out)
    return out


def build_strips() -> Path:
    """Write the finished-strip sheet at 3x for the banding verdict.

    Returns:
        The written path.
    """
    paths = sorted(FINISHED.glob("*.png"))
    scale = 3
    cell_w, cell_h = 256 * scale, 64 * scale
    width = cell_w + 2 * PAD
    height = len(paths) * cell_h + (len(paths) + 1) * PAD
    sheet = Image.new("RGB", (width, height), (24, 24, 28))
    for r, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        img = img.resize((cell_w, cell_h), Image.Resampling.NEAREST)
        sheet.paste(img, (PAD, PAD + r * (cell_h + PAD)))
    out = SHEETS / "SESSION1_STRIPS.png"
    sheet.save(out)
    return out


def main() -> int:
    """Build both sheets.

    Returns:
        0 on success.
    """
    SHEETS.mkdir(parents=True, exist_ok=True)
    for path in (build_portraits(), build_strips()):
        emit(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
