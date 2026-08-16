"""Background removal for continuous-tone renders.

Flooding inward from every border pixel, rather than from the four corners,
is the load-bearing choice here. A corner flood with a colour tolerance cannot
tell a dark object from a dark field: it either leaves a halo or eats into the
subject. Seeding every border pixel and requiring connectivity back to the
border means an interior region that merely happens to match the background
colour is never removed, because nothing connects it to the edge.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Max summed per-channel distance from the border's median colour for a pixel
# to be considered background.
DEFAULT_TOLERANCE = 60

OPAQUE = 255
TRANSPARENT = 0


def _seed_border(
    height: int, width: int
) -> tuple[NDArray[np.bool_], deque[tuple[int, int]]]:
    """Mark every border pixel as visited and queue it for flooding.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        The seen mask and the queue of border coordinates.
    """
    seen = np.zeros((height, width), bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        for y in (0, height - 1):
            if not seen[y, x]:
                seen[y, x] = True
                queue.append((y, x))
    for y in range(height):
        for x in (0, width - 1):
            if not seen[y, x]:
                seen[y, x] = True
                queue.append((y, x))
    return seen, queue


def strip_background(
    rgba: NDArray[np.uint8],
    tol: int = DEFAULT_TOLERANCE,
) -> NDArray[np.uint8]:
    """Clear the background by flooding inward from every border pixel.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.
        tol: Max summed per-channel distance from the border's median colour
            for a pixel to count as background.

    Returns:
        A copy whose background pixels have alpha 0.
    """
    height, width = rgba.shape[:2]
    rgb = rgba[:, :, :3].astype(int)
    border = np.concatenate([rgb[0, :], rgb[-1, :], rgb[:, 0], rgb[:, -1]])
    ref = np.median(border, axis=0)
    is_bg = np.abs(rgb - ref).sum(2) < tol

    keep = np.ones((height, width), bool)
    seen, queue = _seed_border(height, width)

    while queue:
        y, x = queue.popleft()
        if not is_bg[y, x]:
            continue
        keep[y, x] = False
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and not seen[ny, nx]:
                seen[ny, nx] = True
                queue.append((ny, nx))

    out = rgba.copy()
    out[:, :, 3] = np.where(keep, OPAQUE, TRANSPARENT)
    return out
