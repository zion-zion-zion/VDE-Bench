"""Data-generation utilities shared across scripts."""

from __future__ import annotations

import os
from typing import Tuple


def poly2bbox(poly) -> Tuple[float, float, float, float]:
    """Convert an 8-number polygon (x1,y1,x2,y2,x3,y3,x4,y4) to (L, U, R, D).

    The polygon is assumed to be (approximately) axis-aligned, so we simply
    take the min/max of the first two corners' x-coordinates and the first
    and last corners' y-coordinates.  Returned values are floats; cast to
    int in the caller when using them as pixel indices.
    """
    L = poly[0]
    U = poly[1]
    R = poly[2]
    D = poly[5]
    L, R = min(L, R), max(L, R)
    U, D = min(U, D), max(U, D)
    return L, U, R, D


def ensure_dir(path: str) -> str:
    """Create a directory if it does not exist and return it."""
    os.makedirs(path, exist_ok=True)
    return path


__all__ = ["poly2bbox", "ensure_dir"]
