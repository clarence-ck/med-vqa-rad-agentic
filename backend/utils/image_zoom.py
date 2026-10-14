"""Crop-and-enlarge for the zoom tool: a focused re-look sees the named region at higher resolution."""
from __future__ import annotations

from io import BytesIO

from PIL import Image

MARGIN = 0.15        # extra context around the located region, as a fraction of the box size
MIN_SIDE = 0.25      # never zoom in further than a quarter of the image width/height
OUT_LONG_SIDE = 1024  # enlarged crop's longer side, in pixels


def parse_box(box) -> tuple[float, float, float, float] | None:
    """[x0, y0, x1, y1] as fractions of the image width/height (0-1; 0-100 is read as percent)."""
    try:
        values = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if len(values) != 4:
        return None
    if max(values) > 1.0:
        if max(values) > 100.0:
            return None
        values = [v / 100.0 for v in values]
    x0, y0, x1, y1 = (min(max(v, 0.0), 1.0) for v in values)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _expand(low: float, high: float) -> tuple[float, float]:
    """Add a margin, enforce the minimum size around the same centre, and keep it inside [0, 1]."""
    size = high - low
    low, high = low - size * MARGIN, high + size * MARGIN
    if high - low < MIN_SIDE:
        centre = (low + high) / 2
        low, high = centre - MIN_SIDE / 2, centre + MIN_SIDE / 2
    if low < 0:
        low, high = 0.0, high - low
    if high > 1:
        low, high = low - (high - 1), 1.0
    return max(low, 0.0), min(high, 1.0)


def zoom(image_bytes: bytes, box) -> tuple[bytes, tuple[float, float, float, float]] | None:
    """Crop the region (plus margin) and enlarge it; returns (PNG bytes, the box actually cropped) or None."""
    parsed = parse_box(box)
    if not parsed:
        return None
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except Exception:
        return None
    x0, x1 = _expand(parsed[0], parsed[2])
    y0, y1 = _expand(parsed[1], parsed[3])
    width, height = image.size
    crop = image.crop((round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)))
    if crop.mode not in ("L", "RGB"):
        crop = crop.convert("RGB")
    scale = OUT_LONG_SIDE / max(crop.size)
    if scale > 1:
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS)
    out = BytesIO()
    crop.save(out, format="PNG")
    return out.getvalue(), (round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3))
