"""64-bit average-hash for the episode first-frame.

Phase 1 ships average-hash (aHash) using Pillow + numpy (both already base
deps). aHash is less robust than DCT-based pHash for compressed / cropped
images, but it's trivially implementable, fast (<5ms per frame), and
catches identical / very-similar scenes — which is the dominant dedup
case for "same demo recorded twice."

Phase 1.5 swaps to dHash or pHash if the false-negative rate climbs.
That's a 1-line change here; downstream code consumes the hex hash string
identically regardless of algorithm.

Algorithm:
    1. Resize image to 8×8 grayscale
    2. Compute mean pixel value
    3. For each pixel: 1 bit if pixel ≥ mean, else 0
    4. Pack 64 bits → 16-char hex string
"""
from __future__ import annotations

import hashlib
import io
from typing import Any


def compute_average_hash(image_bytes: bytes) -> str:
    """Compute a 64-bit average-hash from image bytes. Returns 16-char hex.

    Raises ValueError on unloadable input. Returns a stable hash for
    visually-identical images (the same byte-stream always produces the
    same hex).
    """
    if not image_bytes:
        raise ValueError("image_bytes is empty")
    try:
        from PIL import Image
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Pillow + numpy required for compute_average_hash"
        ) from exc

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:  # noqa: BLE001 — Pillow raises a variety of types
        raise ValueError(f"failed to decode image: {exc}") from exc

    # Resize to 8×8 grayscale.
    small = img.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.uint8).flatten()
    if pixels.size != 64:
        raise ValueError(f"resized image had {pixels.size} pixels, expected 64")
    mean = float(pixels.mean())
    bits = (pixels >= mean).astype(np.uint8)
    # Pack 64 bits into a 64-bit integer, MSB first.
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:016x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Hamming distance between two hex hash strings.

    Returns the number of differing bits. Mismatched-length hashes return
    the max distance for the longer string (treats as fully different).
    """
    if len(hash_a) != len(hash_b):
        return max(len(hash_a), len(hash_b)) * 4
    try:
        a = int(hash_a, 16)
        b = int(hash_b, 16)
    except ValueError:
        return len(hash_a) * 4
    return bin(a ^ b).count("1")


def fingerprint_bytes(image_bytes: bytes) -> str:
    """SHA-256 hex of the raw image bytes. Useful as a frozen-frame detector
    when the same exact frame appears in multiple episodes (camera frozen,
    not just visually similar). Distinct from compute_average_hash, which is
    a perceptual hash designed to be similar for visually-similar images."""
    return hashlib.sha256(image_bytes).hexdigest()


__all__ = [
    "compute_average_hash",
    "fingerprint_bytes",
    "hamming_distance",
]
