"""Bilder laden/speichern. Intern: float32, Form (H, W) oder (H, W, 3), RGB, Wertebereich 0..1."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

RAW_EXT = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".pef"}
FITS_EXT = {".fit", ".fits", ".fts"}
STD_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
ALL_EXT = RAW_EXT | FITS_EXT | STD_EXT


def load_image(path: str | Path) -> np.ndarray:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in FITS_EXT:
        img = _load_fits(p)
    elif ext in RAW_EXT:
        img = _load_raw(p)
    else:
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if raw is None:
            raise ValueError(f"Kann Bild nicht lesen: {p}")
        if raw.ndim == 3:
            raw = raw[..., :3][..., ::-1]  # BGR(A) -> RGB
        img = _to_unit_float(raw)
    return np.ascontiguousarray(img, dtype=np.float32)


def _to_unit_float(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    return a.astype(np.float32)


def _load_fits(p: Path) -> np.ndarray:
    from astropy.io import fits

    with fits.open(p) as hdul:
        data = next((h.data for h in hdul if h.data is not None and h.data.ndim >= 2), None)
    if data is None:
        raise ValueError(f"Keine Bilddaten in {p}")
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 3:  # (3, H, W) -> (H, W, 3)
        data = np.moveaxis(data, 0, -1) if data.shape[0] in (3, 4) else data[0]
        data = data[..., :3] if data.ndim == 3 else data
    lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
    if hi > 1.5 or lo < 0:  # auf 0..1 normieren
        data = (data - lo) / max(hi - lo, 1e-12)
    return np.nan_to_num(data)


def _load_raw(p: Path) -> np.ndarray:
    import rawpy

    with rawpy.imread(str(p)) as r:
        rgb = r.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True)
    return rgb.astype(np.float32) / 65535.0


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def to_display(img: np.ndarray, stretch: bool = True) -> np.ndarray:
    """uint8 RGB für die Vorschau; optional automatischer Stretch (Sterne sind sonst unsichtbar)."""
    a = img if img.ndim == 3 else np.repeat(img[..., None], 3, axis=2)
    if stretch:
        lo = float(np.percentile(a, 1.0))
        hi = float(np.percentile(a, 99.8))
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        a = np.clip((a - lo) / (hi - lo), 0, 1) ** 0.5
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def save_image(path: str | Path, img: np.ndarray) -> None:
    """Format nach Endung: .tif = 32-bit float, .png/.jpg = 16/8-bit, .fits = float32."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".tif", ".tiff"):
        import tifffile

        tifffile.imwrite(str(p), img.astype(np.float32), photometric="rgb" if img.ndim == 3 else "minisblack")
    elif ext in FITS_EXT:
        from astropy.io import fits

        data = np.moveaxis(img, -1, 0) if img.ndim == 3 else img
        fits.PrimaryHDU(data.astype(np.float32)).writeto(p, overwrite=True)
    else:
        bgr = img[..., ::-1] if img.ndim == 3 else img
        if ext == ".png":
            out = (np.clip(bgr, 0, 1) * 65535 + 0.5).astype(np.uint16)
        else:
            out = (np.clip(bgr, 0, 1) * 255 + 0.5).astype(np.uint8)
        if not cv2.imwrite(str(p), out):
            raise OSError(f"Konnte nicht speichern: {p}")
