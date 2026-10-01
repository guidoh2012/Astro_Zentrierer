"""Stacking wie in Siril & Co.: Mittelwert, Median, Sigma-Clipping, Winsorized; optional Gewichtung nach Qualität."""
from __future__ import annotations

from typing import Callable, Iterable

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numpy\..*")

METHODS = ["Mittelwert", "Median", "Sigma-Clipping", "Winsorized Sigma", "Maximum"]


def _normalize(frames: np.ndarray) -> np.ndarray:
    """Additive+multiplikative Normierung auf den ersten Frame (Median / Streuung), wie Siril 'add+scale'."""
    ref_med = np.median(frames[0])
    ref_sd = np.std(frames[0]) + 1e-12
    out = np.empty_like(frames)
    for i, f in enumerate(frames):
        out[i] = (f - np.median(f)) * (ref_sd / (np.std(f) + 1e-12)) + ref_med
    return out


def stack(frames: Iterable[np.ndarray] | np.ndarray, method: str = "Sigma-Clipping",
          sigma_low: float = 3.0, sigma_high: float = 3.0, iterations: int = 5,
          normalize: bool = False, weights: np.ndarray | None = None,
          valid: np.ndarray | None = None) -> np.ndarray:
    """frames: (N, H, W[, C]). valid: optionale Bool-Maske gleicher Form (False = Pixel ignorieren, z. B. Ränder)."""
    arr = np.asarray(list(frames) if not isinstance(frames, np.ndarray) else frames, dtype=np.float32)
    if arr.shape[0] == 0:
        raise ValueError("Keine Bilder zum Stacken")
    if arr.shape[0] == 1:
        return arr[0].copy()
    if normalize:
        arr = _normalize(arr)
    mask = np.ones(arr.shape, bool) if valid is None else valid.copy()
    w = None if weights is None else np.asarray(weights, np.float32).reshape((-1,) + (1,) * (arr.ndim - 1))

    if method == "Median":
        m = np.where(mask, arr, np.nan)
        return np.nan_to_num(np.nanmedian(m, axis=0)).astype(np.float32)
    if method == "Maximum":
        return np.where(mask, arr, -np.inf).max(axis=0).clip(0).astype(np.float32)

    if method in ("Sigma-Clipping", "Winsorized Sigma"):
        for _ in range(iterations):
            m = np.where(mask, arr, np.nan)
            med = np.nanmedian(m, axis=0)
            sd = np.nanstd(m, axis=0)
            if method == "Winsorized Sigma":  # Ausreißer auf Grenze kappen, bevor sd bestimmt wird
                cl = np.clip(m, med - 1.5 * sd, med + 1.5 * sd)
                sd = np.nanstd(cl, axis=0) * 1.134
            lo, hi = med - sigma_low * sd, med + sigma_high * sd
            new = mask & (arr >= lo) & (arr <= hi)
            if (new.sum(0) == 0).any():  # nie alle Pixel wegwerfen
                new |= (new.sum(0) == 0) & mask
            if (new == mask).all():
                break
            mask = new
    elif method != "Mittelwert":
        raise ValueError(f"Unbekannte Methode: {method}")

    ww = mask.astype(np.float32) if w is None else mask * w
    den = ww.sum(axis=0)
    return np.where(den > 0, (arr * ww).sum(axis=0) / np.maximum(den, 1e-12), 0).astype(np.float32)


def quality(img: np.ndarray) -> float:
    """Schärfemaß (Varianz des Laplace) für Gewichtung / Aussortieren schlechter Frames."""
    import cv2

    from .imageio import to_gray
    g = to_gray(img).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())
