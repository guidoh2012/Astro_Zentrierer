"""Objekt in allen Bildern finden und zentrieren."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .imageio import to_gray


@dataclass
class Match:
    x: float          # Objektposition (Pixel, Subpixel)
    y: float
    score: float      # 0..1, Qualität der Zuordnung


def _prep(img: np.ndarray) -> np.ndarray:
    """Grau, hintergrundfrei, leicht geglättet, auf 0..1 gestreckt -> robust für Template-Matching."""
    g = to_gray(img).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    g = g - np.median(g)
    hi = float(np.percentile(g, 99.9))
    return np.clip(g / hi, 0, 1.5).astype(np.float32) if hi > 1e-9 else g


def refine_centroid(img: np.ndarray, x: float, y: float, radius: int) -> tuple[float, float]:
    """Schwerpunkt des zusammenhängenden Objekts (Mond-Scheibe / Stern) an (x, y).

    Nur der Blob, der die Startposition enthält, zählt – Nachbarsterne bleiben außen vor.
    """
    g = to_gray(img).astype(np.float32)
    h, w = g.shape
    r = max(int(radius), 4)
    for _ in range(6):
        x0, x1 = max(int(round(x)) - r, 0), min(int(round(x)) + r + 1, w)
        y0, y1 = max(int(round(y)) - r, 0), min(int(round(y)) + r + 1, h)
        patch = cv2.GaussianBlur(g[y0:y1, x0:x1], (0, 0), 1.0)
        if patch.size == 0:
            break
        bg = float(np.percentile(patch, 20))
        cx, cy = int(round(x)) - x0, int(round(y)) - y0
        k = max(2, r // 4)  # Spitzenwert nahe der Markierung
        near = patch[max(cy - k, 0):cy + k + 1, max(cx - k, 0):cx + k + 1]
        peak = float(near.max())
        if peak - bg < 1e-6:
            break
        mask = (patch > bg + 0.5 * (peak - bg)).astype(np.uint8)
        n, lab = cv2.connectedComponents(mask)
        yy0, xx0 = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
        circ = (yy0 - cy) ** 2 + (xx0 - cx) ** 2 <= k * k
        py, px = np.unravel_index(np.argmax(np.where(circ, patch, -np.inf)), patch.shape)
        l = lab[py, px]
        if l == 0:
            break
        wgt = np.where(lab == l, patch - bg, 0).clip(0)
        s = float(wgt.sum())
        yy, xx = np.mgrid[y0:y1, x0:x1]
        nx, ny = float((wgt * xx).sum() / s), float((wgt * yy).sum() / s)
        done = abs(nx - x) < 0.01 and abs(ny - y) < 0.01
        x, y = nx, ny
        if done:
            break
    return x, y


def make_template(ref: np.ndarray, x: float, y: float, half: int) -> np.ndarray:
    p = _prep(ref)
    h, w = p.shape
    x0, y0 = int(round(x)) - half, int(round(y)) - half
    if x0 < 0 or y0 < 0 or x0 + 2 * half + 1 > w or y0 + 2 * half + 1 > h:
        raise ValueError("Markierung liegt zu nah am Bildrand")
    return p[y0:y0 + 2 * half + 1, x0:x0 + 2 * half + 1].copy()


def find_object(img: np.ndarray, tmpl: np.ndarray, hint: tuple[float, float] | None = None,
                search_radius: int | None = None, centroid_radius: int | None = None) -> Match:
    """Sucht das Template (NCC). `hint`+`search_radius` begrenzen den Suchbereich (schneller, robuster)."""
    p = _prep(img)
    th, tw = tmpl.shape
    ox = oy = 0
    if hint is not None and search_radius:
        hx, hy = int(round(hint[0])), int(round(hint[1]))
        x0, x1 = max(hx - search_radius - tw // 2, 0), min(hx + search_radius + tw // 2 + 1, p.shape[1])
        y0, y1 = max(hy - search_radius - th // 2, 0), min(hy + search_radius + th // 2 + 1, p.shape[0])
        if x1 - x0 > tw and y1 - y0 > th:
            p, ox, oy = p[y0:y1, x0:x1], x0, y0
    res = cv2.matchTemplate(p, tmpl, cv2.TM_CCOEFF_NORMED)
    # NCC ist in (fast) konstanten Bereichen numerisch instabil -> dort verwerfen
    mean = cv2.boxFilter(p, -1, (tw, th), normalize=True, borderType=cv2.BORDER_CONSTANT)
    sq = cv2.boxFilter(p * p, -1, (tw, th), normalize=True, borderType=cv2.BORDER_CONSTANT)
    std = np.sqrt(np.clip(sq - mean * mean, 0, None))[th // 2:th // 2 + res.shape[0], tw // 2:tw // 2 + res.shape[1]]
    res = np.where(std > 0.05 * float(tmpl.std()), res, -1.0).astype(np.float32)
    _, score, _, loc = cv2.minMaxLoc(res)
    dx = dy = 0.0
    lx, ly = loc
    if 0 < lx < res.shape[1] - 1 and 0 < ly < res.shape[0] - 1:  # Parabel-Subpixel
        def para(a, b, c):
            d = a - 2 * b + c
            return 0.0 if abs(d) < 1e-12 else 0.5 * (a - c) / d
        dx = para(res[ly, lx - 1], res[ly, lx], res[ly, lx + 1])
        dy = para(res[ly - 1, lx], res[ly, lx], res[ly + 1, lx])
    x = ox + lx + dx + (tw - 1) / 2
    y = oy + ly + dy + (th - 1) / 2
    if centroid_radius:
        x, y = refine_centroid(img, x, y, centroid_radius)
    return Match(x, y, float(score))


def shift_to_center(img: np.ndarray, x: float, y: float, out_size: tuple[int, int] | None = None) -> np.ndarray:
    """Verschiebt das Bild so, dass (x, y) exakt in der Mitte der Ausgabe (w, h) liegt. Leere Ränder = 0."""
    h, w = img.shape[:2]
    ow, oh = out_size or (w, h)
    M = np.array([[1, 0, (ow - 1) / 2 - x], [0, 1, (oh - 1) / 2 - y]], dtype=np.float64)
    return cv2.warpAffine(img, M, (ow, oh), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def valid_mask_shift(shape: tuple[int, int], x: float, y: float) -> tuple[float, float, float, float]:
    """Gültiger Bereich (x0, y0, x1, y1) eines zentrierten Bildes in Ausgabekoordinaten."""
    h, w = shape
    sx, sy = (w - 1) / 2 - x, (h - 1) / 2 - y
    return sx, sy, sx + w, sy + h


def common_crop(shape: tuple[int, int], positions: list[tuple[float, float]]) -> tuple[int, int, int, int]:
    """Größtes Rechteck (x, y, w, h), das in ALLEN zentrierten Bildern gültig ist (ohne schwarze Ränder)."""
    h, w = shape
    x0 = y0 = -1e18
    x1 = y1 = 1e18
    for px, py in positions:
        a, b, c, d = valid_mask_shift((h, w), px, py)
        x0, y0, x1, y1 = max(x0, a), max(y0, b), min(x1, c), min(y1, d)
    X0, Y0 = int(np.ceil(x0)) + 1, int(np.ceil(y0)) + 1  # +1: Interpolationsrand
    X1, Y1 = int(np.floor(x1)) - 1, int(np.floor(y1)) - 1
    X0, Y0, X1, Y1 = max(X0, 0), max(Y0, 0), min(X1, w), min(Y1, h)
    return X0, Y0, max(X1 - X0, 0), max(Y1 - Y0, 0)
