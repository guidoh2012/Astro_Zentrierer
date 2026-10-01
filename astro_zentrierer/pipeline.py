"""Ablauf: laden -> Objekt suchen -> zentrieren -> stacken."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import align, stack as stk
from .imageio import load_image

Progress = Callable[[int, int, str], None]


@dataclass
class Item:
    path: str
    x: float = 0.0
    y: float = 0.0
    score: float = 0.0
    enabled: bool = True
    manual: bool = False  # Position vom Nutzer korrigiert


@dataclass
class Project:
    items: list[Item] = field(default_factory=list)
    ref_index: int = 0
    marker: tuple[float, float] | None = None  # (x, y) im Referenzbild
    marker_half: int = 40                       # halbe Kantenlänge des Templates
    centroid: bool = True                       # Schwerpunkt-Verfeinerung
    search_radius: int = 0                      # 0 = ganzes Bild durchsuchen


def detect_all(prj: Project, progress: Progress | None = None) -> None:
    """Findet das markierte Objekt in allen Bildern. Referenzbild: Position = Markierung (ggf. verfeinert)."""
    if prj.marker is None:
        raise ValueError("Erst das Objekt im Referenzbild markieren")
    ref_item = prj.items[prj.ref_index]
    ref = load_image(ref_item.path)
    mx, my = prj.marker
    cr = prj.marker_half if prj.centroid else None
    if cr:
        mx, my = align.refine_centroid(ref, mx, my, cr)
    tmpl = align.make_template(ref, mx, my, prj.marker_half)
    n = len(prj.items)
    for i, it in enumerate(prj.items):
        if progress:
            progress(i, n, it.path)
        if it.manual:
            continue
        if i == prj.ref_index:
            it.x, it.y, it.score = mx, my, 1.0
            continue
        img = load_image(it.path)
        hint = (mx, my) if prj.search_radius else None
        m = align.find_object(img, tmpl, hint, prj.search_radius or None, cr)
        it.x, it.y, it.score = m.x, m.y, m.score
    if progress:
        progress(n, n, "")


def run_stack(prj: Project, method: str, progress: Progress | None = None, **kw):
    """Zentriert alle aktiven Bilder und stackt sie. Gibt (Stack ohne Crop, Crop-Rechteck (x,y,w,h)) zurück."""
    its = [it for it in prj.items if it.enabled]
    if not its:
        raise ValueError("Keine aktiven Bilder")
    frames, valids = [], []
    for i, it in enumerate(its):
        if progress:
            progress(i, len(its), it.path)
        img = load_image(it.path)
        frames.append(align.shift_to_center(img, it.x, it.y))
        ones = np.ones(img.shape[:2], np.float32)
        v = align.shift_to_center(ones, it.x, it.y)
        valids.append(v > 0.999)  # Randpixel (Interpolation) ausschließen
    shape = frames[0].shape
    valid = np.stack([np.broadcast_to(v[..., None], shape) if len(shape) == 3 else v for v in valids])
    result = stk.stack(np.stack(frames), method, valid=valid, **kw)
    crop = align.common_crop(shape[:2], [(it.x, it.y) for it in its])
    if progress:
        progress(len(its), len(its), "")
    return result, crop
