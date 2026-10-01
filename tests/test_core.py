import numpy as np
import cv2
from astro_zentrierer import align, stack as stk
from astro_zentrierer.pipeline import Item, Project, detect_all, run_stack
from astro_zentrierer.imageio import save_image, load_image


def moon_scene(cx, cy, size=(400, 500), seed=0, noise=0.02):
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.zeros((h, w), np.float32)
    cv2.circle(img, (int(round(cx)), int(round(cy))), 30, 0.8, -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 2)
    img += 0.1 + rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def make_project(tmp_path, pos):
    items = []
    for i, (x, y) in enumerate(pos):
        p = tmp_path / f"f{i}.tif"
        save_image(p, moon_scene(x, y, seed=i))
        items.append(Item(str(p)))
    return Project(items=items, marker=(pos[0][0], pos[0][1]), marker_half=45)


def test_detect_and_center(tmp_path):
    pos = [(250, 200), (270, 185), (230, 215), (255, 190)]
    prj = make_project(tmp_path, pos)
    detect_all(prj)
    for it, (x, y) in zip(prj.items, pos):
        assert abs(it.x - x) < 0.7 and abs(it.y - y) < 0.7, (it, x, y)
    res, crop = run_stack(prj, "Sigma-Clipping")
    ys, xs = np.nonzero(res > 0.5)
    assert abs(xs.mean() - (res.shape[1] - 1) / 2) < 1.0
    assert abs(ys.mean() - (res.shape[0] - 1) / 2) < 1.0
    x, y, w, h = crop
    assert w > 0 and h > 0 and x + w <= res.shape[1]
    sub = res[y:y + h, x:x + w]
    assert (sub > 0.05).all()  # kein schwarzer Rand im Crop


def test_stack_rejects_outliers():
    base = np.full((5, 5), 0.3, np.float32)
    fr = np.stack([base + np.random.default_rng(i).normal(0, 0.005, base.shape) for i in range(9)] + [base + 0.7])
    assert abs(stk.stack(fr, "Sigma-Clipping", sigma_high=2).mean() - 0.3) < 0.02
    assert abs(stk.stack(fr, "Median").mean() - 0.3) < 0.01
    assert stk.stack(fr, "Mittelwert").mean() > 0.35


def test_stack_improves_snr():
    clean = moon_scene(100, 100, (200, 200), noise=0)
    fr = np.stack([clean + np.random.default_rng(i).normal(0, 0.05, clean.shape) for i in range(16)]).astype(np.float32)
    assert np.std(stk.stack(fr, "Mittelwert") - clean) < 0.05 / 3


def test_star_field_translation(tmp_path):
    rng = np.random.default_rng(1)
    pts = rng.uniform(40, 360, (30, 2))
    def sky(dx, dy):
        im = np.zeros((400, 400), np.float32)
        for (x, y), b in zip(pts + [dx, dy], rng.uniform(0.3, 1, 30) * 0 + np.linspace(0.3, 1, 30)):
            if 0 <= x < 400 and 0 <= y < 400:
                im[int(round(y)), int(round(x))] = b
        return cv2.GaussianBlur(im, (0, 0), 1.5) * 8
    ref_star = pts[29]
    items = []
    shifts = [(0, 0), (6.4, -4.2), (-8, 7.6)]
    for i, (dx, dy) in enumerate(shifts):
        p = tmp_path / f"s{i}.tif"; save_image(p, np.clip(sky(dx, dy), 0, 1)); items.append(Item(str(p)))
    prj = Project(items, marker=tuple(ref_star), marker_half=30)
    detect_all(prj)
    for it, (dx, dy) in zip(items, shifts):
        assert abs(it.x - (ref_star[0] + dx)) < 1.0 and abs(it.y - (ref_star[1] + dy)) < 1.0
