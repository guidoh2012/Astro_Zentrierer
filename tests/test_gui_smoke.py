import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QApplication
from astro_zentrierer.gui import MainWindow
from astro_zentrierer.imageio import save_image
from test_core import moon_scene


def test_gui_flow(tmp_path):
    app = QApplication.instance() or QApplication([])
    pos = [(250, 200), (270, 185), (230, 215)]
    paths = []
    for i, (x, y) in enumerate(pos):
        p = tmp_path / f"m{i}.tif"; save_image(p, moon_scene(x, y, seed=i)); paths.append(str(p))
    w = MainWindow(); w.show()
    w.add_paths(paths); w.files.setCurrentRow(0)
    w.on_rect(QRectF(200, 150, 100, 100))
    assert w.prj.marker is not None
    w.prj.centroid = True
    from astro_zentrierer.pipeline import detect_all, run_stack
    detect_all(w.prj)
    w.mode.setCurrentIndex(1); w.refresh()
    w.result, w.crop = run_stack(w.prj, "Mittelwert")
    w.mode.setCurrentIndex(2); w.refresh()
    w.on_rect(QRectF(10, 10, 100, 80)); assert w.crop == (10, 10, 100, 80)
    w.auto_crop(); assert w.crop[2] > 0
    w.view.grab().save(str(tmp_path / "shot.png"))
