"""PySide6-Oberfläche: Markieren -> Zentrieren -> Stacken -> Zuschneiden -> Speichern."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGraphicsPixmapItem, QGraphicsRectItem,
                               QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QSplitter, QSpinBox, QVBoxLayout, QWidget)

from . import align
from .imageio import ALL_EXT, load_image, save_image, to_display
from .pipeline import Item, Project, detect_all, run_stack
from .stack import METHODS

SRC, CENTERED, RESULT = 0, 1, 2


def to_qimage(rgb8: np.ndarray) -> QImage:
    h, w, _ = rgb8.shape
    rgb8 = np.ascontiguousarray(rgb8)
    return QImage(rgb8.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class Preview(QGraphicsView):
    """Zoom (Mausrad), Pan (rechte/mittlere Taste), Rechteck aufziehen (linke Taste)."""
    rect_drawn = Signal(QRectF)
    point_clicked = Signal(QPointF)  # Shift+Klick

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor(20, 20, 20))
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.pix = QGraphicsPixmapItem()
        self.scene().addItem(self.pix)
        self.rect_item: QGraphicsRectItem | None = None
        self.overlays: list = []
        self.draw_enabled = True
        self._start: QPointF | None = None
        self._pan = None

    def set_image(self, rgb8: np.ndarray, keep_view: bool):
        self.pix.setPixmap(QPixmap.fromImage(to_qimage(rgb8)))
        self.scene().setSceneRect(self.pix.boundingRect())
        if not keep_view:
            self.fitInView(self.pix, Qt.KeepAspectRatio)

    def clear_overlays(self):
        for o in self.overlays:
            self.scene().removeItem(o)
        self.overlays = []

    def add_rect(self, r: QRectF, color: str, dashed=False):
        it = self.scene().addRect(r, self._pen(color, dashed))
        self.overlays.append(it)
        return it

    def add_cross(self, x: float, y: float, color: str, size=18):
        pen = self._pen(color)
        for a, b in (((x - size, y), (x + size, y)), ((x, y - size), (x, y + size))):
            self.overlays.append(self.scene().addLine(a[0], a[1], b[0], b[1], pen))

    def _pen(self, color, dashed=False):
        pen = QPen(QColor(color), 1.5)
        pen.setCosmetic(True)
        if dashed:
            pen.setStyle(Qt.DashLine)
        return pen

    def wheelEvent(self, e):
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(f, f)

    def mouseDoubleClickEvent(self, e):
        self.fitInView(self.pix, Qt.KeepAspectRatio)

    def mousePressEvent(self, e):
        if e.button() in (Qt.RightButton, Qt.MiddleButton):
            self._pan = e.position()
            self.setCursor(Qt.ClosedHandCursor)
        elif e.button() == Qt.LeftButton:
            p = self.mapToScene(e.position().toPoint())
            if e.modifiers() & Qt.ShiftModifier:
                self.point_clicked.emit(p)
            elif self.draw_enabled:
                self._start = p
                if self.rect_item is None:
                    self.rect_item = self.scene().addRect(QRectF(), self._pen("#ffd400"))

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
        elif self._start is not None and self.rect_item is not None:
            p = self.mapToScene(e.position().toPoint())
            self.rect_item.setRect(QRectF(self._start, p).normalized())

    def mouseReleaseEvent(self, e):
        if self._pan is not None:
            self._pan = None
            self.unsetCursor()
        elif self._start is not None and self.rect_item is not None:
            r = self.rect_item.rect().intersected(self.pix.boundingRect())
            self._start = None
            if r.width() > 3 and r.height() > 3:
                self.rect_item.setRect(r)
                self.rect_drawn.emit(r)

    def reset_rect(self):
        if self.rect_item is not None:
            self.scene().removeItem(self.rect_item)
            self.rect_item = None


class Worker(QObject):
    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(lambda i, n, _p: self.progress.emit(i, n)))
        except Exception as ex:  # noqa: BLE001 – an GUI durchreichen
            self.failed.emit(f"{type(ex).__name__}: {ex}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Astro Zentrierer")
        self.resize(1280, 800)
        self.prj = Project()
        self.result: np.ndarray | None = None
        self.crop: tuple[int, int, int, int] | None = None
        self._cache: tuple[str, np.ndarray] | None = None
        self._threads: list = []

        self.view = Preview()
        self.view.rect_drawn.connect(self.on_rect)
        self.view.point_clicked.connect(self.on_manual_point)

        self.files = QListWidget()
        self.files.setSelectionMode(QAbstractItemView.SingleSelection)
        self.files.currentRowChanged.connect(lambda _i: self.refresh())
        self.files.itemChanged.connect(self.on_check)

        add = QPushButton("Bilder hinzufügen …")
        add.clicked.connect(self.add_files)
        rm = QPushButton("Entfernen")
        rm.clicked.connect(self.remove_file)
        ref = QPushButton("Als Referenz")
        ref.clicked.connect(self.set_reference)
        lbtn = QHBoxLayout()
        for b in (add, rm, ref):
            lbtn.addWidget(b)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(self.files)
        ll.addLayout(lbtn)

        # Ansicht
        self.mode = QComboBox()
        self.mode.addItems(["Original", "Zentriert", "Stack-Ergebnis"])
        self.mode.currentIndexChanged.connect(lambda _i: self.refresh())
        self.stretch = QCheckBox("Auto-Stretch")
        self.stretch.setChecked(True)
        self.stretch.toggled.connect(lambda _b: self.refresh(keep=True))
        g0 = QGroupBox("Ansicht")
        f0 = QFormLayout(g0)
        f0.addRow("Anzeige:", self.mode)
        f0.addRow(self.stretch)

        # 1 + 2 Markieren & Suchen
        self.hint = QLabel("1. Referenz wählen, dann im Bild ein Rechteck um das Objekt ziehen (Mond / Stern).")
        self.hint.setWordWrap(True)
        self.centroid = QCheckBox("Schwerpunkt verfeinern (Subpixel)")
        self.centroid.setChecked(True)
        self.radius = QSpinBox()
        self.radius.setRange(0, 5000)
        self.radius.setSpecialValueText("ganzes Bild")
        self.radius.setSuffix(" px")
        detect = QPushButton("2. Objekt in allen Bildern suchen")
        detect.clicked.connect(self.detect)
        g1 = QGroupBox("Markieren & Suchen")
        f1 = QFormLayout(g1)
        f1.addRow(self.hint)
        f1.addRow(self.centroid)
        f1.addRow("Suchradius:", self.radius)
        f1.addRow(detect)
        f1.addRow(QLabel("Shift+Klick im Bild setzt die Position des gewählten Bildes manuell."))

        # 3 Stacken
        self.method = QComboBox()
        self.method.addItems(METHODS)
        self.method.setCurrentText("Sigma-Clipping")
        self.slow = QDoubleSpinBox(); self.slow.setValue(3.0); self.slow.setRange(0.5, 10)
        self.shigh = QDoubleSpinBox(); self.shigh.setValue(3.0); self.shigh.setRange(0.5, 10)
        self.norm = QCheckBox("Normalisieren (Helligkeit angleichen)")
        stk = QPushButton("3. Zentrieren && Stacken")
        stk.clicked.connect(self.do_stack)
        g2 = QGroupBox("Stacken")
        f2 = QFormLayout(g2)
        f2.addRow("Methode:", self.method)
        f2.addRow("σ niedrig:", self.slow)
        f2.addRow("σ hoch:", self.shigh)
        f2.addRow(self.norm)
        f2.addRow(stk)

        # 4 Zuschneiden
        auto = QPushButton("Automatisch (ohne Ränder)")
        auto.clicked.connect(self.auto_crop)
        full = QPushButton("Zuschnitt zurücksetzen")
        full.clicked.connect(self.reset_crop)
        self.crop_label = QLabel("Kein Zuschnitt – im Ergebnis ein Rechteck ziehen.")
        self.crop_label.setWordWrap(True)
        save = QPushButton("4. Speichern …")
        save.clicked.connect(self.save)
        g3 = QGroupBox("Zuschneiden & Speichern")
        f3 = QVBoxLayout(g3)
        for w in (auto, full, self.crop_label, save):
            f3.addWidget(w)

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        right = QWidget()
        rl = QVBoxLayout(right)
        for w in (g0, g1, g2, g3):
            rl.addWidget(w)
        rl.addStretch()
        rl.addWidget(self.bar)
        right.setFixedWidth(320)

        sp = QSplitter()
        sp.addWidget(left)
        sp.addWidget(self.view)
        sp.addWidget(right)
        sp.setSizes([240, 720, 320])
        self.setCentralWidget(sp)
        self.statusBar().showMessage("Bilder hinzufügen, um zu starten.")

        act = QAction("Bilder öffnen …", self)
        act.setShortcut("Ctrl+O")
        act.triggered.connect(self.add_files)
        self.menuBar().addMenu("Datei").addAction(act)

    # ---------- Dateiliste ----------
    def add_files(self):
        exts = " ".join(f"*{e}" for e in sorted(ALL_EXT))
        paths, _ = QFileDialog.getOpenFileNames(self, "Bilder wählen", "", f"Bilder ({exts})")
        self.add_paths(paths)

    def add_paths(self, paths):
        self.files.blockSignals(True)
        for p in sorted(paths):
            self.prj.items.append(Item(p))
        self.files.blockSignals(False)
        self.rebuild_list()
        if self.files.currentRow() < 0 and self.prj.items:
            self.files.setCurrentRow(0)
        self.refresh()

    def rebuild_list(self):
        cur = self.files.currentRow()
        self.files.blockSignals(True)
        self.files.clear()
        for i, it in enumerate(self.prj.items):
            tag = "★ " if i == self.prj.ref_index else ""
            sc = f"  ({it.score:.2f})" if it.score and i != self.prj.ref_index else ""
            li = QListWidgetItem(f"{tag}{Path(it.path).name}{sc}")
            li.setFlags(li.flags() | Qt.ItemIsUserCheckable)
            li.setCheckState(Qt.Checked if it.enabled else Qt.Unchecked)
            if it.manual:
                li.setForeground(QColor("#2a7fff"))
            elif it.score and it.score < 0.5 and i != self.prj.ref_index:
                li.setForeground(QColor("#d33"))
            self.files.addItem(li)
        self.files.blockSignals(False)
        if 0 <= cur < self.files.count():
            self.files.setCurrentRow(cur)

    def on_check(self, li):
        self.prj.items[self.files.row(li)].enabled = li.checkState() == Qt.Checked

    def remove_file(self):
        r = self.files.currentRow()
        if r < 0:
            return
        del self.prj.items[r]
        if self.prj.ref_index >= len(self.prj.items) or r < self.prj.ref_index:
            self.prj.ref_index = max(0, min(self.prj.ref_index - (r < self.prj.ref_index), len(self.prj.items) - 1))
        self.rebuild_list()
        self.refresh()

    def set_reference(self):
        r = self.files.currentRow()
        if r >= 0:
            self.prj.ref_index = r
            self.prj.marker = None
            self.view.reset_rect()
            self.rebuild_list()
            self.refresh()

    # ---------- Anzeige ----------
    def current_image(self, path: str) -> np.ndarray:
        if self._cache is None or self._cache[0] != path:
            self._cache = (path, load_image(path))
        return self._cache[1]

    def refresh(self, keep=False):
        r = self.files.currentRow()
        mode = self.mode.currentIndex()
        self.view.clear_overlays()
        self.view.draw_enabled = True
        if mode == RESULT:
            if self.result is None:
                self.statusBar().showMessage("Noch kein Stack-Ergebnis.")
                return
            self.view.set_image(to_display(self.result, self.stretch.isChecked()), keep)
            self.view.reset_rect()
            if self.crop and self.crop[2] > 0:
                self.view.add_rect(QRectF(*self.crop), "#00e0ff", dashed=True)
            return
        if r < 0 or r >= len(self.prj.items):
            return
        it = self.prj.items[r]
        try:
            img = self.current_image(it.path)
        except Exception as ex:  # noqa: BLE001
            self.statusBar().showMessage(str(ex))
            return
        has_pos = it.score > 0 or it.manual
        if mode == CENTERED and has_pos:
            img = align.shift_to_center(img, it.x, it.y)
            self.view.set_image(to_display(img, self.stretch.isChecked()), keep)
            h, w = img.shape[:2]
            self.view.add_cross((w - 1) / 2, (h - 1) / 2, "#00ff66")
            self.view.reset_rect()
            return
        self.view.set_image(to_display(img, self.stretch.isChecked()), keep)
        if r == self.prj.ref_index and self.prj.marker:
            mx, my = self.prj.marker
            h = self.prj.marker_half
            self.view.reset_rect()
            self.view.rect_item = self.view.scene().addRect(QRectF(mx - h, my - h, 2 * h, 2 * h), self.view._pen("#ffd400"))
        else:
            self.view.reset_rect()
        if has_pos:
            self.view.add_cross(it.x, it.y, "#00ff66" if not it.manual else "#2a7fff")
            self.view.add_rect(QRectF(it.x - self.prj.marker_half, it.y - self.prj.marker_half,
                                      2 * self.prj.marker_half, 2 * self.prj.marker_half), "#00ff66", dashed=True)

    # ---------- Interaktion ----------
    def on_rect(self, r: QRectF):
        if self.mode.currentIndex() == RESULT:
            self.crop = (int(r.x()), int(r.y()), int(r.width()), int(r.height()))
            self.crop_label.setText(f"Zuschnitt: {self.crop[2]} × {self.crop[3]} px")
            return
        if self.files.currentRow() != self.prj.ref_index:
            self.statusBar().showMessage("Markieren geht im Referenzbild (★). Erst 'Als Referenz' wählen.")
            self.view.reset_rect()
            return
        c = r.center()
        self.prj.marker = (c.x(), c.y())
        self.prj.marker_half = max(8, int(max(r.width(), r.height()) / 2))
        for it in self.prj.items:
            it.score = 0.0
            if not it.manual:
                it.x = it.y = 0.0
        self.statusBar().showMessage(f"Objekt markiert bei ({c.x():.0f}, {c.y():.0f}). Weiter mit 'Objekt suchen'.")

    def on_manual_point(self, p: QPointF):
        r = self.files.currentRow()
        if self.mode.currentIndex() == RESULT or r < 0:
            return
        it = self.prj.items[r]
        x, y = p.x(), p.y()
        if self.centroid.isChecked():
            x, y = align.refine_centroid(self.current_image(it.path), x, y, self.prj.marker_half)
        it.x, it.y, it.score, it.manual = x, y, 1.0, True
        self.rebuild_list()
        self.refresh(keep=True)

    # ---------- Hintergrundjobs ----------
    def run_job(self, fn, on_done):
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        th, wk = QThread(self), Worker(fn)
        wk.moveToThread(th)
        th.started.connect(wk.run)
        wk.progress.connect(lambda i, n: (self.bar.setRange(0, n), self.bar.setValue(i)))

        def finish(res):
            th.quit(); th.wait()
            self.bar.setVisible(False)
            self._threads.remove((th, wk))
            on_done(res)

        def fail(msg):
            th.quit(); th.wait()
            self.bar.setVisible(False)
            self._threads.remove((th, wk))
            QMessageBox.warning(self, "Fehler", msg)

        wk.done.connect(finish)
        wk.failed.connect(fail)
        self._threads.append((th, wk))
        th.start()

    def detect(self):
        if not self.prj.items:
            return
        if self.prj.marker is None:
            QMessageBox.information(self, "Hinweis", "Bitte zuerst im Referenzbild das Objekt markieren.")
            return
        self.prj.centroid = self.centroid.isChecked()
        self.prj.search_radius = self.radius.value()

        def job(cb):
            detect_all(self.prj, cb)

        def done(_):
            self.rebuild_list()
            bad = [Path(i.path).name for k, i in enumerate(self.prj.items)
                   if i.score < 0.5 and k != self.prj.ref_index]
            msg = "Objekt in allen Bildern gefunden." if not bad else \
                f"Unsicher bei {len(bad)} Bild(ern) (rot) – per Shift+Klick korrigieren oder abwählen."
            self.statusBar().showMessage(msg)
            self.mode.setCurrentIndex(CENTERED)
            self.refresh()

        self.run_job(job, done)

    def do_stack(self):
        if not any(i.score > 0 or i.manual for i in self.prj.items):
            QMessageBox.information(self, "Hinweis", "Erst das Objekt suchen (Schritt 2).")
            return
        kw = dict(sigma_low=self.slow.value(), sigma_high=self.shigh.value(), normalize=self.norm.isChecked())

        def job(cb):
            return run_stack(self.prj, self.method.currentText(), cb, **kw)

        def done(res):
            self.result, self.crop = res
            self.crop_label.setText(f"Auto-Zuschnitt: {self.crop[2]} × {self.crop[3]} px (Ränder entfernt)")
            self.mode.setCurrentIndex(RESULT)
            self.refresh()

        self.run_job(job, done)

    def auto_crop(self):
        act = [(i.x, i.y) for i in self.prj.items if i.enabled and (i.score > 0 or i.manual)]
        if self.result is None or not act:
            return
        self.crop = align.common_crop(self.result.shape[:2], act)
        self.crop_label.setText(f"Auto-Zuschnitt: {self.crop[2]} × {self.crop[3]} px")
        self.refresh(keep=True)

    def reset_crop(self):
        self.crop = None
        self.crop_label.setText("Kein Zuschnitt.")
        self.refresh(keep=True)

    def save(self):
        if self.result is None:
            QMessageBox.information(self, "Hinweis", "Noch nichts gestackt.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Speichern", "stack.tif",
                                              "TIFF 32-bit (*.tif);;PNG 16-bit (*.png);;FITS (*.fits);;JPEG (*.jpg)")
        if not path:
            return
        out = self.result
        if self.crop and self.crop[2] > 0 and self.crop[3] > 0:
            x, y, w, h = self.crop
            out = out[y:y + h, x:x + w]
        save_image(path, out)
        self.statusBar().showMessage(f"Gespeichert: {path}")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    w.add_paths([a for a in sys.argv[1:] if Path(a).suffix.lower() in ALL_EXT])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
