# Astro Zentrierer

macOS-Tool (PySide6) zum einheitlichen Zentrieren von Mond-/Sternaufnahmen mit anschließendem Stacking.

## Ablauf
1. **Bilder hinzufügen** (PNG, TIFF, JPG, FITS, RAW wie CR2/CR3/NEF/ARW/DNG) – Referenzbild mit *Als Referenz* (★) wählen.
2. **Markieren:** im Referenzbild ein Rechteck um das Objekt ziehen, das in allen Bildern vorkommt (Mond, heller Stern, Komet …).
3. **Objekt suchen:** Template-Matching (NCC, Subpixel) + optionale Schwerpunkt-Verfeinerung. Unsichere Treffer sind rot; per **Shift+Klick** manuell korrigieren oder Bild abwählen.
4. **Zentrieren & Stacken:** Mittelwert, Median, Sigma-Clipping, Winsorized Sigma, Maximum; optional Normalisierung.
5. **Zuschneiden:** Automatisch ohne schwarze Ränder oder Rechteck im Ergebnis ziehen.
6. **Speichern:** TIFF (32-bit float), PNG (16-bit), FITS, JPEG.

Bedienung der Vorschau: Mausrad = Zoom, rechte/mittlere Taste = Verschieben, Doppelklick = einpassen.

## Starten
```
pip install -r requirements.txt
python -m astro_zentrierer [bilder...]
```
## Als .app bauen (auf dem Mac)
```
./build_mac.sh   # -> dist/Astro Zentrierer.app
```
## Tests
```
PYTHONPATH=.:tests pytest
```
## Grenzen (v0.1)
Nur Translation – keine Bildfeld-Rotation/Skalierung (bei langen Sternfeld-Serien ggf. relevant), keine Darks/Flats/Debayer-Kalibrierung.
