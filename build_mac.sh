#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --windowed --name "Astro Zentrierer" --collect-all rawpy astro_zentrierer/__main__.py
echo "Fertig: dist/Astro Zentrierer.app"
