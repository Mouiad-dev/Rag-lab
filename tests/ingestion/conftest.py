"""Lets tests in this folder import shared helpers (`_pdf_fixtures`); pytest runs with --import-mode=importlib."""

import sys
from pathlib import Path

_HERE = str(Path(__file__).parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
