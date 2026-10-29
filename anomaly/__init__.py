"""Anomaly-detection study: do path-signature representations make anomaly scores
carry information beyond volatility clustering?

Reuses the data, feature and walk-forward code in ``Mihika/`` (put on ``sys.path``
here) instead of copying it.
"""
import sys
from pathlib import Path

_MIHIKA = Path(__file__).resolve().parent.parent / "Mihika"
if str(_MIHIKA) not in sys.path:
    sys.path.insert(0, str(_MIHIKA))
