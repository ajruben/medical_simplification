"""Thin wrapper. Prefer train_t5.py with mode large_medical_sari."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from t5_train.pipeline import run_large_medical_sari

if __name__ == "__main__":
    run_large_medical_sari()
