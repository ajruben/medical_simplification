#!/usr/bin/env python3
"""CLI entry: run train_t5.py with --mode from the src folder, or run src/train_t5.py from repo root."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from t5_train.cli import main

if __name__ == "__main__":
    main()
