"""Insert the src directory that contains paths.py into sys.path for legacy scripts.

Import this module before importing paths from legacy or other non-package scripts so resolution
matches the main training code under src.
"""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
for p in [_here, *_here.parents]:
    if (p / "paths.py").exists():
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
        break
