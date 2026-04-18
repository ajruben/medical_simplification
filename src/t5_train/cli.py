"""Command-line interface for selecting a training mode and running its pipeline.

Run as a module with src on PYTHONPATH, or invoke train_t5.py in the src directory which adds
that path for you.
"""
from __future__ import annotations

import argparse

from t5_train.pipeline import MODES


def main() -> None:
    """Parse --mode and dispatch to the corresponding function in MODES."""
    p = argparse.ArgumentParser(description="T5 simplification training")
    p.add_argument(
        "--mode",
        choices=sorted(MODES),
        required=True,
        help="Training pipeline.",
    )
    args = p.parse_args()
    MODES[args.mode]()


if __name__ == "__main__":
    main()
