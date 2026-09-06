#!/usr/bin/env python3
"""Run the baseline trainer in LP-C-only convergence mode."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    trainer = Path(__file__).with_name("01_train_baselines.py")
    sys.argv = [
        str(trainer),
        "--strategies",
        "LP-C",
        "--learning-rate",
        "1e-4",
        *sys.argv[1:],
    ]
    runpy.run_path(str(trainer), run_name="__main__")
