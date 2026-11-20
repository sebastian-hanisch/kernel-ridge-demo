"""Gemeinsame Fixtures: das Handbeispiel (drei Punkte, Polynomkern Grad 2, in Brüchen) und kleine Plan-Instanzen (DEMO-PLAYBOOK §1b)."""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import krd_exact as E  # noqa: E402


@pytest.fixture
def hand():
    """Drei Punkte x = (-1, 0, 1), y = (1, 0, 1), Polynomkern (x x' + 1)^2, lambda = 1, Testpunkt x* = 2; Handrechnung: K = [[4,1,0],[1,1,1],[0,1,4]], alpha = (1/4, -1/4, 1/4)."""
    return E.hand_example(1, 2)
