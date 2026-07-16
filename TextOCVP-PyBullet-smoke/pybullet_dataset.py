"""Compatibility export for the official raw PyBullet TextOCVP adapter."""

from __future__ import annotations

import sys
from pathlib import Path


TEXTOCVP_SRC = Path("/home/gaoya/Code_Video/TextOCVP-master/src")
sys.path.insert(0, str(TEXTOCVP_SRC))

from data.PyBullet import PyBullet  # noqa: E402,F401


PyBulletTextOCVPDataset = PyBullet
