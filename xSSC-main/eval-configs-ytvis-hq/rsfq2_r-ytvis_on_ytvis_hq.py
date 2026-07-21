from pathlib import Path

from object_centric_bench.util import importlib_cfg


_ROOT = Path(__file__).resolve().parents[1]
globals().update(importlib_cfg(_ROOT / "config-randsfq/rsfq2_r-ytvis.py"))

dataset_v = dataset_v.copy()
dataset_v["data_file"] = "ytvis_hq/val.lmdb"
