from pathlib import Path

from object_centric_bench.datum import YTVIS
from object_centric_bench.util import importlib_cfg


_ROOT = Path(__file__).resolve().parents[1]
globals().update(importlib_cfg(_ROOT / "config-randsfq/rsfq2_c-movi_e.py"))

batch_size_v = 1
dataset_v = dict(
    type=YTVIS,
    data_file="ytvis_hq/val.lmdb",
    extra_keys=["segment", "bbox"],
    transform=dict(type=Compose, transforms=transform_v),
    base_dir=...,
)
