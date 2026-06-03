# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from importlib import import_module

from . import configs, distributed, modules

__all__ = [
    "configs",
    "distributed",
    "modules",
    "WanI2V",
    "WanS2V",
    "WanT2V",
    "WanTI2V",
    "WanAnimate",
]


def __getattr__(name):
    if name == "WanI2V":
        return import_module(".image2video", __name__).WanI2V
    if name == "WanS2V":
        return import_module(".speech2video", __name__).WanS2V
    if name == "WanT2V":
        return import_module(".text2video", __name__).WanT2V
    if name == "WanTI2V":
        return import_module(".textimage2video", __name__).WanTI2V
    if name == "WanAnimate":
        return import_module(".animate", __name__).WanAnimate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
