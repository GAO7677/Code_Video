# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from . import configs, distributed, modules
from .image2video import WanI2V
from .text2video import WanT2V
from .textimage2video import WanTI2V
from .animate import WanAnimate

try:  # Optional audio dependencies such as librosa may be absent in video-only envs.
    from .speech2video import WanS2V
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    WanS2V = None
