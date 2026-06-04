from __future__ import annotations

from .utils import require_torch

torch = require_torch()


def load_torch_checkpoint(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
