from __future__ import annotations

import hashlib
import re
from typing import Iterable, List


def tokenize_prompt(prompt: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", prompt.lower())


def hash_prompt_tokens(prompt: str, vocab_size: int) -> List[int]:
    token_ids: List[int] = []
    for token in tokenize_prompt(prompt):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        token_ids.append(int(digest, 16) % vocab_size)
    if not token_ids:
        token_ids.append(0)
    return token_ids


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "PyTorch is required for this module. Use /data/gaoya/miniconda3/envs/wan/bin/python."
        ) from exc
    return torch


def detach_to_cpu_numpy(tensor):
    torch = require_torch()
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor
