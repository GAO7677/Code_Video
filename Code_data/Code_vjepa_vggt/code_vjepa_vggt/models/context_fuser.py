from __future__ import annotations

import torch
import torch.nn as nn


class ContextTokenFuser(nn.Module):
    def __init__(self, text_dim: int, max_context_len: int, min_text_tokens: int = 64) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.max_context_len = max_context_len
        self.min_text_tokens = min_text_tokens
        self.object_gate = nn.Parameter(torch.tensor(1.0))
        self.norm = nn.LayerNorm(text_dim)
        self.object_score = nn.Linear(text_dim, 1)

    def forward(
        self,
        text_context: list[torch.Tensor],
        object_tokens: torch.Tensor,
    ) -> list[torch.Tensor]:
        out = []
        for i, txt in enumerate(text_context):
            obj = self.norm(self.object_gate * object_tokens[i])
            max_objects = max(0, self.max_context_len - min(int(txt.shape[0]), self.min_text_tokens))
            if obj.shape[0] > max_objects > 0:
                scores = self.object_score(obj).squeeze(-1)
                keep_idx = torch.topk(scores, k=max_objects, dim=0).indices
                keep_idx, _ = torch.sort(keep_idx)
                obj = obj[keep_idx]
            elif max_objects == 0:
                obj = obj[:0]
            keep_text = max(0, self.max_context_len - obj.shape[0])
            fused = torch.cat([txt[:keep_text], obj], dim=0)
            out.append(fused)
        return out
