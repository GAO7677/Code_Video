from __future__ import annotations

import torch
import torch.nn as nn


class ContextTokenFuser(nn.Module):
    def __init__(self, text_dim: int, max_context_len: int) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.max_context_len = max_context_len
        self.object_gate = nn.Parameter(torch.tensor(1.0))
        self.norm = nn.LayerNorm(text_dim)

    def forward(
        self,
        text_context: list[torch.Tensor],
        object_tokens: torch.Tensor,
    ) -> list[torch.Tensor]:
        out = []
        for i, txt in enumerate(text_context):
            obj = self.norm(self.object_gate * object_tokens[i])
            keep_text = max(0, self.max_context_len - obj.shape[0])
            fused = torch.cat([txt[:keep_text], obj], dim=0)
            out.append(fused)
        return out
