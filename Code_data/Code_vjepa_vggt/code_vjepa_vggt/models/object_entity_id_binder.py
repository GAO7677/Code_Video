from __future__ import annotations

from collections.abc import Sequence
import weakref

import torch
import torch.nn as nn

from code_vjepa_vggt.models.object_condition_adapter import ObjectConditionAdapter


def attach_entity_text_binding_adapter(pipe, adapter: nn.Module) -> None:
    """Expose the adapter to inference without registering a duplicate module."""
    object.__setattr__(
        pipe,
        "_entity_text_binding_adapter_ref",
        weakref.ref(adapter),
    )


def find_subsequence_spans(
    sequence: Sequence[int],
    subsequence: Sequence[int],
) -> list[tuple[int, int]]:
    """Return all non-overlapping [start, end) matches."""
    needle = [int(value) for value in subsequence]
    if not needle:
        return []
    haystack = [int(value) for value in sequence]
    spans: list[tuple[int, int]] = []
    index = 0
    while index <= len(haystack) - len(needle):
        if haystack[index : index + len(needle)] == needle:
            spans.append((index, index + len(needle)))
            index += len(needle)
        else:
            index += 1
    return spans


class EntityIDBindingObjectConditionAdapter(ObjectConditionAdapter):
    """Object adapter with hard-routed per-entity text residuals.

    ``slot_entity_ids`` is metadata, not a semantic class index. It selects the
    entity text row routed to each tracked object slot before the unchanged
    ObjectConditionAdapter path runs.
    """

    def __init__(
        self,
        *,
        dim: int = 4096,
        num_slots: int = 8,
        max_time_steps: int = 64,
        output_gate_init: float = 0.1,
        entity_text_dim: int | None = None,
        entity_bottleneck_dim: int = 256,
        entity_gate_init: float = 0.1,
        entity_dropout_prob: float = 0.2,
        entity_residual_max_ratio: float = 0.1,
    ) -> None:
        super().__init__(
            dim=dim,
            num_slots=num_slots,
            max_time_steps=max_time_steps,
            output_gate_init=output_gate_init,
        )
        self.entity_text_dim = int(dim if entity_text_dim is None else entity_text_dim)
        self.entity_bottleneck_dim = int(entity_bottleneck_dim)
        self.entity_dropout_prob = float(entity_dropout_prob)
        self.entity_residual_max_ratio = float(entity_residual_max_ratio)
        if self.entity_bottleneck_dim <= 0:
            raise ValueError("entity_bottleneck_dim must be positive")
        if not 0.0 <= self.entity_dropout_prob <= 1.0:
            raise ValueError("entity_dropout_prob must be in [0, 1]")
        if self.entity_residual_max_ratio < 0.0:
            raise ValueError("entity_residual_max_ratio must be non-negative")

        self.entity_text_norm = nn.LayerNorm(self.entity_text_dim)
        self.entity_text_down = nn.Linear(
            self.entity_text_dim,
            self.entity_bottleneck_dim,
            bias=False,
        )
        self.entity_id_embed = nn.Embedding(
            self.num_slots,
            self.entity_bottleneck_dim,
        )
        self.entity_text_act = nn.GELU()
        self.entity_text_up = nn.Linear(
            self.entity_bottleneck_dim,
            self.dim,
            bias=False,
        )
        self.entity_text_context_up = nn.Linear(
            self.entity_bottleneck_dim,
            self.entity_text_dim,
            bias=False,
        )
        # Exact old-model behavior at initialization while entity_text_up still
        # receives gradients through the nonzero gate.
        nn.init.zeros_(self.entity_text_up.weight)
        nn.init.zeros_(self.entity_text_context_up.weight)
        self.entity_binding_gate = nn.Parameter(
            torch.tensor(float(entity_gate_init), dtype=torch.float32)
        )

        self._entity_text_by_id: torch.Tensor | None = None
        self._entity_text_match_mask: torch.Tensor | None = None
        self._slot_entity_ids: torch.Tensor | None = None
        self._text_token_entity_ids: torch.Tensor | None = None
        self._entity_keep_by_id: torch.Tensor | None = None
        self._last_entity_text_binding_metrics = self._empty_text_binding_metrics()
        self._last_entity_binding_metrics = self._empty_binding_metrics()

    @staticmethod
    def _empty_binding_metrics() -> dict[str, float]:
        return {
            "train/entity_binding_active": 0.0,
            "train/entity_binding_valid_slot_count": 0.0,
            "train/entity_binding_matched_slot_count": 0.0,
            "train/entity_binding_unique_id_count": 0.0,
            "train/entity_binding_id_collision_count": 0.0,
            "train/entity_binding_dropout_fraction": 0.0,
            "train/entity_binding_gate_tanh": 0.0,
            "train/entity_binding_residual_ratio_mean": 0.0,
            "train/entity_binding_residual_ratio_max": 0.0,
            "train/entity_binding_cap_applied_fraction": 0.0,
            "train/entity_binding_cap_scale_min": 1.0,
            **EntityIDBindingObjectConditionAdapter._empty_text_binding_metrics(),
        }

    @staticmethod
    def _empty_text_binding_metrics() -> dict[str, float]:
        return {
            "train/entity_text_binding_active": 0.0,
            "train/entity_text_binding_active_token_count": 0.0,
            "train/entity_text_binding_residual_ratio_mean": 0.0,
            "train/entity_text_binding_residual_ratio_max": 0.0,
            "train/entity_text_binding_cap_applied_fraction": 0.0,
            "train/entity_text_binding_cap_scale_min": 1.0,
        }

    def set_entity_binding_context(
        self,
        *,
        entity_text_by_id: torch.Tensor,
        entity_text_match_mask: torch.Tensor,
        slot_entity_ids: torch.Tensor,
        text_token_entity_ids: torch.Tensor | None = None,
    ) -> None:
        if entity_text_by_id.ndim != 3:
            raise ValueError("entity_text_by_id must be [B,E,D]")
        if entity_text_match_mask.shape != entity_text_by_id.shape[:2]:
            raise ValueError("entity_text_match_mask must match [B,E]")
        if slot_entity_ids.ndim != 2 or int(slot_entity_ids.shape[0]) != int(
            entity_text_by_id.shape[0]
        ):
            raise ValueError("slot_entity_ids must be [B,O]")
        if int(entity_text_by_id.shape[-1]) != self.entity_text_dim:
            raise ValueError(
                f"entity text dim={entity_text_by_id.shape[-1]} does not match "
                f"configured dim={self.entity_text_dim}"
            )
        self._entity_text_by_id = entity_text_by_id.detach()
        self._entity_text_match_mask = entity_text_match_mask.detach()
        self._slot_entity_ids = slot_entity_ids.detach().long()
        if text_token_entity_ids is not None:
            if text_token_entity_ids.ndim != 2 or int(text_token_entity_ids.shape[0]) != int(
                entity_text_by_id.shape[0]
            ):
                raise ValueError("text_token_entity_ids must be [B,L]")
            self._text_token_entity_ids = text_token_entity_ids.detach().long()
        else:
            self._text_token_entity_ids = None
        keep_by_id = entity_text_match_mask.detach().bool()
        if self.training and self.entity_dropout_prob > 0.0:
            keep_by_id = keep_by_id & (
                torch.rand(keep_by_id.shape, device=keep_by_id.device)
                >= self.entity_dropout_prob
            )
        self._entity_keep_by_id = keep_by_id

    def restrict_entity_binding_to_valid_slots(
        self,
        object_valid_mask: torch.Tensor,
    ) -> None:
        """Remove text/object bindings for slots dropped after grounding."""
        slot_entity_ids = self._slot_entity_ids
        keep_by_id = self._entity_keep_by_id
        if slot_entity_ids is None or keep_by_id is None:
            return
        if tuple(object_valid_mask.shape) != tuple(slot_entity_ids.shape):
            raise ValueError("object_valid_mask must match slot_entity_ids")
        valid_ids = slot_entity_ids >= 0
        active_ids = torch.zeros_like(keep_by_id, dtype=torch.bool)
        for batch_id in range(int(slot_entity_ids.shape[0])):
            ids = slot_entity_ids[batch_id][
                valid_ids[batch_id]
                & (object_valid_mask[batch_id].to(slot_entity_ids.device) > 0.5)
            ]
            if int(ids.numel()) > 0:
                active_ids[batch_id, ids] = True
        self._entity_keep_by_id = keep_by_id & active_ids

    def clear_entity_object_context(self) -> None:
        """Clear object-side tensors while retaining text routing for inference."""
        self._entity_text_by_id = None
        self._entity_text_match_mask = None
        self._slot_entity_ids = None

    def clear_entity_binding_context(self) -> None:
        self.clear_entity_object_context()
        self._text_token_entity_ids = None
        self._entity_keep_by_id = None

    def pop_entity_binding_metrics(self) -> dict[str, float]:
        metrics = dict(self._last_entity_binding_metrics)
        metrics.update(self._last_entity_text_binding_metrics)
        self._last_entity_binding_metrics = self._empty_binding_metrics()
        self._last_entity_text_binding_metrics = self._empty_text_binding_metrics()
        return metrics

    def apply_entity_ids_to_text_context(
        self,
        text_context: torch.Tensor,
    ) -> torch.Tensor:
        """Add the same sample-local entity IDs to their routed T5 noun spans."""
        token_entity_ids = self._text_token_entity_ids
        keep_by_id = self._entity_keep_by_id
        if token_entity_ids is None or keep_by_id is None:
            self._last_entity_text_binding_metrics = self._empty_text_binding_metrics()
            return text_context
        if text_context.ndim != 3 or int(text_context.shape[-1]) != self.entity_text_dim:
            raise ValueError(
                f"text_context must be [B,L,{self.entity_text_dim}], got {list(text_context.shape)}"
            )
        if tuple(token_entity_ids.shape) != tuple(text_context.shape[:2]):
            raise ValueError(
                "text_token_entity_ids must match the text context batch/sequence dimensions"
            )

        entity_count = int(keep_by_id.shape[1])
        valid_ids = (token_entity_ids >= 0) & (token_entity_ids < entity_count)
        safe_ids = token_entity_ids.clamp(min=0, max=max(entity_count - 1, 0))
        token_keep = torch.gather(
            keep_by_id.to(device=safe_ids.device),
            dim=1,
            index=safe_ids,
        )
        active = valid_ids & token_keep
        id_hidden = self.entity_id_embed(
            safe_ids.to(device=self.entity_id_embed.weight.device)
        )
        residual = self.entity_text_context_up(id_hidden).to(
            device=text_context.device,
            dtype=text_context.dtype,
        )
        residual = residual * active[:, :, None].to(residual.dtype)

        base_rms = text_context.float().square().mean(dim=-1).clamp_min(1.0e-12).sqrt()
        residual_rms = residual.float().square().mean(dim=-1).clamp_min(1.0e-12).sqrt()
        ratio = residual_rms / base_rms
        cap_scale = torch.ones_like(ratio)
        if self.entity_residual_max_ratio > 0.0:
            cap_scale = (
                self.entity_residual_max_ratio / ratio.clamp_min(1.0e-12)
            ).clamp(max=1.0)
            residual = residual * cap_scale.unsqueeze(-1).to(residual.dtype)
        gate = torch.tanh(self.entity_binding_gate.float()).to(
            device=residual.device,
            dtype=residual.dtype,
        )
        output = text_context + gate * residual

        if bool(active.any()):
            active_ratio = ratio[active]
            active_scale = cap_scale[active]
            metrics = {
                "train/entity_text_binding_active": 1.0,
                "train/entity_text_binding_active_token_count": float(active.sum().item()),
                "train/entity_text_binding_residual_ratio_mean": float(
                    active_ratio.detach().mean().item()
                ),
                "train/entity_text_binding_residual_ratio_max": float(
                    active_ratio.detach().max().item()
                ),
                "train/entity_text_binding_cap_applied_fraction": float(
                    (active_scale.detach() < 0.9999).float().mean().item()
                ),
                "train/entity_text_binding_cap_scale_min": float(
                    active_scale.detach().min().item()
                ),
            }
        else:
            metrics = self._empty_text_binding_metrics()
        self._last_entity_text_binding_metrics = metrics
        return output

    def apply_entity_binding(
        self,
        object_latent_tokens: torch.Tensor,
        *,
        object_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        entity_text = self._entity_text_by_id
        entity_match = self._entity_text_match_mask
        slot_entity_ids = self._slot_entity_ids
        if entity_text is None or entity_match is None or slot_entity_ids is None:
            self._last_entity_binding_metrics = self._empty_binding_metrics()
            return object_latent_tokens

        batch, time_steps, slots, _ = object_latent_tokens.shape
        if tuple(slot_entity_ids.shape) != (batch, slots):
            raise ValueError(
                f"slot_entity_ids={list(slot_entity_ids.shape)} must match B,O={(batch, slots)}"
            )
        entity_count = int(entity_text.shape[1])
        valid_ids = (slot_entity_ids >= 0) & (slot_entity_ids < entity_count)
        safe_ids = slot_entity_ids.clamp(min=0, max=max(entity_count - 1, 0))
        gather_index = safe_ids.unsqueeze(-1).expand(-1, -1, self.entity_text_dim)
        slot_text = torch.gather(
            entity_text.to(device=object_latent_tokens.device),
            dim=1,
            index=gather_index,
        )
        slot_match = torch.gather(
            entity_match.to(device=object_latent_tokens.device),
            dim=1,
            index=safe_ids,
        ).bool()
        slot_match = slot_match & valid_ids.to(device=slot_match.device)
        if object_valid_mask is not None:
            valid_slots = object_valid_mask.to(device=slot_match.device) > 0.5
        else:
            valid_slots = torch.ones_like(slot_match, dtype=torch.bool)
        active_slots = slot_match & valid_slots

        keep_by_id = self._entity_keep_by_id
        if keep_by_id is None:
            keep_slots = active_slots
        else:
            keep_slots = active_slots & torch.gather(
                keep_by_id.to(device=safe_ids.device),
                dim=1,
                index=safe_ids,
            )
        dropped_slots = active_slots & ~keep_slots

        normalized_text = self.entity_text_norm(
            slot_text.to(
                device=self.entity_text_norm.weight.device,
                dtype=self.entity_text_norm.weight.dtype,
            )
        )
        text_hidden = self.entity_text_down(normalized_text)
        id_hidden = self.entity_id_embed(
            safe_ids.to(device=self.entity_id_embed.weight.device)
        ).to(dtype=text_hidden.dtype)
        projected = self.entity_text_up(
            self.entity_text_act(text_hidden + id_hidden)
        )
        projected = projected.to(
            device=object_latent_tokens.device,
            dtype=object_latent_tokens.dtype,
        )
        residual = projected[:, None, :, :].expand(-1, time_steps, -1, -1)
        residual = residual * keep_slots[:, None, :, None].to(dtype=residual.dtype)

        base_rms = (
            object_latent_tokens.float().square().mean(dim=-1).clamp_min(1.0e-12).sqrt()
        )
        residual_rms = (
            residual.float().square().mean(dim=-1).clamp_min(1.0e-12).sqrt()
        )
        ratio = residual_rms / base_rms
        cap_scale = torch.ones_like(ratio)
        if self.entity_residual_max_ratio > 0.0:
            cap_scale = (
                self.entity_residual_max_ratio / ratio.clamp_min(1.0e-12)
            ).clamp(max=1.0)
            residual = residual * cap_scale.unsqueeze(-1).to(dtype=residual.dtype)

        gate = torch.tanh(self.entity_binding_gate.float()).to(
            device=residual.device,
            dtype=residual.dtype,
        )
        output = object_latent_tokens + gate * residual

        active_time_mask = keep_slots[:, None, :].expand(-1, time_steps, -1)
        if bool(active_time_mask.any()):
            active_ratio = ratio[active_time_mask]
            active_scale = cap_scale[active_time_mask]
            ratio_mean = float(active_ratio.detach().mean().item())
            ratio_max = float(active_ratio.detach().max().item())
            cap_fraction = float((active_scale.detach() < 0.9999).float().mean().item())
            cap_scale_min = float(active_scale.detach().min().item())
        else:
            ratio_mean = 0.0
            ratio_max = 0.0
            cap_fraction = 0.0
            cap_scale_min = 1.0

        collision_count = 0
        unique_count = 0
        for batch_id in range(batch):
            ids = slot_entity_ids[batch_id][valid_slots[batch_id] & valid_ids[batch_id]]
            unique = int(torch.unique(ids).numel()) if int(ids.numel()) > 0 else 0
            unique_count += unique
            collision_count += int(ids.numel()) - unique
        active_count = int(active_slots.sum().item())
        self._last_entity_binding_metrics = {
            "train/entity_binding_active": float(active_count > 0),
            "train/entity_binding_valid_slot_count": float(valid_slots.sum().item()),
            "train/entity_binding_matched_slot_count": float(active_count),
            "train/entity_binding_unique_id_count": float(unique_count),
            "train/entity_binding_id_collision_count": float(collision_count),
            "train/entity_binding_dropout_fraction": float(
                dropped_slots.sum().item() / max(active_count, 1)
            ),
            "train/entity_binding_gate_tanh": float(gate.detach().float().item()),
            "train/entity_binding_residual_ratio_mean": ratio_mean,
            "train/entity_binding_residual_ratio_max": ratio_max,
            "train/entity_binding_cap_applied_fraction": cap_fraction,
            "train/entity_binding_cap_scale_min": cap_scale_min,
            **self._last_entity_text_binding_metrics,
        }
        return output

    def forward(
        self,
        object_latent_tokens: torch.Tensor,
        *,
        object_valid_mask: torch.Tensor | None = None,
        bbox_xyxy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if str(getattr(self, "_input_ablation", "none")).strip().lower() in {"zero_entity", "zero_all"}:
            bound_tokens = object_latent_tokens
        else:
            bound_tokens = self.apply_entity_binding(
                object_latent_tokens,
                object_valid_mask=object_valid_mask,
            )
        return super().forward(
            bound_tokens,
            object_valid_mask=object_valid_mask,
            bbox_xyxy=bbox_xyxy,
        )


def upgrade_object_condition_adapter(
    old_adapter: ObjectConditionAdapter,
    *,
    entity_bottleneck_dim: int = 256,
    entity_gate_init: float = 0.1,
    entity_dropout_prob: float = 0.2,
    entity_residual_max_ratio: float = 0.1,
    trainable: bool | None = None,
) -> EntityIDBindingObjectConditionAdapter:
    """Upgrade an existing adapter without changing its legacy state-dict keys."""
    old_param = next(old_adapter.parameters())
    upgraded = EntityIDBindingObjectConditionAdapter(
        dim=int(old_adapter.dim),
        num_slots=int(old_adapter.num_slots),
        max_time_steps=int(old_adapter.max_time_steps),
        entity_text_dim=int(old_adapter.dim),
        entity_bottleneck_dim=int(entity_bottleneck_dim),
        entity_gate_init=float(entity_gate_init),
        entity_dropout_prob=float(entity_dropout_prob),
        entity_residual_max_ratio=float(entity_residual_max_ratio),
    ).to(device=old_param.device, dtype=old_param.dtype)
    load_info = upgraded.load_state_dict(old_adapter.state_dict(), strict=False)
    if load_info.unexpected_keys:
        raise RuntimeError(
            f"unexpected old ObjectConditionAdapter keys: {load_info.unexpected_keys}"
        )
    upgraded.mlp_residual_max_ratio = old_adapter.mlp_residual_max_ratio
    upgraded.train(old_adapter.training)
    if trainable is None:
        trainable = any(param.requires_grad for param in old_adapter.parameters())
    upgraded.requires_grad_(bool(trainable))
    upgraded.entity_binding_gate = nn.Parameter(
        upgraded.entity_binding_gate.detach().float(),
        requires_grad=bool(trainable),
    )
    return upgraded
