"""Diagnostic interventions for Scheme-C text-to-object entity binding.

This wrapper leaves grounding, tracking, object pooling, and object-slot order
unchanged. It only changes the text/entity route into those fixed slots and the
effective entity-binding gate used by the object adapter.
"""
from __future__ import annotations

import math
import sys
from typing import Any

import torch

from code_vjepa_vggt.models.object_entity_id_binder import (
    EntityIDBindingObjectConditionAdapter,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_entity_id_binding_v2v as entity_v2v,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_scheme_c_entity_caption_physical_v2v as scheme_v2v,
)


_VALID_MAP_MODES = {"correct", "swapped", "disabled"}
_MAP_MODE = "correct"
_GATE_SCALE = 1.0


def _pop_option(argv: list[str], option: str, default: str) -> str:
    if option not in argv:
        return default
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise ValueError(f"{option} requires a value")
    value = argv[index + 1]
    del argv[index : index + 2]
    return value


def _parse_intervention_args() -> None:
    global _MAP_MODE, _GATE_SCALE
    _MAP_MODE = _pop_option(
        sys.argv, "--entity-binding-map-mode", "correct"
    ).strip().lower()
    if _MAP_MODE not in _VALID_MAP_MODES:
        raise ValueError(
            f"--entity-binding-map-mode must be one of {sorted(_VALID_MAP_MODES)}, "
            f"got {_MAP_MODE!r}"
        )
    _GATE_SCALE = float(
        _pop_option(sys.argv, "--entity-binding-gate-scale", "1.0")
    )
    if not math.isfinite(_GATE_SCALE) or _GATE_SCALE < 0.0:
        raise ValueError("--entity-binding-gate-scale must be finite and non-negative")


def _install_gate_intervention() -> None:
    original_apply = EntityIDBindingObjectConditionAdapter.apply_entity_binding

    def apply_with_scaled_gate(
        self: EntityIDBindingObjectConditionAdapter,
        object_latent_tokens: torch.Tensor,
        *,
        object_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        raw_parameter = self.entity_binding_gate.detach().float().clone()
        checkpoint_gate = float(torch.tanh(raw_parameter).item())
        effective_gate = max(-0.999999, min(0.999999, checkpoint_gate * _GATE_SCALE))
        effective_parameter = math.atanh(effective_gate)
        with torch.no_grad():
            self.entity_binding_gate.copy_(
                torch.as_tensor(
                    effective_parameter,
                    device=self.entity_binding_gate.device,
                    dtype=self.entity_binding_gate.dtype,
                )
            )
        try:
            return original_apply(
                self,
                object_latent_tokens,
                object_valid_mask=object_valid_mask,
            )
        finally:
            with torch.no_grad():
                self.entity_binding_gate.copy_(raw_parameter)

    EntityIDBindingObjectConditionAdapter.apply_entity_binding = apply_with_scaled_gate


def _install_mapping_intervention() -> None:
    original_install = entity_v2v._install_binding_for_grounded_slots

    def install_with_mapping(*args: Any, **kwargs: Any) -> dict[str, object]:
        debug = original_install(*args, **kwargs)
        model = args[0] if args else kwargs["model"]
        adapter = model.object_adapter
        if not isinstance(adapter, EntityIDBindingObjectConditionAdapter):
            raise TypeError("entity-binding intervention requires the entity adapter")

        original_ids = adapter._slot_entity_ids
        if original_ids is None:
            raise RuntimeError("entity binding context was not installed")
        routed_ids = original_ids.clone()
        valid_positions = torch.nonzero(
            routed_ids[0] >= 0, as_tuple=False
        ).flatten()

        if _MAP_MODE == "disabled":
            routed_ids.fill_(-1)
        elif _MAP_MODE == "swapped":
            if int(valid_positions.numel()) != 2:
                raise RuntimeError(
                    "swapped intervention requires exactly two matched slots; "
                    f"found {int(valid_positions.numel())}"
                )
            first, second = [int(value) for value in valid_positions.tolist()]
            first_id = routed_ids[0, first].clone()
            routed_ids[0, first] = routed_ids[0, second]
            routed_ids[0, second] = first_id

        adapter._slot_entity_ids = routed_ids.detach()
        matched = debug.get("matched", [])
        effective_routes = []
        for record in matched:
            slot_id = int(record["slot_id"])
            effective_routes.append(
                {
                    "slot_id": slot_id,
                    "grounding_phrase": str(record.get("grounding_phrase", "")),
                    "original_entity_id": int(record["entity_id"]),
                    "effective_entity_id": int(routed_ids[0, slot_id].item()),
                }
            )
        debug.update(
            {
                "intervention": {
                    "map_mode": _MAP_MODE,
                    "gate_scale": float(_GATE_SCALE),
                    "checkpoint_gate_tanh": float(
                        torch.tanh(adapter.entity_binding_gate.detach().float()).item()
                    ),
                    "effective_routes": effective_routes,
                    "original_slot_entity_ids": original_ids.detach().cpu().tolist(),
                    "effective_slot_entity_ids": routed_ids.detach().cpu().tolist(),
                }
            }
        )
        return debug

    entity_v2v._install_binding_for_grounded_slots = install_with_mapping


def main() -> None:
    _parse_intervention_args()
    _install_gate_intervention()
    _install_mapping_intervention()
    print(
        "[entity-binding-intervention] "
        f"map_mode={_MAP_MODE} gate_scale={_GATE_SCALE:.6g}"
    )
    scheme_v2v.main()


if __name__ == "__main__":
    main()
