"""Batch v2v inference for entity-ID-bound Stage1B checkpoints.

The command-line interface is identical to
``wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.py``.
"""
from __future__ import annotations

import os
import sys
import types

_DIFFSYNTH_ROOT = os.environ.get(
    "DIFFSYNTH_ROOT",
    "/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main",
)
if _DIFFSYNTH_ROOT not in sys.path:
    sys.path.insert(0, _DIFFSYNTH_ROOT)

import torch

from code_vjepa_vggt.models.object_entity_id_binder import (
    EntityIDBindingObjectConditionAdapter,
    upgrade_object_condition_adapter,
)
from code_vjepa_vggt.train0705 import (
    infer_stage1b_context_only_no_gt_box_v_newtrain0705 as infer0705,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    infer_stage1b_context_only_no_gt_box_v_newtrain_kubric as kubric_infer,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    train_stage1b_no_gt_box_replay_preserve_entity_id_binding as binding_train,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as batch_base,
)


_ORIGINAL_BUILD_MODEL = kubric_infer.trainmod.build_model
_ORIGINAL_BUILD_OBJECT_CONTEXT = kubric_infer._build_object_context


def _build_entity_bound_model(args, accelerator):
    model = _ORIGINAL_BUILD_MODEL(args, accelerator)
    if bool(getattr(model, "enable_object_branch", False)):
        model.object_adapter = upgrade_object_condition_adapter(
            model.object_adapter,
            entity_bottleneck_dim=256,
            entity_gate_init=0.1,
            entity_dropout_prob=0.0,
            entity_residual_max_ratio=0.1,
            trainable=bool(getattr(model, "train_object_adapter", False)),
        )
    return model


@torch.no_grad()
def _encode_prompt_context(pipe, prompt: str) -> torch.Tensor:
    pipe.load_models_to_device(["text_encoder"])
    token_ids, token_mask = pipe.tokenizer(
        str(prompt),
        return_mask=True,
        add_special_tokens=True,
    )
    token_ids = token_ids.to(pipe.device)
    token_mask = token_mask.to(pipe.device)
    context = pipe.text_encoder(token_ids, token_mask)
    valid_length = int(token_mask[0].sum().item())
    context[:, valid_length:] = 0
    return context


def _install_binding_for_grounded_slots(
    model,
    *,
    prompt: str,
    prompt_context: torch.Tensor,
    object_valid_mask: torch.Tensor,
) -> dict[str, object]:
    adapter = model.object_adapter
    if not isinstance(adapter, EntityIDBindingObjectConditionAdapter):
        raise TypeError("inference model does not use the entity-bound adapter")
    phrases = list(
        (getattr(model, "_last_grounding_debug", {}) or {}).get(
            "object_phrases", []
        )
    )
    prompt_ids, prompt_mask = model.pipe.tokenizer(
        str(prompt),
        return_mask=True,
        add_special_tokens=True,
    )
    valid_length = int(prompt_mask[0].sum().item())
    prompt_token_ids = [
        int(value) for value in prompt_ids[0, :valid_length].tolist()
    ]
    slots = int(model.aux_max_objects)
    entity_text_by_id = prompt_context.new_zeros(
        (1, slots, int(prompt_context.shape[-1]))
    )
    entity_match_mask = torch.zeros(
        (1, slots),
        dtype=torch.bool,
        device=prompt_context.device,
    )
    slot_entity_ids = torch.full(
        (1, slots),
        -1,
        dtype=torch.long,
        device=prompt_context.device,
    )
    valid_slots = [
        int(slot)
        for slot in torch.nonzero(
            object_valid_mask[0] > 0.5,
            as_tuple=False,
        ).flatten().tolist()
    ]
    matched: list[dict[str, object]] = []
    unmatched: list[dict[str, object]] = []
    for entity_id, slot_id in enumerate(valid_slots):
        slot_entity_ids[0, slot_id] = int(entity_id)
        phrase = phrases[slot_id] if slot_id < len(phrases) else ""
        pooled, span_count, candidate = binding_train._pool_phrase_from_prompt_context(
            prompt_token_ids=prompt_token_ids,
            prompt_context=prompt_context,
            tokenizer=model.pipe.tokenizer,
            phrase=phrase,
        )
        record = {
            "entity_id": int(entity_id),
            "slot_id": int(slot_id),
            "grounding_phrase": str(phrase),
        }
        if pooled is None:
            unmatched.append(record)
            continue
        entity_text_by_id[0, entity_id] = pooled[0]
        entity_match_mask[0, entity_id] = True
        record.update(
            {
                "matched_candidate": candidate,
                "prompt_span_count": int(span_count),
            }
        )
        matched.append(record)

    adapter.set_entity_binding_context(
        entity_text_by_id=entity_text_by_id,
        entity_text_match_mask=entity_match_mask,
        slot_entity_ids=slot_entity_ids,
    )
    return {
        "enabled": True,
        "id_policy": "deterministic_valid_slot_order",
        "matched": matched,
        "unmatched": unmatched,
        "slot_entity_ids": slot_entity_ids.detach().cpu().tolist(),
    }


def _build_object_context_with_entity_binding(
    model,
    *,
    context_video_single: torch.Tensor,
    prompt: str,
    video_path: str,
):
    if not bool(getattr(model, "enable_object_branch", False)):
        return _ORIGINAL_BUILD_OBJECT_CONTEXT(
            model,
            context_video_single=context_video_single,
            prompt=prompt,
            video_path=video_path,
        )
    adapter = model.object_adapter
    if not isinstance(adapter, EntityIDBindingObjectConditionAdapter):
        raise TypeError("entity-binding checkpoint requires EntityIDBindingObjectConditionAdapter")

    prompt_context = _encode_prompt_context(model.pipe, str(prompt))
    original_query_builder = model._build_object_query_priors
    binding_debug: dict[str, object] = {"enabled": False}

    def query_builder_with_binding(self, sample, *, image_hw):
        nonlocal binding_debug
        outputs = original_query_builder(sample, image_hw=image_hw)
        binding_debug = _install_binding_for_grounded_slots(
            self,
            prompt=str(prompt),
            prompt_context=prompt_context,
            object_valid_mask=outputs[2],
        )
        return outputs

    model._build_object_query_priors = types.MethodType(
        query_builder_with_binding,
        model,
    )
    try:
        object_context, debug = _ORIGINAL_BUILD_OBJECT_CONTEXT(
            model,
            context_video_single=context_video_single,
            prompt=prompt,
            video_path=video_path,
        )
        binding_debug["adapter_metrics"] = adapter.pop_entity_binding_metrics()
        debug["entity_id_binding"] = binding_debug
        return object_context, debug
    finally:
        model._build_object_query_priors = original_query_builder
        adapter.clear_entity_binding_context()


def _install_entity_runtime_hooks() -> None:
    kubric_infer.trainmod.build_model = _build_entity_bound_model
    infer0705.t0705 = kubric_infer.trainmod
    infer0705._build_object_context = _build_object_context_with_entity_binding
    infer0705._build_model_args = kubric_infer._build_model_args


def main() -> None:
    batch_base._install_kubric_runtime_hooks = _install_entity_runtime_hooks
    batch_base.main()


if __name__ == "__main__":
    main()
