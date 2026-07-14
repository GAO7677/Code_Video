"""Scheme-C replay training with hard-routed text-to-object entity IDs."""
from __future__ import annotations

import argparse
import re
from typing import Any

import torch

import code_vjepa_vggt.train_v_newtrain as tvn
import code_vjepa_vggt.train0705_kubric_no_gt_box.train_stage1b_no_gt_box_replay_preserve as replay
from code_vjepa_vggt.headonly_val_loss import HeadOnlyValConfig
from code_vjepa_vggt.models.object_entity_id_binder import (
    EntityIDBindingObjectConditionAdapter,
    find_subsequence_spans,
    upgrade_object_condition_adapter,
)

from diffsynth.diffusion import ModelLogger


_PHRASE_ALIAS_GROUPS = (
    ("ball", "sphere", "round rigid object"),
    ("block", "box", "cube", "box shaped rigid object"),
    ("puck", "flat round rigid object"),
    ("cylinder", "cylindrical rigid object"),
    ("capsule", "capsule shaped rigid object"),
    ("person", "man", "woman", "boy", "girl"),
    ("car", "vehicle"),
)


def _normalize_grounding_phrase(phrase: str) -> str:
    value = str(phrase).strip().lower().replace("_", " ")
    value = re.sub(r"\b(?:empty mask|box) fallback\b.*$", "", value)
    value = re.sub(r"\bmotion (?:component \d+|proxy)\b", "", value)
    value = re.sub(r"[^a-z0-9 -]+", " ", value)
    return " ".join(value.split())


def _phrase_candidates(phrase: str) -> list[str]:
    normalized = _normalize_grounding_phrase(phrase)
    if not normalized:
        return []
    candidates = [normalized]
    normalized_words = set(normalized.split())
    for aliases in _PHRASE_ALIAS_GROUPS:
        if normalized in aliases or normalized_words.intersection(aliases):
            candidates.extend(aliases)
    deduped: list[str] = []
    for candidate in candidates:
        candidate = " ".join(candidate.split())
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _tokenize_without_padding(tokenizer, text: str) -> list[int]:
    ids = tokenizer(
        text,
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )
    return [int(value) for value in ids.reshape(-1).tolist()]


def _pool_phrase_from_prompt_context(
    *,
    prompt_token_ids: list[int],
    prompt_context: torch.Tensor,
    tokenizer,
    phrase: str,
) -> tuple[torch.Tensor | None, int, str | None]:
    for candidate in _phrase_candidates(phrase):
        candidate_ids = _tokenize_without_padding(tokenizer, candidate)
        spans = find_subsequence_spans(prompt_token_ids, candidate_ids)
        if not spans:
            continue
        token_indices = sorted(
            {
                token_id
                for start, end in spans
                for token_id in range(int(start), int(end))
            }
        )
        if not token_indices:
            continue
        pooled = prompt_context[:, token_indices, :].mean(dim=1)
        return pooled, len(spans), candidate
    return None, 0, None


class EntityIDBindingReplayPreserveWanModule(replay.ReplayPreserveNoGTBoxWanModule):
    def __init__(self, *args, **kwargs) -> None:
        self.entity_binding_enabled = bool(kwargs.pop("entity_binding_enabled", True))
        raw_sources = kwargs.pop("entity_binding_sources", "pybullet,kubric")
        self.entity_binding_sources = {
            item.strip().lower()
            for item in str(raw_sources).split(",")
            if item.strip()
        }
        self.entity_binding_bottleneck_dim = int(
            kwargs.pop("entity_binding_bottleneck_dim", 256)
        )
        self.entity_binding_gate_init = float(
            kwargs.pop("entity_binding_gate_init", 0.1)
        )
        self.entity_binding_dropout_prob = float(
            kwargs.pop("entity_binding_dropout_prob", 0.2)
        )
        self.entity_binding_residual_max_ratio = float(
            kwargs.pop("entity_binding_residual_max_ratio", 0.1)
        )
        self.entity_binding_randomize_ids = bool(
            kwargs.pop("entity_binding_randomize_ids", True)
        )
        super().__init__(*args, **kwargs)

        self._entity_binding_runtime: dict[str, Any] | None = None
        self._entity_binding_prepare_metrics: dict[str, float] = {}
        if not self.enable_object_branch or self.object_adapter is None:
            return

        self.object_adapter = upgrade_object_condition_adapter(
            self.object_adapter,
            entity_bottleneck_dim=self.entity_binding_bottleneck_dim,
            entity_gate_init=self.entity_binding_gate_init,
            entity_dropout_prob=self.entity_binding_dropout_prob,
            entity_residual_max_ratio=self.entity_binding_residual_max_ratio,
            trainable=bool(self.train_object_adapter),
        )

    @property
    def entity_bound_adapter(self) -> EntityIDBindingObjectConditionAdapter:
        if not isinstance(self.object_adapter, EntityIDBindingObjectConditionAdapter):
            raise RuntimeError("entity-bound object adapter is not initialized")
        return self.object_adapter

    def _prepare_entity_bindings(
        self,
        *,
        object_valid_mask: torch.Tensor,
    ) -> None:
        runtime = self._entity_binding_runtime
        if runtime is None:
            return
        prompt_context = runtime["prompt_context"]
        prompt = str(runtime["prompt"])
        tokenizer = runtime["tokenizer"]
        phrases = list(
            (getattr(self, "_last_grounding_debug", {}) or {}).get(
                "object_phrases", []
            )
        )
        prompt_ids, prompt_mask = tokenizer(
            prompt,
            return_mask=True,
            add_special_tokens=True,
        )
        valid_length = int(prompt_mask[0].sum().item())
        prompt_token_ids = [
            int(value) for value in prompt_ids[0, :valid_length].tolist()
        ]
        if int(prompt_context.shape[0]) != 1:
            raise ValueError("entity ID binder currently expects batch size 1")
        if int(prompt_context.shape[-1]) != int(self.entity_bound_adapter.entity_text_dim):
            raise ValueError(
                f"T5 context dim={prompt_context.shape[-1]} does not match object dim="
                f"{self.entity_bound_adapter.entity_text_dim}"
            )

        slots = int(self.aux_max_objects)
        valid_slots = [
            int(slot)
            for slot in torch.nonzero(
                object_valid_mask[0] > 0.5,
                as_tuple=False,
            ).flatten().tolist()
        ]
        if self.entity_binding_randomize_ids and valid_slots:
            randomized_ids = torch.randperm(len(valid_slots)).tolist()
        else:
            randomized_ids = list(range(len(valid_slots)))

        entity_text_by_id = prompt_context.new_zeros(
            (1, slots, int(prompt_context.shape[-1]))
        )
        entity_match_mask = torch.zeros(
            (1, slots),
            device=prompt_context.device,
            dtype=torch.bool,
        )
        slot_entity_ids = torch.full(
            (1, slots),
            -1,
            device=prompt_context.device,
            dtype=torch.long,
        )
        matched_phrases = 0
        matched_spans = 0
        matched_candidates: set[str] = set()
        for local_index, slot_id in enumerate(valid_slots):
            entity_id = int(randomized_ids[local_index])
            slot_entity_ids[0, slot_id] = entity_id
            phrase = phrases[slot_id] if slot_id < len(phrases) else ""
            pooled, span_count, matched_candidate = _pool_phrase_from_prompt_context(
                prompt_token_ids=prompt_token_ids,
                prompt_context=prompt_context,
                tokenizer=tokenizer,
                phrase=phrase,
            )
            if pooled is None:
                continue
            entity_text_by_id[0, entity_id] = pooled[0]
            entity_match_mask[0, entity_id] = True
            matched_phrases += 1
            matched_spans += int(span_count)
            if matched_candidate is not None:
                matched_candidates.add(matched_candidate)

        self.entity_bound_adapter.set_entity_binding_context(
            entity_text_by_id=entity_text_by_id,
            entity_text_match_mask=entity_match_mask,
            slot_entity_ids=slot_entity_ids,
        )
        self._entity_binding_prepare_metrics = {
            "train/entity_binding_grounding_phrase_count": float(len(phrases)),
            "train/entity_binding_prompt_matched_phrase_count": float(matched_phrases),
            "train/entity_binding_prompt_matched_span_count": float(matched_spans),
            "train/entity_binding_prompt_unique_candidate_count": float(
                len(matched_candidates)
            ),
            "train/entity_binding_id_randomized": float(
                self.entity_binding_randomize_ids
            ),
        }

    def _build_object_query_priors(self, sample: dict, *, image_hw: tuple[int, int]):
        outputs = super()._build_object_query_priors(sample, image_hw=image_hw)
        if self._entity_binding_runtime is not None:
            self._prepare_entity_bindings(object_valid_mask=outputs[2])
        return outputs

    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        source = self._dataset_source(inputs_shared)
        source_enabled = bool(
            self.entity_binding_enabled and source in self.entity_binding_sources
        )
        self._entity_binding_prepare_metrics = {
            "train/entity_binding_source_enabled": float(source_enabled),
            "train/entity_binding_grounding_phrase_count": 0.0,
            "train/entity_binding_prompt_matched_phrase_count": 0.0,
            "train/entity_binding_prompt_matched_span_count": 0.0,
            "train/entity_binding_prompt_unique_candidate_count": 0.0,
            "train/entity_binding_id_randomized": float(
                self.entity_binding_randomize_ids
            ),
        }
        if self.enable_object_branch and self.object_adapter is not None:
            self.entity_bound_adapter.clear_entity_binding_context()
            self.entity_bound_adapter.pop_entity_binding_metrics()

        sample = inputs_shared.get("raw_sample", {})
        if source_enabled and "context" in inputs_posi:
            self._entity_binding_runtime = {
                "prompt_context": inputs_posi["context"],
                "prompt": str(sample.get("caption", "")),
                "tokenizer": pipe.tokenizer,
            }
        else:
            self._entity_binding_runtime = None
        try:
            total, metrics = super()._compute_object_losses(
                pipe,
                inputs_shared,
                inputs_posi,
            )
            if self.enable_object_branch and self.object_adapter is not None:
                metrics.update(self.entity_bound_adapter.pop_entity_binding_metrics())
            metrics.update(self._entity_binding_prepare_metrics)
            return total, metrics
        finally:
            self._entity_binding_runtime = None
            if self.enable_object_branch and self.object_adapter is not None:
                self.entity_bound_adapter.clear_entity_binding_context()


def build_parser() -> argparse.ArgumentParser:
    parser = replay.build_parser()
    parser.description = (
        "Scheme-C replay-preservation training with hard-routed text/object entity IDs."
    )
    group = parser.add_argument_group("entity_id_binding")
    group.add_argument(
        "--disable_entity_id_binding",
        action="store_true",
        help="Disable the entity binder while retaining the same training entrypoint.",
    )
    group.add_argument(
        "--entity_binding_sources",
        default="pybullet,kubric",
        help="Comma-separated dataset sources using phrase-to-slot binding.",
    )
    group.add_argument("--entity_binding_bottleneck_dim", type=int, default=256)
    group.add_argument("--entity_binding_gate_init", type=float, default=0.1)
    group.add_argument("--entity_binding_dropout_prob", type=float, default=0.2)
    group.add_argument("--entity_binding_residual_max_ratio", type=float, default=0.1)
    group.add_argument(
        "--disable_entity_binding_id_randomization",
        action="store_true",
        help="Use slot-order entity IDs instead of a fresh per-sample permutation.",
    )
    return parser


def build_model(
    args: argparse.Namespace,
    accelerator,
) -> EntityIDBindingReplayPreserveWanModule:
    original_factory = replay.ReplayPreserveNoGTBoxWanModule

    def entity_factory(*model_args, **model_kwargs):
        return EntityIDBindingReplayPreserveWanModule(
            *model_args,
            **model_kwargs,
            entity_binding_enabled=not args.disable_entity_id_binding,
            entity_binding_sources=args.entity_binding_sources,
            entity_binding_bottleneck_dim=args.entity_binding_bottleneck_dim,
            entity_binding_gate_init=args.entity_binding_gate_init,
            entity_binding_dropout_prob=args.entity_binding_dropout_prob,
            entity_binding_residual_max_ratio=(
                args.entity_binding_residual_max_ratio
            ),
            entity_binding_randomize_ids=(
                not args.disable_entity_binding_id_randomization
            ),
        )

    replay.ReplayPreserveNoGTBoxWanModule = entity_factory
    try:
        model = replay.build_model(args, accelerator)
    finally:
        replay.ReplayPreserveNoGTBoxWanModule = original_factory
    if not isinstance(model, EntityIDBindingReplayPreserveWanModule):
        raise TypeError(f"unexpected model type: {type(model).__name__}")
    return model


def main() -> None:
    args = tvn.prepare_args(build_parser().parse_args())
    if args.stage2_init_from is not None and args.stage2_resume_from is not None:
        raise ValueError("--stage2_init_from and --stage2_resume_from are mutually exclusive")
    previous_handlers = tvn.install_interrupt_handlers()
    accelerator = tvn.build_accelerator(args)
    tvn.init_trackers(accelerator, args)

    dataset = replay.build_dataset(args)
    if accelerator.is_main_process and hasattr(dataset, "dataset_stats"):
        accelerator.print(f"Replay mixture: {dataset.dataset_stats}")
    disabled_val = HeadOnlyValConfig(
        enabled=False,
        split="val",
        every_steps=None,
        num_batches=1,
    )
    model = build_model(args, accelerator)

    if args.stage1a_init_from is not None:
        info = tvn._load_filtered_checkpoint_into_model(
            model,
            args.stage1a_init_from,
            include_prefixes=("object_pooler.", "object_aux_heads."),
        )
        accelerator.print(
            "Loaded Stage1A token builder: "
            f"loaded={info['loaded_count']} "
            f"shape_mismatch={len(info['skipped_shape_mismatch'])}"
        )
    stage2_source = args.stage2_init_from or args.stage2_resume_from
    if stage2_source is not None:
        info = replay._load_stage2_trainables(model, stage2_source)
        mode = "model-only initialization (fresh optimizer)" if args.stage2_init_from else "resume"
        accelerator.print(
            f"Loaded Stage1B {mode}: source={stage2_source} "
            f"loaded={info['loaded_count']} "
            f"shape_mismatch={len(info['skipped_shape_mismatch'])}"
        )

    replay.base._log_stage_summary(accelerator, model, args)
    accelerator.print(
        "Entity ID binding: "
        f"enabled={not args.disable_entity_id_binding}, "
        f"sources={args.entity_binding_sources}, "
        f"randomize_ids={not args.disable_entity_binding_id_randomization}, "
        f"bottleneck={args.entity_binding_bottleneck_dim}, "
        f"gate_init={args.entity_binding_gate_init:.4f}, "
        f"dropout={args.entity_binding_dropout_prob:.4f}, "
        f"residual_max_ratio={args.entity_binding_residual_max_ratio:.4f}"
    )
    model_logger = ModelLogger(
        tvn.get_checkpoint_dir(args),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    runtime_state: dict[str, Any] = {}
    try:
        tvn.train_loop(
            accelerator,
            dataset,
            model,
            model_logger,
            args,
            runtime_state=runtime_state,
            headonly_val_dataloader=None,
            headonly_val_config=disabled_val,
        )
    except (KeyboardInterrupt, tvn.TrainingInterrupted) as exc:
        checkpoint_root = tvn.get_checkpoint_dir(args)
        model_logger.save_model(
            accelerator,
            model,
            tvn.training_checkpoint_file(checkpoint_root, "interrupted-latest"),
        )
        optimizer = runtime_state.get("optimizer")
        scheduler = runtime_state.get("scheduler")
        progress = runtime_state.get("progress", {})
        if optimizer is not None and scheduler is not None:
            tvn.save_training_state(
                accelerator=accelerator,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=progress.get("global_step", 0),
                epoch_id=progress.get("epoch_id", 0),
                batch_in_epoch=progress.get("batch_in_epoch", 0),
                model_logger=model_logger,
                state_path=tvn.training_state_file(
                    checkpoint_root,
                    "interrupted-latest",
                ),
            )
        accelerator.end_training()
        tvn.restore_interrupt_handlers(previous_handlers)
        raise SystemExit(130) from exc

    accelerator.end_training()
    tvn.restore_interrupt_handlers(previous_handlers)


if __name__ == "__main__":
    main()
