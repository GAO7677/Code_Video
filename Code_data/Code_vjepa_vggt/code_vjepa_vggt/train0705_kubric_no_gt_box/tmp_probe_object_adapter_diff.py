from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from code_vjepa_vggt import batch_infer_v_newtrain_from_jsonl as core
from code_vjepa_vggt.models.object_condition_adapter import ObjectConditionAdapter
from code_vjepa_vggt.train0705 import infer_stage1b_context_only_no_gt_box_v_newtrain0705 as infer0705
from code_vjepa_vggt.utils.vggt_cache import load_vggt_cache
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, read_video_prefix, read_video_uniform


DEFAULT_INPUT_JSON = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/"
    "physicIQ_026_Solid_Mechanics_0005_perspective-center_trimmed-ball-behind-rotating-paper.json"
)
DEFAULT_CHECKPOINT_A = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
    "train_stage1b_kubric0708/checkpoints/step-003500"
)
DEFAULT_CHECKPOINT_B = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
    "train_stage1b_kubric0708/checkpoints/step-005000"
)
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/query_prior_compare_20260710")
DEFAULT_CASE_NAME = "physicIQ_026_object_adapter_probe_step003500_vs_step005000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe ObjectConditionAdapter internals for two stage1b checkpoints "
            "using the same precomputed object_latent_tokens."
        )
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--checkpoint-a", type=Path, default=DEFAULT_CHECKPOINT_A)
    parser.add_argument("--checkpoint-b", type=Path, default=DEFAULT_CHECKPOINT_B)
    parser.add_argument("--label-a", default="step-003500")
    parser.add_argument("--label-b", default="step-005000")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case-name", default=DEFAULT_CASE_NAME)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--context-frames", type=int, default=20)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aux-device", default=None)
    parser.add_argument("--grounding-device", default=None)
    parser.add_argument("--vggt-cache-root", default=None)
    parser.add_argument("--initialize-model-on-cpu", action="store_true")
    parser.add_argument("--wan-root", default="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
    parser.add_argument("--diffsynth-root", default="/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
    parser.add_argument(
        "--lora-checkpoint",
        default="/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors",
    )
    parser.add_argument(
        "--stage1a-init-from",
        default="/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt",
    )
    return parser.parse_args()


def _resolve_runtime_device(device_arg: str) -> str:
    if str(device_arg).strip() and str(device_arg).strip().lower() != "cuda":
        return str(device_arg).strip()
    return infer0705._resolve_launch_device()


def _build_infer_args(
    args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
    output_dir: Path,
) -> argparse.Namespace:
    old_argv = list(sys.argv)
    try:
        sys.argv = [
            old_argv[0],
            "--checkpoint",
            str(checkpoint_dir),
            "--context-video",
            "/tmp/dummy_context.mp4",
            "--prompt",
            "dummy",
            "--output-dir",
            str(output_dir),
        ]
        infer_args = infer0705.parse_args()
    finally:
        sys.argv = old_argv

    infer_args.checkpoint = str(checkpoint_dir)
    infer_args.output_dir = str(output_dir)
    infer_args.device = _resolve_runtime_device(args.device)
    infer_args.height = int(args.height)
    infer_args.width = int(args.width)
    infer_args.context_frames = int(args.context_frames)
    infer_args.fps = int(args.fps)
    infer_args.seed = int(args.seed)
    infer_args.wan_root = str(args.wan_root)
    infer_args.diffsynth_root = str(args.diffsynth_root)
    infer_args.lora_checkpoint = str(args.lora_checkpoint)
    infer_args.stage1a_init_from = str(args.stage1a_init_from)
    infer_args.grounding_device = args.grounding_device
    infer_args.aux_device = args.aux_device
    infer_args.vggt_cache_root = args.vggt_cache_root
    infer_args.initialize_model_on_cpu = bool(args.initialize_model_on_cpu)
    infer_args.object_context_ablation = "none"
    return infer_args


def _load_context_video(
    *,
    source_video: Path,
    sampling_mode: str,
    context_frames: int,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_mode == "uniform":
        frames, frame_indices = read_video_uniform(source_video, int(context_frames))
    else:
        frames, frame_indices = read_video_prefix(source_video, int(context_frames))
    context_video_single = preprocess_video_rgb_uint8(
        frames,
        (int(height), int(width)),
        resize_mode="cover_crop",
        cover_crop_hw=(int(height), int(width)),
    )
    return context_video_single, frame_indices


def _tensor_stats(tensor: torch.Tensor) -> dict[str, object]:
    cpu = tensor.detach().cpu().float().contiguous()
    return {
        "shape": list(cpu.shape),
        "mean": float(cpu.mean().item()),
        "std": float(cpu.std(unbiased=False).item()) if cpu.numel() > 1 else 0.0,
        "abs_mean": float(cpu.abs().mean().item()),
        "abs_max": float(cpu.abs().max().item()),
        "min": float(cpu.min().item()),
        "max": float(cpu.max().item()),
    }


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, object]:
    a_cpu = a.detach().cpu().float().contiguous()
    b_cpu = b.detach().cpu().float().contiguous()
    diff = (a_cpu - b_cpu).abs()
    l2 = torch.linalg.vector_norm((a_cpu - b_cpu).reshape(-1), ord=2)
    ref_l2 = torch.linalg.vector_norm(a_cpu.reshape(-1), ord=2)
    return {
        "shape": list(a_cpu.shape),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "l2_diff": float(l2.item()),
        "relative_l2_vs_a": float((l2 / ref_l2).item()) if float(ref_l2.item()) > 0.0 else 0.0,
    }


def _extract_latents_and_template(
    args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    context_video_single: torch.Tensor,
    prompt: str,
    source_video: Path,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    infer_args = _build_infer_args(args, checkpoint_dir=checkpoint_dir, output_dir=output_dir)
    torch.manual_seed(int(infer_args.seed))
    model, _, load_info = infer0705._build_runtime_model(infer_args)
    try:
        pipe = model.pipe
        device = torch.device(pipe.device)
        context_video = context_video_single.unsqueeze(0).to(device=device, dtype=pipe.torch_dtype)
        image_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))
        sample = {
            "context_video": context_video_single,
            "num_context_frames": int(context_video_single.shape[1]),
            "caption": prompt,
            "video_path": str(source_video),
        }

        query_points_prior, query_frame_ids, object_valid_mask, box_prior_xyxy = model._build_object_query_priors(
            sample,
            image_hw=image_hw,
        )
        query_points_prior = query_points_prior.to(device=device, dtype=pipe.torch_dtype)
        query_frame_ids = query_frame_ids.to(device=device, dtype=pipe.torch_dtype)
        object_valid_mask = object_valid_mask.to(device=device, dtype=pipe.torch_dtype)
        box_prior_xyxy = box_prior_xyxy.to(device=device, dtype=pipe.torch_dtype)

        frames_bthwc_01 = ((context_video.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0.0, 1.0)
        cotracker_out = model._run_cotracker(
            frames_bthwc_01,
            query_points_prior=query_points_prior,
            query_frame_ids=query_frame_ids,
            query_image_hw=image_hw,
        )
        tracks_grouped, visibility_grouped, confidence_grouped = model._group_tracks_to_objects(
            cotracker_out.tracks,
            cotracker_out.visibility,
            cotracker_out.confidence,
            max_objects=model.aux_max_objects,
            points_per_object=model.object_num_queries,
        )
        if model.vggt_cache_root:
            vggt_out = load_vggt_cache(sample, model.vggt_cache_root, allow_missing=False)
            if vggt_out is None:
                raise RuntimeError(f"VGGT cache missing for {source_video}")
        elif getattr(model, "vggt_adapter", None) is not None:
            vggt_device = getattr(model.vggt_adapter, "device_obj", device)
            vggt_out = model.vggt_adapter(
                frames_bthwc_01.to(vggt_device),
                query_points_prior=query_points_prior.to(vggt_device),
                query_image_hw=image_hw,
            )
            for attr_name in (
                "query_points",
                "tracks",
                "visibility",
                "confidence",
                "pose_enc",
                "depth",
                "depth_conf",
                "world_points",
                "world_points_conf",
                "dense_patch_tokens",
            ):
                attr_value = getattr(vggt_out, attr_name, None)
                if isinstance(attr_value, torch.Tensor):
                    setattr(vggt_out, attr_name, attr_value.to(device))
        else:
            vggt_out = SimpleNamespace(
                world_points=None,
                world_points_conf=None,
                depth=None,
                depth_conf=None,
                dense_patch_tokens=None,
                patch_grid_hw=None,
                input_hw=None,
                image_hw=None,
            )
        jepa_out = model._run_jepa(context_video)
        context_latents = infer0705._encode_context_latents(pipe, context_video_single)
        object_out = model.object_pooler(
            jepa_patch_tokens=jepa_out.patch_tokens,
            context_latents=context_latents,
            tracks=tracks_grouped,
            visibility=visibility_grouped,
            confidence=confidence_grouped,
            track_image_hw=image_hw,
            object_valid_mask=object_valid_mask,
            box_prior_xyxy=box_prior_xyxy,
            vggt_world_points=getattr(vggt_out, "world_points", None),
            vggt_world_points_conf=getattr(vggt_out, "world_points_conf", None),
            vggt_depth=getattr(vggt_out, "depth", None),
            vggt_depth_conf=getattr(vggt_out, "depth_conf", None),
            vggt_dense_patch_tokens=getattr(vggt_out, "dense_patch_tokens", None),
            vggt_patch_grid_hw=getattr(vggt_out, "patch_grid_hw", None),
            vggt_geometry_image_hw=getattr(vggt_out, "input_hw", None)
            if getattr(vggt_out, "input_hw", None) is not None
            else getattr(vggt_out, "image_hw", None),
            frame_valid_mask=None,
        )
        adapter = model.object_adapter
        template_info = {
            "dim": int(adapter.dim),
            "num_slots": int(adapter.num_slots),
            "max_time_steps": int(adapter.max_time_steps),
            "output_gate_init_loaded": float(torch.sigmoid(adapter.output_gate_logit.detach().float()).item()),
            "load_info": infer0705._summarize_load_info(load_info),
            "object_latent_tokens": _tensor_stats(object_out.object_latent_tokens),
            "object_valid_mask": _tensor_stats(object_valid_mask),
        }
        return (
            object_out.object_latent_tokens.detach().cpu().float(),
            object_valid_mask.detach().cpu().float(),
            template_info,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_object_adapter_from_checkpoint(
    checkpoint_dir: Path,
    *,
    dim: int,
    num_slots: int,
    max_time_steps: int,
) -> ObjectConditionAdapter:
    adapter = ObjectConditionAdapter(
        dim=dim,
        num_slots=num_slots,
        max_time_steps=max_time_steps,
    ).eval()
    state = load_file(str((checkpoint_dir / "checkpoint.safetensors").resolve()), device="cpu")
    adapter_state = {}
    prefix = "object_adapter."
    for key, value in state.items():
        if key.startswith(prefix):
            adapter_state[key[len(prefix) :]] = value.detach().cpu()
    missing, unexpected = adapter.load_state_dict(adapter_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected adapter load mismatch for {checkpoint_dir}: missing={missing}, unexpected={unexpected}"
        )
    return adapter.float().eval()


def _adapter_forward_breakdown(
    adapter: ObjectConditionAdapter,
    *,
    object_latent_tokens: torch.Tensor,
    object_valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    latent = object_latent_tokens.detach().cpu().float()
    valid = object_valid_mask.detach().cpu().float()
    batch, time_steps, slots, dim = latent.shape
    slot_ids = torch.arange(slots, dtype=torch.long)
    time_ids = torch.arange(time_steps, dtype=torch.long)
    slot_bias = adapter.slot_embed(slot_ids).view(1, 1, slots, dim).float()
    time_bias = adapter.time_embed(time_ids).view(1, time_steps, 1, dim).float()
    x0 = torch.nan_to_num(latent, nan=0.0, posinf=0.0, neginf=0.0)
    x1 = x0 + slot_bias + time_bias
    x2 = adapter.norm(x1)
    x3 = adapter.mlp(x2)
    x4 = x2 + x3
    x5 = adapter.out_norm(x4)
    gate = torch.sigmoid(adapter.output_gate_logit.detach().cpu().float())
    x6 = x5 * gate
    slot_mask = valid[:, None, :, None]
    x7 = x6 * slot_mask
    x8 = x7.view(batch, time_steps * slots, dim)
    return {
        "slot_bias": slot_bias,
        "time_bias": time_bias,
        "x0_object_latent_tokens": x0,
        "x1_plus_slot_time": x1,
        "x2_norm": x2,
        "x3_mlp": x3,
        "x4_residual": x4,
        "x5_out_norm": x5,
        "x6_gate_applied": x6,
        "x7_mask_applied": x7,
        "x8_flattened": x8,
        "gate_scalar": gate.reshape(1),
    }


def _clone_adapter(adapter: ObjectConditionAdapter) -> ObjectConditionAdapter:
    return copy.deepcopy(adapter).float().eval()


def _swap_component_from_b(
    adapter_a: ObjectConditionAdapter,
    adapter_b: ObjectConditionAdapter,
    component: str,
) -> ObjectConditionAdapter:
    mixed = _clone_adapter(adapter_a)
    if component == "slot_embed":
        mixed.slot_embed.load_state_dict(adapter_b.slot_embed.state_dict())
    elif component == "time_embed":
        mixed.time_embed.load_state_dict(adapter_b.time_embed.state_dict())
    elif component == "slot_time_embed":
        mixed.slot_embed.load_state_dict(adapter_b.slot_embed.state_dict())
        mixed.time_embed.load_state_dict(adapter_b.time_embed.state_dict())
    elif component == "norm":
        mixed.norm.load_state_dict(adapter_b.norm.state_dict())
    elif component == "mlp":
        mixed.mlp.load_state_dict(adapter_b.mlp.state_dict())
    elif component == "out_norm":
        mixed.out_norm.load_state_dict(adapter_b.out_norm.state_dict())
    elif component == "gate":
        mixed.output_gate_logit.data.copy_(adapter_b.output_gate_logit.detach())
    else:
        raise ValueError(f"Unsupported component={component}")
    return mixed


def main() -> None:
    args = parse_args()
    output_dir = (args.output_root / args.case_name).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = core._load_input_json(args.input_json)
    source_video = Path(str(payload["source_video"])).expanduser().resolve()
    prompt = core._ensure_str_field(payload, "input_caption", args.input_json)
    context_video_single, frame_indices = _load_context_video(
        source_video=source_video,
        sampling_mode=str(args.sampling_mode),
        context_frames=int(args.context_frames),
        height=int(args.height),
        width=int(args.width),
    )

    object_latent_tokens, object_valid_mask, template_info = _extract_latents_and_template(
        args,
        checkpoint_dir=args.checkpoint_a.expanduser().resolve(),
        output_dir=output_dir / str(args.label_a),
        context_video_single=context_video_single,
        prompt=prompt,
        source_video=source_video,
    )

    adapter_a = _load_object_adapter_from_checkpoint(
        args.checkpoint_a.expanduser().resolve(),
        dim=int(template_info["dim"]),
        num_slots=int(template_info["num_slots"]),
        max_time_steps=int(template_info["max_time_steps"]),
    )
    adapter_b = _load_object_adapter_from_checkpoint(
        args.checkpoint_b.expanduser().resolve(),
        dim=int(template_info["dim"]),
        num_slots=int(template_info["num_slots"]),
        max_time_steps=int(template_info["max_time_steps"]),
    )

    stages_a = _adapter_forward_breakdown(
        adapter_a,
        object_latent_tokens=object_latent_tokens,
        object_valid_mask=object_valid_mask,
    )
    stages_b = _adapter_forward_breakdown(
        adapter_b,
        object_latent_tokens=object_latent_tokens,
        object_valid_mask=object_valid_mask,
    )

    stage_names = [
        "slot_bias",
        "time_bias",
        "x0_object_latent_tokens",
        "x1_plus_slot_time",
        "x2_norm",
        "x3_mlp",
        "x4_residual",
        "x5_out_norm",
        "x6_gate_applied",
        "x7_mask_applied",
        "x8_flattened",
        "gate_scalar",
    ]
    stage_diffs = []
    for stage_name in stage_names:
        stage_diffs.append(
            {
                "stage": stage_name,
                "a_stats": _tensor_stats(stages_a[stage_name]),
                "b_stats": _tensor_stats(stages_b[stage_name]),
                "diff": _diff_stats(stages_a[stage_name], stages_b[stage_name]),
            }
        )

    output_a = stages_a["x8_flattened"]
    output_b = stages_b["x8_flattened"]
    full_ab_mean_abs = float((output_a - output_b).abs().mean().item())

    components = [
        "slot_embed",
        "time_embed",
        "slot_time_embed",
        "norm",
        "mlp",
        "out_norm",
        "gate",
    ]
    component_swaps = []
    for component in components:
        mixed_adapter = _swap_component_from_b(adapter_a, adapter_b, component)
        mixed_output = _adapter_forward_breakdown(
            mixed_adapter,
            object_latent_tokens=object_latent_tokens,
            object_valid_mask=object_valid_mask,
        )["x8_flattened"]
        diff_from_a = _diff_stats(output_a, mixed_output)
        diff_from_b = _diff_stats(output_b, mixed_output)
        moved_fraction = (
            float(diff_from_a["mean_abs_diff"]) / full_ab_mean_abs if full_ab_mean_abs > 0.0 else 0.0
        )
        closeness_to_b = 1.0 - (
            float(diff_from_b["mean_abs_diff"]) / full_ab_mean_abs if full_ab_mean_abs > 0.0 else 0.0
        )
        component_swaps.append(
            {
                "component": component,
                "diff_vs_a_output": diff_from_a,
                "diff_vs_b_output": diff_from_b,
                "mean_abs_move_fraction_of_full_ab": moved_fraction,
                "mean_abs_closeness_to_b_ratio": closeness_to_b,
            }
        )

    param_groups = {
        "slot_embed": ["object_adapter.slot_embed.weight"],
        "time_embed": ["object_adapter.time_embed.weight"],
        "bbox_embed": ["object_adapter.bbox_embed.weight"],
        "norm": ["object_adapter.norm.weight", "object_adapter.norm.bias"],
        "mlp": [
            "object_adapter.mlp.0.weight",
            "object_adapter.mlp.0.bias",
            "object_adapter.mlp.2.weight",
            "object_adapter.mlp.2.bias",
        ],
        "out_norm": ["object_adapter.out_norm.weight", "object_adapter.out_norm.bias"],
        "gate": ["object_adapter.output_gate_logit"],
    }
    state_a = load_file(str((args.checkpoint_a.expanduser().resolve() / "checkpoint.safetensors")), device="cpu")
    state_b = load_file(str((args.checkpoint_b.expanduser().resolve() / "checkpoint.safetensors")), device="cpu")
    parameter_group_diffs = []
    for group_name, keys in param_groups.items():
        total_sum = 0.0
        total_numel = 0
        max_abs = 0.0
        for key in keys:
            diff = (state_a[key].float() - state_b[key].float()).abs()
            total_sum += float(diff.sum().item())
            total_numel += int(diff.numel())
            if diff.numel():
                max_abs = max(max_abs, float(diff.max().item()))
        parameter_group_diffs.append(
            {
                "group": group_name,
                "keys": keys,
                "mean_abs_diff": total_sum / max(total_numel, 1),
                "max_abs_diff": max_abs,
                "numel": total_numel,
            }
        )

    report = {
        "input_json": str(args.input_json),
        "source_video": str(source_video),
        "prompt": prompt,
        "frame_indices": frame_indices.tolist(),
        "checkpoint_a": {
            "label": str(args.label_a),
            "path": str(args.checkpoint_a.expanduser().resolve()),
        },
        "checkpoint_b": {
            "label": str(args.label_b),
            "path": str(args.checkpoint_b.expanduser().resolve()),
        },
        "template_info": template_info,
        "stage_diffs": stage_diffs,
        "component_swaps": component_swaps,
        "parameter_group_diffs": parameter_group_diffs,
    }
    report_path = output_dir / "object_adapter_probe_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "output_dir": str(output_dir),
        "report": str(report_path),
        "final_output_diff": _diff_stats(output_a, output_b),
        "largest_stage_by_mean_abs_diff": max(
            stage_diffs,
            key=lambda item: float(item["diff"]["mean_abs_diff"]),
        )["stage"],
        "largest_component_swap_by_move_fraction": max(
            component_swaps,
            key=lambda item: float(item["mean_abs_move_fraction_of_full_ab"]),
        )["component"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
