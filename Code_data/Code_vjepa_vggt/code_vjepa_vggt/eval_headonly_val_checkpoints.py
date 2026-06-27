from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset
from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackAdapter
from code_vjepa_vggt.infer_v_newtrain_context_video_wan import (
    _load_v_newtrain_state_into_model,
    _resolve_checkpoint_file,
)
from code_vjepa_vggt.inspect_train_aux_losses_v_newtrain_compare import (
    _build_report,
    _checkpoint_label,
    _run_case_for_checkpoint,
)
from code_vjepa_vggt.train_v_newtrain import (
    WanTrainingModule,
    build_wan22_ti2v5b_model_paths,
    find_tokenizer_path,
)


DEFAULT_DATASET_ROOT = (
    "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/"
    "industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500"
)
DEFAULT_WAN_ROOT = "/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"
DEFAULT_JEPA_CKPT = "/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth"
DEFAULT_COTRACKER_CKPT = "/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth"
DEFAULT_VGGT_ROOT = "/data/gaoya/ckpt/facebook-VGGT-1B"
DEFAULT_VGGT_CACHE_ROOT = "/data/gaoya/AAA_test_video/0623/train/train0624/vggt_cache"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wan-root", default=DEFAULT_WAN_ROOT)
    parser.add_argument("--overlay-indices", type=int, nargs="+", default=None)
    parser.add_argument("--cases-per-family", type=int, default=0)
    parser.add_argument("--families", nargs="+", default=["F1", "F2", "F3", "F4", "F5"])
    parser.add_argument("--native-only-report", action="store_true")
    parser.add_argument("--latent-only-report", action="store_true")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-context-frames", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--object-num-queries", type=int, default=8)
    parser.add_argument("--aux-max-objects", type=int, default=4)
    parser.add_argument("--jepa-ckpt-path", default=DEFAULT_JEPA_CKPT)
    parser.add_argument("--jepa-input-size", type=int, default=384)
    parser.add_argument("--jepa-patch-size", type=int, default=16)
    parser.add_argument("--jepa-tubelet-size", type=int, default=2)
    parser.add_argument("--cotracker-checkpoint", default=DEFAULT_COTRACKER_CKPT)
    parser.add_argument("--cotracker-input-h", type=int, default=384)
    parser.add_argument("--cotracker-input-w", type=int, default=512)
    parser.add_argument("--cotracker-window-len", type=int, default=60)
    parser.add_argument("--vggt-model-path", default=DEFAULT_VGGT_ROOT)
    parser.add_argument("--vggt-input-h", type=int, default=420)
    parser.add_argument("--vggt-input-w", type=int, default=728)
    parser.add_argument("--vggt-cache-root", default=DEFAULT_VGGT_CACHE_ROOT)
    parser.add_argument("--object-pooler-latent-dim", type=int, default=16)
    parser.add_argument("--cond-proj-dim", type=int, default=4096)
    parser.add_argument("--jepa-window-radius", type=int, default=1)
    parser.add_argument("--latent-window-radius", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=["full", "val-only", "overlay-only"],
        default="full",
    )
    parser.add_argument("--val-metrics-json", default=None)
    return parser.parse_args()


def _discover_checkpoints(run_dirs: list[str], explicit_checkpoints: list[str] | None) -> list[Path]:
    if explicit_checkpoints:
        return [Path(path).expanduser().resolve() for path in explicit_checkpoints]
    checkpoints: list[Path] = []
    for raw_run_dir in run_dirs:
        run_dir = Path(raw_run_dir).expanduser().resolve()
        checkpoint_root = run_dir / "checkpoints" if (run_dir / "checkpoints").is_dir() else run_dir
        step_dirs = sorted(path for path in checkpoint_root.glob("step-*") if path.is_dir())
        if not step_dirs:
            raise FileNotFoundError(f"no step-* directories found under {checkpoint_root}")
        checkpoints.extend(step_dirs)
    return checkpoints


def _build_model(args: argparse.Namespace) -> WanTrainingModule:
    model_paths = build_wan22_ti2v5b_model_paths(args.wan_root)
    tokenizer_path = find_tokenizer_path(args.wan_root)
    model = WanTrainingModule(
        model_paths=model_paths,
        tokenizer_path=tokenizer_path,
        trainable_models=None,
        lora_base_model="dit",
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=int(args.lora_rank),
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs="input_image",
        fp8_models=None,
        offload_models=None,
        device=args.device,
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        context_sampling_profile="legacy_prefix",
        min_context_frames=1,
        max_context_ratio=0.5,
        context_reference_frames=49,
        context_reference_prefixes="1,4,8,12,16",
        prefix_context_ratio=0.55,
        first_frame_context_ratio=0.20,
        sparse_context_ratio=0.15,
        random_context_ratio=0.05,
        no_context_ratio=0.05,
        fixed_num_context_frames=int(args.num_context_frames),
        enable_object_branch=True,
        object_num_queries=int(args.object_num_queries),
        aux_max_objects=int(args.aux_max_objects),
        jepa_ckpt_path=args.jepa_ckpt_path,
        jepa_input_size=int(args.jepa_input_size),
        jepa_patch_size=int(args.jepa_patch_size),
        jepa_tubelet_size=int(args.jepa_tubelet_size),
        cotracker_checkpoint=args.cotracker_checkpoint,
        cotracker_input_h=int(args.cotracker_input_h),
        cotracker_input_w=int(args.cotracker_input_w),
        cotracker_window_len=int(args.cotracker_window_len),
        vggt_model_path=args.vggt_model_path,
        vggt_input_h=int(args.vggt_input_h),
        vggt_input_w=int(args.vggt_input_w),
        vggt_cache_root=args.vggt_cache_root,
        object_aux_devices=None,
        train_vggt=False,
        object_pooler_latent_dim=int(args.object_pooler_latent_dim),
        cond_proj_dim=int(args.cond_proj_dim),
        jepa_window_radius=int(args.jepa_window_radius),
        latent_window_radius=int(args.latent_window_radius),
        object_track_delta_scale=0.25,
        object_track_gate_init=0.05,
        object_box_delta_scale=0.25,
        object_box_wh_log_scale=2.25,
        object_box_wh_max_scale=2.0,
        object_min_box_px=16.0,
        object_gate_init=0.1,
        lambda_main=0.0,
        lambda_track_aux=0.1,
        lambda_box_aux=0.1,
        lambda_depth_aux=0.1,
        lambda_track_box_aux=0.0,
        lambda_track_iou_aux=0.0,
        lambda_track_anchor_reg=0.0,
        lambda_box_anchor_reg=0.0,
        lambda_object_context_reg=0.0,
        depth_target_state_index=2,
        train_object_pooler=True,
        train_object_aux_heads=True,
        train_object_adapter=False,
        train_object_dit_branch=False,
        freeze_non_object_trainables=False,
    )
    if getattr(model, "vggt_adapter", None) is None:
        model.vggt_adapter = VGGTTrackAdapter(
            model_path=args.vggt_model_path,
            num_queries=int(args.object_num_queries) * int(args.aux_max_objects),
            device=args.device,
            input_hw=(int(args.vggt_input_h), int(args.vggt_input_w)),
            trainable=False,
        )
    return model


def _iter_dataset_indices(dataset_len: int, max_val_samples: int | None) -> list[int]:
    if max_val_samples is None:
        return list(range(dataset_len))
    return list(range(min(dataset_len, int(max_val_samples))))


def _extract_family(sample: dict[str, Any]) -> str | None:
    metadata = sample.get("metadata", {}) or {}
    sample_dir = str(metadata.get("sample_dir", ""))
    match = re.search(r"/(F[1-5])_", sample_dir)
    if match:
        return match.group(1)
    prompt = str(sample.get("caption", ""))
    match = re.search(r"\b(f[1-5])\b", prompt, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _select_overlay_indices(
    dataset: PhysStateEpisodeDataset,
    *,
    explicit_indices: list[int] | None,
    families: list[str],
    cases_per_family: int,
) -> list[int]:
    if explicit_indices:
        return [index for index in explicit_indices if 0 <= int(index) < len(dataset)]
    if int(cases_per_family) <= 0:
        return []
    requested = {str(f).upper() for f in families}
    selected: list[int] = []
    counts = {family: 0 for family in requested}
    seen_sample_ids: dict[str, set[str]] = {family: set() for family in requested}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        family = _extract_family(sample)
        if family is None or family.upper() not in requested:
            continue
        family = family.upper()
        if counts[family] >= int(cases_per_family):
            continue
        sample_id = str(sample.get("metadata", {}).get("sample_id", f"idx-{idx}"))
        if sample_id in seen_sample_ids[family]:
            continue
        selected.append(int(idx))
        counts[family] += 1
        seen_sample_ids[family].add(sample_id)
        if all(counts[f] >= int(cases_per_family) for f in requested):
            break
    missing = [family for family, count in counts.items() if count < int(cases_per_family)]
    if missing:
        raise RuntimeError(
            "failed to collect enough samples for families: "
            + ", ".join(f"{family} ({counts[family]}/{int(cases_per_family)})" for family in missing)
        )
    return selected


def _aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float, np.floating)):
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / max(counts.get(key, 0), 1) for key in sums}


def _evaluate_checkpoint_on_val(
    *,
    model: WanTrainingModule,
    dataset: PhysStateEpisodeDataset,
    checkpoint_path: Path,
    sample_indices: list[int],
    seed: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    load_info = _load_v_newtrain_state_into_model(model, checkpoint_path)
    metric_rows: list[dict[str, float]] = []
    for dataset_index in tqdm(sample_indices, desc=f"val {checkpoint_path.parent.name}", leave=False):
        sample = dataset[int(dataset_index)]
        local_seed = int(seed) + int(dataset_index)
        random.seed(local_seed)
        np.random.seed(local_seed % (2**32 - 1))
        torch.manual_seed(local_seed)
        with torch.no_grad():
            loss = model(sample)
        metrics = dict(getattr(model, "last_train_metrics", {}))
        metrics["train/loss_total"] = float(loss.detach().item())
        metric_rows.append(metrics)
    return _aggregate_metric_rows(metric_rows), load_info


def _build_summary_record(
    *,
    checkpoint_path: Path,
    mean_metrics: dict[str, float],
    num_val_samples: int,
) -> dict[str, Any]:
    return {
        "checkpoint_label": _checkpoint_label(checkpoint_path),
        "checkpoint": str(_resolve_checkpoint_file(checkpoint_path)),
        "num_val_samples": int(num_val_samples),
        "mean_track_aux": float(mean_metrics.get("train/loss_track_aux", 0.0)),
        "mean_box_aux": float(mean_metrics.get("train/loss_box_aux", 0.0)),
        "mean_depth_aux": float(mean_metrics.get("train/loss_depth_aux", 0.0)),
        "mean_track_box": float(mean_metrics.get("train/track_box_loss", 0.0)),
        "mean_track_iou": float(mean_metrics.get("train/track_iou_loss", 0.0)),
        "mean_loss_total": float(mean_metrics.get("train/loss_total", 0.0)),
        "mean_object_context_abs_max": float(mean_metrics.get("train/object_context_abs_max", 0.0)),
        "mean_object_latent_tokens_abs_max": float(mean_metrics.get("train/object_latent_tokens_abs_max", 0.0)),
        "mean_grad_norm": float(mean_metrics.get("train/grad_norm", 0.0)),
        "all_mean_metrics": mean_metrics,
    }


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = _discover_checkpoints(args.run_dirs, args.checkpoints)
    dataset = PhysStateEpisodeDataset(
        root=args.dataset_root,
        split=args.split,
        resolution=(int(args.height), int(args.width)),
        num_context_frames=int(args.num_context_frames),
        context_fraction=0.5,
        random_context_frames=False,
        seed=int(args.seed),
    )
    val_sample_indices = _iter_dataset_indices(len(dataset), args.max_val_samples)
    overlay_indices = _select_overlay_indices(
        dataset,
        explicit_indices=None if args.overlay_indices is None else list(args.overlay_indices),
        families=[str(f) for f in args.families],
        cases_per_family=int(args.cases_per_family),
    )
    overlay_samples = {int(index): dataset[int(index)] for index in overlay_indices} if overlay_indices else {}

    model = _build_model(args)
    model.to(torch.device(args.device))
    model.eval()

    per_case_records = {}
    if args.mode in ("full", "overlay-only"):
        if not overlay_indices:
            raise ValueError("overlay indices are empty after clipping to dataset range")
        per_case_records = {
            int(sample_index): {
                "sample_index": int(sample_index),
                "caption": overlay_samples[int(sample_index)]["caption"],
                "video_path": overlay_samples[int(sample_index)]["video_path"],
                "context_frame_indices": overlay_samples[int(sample_index)]["context_frame_indices"].tolist(),
                "sample_id": str(overlay_samples[int(sample_index)].get("metadata", {}).get("sample_id", "")),
                "case_group": str(_extract_family(overlay_samples[int(sample_index)]) or "unknown"),
                "checkpoints": [],
            }
            for sample_index in overlay_indices
        }
    summary_by_checkpoint: list[dict[str, Any]] = []

    if args.mode == "overlay-only":
        if not args.val_metrics_json:
            raise ValueError("--val-metrics-json is required in overlay-only mode")
        payload = json.loads(Path(args.val_metrics_json).read_text(encoding="utf-8"))
        summary_by_checkpoint = list(payload.get("checkpoints", []))

    for checkpoint_path in checkpoints:
        mean_metrics = None
        load_info = None
        checkpoint_record = None
        if args.mode in ("full", "val-only"):
            mean_metrics, load_info = _evaluate_checkpoint_on_val(
                model=model,
                dataset=dataset,
                checkpoint_path=checkpoint_path,
                sample_indices=val_sample_indices,
                seed=int(args.seed),
            )
            checkpoint_record = _build_summary_record(
                checkpoint_path=checkpoint_path,
                mean_metrics=mean_metrics,
                num_val_samples=len(val_sample_indices),
            )
            checkpoint_record["load_info"] = load_info
            summary_by_checkpoint.append(checkpoint_record)
        else:
            load_info = _load_v_newtrain_state_into_model(model, checkpoint_path)
            for item in summary_by_checkpoint:
                if str(_resolve_checkpoint_file(checkpoint_path)) == str(item.get("checkpoint")):
                    checkpoint_record = item
                    mean_metrics = dict(item.get("all_mean_metrics", {}))
                    break

        if args.mode in ("full", "overlay-only"):
            checkpoint_label = _checkpoint_label(checkpoint_path)
            for sample_index in overlay_indices:
                item = _run_case_for_checkpoint(
                    model=model,
                    sample=overlay_samples[int(sample_index)],
                    checkpoint_path=checkpoint_path,
                    checkpoint_label=checkpoint_label,
                    sample_index=int(sample_index),
                    output_dir=output_dir,
                    fps=int(args.fps),
                    export_aux_visuals=not bool(args.native_only_report),
                    export_native_visuals=not bool(args.latent_only_report),
                )
                item["load_info"] = load_info
                item["val_mean_metrics"] = mean_metrics
                per_case_records[int(sample_index)]["checkpoints"].append(item)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    html_path = None
    results_by_case: list[dict[str, Any]] = []
    if args.mode in ("full", "overlay-only"):
        results_by_case = [per_case_records[int(sample_index)] for sample_index in overlay_indices]
        html_path = _build_report(
            results_by_case=results_by_case,
            summary_by_checkpoint=summary_by_checkpoint,
            output_dir=output_dir,
            native_only=bool(args.native_only_report),
            latent_only=bool(args.latent_only_report),
        )

    payload = {
        "split": args.split,
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "num_val_samples": len(val_sample_indices),
        "overlay_indices": overlay_indices,
        "checkpoints": summary_by_checkpoint,
        "html_report": None if html_path is None else str(html_path),
    }
    metrics_json_path = output_dir / "full_val_metrics.json"
    metrics_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"full val metrics: {metrics_json_path}")
    if html_path is not None:
        print(f"overlay report: {html_path}")


if __name__ == "__main__":
    main()
