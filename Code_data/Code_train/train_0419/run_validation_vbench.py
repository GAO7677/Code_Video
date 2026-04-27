#!/usr/bin/env python3
"""Run context-sweep validation generation and VBench-short evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


TRAIN0419_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419")
MY_BENCH_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench")
DINO_REPO_ROOT = Path("/home/gaoya/.cache/torch/hub/facebookresearch_dinov2_main")
DINO_CHECKPOINT_PATH = Path("/home/gaoya/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth")

if str(TRAIN0419_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN0419_ROOT))
if str(MY_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(MY_BENCH_ROOT))

from benchlib.config import load_config  # noqa: E402
from benchlib.manifest import load_manifest  # noqa: E402
from benchlib.vbench_wrappers import run_vbench_short  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_resize_mode(dataset_name: str) -> str:
    if str(dataset_name).strip().lower() == "movi-d":
        return "pad"
    return "crop"


def crop_and_resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    image_height, image_width = frame.shape[:2]
    if image_height / image_width < height / width:
        cropped_width = int(image_height / height * width)
        left = (image_width - cropped_width) // 2
        frame = frame[:, left : left + cropped_width]
    else:
        cropped_height = int(image_width / width * height)
        top = (image_height - cropped_height) // 2
        frame = frame[top : top + cropped_height, :]
    return np.asarray(frame, dtype=np.uint8)


def resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    image = np.asarray(frame, dtype=np.uint8)
    pil_image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    pil_image = F.interpolate(
        pil_image,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    output = (pil_image.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0).round()
    return output.astype(np.uint8)


def pad_and_resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    image_height, image_width = frame.shape[:2]
    scale = min(width / image_width, height / image_height)
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    resized = resize_frame(frame, resized_height, resized_width)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized_height) // 2
    left = (width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def preprocess_frame_for_metrics(
    frame: np.ndarray,
    *,
    height: int,
    width: int,
    resize_mode: str,
) -> np.ndarray:
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame.astype(np.uint8)
    if resize_mode == "pad":
        return pad_and_resize_frame(frame, height=height, width=width)
    cropped = crop_and_resize_frame(frame, height=height, width=width)
    return resize_frame(cropped, height=height, width=width)


def load_video_frames(video_path: str) -> list[np.ndarray]:
    reader = imageio.get_reader(video_path)
    try:
        total = reader.count_frames()
        return [np.asarray(reader.get_data(index), dtype=np.uint8) for index in range(total)]
    finally:
        reader.close()


def metric_prefers_lower(metric_name: str) -> bool:
    lowered = metric_name.lower()
    return "lpips" in lowered or lowered.endswith("_mse")


def is_curve_metric(metric_name: str) -> bool:
    lowered = metric_name.lower()
    if lowered.endswith("pair_count"):
        return False
    return True


class ValidationMetricSuite:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex",
            normalize=True,
        ).to(self.device).eval()
        self.dino = self._load_dino_model().to(self.device).eval()
        self.dino_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.dino_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _load_dino_model(self) -> torch.nn.Module:
        if not DINO_REPO_ROOT.is_dir():
            raise FileNotFoundError(f"DINO repo cache not found: {DINO_REPO_ROOT}")
        if not DINO_CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"DINO checkpoint not found: {DINO_CHECKPOINT_PATH}")
        model = torch.hub.load(
            str(DINO_REPO_ROOT),
            "dinov2_vitb14",
            source="local",
            pretrained=False,
        )
        state_dict = torch.load(DINO_CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        return model

    def _frame_tensor(self, frame: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        return tensor.to(self.device)

    def dino_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        ta = self._frame_tensor(a)
        tb = self._frame_tensor(b)
        ta = F.interpolate(ta, size=(224, 224), mode="bilinear", align_corners=False)
        tb = F.interpolate(tb, size=(224, 224), mode="bilinear", align_corners=False)
        ta = (ta - self.dino_mean) / self.dino_std
        tb = (tb - self.dino_mean) / self.dino_std
        with torch.no_grad():
            fa = self.dino(ta)
            fb = self.dino(tb)
            score = F.cosine_similarity(fa, fb, dim=1)
        return float(score.item())

    def lpips_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        ta = self._frame_tensor(a)
        tb = self._frame_tensor(b)
        with torch.no_grad():
            score = self.lpips(ta, tb)
        return float(score.item())


def compute_future_gt_metrics(
    entries: list[dict[str, Any]],
    *,
    height: int,
    width: int,
    metric_suite: ValidationMetricSuite,
) -> dict[str, Any]:
    per_sample: list[dict[str, Any]] = []
    aggregate_buckets: dict[str, list[float]] = {}

    for entry in entries:
        if entry.get("status") not in {"generated", "skipped_existing"}:
            continue
        paths = entry.get("paths", {})
        if not isinstance(paths, dict):
            continue
        output_video_path = paths.get("output_video_path") or paths.get("output_path")
        gt_video_path = paths.get("future_gt_video_path")
        if not output_video_path or not gt_video_path:
            continue

        resize_mode = resolve_resize_mode(str(entry.get("dataset", "")))
        used_context_frames = int(
            entry.get("generation_params", {}).get(
                "used_context_frames",
                entry.get("generation_params", {}).get("context_frames", 0),
            )
        )
        generated_frames = load_video_frames(str(output_video_path))
        gt_frames = load_video_frames(str(gt_video_path))
        generated_eval = generated_frames[used_context_frames:]
        pair_count = min(len(generated_eval), len(gt_frames))
        if pair_count <= 0:
            continue

        psnr_values = []
        ssim_values = []
        lpips_values = []
        dino_values = []
        for index in range(pair_count):
            generated = preprocess_frame_for_metrics(
                generated_eval[index],
                height=height,
                width=width,
                resize_mode=resize_mode,
            )
            gt = preprocess_frame_for_metrics(
                gt_frames[index],
                height=height,
                width=width,
                resize_mode=resize_mode,
            )
            generated_float = generated.astype(np.float32) / 255.0
            gt_float = gt.astype(np.float32) / 255.0
            psnr_values.append(float(peak_signal_noise_ratio(gt_float, generated_float, data_range=1.0)))
            ssim_values.append(
                float(structural_similarity(gt_float, generated_float, channel_axis=2, data_range=1.0))
            )
            lpips_values.append(metric_suite.lpips_distance(generated, gt))
            dino_values.append(metric_suite.dino_similarity(generated, gt))

        result = {
            "sample_id": entry.get("sample_id"),
            "dataset": entry.get("dataset"),
            "future_pair_count": float(pair_count),
            "future_psnr": float(np.mean(psnr_values)),
            "future_ssim": float(np.mean(ssim_values)),
            "future_lpips": float(np.mean(lpips_values)),
            "future_dino": float(np.mean(dino_values)),
        }
        per_sample.append(result)
        for key, value in result.items():
            if key in {"sample_id", "dataset"}:
                continue
            aggregate_buckets.setdefault(key, []).append(float(value))

    aggregate = {
        key: float(np.mean(values))
        for key, values in aggregate_buckets.items()
        if values
    }
    return {
        "aggregate": aggregate,
        "per_sample": per_sample,
    }


def entry_sort_index(entry: dict[str, Any]) -> int:
    runtime = entry.get("runtime")
    if isinstance(runtime, dict) and isinstance(runtime.get("index_in_sorted_list"), int):
        return int(runtime["index_in_sorted_list"])
    if isinstance(entry.get("index_in_sorted_list"), int):
        return int(entry["index_in_sorted_list"])
    return 10**18


def load_entries_for_compare(
    model_name: str,
    generated_dir: Path,
    runtime_root: Path | None,
) -> list[dict[str, Any]]:
    if runtime_root is not None:
        jsonl_path = runtime_root / "metadata" / model_name / f"{model_name}_per_case.jsonl"
        if jsonl_path.is_file():
            return load_jsonl(jsonl_path)

    entries: list[dict[str, Any]] = []
    for sidecar_path in sorted(generated_dir.glob("*.json")):
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            entries.append(payload)
    if not entries:
        raise FileNotFoundError(
            f"no entries found for model={model_name} under runtime_root={runtime_root} or generated_dir={generated_dir}"
        )
    entries.sort(key=entry_sort_index)
    return entries


def build_dataset_breakdown(
    entries: list[dict[str, Any]],
    *,
    height: int,
    width: int,
    metric_suite: ValidationMetricSuite,
) -> dict[str, Any]:
    datasets = sorted({str(entry.get("dataset", "unknown")) for entry in entries})
    payload: dict[str, Any] = {}
    for dataset_name in datasets:
        subset = [entry for entry in entries if str(entry.get("dataset", "unknown")) == dataset_name]
        metrics = compute_future_gt_metrics(
            subset,
            height=height,
            width=width,
            metric_suite=metric_suite,
        )
        payload[dataset_name] = {
            "num_entries": len(subset),
            "num_success": sum(1 for entry in subset if entry.get("status") in {"generated", "skipped_existing"}),
            "aggregate": metrics.get("aggregate", {}),
        }
    return payload


def build_run_payload(
    *,
    model_name: str,
    generated_dir: Path,
    runtime_root: Path | None,
    height: int,
    width: int,
    metric_suite: ValidationMetricSuite,
) -> dict[str, Any]:
    entries = load_entries_for_compare(model_name, generated_dir, runtime_root)
    overall_metrics = compute_future_gt_metrics(
        entries,
        height=height,
        width=width,
        metric_suite=metric_suite,
    )
    return {
        "model_name": model_name,
        "generated_dir": str(generated_dir),
        "runtime_root": str(runtime_root) if runtime_root is not None else None,
        "num_entries": len(entries),
        "num_success": sum(1 for entry in entries if entry.get("status") in {"generated", "skipped_existing"}),
        "aggregate": overall_metrics.get("aggregate", {}),
        "per_dataset": build_dataset_breakdown(
            entries,
            height=height,
            width=width,
            metric_suite=metric_suite,
        ),
    }


def metric_delta(ft_value: float | None, base_value: float | None) -> float | None:
    if ft_value is None or base_value is None:
        return None
    return float(ft_value - base_value)


def build_comparison_rows(base_payload: dict[str, Any], ft_payload: dict[str, Any]) -> list[dict[str, Any]]:
    metric_names = ["future_psnr", "future_ssim", "future_lpips", "future_dino"]
    row_names = ["overall"] + sorted(
        set(base_payload.get("per_dataset", {}).keys()) | set(ft_payload.get("per_dataset", {}).keys())
    )
    rows: list[dict[str, Any]] = []
    for row_name in row_names:
        if row_name == "overall":
            base_block = {
                "num_entries": base_payload.get("num_entries"),
                "num_success": base_payload.get("num_success"),
                "aggregate": base_payload.get("aggregate", {}),
            }
            ft_block = {
                "num_entries": ft_payload.get("num_entries"),
                "num_success": ft_payload.get("num_success"),
                "aggregate": ft_payload.get("aggregate", {}),
            }
        else:
            base_block = base_payload.get("per_dataset", {}).get(row_name, {})
            ft_block = ft_payload.get("per_dataset", {}).get(row_name, {})

        row: dict[str, Any] = {
            "scope": row_name,
            "base_num_entries": base_block.get("num_entries"),
            "base_num_success": base_block.get("num_success"),
            "ft_num_entries": ft_block.get("num_entries"),
            "ft_num_success": ft_block.get("num_success"),
        }
        for metric_name in metric_names:
            base_value = base_block.get("aggregate", {}).get(metric_name)
            ft_value = ft_block.get("aggregate", {}).get(metric_name)
            row[f"base_{metric_name}"] = base_value
            row[f"ft_{metric_name}"] = ft_value
            row[f"delta_{metric_name}"] = metric_delta(ft_value, base_value)
        rows.append(row)
    return rows


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_compare_mode(args: argparse.Namespace) -> None:
    if not args.compare_base_name or not args.compare_base_generated_dir or not args.compare_ft_name or not args.compare_ft_generated_dir:
        raise ValueError("Compare mode requires base/finetuned names and generated directories.")
    if args.compare_output_root is None:
        raise ValueError("Compare mode requires --compare_output_root.")

    args.compare_output_root.mkdir(parents=True, exist_ok=True)
    metric_suite = ValidationMetricSuite()
    base_payload = build_run_payload(
        model_name=args.compare_base_name,
        generated_dir=args.compare_base_generated_dir,
        runtime_root=args.compare_base_runtime_root,
        height=args.height,
        width=args.width,
        metric_suite=metric_suite,
    )
    ft_payload = build_run_payload(
        model_name=args.compare_ft_name,
        generated_dir=args.compare_ft_generated_dir,
        runtime_root=args.compare_ft_runtime_root,
        height=args.height,
        width=args.width,
        metric_suite=metric_suite,
    )
    rows = build_comparison_rows(base_payload, ft_payload)
    comparison = {
        "base": base_payload,
        "finetuned": ft_payload,
        "rows": rows,
    }
    write_json(args.compare_output_root / "base_metrics.json", base_payload)
    write_json(args.compare_output_root / "finetuned_metrics.json", ft_payload)
    write_json(args.compare_output_root / "comparison_summary.json", comparison)
    write_rows_csv(args.compare_output_root / "comparison_metrics.csv", rows)
    print(args.compare_output_root / "comparison_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-context validation generation followed by VBench-short."
    )
    parser.add_argument("--wan_root", type=Path, default=None)
    parser.add_argument("--lora_path", type=Path, default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--meta_list_path", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--runtime_root", type=Path, default=None)
    parser.add_argument("--batch_eval_script_path", type=Path, default=None)
    parser.add_argument("--vbench_config_path", type=Path, default=None)

    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num_frames", type=int, default=161)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--context_frames_list", default="0,1,2,4,6,8")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compare_base_name", default=None)
    parser.add_argument("--compare_base_generated_dir", type=Path, default=None)
    parser.add_argument("--compare_base_runtime_root", type=Path, default=None)
    parser.add_argument("--compare_ft_name", default=None)
    parser.add_argument("--compare_ft_generated_dir", type=Path, default=None)
    parser.add_argument("--compare_ft_runtime_root", type=Path, default=None)
    parser.add_argument("--compare_output_root", type=Path, default=None)
    return parser.parse_args()


def parse_context_frames_list(raw_value: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError("context_frames_list must contain at least one integer.")
    if any(value < 0 for value in values):
        raise ValueError(f"context_frames_list must be non-negative, got {values}.")
    return values


def run_generation_for_context(args: argparse.Namespace, context_frames: int) -> tuple[Path, Path]:
    context_tag = f"ctx{context_frames:02d}"
    output_root = args.output_root / context_tag
    runtime_root = args.runtime_root / context_tag
    summary_path = runtime_root / "summary.json"

    command = [
        sys.executable,
        str(args.batch_eval_script_path),
        "--wan_root",
        str(args.wan_root),
        "--output_root",
        str(output_root),
        "--runtime_root",
        str(runtime_root),
        "--lora_path",
        str(args.lora_path),
        "--meta_list_path",
        str(args.meta_list_path),
        "--model_name",
        f"{args.model_name}_{context_tag}",
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--fps",
        str(args.fps),
        "--num_frames",
        str(args.num_frames),
        "--context_frames",
        str(context_frames),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--cfg_scale",
        str(args.cfg_scale),
        "--seed",
        str(args.seed),
        "--overwrite",
    ]
    if args.negative_prompt:
        command.extend(["--negative_prompt", args.negative_prompt])
    if args.multi_gpu:
        command.append("--multi_gpu")
    if not args.overwrite:
        command.remove("--overwrite")

    subprocess.run(
        command,
        check=True,
        cwd=str(TRAIN0419_ROOT),
        env=os.environ.copy(),
    )
    if not summary_path.is_file():
        raise FileNotFoundError(f"Generation summary not found: {summary_path}")
    return output_root, runtime_root


def parse_vbench_eval(path: Path) -> dict[str, float]:
    payload = load_json(path)
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, list) and value:
            metrics[key] = float(value[0])
        elif isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def build_vbench_manifest(runtime_root: Path, model_name: str, manifest_path: Path) -> Path:
    jsonl_path = runtime_root / "metadata" / model_name / f"{model_name}_per_case.jsonl"
    entries = load_jsonl(jsonl_path)
    samples = []
    for entry in entries:
        if entry.get("status") not in {"generated", "skipped_existing"}:
            continue
        paths = entry.get("paths", {})
        output_path = paths.get("output_video_path") or paths.get("output_path")
        if not output_path:
            continue
        samples.append(
            {
                "sample_id": entry["sample_id"],
                "prompt": entry["caption"],
                "video_path": output_path,
            }
        )
    if not samples:
        raise ValueError(f"No successful generation samples found under {jsonl_path}.")
    write_json(manifest_path, samples)
    return manifest_path


def load_generation_entries(runtime_root: Path, model_name: str) -> list[dict[str, Any]]:
    merged_jsonl = runtime_root / "metadata" / model_name / f"{model_name}_per_case.jsonl"
    return load_jsonl(merged_jsonl)


def summarize_context_results(results_by_context: dict[int, dict[str, Any]]) -> dict[str, Any]:
    metrics_by_name: dict[str, list[tuple[int, float]]] = {}
    for context_frames, payload in sorted(results_by_context.items()):
        for metric_name, metric_value in payload["curve_metrics"].items():
            if not is_curve_metric(metric_name):
                continue
            metrics_by_name.setdefault(metric_name, []).append((context_frames, metric_value))

    best_context_by_metric = {}
    for metric_name, points in metrics_by_name.items():
        if metric_prefers_lower(metric_name):
            best_context, best_value = min(points, key=lambda item: item[1])
        else:
            best_context, best_value = max(points, key=lambda item: item[1])
        best_context_by_metric[metric_name] = {
            "context_frames": int(best_context),
            "value": float(best_value),
        }

    vbench_aggregate_curve = []
    for context_frames, payload in sorted(results_by_context.items()):
        values = list(payload["vbench_short_metrics"].values())
        vbench_aggregate_curve.append(
            {
                "context_frames": int(context_frames),
                "vbench_aggregate_mean": round(sum(values) / len(values), 6) if values else 0.0,
            }
        )

    best_aggregate = max(vbench_aggregate_curve, key=lambda item: item["vbench_aggregate_mean"])
    return {
        "num_context_settings": len(results_by_context),
        "best_aggregate_context_frames": int(best_aggregate["context_frames"]),
        "best_aggregate_mean": float(best_aggregate["vbench_aggregate_mean"]),
        "best_context_by_metric": best_context_by_metric,
        "vbench_aggregate_curve": vbench_aggregate_curve,
    }


def write_curve_csv(path: Path, results_by_context: dict[int, dict[str, Any]]) -> None:
    metric_names = sorted(
        {
            metric_name
            for payload in results_by_context.values()
            for metric_name in payload["curve_metrics"].keys()
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["context_frames", *metric_names],
        )
        writer.writeheader()
        for context_frames, payload in sorted(results_by_context.items()):
            metrics = payload["curve_metrics"]
            row = {"context_frames": context_frames}
            row.update(metrics)
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.compare_base_name is not None:
        run_compare_mode(args)
        return

    required_args = {
        "--wan_root": args.wan_root,
        "--lora_path": args.lora_path,
        "--model_name": args.model_name,
        "--meta_list_path": args.meta_list_path,
        "--output_root": args.output_root,
        "--runtime_root": args.runtime_root,
        "--batch_eval_script_path": args.batch_eval_script_path,
        "--vbench_config_path": args.vbench_config_path,
    }
    missing = [name for name, value in required_args.items() if value is None]
    if missing:
        raise ValueError(f"Standard validation mode is missing required arguments: {', '.join(missing)}")
    context_frames_list = parse_context_frames_list(args.context_frames_list)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.runtime_root.mkdir(parents=True, exist_ok=True)

    bench_config = load_config(str(args.vbench_config_path))
    metric_suite = ValidationMetricSuite()
    results_by_context: dict[int, dict[str, Any]] = {}

    for context_frames in context_frames_list:
        context_tag = f"ctx{context_frames:02d}"
        context_model_name = f"{args.model_name}_{context_tag}"
        generation_output_root, generation_runtime_root = run_generation_for_context(
            args,
            context_frames,
        )
        generation_summary = load_json(generation_runtime_root / "summary.json")
        generation_entries = load_generation_entries(generation_runtime_root, context_model_name)
        gt_metrics_payload = compute_future_gt_metrics(
            generation_entries,
            height=args.height,
            width=args.width,
            metric_suite=metric_suite,
        )
        gt_metrics_json = args.runtime_root / "gt_metrics" / f"{context_model_name}.json"
        write_json(gt_metrics_json, gt_metrics_payload)

        manifest_path = build_vbench_manifest(
            generation_runtime_root,
            context_model_name,
            args.runtime_root / "manifests" / f"{context_model_name}.json",
        )
        samples = load_manifest(str(manifest_path))
        vbench_output_dir = args.runtime_root / "vbench_short" / context_tag
        vbench_eval_json = Path(
            run_vbench_short(
                config=bench_config,
                samples=samples,
                output_dir=str(vbench_output_dir),
                run_name=context_model_name,
            )
        )
        vbench_metrics = parse_vbench_eval(vbench_eval_json)
        curve_metrics = dict(gt_metrics_payload.get("aggregate", {}))
        curve_metrics.update({f"vbench_{key}": value for key, value in vbench_metrics.items()})
        vbench_values = list(vbench_metrics.values())
        curve_metrics["vbench_aggregate_mean"] = (
            round(sum(vbench_values) / len(vbench_values), 6) if vbench_values else 0.0
        )
        curve_metrics["aggregate_mean"] = curve_metrics["vbench_aggregate_mean"]
        results_by_context[context_frames] = {
            "context_frames": context_frames,
            "generation_output_root": str(generation_output_root),
            "generation_runtime_root": str(generation_runtime_root),
            "generation_summary": generation_summary.get("summary", {}),
            "future_gt_metrics_json": str(gt_metrics_json),
            "future_gt_metrics": gt_metrics_payload.get("aggregate", {}),
            "vbench_short_eval_json": str(vbench_eval_json),
            "vbench_short_metrics": vbench_metrics,
            "curve_metrics": curve_metrics,
            "manifest_path": str(manifest_path),
        }

    summary = summarize_context_results(results_by_context)
    payload = {
        "model_name": args.model_name,
        "lora_path": str(args.lora_path),
        "meta_list_path": str(args.meta_list_path),
        "contexts": {str(key): value for key, value in sorted(results_by_context.items())},
        "summary": summary,
    }
    write_json(args.runtime_root / "summary.json", payload)
    write_curve_csv(args.runtime_root / "context_curve.csv", results_by_context)
    print(args.runtime_root / "summary.json")


if __name__ == "__main__":
    main()
