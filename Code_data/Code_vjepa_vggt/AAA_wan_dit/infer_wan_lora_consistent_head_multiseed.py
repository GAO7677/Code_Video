#!/usr/bin/env python3
"""Paired multi-seed Wan+LoRA baseline and consistent-category Head-zero sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from code_vjepa_vggt.train0419_reference import batch_eval_lora as core

from consistent_head_targets import (
    CATEGORIES,
    load_consistent_category_targets,
)
from dit_ablation import install_dynamic_grouped_head_ablator


CASE = "0613pybullet_sample_001460_w002"
VARIANTS = ("baseline", *CATEGORIES)
FFPROBE = Path("/data/gaoya/miniconda3/envs/vjepa2/bin/ffprobe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--classification-metadata", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--input-json-list-path", type=Path, required=True)
    parser.add_argument("--wan-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--negative-prompt", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reversibility-check-seed", type=int, default=43)
    return parser.parse_args()


def _variant_tag(variant: str) -> str:
    if variant == "baseline":
        return "baseline"
    return f"self_attn_consistent_head_zero_category_{variant.lower()}"


def _paths(root: Path, seed: int, variant: str) -> tuple[Path, Path]:
    directory = root / f"seed_{seed:04d}" / _variant_tag(variant)
    return directory / f"{CASE}.mp4", directory / f"{CASE}.json"


def _probe_video(path: Path) -> str:
    return subprocess.check_output(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        text=True,
    ).strip()


def _is_complete(
    video_path: Path,
    sidecar_path: Path,
    *,
    seed: int,
    variant: str,
    classification_sha256: str,
    expected_probe: str,
) -> bool:
    if not video_path.is_file() or not sidecar_path.is_file():
        return False
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        metadata = payload["dit_ablation"]
        if (
            payload.get("status") != "generated"
            or payload.get("seed") != seed
            or payload.get("experiment", {}).get("variant") != variant
            or _probe_video(video_path) != expected_probe
        ):
            return False
        if variant == "baseline":
            return (
                metadata.get("mode") == "baseline"
                and metadata.get("num_targets") == 0
                and metadata.get("observed_target_forward_calls") == 0
            )
        return (
            metadata.get("mode") == "self_attn_grouped_head_zero"
            and metadata.get("category") == variant
            and metadata.get("target_forward_call_count_ok") is True
            and metadata.get("target_selection", {}).get("sha256")
            == classification_sha256
        )
    except (KeyError, OSError, ValueError, subprocess.SubprocessError):
        return False


def _frame_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    capture = cv2.VideoCapture(str(path))
    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            digest.update(frame.tobytes())
            frame_count += 1
    finally:
        capture.release()
    return digest.hexdigest(), frame_count


def _write_state(
    root: Path,
    *,
    status: str,
    seeds: list[int],
    completed: list[dict[str, Any]],
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "variants": list(VARIANTS),
        "num_expected": len(seeds) * len(VARIANTS),
        "num_completed": len(completed),
        "completed": completed,
    }
    if error is not None:
        payload["error"] = error
    core.write_json(root / "sweep_state.json", payload)


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    if len(seeds) != len(args.seeds):
        raise ValueError(f"Duplicate seeds are not allowed: {args.seeds}")
    if args.reversibility_check_seed not in seeds:
        raise ValueError("--reversibility-check-seed must be one of --seeds")

    output_root = args.output_root.expanduser().resolve()
    classification_path = args.classification_metadata.expanduser().resolve()
    weights_root = args.weights_root.expanduser().resolve()
    lora_path = weights_root / "checkpoint.safetensors"
    input_list = args.input_json_list_path.expanduser().resolve()
    wan_root = args.wan_root.expanduser().resolve()
    for path, label in (
        (classification_path, "classification metadata"),
        (lora_path, "LoRA checkpoint"),
        (input_list, "input JSON list"),
        (wan_root, "Wan root"),
    ):
        core.assert_exists(path, label)

    targets, classification_source = load_consistent_category_targets(
        classification_path
    )
    rows = core.collect_cases_from_input_json_list(
        core.load_input_json_paths(input_list),
        limit=None,
    )
    if len(rows) != 1 or rows[0]["sample_id"] != CASE:
        raise RuntimeError(f"Expected only {CASE}, found {[row['sample_id'] for row in rows]}")
    row = rows[0]
    aligned_height, aligned_width = core.align_generation_size(
        args.height,
        args.width,
    )
    aligned_frames = core.align_generation_num_frames(args.num_frames)
    expected_probe = f"{aligned_width},{aligned_height},{args.num_frames}"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "wan_lora_consistent_category_head_zero_multiseed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "variants": list(VARIANTS),
        "case": CASE,
        "classification_source": classification_source,
        "weights_root": str(weights_root),
        "lora_path": str(lora_path),
        "lora_sha256": core.sha256_file(lora_path),
        "input_json_list_path": str(input_list),
        "wan_root": str(wan_root),
        "generation": {
            "height": aligned_height,
            "width": aligned_width,
            "requested_num_frames": args.num_frames,
            "aligned_num_frames": aligned_frames,
            "context_frames": args.context_frames,
            "context_resize_mode": row["context_resize_mode"],
            "conditioning_mode": "context_aware",
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "fps": args.fps,
            "quality": args.quality,
            "negative_prompt": args.negative_prompt,
        },
        "execution": {
            "pipeline_loads": 1,
            "same_pipeline_reused_for_all_variants": True,
            "seed_reset_before_every_generation": True,
            "dynamic_mask_location": (
                "self_attention_output_projection_input"
            ),
            "reversibility_check_seed": args.reversibility_check_seed,
        },
    }
    core.write_json(output_root / "run_manifest.json", manifest)

    completed: list[dict[str, Any]] = []
    for seed in seeds:
        for variant in VARIANTS:
            video_path, sidecar_path = _paths(output_root, seed, variant)
            if not args.overwrite and _is_complete(
                video_path,
                sidecar_path,
                seed=seed,
                variant=variant,
                classification_sha256=classification_source["sha256"],
                expected_probe=expected_probe,
            ):
                completed.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "video": str(video_path),
                        "reused_existing": True,
                    }
                )
    _write_state(
        output_root,
        status="loading_pipeline",
        seeds=seeds,
        completed=completed,
    )

    print("[multiseed] loading Wan+LoRA pipeline once", flush=True)
    pipe = core.build_pipeline(wan_root, args.device, lora_path)
    controller = install_dynamic_grouped_head_ablator(pipe.dit)
    method = core.build_method_name(lora_path)
    first_frame_path = None
    raw_first_frame = row.get("source_paths", {}).get("first_frame_path")
    if isinstance(raw_first_frame, str) and raw_first_frame:
        first_frame_path = Path(raw_first_frame)

    try:
        for seed in seeds:
            for variant in VARIANTS:
                video_path, sidecar_path = _paths(output_root, seed, variant)
                if not args.overwrite and _is_complete(
                    video_path,
                    sidecar_path,
                    seed=seed,
                    variant=variant,
                    classification_sha256=classification_source["sha256"],
                    expected_probe=expected_probe,
                ):
                    print(
                        f"[multiseed] reuse seed={seed} variant={variant}",
                        flush=True,
                    )
                    continue
                video_path.parent.mkdir(parents=True, exist_ok=True)
                video_path.unlink(missing_ok=True)
                sidecar_path.unlink(missing_ok=True)

                if variant == "baseline":
                    ablation = controller.set_targets(
                        category=None,
                        targets=[],
                    )
                else:
                    ablation = controller.set_targets(
                        category=variant,
                        targets=targets[variant],
                    )
                    ablation["target_selection"] = {
                        "kind": (
                            "protocol_consistent_object_query_category"
                        ),
                        **classification_source,
                    }

                print(
                    f"[multiseed] generate seed={seed} variant={variant} "
                    f"targets={ablation['num_targets']}",
                    flush=True,
                )
                started = time.monotonic()
                video, used_context_frames = core.generate_one_video(
                    pipe=pipe,
                    context_path=Path(row["context_path"]),
                    first_frame_path=first_frame_path,
                    prompt=row["caption"],
                    negative_prompt=args.negative_prompt,
                    seed=seed,
                    height=aligned_height,
                    width=aligned_width,
                    num_frames=aligned_frames,
                    fps=args.fps,
                    cfg_scale=args.cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                    context_frames=args.context_frames,
                    output_num_frames=args.num_frames,
                    context_resize_mode=row["context_resize_mode"],
                    conditioning_mode="context_aware",
                )
                core.save_video(
                    video,
                    str(video_path),
                    fps=args.fps,
                    quality=args.quality,
                )
                observed = int(controller.call_count)
                expected = (
                    int(ablation["num_targets"])
                    * args.num_inference_steps
                    * 2
                )
                ablation.update(
                    {
                        "observed_target_forward_calls": observed,
                        "expected_target_forward_calls": expected,
                        "target_forward_call_count_ok": observed == expected,
                    }
                )
                if observed != expected:
                    raise RuntimeError(
                        f"seed={seed} variant={variant}: expected "
                        f"{expected} target calls, observed {observed}"
                    )
                probe = _probe_video(video_path)
                if probe != expected_probe:
                    raise RuntimeError(
                        f"seed={seed} variant={variant}: expected video "
                        f"{expected_probe}, found {probe}"
                    )
                payload = core.build_simple_result_payload(
                    input_json_path=Path(row["meta_path"]),
                    input_video=str(row["context_path"]),
                    input_caption=str(row["caption"]),
                    output_video=video_path,
                    method=method,
                    seed=seed,
                    step=args.num_inference_steps,
                    guidance=args.cfg_scale,
                    ckpt=lora_path,
                    status="generated",
                )
                payload.update(
                    {
                        "experiment": {
                            "name": (
                                "consistent_category_head_zero_multiseed"
                            ),
                            "variant": variant,
                            "paired_seed": seed,
                            "same_pipeline_reused": True,
                            "seed_reset_before_generation": True,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        "generation_params": manifest["generation"],
                        "used_context_frames": used_context_frames,
                        "dit_ablation": ablation,
                        "negative_prompt": args.negative_prompt,
                    }
                )
                core.write_json(sidecar_path, payload)
                completed = [
                    item
                    for item in completed
                    if not (
                        item["seed"] == seed
                        and item["variant"] == variant
                    )
                ]
                completed.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "video": str(video_path),
                        "reused_existing": False,
                    }
                )
                _write_state(
                    output_root,
                    status="running",
                    seeds=seeds,
                    completed=completed,
                )
                print(
                    f"[multiseed] complete seed={seed} variant={variant} "
                    f"calls={observed}",
                    flush=True,
                )

            if seed == args.reversibility_check_seed:
                baseline_path, _ = _paths(output_root, seed, "baseline")
                control_dir = output_root / "_control"
                repeat_path = (
                    control_dir / f"seed_{seed:04d}_baseline_repeat.mp4"
                )
                controller.set_targets(category=None, targets=[])
                repeat_video, _ = core.generate_one_video(
                    pipe=pipe,
                    context_path=Path(row["context_path"]),
                    first_frame_path=first_frame_path,
                    prompt=row["caption"],
                    negative_prompt=args.negative_prompt,
                    seed=seed,
                    height=aligned_height,
                    width=aligned_width,
                    num_frames=aligned_frames,
                    fps=args.fps,
                    cfg_scale=args.cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                    context_frames=args.context_frames,
                    output_num_frames=args.num_frames,
                    context_resize_mode=row["context_resize_mode"],
                    conditioning_mode="context_aware",
                )
                control_dir.mkdir(parents=True, exist_ok=True)
                core.save_video(
                    repeat_video,
                    str(repeat_path),
                    fps=args.fps,
                    quality=args.quality,
                )
                baseline_digest, baseline_frames = _frame_digest(
                    baseline_path
                )
                repeat_digest, repeat_frames = _frame_digest(repeat_path)
                check = {
                    "seed": seed,
                    "baseline": str(baseline_path),
                    "repeat": str(repeat_path),
                    "baseline_frame_digest": baseline_digest,
                    "repeat_frame_digest": repeat_digest,
                    "baseline_frames": baseline_frames,
                    "repeat_frames": repeat_frames,
                    "decoded_frames_exact_match": (
                        baseline_digest == repeat_digest
                        and baseline_frames == repeat_frames
                    ),
                    "purpose": (
                        "verify that switching all six masks and restoring "
                        "baseline leaves no dynamic-ablation state"
                    ),
                }
                core.write_json(
                    output_root / "reversibility_check.json",
                    check,
                )
                if not check["decoded_frames_exact_match"]:
                    raise RuntimeError(
                        "Dynamic ablation reversibility check failed"
                    )
                print(
                    "[multiseed] baseline reversibility check passed",
                    flush=True,
                )
    except Exception as error:
        _write_state(
            output_root,
            status="failed",
            seeds=seeds,
            completed=completed,
            error=repr(error),
        )
        raise

    completed.sort(key=lambda item: (item["seed"], VARIANTS.index(item["variant"])))
    _write_state(
        output_root,
        status="complete",
        seeds=seeds,
        completed=completed,
    )
    print(f"[multiseed] complete outputs={len(completed)}", flush=True)


if __name__ == "__main__":
    main()
