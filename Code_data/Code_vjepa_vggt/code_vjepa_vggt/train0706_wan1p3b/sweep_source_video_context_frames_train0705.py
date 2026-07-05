from __future__ import annotations

# Run command example:
# PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
# CUDA_VISIBLE_DEVICES=7 \
# /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
# /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/sweep_source_video_context_frames_train0705.py \
#   --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0706_wan21_13b/run_gpu0235_20260703/checkpoints/step-007000 \
#   --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
#   --model-name train_stage1b_diffsynth_native0706_wan21_13b_sourcectx_sweep \
#   --context-frames-list 4 16 24 \
#   --num-inference-steps 40

import argparse
import av
import cv2
import json
import numpy as np
import os
import subprocess
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
TRAIN0706_DIR = THIS_FILE.parent
PACKAGE_DIR = TRAIN0706_DIR.parent
REPO_ROOT = PACKAGE_DIR.parent
DEFAULT_BATCH_SCRIPT = TRAIN0706_DIR / "wan_stage1b_context_only_no_gt_box_wan21_13b_v2v.py"
DEFAULT_DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/v2v_1p3b")
DEFAULT_ORIGINAL_JSON_FIELD = "train0706_saved_input_context_videos"
DEFAULT_SAVED_INPUT_VIDEO_CONTAINER = "mp4"
DEFAULT_SAVED_INPUT_VIDEO_CODEC = "libx264"
DEFAULT_SAVED_INPUT_VIDEO_BACKEND = "pyav"


def _read_list_file(list_path: Path, *, deduplicate: bool) -> list[Path]:
    items: list[Path] = []
    seen: set[Path] = set()
    with list_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            item = Path(line).expanduser().resolve()
            if deduplicate and item in seen:
                continue
            seen.add(item)
            items.append(item)
    return items


def _filter_input_json_paths(
    input_json_paths: list[Path],
    *,
    case_stems: list[str] | None,
) -> list[Path]:
    if not case_stems:
        return input_json_paths
    allowed = {str(item).strip() for item in case_stems if str(item).strip()}
    return [path for path in input_json_paths if path.stem in allowed]


def _load_input_json(json_path: Path) -> dict[str, object]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"input json must be an object: {json_path}")
    return payload


def _resolve_source_video(
    payload: dict[str, object],
    *,
    json_path: Path,
    source_video_field: str,
    fallback_to_input_video: bool,
) -> str:
    value = payload.get(source_video_field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if fallback_to_input_video:
        value = payload.get("input_video")
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(
        f"missing usable source video in {json_path}; checked {source_video_field!r}"
    )


def _build_patched_jsons(
    *,
    input_json_paths: list[Path],
    context_frames: int,
    scratch_root: Path,
    source_video_field: str,
    fallback_to_input_video: bool,
) -> tuple[list[Path], list[dict[str, object]]]:
    ctx_dir = scratch_root / f"context_frames_{int(context_frames):02d}"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    patched_jsons: list[Path] = []
    manifest_items: list[dict[str, object]] = []
    for input_json_path in input_json_paths:
        payload = _load_input_json(input_json_path)
        source_video = _resolve_source_video(
            payload,
            json_path=input_json_path,
            source_video_field=source_video_field,
            fallback_to_input_video=fallback_to_input_video,
        )
        original_input_video = payload.get("input_video")
        payload["input_video"] = str(source_video)
        payload["sweep_source_video_field"] = str(source_video_field)
        payload["sweep_context_frames"] = int(context_frames)
        payload["original_input_video"] = original_input_video
        patched_json_path = ctx_dir / input_json_path.name
        patched_json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        patched_jsons.append(patched_json_path)
        manifest_items.append(
            {
                "input_json": str(input_json_path),
                "patched_json": str(patched_json_path),
                "source_video": str(source_video),
                "original_input_video": original_input_video,
            }
        )
    return patched_jsons, manifest_items


def _select_frame_indices(total_frames: int, target_context_frames: int, sampling_mode: str) -> np.ndarray:
    if total_frames < target_context_frames:
        raise RuntimeError(
            f"video only has {total_frames} frames, smaller than required {target_context_frames}"
        )
    if sampling_mode == "uniform":
        return np.linspace(0, total_frames - 1, num=target_context_frames, dtype=np.int32)
    return np.arange(target_context_frames, dtype=np.int32)


def _write_selected_frames_as_video(
    *,
    source_video_path: Path,
    output_video_path: Path,
    frame_indices: np.ndarray,
    fallback_fps: int,
    video_writer_fourcc: str,
    video_backend: str,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(source_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open source video: {source_video_path}")
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = float(fallback_fps)
        if width <= 0 or height <= 0:
            raise RuntimeError(f"invalid source video spatial size: {source_video_path}")
        frames_bgr: list[np.ndarray] = []
        for frame_index in frame_indices.tolist():
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"failed to read frame {int(frame_index)} from {source_video_path}"
                )
            frames_bgr.append(frame)
    finally:
        capture.release()

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    if output_video_path.exists():
        output_video_path.unlink()

    if video_backend == "pyav":
        container = av.open(str(output_video_path), mode="w")
        try:
            stream = container.add_stream(str(video_writer_fourcc), rate=max(1, int(round(fps))))
            stream.width = int(width)
            stream.height = int(height)
            stream.pix_fmt = "yuv420p"
            if str(video_writer_fourcc) in {"libx264", "h264"}:
                stream.options = {"crf": "18", "preset": "medium"}
            for frame_bgr in frames_bgr:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
    else:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*video_writer_fourcc),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to open writer for: {output_video_path}")
        try:
            for frame_bgr in frames_bgr:
                writer.write(frame_bgr)
        finally:
            writer.release()

    return {
        "path": str(output_video_path.resolve()),
        "frame_indices": [int(index) for index in frame_indices.tolist()],
        "fps": fps,
        "source_video": str(source_video_path.resolve()),
        "num_frames": int(frame_indices.shape[0]),
        "codec": str(video_writer_fourcc),
        "container": str(output_video_path.suffix.lstrip(".")),
        "backend": str(video_backend),
    }


def _build_saved_input_video_path(
    *,
    context_output_root: Path,
    step_name: str,
    case_stem: str,
    context_frames: int,
    container: str,
    placement_mode: str,
    source_video_path: Path,
) -> Path:
    if placement_mode == "sample_dir":
        sample_dir = context_output_root / step_name / case_stem
        file_name = f"{case_stem}_ctx{int(context_frames):02d}f.{container}"
        return sample_dir / file_name
    if placement_mode == "source_video_dir":
        sample_dir = source_video_path.parent
        file_name = f"{case_stem}_ctx{int(context_frames):02d}f.{container}"
        return sample_dir / file_name
    return context_output_root / "input_context_videos" / f"{case_stem}.{container}"


def _load_json_object(json_path: Path) -> dict[str, object]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"json must be an object: {json_path}")
    return payload


def _write_json_object(json_path: Path, payload: dict[str, object]) -> None:
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sync_result_json_input_context_video(
    *,
    result_json_path: Path,
    saved_video_info: dict[str, object],
) -> None:
    if not result_json_path.exists():
        return
    payload = _load_json_object(result_json_path)
    payload["saved_input_context_video"] = str(saved_video_info["path"])
    payload["saved_input_context_video_frame_indices"] = list(saved_video_info["frame_indices"])
    payload["saved_input_context_video_num_frames"] = int(saved_video_info["num_frames"])
    payload["saved_input_context_video_fps"] = float(saved_video_info["fps"])
    payload["saved_input_context_video_codec"] = str(saved_video_info["codec"])
    payload["saved_input_context_video_container"] = str(saved_video_info["container"])
    payload["saved_input_context_video_backend"] = str(saved_video_info["backend"])
    original_input_video = payload.get("input_video")
    if not payload.get("source_video") and isinstance(original_input_video, str):
        payload["source_video"] = original_input_video
    payload["input_video"] = str(saved_video_info["path"])
    _write_json_object(result_json_path, payload)


def _sync_original_json_input_context_video(
    *,
    original_json_path: Path,
    context_frames: int,
    saved_video_info: dict[str, object],
) -> None:
    payload = _load_json_object(original_json_path)
    payload[f"input_video_{int(context_frames)}f"] = str(saved_video_info["path"])
    payload.pop(DEFAULT_ORIGINAL_JSON_FIELD, None)
    _write_json_object(original_json_path, payload)


def _prepare_saved_input_context_videos(
    *,
    args: argparse.Namespace,
    context_frames: int,
    context_output_root: Path,
    manifest_items: list[dict[str, object]],
) -> list[dict[str, object]]:
    saved_items: list[dict[str, object]] = []
    for item in manifest_items:
        original_json_path = Path(str(item["input_json"])).expanduser().resolve()
        source_video_path = Path(str(item["source_video"])).expanduser().resolve()
        case_stem = original_json_path.stem
        saved_video_path = _build_saved_input_video_path(
            context_output_root=context_output_root,
            step_name=args.weights_root.name,
            case_stem=case_stem,
            context_frames=int(context_frames),
            container=str(args.saved_input_video_container),
            placement_mode=str(args.saved_input_video_placement),
            source_video_path=source_video_path,
        )

        capture = cv2.VideoCapture(str(source_video_path))
        if not capture.isOpened():
            raise RuntimeError(f"failed to open source video: {source_video_path}")
        try:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        frame_indices = _select_frame_indices(
            total_frames=total_frames,
            target_context_frames=int(context_frames),
            sampling_mode=str(args.sampling_mode),
        )
        saved_video_info = _write_selected_frames_as_video(
            source_video_path=source_video_path,
            output_video_path=saved_video_path,
            frame_indices=frame_indices,
            fallback_fps=int(args.fps),
            video_writer_fourcc=str(args.saved_input_video_codec),
            video_backend=str(args.saved_input_video_backend),
        )

        patched_json_path = Path(str(item["patched_json"])).expanduser().resolve()
        patched_payload = _load_json_object(patched_json_path)
        patched_payload["saved_input_context_video"] = str(saved_video_info["path"])
        patched_payload["saved_input_context_video_frame_indices"] = list(saved_video_info["frame_indices"])
        patched_payload["saved_input_context_video_num_frames"] = int(saved_video_info["num_frames"])
        patched_payload["saved_input_context_video_fps"] = float(saved_video_info["fps"])
        patched_payload["saved_input_context_video_codec"] = str(saved_video_info["codec"])
        patched_payload["saved_input_context_video_container"] = str(saved_video_info["container"])
        patched_payload["saved_input_context_video_backend"] = str(saved_video_info["backend"])
        _write_json_object(patched_json_path, patched_payload)

        _sync_original_json_input_context_video(
            original_json_path=original_json_path,
            context_frames=int(context_frames),
            saved_video_info=saved_video_info,
        )
        saved_items.append(
            {
                "input_json": str(original_json_path),
                "patched_json": str(patched_json_path),
                "saved_input_context_video": str(saved_video_info["path"]),
                "source_video": str(saved_video_info["source_video"]),
                "frame_indices": list(saved_video_info["frame_indices"]),
                "num_frames": int(saved_video_info["num_frames"]),
                "fps": float(saved_video_info["fps"]),
                "codec": str(saved_video_info["codec"]),
                "container": str(saved_video_info["container"]),
                "backend": str(saved_video_info["backend"]),
                "result_json": str(context_output_root / args.weights_root.name / f"{case_stem}.json"),
            }
        )
    return saved_items


def _sync_saved_input_context_videos_to_result_jsons(saved_items: list[dict[str, object]]) -> None:
    for item in saved_items:
        result_json_path = Path(str(item["result_json"])).expanduser().resolve()
        if not result_json_path.exists():
            continue
        saved_video_info = {
            "path": str(item["saved_input_context_video"]),
            "source_video": str(item.get("source_video", "")),
            "frame_indices": list(item["frame_indices"]),
            "num_frames": int(item["num_frames"]),
            "fps": float(item.get("fps", 0.0)),
            "codec": str(item.get("codec", "")),
            "container": str(item.get("container", "")),
            "backend": str(item.get("backend", "")),
        }
        _sync_result_json_input_context_video(
            result_json_path=result_json_path,
            saved_video_info=saved_video_info,
        )


def _sync_step_result_summary(
    *,
    step_result_json_path: Path,
    saved_items: list[dict[str, object]],
) -> None:
    if not step_result_json_path.exists():
        return
    payload = _load_json_object(step_result_json_path)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return
    saved_by_input_json = {
        str(item["patched_json"]): item
        for item in saved_items
    }
    changed = False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        input_json = str(entry.get("input_json", ""))
        saved_item = saved_by_input_json.get(input_json)
        if saved_item is None:
            continue
        original_input_video = entry.get("input_video")
        if not entry.get("source_video") and isinstance(original_input_video, str):
            entry["source_video"] = original_input_video
        entry["input_video"] = str(saved_item["saved_input_context_video"])
        entry["saved_input_context_video"] = str(saved_item["saved_input_context_video"])
        entry["saved_input_context_video_frame_indices"] = list(saved_item["frame_indices"])
        entry["saved_input_context_video_num_frames"] = int(saved_item["num_frames"])
        entry["saved_input_context_video_fps"] = float(saved_item["fps"])
        entry["saved_input_context_video_codec"] = str(saved_item["codec"])
        entry["saved_input_context_video_container"] = str(saved_item["container"])
        entry["saved_input_context_video_backend"] = str(saved_item["backend"])
        changed = True
    if changed:
        _write_json_object(step_result_json_path, payload)


def _write_list_file(list_path: Path, items: list[Path]) -> None:
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with list_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(str(item))
            handle.write("\n")


def _build_command(
    *,
    args: argparse.Namespace,
    context_frames: int,
    patched_list_path: Path,
    context_output_root: Path,
) -> list[str]:
    cmd = [
        str(args.python_executable),
        str(args.batch_script),
        "--weights-root",
        str(args.weights_root),
        "--input-json-list-path",
        str(patched_list_path),
        "--model-name",
        str(args.model_name),
        "--output-root",
        str(context_output_root),
        "--device",
        str(args.device),
        "--wan-root",
        str(args.wan_root),
        "--diffsynth-root",
        str(args.diffsynth_root),
        "--lora-checkpoint",
        str(args.lora_checkpoint),
        "--stage1a-init-from",
        str(args.stage1a_init_from),
        "--height",
        str(int(args.height)),
        "--width",
        str(int(args.width)),
        "--num-frames",
        str(int(args.num_frames)),
        "--context-frames",
        str(int(context_frames)),
        "--fps",
        str(int(args.fps)),
        "--sampling-mode",
        str(args.sampling_mode),
        "--num-inference-steps",
        str(int(args.num_inference_steps)),
        "--cfg-scale",
        str(float(args.cfg_scale)),
        "--seed",
        str(int(args.seed)),
        "--quality",
        str(int(args.quality)),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(int(args.limit))])
    if args.force:
        cmd.append("--force")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.initialize_model_on_cpu:
        cmd.append("--initialize-model-on-cpu")
    return cmd


def _build_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath_items = [
        str(REPO_ROOT),
        str(args.diffsynth_root),
    ]
    existing = env.get("PYTHONPATH", "").strip()
    if existing:
        pythonpath_items.append(existing)
    env["PYTHONPATH"] = ":".join(item for item in pythonpath_items if item)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep train0706 inference over multiple context lengths by replacing "
            "input_video in each json with source_video and reusing the existing "
            "batch inference script."
        )
    )
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--input-json-list-path", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--scratch-root", type=Path, default=None)
    parser.add_argument("--context-frames-list", type=int, nargs="+", required=True)
    parser.add_argument("--case-stems", type=str, nargs="+", default=None)
    parser.add_argument("--source-video-field", type=str, default="source_video")
    parser.add_argument("--fallback-to-input-video", action="store_true")
    parser.add_argument("--no-deduplicate", action="store_true")
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--batch-script", type=Path, default=DEFAULT_BATCH_SCRIPT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wan-root", type=Path, default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B"))
    parser.add_argument("--diffsynth-root", type=Path, default=DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument(
        "--lora-checkpoint",
        type=Path,
        default=Path(
            "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/"
            "raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors"
        ),
    )
    parser.add_argument(
        "--stage1a-init-from",
        type=Path,
        default=Path(
            "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
            "pybullet0629_teacher_student/stage1a_full_token/step_0003000.pt"
        ),
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--initialize-model-on-cpu", action="store_true")
    parser.add_argument(
        "--saved-input-video-container",
        type=str,
        default=DEFAULT_SAVED_INPUT_VIDEO_CONTAINER,
        help="Container extension for exported input context videos.",
    )
    parser.add_argument(
        "--saved-input-video-codec",
        type=str,
        default=DEFAULT_SAVED_INPUT_VIDEO_CODEC,
        help="Codec name used for exported input context videos.",
    )
    parser.add_argument(
        "--saved-input-video-backend",
        choices=["pyav", "opencv"],
        default=DEFAULT_SAVED_INPUT_VIDEO_BACKEND,
        help="Video writer backend used for exported input context videos.",
    )
    parser.add_argument(
        "--saved-input-video-placement",
        choices=["shared_root", "sample_dir", "source_video_dir"],
        default="shared_root",
        help="Where to place exported input context videos.",
    )
    parser.add_argument(
        "--original-json-field-name",
        type=str,
        default=DEFAULT_ORIGINAL_JSON_FIELD,
        help="Deprecated no-op; original json now uses flat input_video_<frames>f fields.",
    )
    parser.add_argument(
        "--sync-input-context-videos-only",
        action="store_true",
        help="Only export input context clips and sync json metadata; skip inference subprocess.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.weights_root = args.weights_root.expanduser().resolve()
    args.input_json_list_path = args.input_json_list_path.expanduser().resolve()
    args.python_executable = args.python_executable.expanduser().resolve()
    args.batch_script = args.batch_script.expanduser().resolve()
    args.wan_root = args.wan_root.expanduser().resolve()
    args.diffsynth_root = args.diffsynth_root.expanduser().resolve()
    args.lora_checkpoint = args.lora_checkpoint.expanduser().resolve()
    args.stage1a_init_from = args.stage1a_init_from.expanduser().resolve()

    if args.output_root is None:
        args.output_root = DEFAULT_OUTPUT_ROOT / str(args.model_name).strip()
    else:
        args.output_root = args.output_root.expanduser().resolve()
    if args.scratch_root is None:
        args.scratch_root = args.output_root / "_source_video_patched_jsons"
    else:
        args.scratch_root = args.scratch_root.expanduser().resolve()

    input_json_paths = _read_list_file(
        args.input_json_list_path,
        deduplicate=not bool(args.no_deduplicate),
    )
    input_json_paths = _filter_input_json_paths(
        input_json_paths,
        case_stems=args.case_stems,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)

    summary_entries: list[dict[str, object]] = []
    for context_frames in args.context_frames_list:
        patched_jsons, manifest_items = _build_patched_jsons(
            input_json_paths=input_json_paths,
            context_frames=int(context_frames),
            scratch_root=args.scratch_root,
            source_video_field=str(args.source_video_field),
            fallback_to_input_video=bool(args.fallback_to_input_video),
        )
        patched_list_path = args.scratch_root / f"context_frames_{int(context_frames):02d}.txt"
        _write_list_file(patched_list_path, patched_jsons)

        context_output_root = args.output_root / f"context_frames_{int(context_frames):02d}"
        context_output_root.mkdir(parents=True, exist_ok=True)

        manifest = {
            "weights_root": str(args.weights_root),
            "input_json_list_path": str(args.input_json_list_path),
            "deduplicated": not bool(args.no_deduplicate),
            "context_frames": int(context_frames),
            "num_items": len(patched_jsons),
            "sampling_mode": str(args.sampling_mode),
            "patched_items": manifest_items,
        }
        (context_output_root / "source_video_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        saved_items = _prepare_saved_input_context_videos(
            args=args,
            context_frames=int(context_frames),
            context_output_root=context_output_root,
            manifest_items=manifest_items,
        )
        manifest["saved_input_context_videos"] = saved_items
        (context_output_root / "source_video_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if not args.sync_input_context_videos_only:
            cmd = _build_command(
                args=args,
                context_frames=int(context_frames),
                patched_list_path=patched_list_path,
                context_output_root=context_output_root,
            )
            print(f"[run] context_frames={int(context_frames)}")
            print(" ".join(cmd))
            subprocess.run(
                cmd,
                check=True,
                cwd=str(REPO_ROOT),
                env=_build_env(args),
            )
        _sync_saved_input_context_videos_to_result_jsons(saved_items)
        step_output_root = context_output_root / args.weights_root.name
        _sync_step_result_summary(
            step_result_json_path=step_output_root / "result.json",
            saved_items=saved_items,
        )
        summary_entries.append(
            {
                "context_frames": int(context_frames),
                "output_root": str(context_output_root),
                "patched_list_path": str(patched_list_path),
                "num_items": len(patched_jsons),
                "saved_input_context_video_placement": str(args.saved_input_video_placement),
                "saved_input_context_video_root": (
                    str(context_output_root / "input_context_videos")
                    if str(args.saved_input_video_placement) == "shared_root"
                    else (
                        str(context_output_root / args.weights_root.name)
                        if str(args.saved_input_video_placement) == "sample_dir"
                        else "source_video_parent_dirs"
                    )
                ),
                "step_output_root": str(step_output_root),
                "inference_executed_in_this_run": not bool(args.sync_input_context_videos_only),
                "step_result_json_present": bool((step_output_root / "result.json").exists()),
            }
        )

    summary = {
        "weights_root": str(args.weights_root),
        "input_json_list_path": str(args.input_json_list_path),
        "model_name": str(args.model_name),
        "output_root": str(args.output_root),
        "context_frames_list": [int(item) for item in args.context_frames_list],
        "entries": summary_entries,
    }
    (args.output_root / "context_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[done] outputs under {args.output_root}")


if __name__ == "__main__":
    main()
