'''
# 评估pdi指标
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py \
  --metric pdi \
  --result-root /data/gaoya/AAA_test_video/0623/test/v2v

# 评估wmreward指标
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py \
  --metric wmreward \
  --result-root /data/gaoya/AAA_test_video/0623/test/v2v

# 评估单视角近似 Physics-IQ 指标
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py \
  --metric physics_iq \
  --result-root /data/gaoya/AAA_test_video/0623/test/v2v

  
# 一键启动所有指标的评估
CUDA_VISIBLE_DEVICES=0 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.sh /data/gaoya/AAA_test_video/0623/test/v2v


# 统计并可视化指标报告
/home/gaoya/miniconda3/envs/wan-cu128/bin/python /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/render_v2v_metric_report.py --result-root /data/gaoya/AAA_test_video/0623/test/v2v
pyport /data/gaoya/AAA_test_video/0623/test/report/v2v 8991


# 把test_5.txt中的json路径对应的所有方法输出视频复制到output-root中

/home/gaoya/miniconda3/envs/wan-cu128/bin/python /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/export_v2v_case_videos.py \
    --txt-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
    --output-root /data/gaoya/agent-data/outputs/v2v_case_export_test5




'''
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable
import shutil

ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
TRY0526_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
for path in [ROOT, TRY0526_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from physv_eval.paths import FLUX_PYTHON


DEFAULT_RESULT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/v2v")
CONTEXT_PREFIX_CACHE_ROOT = Path("/data/gaoya/agent-data/cache/AAAinfer_bench_context_prefix")


def remap_metric_path(path: Path) -> Path:
    """Resolve a path recorded on a remote mount without changing provenance.

    Result metadata must retain the path used by the remote inference host.
    Metric workers may opt into a narrowly scoped mount translation through
    PHYSV_BENCH_PATH_REMAP_FROM/TO. With the variables unset this is a no-op.
    """
    source = os.environ.get("PHYSV_BENCH_PATH_REMAP_FROM", "").strip()
    target = os.environ.get("PHYSV_BENCH_PATH_REMAP_TO", "").strip()
    if not source or not target:
        return path
    source = source.rstrip("/")
    raw = str(path)
    if raw != source and not raw.startswith(source + "/"):
        return path
    mapped = target.rstrip("/") + raw[len(source):]
    return Path(mapped).expanduser().resolve()


@dataclass
class CaseRecord:
    result_json_path: Path
    result_payload: dict[str, Any]
    input_json_path: Path
    gt_video_path: Path
    candidate_video_path: Path


MetricFunc = Callable[[CaseRecord], dict[str, Any] | None]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    field: str
    builder: Callable[[argparse.Namespace], MetricFunc]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-evaluate one metric over all result jsons under result-root, "
            "loading one metric model per process and backfilling immediately."
        )
    )
    metric_choices = [
        "pdi",
        "wmreward",
        "proxy",
        "videophy2",
        "phyground",
        "cosmos_reason1",
        "physics_iq",
        "physics_iq_with_context",
        "physics_iq_without_context",
        "physics_iq_verified_proxy",
        "pmf_with_context",
        "pmf_without_context",
        "vbench_subject_consistency",
        "vbench_background_consistency",
        "vbench_temporal_flickering",
        "vbench_motion_smoothness",
        "vbench_dynamic_degree",
        "vbench_aesthetic_quality",
        "vbench_imaging_quality",
    ]
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--input-json-allowlist",
        type=Path,
        default=None,
        help="Optional txt file of input_json paths to evaluate; all other result JSONs are skipped.",
    )
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--metric", required=True, choices=metric_choices)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--videophy2-task",
        default="generated_only_sa_pc_joint",
        choices=["sa", "pc", "rule", "generated_only_sa_pc_joint"],
    )
    parser.add_argument("--videophy2-caption", default=None)
    parser.add_argument("--phyground-general-only", action="store_true")
    parser.add_argument("--pdi-caption", default="ball")
    parser.add_argument("--wmreward-reset-interval", type=int, default=16)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--flux-python", type=Path, default=FLUX_PYTHON, help=argparse.SUPPRESS)
    parser.add_argument("--cosmos-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--flux-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--physics-iq-output-root",
        type=Path,
        default=Path("/tmp/gaoya/physics_iq_single_case/AAAinfer_bench"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--physics-iq-verified-output-root",
        type=Path,
        default=Path("/tmp/gaoya/physics_iq_verified_proxy/AAAinfer_bench"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--physics-iq-verified-dataset-root",
        type=Path,
        default=Path("/data/gaoya/dataset/Anates-Labs-Research-Physics-IQ-Verified"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--physics-iq-official-repo",
        type=Path,
        default=Path(
            "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/"
            "code_phys_papers_compare/google-deepmind-physics-iq-benchmark"
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pmf-output-root",
        type=Path,
        default=Path("/tmp/gaoya/physinone_pmf_single_case/AAAinfer_bench"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--pmf-device", default="cpu", help=argparse.SUPPRESS)
    parser.add_argument(
        "--vbench-output-root",
        type=Path,
        default=Path("/data/gaoya/agent-data/outputs/vbench_single_case/AAAinfer_bench"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--vbench-device", default="cuda", help=argparse.SUPPRESS)
    parser.add_argument(
        "--vbench-load-ckpt-from-local",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--vbench-read-frame", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--vbench-imaging-quality-preprocessing-mode",
        default="longer",
        choices=["shorter", "longer", "shorter_centercrop", "None"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--physics-iq-threshold-value", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--physics-iq-downsample-factor", type=int, default=4, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


@contextmanager
def locked_result_json(path: Path):
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def round_floats(value: Any, ndigits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {key: round_floats(item, ndigits=ndigits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item, ndigits=ndigits) for item in value]
    if isinstance(value, tuple):
        return [round_floats(item, ndigits=ndigits) for item in value]
    return value


def resolve_input_json_path(result_payload: dict[str, Any], result_json_path: Path) -> Path:
    input_json = result_payload.get("input_json")
    if not isinstance(input_json, str) or not input_json.strip():
        input_json = result_payload.get("case_json")
    if not isinstance(input_json, str) or not input_json.strip():
        raise ValueError(f"Missing input_json/case_json in {result_json_path}")
    candidate = remap_metric_path(Path(input_json).expanduser().resolve())
    if not candidate.is_absolute():
        raise ValueError(f"input_json must be an absolute path in {result_json_path}: {input_json}")
    if candidate.is_file():
        return candidate
    # Some remote inference jobs recorded a legacy shared input directory
    # that is not present on the metric host.  The active metric queue can
    # provide the task's canonical input directory explicitly; resolve only
    # by basename there, preserving the original result metadata untouched.
    fallback_root = os.environ.get("PHYSV_BENCH_INPUT_ROOT", "").strip()
    if fallback_root:
        fallback = Path(fallback_root).expanduser() / Path(input_json).name
        if fallback.is_file():
            return fallback.resolve()
    raise FileNotFoundError(f"Cannot resolve input_json for {result_json_path}: {input_json}")


def resolve_gt_video_path(input_json_path: Path) -> Path:
    source_payload = load_json(input_json_path)
    source_video = source_payload.get("source_video")
    if not isinstance(source_video, str) or not source_video.strip():
        raise ValueError(f"Missing source_video in source json: {input_json_path}")
    candidate = remap_metric_path(Path(source_video).expanduser().resolve())
    if not candidate.is_absolute():
        raise ValueError(f"source_video must be an absolute path in {input_json_path}: {source_video}")
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Cannot resolve source_video from {input_json_path}: {source_video}")


def resolve_context_video_path(input_json_path: Path) -> Path | None:
    source_payload = load_json(input_json_path)
    for key in ("input_video", "context_video"):
        candidate_value = source_payload.get(key)
        if not isinstance(candidate_value, str) or not candidate_value.strip():
            continue
        candidate = remap_metric_path(Path(candidate_value).expanduser().resolve())
        if candidate.is_file():
            return candidate
    return None


def parse_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def resolve_context_frames_override(record: CaseRecord) -> int | None:
    for key in ("effective_context_frames", "context_frames"):
        resolved = parse_nonnegative_int(record.result_payload.get(key))
        if resolved is not None:
            return resolved
    inference = record.result_payload.get("inference")
    if isinstance(inference, dict):
        for key in ("effective_context_frames", "requested_context_frames"):
            resolved = parse_nonnegative_int(inference.get(key))
            if resolved is not None:
                return resolved
    return None


def write_bgr_frames_to_video(path: Path, frames: list[Any], fps: float) -> None:
    import av
    import cv2
    import numpy as np

    if not frames:
        raise ValueError("Cannot write an empty context video")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.stem}.tmp.{os.getpid()}{path.suffix}")
    height, width = frames[0].shape[:2]
    container = av.open(str(temp_path), mode="w")
    rate = Fraction(str(fps if fps > 0 else 30.0)).limit_denominator(1000)
    stream = container.add_stream("libx264", rate=rate)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    try:
        for frame in frames:
            if getattr(frame, "dtype", None) != np.uint8:
                frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)
            else:
                frame_u8 = frame
            rgb_frame = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2RGB)
            video_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    os.replace(temp_path, path)


def materialize_source_prefix_context_video(source_video_path: Path, context_frames: int) -> Path:
    import cv2

    if context_frames <= 0:
        raise ValueError(f"context_frames must be positive, got {context_frames}")
    cache_id = hashlib.sha1(
        f"{source_video_path.resolve()}::{context_frames}".encode("utf-8")
    ).hexdigest()[:16]
    output_dir = CONTEXT_PREFIX_CACHE_ROOT / cache_id
    output_path = output_dir / f"{source_video_path.stem}_prefix_ctx{context_frames:02d}.mp4"
    if output_path.is_file():
        return output_path.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video for context prefix extraction: {source_video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
        frames: list[Any] = []
        for _ in range(context_frames):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) != context_frames:
        raise ValueError(
            f"source_video {source_video_path} has only {len(frames)} readable prefix frames, "
            f"cannot build context prefix with context_frames={context_frames}"
        )
    write_bgr_frames_to_video(output_path, frames, fps)
    return output_path.resolve()


def collect_result_jsons(result_root: Path) -> list[Path]:
    result_jsons: list[Path] = []
    for dirpath, _, filenames in os.walk(result_root, followlinks=True):
        current_dir = Path(dirpath)
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            if filename in {"summary.json", "batch_manifest.json", "eval_summary.json"}:
                continue
            result_jsons.append((current_dir / filename).resolve())
    return sorted(result_jsons)


def resolve_candidate_video_path(result_json_path: Path, result_payload: dict[str, Any]) -> Path:
    candidate_video_path = result_json_path.with_suffix(".mp4")
    if candidate_video_path.is_file():
        return candidate_video_path.resolve()
    generated_video_path = result_json_path.parent / "generated.mp4"
    if generated_video_path.is_file():
        return generated_video_path.resolve()
    candidate_video = result_payload.get("output_video")
    if isinstance(candidate_video, str) and candidate_video.strip():
        path = Path(candidate_video).expanduser().resolve()
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing candidate video for {result_json_path}")


def derive_method_name(result_payload: dict[str, Any], fallback_video_path: Path | None = None) -> str | None:
    def normalize_ckpt_method_name(name: str) -> str:
        normalized = re.sub(r"^[A-Za-z]+\d+_", "", name, count=1)
        return normalized or name

    def derive_method_name_from_ckpt_path(ckpt_path: Path) -> str | None:
        candidate_path = ckpt_path.expanduser()
        if candidate_path.is_file() or candidate_path.suffix:
            step_dir = candidate_path.parent
            if not step_dir.name.startswith("step-"):
                return None
            checkpoint_parent = step_dir.parent
            step_name = step_dir.name
        else:
            step_name = candidate_path.name
            checkpoint_parent = candidate_path.parent
        if not step_name:
            return None
        if checkpoint_parent.name == "checkpoints" and checkpoint_parent.parent.name:
            method_root = normalize_ckpt_method_name(checkpoint_parent.parent.name)
            return f"{method_root}_{step_name}"
        if checkpoint_parent.name:
            method_root = normalize_ckpt_method_name(checkpoint_parent.name)
            return f"{method_root}_{step_name}"
        return None

    ckpt = result_payload.get("ckpt")
    if isinstance(ckpt, str) and ckpt.strip():
        derived_from_ckpt = derive_method_name_from_ckpt_path(Path(ckpt))
        if derived_from_ckpt is not None:
            return derived_from_ckpt

    output_video = result_payload.get("output_video")
    if isinstance(output_video, str) and output_video.strip():
        output_video_path = Path(output_video).expanduser()
        if output_video_path.parent.name:
            return output_video_path.parent.name
    if fallback_video_path is not None and fallback_video_path.parent.name:
        return fallback_video_path.parent.name
    return None


def build_case_payload(record: CaseRecord) -> dict[str, Any]:
    payload = dict(record.result_payload)
    payload["video"] = str(record.candidate_video_path)
    if not isinstance(payload.get("caption"), str) or not payload["caption"].strip():
        input_caption = payload.get("input_caption")
        if isinstance(input_caption, str) and input_caption.strip():
            payload["caption"] = input_caption.strip()
        else:
            nested_input = payload.get("input")
            if isinstance(nested_input, dict):
                nested_caption = nested_input.get("caption")
                if isinstance(nested_caption, str) and nested_caption.strip():
                    payload["caption"] = nested_caption.strip()
    context_video_path = resolve_context_video_path(record.input_json_path)
    if context_video_path is not None:
        payload["context_video"] = str(context_video_path)
        payload.setdefault("input_video", str(context_video_path))
    payload["source_video"] = str(record.gt_video_path)
    return payload


def build_context_metric_case_payload(record: CaseRecord) -> tuple[dict[str, Any], int | None]:
    payload = dict(record.result_payload)
    payload["video"] = str(record.candidate_video_path)
    payload["source_video"] = str(record.gt_video_path)

    context_frames = resolve_context_frames_override(record)
    if context_frames is not None:
        context_video_path = materialize_source_prefix_context_video(record.gt_video_path, context_frames)
        payload["context_video"] = str(context_video_path)
        payload["input_video"] = str(context_video_path)
        payload["context_frames"] = int(context_frames)
        payload["effective_context_frames"] = int(context_frames)
        return payload, int(context_frames)

    fallback_context_video_path = resolve_context_video_path(record.input_json_path)
    if fallback_context_video_path is not None:
        payload["context_video"] = str(fallback_context_video_path)
        payload.setdefault("input_video", str(fallback_context_video_path))
    return payload, None


def sanitize_metric_value(metric_name: str, value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = round_floats(value)
    if metric_name == "pdi":
        for key in [
            "ra_math_pass",
            "ra_ground_rmse",
            "ra_scale_jump",
            "ra_reproj_err",
            "ra_overall_pass",
            "raw_report_path",
        ]:
            payload.pop(key, None)
        raw_report_path = value.get("raw_report_path")
        if isinstance(raw_report_path, str) and raw_report_path:
            report_path = Path(raw_report_path)
            report_dir = report_path.parent
            if report_path.exists():
                report_path.unlink(missing_ok=True)
            if report_dir.exists():
                shutil.rmtree(report_dir, ignore_errors=True)
    return payload


def cleanup_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload.pop("eval_metrics", None)
    payload.pop("gt_video", None)
    if isinstance(payload.get("pdi"), dict):
        payload["pdi"] = sanitize_metric_value("pdi", payload["pdi"])
    return payload


def metric_already_completed(payload: dict[str, Any], field: str) -> bool:
    if field not in payload:
        return False
    return payload.get(field) is not None


def apply_payload_defaults(payload: dict[str, Any], *, candidate_video_path: Path) -> dict[str, Any]:
    existing_method = payload.get("method")
    method = derive_method_name(payload, fallback_video_path=candidate_video_path)
    if method is not None:
        should_replace = not isinstance(existing_method, str) or not existing_method.strip()
        if isinstance(existing_method, str):
            stripped_method = existing_method.strip()
            if re.fullmatch(r"step-\d+", stripped_method):
                should_replace = True
        if should_replace:
            payload["method"] = method
    cleanup_result_payload(payload)
    return payload


def maybe_delegate_flux_metric(args: argparse.Namespace) -> bool:
    if args.metric not in {"phyground", "cosmos_reason1"}:
        return False
    if args.flux_worker:
        return False
    if args.metric == "cosmos_reason1" and args.cosmos_worker:
        return False

    flux_python = args.flux_python.expanduser().resolve()
    cmd = [
        str(flux_python),
        str(Path(__file__).resolve()),
        "--metric",
        args.metric,
        "--result-root",
        str(args.result_root.expanduser().resolve()),
        "--flux-worker",
    ]
    if args.metric == "cosmos_reason1":
        cmd.append("--cosmos-worker")
    if args.output_summary is not None:
        cmd.extend(["--output-summary", str(args.output_summary.expanduser().resolve())])
    if args.input_json_allowlist is not None:
        cmd.extend(["--input-json-allowlist", str(args.input_json_allowlist.expanduser().resolve())])
    if int(args.num_shards) > 1:
        cmd.extend(["--num-shards", str(int(args.num_shards)), "--shard-index", str(int(args.shard_index))])
    if args.overwrite:
        cmd.append("--overwrite")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.metric == "phyground" and args.phyground_general_only:
        cmd.append("--phyground-general-only")

    env = os.environ.copy()
    pythonpath_entries = [str(ROOT), str(TRY0526_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["PYTHONNOUSERSITE"] = "1"

    print(f"[{args.metric}:delegate] python={flux_python}")
    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))
    return True


def build_metric_spec(args: argparse.Namespace) -> MetricSpec:
    def build_method_case_dir(base_root: Path, record: CaseRecord, metric_name: str | None = None) -> Path:
        method_name = derive_method_name(record.result_payload, fallback_video_path=record.candidate_video_path)
        method_dir = method_name if method_name else record.result_json_path.stem
        path = base_root
        if metric_name is not None:
            path = path / metric_name
        return path / method_dir / record.input_json_path.stem

    def build_pdi(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.official_pdi import OfficialPDIRunner
        from physv_eval.single_case.pdi import score_case as score_pdi_case

        runner = OfficialPDIRunner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            case = build_case_payload(record)
            caption = case.get("input_caption") or case.get("caption") or args.pdi_caption
            return score_pdi_case(case, text_query=caption, runner=runner)

        return run

    def build_wmreward(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.single_case.wmreward import score_case as score_wmreward_case
        from physv_eval.wmreward_official import WMRewardRunner

        reset_interval = max(1, int(args.wmreward_reset_interval))
        runner: WMRewardRunner | None = None
        cases_since_reset = 0

        def post_case_cleanup(active_runner: WMRewardRunner | None) -> None:
            if active_runner is None:
                return
            torch_module = getattr(active_runner, "_torch", None)
            if torch_module is not None and torch_module.cuda.is_available():
                try:
                    torch_module.cuda.empty_cache()
                except Exception:
                    pass
            gc.collect()

        def cleanup_runner(active_runner: WMRewardRunner | None) -> None:
            if active_runner is None:
                return
            models = getattr(active_runner, "_models", None)
            if isinstance(models, tuple):
                for model in models[:3]:
                    if hasattr(model, "cpu"):
                        try:
                            model.cpu()
                        except Exception:
                            pass
            if hasattr(active_runner, "_models"):
                active_runner._models = None
            post_case_cleanup(active_runner)

        def run(record: CaseRecord) -> dict[str, Any] | None:
            nonlocal runner, cases_since_reset
            if runner is None or cases_since_reset >= reset_interval:
                cleanup_runner(runner)
                runner = WMRewardRunner()
                cases_since_reset = 0
            try:
                return score_wmreward_case(build_case_payload(record), runner=runner)
            finally:
                cases_since_reset += 1
                post_case_cleanup(runner)

        return run

    def build_proxy(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.proxy_runner import ProxyRunner
        from physv_eval.single_case.proxy import score_case as score_proxy_case

        runner = ProxyRunner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            case = build_case_payload(record)
            return score_proxy_case(case, context_video_path=record.gt_video_path, runner=runner)

        return run

    def build_videophy2(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.single_case.videophy2 import score_case as score_videophy2_case
        from physv_eval.videophy2_auto import VideoPhy2Runner

        runner = VideoPhy2Runner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            case = build_case_payload(record)
            caption = case.get("input_caption") or case.get("caption") or args.videophy2_caption
            rule = case.get("rule") or case.get("physical_law") or case.get("law")
            context_frames = resolve_context_frames_override(record)
            return score_videophy2_case(
                case,
                task=args.videophy2_task,
                caption=caption,
                rule=rule,
                context_frames=context_frames,
                runner=runner,
            )

        return run

    def build_phyground(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.phyground_official import OfficialPhyGroundRunner
        from physv_eval.single_case.phyground import score_case as score_phyground_case

        runner = OfficialPhyGroundRunner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            case = build_case_payload(record)
            caption = case.get("input_caption") or case.get("caption")
            metrics = None
            laws = [] if args.phyground_general_only else None
            return score_phyground_case(case, caption=caption, metrics=metrics, laws=laws, runner=runner)

        return run

    def build_cosmos_reason1(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.cosmos_reason1_official import OfficialCosmosReason1Runner
        from physv_eval.single_case.cosmos_reason1 import score_case as score_cosmos_reason1_case

        runner = OfficialCosmosReason1Runner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            return score_cosmos_reason1_case(build_case_payload(record), runner=runner)

        return run

    def build_physics_iq(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.single_case.physics_iq import score_case as score_physics_iq_case

        physics_iq_output_root = args.physics_iq_output_root.expanduser().resolve()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            case = build_case_payload(record)
            aligned_video_dir = build_method_case_dir(physics_iq_output_root, record)
            return score_physics_iq_case(
                case,
                source_video_path=record.gt_video_path,
                threshold_value=int(args.physics_iq_threshold_value),
                downsample_factor=int(args.physics_iq_downsample_factor),
                aligned_video_dir=aligned_video_dir,
            )

        return run

    def build_physics_iq_context_metric(context_mode: str, metric_name: str) -> Callable[[argparse.Namespace], MetricFunc]:
        def factory(_: argparse.Namespace) -> MetricFunc:
            from physv_eval.single_case.physics_iq import score_case as score_physics_iq_case

            physics_iq_output_root = args.physics_iq_output_root.expanduser().resolve()

            def run(record: CaseRecord) -> dict[str, Any] | None:
                case, context_frames_override = build_context_metric_case_payload(record)
                aligned_video_dir = build_method_case_dir(physics_iq_output_root, record, metric_name)
                return score_physics_iq_case(
                    case,
                    source_video_path=record.gt_video_path,
                    context_mode=context_mode,
                    context_frames=context_frames_override,
                    threshold_value=int(args.physics_iq_threshold_value),
                    downsample_factor=int(args.physics_iq_downsample_factor),
                    aligned_video_dir=aligned_video_dir,
                )

            return run

        return factory

    def build_physics_iq_verified_proxy(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.single_case.physics_iq_verified_proxy import (
            score_case as score_physics_iq_verified_proxy_case,
        )

        output_root = args.physics_iq_verified_output_root.expanduser().resolve()
        benchmark_root = args.physics_iq_verified_dataset_root.expanduser().resolve()
        official_repo_root = args.physics_iq_official_repo.expanduser().resolve()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            case = build_case_payload(record)
            context_frames_override = resolve_context_frames_override(record)
            context_frames = 8 if context_frames_override is None else int(context_frames_override)
            aligned_video_dir = build_method_case_dir(
                output_root,
                record,
                "physics_iq_verified_proxy",
            )
            return score_physics_iq_verified_proxy_case(
                case,
                benchmark_hint=record.input_json_path,
                context_frames=context_frames,
                benchmark_root=benchmark_root,
                official_repo_root=official_repo_root,
                threshold_value=int(args.physics_iq_threshold_value),
                aligned_video_dir=aligned_video_dir,
            )

        return run

    def build_pmf_context_metric(context_mode: str, metric_name: str) -> Callable[[argparse.Namespace], MetricFunc]:
        def factory(_: argparse.Namespace) -> MetricFunc:
            from physv_eval.single_case.pmf import score_case as score_pmf_case

            pmf_output_root = args.pmf_output_root.expanduser().resolve()

            def run(record: CaseRecord) -> dict[str, Any] | None:
                case, context_frames_override = build_context_metric_case_payload(record)
                aligned_video_dir = build_method_case_dir(pmf_output_root, record, metric_name)
                return score_pmf_case(
                    case,
                    source_video_path=record.gt_video_path,
                    context_mode=context_mode,
                    context_frames=context_frames_override,
                    device=str(args.pmf_device),
                    aligned_video_dir=aligned_video_dir,
                )

            return run

        return factory

    def build_vbench_metric(dimension: str, metric_name: str) -> Callable[[argparse.Namespace], MetricFunc]:
        def factory(_: argparse.Namespace) -> MetricFunc:
            from physv_eval.single_case.vbench import score_case as score_vbench_case
            from physv_eval.vbench_official import OfficialVBenchRunner

            runner = OfficialVBenchRunner(
                output_root=args.vbench_output_root.expanduser().resolve() / metric_name,
                device=str(args.vbench_device),
                load_ckpt_from_local=bool(args.vbench_load_ckpt_from_local),
                read_frame=bool(args.vbench_read_frame),
                imaging_quality_preprocessing_mode=str(args.vbench_imaging_quality_preprocessing_mode),
            )

            def run(record: CaseRecord) -> dict[str, Any] | None:
                case = build_case_payload(record)
                caption = case.get("input_caption") or case.get("caption")
                output_path = build_method_case_dir(args.vbench_output_root.expanduser().resolve(), record, metric_name)
                return score_vbench_case(
                    case,
                    dimension=dimension,
                    caption=caption,
                    output_path=output_path,
                    runner=runner,
                )

            return run

        return factory

    builders: dict[str, Callable[[argparse.Namespace], MetricFunc]] = {
        "pdi": build_pdi,
        "wmreward": build_wmreward,
        "proxy": build_proxy,
        "videophy2": build_videophy2,
        "phyground": build_phyground,
        "cosmos_reason1": build_cosmos_reason1,
        "physics_iq": build_physics_iq,
        "physics_iq_with_context": build_physics_iq_context_metric("with_context", "physics_iq_with_context"),
        "physics_iq_without_context": build_physics_iq_context_metric("without_context", "physics_iq_without_context"),
        "physics_iq_verified_proxy": build_physics_iq_verified_proxy,
        "pmf_with_context": build_pmf_context_metric("with_context", "pmf_with_context"),
        "pmf_without_context": build_pmf_context_metric("without_context", "pmf_without_context"),
        "vbench_subject_consistency": build_vbench_metric("subject_consistency", "vbench_subject_consistency"),
        "vbench_background_consistency": build_vbench_metric("background_consistency", "vbench_background_consistency"),
        "vbench_temporal_flickering": build_vbench_metric("temporal_flickering", "vbench_temporal_flickering"),
        "vbench_motion_smoothness": build_vbench_metric("motion_smoothness", "vbench_motion_smoothness"),
        "vbench_dynamic_degree": build_vbench_metric("dynamic_degree", "vbench_dynamic_degree"),
        "vbench_aesthetic_quality": build_vbench_metric("aesthetic_quality", "vbench_aesthetic_quality"),
        "vbench_imaging_quality": build_vbench_metric("imaging_quality", "vbench_imaging_quality"),
    }
    return MetricSpec(name=args.metric, field=args.metric, builder=builders[args.metric])


def prepare_cases(result_root: Path) -> tuple[list[CaseRecord], list[dict[str, Any]]]:
    cases: list[CaseRecord] = []
    errors: list[dict[str, Any]] = []
    for result_json_path in collect_result_jsons(result_root):
        try:
            result_payload = load_json(result_json_path)
            if not (
                isinstance(result_payload.get("input_json"), str)
                or isinstance(result_payload.get("case_json"), str)
            ):
                continue
            input_json_path = resolve_input_json_path(result_payload, result_json_path)
            gt_video_path = resolve_gt_video_path(input_json_path)
            candidate_video_path = resolve_candidate_video_path(result_json_path, result_payload)
            cases.append(
                CaseRecord(
                    result_json_path=result_json_path,
                    result_payload=result_payload,
                    input_json_path=input_json_path,
                    gt_video_path=gt_video_path,
                    candidate_video_path=candidate_video_path,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "result_json": str(result_json_path),
                    "stage": "prepare",
                    "error": str(exc),
                }
            )
    return cases, errors


def write_summary(
    summary_path: Path,
    *,
    args: argparse.Namespace,
    result_root: Path,
    metric_spec: MetricSpec,
    cases: list[CaseRecord],
    metric_status: dict[str, Any],
    errors: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    summary_payload = {
        "result_root": str(result_root),
        "num_result_jsons": len(cases),
        "metric": metric_spec.name,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "metric_status": round_floats(metric_status),
        "errors": errors,
    }
    remap_from = os.environ.get("PHYSV_BENCH_PATH_REMAP_FROM", "").strip()
    remap_to = os.environ.get("PHYSV_BENCH_PATH_REMAP_TO", "").strip()
    if remap_from and remap_to:
        summary_payload["path_remap"] = {
            "from": remap_from,
            "to": remap_to,
            "scope": "metric-worker-only",
        }
    fallback_root = os.environ.get("PHYSV_BENCH_INPUT_ROOT", "").strip()
    if fallback_root:
        summary_payload["input_json_fallback_root"] = fallback_root
    if not dry_run:
        write_json(summary_path, summary_payload)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))

def main() -> None:
    args = parse_args()
    if int(args.num_shards) <= 0:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError(
            f"--shard-index must satisfy 0 <= shard-index < num-shards, got "
            f"shard-index={args.shard_index}, num-shards={args.num_shards}"
        )
    if maybe_delegate_flux_metric(args):
        return
    result_root = args.result_root.expanduser().resolve()
    summary_path = (
        args.output_summary.expanduser().resolve()
        if args.output_summary is not None
        else result_root / f"eval_summary_{args.metric}.json"
    )
    metric_spec = build_metric_spec(args)

    cases, errors = prepare_cases(result_root)
    if args.input_json_allowlist is not None:
        allowlist_path = args.input_json_allowlist.expanduser().resolve()
        allowed_inputs = {
            Path(line.strip()).expanduser().resolve()
            for line in allowlist_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        cases = [record for record in cases if record.input_json_path in allowed_inputs]
    if int(args.num_shards) > 1:
        cases = [
            record
            for case_index, record in enumerate(cases)
            if case_index % int(args.num_shards) == int(args.shard_index)
        ]
    metric_status: dict[str, Any] = {}
    if not args.dry_run:
        write_json(summary_path, {})

    print(
        f"[metric:start] {metric_spec.name} cases={len(cases)} "
        f"shard={int(args.shard_index) + 1}/{int(args.num_shards)}"
    )
    runner = metric_spec.builder(args)
    num_success = 0
    num_failed = 0
    for index, record in enumerate(cases, start=1):
        try:
            with locked_result_json(record.result_json_path):
                current_payload = load_json(record.result_json_path)
                current_payload = apply_payload_defaults(
                    copy.deepcopy(current_payload),
                    candidate_video_path=record.candidate_video_path,
                )
                if not args.overwrite and metric_already_completed(current_payload, metric_spec.field):
                    if not args.dry_run:
                        write_json(record.result_json_path, current_payload)
                    print(f"[metric:skip] {metric_spec.name} {index}/{len(cases)} {record.result_json_path.name}")
                    num_success += 1
                    metric_status = {
                        "num_cases": len(cases),
                        "num_success": num_success,
                        "num_failed": num_failed,
                        "completed": index,
                    }
                    write_summary(
                        summary_path,
                        args=args,
                        result_root=result_root,
                        metric_spec=metric_spec,
                        cases=cases,
                        metric_status=metric_status,
                        errors=errors,
                        dry_run=args.dry_run,
                    )
                    continue

            metric_value = sanitize_metric_value(metric_spec.name, runner(record))

            with locked_result_json(record.result_json_path):
                latest_payload = load_json(record.result_json_path)
                latest_payload = apply_payload_defaults(
                    copy.deepcopy(latest_payload),
                    candidate_video_path=record.candidate_video_path,
                )
                if not args.overwrite and metric_already_completed(latest_payload, metric_spec.field):
                    if not args.dry_run:
                        write_json(record.result_json_path, latest_payload)
                    print(f"[metric:skip-race] {metric_spec.name} {index}/{len(cases)} {record.result_json_path.name}")
                    num_success += 1
                else:
                    latest_payload[metric_spec.field] = metric_value
                    latest_payload = apply_payload_defaults(
                        latest_payload,
                        candidate_video_path=record.candidate_video_path,
                    )
                    if not args.dry_run:
                        write_json(record.result_json_path, latest_payload)
                    num_success += 1
                    print(f"[metric:done] {metric_spec.name} {index}/{len(cases)} {record.result_json_path.name}")
        except Exception as exc:
            num_failed += 1
            errors.append(
                {
                    "metric": metric_spec.name,
                    "result_json": str(record.result_json_path),
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=3),
                }
            )
            print(f"[metric:error] {metric_spec.name} {index}/{len(cases)} {record.result_json_path.name}: {exc}")
        metric_status = {
            "num_cases": len(cases),
            "num_success": num_success,
            "num_failed": num_failed,
            "completed": index,
        }
        write_summary(
            summary_path,
            args=args,
            result_root=result_root,
            metric_spec=metric_spec,
            cases=cases,
            metric_status=metric_status,
            errors=errors,
            dry_run=args.dry_run,
        )
    print(f"[metric:finish] {metric_spec.name} success={num_success} failed={num_failed}")


if __name__ == "__main__":
    main()
