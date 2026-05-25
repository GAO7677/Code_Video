#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = Path("/home/gaoya/Code_Video/physics-IQ-benchmark-main")
BENCHMARK_CODE = BENCHMARK_ROOT / "code"
TRAIN0419_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419")
DATASET_ROOT = Path("/data/gaoya/dataset/physics-iq-benchmark")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/physics_IQ_demo")
DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_VACE_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B")
DEFAULT_DESCRIPTIONS_CSV = BENCHMARK_ROOT / "descriptions" / "descriptions.csv"
DEFAULT_PORT = 18701

WAN_SCRIPT = BENCH_DIR / "wan22_ti2v_physics_iq_eval_multigpu.py"
BATCH_EVAL_VACE = TRAIN0419_ROOT / "batch_eval_vace.py"

if str(BENCHMARK_CODE) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_CODE))

from binary_mask_generator import generate_binary_masks  # noqa: E402
from calculate_and_write_metrics_to_csv import process_videos  # noqa: E402
from calculate_iq_score import calculate_iq_score  # noqa: E402


@dataclass(frozen=True)
class DemoMethod:
    key: str
    model_name: str
    mode: str
    display_name: str


METHODS: dict[str, DemoMethod] = {
    "wan": DemoMethod(
        key="wan",
        model_name="wan22_ti2v_5b_demo",
        mode="wan_ti2v",
        display_name="Wan 2.2 TI2V 5B",
    ),
    "vace": DemoMethod(
        key="vace",
        model_name="vace_v2v_ctx08f_demo",
        mode="vace_v2v_ctx8f",
        display_name="VACE 1.3B V2V ctx08",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Physics-IQ demo subset for Wan and/or VACE, then compute subset score."
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="wan,vace",
        help="Comma-separated methods. Supported: wan,vace",
    )
    parser.add_argument(
        "--cases",
        type=str,
        required=True,
        help="Comma-separated take-1 case ids, e.g. 0002,0041. Internally expanded to full 3-view scenarios.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run folder name under output_root/runs/. Defaults to scenario ids.",
    )
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--descriptions_csv", type=Path, default=DEFAULT_DESCRIPTIONS_CSV)
    parser.add_argument("--dataset_root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--wan_root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--vace_root", type=Path, default=DEFAULT_VACE_ROOT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wan_device", type=str, default=None)
    parser.add_argument("--vace_device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wan_height", type=int, default=720)
    parser.add_argument("--wan_width", type=int, default=1280)
    parser.add_argument("--wan_fps", type=int, default=30)
    parser.add_argument("--wan_num_frames", type=int, default=151)
    parser.add_argument("--wan_steps", type=int, default=50)
    parser.add_argument("--wan_cfg_scale", type=float, default=5.0)
    parser.add_argument("--vace_height", type=int, default=544)
    parser.add_argument("--vace_width", type=int, default=720)
    parser.add_argument("--vace_fps", type=int, default=16)
    parser.add_argument("--vace_num_frames", type=int, default=81)
    parser.add_argument("--vace_context_frames", type=int, default=8)
    parser.add_argument("--vace_steps", type=int, default=50)
    parser.add_argument("--vace_cfg_scale", type=float, default=5.0)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    return parser.parse_args()


def load_rows(descriptions_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with descriptions_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "take-1" not in row["scenario"]:
                continue
            rows.append(row)
    rows.sort(key=lambda row: row["generated_video_name"])
    return rows


def load_csv_fieldnames(descriptions_csv: Path) -> list[str]:
    with descriptions_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def parse_case_ids(raw_cases: str) -> list[str]:
    values = []
    for item in raw_cases.split(","):
        token = item.strip()
        if not token:
            continue
        if not token.isdigit():
            raise ValueError(f"Invalid case id: {token}")
        values.append(f"{int(token):04d}")
    if not values:
        raise ValueError("No valid cases provided.")
    return values


def select_rows(rows: list[dict[str, str]], case_ids: list[str]) -> list[dict[str, str]]:
    by_id = {row["generated_video_name"].split("_", 1)[0]: row for row in rows}
    scenario_names: set[str] = set()
    for case_id in case_ids:
        row = by_id.get(case_id)
        if row is None:
            raise KeyError(f"case id not found in take-1 rows: {case_id}")
        scenario_names.add(Path(row["scenario"]).stem.split("_", 3)[3])

    selected = [row for row in rows if Path(row["scenario"]).stem.split("_", 3)[3] in scenario_names]
    if not selected:
        raise ValueError("No rows selected after scenario expansion.")
    return selected


def build_run_name(case_ids: list[str], explicit: str | None) -> str:
    if explicit:
        return explicit
    joined = "-".join(case_ids[:8])
    if len(case_ids) > 8:
        joined += f"-plus{len(case_ids) - 8}"
    return f"cases_{joined}"


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(" ".join(str(part) for part in cmd))
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=env)


def real_testing_path(dataset_root: Path, row: dict[str, str]) -> Path:
    return dataset_root / "split-videos" / "testing" / "30FPS" / row["scenario"]


def real_mask_path(dataset_root: Path, row: dict[str, str]) -> Path:
    return dataset_root / "video-masks" / "real" / "30FPS" / row["scenario"].replace("testing-videos", "video-masks")


def conditioning_path(dataset_root: Path, row: dict[str, str]) -> Path:
    return dataset_root / "split-videos" / "conditioning" / "30FPS" / row["scenario"].replace("testing-videos", "conditioning-videos")


def sample_id_from_row(row: dict[str, str]) -> str:
    return Path(row["generated_video_name"]).stem


def build_vace_meta(case_row: dict[str, str], run_root: Path) -> dict[str, Any]:
    sample_id = sample_id_from_row(case_row)
    sample_dir = DATASET_ROOT / "mytest" / sample_id
    first_frame = sample_dir / "first_frame.png"
    context_path = conditioning_path(DATASET_ROOT, case_row)
    source_paths: dict[str, str] = {
        "sample_dir": str(sample_dir),
        "context_video_path": str(context_path),
        "future_gt_video_path": str(real_testing_path(DATASET_ROOT, case_row)),
        "full_video_path": str(real_testing_path(DATASET_ROOT, case_row)),
        "meta_json_path": str(sample_dir / "meta.json"),
    }
    if first_frame.exists():
        source_paths["first_frame_path"] = str(first_frame)
    return {
        "dataset": "physics-IQ-demo",
        "sample_id": sample_id,
        "caption": case_row["description"],
        "scenario": case_row["scenario"],
        "context_path": str(context_path),
        "context_resize_mode": "crop",
        "source_paths": source_paths,
        "output_name": f"{sample_id}.mp4",
        "run_root": str(run_root),
    }


def stage_subset_eval_assets(
    *,
    selected_rows: list[dict[str, str]],
    method_name: str,
    run_root: Path,
    generated_dir: Path,
) -> tuple[Path, Path, Path]:
    staging_root = run_root / "eval_subset_staging" / method_name
    real_dir = staging_root / "real_testing_videos"
    real_mask_dir = staging_root / "real_masks"
    gen_dir = staging_root / "generated_videos"
    ensure_clean_dir(staging_root, overwrite=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    real_mask_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    for row in selected_rows:
        gen_src = generated_dir / f"{sample_id_from_row(row)}.mp4"
        if not gen_src.exists():
            raise FileNotFoundError(f"Missing generated video: {gen_src}")
        os.symlink(gen_src, gen_dir / gen_src.name)
    for row in selected_rows:
        parts = Path(row["scenario"]).stem.split("_", 3)
        view = parts[1]
        scenario_name = parts[3]
        take1_glob = f"*_testing-videos_30FPS_{view}_take-1_{scenario_name}.mp4"
        take2_glob = f"*_testing-videos_30FPS_{view}_take-2_{scenario_name}.mp4"
        real_matches = list((DATASET_ROOT / "split-videos" / "testing" / "30FPS").glob(take1_glob))
        real_matches += list((DATASET_ROOT / "split-videos" / "testing" / "30FPS").glob(take2_glob))
        mask_matches = list((DATASET_ROOT / "video-masks" / "real" / "30FPS").glob(take1_glob.replace("testing-videos", "video-masks")))
        mask_matches += list((DATASET_ROOT / "video-masks" / "real" / "30FPS").glob(take2_glob.replace("testing-videos", "video-masks")))
        if len(real_matches) != 2:
            raise FileNotFoundError(f"Expected 2 real videos for {scenario_name} {view}, got {len(real_matches)}")
        if len(mask_matches) != 2:
            raise FileNotFoundError(f"Expected 2 real masks for {scenario_name} {view}, got {len(mask_matches)}")
        for src in real_matches:
            dst = real_dir / src.name
            if not dst.exists():
                os.symlink(src, dst)
        for src in mask_matches:
            dst = real_mask_dir / src.name
            if not dst.exists():
                os.symlink(src, dst)
    return real_dir, real_mask_dir, gen_dir


def detect_single_fps(folder: Path) -> int:
    import cv2

    fps_values: set[int] = set()
    for path in sorted(folder.glob("*.mp4")):
        cap = cv2.VideoCapture(str(path))
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        fps_values.add(int(fps))
    if len(fps_values) != 1:
        raise ValueError(f"Inconsistent FPS in {folder}: {fps_values}")
    return next(iter(fps_values))


def build_subset_csv(
    *,
    run_root: Path,
    method: DemoMethod,
    selected_rows: list[dict[str, str]],
    generated_dir: Path,
) -> tuple[Path, float, float]:
    real_dir, real_mask_dir, gen_dir = stage_subset_eval_assets(
        selected_rows=selected_rows,
        method_name=method.model_name,
        run_root=run_root,
        generated_dir=generated_dir,
    )
    fps = detect_single_fps(gen_dir)
    subset_mask_stage = run_root / "eval_subset_staging" / method.model_name / "generated_masks"
    if subset_mask_stage.exists():
        shutil.rmtree(subset_mask_stage)
    subset_mask_stage.mkdir(parents=True, exist_ok=True)
    generate_binary_masks(str(gen_dir), str(subset_mask_stage), False)

    result_dir = run_root / "eval_outputs" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / f"{method.model_name}.csv"
    process_videos(
        real_folders=str(real_dir),
        generated_folders=str(gen_dir),
        binary_real_folders=str(real_mask_dir),
        binary_generated_folders=str(subset_mask_stage),
        csv_file_path=str(csv_path),
        fps=fps,
        video_time_selection="first",
    )
    score, variance = calculate_iq_score(str(csv_path))
    return csv_path, score, variance


def build_viewer_payload(
    *,
    run_root: Path,
    selected_rows: list[dict[str, str]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    rows_by_method = {item["method_key"]: item for item in results}
    cases: list[dict[str, Any]] = []
    for row in selected_rows:
        sample_id = sample_id_from_row(row)
        sample_dir = DATASET_ROOT / "mytest" / sample_id
        first_frame_path = sample_dir / "first_frame.png"
        if not first_frame_path.exists():
            first_frame_path = DATASET_ROOT / "switch-frames" / (
                row["generated_video_name"].replace(".mp4", ".jpg").replace(
                    f"{sample_id.split('_', 1)[0]}_{sample_id.split('_', 2)[1]}_",
                    f"{sample_id.split('_', 1)[0]}_switch-frames_anyFPS_{sample_id.split('_', 2)[1]}_",
                )
            )
        case_payload = {
            "sample_id": sample_id,
            "caption": row["description"],
            "scenario": row["scenario"],
            "paths": {
                "first_frame_path": str(first_frame_path),
                "context_video_path": str(conditioning_path(DATASET_ROOT, row)),
                "future_gt_video_path": str(real_testing_path(DATASET_ROOT, row)),
            },
            "predictions": [],
        }
        for method_key, method_result in rows_by_method.items():
            pred_path = Path(method_result["generated_dir"]) / f"{sample_id}.mp4"
            if pred_path.exists():
                case_payload["predictions"].append(
                    {
                        "method_key": method_key,
                        "method_name": method_result["display_name"],
                        "video_path": str(pred_path),
                    }
                )
        cases.append(case_payload)

    return {
        "run_root": str(run_root),
        "summary": results,
        "cases": cases,
    }


def build_viewer_html(payload: dict[str, Any]) -> str:
    summary_cards = []
    for item in payload["summary"]:
        summary_cards.append(
            f"""
            <div class="stat">
              <div class="stat-name">{item['display_name']}</div>
              <div class="stat-score">{item.get('subset_score', 'NA')}</div>
              <div class="stat-note">{item['method_name']}</div>
            </div>
            """
        )
    cards_html = "\n".join(summary_cards)

    rows_html = []
    for case in payload["cases"]:
        pred_cells = []
        for pred in case["predictions"]:
            pred_cells.append(
                f"""
                <div class="panel">
                  <div class="label">{pred['method_name']}</div>
                  <video controls preload="metadata" src="/files/{Path(pred['video_path']).as_posix()}"></video>
                </div>
                """
            )
        rows_html.append(
            f"""
            <section class="row">
              <div class="meta">
                <div class="sample-id">{case['sample_id']}</div>
                <p>{case['caption']}</p>
              </div>
              <div class="panel">
                <div class="label">first frame</div>
                <img src="/files/{Path(case['paths']['first_frame_path']).as_posix()}" alt="{case['sample_id']}">
              </div>
              <div class="panel">
                <div class="label">context</div>
                <video controls preload="metadata" src="/files/{Path(case['paths']['context_video_path']).as_posix()}"></video>
              </div>
              <div class="panel">
                <div class="label">future gt</div>
                <video controls preload="metadata" src="/files/{Path(case['paths']['future_gt_video_path']).as_posix()}"></video>
              </div>
              {''.join(pred_cells)}
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Physics-IQ Demo</title>
  <style>
    :root {{
      --bg: #f5f1e9;
      --panel: #fffdf9;
      --ink: #1d252c;
      --muted: #5e6972;
      --line: #d9d0c3;
      --accent: #a34d2b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(163,77,43,0.1), transparent 24%),
        linear-gradient(180deg, #fbf7ef 0%, var(--bg) 100%);
    }}
    .wrap {{ width: min(1800px, calc(100vw - 20px)); margin: 0 auto; padding: 16px 0 28px; }}
    h1 {{ margin: 0 0 6px; font-size: 30px; }}
    .sub {{ color: var(--muted); margin: 0 0 16px; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin-bottom: 16px; }}
    .stat, .panel, .meta {{
      background: rgba(255, 253, 249, 0.92);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
      box-shadow: 0 10px 26px rgba(50, 42, 31, 0.06);
    }}
    .stat-name {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .stat-score {{ margin-top: 6px; font-size: 28px; font-weight: 700; }}
    .stat-note {{ margin-top: 4px; font-size: 13px; color: var(--muted); }}
    .row {{ display: grid; grid-template-columns: 340px repeat(4, minmax(220px, 1fr)); gap: 10px; margin-bottom: 10px; align-items: start; }}
    .sample-id {{ font-family: "IBM Plex Mono", "SFMono-Regular", monospace; font-size: 13px; margin-bottom: 8px; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }}
    img, video {{ width: 100%; display: block; border-radius: 10px; background: #ebe4d8; border: 1px solid #ddd4c8; }}
    img {{ aspect-ratio: 1 / 1; object-fit: cover; }}
    video {{ aspect-ratio: 1 / 1; object-fit: contain; }}
    p {{ margin: 0; font-size: 14px; line-height: 1.5; white-space: pre-wrap; }}
    @media (max-width: 1400px) {{ .row {{ grid-template-columns: 1fr 1fr; }} }}
    @media (max-width: 720px) {{ .row {{ grid-template-columns: 1fr; }} .wrap {{ width: min(100vw, calc(100vw - 10px)); }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Physics-IQ Demo</h1>
    <p class="sub">Subset demo score only. Each selected scenario is expanded to all three take-1 views before evaluation.</p>
    <section class="stats">{cards_html}</section>
    {''.join(rows_html)}
  </div>
</body>
</html>
"""


def build_viewer(run_root: Path, payload: dict[str, Any]) -> Path:
    viewer_dir = run_root / "viewer"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    write_json(viewer_dir / "manifest.json", payload)
    (viewer_dir / "index.html").write_text(build_viewer_html(payload), encoding="utf-8")
    return viewer_dir / "index.html"


def launch_server(viewer_dir: Path, host: str, port: int) -> None:
    server_code = f"""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import mimetypes
import os
from urllib.parse import unquote

viewer_dir = Path({viewer_dir.as_posix()!r})

class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        raw = unquote(path.split('?', 1)[0])
        if raw.startswith('/files/'):
            return raw[len('/files'):]
        target = viewer_dir / raw.lstrip('/')
        if raw == '/':
            target = viewer_dir / 'index.html'
        return str(target)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

ThreadingHTTPServer(({host!r}, {port}), Handler).serve_forever()
"""
    cmd = [sys.executable, "-c", server_code]
    print(f"Serving {viewer_dir} at http://127.0.0.1:{port}")
    subprocess.run(cmd, check=True)


def run_wan_generation(
    *,
    args: argparse.Namespace,
    method: DemoMethod,
    selected_rows: list[dict[str, str]],
    run_root: Path,
) -> Path:
    generated_dir = run_root / "generated_videos" / method.model_name
    generated_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_generation:
        return generated_dir

    fieldnames = load_csv_fieldnames(args.descriptions_csv)
    subset_desc = run_root / "meta" / method.model_name / "descriptions_subset.csv"
    subset_desc.parent.mkdir(parents=True, exist_ok=True)
    with subset_desc.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    script_args = [
        sys.executable,
        str(WAN_SCRIPT),
        "--model_root",
        str(args.wan_root),
        "--device",
        args.wan_device or args.device,
        "--output_root",
        str(run_root),
        "--descriptions_csv",
        str(subset_desc),
        "--height",
        str(args.wan_height),
        "--width",
        str(args.wan_width),
        "--fps",
        str(args.wan_fps),
        "--num_frames",
        str(args.wan_num_frames),
        "--seed",
        str(args.seed),
        "--model_name",
        method.model_name,
        "--skip_evaluation",
        "--allow_subset_cases",
    ]
    if args.overwrite:
        script_args.append("--overwrite")
    run_cmd(script_args, cwd=BENCH_DIR)
    return generated_dir


def run_vace_generation(
    *,
    args: argparse.Namespace,
    method: DemoMethod,
    selected_rows: list[dict[str, str]],
    run_root: Path,
) -> Path:
    generated_dir = run_root / "generated_videos" / method.model_name
    generated_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_generation:
        return generated_dir

    meta_dir = run_root / "meta" / method.model_name
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_paths: list[Path] = []
    for row in selected_rows:
        meta = build_vace_meta(row, run_root)
        meta_path = meta_dir / f"{meta['sample_id']}.json"
        write_json(meta_path, meta)
        meta_paths.append(meta_path)
    meta_list_path = meta_dir / "meta_paths.txt"
    meta_list_path.write_text("\n".join(str(path) for path in meta_paths) + "\n", encoding="utf-8")

    runtime_root = run_root / "metadata_runtime" / method.model_name
    runtime_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(BATCH_EVAL_VACE),
        "--vace_root",
        str(args.vace_root),
        "--meta_list_path",
        str(meta_list_path),
        "--output_root",
        str(generated_dir),
        "--runtime_root",
        str(runtime_root),
        "--model_name",
        method.model_name,
        "--mode",
        "v2v_clipref",
        "--device",
        args.vace_device or args.device,
        "--height",
        str(args.vace_height),
        "--width",
        str(args.vace_width),
        "--fps",
        str(args.vace_fps),
        "--num_frames",
        str(args.vace_num_frames),
        "--context_frames",
        str(args.vace_context_frames),
        "--num_inference_steps",
        str(args.vace_steps),
        "--cfg_scale",
        str(args.vace_cfg_scale),
        "--seed",
        str(args.seed),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    run_cmd(cmd, cwd=TRAIN0419_ROOT)
    return generated_dir


def main() -> None:
    args = parse_args()
    methods = []
    for item in args.methods.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in METHODS:
            raise KeyError(f"Unsupported method: {key}")
        methods.append(METHODS[key])
    if not methods:
        raise ValueError("No methods selected.")

    rows = load_rows(args.descriptions_csv)
    case_ids = parse_case_ids(args.cases)
    selected_rows = select_rows(rows, case_ids)
    run_name = build_run_name(case_ids, args.run_name)
    run_root = args.output_root / "runs" / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    selection_payload = {
        "requested_case_ids": case_ids,
        "expanded_sample_ids": [sample_id_from_row(row) for row in selected_rows],
        "num_requested_case_ids": len(case_ids),
        "num_expanded_cases": len(selected_rows),
        "expanded_scenarios": sorted({Path(row["scenario"]).stem.split("_", 3)[3] for row in selected_rows}),
    }
    write_json(run_root / "selection.json", selection_payload)
    write_jsonl(run_root / "selection_rows.jsonl", selected_rows)

    results: list[dict[str, Any]] = []
    for method in methods:
        if method.mode == "wan_ti2v":
            generated_dir = run_wan_generation(
                args=args,
                method=method,
                selected_rows=selected_rows,
                run_root=run_root,
            )
        elif method.mode == "vace_v2v_ctx8f":
            generated_dir = run_vace_generation(
                args=args,
                method=method,
                selected_rows=selected_rows,
                run_root=run_root,
            )
        else:
            raise ValueError(f"Unknown mode: {method.mode}")

        result_item: dict[str, Any] = {
            "method_key": method.key,
            "method_name": method.model_name,
            "display_name": method.display_name,
            "generated_dir": str(generated_dir),
        }
        if not args.skip_evaluation:
            csv_path, score, variance = build_subset_csv(
                run_root=run_root,
                method=method,
                selected_rows=selected_rows,
                generated_dir=generated_dir,
            )
            result_item.update(
                {
                    "eval_csv": str(csv_path),
                    "subset_score": score,
                    "physical_variance_mean": variance,
                    "num_subset_cases": len(selected_rows),
                    "num_requested_case_ids": len(case_ids),
                }
            )
            write_json(
                run_root / "eval_outputs" / "results" / f"{method.model_name}.subset_score.json",
                result_item,
            )
        results.append(result_item)

    write_json(run_root / "summary.json", {"run_name": run_name, "results": results})
    viewer_payload = build_viewer_payload(run_root=run_root, selected_rows=selected_rows, results=results)
    viewer_index = build_viewer(run_root, viewer_payload)

    print(f"Run root: {run_root}")
    print(f"Viewer: {viewer_index}")
    if args.serve:
        launch_server(viewer_index.parent, args.host, args.port)


if __name__ == "__main__":
    main()
