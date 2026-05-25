#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_DESCRIPTIONS_CSV = Path("/home/gaoya/Code_Video/physics-IQ-benchmark-main/descriptions/descriptions.csv")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/physics_IQ_demo/compare_viewer")
DEFAULT_WAN_DIR = Path("/data/gaoya/AAA_test_video/Benchmark/physics_IQ/generated_videos/wan_22_ti2v_5b")
DEFAULT_VACE_DIR = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption/output/VACE_1_3B_V2V/context_08f")
DEFAULT_DATASET_ROOT = Path("/data/gaoya/dataset/physics-iq-benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local Physics-IQ compare viewer from existing Wan/VACE outputs.")
    parser.add_argument("--cases", type=str, required=True, help="Comma-separated take-1 center-view case ids.")
    parser.add_argument("--descriptions_csv", type=Path, default=DEFAULT_DESCRIPTIONS_CSV)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--wan_dir", type=Path, default=DEFAULT_WAN_DIR)
    parser.add_argument("--vace_dir", type=Path, default=DEFAULT_VACE_DIR)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    return parser.parse_args()


def parse_case_ids(raw: str) -> list[str]:
    values = []
    for item in raw.split(","):
        token = item.strip()
        if token:
            values.append(f"{int(token):04d}")
    return values


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if "take-1" in row["scenario"]]
    rows.sort(key=lambda row: row["generated_video_name"])
    return rows


def build_first_frame_path(dataset_root: Path, row: dict[str, str]) -> Path:
    stem = Path(row["generated_video_name"]).stem
    file_id, perspective, scenario = stem.split("_", 2)
    return dataset_root / "switch-frames" / f"{file_id}_switch-frames_anyFPS_{perspective}_{scenario}.jpg"


def split_scenario_parts(row: dict[str, str]) -> tuple[str, str, str, str]:
    return Path(row["scenario"]).stem.split("_", 3)


def build_context_path(dataset_root: Path, row: dict[str, str]) -> Path:
    file_id, perspective, take, scenario = split_scenario_parts(row)
    filename = f"{file_id}_conditioning-videos_30FPS_{perspective}_{take}_{scenario}.mp4"
    return dataset_root / "split-videos" / "conditioning" / "30FPS" / filename


def build_future_gt_path(dataset_root: Path, row: dict[str, str]) -> Path:
    file_id, perspective, take, scenario = split_scenario_parts(row)
    filename = f"{file_id}_testing-videos_30FPS_{perspective}_{take}_{scenario}.mp4"
    return dataset_root / "split-videos" / "testing" / "30FPS" / filename


def build_vace_name(row: dict[str, str]) -> str:
    stem = Path(row["generated_video_name"]).stem
    return f"physics-iq-benchmark__{stem}.mp4"


def build_payload(args: argparse.Namespace) -> dict:
    case_ids = set(parse_case_ids(args.cases))
    rows = load_rows(args.descriptions_csv)
    selected = [row for row in rows if row["generated_video_name"].split("_", 1)[0] in case_ids]
    selected.sort(key=lambda row: row["generated_video_name"])
    cases = []
    for row in selected:
        wan_path = args.wan_dir / row["generated_video_name"]
        vace_path = args.vace_dir / build_vace_name(row)
        cases.append(
            {
                "sample_id": Path(row["generated_video_name"]).stem,
                "caption": row["description"],
                "first_frame_path": str(build_first_frame_path(args.dataset_root, row)),
                "context_video_path": str(build_context_path(args.dataset_root, row)),
                "future_gt_video_path": str(build_future_gt_path(args.dataset_root, row)),
                "wan_video_path": str(wan_path),
                "vace_video_path": str(vace_path),
                "has_wan": wan_path.exists(),
                "has_vace": vace_path.exists(),
            }
        )
    return {
        "summary": {
            "case_ids": sorted(case_ids),
            "num_cases": len(cases),
            "wan_input": "first-frame image + caption",
            "vace_input": "8-frame context video + caption",
            "context_frames": 8,
            "output": "predicted future video",
            "context_source": "official Physics-IQ conditioning split",
            "future_gt_source": "official Physics-IQ testing split",
        },
        "cases": cases,
    }


def build_html(payload: dict) -> str:
    summary = payload["summary"]
    rows = []
    for case in payload["cases"]:
        rows.append(
            f"""
            <section class="row">
              <div class="meta">
                <div class="sample-id">{case['sample_id']}</div>
                <p>{case['caption']}</p>
              </div>
              <div class="panel">
                <div class="label">first frame</div>
                <img src="/files/{case['first_frame_path']}" alt="{case['sample_id']}">
              </div>
              <div class="panel">
                <div class="label">context</div>
                <video controls preload="metadata" src="/files/{case['context_video_path']}"></video>
              </div>
              <div class="panel">
                <div class="label">future gt</div>
                <video controls preload="metadata" src="/files/{case['future_gt_video_path']}"></video>
              </div>
              <div class="panel">
                <div class="label">wan</div>
                <video controls preload="metadata" src="/files/{case['wan_video_path']}"></video>
              </div>
              <div class="panel">
                <div class="label">vace ctx08</div>
                <video controls preload="metadata" src="/files/{case['vace_video_path']}"></video>
              </div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Physics-IQ Compare Viewer</title>
  <style>
    :root {{
      --bg: #f5f1e9; --panel: #fffdf9; --ink: #1d252c; --muted: #5e6972; --line: #d9d0c3;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; color: var(--ink); background: linear-gradient(180deg, #fbf7ef 0%, var(--bg) 100%); }}
    .wrap {{ width: min(1900px, calc(100vw - 20px)); margin: 0 auto; padding: 16px 0 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .sub {{ margin: 0 0 16px; color: var(--muted); }}
    .top {{ display: grid; grid-template-columns: 1.3fr 1fr; gap: 10px; margin-bottom: 14px; }}
    .config {{ background: rgba(255, 253, 249, 0.94); border: 1px solid var(--line); border-radius: 16px; padding: 14px; }}
    .config-grid {{ display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 10px 16px; }}
    .config-item {{ font-size: 14px; line-height: 1.45; }}
    .config-key {{ display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }}
    .row {{ display: grid; grid-template-columns: 300px repeat(5, minmax(220px, 1fr)); gap: 10px; margin-bottom: 10px; }}
    .panel, .meta {{ background: rgba(255, 253, 249, 0.92); border: 1px solid var(--line); border-radius: 16px; padding: 12px; }}
    .sample-id {{ font-family: "IBM Plex Mono", "SFMono-Regular", monospace; font-size: 13px; margin-bottom: 8px; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }}
    img, video {{ width: 100%; display: block; border-radius: 10px; background: #ebe4d8; border: 1px solid #ddd4c8; }}
    img {{ aspect-ratio: 1 / 1; object-fit: cover; }}
    video {{ aspect-ratio: 1 / 1; object-fit: contain; }}
    p {{ margin: 0; font-size: 14px; line-height: 1.5; white-space: pre-wrap; }}
    @media (max-width: 1500px) {{ .row {{ grid-template-columns: 1fr 1fr; }} .top {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 720px) {{ .row {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Physics-IQ Compare Viewer</h1>
    <p class="sub">Existing local outputs only. Current page compares center-view cases where both Wan and VACE ctx08 results already exist locally.</p>
    <section class="top">
      <div class="config">
        <div class="config-grid">
          <div class="config-item"><span class="config-key">Wan Input</span>{summary['wan_input']}</div>
          <div class="config-item"><span class="config-key">VACE Input</span>{summary['vace_input']}</div>
          <div class="config-item"><span class="config-key">Caption</span>Each row uses the official Physics-IQ text description shown in the left meta panel.</div>
          <div class="config-item"><span class="config-key">Output</span>{summary['output']}</div>
          <div class="config-item"><span class="config-key">Context Frames</span>{summary['context_frames']} frames for VACE ctx08. Wan uses the switch frame as image condition.</div>
          <div class="config-item"><span class="config-key">GT Source</span>{summary['future_gt_source']}</div>
        </div>
      </div>
      <div class="config">
        <div class="config-grid">
          <div class="config-item"><span class="config-key">Cases</span>{", ".join(summary['case_ids'])}</div>
          <div class="config-item"><span class="config-key">Num Cases</span>{summary['num_cases']}</div>
          <div class="config-item"><span class="config-key">Context Source</span>{summary['context_source']}</div>
          <div class="config-item"><span class="config-key">Columns</span>first frame / context / future gt / wan / vace ctx08</div>
        </div>
      </div>
    </section>
    {''.join(rows)}
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args)
    (args.output_root / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "index.html").write_text(build_html(payload), encoding="utf-8")
    print(args.output_root / "index.html")


if __name__ == "__main__":
    main()
