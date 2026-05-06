#!/usr/bin/env python3
"""Build a portal with one representative case per dataset for context-length comparison."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import batch_eval_lora as bel


DEFAULT_BENCHMARK_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption")
DEFAULT_ORIG_BENCHMARK_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V")
DEFAULT_CONTEXT_LENGTHS = [8, 16, 32, 38]
REPRESENTATIVES = [
    (
        "physics-iq-benchmark",
        "0005_perspective-center_trimmed-ball-behind-rotating-paper",
        "/data/gaoya/dataset/physics-iq-benchmark/mytest/0005_perspective-center_trimmed-ball-behind-rotating-paper/meta.json",
    ),
    (
        "kubric_tfds_movi-d",
        "movi_d_test_0005__video_668",
        "/data/gaoya/dataset/kubric_tfds_movi-d/mytest/movi_d_test_0005__video_668/meta.json",
    ),
    (
        "mvp-lab-OpenVidHD-0.4M-720p-48fps",
        "rank0_1761115610.0727706_720x1280__00065__zy_NvBKW6O4_35_0to121",
        "/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/mytest/rank0_1761115610.0727706_720x1280__00065__zy_NvBKW6O4_35_0to121/meta.json",
    ),
    (
        "vLAR-PhysInOne",
        "A__ObliqueProjectile_RollUpSlope_LinCarryInertia__bg158__LoFxHr_trajectory__CineCamera_0",
        "/data/gaoya/dataset/vLAR-PhysInOne/mytest/A__ObliqueProjectile_RollUpSlope_LinCarryInertia__bg158__LoFxHr_trajectory__CineCamera_0/meta.json",
    ),
    (
        "version_1_genesis_rigid_data_all_cases",
        "genesis_heldout_0001__10005__case000_static_center_v2",
        "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/mytest/genesis_heldout_0001__10005__case000_static_center_v2/meta.json",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--orig_benchmark_root", type=Path, default=DEFAULT_ORIG_BENCHMARK_ROOT)
    parser.add_argument(
        "--portal_subdir",
        type=Path,
        default=Path("tools/visualization/dataset_representative_context_sweeps"),
    )
    parser.add_argument("--context-lengths", default="8,16,32,38")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def relative_to_root(root: Path, path: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def web_path(path: str | None) -> str | None:
    if not path:
        return None
    return "/" + path.replace(os.sep, "/").lstrip("/")


def ensure_clean_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()


def ensure_symlink(target: Path, link_path: Path) -> Path | None:
    if not target.exists():
        return None
    ensure_clean_parent(link_path)
    link_path.symlink_to(target)
    return link_path


def expose_asset(
    *,
    target: Path | None,
    benchmark_root: Path,
    assets_dir: Path,
    link_name: str,
) -> str | None:
    if target is None or not target.exists():
        return None
    if target.is_relative_to(benchmark_root):
        return relative_to_root(benchmark_root, target)
    linked = ensure_symlink(target, assets_dir / link_name)
    if linked is None:
        return None
    return relative_to_root(benchmark_root, linked)


def save_context_clip(
    *,
    context_path: Path,
    context_frames: int,
    height: int,
    width: int,
    resize_mode: str,
    fps: int,
    output_path: Path,
) -> Path:
    frames = bel.load_context_frames(
        context_path=context_path,
        context_frames=context_frames,
        height=height,
        width=width,
        resize_mode=resize_mode,
    )
    bel.save_video(frames, str(output_path), fps=fps, quality=5)
    return output_path


def media_html(path: str | None) -> str:
    if not path:
        return "<div class='missing'>Missing</div>"
    resolved = web_path(path)
    return (
        f"<video controls preload='metadata' muted playsinline>"
        f"<source src='{html.escape(resolved or '')}' type='video/mp4'>"
        "</video>"
    )


def text_html(text: str) -> str:
    return f"<div class='text-body'>{html.escape(text)}</div>"


def render_slot(title: str, body: str, extra_class: str = "") -> str:
    class_attr = "slot"
    if extra_class:
        class_attr += f" {extra_class}"
    return (
        f"<section class='{class_attr}'>"
        f"<div class='slot-head'>{html.escape(title)}</div>"
        f"{body}"
        "</section>"
    )


def dataset_folder_name(dataset: str, sample_id: str) -> str:
    return f"{dataset}__{sample_id}"


def resolve_generated_video(
    *,
    benchmark_root: Path,
    orig_benchmark_root: Path,
    dataset: str,
    sample_id: str,
    context_frames: int,
    assets_dir: Path,
) -> str | None:
    if context_frames == 8:
        target = (
            orig_benchmark_root
            / "output"
            / "VACE_1_3B_V2V"
            / "context_08f"
            / f"{dataset}__{sample_id}.mp4"
        )
        return expose_asset(
            target=target,
            benchmark_root=benchmark_root,
            assets_dir=assets_dir,
            link_name=f"generated_context_{context_frames:02d}f.mp4",
        )
    sidecar_path = (
        benchmark_root
        / "tools"
        / "dataset_representative_context_sweeps"
        / "generated"
        / f"batch_context_{context_frames:02d}f"
        / f"{dataset}__{sample_id}.json"
    )
    if not sidecar_path.exists():
        return None
    sidecar = read_json(sidecar_path)
    raw_output = sidecar.get("paths", {}).get("output_video_path")
    if not isinstance(raw_output, str):
        return None
    return expose_asset(
        target=Path(raw_output),
        benchmark_root=benchmark_root,
        assets_dir=assets_dir,
        link_name=f"generated_context_{context_frames:02d}f.mp4",
    )


def build_case_block(
    *,
    benchmark_root: Path,
    orig_benchmark_root: Path,
    dataset: str,
    sample_id: str,
    meta_json_path: Path,
    context_lengths: list[int],
    case_assets_dir: Path,
) -> str:
    meta = read_json(meta_json_path)
    paths = meta.get("paths", {})
    if not isinstance(paths, dict):
        return ""
    context_path = Path(paths["context_video_path"])
    gt_full_path = Path(paths["full_video_path"])
    caption = str(meta.get("caption") or "")
    source_fps = int(float(meta.get("fps") or 30))
    context_range = meta.get("context_frame_range") or [0, 0]
    source_context_frames = int(context_range[1]) - int(context_range[0]) + 1 if len(context_range) == 2 else 0
    height = 544
    width = 720
    resize_mode = bel.resolve_context_resize_mode(dataset)

    full_context_asset_path = case_assets_dir / "source_context_full_clip.mp4"
    if source_context_frames > 0:
        save_context_clip(
            context_path=context_path,
            context_frames=source_context_frames,
            height=height,
            width=width,
            resize_mode=resize_mode,
            fps=source_fps,
            output_path=full_context_asset_path,
        )
    else:
        save_context_clip(
            context_path=context_path,
            context_frames=max(context_lengths),
            height=height,
            width=width,
            resize_mode=resize_mode,
            fps=source_fps,
            output_path=full_context_asset_path,
        )

    rows = []
    for context_frames in context_lengths:
        context_clip_path = case_assets_dir / f"actual_context_{context_frames:02d}f.mp4"
        save_context_clip(
            context_path=context_path,
            context_frames=context_frames,
            height=height,
            width=width,
            resize_mode=resize_mode,
            fps=source_fps,
            output_path=context_clip_path,
        )
        summary_text = (
            f"used_context_frames: {context_frames}\n"
            f"source_context_fps: {source_fps}\n"
            f"approx_context_seconds: {context_frames / source_fps:.3f}\n"
            "generated_output_fps: 16\n"
            "generated_output_frames: 49"
        )
        rows.append(
            "<section class='variant-card'>"
            "<div class='variant-head'>"
            f"<h3>context_{context_frames:02d}f</h3>"
            f"{text_html(summary_text)}"
            "</div>"
            "<div class='variant-row'>"
            f"{render_slot('actual_context_video', media_html(relative_to_root(benchmark_root, context_clip_path)), 'context-slot')}"
            f"{render_slot('generated_output', media_html(resolve_generated_video(benchmark_root=benchmark_root, orig_benchmark_root=orig_benchmark_root, dataset=dataset, sample_id=sample_id, context_frames=context_frames, assets_dir=case_assets_dir)), 'output-slot')}"
            "</div>"
            "</section>"
        )

    return (
        "<article class='case-card'>"
        f"<div class='case-head'><span class='badge'>{html.escape(dataset)}</span><h2>{html.escape(sample_id)}</h2></div>"
        "<div class='shared-grid'>"
        f"{render_slot('source_context_full_clip', media_html(relative_to_root(benchmark_root, full_context_asset_path)), 'context-slot')}"
        f"{render_slot('gt_full_video', media_html(expose_asset(target=gt_full_path, benchmark_root=benchmark_root, assets_dir=case_assets_dir, link_name='gt_full_video.mp4')), 'context-slot')}"
        f"{render_slot('caption', text_html(caption), 'meta-slot')}"
        "</div>"
        f"<div class='variant-stack'>{''.join(rows)}</div>"
        "</article>"
    )


def build_html(case_blocks: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dataset Representative Context Sweeps</title>
  <style>
    :root {{
      --bg: #f4f1e8;
      --panel: #fffdf8;
      --ink: #1e1b16;
      --muted: #6e675d;
      --line: #d8cfbf;
      --ok-soft: #d6ead9;
      --ok-ink: #28563c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(181,83,45,0.10), transparent 26%),
        linear-gradient(180deg, #f9f6ef 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .shell {{
      width: min(1900px, calc(100vw - 20px));
      margin: 0 auto;
      padding: 14px 0 24px;
    }}
    .hero {{
      margin-bottom: 12px;
      padding: 14px 18px;
      background: rgba(255,253,248,0.90);
      border: 1px solid var(--line);
      border-radius: 14px;
    }}
    .case-list {{
      display: grid;
      gap: 14px;
    }}
    .case-card {{
      padding: 10px;
      background: rgba(255,253,248,0.96);
      border: 1px solid var(--line);
      border-radius: 12px;
    }}
    .case-head {{
      display: flex;
      gap: 8px;
      align-items: center;
      margin-bottom: 8px;
    }}
    .case-head h2 {{
      margin: 0;
      font-size: 16px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(214, 234, 217, 0.82);
      color: var(--ok-ink);
      font-size: 11px;
      font-weight: 700;
    }}
    .shared-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(220px, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }}
    .variant-stack {{
      display: grid;
      gap: 10px;
    }}
    .variant-card {{
      padding: 10px;
      background: #fbf8f2;
      border: 1px solid var(--line);
      border-radius: 12px;
    }}
    .variant-head {{
      display: grid;
      grid-template-columns: minmax(180px, 240px) 1fr;
      gap: 8px;
      align-items: start;
      margin-bottom: 8px;
    }}
    .variant-head h3 {{
      margin: 0;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(214, 234, 217, 0.82);
      color: var(--ok-ink);
      font-size: 15px;
    }}
    .variant-row {{
      display: grid;
      grid-template-columns: minmax(320px, 1fr) minmax(320px, 1fr);
      gap: 8px;
    }}
    .slot {{
      background: #fffdf8;
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      min-height: 156px;
    }}
    .slot-head {{
      padding: 6px 8px;
      border-bottom: 1px solid var(--line);
      font-size: 11px;
      font-weight: 700;
      color: #55493d;
      background: rgba(239, 231, 218, 0.65);
    }}
    .context-slot .slot-head {{
      background: rgba(232, 223, 208, 0.82);
    }}
    .output-slot .slot-head {{
      background: rgba(243, 215, 201, 0.78);
      color: #6e2a13;
    }}
    .meta-slot .slot-head {{
      background: rgba(214, 234, 217, 0.82);
      color: var(--ok-ink);
    }}
    video {{
      display: block;
      width: 100%;
      min-height: 156px;
      background: #0d0d0d;
    }}
    .text-body {{
      min-height: 156px;
      padding: 8px 10px;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .missing {{
      display: grid;
      place-items: center;
      min-height: 156px;
      padding: 12px;
      color: var(--muted);
      background: repeating-linear-gradient(45deg, rgba(216,207,191,0.35), rgba(216,207,191,0.35) 10px, rgba(255,253,248,0.75) 10px, rgba(255,253,248,0.75) 20px);
    }}
    @media (max-width: 1100px) {{
      .shared-grid, .variant-head, .variant-row {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Dataset Representative Context Sweeps</h1>
      <p>One representative case per dataset. Each context length is shown as a single row with the actual given context video on the left and its corresponding generated output on the right.</p>
    </section>
    <section class="case-list">
      {case_blocks}
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    orig_benchmark_root = args.orig_benchmark_root.expanduser().resolve()
    portal_dir = (benchmark_root / args.portal_subdir).resolve()
    assets_root = portal_dir / "assets"
    portal_dir.mkdir(parents=True, exist_ok=True)
    assets_root.mkdir(parents=True, exist_ok=True)
    context_lengths = [int(x.strip()) for x in args.context_lengths.split(",") if x.strip()]

    blocks = []
    for dataset, sample_id, meta_json in REPRESENTATIVES:
        meta_path = Path(meta_json)
        if not meta_path.exists():
            continue
        case_assets_dir = assets_root / dataset_folder_name(dataset, sample_id)
        case_assets_dir.mkdir(parents=True, exist_ok=True)
        blocks.append(
            build_case_block(
                benchmark_root=benchmark_root,
                orig_benchmark_root=orig_benchmark_root,
                dataset=dataset,
                sample_id=sample_id,
                meta_json_path=meta_path,
                context_lengths=context_lengths,
                case_assets_dir=case_assets_dir,
            )
        )

    html_path = portal_dir / "index.html"
    html_path.write_text(build_html("".join(blocks)), encoding="utf-8")
    write_json(
        portal_dir / "build_summary.json",
        {
            "html_path": str(html_path),
            "portal_url_path": f"/{relative_to_root(benchmark_root, html_path)}",
            "num_cases": len(blocks),
        },
    )
    print(json.dumps({"html_path": str(html_path), "num_cases": len(blocks)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
