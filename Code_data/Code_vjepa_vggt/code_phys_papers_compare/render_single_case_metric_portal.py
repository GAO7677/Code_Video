from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

import cv2


ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
TRY0526_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
for path in (ROOT, TRY0526_ROOT):
    path_str = str(path)
    if path_str not in os.sys.path:
        os.sys.path.insert(0, path_str)

from physv_eval.single_case.physics_iq import score_case as score_physics_iq
from physv_eval.single_case.pmf import score_case as score_pmf


DEFAULT_OUT_ROOT = Path("/data/gaoya/agent-data/outputs/single_case_metric_portal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a browser-friendly portal for one result case, showing the exact videos "
            "or video pairs used by each metric."
        )
    )
    parser.add_argument("--case-json", type=Path, required=True, help="Per-case result json.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--pmf-device", default="cpu")
    parser.add_argument("--physics-iq-downsample-factor", type=int, default=4)
    parser.add_argument("--physics-iq-threshold-value", type=int, default=10)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def ensure_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def safe_stem(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "case"


def first_existing_file(candidates: list[str | None]) -> Path | None:
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        path = Path(candidate).expanduser().resolve()
        if path.is_file():
            return path
    return None


def count_video_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for frame counting: {video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            return frame_count
        decoded = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            decoded += 1
        if decoded <= 0:
            raise RuntimeError(f"Video has no readable frames: {video_path}")
        return decoded
    finally:
        cap.release()


def resolve_source_video(case_payload: dict[str, Any], case_json_path: Path) -> Path:
    direct = first_existing_file(
        [
            case_payload.get("source_video"),
            case_payload.get("reference_video"),
        ]
    )
    if direct is not None:
        return direct

    input_json_value = case_payload.get("input_json")
    if isinstance(input_json_value, str) and input_json_value.strip():
        input_json_path = Path(input_json_value).expanduser().resolve()
        if input_json_path.is_file():
            source_payload = load_json(input_json_path)
            direct = first_existing_file(
                [
                    source_payload.get("source_video"),
                    source_payload.get("gt_full"),
                    source_payload.get("reference_video"),
                ]
            )
            if direct is not None:
                return direct

    input_video_original = first_existing_file([case_payload.get("input_video_original")])
    if input_video_original is not None:
        if (
            input_video_original.name.startswith("context_video_")
            and input_video_original.parent.name == "source_video"
        ):
            candidate = input_video_original.parent.parent / "source_video.mp4"
            if candidate.is_file():
                return candidate.resolve()

    raise FileNotFoundError(f"Could not resolve source/GT video for {case_json_path}")


def resolve_context_video(case_payload: dict[str, Any]) -> Path | None:
    direct = first_existing_file(
        [
            case_payload.get("input_video_original"),
            case_payload.get("context_video"),
            case_payload.get("input_video"),
        ]
    )
    if direct is not None and direct.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        return direct

    input_json_value = case_payload.get("input_json")
    if isinstance(input_json_value, str) and input_json_value.strip():
        input_json_path = Path(input_json_value).expanduser().resolve()
        if input_json_path.is_file():
            source_payload = load_json(input_json_path)
            return first_existing_file(
                [
                    source_payload.get("input_video"),
                    source_payload.get("context_video"),
                ]
            )
    return None


def resolve_context_frames(case_payload: dict[str, Any], context_video: Path | None) -> int:
    for candidate in (
        case_payload.get("effective_context_frames"),
        case_payload.get("context_frames"),
        (case_payload.get("model_args") or {}).get("context_frames")
        if isinstance(case_payload.get("model_args"), dict)
        else None,
    ):
        if isinstance(candidate, int) and candidate >= 0:
            return int(candidate)

    if context_video is not None:
        return count_video_frames(context_video)
    return 0


def symlink_into_assets(asset_dir: Path, source: Path, name: str | None = None) -> Path:
    asset_dir.mkdir(parents=True, exist_ok=True)
    link_path = asset_dir / (name or source.name)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(source)
    return link_path


def relpath(target: Path, base: Path) -> str:
    return os.path.relpath(target.resolve(), start=base.resolve()).replace("\\", "/")


def build_metric_card(
    *,
    title: str,
    summary: str,
    details: list[str],
    videos: list[tuple[str, str]],
) -> str:
    videos_html = "".join(
        f"""
        <div class="video-block">
          <div class="video-label">{html.escape(label)}</div>
          <video controls playsinline preload="metadata" src="{html.escape(src)}"></video>
        </div>
        """
        for label, src in videos
    )
    details_html = "".join(f"<div class='detail'>{html.escape(item)}</div>" for item in details)
    return f"""
    <section class="metric-card">
      <h2>{html.escape(title)}</h2>
      <div class="summary">{html.escape(summary)}</div>
      <div class="details">{details_html}</div>
      <div class="video-grid">{videos_html}</div>
    </section>
    """


def render_index_html(
    *,
    portal_dir: Path,
    case_payload: dict[str, Any],
    source_video: Path,
    context_video: Path | None,
    context_frames: int,
    output_video: Path,
    physics_iq_result: dict[str, Any],
    physics_iq_without_context_result: dict[str, Any],
    pmf_with_context_result: dict[str, Any],
    pmf_without_context_result: dict[str, Any],
) -> str:
    asset_dir = portal_dir / "assets"
    output_link = symlink_into_assets(asset_dir, output_video, "output_video.mp4")
    source_link = symlink_into_assets(asset_dir, source_video, "source_video.mp4")
    context_link = symlink_into_assets(asset_dir, context_video, "context_video.mp4") if context_video else None

    def link_from_result(result: dict[str, Any], key: str, name: str) -> str:
        return relpath(symlink_into_assets(asset_dir, Path(str(result[key])).resolve(), name), portal_dir)

    sections: list[str] = []
    intro_videos = [
        ("Generated output full video", relpath(output_link, portal_dir)),
        ("Source / GT full video", relpath(source_link, portal_dir)),
    ]
    if context_link is not None:
        intro_videos.insert(0, ("Context video fed to PhysRVG", relpath(context_link, portal_dir)))
    sections.append(
        build_metric_card(
            title="Core Inputs",
            summary="This is the original input/output material around which all metrics are computed.",
            details=[
                f"method={case_payload.get('method')}",
                f"case_json={case_payload.get('input_json')}",
                f"context_frames={context_frames}",
                f"caption={case_payload.get('input_caption')}",
            ],
            videos=intro_videos,
        )
    )

    sections.append(
        build_metric_card(
            title="physics_iq",
            summary="Same effective input pair as physics_iq_with_context. The scorer uses full output vs full GT, then internally aligns and downsamples them.",
            details=[
                f"score={physics_iq_result['score']}",
                f"context_mode={physics_iq_result['context_mode']}",
                f"context_frames_used={physics_iq_result['context_frames_used']}",
                f"num_frames_compared={physics_iq_result['num_frames_compared']}",
            ],
            videos=[
                ("Scored output used by physics_iq", link_from_result(physics_iq_result, "scored_output_video", "physics_iq_scored_output.mp4")),
                ("Scored GT used by physics_iq", link_from_result(physics_iq_result, "scored_source_video", "physics_iq_scored_source.mp4")),
                ("Side-by-side compare", link_from_result(physics_iq_result, "compare_side_by_side", "physics_iq_compare.mp4")),
            ],
        )
    )

    sections.append(
        build_metric_card(
            title="physics_iq_with_context",
            summary="Uses the same full generated video and full GT video as the plain physics_iq metric.",
            details=[
                f"score={physics_iq_result['score']}",
                f"context_mode={physics_iq_result['context_mode']}",
                f"output_start_frame={physics_iq_result['output_start_frame']}",
                f"source_start_frame={physics_iq_result['source_start_frame']}",
            ],
            videos=[
                ("Scored output with context", link_from_result(physics_iq_result, "scored_output_video", "physics_iq_with_context_output.mp4")),
                ("Scored GT with context", link_from_result(physics_iq_result, "scored_source_video", "physics_iq_with_context_source.mp4")),
                ("Side-by-side compare", link_from_result(physics_iq_result, "compare_side_by_side", "physics_iq_with_context_compare.mp4")),
            ],
        )
    )

    sections.append(
        build_metric_card(
            title="physics_iq_without_context",
            summary="Drops the leading context frames from both output and GT before alignment and scoring.",
            details=[
                f"score={physics_iq_without_context_result['score']}",
                f"context_mode={physics_iq_without_context_result['context_mode']}",
                f"context_frames_used={physics_iq_without_context_result['context_frames_used']}",
                f"output_start_frame={physics_iq_without_context_result['output_start_frame']}",
            ],
            videos=[
                ("Scored output without context", link_from_result(physics_iq_without_context_result, "scored_output_video", "physics_iq_without_context_output.mp4")),
                ("Scored GT without context", link_from_result(physics_iq_without_context_result, "scored_source_video", "physics_iq_without_context_source.mp4")),
                ("Side-by-side compare", link_from_result(physics_iq_without_context_result, "compare_side_by_side", "physics_iq_without_context_compare.mp4")),
            ],
        )
    )

    sections.append(
        build_metric_card(
            title="pmf_with_context",
            summary="Runs PMF on the full generated video and full GT after resizing GT to output size.",
            details=[
                f"score={pmf_with_context_result['score']}",
                f"context_mode={pmf_with_context_result['context_mode']}",
                f"context_frames_used={pmf_with_context_result['context_frames_used']}",
                f"compare_fps={pmf_with_context_result['compare_fps']}",
            ],
            videos=[
                ("Prediction used for PMF", link_from_result(pmf_with_context_result, "pred_used_for_pmf", "pmf_with_context_pred.mp4")),
                ("GT used for PMF", link_from_result(pmf_with_context_result, "gt_used_for_pmf", "pmf_with_context_gt.mp4")),
                ("Side-by-side compare", link_from_result(pmf_with_context_result, "compare_side_by_side", "pmf_with_context_compare.mp4")),
            ],
        )
    )

    sections.append(
        build_metric_card(
            title="pmf_without_context",
            summary="Drops the context prefix from both generated video and GT, then computes PMF on the remaining future segment.",
            details=[
                f"score={pmf_without_context_result['score']}",
                f"context_mode={pmf_without_context_result['context_mode']}",
                f"context_frames_used={pmf_without_context_result['context_frames_used']}",
                f"output_start_frame={pmf_without_context_result['output_start_frame']}",
            ],
            videos=[
                ("Prediction used for PMF", link_from_result(pmf_without_context_result, "pred_used_for_pmf", "pmf_without_context_pred.mp4")),
                ("GT used for PMF", link_from_result(pmf_without_context_result, "gt_used_for_pmf", "pmf_without_context_gt.mp4")),
                ("Side-by-side compare", link_from_result(pmf_without_context_result, "compare_side_by_side", "pmf_without_context_compare.mp4")),
            ],
        )
    )

    output_video_rel = relpath(output_link, portal_dir)
    sections.append(
        build_metric_card(
            title="wmreward",
            summary="WMReward consumes only the generated output video. No GT pair is used inside the metric.",
            details=[
                "actual_metric_input=output_video_only",
                "bench_default_context_frames=8",
                "bench_default_window_size=16",
                "max_frames=49",
            ],
            videos=[("Generated output video", output_video_rel)],
        )
    )

    sections.append(
        build_metric_card(
            title="videophy2",
            summary="AAAinfer bench default is task=pc, so VideoPhy-2 consumes only the generated output video. The caption field exists in the case JSON but is not used for task=pc.",
            details=[
                "actual_metric_input=output_video_only",
                "bench_default_task=pc",
                "internal_sampled_frames=32",
                f"case_caption={case_payload.get('input_caption')}",
            ],
            videos=[("Generated output video", output_video_rel)],
        )
    )

    sections.append(
        build_metric_card(
            title="cosmos_reason1",
            summary="Cosmos-Reason1 consumes only the generated output video and applies its own prompt template plus internal video sampling.",
            details=[
                "actual_metric_input=output_video_only",
                "bench_default_fps=16",
                "bench_default_total_pixels=8192*28*28",
                "official_prompt=video_reward.yaml",
            ],
            videos=[("Generated output video", output_video_rel)],
        )
    )

    sections_html = "\n".join(sections)
    title = f"{case_payload.get('method')} | {Path(str(case_payload.get('output_video'))).stem}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #eee6d8;
      --panel: rgba(255, 252, 246, 0.95);
      --ink: #1f1a14;
      --muted: #655b4f;
      --accent: #9b4d24;
      --line: #d7c8b3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(155, 77, 36, 0.18), transparent 26%),
        radial-gradient(circle at bottom right, rgba(47, 88, 80, 0.15), transparent 24%),
        linear-gradient(180deg, #f6f0e3 0%, var(--bg) 100%);
    }}
    main {{
      width: min(100vw - 28px, 1560px);
      margin: 20px auto 48px;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 24px;
      box-shadow: 0 18px 40px rgba(31, 26, 20, 0.08);
      margin-bottom: 18px;
    }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{
      font-size: clamp(28px, 4vw, 52px);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .hero-meta {{
      margin-top: 12px;
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 14px;
    }}
    .metric-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      margin-bottom: 16px;
      box-shadow: 0 14px 30px rgba(31, 26, 20, 0.06);
    }}
    .summary {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 15px;
    }}
    .details {{
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 8px 14px;
    }}
    .detail {{
      background: rgba(155, 77, 36, 0.07);
      border-radius: 12px;
      padding: 8px 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    .video-block {{
      background: #fffefb;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 10px;
    }}
    .video-label {{
      font-weight: 700;
      font-size: 13px;
      letter-spacing: 0.02em;
      color: var(--accent);
      margin-bottom: 8px;
    }}
    video {{
      width: 100%;
      display: block;
      border-radius: 12px;
      background: #000;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <div class="hero-meta">
        <div>Portal intent: show the exact video or video pair each metric actually consumes.</div>
        <div>Case json: {html.escape(str(case_payload.get("input_json")))}</div>
        <div>Generated output: {html.escape(str(output_video))}</div>
        <div>Source / GT: {html.escape(str(source_video))}</div>
        <div>Context video: {html.escape(str(context_video)) if context_video else "None"}</div>
      </div>
    </section>
    {sections_html}
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    case_json = ensure_file(args.case_json, "case json")
    case_payload = load_json(case_json)

    output_video = ensure_file(Path(str(case_payload["output_video"])), "output video")
    source_video = resolve_source_video(case_payload, case_json)
    context_video = resolve_context_video(case_payload)
    context_frames = resolve_context_frames(case_payload, context_video)

    method_name = safe_stem(str(case_payload.get("method") or case_json.parent.name))
    case_name = safe_stem(case_json.stem)
    portal_dir = ensure_dir(args.out_root / method_name / case_name)
    artifact_dir = ensure_dir(portal_dir / "artifacts")

    physics_iq_result = score_physics_iq(
        case_payload,
        source_video_path=source_video,
        threshold_value=int(args.physics_iq_threshold_value),
        downsample_factor=int(args.physics_iq_downsample_factor),
        aligned_video_dir=artifact_dir / "physics_iq",
    )
    physics_iq_without_context_result = score_physics_iq(
        case_payload,
        source_video_path=source_video,
        context_mode="without_context",
        context_frames=int(context_frames),
        threshold_value=int(args.physics_iq_threshold_value),
        downsample_factor=int(args.physics_iq_downsample_factor),
        aligned_video_dir=artifact_dir / "physics_iq_without_context",
    )
    pmf_with_context_result = score_pmf(
        case_payload,
        source_video_path=source_video,
        context_mode="with_context",
        context_frames=int(context_frames),
        device=str(args.pmf_device),
        aligned_video_dir=artifact_dir / "pmf_with_context",
    )
    pmf_without_context_result = score_pmf(
        case_payload,
        source_video_path=source_video,
        context_mode="without_context",
        context_frames=int(context_frames),
        device=str(args.pmf_device),
        aligned_video_dir=artifact_dir / "pmf_without_context",
    )

    portal_summary = {
        "case_json": str(case_json),
        "method": case_payload.get("method"),
        "output_video": str(output_video),
        "source_video": str(source_video),
        "context_video": str(context_video) if context_video else None,
        "context_frames": int(context_frames),
        "physics_iq": physics_iq_result,
        "physics_iq_with_context": physics_iq_result,
        "physics_iq_without_context": physics_iq_without_context_result,
        "pmf_with_context": pmf_with_context_result,
        "pmf_without_context": pmf_without_context_result,
        "input_only_metrics": {
            "wmreward": {
                "actual_metric_input": str(output_video),
                "notes": "Consumes generated output video only.",
            },
            "videophy2": {
                "actual_metric_input": str(output_video),
                "bench_default_task": "pc",
                "notes": "Consumes generated output video only for AAAinfer bench default task=pc.",
            },
            "cosmos_reason1": {
                "actual_metric_input": str(output_video),
                "notes": "Consumes generated output video only, with official prompt template.",
            },
        },
    }
    write_json(portal_dir / "portal_summary.json", portal_summary)
    (portal_dir / "index.html").write_text(
        render_index_html(
            portal_dir=portal_dir,
            case_payload=case_payload,
            source_video=source_video,
            context_video=context_video,
            context_frames=context_frames,
            output_video=output_video,
            physics_iq_result=physics_iq_result,
            physics_iq_without_context_result=physics_iq_without_context_result,
            pmf_with_context_result=pmf_with_context_result,
            pmf_without_context_result=pmf_without_context_result,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"portal_dir": str(portal_dir), "index_html": str(portal_dir / "index.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
