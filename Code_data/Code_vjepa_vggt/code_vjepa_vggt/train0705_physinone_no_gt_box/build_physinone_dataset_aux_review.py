from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from code_vjepa_vggt.inspect_cotracker_vggt_geometry import write_mp4
from code_vjepa_vggt.train0705_physinone_no_gt_box import (
    inspect_physinone_train_forward_aux_overlay as inspectmod,
)
from code_vjepa_vggt.train0705_physinone_no_gt_box import (
    train_stage1b_context_only_no_gt_box_v_newtrain_physinone as trainmod,
)


def _video_cthw_to_uint8_thwc(video_cthw: torch.Tensor) -> np.ndarray:
    video = video_cthw.detach().cpu().clamp(-1.0, 1.0)
    video = ((video + 1.0) * 127.5).to(torch.uint8)
    return video.permute(1, 2, 3, 0).contiguous().numpy()


def _move_optional_module(module: torch.nn.Module | None, device: torch.device) -> None:
    if module is not None:
        module.to(device)


def _build_storyboard(frames_thwc: np.ndarray, *, max_frames: int = 6, columns: int = 3) -> np.ndarray:
    total_frames = int(frames_thwc.shape[0])
    if total_frames <= 0:
        raise ValueError("frames_thwc must contain at least one frame")
    sample_count = min(total_frames, max_frames)
    frame_ids = np.linspace(0, total_frames - 1, sample_count).round().astype(np.int64)
    sampled = frames_thwc[frame_ids]
    rows = int(np.ceil(sample_count / max(columns, 1)))
    columns = max(columns, 1)
    height = int(sampled.shape[1])
    width = int(sampled.shape[2])
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for idx, frame in enumerate(sampled):
        row = idx // columns
        col = idx % columns
        y0 = row * height
        x0 = col * width
        canvas[y0 : y0 + height, x0 : x0 + width] = frame
    return canvas


def _build_case_page(output_dir: Path, result: dict[str, Any], sample_meta: dict[str, Any]) -> None:
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>PhysInOne Dataset + Aux Review</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #f6f4ee; color: #222; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; background: #fff; border: 1px solid #ddd; padding: 12px; }}
    video, img {{ width: 100%; border: 1px solid #ccc; background: #000; }}
    figcaption {{ font-size: 12px; color: #444; margin-top: 6px; }}
    pre {{ background: #fff; border: 1px solid #ddd; padding: 16px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>PhysInOne Dataset + Inspect Review</h1>
  <p><b>Caption:</b> {html.escape(str(result["caption"]))}</p>
  <p><b>Sample key:</b> {html.escape(str(result["sample_key"]))}</p>
  <p><b>Video:</b> {html.escape(str(result["video_path"]))}</p>
  <p><b>Context frames:</b> {html.escape(str(result["context_frame_indices"]))}</p>
  <div class="grid">
    <figure>
      <video controls preload="none" playsinline src="dataset_full.browser.mp4"></video>
      <figcaption>Dataset full training sample</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="dataset_context.browser.mp4"></video>
      <figcaption>Dataset context clip</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(result['prompt_preview_png'])}" />
      <figcaption>Viewer-grounding prompt boxes + sampled query points</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(result['input_overlay_video'])}"></video>
      <figcaption>Input pre-pipe overlay</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(result['box_overlay_png'])}" />
      <figcaption>Aux boxes storyboard</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(result['track_overlay_png'])}" />
      <figcaption>Aux tracks storyboard</figcaption>
    </figure>
  </div>
  <h2>Metrics</h2>
  <pre>{html.escape(json.dumps(result["metrics"], ensure_ascii=False, indent=2))}</pre>
  <h2>Sample Metadata</h2>
  <pre>{html.escape(json.dumps(sample_meta, ensure_ascii=False, indent=2))}</pre>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def _build_summary_page(cases: list[dict[str, Any]], output_dir: Path) -> None:
    sections: list[str] = []
    for case in cases:
        metrics_json = html.escape(json.dumps(case["metrics"], ensure_ascii=False, indent=2))
        sections.append(
            f"""
<section class="case-card">
  <div class="case-header">
    <div>
      <h2>{html.escape(case['case_name'])}</h2>
      <p class="meta"><b>sample_key:</b> {html.escape(case['sample_key'])}</p>
      <p class="meta"><b>camera:</b> {html.escape(case['camera_name'])} &nbsp; <b>physics_group:</b> {html.escape(case['physics_group'])}</p>
      <p class="meta"><b>scene:</b> {html.escape(case['scene_name'])}</p>
      <p class="caption">{html.escape(case['caption'])}</p>
    </div>
    <div class="actions">
      <a href="{html.escape(case['case_name'])}/index.html">open case report</a>
    </div>
  </div>
  <div class="media-grid">
    <figure>
      <video controls preload="none" playsinline src="{html.escape(case['case_name'])}/dataset_full.browser.mp4"></video>
      <figcaption>Dataset full sample</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(case['case_name'])}/dataset_context.browser.mp4"></video>
      <figcaption>Dataset context clip</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(case['case_name'])}/{html.escape(case['input_overlay_video'])}"></video>
      <figcaption>Input pre-pipe overlay</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(case['case_name'])}/{html.escape(case['box_overlay_png'])}" />
      <figcaption>Aux box storyboard</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(case['case_name'])}/{html.escape(case['track_overlay_png'])}" />
      <figcaption>Aux track storyboard</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(case['case_name'])}/{html.escape(case['prompt_preview_png'])}" />
      <figcaption>Prompt preview</figcaption>
    </figure>
  </div>
  <details>
    <summary>Metrics</summary>
    <pre>{metrics_json}</pre>
  </details>
</section>
"""
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>PhysInOne Dataset + Aux Review Gallery</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3efe6;
      --panel: #fffdf8;
      --line: #d9d0c2;
      --text: #1f1f1f;
      --muted: #5f5a53;
      --link: #0b5cad;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      font-family: sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top right, #efe6d3 0, transparent 24%),
        linear-gradient(180deg, #f6f1e7 0%, #f2ede3 100%);
    }}
    .page {{ max-width: 1800px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 32px; }}
    .intro {{ margin: 0 0 24px; color: var(--muted); line-height: 1.5; }}
    .case-list {{ display: grid; gap: 20px; }}
    .case-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
    }}
    .case-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 16px;
    }}
    .case-header h2 {{ margin: 0 0 8px; font-size: 22px; line-height: 1.25; }}
    .meta {{ margin: 4px 0; color: var(--muted); word-break: break-word; }}
    .caption {{ margin: 12px 0 0; line-height: 1.6; }}
    .actions a {{ color: var(--link); text-decoration: none; font-weight: 600; }}
    .actions a:hover {{ text-decoration: underline; }}
    .media-grid {{ display: grid; grid-template-columns: repeat(3, minmax(320px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: #ffffff; }}
    img, video {{ display: block; width: 100%; background: #000; }}
    figcaption {{ padding: 10px 12px; font-size: 13px; color: var(--muted); border-top: 1px solid var(--line); }}
    details {{ margin-top: 14px; border-top: 1px solid var(--line); padding-top: 14px; }}
    summary {{ cursor: pointer; font-weight: 600; }}
    pre {{ margin: 12px 0 0; padding: 14px; overflow-x: auto; border-radius: 10px; background: #faf7f0; border: 1px solid var(--line); white-space: pre-wrap; }}
    @media (max-width: 1300px) {{ .media-grid {{ grid-template-columns: 1fr; }} .case-header {{ flex-direction: column; }} }}
  </style>
</head>
<body>
  <div class="page">
    <h1>PhysInOne Dataset + Aux Review Gallery</h1>
    <p class="intro">
      Each case shows the sampled training video from PhysInOne plus the corresponding
      single-sample forward inspection outputs generated through inspect_physinone_train_forward_aux_overlay.py.
      Total cases: {len(cases)}.
    </p>
    <div class="case-list">
      {''.join(sections)}
    </div>
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = trainmod.build_parser()
    parser.set_defaults(
        wan_root="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B",
        dataset_type="phisinone_no_gt_box",
        phisinone_root="/data/gaoya/dataset/vLAR-PhysInOne/PhysInOneP01-PhysInOneP01",
        phisinone_split="train",
        phisinone_sampling_strategy="prefix",
        phisinone_cache_root="/data/gaoya/agent-data/cache/phisinone_no_gt_box_dataset",
        height=512,
        width=896,
        num_frames=24,
        fixed_num_context_frames=8,
        lora_base_model="dit",
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=32,
        lora_alpha=32,
        lora_checkpoint="/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors",
        extra_inputs="input_image",
        enable_object_branch=True,
        freeze_non_object_trainables=True,
        train_object_adapter=True,
        train_object_dit_branch=True,
        object_num_queries=8,
        aux_max_objects=4,
        jepa_ckpt_path="/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth",
        jepa_input_size=384,
        jepa_patch_size=16,
        jepa_tubelet_size=2,
        cotracker_checkpoint="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth",
        cotracker_input_h=384,
        cotracker_input_w=512,
        cotracker_window_len=60,
        vggt_model_path="/data/gaoya/ckpt/facebook-VGGT-1B",
        vggt_input_h=420,
        vggt_input_w=728,
        object_pooler_latent_dim=16,
        cond_proj_dim=4096,
        jepa_window_radius=1,
        latent_window_radius=1,
        object_gate_init=0.1,
        lambda_main=1.0,
        lambda_track_aux=0.0,
        lambda_box_aux=0.0,
        lambda_depth_aux=0.0,
        stage1a_init_from="/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt",
        grounding_proposal_source="gdino_only",
        grounding_motion_score_ratio=0.15,
        grounding_text_prompt="box . cube . block . cylinder . capsule . sphere . ball .",
        grounding_disable_caption_terms=True,
        grounding_gdino_box_threshold=0.20,
        grounding_gdino_text_threshold=0.15,
        grounding_prompt_frame_mode="first",
        grounding_track_dedupe_iou_threshold=0.75,
        grounding_container_suppress_ratio_threshold=0.95,
        grounding_container_suppress_min_contained=2,
        grounding_container_suppress_min_area_ratio=1.5,
        grounding_container_suppress_small_iou_threshold=0.7,
        sam2_segment_len=8,
        stage2_resume_from="/data/gaoya/agent-data/checkpoints/stage1b_physinone_no_gt_box_smoke_gpu3567/checkpoints/step-000001",
    )
    parser.add_argument("--review_count", type=int, default=6)
    parser.add_argument("--review_seed", type=int, default=42)
    parser.add_argument(
        "--review_output_dir",
        type=str,
        default="/data/gaoya/agent-data/outputs/phisinone_dataset_aux_review",
    )
    parser.add_argument("--review_fps", type=int, default=30)
    return parser


def main() -> None:
    args = trainmod.tvn.prepare_args(build_parser().parse_args())
    output_dir = Path(args.review_output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = trainmod.build_dataset(args)
    if len(dataset) < int(args.review_count):
        raise RuntimeError(
            f"dataset has only {len(dataset)} samples, smaller than requested review_count={args.review_count}"
        )

    indices = list(range(len(dataset)))
    rng = random.Random(int(args.review_seed))
    rng.shuffle(indices)
    selected_indices = indices[: int(args.review_count)]

    accelerator = SimpleNamespace(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = trainmod.build_model(args, accelerator)
    target_device = torch.device(model.pipe.device)
    _move_optional_module(model.object_pooler, target_device)
    _move_optional_module(model.object_aux_heads, target_device)
    _move_optional_module(model.object_adapter, target_device)
    _move_optional_module(model.vggt_adapter, target_device)

    if args.stage1a_init_from is not None:
        trainmod.tvn._load_filtered_checkpoint_into_model(
            model,
            args.stage1a_init_from,
            include_prefixes=("object_pooler.", "object_aux_heads."),
        )
    inspectmod._load_optional_stage2_weights(model, args.stage2_resume_from)

    torch.nn.Module.train(model, False)

    case_manifest: list[dict[str, Any]] = []
    for rank, dataset_index in enumerate(selected_indices):
        sample = dataset[int(dataset_index)]
        sample_name = str(sample.get("sample_name") or sample.get("metadata", {}).get("sample_name") or f"sample_{dataset_index:06d}")
        case_name = f"{rank:02d}_{sample_name}"
        case_dir = output_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        full_video_raw = case_dir / "dataset_full.mp4"
        context_video_raw = case_dir / "dataset_context.mp4"
        write_mp4(full_video_raw, _video_cthw_to_uint8_thwc(sample["video"]), fps=int(args.review_fps))
        write_mp4(context_video_raw, _video_cthw_to_uint8_thwc(sample["context_video"]), fps=int(args.review_fps))
        full_video_browser = inspectmod._ensure_browser_video(full_video_raw)
        context_video_browser = inspectmod._ensure_browser_video(context_video_raw)

        inputs_shared, inputs_posi, _ = inspectmod._prepare_forward_inputs(model, sample)
        with torch.no_grad():
            debug = inspectmod._run_forward_debug(
                model,
                sample,
                inputs_shared=inputs_shared,
                inputs_posi=inputs_posi,
            )

        context_video = sample["context_video"]
        image_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))
        grounding_sample = model.viewer_grounding.build_sample(
            frames_tchw_01=((context_video.permute(1, 0, 2, 3).float() + 1.0) / 2.0).cpu().numpy(),
            caption=str(sample["caption"]),
            image_hw=image_hw,
        )

        valid_queries = inspectmod._valid_query_count(debug["object_valid_mask"], model.object_num_queries)
        valid_queries = max(valid_queries, 0)
        query_points_prior = debug["query_points_prior"][0].detach().float().cpu().numpy()
        cotracker_tracks = debug["cotracker_out"].tracks[0].detach().float().cpu().numpy()
        cotracker_visibility = debug["cotracker_out"].visibility[0].detach().float().cpu().numpy()
        valid_queries_px = query_points_prior[:valid_queries]
        valid_tracks = cotracker_tracks[:, :valid_queries]
        valid_visibility = cotracker_visibility[:, :valid_queries]
        query_owner = [
            obj_idx
            for obj_idx in range(int((debug["object_valid_mask"][0] > 0.5).sum().item()))
            for _ in range(int(model.object_num_queries))
        ]

        prompt_preview = inspectmod._render_prompt_preview(
            context_video=context_video,
            grounding_sample=grounding_sample,
            valid_queries_px=valid_queries_px,
            query_owner=query_owner,
        )
        prompt_preview_path = case_dir / "prompt_preview.png"
        inspectmod._write_rgb_png(prompt_preview_path, prompt_preview)

        input_overlay_video = inspectmod.render_track_overlay(
            context_video=context_video,
            object_tracks=getattr(grounding_sample, "object_tracks", []),
            prompt_frame_idx=int(getattr(grounding_sample, "prompt_frame_idx", 0)),
            query_points_px_k2=valid_queries_px.astype(np.float32),
            query_owner=query_owner,
            tracks_tk2=valid_tracks.astype(np.float32),
            visibility_tk=valid_visibility.astype(np.float32),
            color_rgb=inspectmod.INPUT_TRACK_COLOR,
            prefix="trk",
        )
        input_overlay_raw = case_dir / "input_prepipe_overlay.mp4"
        inspectmod.write_mp4(input_overlay_raw, input_overlay_video, fps=int(args.review_fps))
        input_overlay_browser = inspectmod._ensure_browser_video(input_overlay_raw)

        ref_box_xyxy = debug["object_out"].active_box_xyxy[0].detach().float().cpu().numpy()
        pred_box_xyxy = debug["object_aux_out"].pred_box_xyxy[0].detach().float().cpu().numpy()
        ref_track_summary = debug["object_out"].active_track_summary[0, ..., :4].detach().float().cpu().numpy()
        pred_track_summary = debug["object_aux_out"].pred_track_summary[0].detach().float().cpu().numpy()
        latent_valid_mask = debug["latent_valid_mask"]

        box_overlay_video = inspectmod._render_ref_pred_box_overlay(
            context_video=context_video,
            ref_box_xyxy=ref_box_xyxy,
            pred_box_xyxy=pred_box_xyxy,
            valid_mask=latent_valid_mask,
            image_hw=image_hw,
        )
        box_overlay_storyboard = _build_storyboard(box_overlay_video)
        box_overlay_png = case_dir / "aux_pred_box_overlay.png"
        inspectmod._write_rgb_png(box_overlay_png, box_overlay_storyboard)

        track_overlay_video = inspectmod._render_ref_pred_track_overlay(
            context_video=context_video,
            ref_track_summary=ref_track_summary,
            pred_track_summary=pred_track_summary,
            valid_mask=latent_valid_mask,
            image_hw=image_hw,
        )
        track_overlay_storyboard = _build_storyboard(track_overlay_video)
        track_overlay_png = case_dir / "aux_pred_track_overlay.png"
        inspectmod._write_rgb_png(track_overlay_png, track_overlay_storyboard)

        result = {
            "caption": str(sample["caption"]),
            "video_path": str(sample["video_path"]),
            "sample_key": str(sample.get("metadata", {}).get("sample_key", "")),
            "context_frame_indices": sample["context_frame_indices"].tolist(),
            "prepipe_gallery_url": "",
            "prompt_preview_png": str(prompt_preview_path.name),
            "input_overlay_video": str(input_overlay_browser.name),
            "box_overlay_png": str(box_overlay_png.name),
            "track_overlay_png": str(track_overlay_png.name),
            "metrics": debug["metrics"],
        }
        (case_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        sample_meta = {
            "dataset_index": int(dataset_index),
            "sample_name": sample_name,
            "video_path": str(sample["video_path"]),
            "caption": str(sample["caption"]),
            "metadata": sample.get("metadata", {}),
        }
        (case_dir / "sample_metadata.json").write_text(
            json.dumps(sample_meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _build_case_page(case_dir, result, sample_meta)

        case_manifest.append(
            {
                "case_name": case_name,
                "dataset_index": int(dataset_index),
                "sample_key": result["sample_key"],
                "caption": result["caption"],
                "camera_name": str(sample.get("metadata", {}).get("camera_name", "")),
                "physics_group": str(sample.get("metadata", {}).get("physics_group", "")),
                "scene_name": str(sample.get("metadata", {}).get("scene_name", "")),
                "dataset_full_video": full_video_browser.name,
                "dataset_context_video": context_video_browser.name,
                "prompt_preview_png": result["prompt_preview_png"],
                "input_overlay_video": result["input_overlay_video"],
                "box_overlay_png": result["box_overlay_png"],
                "track_overlay_png": result["track_overlay_png"],
                "metrics": result["metrics"],
            }
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "review_count": int(args.review_count),
                "review_seed": int(args.review_seed),
                "selected_indices": [int(i) for i in selected_indices],
                "cases": case_manifest,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    _build_summary_page(case_manifest, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "index_html": str(output_dir / "index.html"),
                "manifest": str(output_dir / "manifest.json"),
                "case_count": len(case_manifest),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
