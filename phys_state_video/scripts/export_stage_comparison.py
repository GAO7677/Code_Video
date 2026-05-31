from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.adapter import TinyVideoBackbone, adapter_loss
from phys_state_video.conditioning import build_condition_bundle
from phys_state_video.config import AdapterConfig, ConditioningConfig
from phys_state_video.dataset import NpzEpisodeDataset, collate_episodes
from phys_state_video.experiment import apply_condition_mode, compute_state_metrics, perturb_condition_bundle
from phys_state_video.proxy_state import extract_primary_track
from phys_state_video.utils import require_torch

torch = require_torch()


MODEL_SPECS = [
    {
        "id": "baseline_none",
        "label": "Baseline None",
        "checkpoint": "/data/gaoya/AAA_test_video/0529/phys_state_video/runs/openvid_phase1_merged0531_opt/adapter_baseline_opt0531_gpu23.pt",
    },
    {
        "id": "state_opt",
        "label": "State Opt",
        "checkpoint": "/data/gaoya/AAA_test_video/0529/phys_state_video/runs/openvid_phase1_merged0531_opt/adapter_state_opt0531_gpu01.pt",
    },
    {
        "id": "temporal",
        "label": "Temporal",
        "checkpoint": "/data/gaoya/AAA_test_video/0529/phys_state_video/runs/openvid_phase1_temporal0531/adapter_state_temporal_bs128_resume0531_gpu0123.best.pt",
    },
    {
        "id": "spaux025",
        "label": "Spatial Aux 0.25",
        "checkpoint": "/data/gaoya/AAA_test_video/0529/phys_state_video/runs/openvid_phase1_spatialaux0531/adapter_state_spaux025_bs128_gpu01.best.pt",
    },
    {
        "id": "ctxprompt_bias",
        "label": "Ctx+Prompt Bias",
        "checkpoint": "/data/gaoya/AAA_test_video/0529/phys_state_video/runs/openvid_phase1_ctxprompt0531/adapter_state_ctxprompt_spaux025_true_lr5e4_gpu01.best.pt",
    },
    {
        "id": "ctxprompt_tokens",
        "label": "Ctx+Prompt Tokens",
        "checkpoint": "/data/gaoya/AAA_test_video/0529/phys_state_video/runs/openvid_phase1_ctxprompt0531/adapter_state_ctxprompt_tokens_typed_spaux025_lr5e4_gpu01.best.pt",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Export stage-wise adapter comparison assets.")
    parser.add_argument("--data", required=True, help="Validation episode directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for html/assets/json.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--max-cases", type=int, default=12)
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, map_location):
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def load_model_state(module, state_dict, checkpoint_label: str) -> dict[str, list[str]]:
    try:
        module.load_state_dict(state_dict)
        return {"missing": [], "unexpected": []}
    except RuntimeError as exc:
        message = str(exc)
        key_mismatch = "Missing key(s) in state_dict" in message or "Unexpected key(s) in state_dict" in message
        if not key_mismatch:
            raise
        incompatible = module.load_state_dict(state_dict, strict=False)
        return {
            "missing": list(incompatible.missing_keys),
            "unexpected": list(incompatible.unexpected_keys),
        }


def to_uint8_rgb(frame_chw: np.ndarray) -> np.ndarray:
    image = np.transpose(np.clip(frame_chw, 0.0, 1.0), (1, 2, 0))
    return (image * 255.0).round().astype(np.uint8)


def write_mp4(path: Path, frames_tchw: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t, _, height, width = frames_tchw.shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer for {path}")
    for idx in range(t):
        rgb = to_uint8_rgb(frames_tchw[idx])
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def make_strip(frames_tchw: np.ndarray) -> np.ndarray:
    tiles = [to_uint8_rgb(frame) for frame in frames_tchw]
    return np.concatenate(tiles, axis=1)


def save_png(path: Path, rgb_image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))


def normalize_map(channel_thw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    lo = float(channel_thw.min())
    hi = float(channel_thw.max())
    if hi - lo < eps:
        return np.zeros_like(channel_thw, dtype=np.float32)
    return (channel_thw - lo) / (hi - lo)


def draw_text(rgb: np.ndarray, text: str) -> np.ndarray:
    canvas = rgb.copy()
    cv2.putText(
        canvas,
        text,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        text,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return canvas


def gray_to_rgb(gray_thw: np.ndarray, label: str) -> np.ndarray:
    frames = []
    for frame in gray_thw:
        rgb = np.repeat((np.clip(frame, 0.0, 1.0) * 255.0).round().astype(np.uint8)[..., None], 3, axis=2)
        frames.append(draw_text(rgb, label))
    return np.stack(frames, axis=0)


def build_condition_video(cond_maps: np.ndarray) -> np.ndarray:
    heat = cond_maps[:, 0]
    bbox = cond_maps[:, 1]
    depth = normalize_map(cond_maps[:, 2])
    vis = cond_maps[:, 3]
    heat_rgb = gray_to_rgb(heat, "target heatmap")
    bbox_rgb = gray_to_rgb(bbox, "target bbox")
    depth_rgb = gray_to_rgb(depth, "target depth")
    vis_rgb = gray_to_rgb(vis, "target vis")
    rows = []
    for idx in range(cond_maps.shape[0]):
        top = np.concatenate([heat_rgb[idx], bbox_rgb[idx]], axis=1)
        bottom = np.concatenate([depth_rgb[idx], vis_rgb[idx]], axis=1)
        rows.append(np.concatenate([top, bottom], axis=0))
    return np.stack(rows, axis=0)


def build_spatial_video(spatial_logits: np.ndarray, cond_maps: np.ndarray) -> np.ndarray:
    spatial_prob = 1.0 / (1.0 + np.exp(-np.clip(spatial_logits, -20.0, 20.0)))
    pred_heat = spatial_prob[:, 0]
    pred_box = spatial_prob[:, 1]
    tgt_heat = cond_maps[:, 0]
    tgt_box = cond_maps[:, 1]
    pred_heat_rgb = gray_to_rgb(pred_heat, "pred heatmap")
    pred_box_rgb = gray_to_rgb(pred_box, "pred bbox")
    tgt_heat_rgb = gray_to_rgb(tgt_heat, "target heatmap")
    tgt_box_rgb = gray_to_rgb(tgt_box, "target bbox")
    rows = []
    for idx in range(spatial_logits.shape[0]):
        top = np.concatenate([pred_heat_rgb[idx], pred_box_rgb[idx]], axis=1)
        bottom = np.concatenate([tgt_heat_rgb[idx], tgt_box_rgb[idx]], axis=1)
        rows.append(np.concatenate([top, bottom], axis=0))
    return np.stack(rows, axis=0)


def render_html(report: dict) -> str:
    summary_rows = []
    for item in report["model_summaries"]:
        correct = item["correct"]
        perturbed = item["perturbed"]
        summary_rows.append(
            f"""
            <tr>
              <td>{item['label']}</td>
              <td>{correct['loss']:.4f}</td>
              <td>{correct['recon']:.4f}</td>
              <td>{correct['center_error']:.4f}</td>
              <td>{correct['log_scale_error']:.4f}</td>
              <td>{perturbed['center_error']:.4f}</td>
              <td>{perturbed['log_scale_error']:.4f}</td>
            </tr>
            """
        )

    case_blocks = []
    for case in report["cases"]:
        model_cards = []
        for model in case["models"]:
            spatial_html = ""
            if model["show_spatial"]:
                spatial_html = f'<video controls preload="metadata" src="{model["spatial_video"]}"></video>'
            model_cards.append(
                f"""
                <div class="model-card">
                  <div class="model-name">{model['label']}</div>
                  <video controls preload="metadata" src="{model['video']}"></video>
                  {spatial_html}
                  <div class="metrics">
                    <span>loss {model['loss']:.3f}</span>
                    <span>recon {model['recon']:.3f}</span>
                    <span>center {model['center_error']:.3f}</span>
                    <span>scale {model['log_scale_error']:.3f}</span>
                  </div>
                </div>
                """
            )
        case_blocks.append(
            f"""
            <section class="case-card">
              <h2>{case['case_id']}</h2>
              <div class="prompt">{case['prompt']}</div>
              <div class="ref-row">
                <div class="ref-card">
                  <div class="ref-name">Context Strip</div>
                  <img src="{case['context_strip']}" />
                </div>
                <div class="ref-card">
                  <div class="ref-name">GT Future</div>
                  <video controls preload="metadata" src="{case['gt_video']}"></video>
                </div>
                <div class="ref-card">
                  <div class="ref-name">Target Conditions</div>
                  <video controls preload="metadata" src="{case['condition_video']}"></video>
                </div>
              </div>
              <div class="model-grid">
                {''.join(model_cards)}
              </div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phys-State Video Stage Compare</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --ink: #1b1b18;
      --muted: #6d685d;
      --panel: #fffdf8;
      --line: #d7cfbf;
      --accent: #174f44;
      --accent-2: #ab5f2c;
    }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #f7f2e9 0%, #efe6d7 100%);
      color: var(--ink);
      font-family: "Source Han Sans SC", "Noto Sans SC", sans-serif;
    }}
    .wrap {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      letter-spacing: 0.02em;
    }}
    .lead {{
      color: var(--muted);
      max-width: 980px;
      line-height: 1.6;
      margin-bottom: 20px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: rgba(255,255,255,0.72);
      backdrop-filter: blur(8px);
      border: 1px solid var(--line);
      margin-bottom: 28px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      background: #f1e7d5;
    }}
    .case-card {{
      background: rgba(255,253,248,0.84);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      margin-bottom: 24px;
      box-shadow: 0 18px 50px rgba(58, 44, 20, 0.08);
    }}
    .prompt {{
      color: var(--muted);
      margin-bottom: 14px;
      white-space: pre-wrap;
    }}
    .ref-row, .model-grid {{
      display: grid;
      gap: 14px;
    }}
    .ref-row {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin-bottom: 18px;
    }}
    .model-grid {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    .ref-card, .model-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
    }}
    .ref-name, .model-name {{
      font-weight: 700;
      margin-bottom: 10px;
      color: var(--accent);
    }}
    img, video {{
      width: 100%;
      display: block;
      border-radius: 10px;
      background: #000;
      margin-bottom: 10px;
    }}
    .metrics {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      font-size: 12px;
      color: var(--accent-2);
    }}
    .metrics span {{
      background: #f3eadc;
      border-radius: 999px;
      padding: 4px 8px;
    }}
    @media (max-width: 1100px) {{
      .ref-row, .model-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Phys-State Video 阶段对比总览</h1>
    <div class="lead">
      这页把 baseline、原始 state、temporal、spatial-aux、ctxprompt bias、ctxprompt token 等阶段模型放到同一批验证样本上横向比较。
      上面是定量摘要，下面每个 case 都给出 context、GT future、target condition，以及各模型的生成结果与部分中间 spatial map。
    </div>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Correct Loss</th>
          <th>Correct Recon</th>
          <th>Correct Center</th>
          <th>Correct Scale</th>
          <th>Perturbed Center</th>
          <th>Perturbed Scale</th>
        </tr>
      </thead>
      <tbody>
        {''.join(summary_rows)}
      </tbody>
    </table>
    {''.join(case_blocks)}
  </div>
</body>
</html>"""


def main():
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    dataset = NpzEpisodeDataset(args.data)
    case_count = min(args.max_cases, len(dataset))
    episode_records = [(dataset.files[idx].stem, dataset[idx]) for idx in range(case_count)]
    episodes = [episode for _, episode in episode_records]
    cond_cfg = ConditioningConfig(
        frame_height=episodes[0].context_frames.shape[-2],
        frame_width=episodes[0].context_frames.shape[-1],
    )

    models = []
    for spec in MODEL_SPECS:
        ckpt = load_checkpoint(spec["checkpoint"], map_location=args.device)
        adapter_cfg = AdapterConfig(**ckpt["config"])
        model = TinyVideoBackbone(adapter_cfg).to(args.device)
        load_info = load_model_state(model, ckpt["model"], spec["checkpoint"])
        model.eval()
        models.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "checkpoint": spec["checkpoint"],
                "condition_mode": ckpt.get("condition_mode", "state"),
                "state_loss_weights": ckpt.get("state_loss_weights"),
                "state_loss_scale": float(ckpt.get("state_loss_scale", 0.1)),
                "spatial_loss_scale": float(ckpt.get("spatial_loss_scale", 0.0)),
                "spatial_foreground_weight": float(ckpt.get("spatial_foreground_weight", 4.0)),
                "load_info": load_info,
                "model": model,
            }
        )

    model_summaries = []
    report_cases = []

    for model_info in models:
        totals_correct = {"loss": 0.0, "recon": 0.0, "state_aux": 0.0, "spatial_aux": 0.0, "center_error": 0.0, "log_scale_error": 0.0, "visibility_error": 0.0}
        totals_perturbed = {"loss": 0.0, "recon": 0.0, "state_aux": 0.0, "spatial_aux": 0.0, "center_error": 0.0, "log_scale_error": 0.0, "visibility_error": 0.0}
        model_info["per_case"] = {}

        with torch.no_grad():
            for file_stem, episode in episode_records:
                batch = collate_episodes([episode])
                future_states = batch["future_states"].to(args.device)
                future_boxes = batch["future_boxes"].to(args.device)
                appearance = batch["appearance"].to(args.device)
                bundle = build_condition_bundle(future_states, future_boxes, appearance, cond_cfg)
                bundle = apply_condition_mode(bundle, model_info["condition_mode"])

                outputs = model_info["model"](
                    batch["context_frames"].to(args.device),
                    bundle.maps,
                    bundle.memory_tokens,
                    context_states=batch["context_states"].to(args.device),
                    prompt_token_ids=batch["prompt_token_ids"].to(args.device),
                    prompt_token_mask=batch["prompt_token_mask"].to(args.device),
                )
                target_spatial_maps = bundle.maps[:, :, 0:2]
                state_loss_weight_tensor = None
                if model_info["state_loss_weights"] is not None:
                    state_loss_weight_tensor = torch.tensor(model_info["state_loss_weights"], dtype=torch.float32, device=args.device)
                losses = adapter_loss(
                    outputs["frames"],
                    batch["future_frames"].to(args.device),
                    outputs["state_logits"],
                    future_states,
                    state_loss_weights=state_loss_weight_tensor,
                    state_loss_scale=model_info["state_loss_scale"],
                    predicted_spatial_logits=outputs.get("spatial_logits"),
                    target_spatial_maps=target_spatial_maps,
                    spatial_loss_scale=model_info["spatial_loss_scale"],
                    spatial_foreground_weight=model_info["spatial_foreground_weight"],
                )

                generated_np = outputs["frames"][0].detach().cpu().numpy()
                proxy = extract_primary_track(generated_np)
                target_states_np = batch["future_states"][0].detach().cpu().numpy()
                state_metrics = compute_state_metrics(proxy.states, target_states_np)
                for key in ("loss", "recon", "state_aux", "spatial_aux"):
                    totals_correct[key] += float(losses[key].detach().cpu())
                for key, value in state_metrics.items():
                    totals_correct[key] += float(value)

                perturbed_bundle = perturb_condition_bundle(bundle)
                perturbed_outputs = model_info["model"](
                    batch["context_frames"].to(args.device),
                    perturbed_bundle.maps,
                    perturbed_bundle.memory_tokens,
                    context_states=batch["context_states"].to(args.device),
                    prompt_token_ids=batch["prompt_token_ids"].to(args.device),
                    prompt_token_mask=batch["prompt_token_mask"].to(args.device),
                )
                perturbed_losses = adapter_loss(
                    perturbed_outputs["frames"],
                    batch["future_frames"].to(args.device),
                    perturbed_outputs["state_logits"],
                    future_states,
                    state_loss_weights=state_loss_weight_tensor,
                    state_loss_scale=model_info["state_loss_scale"],
                    predicted_spatial_logits=perturbed_outputs.get("spatial_logits"),
                    target_spatial_maps=perturbed_bundle.maps[:, :, 0:2],
                    spatial_loss_scale=model_info["spatial_loss_scale"],
                    spatial_foreground_weight=model_info["spatial_foreground_weight"],
                )
                perturbed_generated_np = perturbed_outputs["frames"][0].detach().cpu().numpy()
                perturbed_proxy = extract_primary_track(perturbed_generated_np)
                perturbed_metrics = compute_state_metrics(perturbed_proxy.states, target_states_np)
                for key in ("loss", "recon", "state_aux", "spatial_aux"):
                    totals_perturbed[key] += float(perturbed_losses[key].detach().cpu())
                for key, value in perturbed_metrics.items():
                    totals_perturbed[key] += float(value)

                model_info["per_case"][file_stem] = {
                    "loss": float(losses["loss"].detach().cpu()),
                    "recon": float(losses["recon"].detach().cpu()),
                    "state_aux": float(losses["state_aux"].detach().cpu()),
                    "spatial_aux": float(losses["spatial_aux"].detach().cpu()),
                    "center_error": float(state_metrics["center_error"]),
                    "log_scale_error": float(state_metrics["log_scale_error"]),
                    "visibility_error": float(state_metrics["visibility_error"]),
                    "generated": generated_np,
                    "spatial_logits": outputs["spatial_logits"][0].detach().cpu().numpy(),
                }

        denom = float(case_count)
        model_summaries.append(
            {
                "id": model_info["id"],
                "label": model_info["label"],
                "checkpoint": model_info["checkpoint"],
                "condition_mode": model_info["condition_mode"],
                "spatial_loss_scale": model_info["spatial_loss_scale"],
                "load_info": model_info["load_info"],
                "correct": {key: value / denom for key, value in totals_correct.items()},
                "perturbed": {key: value / denom for key, value in totals_perturbed.items()},
            }
        )

    for file_stem, episode in episode_records:
        batch = collate_episodes([episode])
        cond_bundle = build_condition_bundle(
            batch["future_states"].to(args.device),
            batch["future_boxes"].to(args.device),
            batch["appearance"].to(args.device),
            cond_cfg,
        )
        cond_video = build_condition_video(cond_bundle.maps[0].detach().cpu().numpy())

        case_dir = assets_dir / file_stem
        case_dir.mkdir(parents=True, exist_ok=True)
        context_strip_path = case_dir / "context_strip.png"
        gt_video_path = case_dir / "gt_future.mp4"
        cond_video_path = case_dir / "target_conditions.mp4"
        save_png(context_strip_path, make_strip(batch["context_frames"][0].numpy()))
        write_mp4(gt_video_path, batch["future_frames"][0].numpy(), args.fps)
        write_mp4(cond_video_path, np.transpose(cond_video, (0, 3, 1, 2)).astype(np.float32) / 255.0, args.fps)

        model_entries = []
        for model_info, summary in zip(models, model_summaries):
            per_case = model_info["per_case"][file_stem]
            video_rel = f"assets/{file_stem}/{model_info['id']}_generated.mp4"
            spatial_rel = f"assets/{file_stem}/{model_info['id']}_spatial.mp4"
            write_mp4(output_dir / video_rel, per_case["generated"], args.fps)
            show_spatial = summary["spatial_loss_scale"] > 0.0
            if show_spatial:
                spatial_video = build_spatial_video(per_case["spatial_logits"], cond_bundle.maps[0].detach().cpu().numpy())
                write_mp4(output_dir / spatial_rel, np.transpose(spatial_video, (0, 3, 1, 2)).astype(np.float32) / 255.0, args.fps)
            model_entries.append(
                {
                    "id": model_info["id"],
                    "label": model_info["label"],
                    "video": video_rel,
                    "spatial_video": spatial_rel,
                    "show_spatial": show_spatial,
                    "loss": per_case["loss"],
                    "recon": per_case["recon"],
                    "center_error": per_case["center_error"],
                    "log_scale_error": per_case["log_scale_error"],
                    "visibility_error": per_case["visibility_error"],
                }
            )

        report_cases.append(
            {
                "case_id": file_stem,
                "prompt": batch["prompts"][0],
                "context_strip": f"assets/{file_stem}/context_strip.png",
                "gt_video": f"assets/{file_stem}/gt_future.mp4",
                "condition_video": f"assets/{file_stem}/target_conditions.mp4",
                "models": model_entries,
            }
        )

    report = {
        "data": args.data,
        "output_dir": str(output_dir),
        "model_summaries": model_summaries,
        "cases": report_cases,
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(render_html(report), encoding="utf-8")
    print(f"exported stage comparison to {output_dir}")


if __name__ == "__main__":
    main()
