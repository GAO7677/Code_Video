import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from decord import VideoReader, cpu

from src.masks.multiseq_multiblock3d import _MaskGenerator


IMAGENET_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODEL_EMBED_DIMS = {
    "vit_base": 768,
    "vit_large": 1024,
    "vit_huge": 1280,
    "vit_giant_xformers": 1408,
    "vit_gigantic_xformers": 1664,
}


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def sample_frame_indices(frame_count: int, num_frames: int) -> np.ndarray:
    idx = np.linspace(0, frame_count - 1, num_frames)
    return np.round(idx).astype(np.int64)


def read_video(video_path: str, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    vr = VideoReader(video_path, ctx=cpu(0))
    frame_idx = sample_frame_indices(len(vr), num_frames)
    video = vr.get_batch(frame_idx).asnumpy()
    return video, frame_idx


def preprocess_frames(video_thwc: np.ndarray, crop_size: int) -> dict[str, Any]:
    x = torch.from_numpy(video_thwc).permute(0, 3, 1, 2).contiguous()

    records = [
        {
            "name": "sampled_frames",
            "shape": list(video_thwc.shape),
            "dtype": str(video_thwc.dtype),
            "zh": "等间隔采样后的原始视频帧，格式是 [T, H, W, C]。",
        },
        {
            "name": "tchw",
            "shape": list(x.shape),
            "dtype": str(x.dtype).replace("torch.", ""),
            "zh": "转成 PyTorch 视频张量 [T, C, H, W]。",
        },
    ]

    short_side = int(256.0 / 224 * crop_size)
    resized = []
    for frame in x:
        h, w = frame.shape[-2:]
        if h < w:
            new_h = short_side
            new_w = int(round(w * short_side / h))
        else:
            new_w = short_side
            new_h = int(round(h * short_side / w))
        frame_f = frame.unsqueeze(0).float()
        frame_f = torch.nn.functional.interpolate(
            frame_f, size=(new_h, new_w), mode="bilinear", align_corners=False
        ).squeeze(0)
        resized.append(frame_f)
    x = torch.stack(resized, dim=0)
    records.append(
        {
            "name": "resize_short_side",
            "shape": list(x.shape),
            "dtype": str(x.dtype).replace("torch.", ""),
            "zh": f"短边缩放到 {short_side}，保持宽高比。",
        }
    )

    h, w = x.shape[-2:]
    top = max((h - crop_size) // 2, 0)
    left = max((w - crop_size) // 2, 0)
    x = x[:, :, top : top + crop_size, left : left + crop_size]
    records.append(
        {
            "name": "center_crop",
            "shape": list(x.shape),
            "dtype": str(x.dtype).replace("torch.", ""),
            "zh": f"中心裁剪成 {crop_size}x{crop_size}。",
        }
    )

    cropped_uint8 = x.permute(0, 2, 3, 1).clamp(0, 255).byte().numpy()

    x = x / 255.0
    records.append(
        {
            "name": "rescale_0_1",
            "shape": list(x.shape),
            "dtype": str(x.dtype).replace("torch.", ""),
            "zh": "像素缩放到 0-1。",
        }
    )

    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    records.append(
        {
            "name": "normalize",
            "shape": list(x.shape),
            "dtype": str(x.dtype).replace("torch.", ""),
            "zh": "按 ImageNet mean/std 归一化。",
        }
    )

    x = x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    records.append(
        {
            "name": "to_bcthw",
            "shape": list(x.shape),
            "dtype": str(x.dtype).replace("torch.", ""),
            "zh": "整理成训练时输入 encoder 的 [B, C, T, H, W]。",
        }
    )

    return {"tensor": x, "records": records, "cropped_uint8": cropped_uint8}


def build_mask_generators(cfg: dict[str, Any]) -> list[_MaskGenerator]:
    data_cfg = cfg["data"]
    return [
        _MaskGenerator(
            crop_size=data_cfg["crop_size"],
            num_frames=data_cfg["dataset_fpcs"][0],
            spatial_patch_size=data_cfg["patch_size"],
            temporal_patch_size=data_cfg["tubelet_size"],
            spatial_pred_mask_scale=m.get("spatial_scale"),
            temporal_pred_mask_scale=m.get("temporal_scale"),
            aspect_ratio=m.get("aspect_ratio"),
            npred=m.get("num_blocks"),
            max_context_frames_ratio=m.get("max_temporal_keep", 1.0),
            max_keep=m.get("max_keep", None),
            full_complement=m.get("full_complement", False),
            pred_full_complement=m.get("pred_full_complement", False),
            inv_block=m.get("inv_block", False),
        )
        for m in cfg["mask"]
    ]


def mask_indices_to_grid(mask_pred: torch.Tensor, t_tokens: int, h_tokens: int, w_tokens: int) -> np.ndarray:
    grid = np.zeros((t_tokens, h_tokens, w_tokens), dtype=np.uint8)
    indices = mask_pred.squeeze(0).cpu().numpy().astype(np.int64)
    for idx in indices:
        t = idx // (h_tokens * w_tokens)
        rem = idx % (h_tokens * w_tokens)
        h = rem // w_tokens
        w = rem % w_tokens
        grid[t, h, w] = 1
    return grid


def overlay_mask_on_frame(frame_hwc: np.ndarray, mask_grid_2d: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    h, w = frame_hwc.shape[:2]
    mask_img = cv2.resize(mask_grid_2d.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    base = frame_hwc.astype(np.float32).copy()
    color = np.zeros_like(base)
    color[:, :, 0] = 245
    color[:, :, 1] = 99
    color[:, :, 2] = 56
    mask_bool = mask_img > 0
    base[mask_bool] = (1.0 - alpha) * base[mask_bool] + alpha * color[mask_bool]

    # draw patch grid
    patch_h = h // mask_grid_2d.shape[0]
    patch_w = w // mask_grid_2d.shape[1]
    for y in range(0, h, patch_h):
        cv2.line(base, (0, y), (w, y), (255, 255, 255), 1)
    for x in range(0, w, patch_w):
        cv2.line(base, (x, 0), (x, h), (255, 255, 255), 1)
    return np.clip(base, 0, 255).astype(np.uint8)


def save_mask_assets(
    cropped_frames: np.ndarray,
    frame_indices: np.ndarray,
    mask_specs: list[dict[str, Any]],
    t_tokens: int,
    h_tokens: int,
    w_tokens: int,
    out_dir: Path,
) -> None:
    overlay_dir = out_dir / "mask_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    preview_positions = [0, len(cropped_frames) // 3, 2 * len(cropped_frames) // 3, len(cropped_frames) - 1]

    for spec in mask_specs:
        mask_grid = spec["mask_grid"]
        video_path = overlay_dir / f"{spec['mask_name']}_overlay.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            8.0,
            (cropped_frames.shape[2], cropped_frames.shape[1]),
        )
        for f in range(len(cropped_frames)):
            t = min(f // 2, t_tokens - 1)
            overlay = overlay_mask_on_frame(cropped_frames[f], mask_grid[t])
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        writer.release()

        for pos in preview_positions:
            t = min(pos // 2, t_tokens - 1)
            overlay = overlay_mask_on_frame(cropped_frames[pos], mask_grid[t])
            out_path = overlay_dir / f"{spec['mask_name']}_frame_{pos:02d}_srcidx_{int(frame_indices[pos]):03d}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def compute_training_flow(cfg: dict[str, Any], masks_enc: list[torch.Tensor], masks_pred: list[torch.Tensor]) -> dict[str, Any]:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    batch_vis = 1
    num_frames = data_cfg["dataset_fpcs"][0]
    crop_size = data_cfg["crop_size"]
    patch_size = data_cfg["patch_size"]
    tubelet_size = data_cfg["tubelet_size"]
    t_tokens = num_frames // tubelet_size
    h_tokens = crop_size // patch_size
    w_tokens = crop_size // patch_size
    total_tokens = t_tokens * h_tokens * w_tokens
    embed_dim = MODEL_EMBED_DIMS[model_cfg["model_name"]]
    pred_embed_dim = model_cfg["pred_embed_dim"]

    mask_flows = []
    for i, (enc, pred) in enumerate(zip(masks_enc, masks_pred)):
        k_enc = int(enc.shape[1])
        k_pred = int(pred.shape[1])
        mask_ratio = k_pred / total_tokens
        mask_flows.append(
            {
                "mask_name": f"mask_{i}",
                "config": cfg["mask"][i],
                "k_enc": k_enc,
                "k_pred": k_pred,
                "mask_ratio": mask_ratio,
                "encoder_in": [batch_vis, 3, num_frames, crop_size, crop_size],
                "target_encoder_out": [batch_vis, total_tokens, embed_dim],
                "encoder_visible_tokens": [batch_vis, k_enc, embed_dim],
                "predictor_embed_context": [batch_vis, k_enc, pred_embed_dim],
                "predictor_mask_tokens": [batch_vis, k_pred, pred_embed_dim],
                "predictor_concat_sorted": [batch_vis, k_enc + k_pred, pred_embed_dim],
                "predictor_out_masked": [batch_vis, k_pred, embed_dim],
                "target_masked_for_loss": [batch_vis, k_pred, embed_dim],
            }
        )

    return {
        "num_frames": num_frames,
        "crop_size": crop_size,
        "patch_size": patch_size,
        "tubelet_size": tubelet_size,
        "t_tokens": t_tokens,
        "h_tokens": h_tokens,
        "w_tokens": w_tokens,
        "total_tokens": total_tokens,
        "embed_dim": embed_dim,
        "pred_embed_dim": pred_embed_dim,
        "mask_flows": mask_flows,
    }


def render_html(
    report: dict[str, Any],
    out_dir: Path,
    title: str,
) -> None:
    mask_cards = []
    for spec in report["training_flow"]["mask_flows"]:
        config_json = json.dumps(spec["config"], ensure_ascii=False, indent=2)
        overlay_frames = []
        for img in sorted((out_dir / "mask_overlays").glob(f"{spec['mask_name']}_frame_*.jpg")):
            overlay_frames.append(
                f'<figure><img src="mask_overlays/{img.name}" alt="{img.name}"><figcaption>{img.name}</figcaption></figure>'
            )
        mask_cards.append(
            f"""
            <section class="mask-card">
              <h3>{spec['mask_name']}</h3>
              <p>被预测 token: <code>{spec['k_pred']}</code>，可见 context token: <code>{spec['k_enc']}</code>，mask 比例: <code>{spec['mask_ratio']:.2%}</code></p>
              <div class="video-grid">
                <div>
                  <video controls muted loop playsinline src="mask_overlays/{spec['mask_name']}_overlay.mp4"></video>
                </div>
                <div class="flow-box">
                  <div><code>encoder(clips, masks_enc)</code> -> {spec['encoder_visible_tokens']}</div>
                  <div><code>predictor_embed(context)</code> -> {spec['predictor_embed_context']}</div>
                  <div><code>mask tokens</code> -> {spec['predictor_mask_tokens']}</div>
                  <div><code>concat + sort</code> -> {spec['predictor_concat_sorted']}</div>
                  <div><code>predictor output</code> -> {spec['predictor_out_masked']}</div>
                  <div><code>target masked</code> -> {spec['target_masked_for_loss']}</div>
                </div>
              </div>
              <div class="thumb-grid">{''.join(overlay_frames)}</div>
              <pre>{config_json}</pre>
            </section>
            """
        )

    preprocess_rows = []
    for row in report["preprocess"]["records"]:
        preprocess_rows.append(
            f"<tr><td>{row['name']}</td><td>{row['shape']}</td><td>{row['dtype']}</td><td>{row['zh']}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #efe7da;
      --panel: rgba(255, 251, 245, 0.95);
      --ink: #13212f;
      --muted: #53616d;
      --line: #d6c7b4;
      --accent: #b84a1b;
      --accent-soft: #f7d8c8;
      --green: #1f7a5c;
      --blue: #205b8f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Source Han Sans SC", "Noto Sans CJK SC", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(247, 216, 200, 0.9) 0, transparent 28%),
        linear-gradient(180deg, #f6f0e8 0%, #ece3d7 100%);
    }}
    main {{
      width: min(1320px, calc(100vw - 28px));
      margin: 18px auto 48px;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      margin-bottom: 16px;
      box-shadow: 0 12px 30px rgba(19, 33, 47, 0.08);
      backdrop-filter: blur(8px);
    }}
    h1, h2, h3 {{ margin: 0 0 10px; }}
    p, li {{ color: var(--muted); line-height: 1.65; }}
    code {{
      background: #f4ece3;
      border-radius: 6px;
      padding: 2px 6px;
      font-size: 0.95em;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{ background: var(--accent-soft); }}
    .hero {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 18px;
      align-items: start;
    }}
    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .pill {{
      display: inline-block;
      background: #eef3f7;
      color: var(--blue);
      border: 1px solid #cfdae2;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 13px;
    }}
    .flow-line {{
      display: grid;
      grid-template-columns: repeat(7, minmax(120px, 1fr));
      gap: 10px;
      align-items: center;
      margin-top: 14px;
    }}
    .flow-line-4 {{
      display: grid;
      grid-template-columns: repeat(4, minmax(180px, 1fr));
      gap: 10px;
      align-items: stretch;
      margin-top: 14px;
    }}
    .flow-node {{
      border-radius: 16px;
      padding: 14px;
      min-height: 106px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #fffdf9 0%, #f6efe7 100%);
    }}
    .flow-node strong {{
      display: block;
      margin-bottom: 6px;
    }}
    .arrow {{
      text-align: center;
      color: var(--accent);
      font-size: 22px;
      font-weight: 700;
    }}
    .thumb-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .thumb-grid img, video {{
      width: 100%;
      display: block;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #000;
    }}
    figure {{ margin: 0; }}
    figcaption {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
    }}
    .mask-card {{
      background: linear-gradient(180deg, rgba(255,251,245,0.98), rgba(247,239,230,0.98));
    }}
    .concept-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(280px, 1fr));
      gap: 14px;
    }}
    .concept-box {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: rgba(255,255,255,0.72);
    }}
    .concept-box h3 {{
      margin: 0 0 8px;
    }}
    .concept-box pre {{
      margin: 10px 0 0;
      background: #15202b;
      color: #f1f7fa;
    }}
    .code-callout {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: linear-gradient(180deg, #fffdf8 0%, #f7efe6 100%);
      margin-top: 12px;
    }}
    .code-callout pre {{
      margin: 0;
      background: #101923;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(280px, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    .detail-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: rgba(255,255,255,0.72);
    }}
    .detail-card h3 {{
      margin: 0 0 8px;
    }}
    .detail-card .mono-row {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
      color: var(--ink);
      background: #f4ece3;
      border-radius: 8px;
      padding: 8px 10px;
      margin: 8px 0;
      word-break: break-word;
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: 0.8fr 1.2fr;
      gap: 14px;
      align-items: start;
    }}
    .flow-box {{
      border: 1px dashed var(--line);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255,255,255,0.65);
    }}
    .flow-box div {{
      padding: 8px 0;
      border-bottom: 1px solid #eadfce;
      color: var(--muted);
    }}
    .flow-box div:last-child {{ border-bottom: 0; }}
    pre {{
      overflow: auto;
      padding: 12px;
      border-radius: 12px;
      background: #1c252f;
      color: #f3f7fa;
      font-size: 12px;
      line-height: 1.5;
    }}
    @media (max-width: 980px) {{
      .hero, .video-grid, .flow-line {{
        grid-template-columns: 1fr;
      }}
      .arrow {{ display: none; }}
    }}
  </style>
</head>
<body>
  <main>
    <section>
      <div class="hero">
        <div>
          <h1>{title}</h1>
          <p>这个页面把 <code>V-JEPA 2</code> 标准训练线可视化成可读流程：从输入视频、训练预处理、mask 采样、encoder/target encoder/predictor 的变量 shape，到最终 loss 对齐位置，全部落成一个本地页面。</p>
          <div class="stats">
            <span class="pill">输入视频: {report['video_meta']['width']}x{report['video_meta']['height']}</span>
            <span class="pill">原始帧数: {report['video_meta']['frame_count']}</span>
            <span class="pill">训练采样帧数: {report['training_flow']['num_frames']}</span>
            <span class="pill">token 网格: {report['training_flow']['t_tokens']} x {report['training_flow']['h_tokens']} x {report['training_flow']['w_tokens']}</span>
            <span class="pill">总 token: {report['training_flow']['total_tokens']}</span>
            <span class="pill">encoder hidden: {report['training_flow']['embed_dim']}</span>
            <span class="pill">predictor hidden: {report['training_flow']['pred_embed_dim']}</span>
          </div>
        </div>
        <div>
          <video controls muted loop playsinline src="input_video.mp4"></video>
        </div>
      </div>
    </section>

    <section>
      <h2>训练主流程</h2>
      <div class="flow-line">
        <div class="flow-node"><strong>1. clips</strong><div><code>[1, 3, 64, 384, 384]</code></div><div>单视频可视化时用 batch=1，实际训练 batch 见配置。</div></div>
        <div class="arrow">→</div>
        <div class="flow-node"><strong>2. target_encoder</strong><div><code>[1, 18432, 1408]</code></div><div>teacher 分支看完整 token。</div></div>
        <div class="arrow">→</div>
        <div class="flow-node"><strong>3. encoder + masks_enc</strong><div><code>[1, K_enc, 1408]</code></div><div>student 只看可见 context token。</div></div>
        <div class="arrow">→</div>
        <div class="flow-node"><strong>4. predictor + masks_pred</strong><div><code>[1, K_pred, 1408]</code></div><div>预测被遮住的 token。</div></div>
      </div>
      <p>loss 不是像素重建，而是把 predictor 输出的 masked latent，和 target_encoder 在同位置的 latent 做对齐。</p>
    </section>

    <section>
      <h2>源码两行的细化流程</h2>
      <div class="code-callout">
        <pre>def forward_context(c):
    z = encoder(c, masks_enc)
    z = predictor(z, masks_enc, masks_pred)
    return z</pre>
      </div>
      <p>这是 [`app/vjepa/train.py`](/home/gaoya/Code_Video/WAN_2p2/vjepa2-main/app/vjepa/train.py#L435) 里的原始训练代码。这里的 <code>encoder</code> 和 <code>predictor</code> 都带了 wrapper，因此真实输入输出首先是嵌套 list，然后才可以落到单个 mask 的简化 tensor shape。</p>

      <div class="detail-grid">
        <div class="detail-card">
          <h3>第一行：<code>z = encoder(c, masks_enc)</code></h3>
          <div class="mono-row">真实训练输入: c = list[Tensor[B, 3, 64, 384, 384]]</div>
          <div class="mono-row">真实训练输入: masks_enc = list[list[Tensor[B, K_enc]]]</div>
          <div class="mono-row">真实训练输出: z = list[list[Tensor[B, K_enc, 1408]]]</div>
          <p>含义：每个 clip 先被切成完整 token 序列 <code>[B, 18432, 1408]</code>，再按 <code>masks_enc</code> 只保留可见 token。对于当前页面展示的 <code>mask_0</code>，单视频时简化成 <code>[1, 3, 64, 384, 384] -&gt; [1, 5824, 1408]</code>。</p>
        </div>
        <div class="detail-card">
          <h3>第二行：<code>z = predictor(z, masks_enc, masks_pred)</code></h3>
          <div class="mono-row">真实训练输入: z = list[list[Tensor[B, K_enc, 1408]]]</div>
          <div class="mono-row">真实训练输入: masks_enc = list[list[Tensor[B, K_enc]]]</div>
          <div class="mono-row">真实训练输入: masks_pred = list[list[Tensor[B, K_pred]]]</div>
          <div class="mono-row">真实训练输出: z = list[list[Tensor[B, K_pred, 1408]]]</div>
          <p>含义：predictor 知道可见 token 原来在什么位置，也知道哪些位置被遮住了要补出来。它会把 context token 映射到 predictor 维度，再造出对应数量的 mask token，占位后拼回完整序列，最后只返回 target 部分的预测结果。</p>
        </div>
      </div>

      <div class="flow-line-4">
        <div class="flow-node">
          <strong>单个 mask 的 encoder 视角</strong>
          <div><code>[1, 3, 64, 384, 384]</code></div>
          <div>先 patch embed 成完整 <code>[1, 18432, 1408]</code>，再按 <code>masks_enc</code> 保留可见 token。</div>
        </div>
        <div class="flow-node">
          <strong>encoder 输出</strong>
          <div><code>[1, K_enc, 1408]</code></div>
          <div>当前 <code>mask_0</code> 是 <code>K_enc=5824</code>。这就是 context token。</div>
        </div>
        <div class="flow-node">
          <strong>predictor 内部拼接</strong>
          <div><code>[1, K_enc, 384]</code> + <code>[1, K_pred, 384]</code></div>
          <div>当前 <code>mask_0</code> 里 <code>K_pred=12608</code>，拼完会回到完整 token 数。</div>
        </div>
        <div class="flow-node">
          <strong>predictor 输出</strong>
          <div><code>[1, K_pred, 1408]</code></div>
          <div>只返回被遮住位置的预测特征，随后与 target encoder 同位置特征做 loss。</div>
        </div>
      </div>

      <div class="detail-grid">
        <div class="detail-card">
          <h3>为什么 predictor 知道要预测多少个 token</h3>
          <p>不是从 <code>[1, 5824, 1408]</code> 这个 shape 猜出来的，而是直接从 <code>masks_pred</code> 得知。比如当前 <code>mask_0</code> 里，<code>masks_pred.shape = [1, 12608]</code>，所以 predictor 会生成 <code>12608</code> 个 mask token，占据这些目标位置。</p>
        </div>
        <div class="detail-card">
          <h3>loss 比较的两边是什么</h3>
          <div class="mono-row">predictor output = [1, K_pred, 1408]</div>
          <div class="mono-row">target masked = [1, K_pred, 1408]</div>
          <p>target encoder 先看完整 clip，得到 <code>[1, 18432, 1408]</code>，再按 <code>masks_pred</code> 取出同位置 token。训练时比较的是两个 latent 表征，不是比较像素。</p>
        </div>
      </div>
    </section>

    <section>
      <h2>训练预处理 Shape</h2>
      <table>
        <thead><tr><th>变量</th><th>Shape</th><th>DType</th><th>中文解释</th></tr></thead>
        <tbody>{''.join(preprocess_rows)}</tbody>
      </table>
    </section>

    <section>
      <h2>采样帧预览</h2>
      <p>这些帧来自本次训练可视化实际使用的 64 帧采样序列。页面右下的 mask 叠加视频是在中心裁剪后的训练输入帧上生成的。</p>
      <div class="thumb-grid">
        <figure><img src="sample_frames/frame_00.jpg" alt="frame_00"><figcaption>frame_00</figcaption></figure>
        <figure><img src="sample_frames/frame_21.jpg" alt="frame_21"><figcaption>frame_21</figcaption></figure>
        <figure><img src="sample_frames/frame_42.jpg" alt="frame_42"><figcaption>frame_42</figcaption></figure>
        <figure><img src="sample_frames/frame_63.jpg" alt="frame_63"><figcaption>frame_63</figcaption></figure>
      </div>
    </section>

    <section>
      <h2>Patch / Token / Block 示意</h2>
      <div class="concept-grid">
        <div class="concept-box">
          <h3>1. 单帧 Patch 网格</h3>
          <p>训练输入每一帧是 <code>384 x 384</code>，而 <code>patch_size = 16</code>，所以一帧会被切成 <code>24 x 24 = 576</code> 个空间 patch。</p>
          <pre>384 px frame
┌──────────────────────────────┐
│[][][][][][][][][][][][][][][]│
│[][][][][][][][][][][][][][][]│
│[][][][][][][][][][][][][][][]│  24 x 24 cells
│............ each cell .......│  each cell = 16 x 16 pixels
│[][][][][][][][][][][][][][][]│
└──────────────────────────────┘</pre>
        </div>
        <div class="concept-box">
          <h3>2. 视频 Token 网格</h3>
          <p><code>tubelet_size = 2</code>，所以 64 帧会先变成 32 个时间片。整个视频 token 网格是 <code>32 x 24 x 24</code>。</p>
          <pre>time depth = 32

slice t=0   : 24 x 24 patches
slice t=1   : 24 x 24 patches
...
slice t=31  : 24 x 24 patches

total tokens = 32 x 24 x 24 = 18432</pre>
        </div>
        <div class="concept-box">
          <h3>3. “覆盖整个时间维”</h3>
          <p>配置里 <code>temporal_scale = 1.0</code>，意味着一个 mask block 的时间长度 <code>t = 32</code>，也就是从视频开头一直延伸到结尾，不只是盖住几帧。</p>
          <pre>time axis
0 ─────────────────────────── 31
██████████████████████████████

same spatial area
repeats through all time slices</pre>
        </div>
        <div class="concept-box">
          <h3>4. “空间覆盖 15% 面积”</h3>
          <p>配置里 <code>spatial_scale = 0.15</code>，是相对单帧 patch 网格面积来说的。单帧有 <code>24 x 24 = 576</code> 个 patch 位置，15% 约等于 <code>86</code> 个 patch。</p>
          <pre>single frame patch map: 24 x 24 = 576 cells

masked spatial area per block
≈ 576 x 0.15
≈ 86 cells

possible rectangle examples:
8 x 11 = 88
9 x 10 = 90
7 x 12 = 84</pre>
        </div>
      </div>
      <p>所以一个 “mask block” 不是一个像素块，而是视频 token 网格里的一个三维长方体区域：<code>t x h x w</code>。在当前配置下，它大致是 “时间贯穿 32 层，空间每层盖住约 86 个 patch 位置”。</p>
    </section>

    <section>
      <h2>Mask 与变量 Shape</h2>
      <p>下面每张卡片对应配置里的一个 mask generator。当前配置有两种 mask：一种是很多个小块，另一种是少量大块。它们都会在同一个训练 step 里产生一个 predictor 输出，并各自参与 loss。</p>
      {''.join(mask_cards)}
    </section>
  </main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def save_sample_frames(cropped_frames: np.ndarray, out_dir: Path) -> None:
    sample_dir = out_dir / "sample_frames"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for pos in [0, len(cropped_frames) // 3, 2 * len(cropped_frames) // 3, len(cropped_frames) - 1]:
        out_path = sample_dir / f"frame_{pos:02d}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(cropped_frames[pos], cv2.COLOR_RGB2BGR))


def get_video_meta(video_path: str) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    meta = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/gaoya/Code_Video/WAN_2p2/vjepa2-main/configs/train/vitg16/cooldown-384px-64f.yaml")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-dir", default="/home/gaoya/Code_Video/WAN_2p2/vjepa2-main/analysis_outputs/bear_training_flow")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_meta = get_video_meta(args.video_path)
    num_frames = cfg["data"]["dataset_fpcs"][0]
    crop_size = cfg["data"]["crop_size"]
    patch_size = cfg["data"]["patch_size"]
    tubelet_size = cfg["data"]["tubelet_size"]
    t_tokens = num_frames // tubelet_size
    h_tokens = crop_size // patch_size
    w_tokens = crop_size // patch_size

    video_thwc, frame_indices = read_video(args.video_path, num_frames=num_frames)
    pre = preprocess_frames(video_thwc, crop_size=crop_size)
    cropped_frames = pre["cropped_uint8"]
    save_sample_frames(cropped_frames, out_dir)

    shutil.copy2(args.video_path, out_dir / "input_video.mp4")

    generators = build_mask_generators(cfg)
    masks_enc = []
    masks_pred = []
    mask_specs = []
    for i, gen in enumerate(generators):
        enc, pred = gen(1)
        masks_enc.append(enc)
        masks_pred.append(pred)
        mask_specs.append(
            {
                "mask_name": f"mask_{i}",
                "mask_grid": mask_indices_to_grid(pred, t_tokens, h_tokens, w_tokens),
            }
        )
    save_mask_assets(cropped_frames, frame_indices, mask_specs, t_tokens, h_tokens, w_tokens, out_dir)

    training_flow = compute_training_flow(cfg, masks_enc, masks_pred)
    for spec, flow in zip(mask_specs, training_flow["mask_flows"]):
        spec.update(flow)

    report = {
        "config_path": args.config,
        "video_path": args.video_path,
        "video_meta": video_meta,
        "frame_indices": frame_indices.tolist(),
        "preprocess": {"records": pre["records"]},
        "training_flow": training_flow,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    render_html(
        report={
            **report,
            "training_flow": {**training_flow, "mask_flows": mask_specs},
        },
        out_dir=out_dir,
        title="V-JEPA 2 训练流程可视化",
    )
    print(json.dumps({"output_dir": str(out_dir), "index": str(out_dir / "index.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
