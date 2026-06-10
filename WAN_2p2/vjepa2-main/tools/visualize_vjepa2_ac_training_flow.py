import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from decord import VideoReader, cpu


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


def get_video_meta(video_path: str) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    meta = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return meta


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
            "zh": "等间隔采样后的原始视频帧，格式 [T, H, W, C]。",
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
            "zh": "整理成训练时输入的 [B, C, T, H, W]。",
        }
    )

    return {"tensor": x, "records": records, "cropped_uint8": cropped_uint8}


def save_sample_frames(cropped_frames: np.ndarray, out_dir: Path) -> None:
    sample_dir = out_dir / "sample_frames"
    sample_dir.mkdir(parents=True, exist_ok=True)
    preview_positions = [0, len(cropped_frames) // 3, 2 * len(cropped_frames) // 3, len(cropped_frames) - 1]
    for pos in preview_positions:
        out_path = sample_dir / f"frame_{pos:02d}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(cropped_frames[pos], cv2.COLOR_RGB2BGR))


def build_report(cfg: dict[str, Any], video_path: str, frame_indices: np.ndarray, preprocess_records: list[dict[str, Any]]) -> dict[str, Any]:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]

    num_frames = data_cfg["dataset_fpcs"][0]
    crop_size = data_cfg["crop_size"]
    patch_size = data_cfg["patch_size"]
    embed_dim = MODEL_EMBED_DIMS[model_cfg["model_name"]]
    pred_embed_dim = model_cfg["pred_embed_dim"]
    tokens_per_frame = int((crop_size // patch_size) ** 2)
    total_tokens = num_frames * tokens_per_frame
    auto_steps = min(loss_cfg["auto_steps"], num_frames)
    batch_vis = 1

    return {
        "config_path": cfg["_config_path"],
        "video_path": video_path,
        "video_meta": get_video_meta(video_path),
        "frame_indices": frame_indices.tolist(),
        "preprocess": {"records": preprocess_records},
        "flow": {
            "num_frames": num_frames,
            "crop_size": crop_size,
            "patch_size": patch_size,
            "tubelet_size": data_cfg["tubelet_size"],
            "tokens_per_frame": tokens_per_frame,
            "total_tokens": total_tokens,
            "embed_dim": embed_dim,
            "pred_embed_dim": pred_embed_dim,
            "auto_steps": auto_steps,
            "loss_exp": float(loss_cfg["loss_exp"]),
            "normalize_reps": bool(loss_cfg["normalize_reps"]),
            "clips": [batch_vis, 3, num_frames, crop_size, crop_size],
            "forward_target_in": [batch_vis * num_frames, 3, 2, crop_size, crop_size],
            "target_encoder_out_per_frame": [batch_vis * num_frames, tokens_per_frame, embed_dim],
            "target_flattened_h": [batch_vis, total_tokens, embed_dim],
            "teacher_forcing_in": [batch_vis, total_tokens - tokens_per_frame, embed_dim],
            "teacher_forcing_actions": [batch_vis, num_frames - 1, 7],
            "teacher_forcing_states": [batch_vis, num_frames - 1, 7],
            "teacher_forcing_out": [batch_vis, total_tokens - tokens_per_frame, embed_dim],
            "ar_seed_in": [batch_vis, 2 * tokens_per_frame, embed_dim],
            "ar_step_out_full": [batch_vis, 2 * tokens_per_frame, embed_dim],
            "ar_step_out_last_frame": [batch_vis, tokens_per_frame, embed_dim],
            "ar_concat": [batch_vis, 3 * tokens_per_frame, embed_dim],
            "ar_final": [batch_vis, 2 * tokens_per_frame, embed_dim],
            "jloss_target": [batch_vis, total_tokens - tokens_per_frame, embed_dim],
            "sloss_target": [batch_vis, 2 * tokens_per_frame, embed_dim],
            "predictor_internal": {
                "predictor_embed_in": [batch_vis, 2 * tokens_per_frame, embed_dim],
                "predictor_embed_out": [batch_vis, 2 * tokens_per_frame, pred_embed_dim],
                "predictor_view": [batch_vis, 2, tokens_per_frame, pred_embed_dim],
                "predictor_with_action_state": [batch_vis, 2 * (tokens_per_frame + 2), pred_embed_dim],
                "predictor_proj_out": [batch_vis, 2 * tokens_per_frame, embed_dim],
            },
        },
    }


def render_html(report: dict[str, Any], out_dir: Path) -> None:
    flow = report["flow"]
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
  <title>V-JEPA 2-AC 训练流程可视化</title>
  <style>
    :root {{
      --bg: #f0eee8;
      --panel: rgba(255, 252, 247, 0.95);
      --ink: #162331;
      --muted: #56646f;
      --line: #d8cdbd;
      --accent: #b45309;
      --accent-soft: #f5e1cf;
      --blue: #1d4f7a;
      --green: #116149;
      --red: #9d2d2d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Source Han Sans SC", "Noto Sans CJK SC", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(245, 225, 207, 0.9) 0, transparent 28%),
        linear-gradient(180deg, #f7f2ea 0%, #ece5d9 100%);
    }}
    main {{
      width: min(1360px, calc(100vw - 28px));
      margin: 18px auto 48px;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      margin-bottom: 16px;
      box-shadow: 0 12px 30px rgba(22, 35, 49, 0.08);
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
    pre {{
      overflow: auto;
      padding: 12px;
      border-radius: 12px;
      background: #18222d;
      color: #f3f7fa;
      font-size: 12px;
      line-height: 1.55;
      margin: 0;
    }}
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
      grid-template-columns: repeat(5, minmax(150px, 1fr));
      gap: 10px;
      align-items: stretch;
      margin-top: 14px;
    }}
    .flow-node {{
      border-radius: 16px;
      padding: 14px;
      min-height: 118px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #fffdf9 0%, #f6efe7 100%);
    }}
    .flow-node strong {{
      display: block;
      margin-bottom: 6px;
    }}
    .timeline {{
      display: grid;
      grid-template-columns: repeat(8, minmax(70px, 1fr));
      gap: 8px;
      margin-top: 12px;
    }}
    .frame-box {{
      padding: 12px 8px;
      border: 1px solid var(--line);
      border-radius: 12px;
      text-align: center;
      background: rgba(255,255,255,0.78);
    }}
    .frame-box.real {{
      border-color: #bdd8c6;
      background: #eff9f2;
      color: var(--green);
    }}
    .frame-box.pred {{
      border-color: #e7c4c4;
      background: #fff2f2;
      color: var(--red);
    }}
    .two-col {{
      display: grid;
      grid-template-columns: repeat(2, minmax(300px, 1fr));
      gap: 14px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: rgba(255,255,255,0.72);
    }}
    .mono-row {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
      color: var(--ink);
      background: #f4ece3;
      border-radius: 8px;
      padding: 8px 10px;
      margin: 8px 0;
      word-break: break-word;
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
    .backprop {{
      display: grid;
      grid-template-columns: 1.2fr 1fr 1fr 1fr;
      gap: 12px;
      margin-top: 14px;
    }}
    .backprop .card {{
      min-height: 160px;
    }}
    @media (max-width: 980px) {{
      .hero, .two-col, .backprop, .flow-line, .timeline {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section>
      <div class="hero">
        <div>
          <h1>V-JEPA 2-AC 训练流程可视化</h1>
          <p>这个页面对应 <code>app/vjepa_droid/train.py</code> 的实际训练线。重点不是 mask 重建，而是 action-conditioned predictor 在 target latent 上做 teacher forcing 和短程 auto-regressive rollout，并用 latent L1 loss 训练。</p>
          <div class="stats">
            <span class="pill">输入视频: {report['video_meta']['width']}x{report['video_meta']['height']}</span>
            <span class="pill">原始帧数: {report['video_meta']['frame_count']}</span>
            <span class="pill">训练帧数: {flow['num_frames']}</span>
            <span class="pill">patch/frame: {flow['tokens_per_frame']}</span>
            <span class="pill">total tokens: {flow['total_tokens']}</span>
            <span class="pill">encoder dim: {flow['embed_dim']}</span>
            <span class="pill">predictor dim: {flow['pred_embed_dim']}</span>
            <span class="pill">auto_steps: {flow['auto_steps']}</span>
            <span class="pill">loss_exp: {flow['loss_exp']}</span>
          </div>
        </div>
        <div>
          <video controls muted loop playsinline src="input_video.mp4"></video>
        </div>
      </div>
    </section>

    <section>
      <h2>主流程</h2>
      <div class="flow-line">
        <div class="flow-node">
          <strong>1. clips</strong>
          <div><code>{flow['clips']}</code></div>
          <div>输入 8 帧视频，shape 是 <code>[B, C, T, H, W]</code>。</div>
        </div>
        <div class="flow-node">
          <strong>2. target encoder</strong>
          <div><code>{flow['forward_target_in']}</code> -> <code>{flow['target_flattened_h']}</code></div>
          <div>每帧被复制成 2 帧 tubelet 输入，编码成完整目标 latent <code>h</code>。</div>
        </div>
        <div class="flow-node">
          <strong>3. teacher forcing</strong>
          <div><code>{flow['teacher_forcing_in']}</code> -> <code>{flow['teacher_forcing_out']}</code></div>
          <div>真实的 <code>[0..6]</code> 作为输入，预测 <code>[1..7]</code>。</div>
        </div>
        <div class="flow-node">
          <strong>4. rollout</strong>
          <div><code>{flow['ar_seed_in']}</code> -> <code>{flow['ar_final']}</code></div>
          <div>从 <code>[0(real), 1(pred)]</code> 继续滚到 <code>2(pred)</code>。</div>
        </div>
        <div class="flow-node">
          <strong>5. latent loss</strong>
          <div><code>jloss + sloss</code></div>
          <div>两项都是 latent L1，对齐 target encoder 特征，不比较像素。</div>
        </div>
      </div>
    </section>

    <section>
      <h2>8 帧时间轴</h2>
      <p>下面统一用 <code>[0,1,2,3,4,5,6,7]</code> 表示一段训练样本的 8 帧。</p>
      <div class="timeline">
        <div class="frame-box real">0<br>real</div>
        <div class="frame-box real">1<br>real</div>
        <div class="frame-box real">2<br>real</div>
        <div class="frame-box real">3<br>real</div>
        <div class="frame-box real">4<br>real</div>
        <div class="frame-box real">5<br>real</div>
        <div class="frame-box real">6<br>real</div>
        <div class="frame-box real">7<br>real</div>
      </div>
      <div class="two-col" style="margin-top:14px;">
        <div class="card">
          <h3>target encoder 看哪些帧</h3>
          <div class="mono-row">target encoder input frames = [0,1,2,3,4,5,6,7]</div>
          <div class="mono-row">h.shape = {flow['target_flattened_h']}</div>
          <p>完整 8 帧都被编码成目标 latent <code>h</code>。每帧有 <code>{flow['tokens_per_frame']}</code> 个 patch token，所以总长度是 <code>8 x {flow['tokens_per_frame']} = {flow['total_tokens']}</code>。</p>
        </div>
        <div class="card">
          <h3>teacher forcing 输入哪些帧</h3>
          <div class="mono-row">predictor input = [0(real),1(real),2(real),3(real),4(real),5(real),6(real)]</div>
          <div class="mono-row">z_tf.shape = {flow['teacher_forcing_out']}</div>
          <p>teacher forcing 阶段输入真实的前 7 帧 latent，输出对齐到下一帧序列，即 <code>[1(pred),2(pred),3(pred),4(pred),5(pred),6(pred),7(pred)]</code>。</p>
        </div>
      </div>
    </section>

    <section>
      <h2>auto_steps=2 的 rollout 细化</h2>
      <div class="two-col">
        <div class="card">
          <h3>第 1 步：构造 rollout 起点</h3>
          <div class="mono-row">_z = concat(z[:, :tokens_per_frame], z_tf[:, :tokens_per_frame])</div>
          <div class="mono-row">shape = {flow['ar_seed_in']}</div>
          <div class="mono-row">语义 = [0(real), 1(pred)]</div>
          <p>这里不是把整段 <code>z_tf</code> 都拿来滚，而是只取它的第一帧预测，也就是 <code>1(pred)</code>，和真实的 <code>0(real)</code> 拼成 rollout 的起点。</p>
        </div>
        <div class="card">
          <h3>第 2 步：继续滚出 2(pred)</h3>
          <div class="mono-row">_z_nxt = _step_predictor(_z, actions[:, :2], states[:, :2], extrinsics[:, :2])[:, -tokens_per_frame:]</div>
          <div class="mono-row">full predictor out = {flow['ar_step_out_full']}</div>
          <div class="mono-row">last frame only = {flow['ar_step_out_last_frame']}</div>
          <div class="mono-row">语义 = 2(pred)</div>
          <p>因为 <code>auto_steps=2</code>，循环只执行一次。输入是 <code>[0(real),1(pred)]</code>，最后一帧切出来就是 <code>2(pred)</code>。</p>
        </div>
      </div>
      <div class="card" style="margin-top:14px;">
        <h3>最终 rollout 序列</h3>
        <div class="mono-row">_z = [0(real), 1(pred), 2(pred)]</div>
        <div class="mono-row">z_ar = _z[:, tokens_per_frame:] = [1(pred), 2(pred)]</div>
        <div class="mono-row">z_ar.shape = {flow['ar_final']}</div>
        <p>训练里真正参与 <code>sloss</code> 的不是三帧，而是去掉第一帧 context 之后的两帧预测：<code>[1(pred),2(pred)]</code>。</p>
      </div>
    </section>

    <section>
      <h2>jloss 和 sloss 各自比较什么</h2>
      <div class="two-col">
        <div class="card">
          <h3>jloss: teacher forcing loss</h3>
          <div class="mono-row">z_tf = [1(pred),2(pred),3(pred),4(pred),5(pred),6(pred),7(pred)]</div>
          <div class="mono-row">target = h[:, tokens_per_frame : z_tf.size(1) + tokens_per_frame]</div>
          <div class="mono-row">target = [1(target),2(target),3(target),4(target),5(target),6(target),7(target)]</div>
          <div class="mono-row">shape = {flow['jloss_target']}</div>
          <p><code>jloss</code> 监督 teacher forcing 输出整段下一帧预测序列，对齐到目标的帧 1 到帧 7。</p>
        </div>
        <div class="card">
          <h3>sloss: rollout loss</h3>
          <div class="mono-row">z_ar = [1(pred),2(pred)]</div>
          <div class="mono-row">target = h[:, tokens_per_frame : z_ar.size(1) + tokens_per_frame]</div>
          <div class="mono-row">target = [1(target),2(target)]</div>
          <div class="mono-row">shape = {flow['sloss_target']}</div>
          <p>这就是你问的那一项。<code>sloss</code> 比较的是 rollout 结果 <code>[1(pred),2(pred)]</code> 和 target encoder 编码得到的 <code>[1,2]</code> 两帧特征。</p>
        </div>
      </div>
      <div class="card" style="margin-top:14px;">
        <h3>loss 公式</h3>
        <pre>def loss_fn(z, h):
    _h = h[:, tokens_per_frame : z.size(1) + tokens_per_frame]
    return torch.mean(torch.abs(z - _h) ** loss_exp) / loss_exp</pre>
        <p>当前配置 <code>loss_exp = 1.0</code>，所以就是逐 token、逐通道的 L1 平均。对 <code>sloss</code> 来说，实际比较张量 shape 是 <code>{flow['ar_final']}</code> 对 <code>{flow['sloss_target']}</code>。</p>
      </div>
    </section>

    <section>
      <h2>predictor 内部 shape</h2>
      <div class="two-col">
        <div class="card">
          <h3>输入与动作状态拼接</h3>
          <div class="mono-row">x in = {flow['predictor_internal']['predictor_embed_in']}</div>
          <div class="mono-row">predictor_embed(x) = {flow['predictor_internal']['predictor_embed_out']}</div>
          <div class="mono-row">view to [B, T, H*W, D] = {flow['predictor_internal']['predictor_view']}</div>
          <div class="mono-row">concat action/state = {flow['predictor_internal']['predictor_with_action_state']}</div>
          <p>predictor 每帧除了 patch token，还会在帧前面插入 action token 和 state token，所以每帧内部长度从 <code>H*W</code> 变成 <code>H*W+2</code>。</p>
        </div>
        <div class="card">
          <h3>输出</h3>
          <div class="mono-row">predictor_proj out = {flow['predictor_internal']['predictor_proj_out']}</div>
          <p>predictor 最后会把 action/state token 再拆掉，只保留视频 patch token，并投影回 encoder latent 维度 <code>{flow['embed_dim']}</code>。</p>
        </div>
      </div>
    </section>

    <section>
      <h2>loss 回传路径</h2>
      <div class="backprop">
        <div class="card">
          <h3>前向</h3>
          <div class="mono-row">clips -> forward_target(clips) -> h</div>
          <div class="mono-row">h -> forward_predictions(h) -> z_tf, z_ar</div>
          <div class="mono-row">loss = jloss + sloss</div>
          <p>这条训练线里，predictor 的输入直接就是 <code>h</code>，也就是 target encoder 产生的目标 latent。</p>
        </div>
        <div class="card">
          <h3>不会回传到 target encoder</h3>
          <div class="mono-row">with torch.no_grad(): h = target_encoder(...)</div>
          <div class="mono-row">for p in target_encoder.parameters(): p.requires_grad = False</div>
          <p>target encoder 是冻结的，而且前向在 <code>no_grad</code> 里，因此 <code>jloss/sloss</code> 都不会回传到 target encoder。</p>
        </div>
        <div class="card">
          <h3>会回传到 predictor</h3>
          <div class="mono-row">loss -> z_tf / z_ar -> predictor(...)</div>
          <div class="mono-row">包括: predictor_embed, action_encoder, state_encoder, transformer blocks, predictor_proj</div>
          <p>有梯度的是 predictor 全部内部参数，因为 loss 直接由 predictor 输出张量算出来。</p>
        </div>
        <div class="card">
          <h3>当前代码里 encoder 也不吃梯度</h3>
          <div class="mono-row">train_step 没有调用 encoder(...)</div>
          <div class="mono-row">loss 计算图里没有 encoder 分支</div>
          <p>按当前 <code>app/vjepa_droid/train.py</code> 实现，student encoder 并没有参与这条前向，因此这一步训练里它也拿不到梯度。实际被训练的是 predictor，不是标准 JEPA 里的 encoder + predictor 联合更新。</p>
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
      <p>这个页面用的是同一个输入视频做可视化，但只采样 8 帧并裁到当前 droid 配置的 <code>256x256</code>。它只是帮助看 shape，不代表机器人数据集语义本身。</p>
      <div class="thumb-grid">
        <img src="sample_frames/frame_00.jpg" alt="frame_00">
        <img src="sample_frames/frame_02.jpg" alt="frame_02">
        <img src="sample_frames/frame_05.jpg" alt="frame_05">
        <img src="sample_frames/frame_07.jpg" alt="frame_07">
      </div>
    </section>
  </main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/gaoya/Code_Video/WAN_2p2/vjepa2-main/configs/train/vitg16/droid-256px-8f.yaml",
    )
    parser.add_argument(
        "--video-path",
        default="/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench/output/GT/Biological_Motion/bear.mp4",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/gaoya/Code_Video/WAN_2p2/vjepa2-main/analysis_outputs/bear_training_flow_ac",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg["_config_path"] = args.config
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_frames = cfg["data"]["dataset_fpcs"][0]
    crop_size = cfg["data"]["crop_size"]

    video_thwc, frame_indices = read_video(args.video_path, num_frames=num_frames)
    pre = preprocess_frames(video_thwc, crop_size=crop_size)
    save_sample_frames(pre["cropped_uint8"], out_dir)
    shutil.copy2(args.video_path, out_dir / "input_video.mp4")

    report = build_report(
        cfg=cfg,
        video_path=args.video_path,
        frame_indices=frame_indices,
        preprocess_records=pre["records"],
    )
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    render_html(report, out_dir)
    print(json.dumps({"output_dir": str(out_dir), "index": str(out_dir / "index.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
