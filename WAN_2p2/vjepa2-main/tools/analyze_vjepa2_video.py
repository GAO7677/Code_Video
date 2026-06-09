import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu


IMAGENET_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class StepRecord:
    step: str
    shape: list[int]
    dtype: str
    explanation_zh: str


def shape_list(x: Any) -> list[int]:
    return list(x.shape)


def tensor_dtype(x: Any) -> str:
    return str(x.dtype).replace("torch.", "")


def record(records: list[StepRecord], step: str, x: Any, explanation_zh: str) -> None:
    records.append(
        StepRecord(
            step=step,
            shape=shape_list(x),
            dtype=tensor_dtype(x),
            explanation_zh=explanation_zh,
        )
    )


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


def sample_frame_indices(frame_count: int, num_frames: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("视频帧数为 0")
    if frame_count >= num_frames:
        idx = np.linspace(0, frame_count - 1, num_frames)
    else:
        idx = np.linspace(0, frame_count - 1, num_frames)
    return np.round(idx).astype(np.int64)


def load_sampled_video(video_path: str, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    vr = VideoReader(video_path, ctx=cpu(0))
    frame_count = len(vr)
    frame_idx = sample_frame_indices(frame_count, num_frames)
    video = vr.get_batch(frame_idx).asnumpy()
    return video, frame_idx


def manual_preprocess(video_thwc: np.ndarray, crop_size: int) -> tuple[torch.Tensor, list[StepRecord]]:
    records: list[StepRecord] = []

    x = torch.from_numpy(video_thwc)
    record(records, "sampled_frames_thwc", x, "从原视频中等间隔采样得到 64 帧，格式是 [时间, 高, 宽, 通道]。")

    x = x.permute(0, 3, 1, 2).contiguous()
    record(records, "permute_to_tchw", x, "把视频转成 PyTorch 常用格式 [时间, 通道, 高, 宽]，方便后续缩放和归一化。")

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
    record(
        records,
        "resize_short_side",
        x,
        f"先按短边缩放到 {short_side}，长边按比例变化；这是论文/仓库默认的评测预处理第一步。",
    )

    h, w = x.shape[-2:]
    top = max((h - crop_size) // 2, 0)
    left = max((w - crop_size) // 2, 0)
    x = x[:, :, top : top + crop_size, left : left + crop_size]
    record(
        records,
        "center_crop",
        x,
        f"从缩放后的每一帧中心裁剪出 {crop_size}x{crop_size}，保证输入空间尺寸固定。",
    )

    x = x / 255.0
    record(records, "rescale_0_1", x, "把像素从 0-255 缩放到 0-1 浮点范围。")

    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    record(records, "normalize", x, "按 ImageNet 均值和方差归一化，让输入分布与预训练时一致。")

    x = x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    record(records, "to_bcthw", x, "整理成模型真正接收的 [批次, 通道, 时间, 高, 宽]。")

    return x, records


def analyze_encoder(model: torch.nn.Module, x_bcthw: torch.Tensor) -> dict[str, Any]:
    records: list[StepRecord] = []
    with torch.no_grad():
        patch_tokens = model.patch_embed(x_bcthw)
        record(
            records,
            "patch_embed",
            patch_tokens,
            "3D patch embedding：以 tubelet=2、patch=16x16 的卷积把视频切成时空 token，并映射到隐藏维度。",
        )

        hidden = patch_tokens
        block_shapes = []
        for i, blk in enumerate(model.blocks):
            hidden = blk(hidden, mask=None, attn_mask=None, T=32, H_patches=24, W_patches=24)
            block_shapes.append(
                {
                    "block_index": i,
                    "shape": shape_list(hidden),
                    "dtype": tensor_dtype(hidden),
                    "explanation_zh": "Transformer block 不改变 token 数和通道维，只在时空 token 间做注意力与 MLP 变换。",
                }
            )
        hidden = model.norm(hidden)
        record(records, "final_norm", hidden, "最后一层 LayerNorm 后得到最终 patch-level 视觉表征。")
    return {
        "summary_records": [asdict(r) for r in records],
        "block_shapes": block_shapes,
        "final_features_shape": shape_list(hidden),
    }


def infer_encoder_shapes(
    *,
    num_frames: int,
    crop_size: int,
    patch_size: int,
    tubelet_size: int,
    hidden_size: int,
    num_hidden_layers: int,
) -> dict[str, Any]:
    records: list[StepRecord] = []
    batch_size = 1
    t_tokens = num_frames // tubelet_size
    h_tokens = crop_size // patch_size
    w_tokens = crop_size // patch_size
    n_tokens = t_tokens * h_tokens * w_tokens

    patch_embed = torch.empty((batch_size, n_tokens, hidden_size), dtype=torch.float32)
    record(
        records,
        "patch_embed",
        patch_embed,
        "3D patch embedding：以 tubelet=2、patch=16x16 的卷积把视频切成时空 token，并映射到隐藏维度。",
    )

    block_shapes = []
    hidden = patch_embed
    for i in range(num_hidden_layers):
        block_shapes.append(
            {
                "block_index": i,
                "shape": shape_list(hidden),
                "dtype": tensor_dtype(hidden),
                "explanation_zh": "Transformer block 不改变 token 数和通道维，只在时空 token 间做注意力与 MLP 变换。",
            }
        )

    final_hidden = torch.empty((batch_size, n_tokens, hidden_size), dtype=torch.float32)
    record(records, "final_norm", final_hidden, "最后一层 LayerNorm 后得到最终 patch-level 视觉表征。")

    return {
        "summary_records": [asdict(r) for r in records],
        "block_shapes": block_shapes,
        "final_features_shape": shape_list(final_hidden),
        "token_grid": {
            "temporal_tokens": t_tokens,
            "height_tokens": h_tokens,
            "width_tokens": w_tokens,
            "total_tokens": n_tokens,
        },
    }


def write_html(
    output_html: str,
    title: str,
    video_path: str,
    preview_dir: str,
    report: dict[str, Any],
) -> None:
    rows = []
    for item in report["preprocess_records"] + report["encoder_records"]:
        rows.append(
            f"""
            <tr>
              <td>{item['step']}</td>
              <td>{item['shape']}</td>
              <td>{item['dtype']}</td>
              <td>{item['explanation_zh']}</td>
            </tr>
            """
        )
    block_rows = []
    for item in report["block_shapes"]:
        block_rows.append(
            f"""
            <tr>
              <td>{item['block_index']}</td>
              <td>{item['shape']}</td>
              <td>{item['dtype']}</td>
              <td>{item['explanation_zh']}</td>
            </tr>
            """
        )

    preview_imgs = []
    for name in sorted(os.listdir(preview_dir)):
        if name.endswith(".jpg"):
            preview_imgs.append(
                f'<figure><img src="preview_frames/{name}" alt="{name}"><figcaption>{name}</figcaption></figure>'
            )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: #fffdf8;
      --ink: #16202a;
      --muted: #56616b;
      --line: #d8cfc3;
      --accent: #c25b2a;
      --accent-soft: #f4d8ca;
    }}
    body {{
      margin: 0;
      font-family: "Source Han Sans SC", "Noto Sans CJK SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #f8d9c7 0, transparent 28%),
        linear-gradient(180deg, #f7f1e9 0%, #efe7dc 100%);
    }}
    main {{
      width: min(1200px, calc(100vw - 32px));
      margin: 24px auto 48px;
    }}
    section {{
      background: rgba(255, 253, 248, 0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
      margin-bottom: 18px;
      box-shadow: 0 12px 30px rgba(22, 32, 42, 0.07);
      backdrop-filter: blur(8px);
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    p, li {{
      color: var(--muted);
      line-height: 1.65;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 12px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      background: var(--accent-soft);
      color: var(--ink);
    }}
    code {{
      background: #f3ece5;
      padding: 2px 6px;
      border-radius: 6px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px;
    }}
    figure {{
      margin: 0;
    }}
    img {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      display: block;
    }}
    .pill {{
      display: inline-block;
      background: var(--accent-soft);
      color: var(--accent);
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 13px;
      margin-right: 8px;
      margin-bottom: 8px;
    }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>{title}</h1>
      <p>输入视频：<code>{video_path}</code></p>
      <span class="pill">采样帧数：{report['num_frames']}</span>
      <span class="pill">裁剪尺寸：{report['crop_size']}</span>
      <span class="pill">最终 token 数：{report['final_features_shape'][1]}</span>
      <span class="pill">隐藏维：{report['final_features_shape'][2]}</span>
      <p>{report['summary_zh']}</p>
    </section>
    <section>
      <h2>视频概况</h2>
      <p>原视频尺寸 {report['video_meta']['width']}x{report['video_meta']['height']}，共 {report['video_meta']['frame_count']} 帧，FPS 为 {report['video_meta']['fps']}。</p>
      <p>当前示例从 82 帧里等间隔采样 64 帧，因此没有补帧；只是轻微跳帧压缩时间轴。</p>
      <p>采样索引：<code>{report['frame_indices']}</code></p>
    </section>
    <section>
      <h2>每一步 Shape 变化</h2>
      <table>
        <thead>
          <tr><th>步骤</th><th>Shape</th><th>DType</th><th>中文解释</th></tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </section>
    <section>
      <h2>Transformer Block Shape</h2>
      <table>
        <thead>
          <tr><th>Block</th><th>Shape</th><th>DType</th><th>中文解释</th></tr>
        </thead>
        <tbody>
          {''.join(block_rows)}
        </tbody>
      </table>
    </section>
    <section>
      <h2>关键理解</h2>
      <p><code>[1, 3, 64, 384, 384]</code> 进入模型后，经 3D patch embedding 变成 <code>[1, 18432, 1408]</code>。</p>
      <p>原因是时间维用 <code>tubelet_size=2</code> 压成 <code>64 / 2 = 32</code> 个时间块，空间维用 <code>patch_size=16</code> 压成 <code>384 / 16 = 24</code>，所以 token 总数为 <code>32 x 24 x 24 = 18432</code>。</p>
      <p>后续 40 层 ViT block 不再改变 token 数，主要是在 18432 个时空 token 上做特征混合，最后输出每个 token 的 1408 维表示。</p>
    </section>
    <section>
      <h2>采样帧预览</h2>
      <div class="grid">
        {''.join(preview_imgs)}
      </div>
    </section>
  </main>
</body>
</html>
"""
    Path(output_html).write_text(html, encoding="utf-8")


def save_preview_frames(video_thwc: np.ndarray, frame_idx: np.ndarray, preview_dir: str) -> None:
    Path(preview_dir).mkdir(parents=True, exist_ok=True)
    preview_positions = np.linspace(0, len(frame_idx) - 1, 8).round().astype(int)
    for pos in preview_positions:
        frame = video_thwc[pos]
        out_path = Path(preview_dir) / f"sample_{pos:02d}_srcidx_{int(frame_idx[pos]):03d}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def read_local_hf_config(model_dir: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    proc = json.loads((Path(model_dir) / "video_preprocessor_config.json").read_text(encoding="utf-8"))
    return cfg, proc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--hf-model-dir", default="/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384")
    parser.add_argument("--output-dir", default="analysis_outputs/bear_vjepa2")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "preview_frames"
    report_json = out_dir / "report.json"
    report_html = out_dir / "index.html"

    video_meta = get_video_meta(args.video_path)
    hf_cfg, proc_cfg = read_local_hf_config(args.hf_model_dir)
    crop_size = int(proc_cfg["crop_size"]["height"])
    num_frames = int(hf_cfg.get("frames_per_clip", 64))
    patch_size = int(hf_cfg["patch_size"])
    tubelet_size = int(hf_cfg["tubelet_size"])
    hidden_size = int(hf_cfg["hidden_size"])
    num_hidden_layers = int(hf_cfg["num_hidden_layers"])

    video_thwc, frame_idx = load_sampled_video(args.video_path, num_frames=num_frames)
    save_preview_frames(video_thwc, frame_idx, str(preview_dir))

    x_bcthw, preprocess_records = manual_preprocess(video_thwc, crop_size=crop_size)
    encoder_result = infer_encoder_shapes(
        num_frames=num_frames,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
    )

    report = {
        "video_path": args.video_path,
        "video_meta": video_meta,
        "frame_indices": frame_idx.tolist(),
        "crop_size": crop_size,
        "num_frames": num_frames,
        "patch_size": patch_size,
        "tubelet_size": tubelet_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "preprocess_records": [asdict(r) for r in preprocess_records],
        "encoder_records": encoder_result["summary_records"],
        "block_shapes": encoder_result["block_shapes"],
        "final_features_shape": encoder_result["final_features_shape"],
        "hf_features_shape": encoder_result["final_features_shape"],
        "token_grid": encoder_result["token_grid"],
        "summary_zh": (
            "这个示例对应 V-JEPA 2 的 ViT-g/16 384 版本。输入视频先被标准化为 64 帧、"
            "384x384 的张量，再通过 3D patch embedding 压成 18432 个时空 token，"
            "每个 token 的维度是 1408，随后 40 层 Transformer 在不改变 token 数的前提下完成特征建模。"
        ),
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(
        output_html=str(report_html),
        title="V-JEPA 2 视频 Shape 分析",
        video_path=args.video_path,
        preview_dir=str(preview_dir),
        report=report,
    )
    print(json.dumps({"report_json": str(report_json), "report_html": str(report_html)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
