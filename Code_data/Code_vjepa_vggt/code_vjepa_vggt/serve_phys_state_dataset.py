from __future__ import annotations

import argparse
import base64
import http.server
import io
import json
import socketserver
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset


def tensor_frame_to_pil(frame_chw: torch.Tensor) -> Image.Image:
    x = frame_chw.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).contiguous()
    return Image.fromarray(x.numpy())


def draw_boxes(image: Image.Image, boxes_k4: torch.Tensor) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    colors = ["#d62828", "#f77f00", "#fcbf49", "#2a9d8f", "#277da1", "#6a4c93"]
    for idx, box in enumerate(boxes_k4.tolist()):
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle(
            [x0 * width, y0 * height, x1 * width, y1 * height],
            outline=colors[idx % len(colors)],
            width=3,
        )
        draw.text((x0 * width + 2, y0 * height + 2), str(idx), fill=colors[idx % len(colors)])
    return out


def pil_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_case_block(sample: dict, case_id: int) -> str:
    context_video = sample["context_video"].permute(1, 0, 2, 3)
    context_boxes = sample["context_boxes"]
    context_images = []
    for i in range(context_video.shape[0]):
        img = tensor_frame_to_pil(context_video[i])
        img = draw_boxes(img, context_boxes[i])
        context_images.append(pil_to_data_url(img))

    payload = {
        "caption": sample["caption"],
        "video_shape": list(sample["video"].shape),
        "context_video_shape": list(sample["context_video"].shape),
        "context_boxes_shape": list(sample["context_boxes"].shape),
        "future_boxes_shape": list(sample["future_boxes"].shape),
        "context_states_shape": list(sample["context_states"].shape),
        "future_states_shape": list(sample["future_states"].shape),
        "appearance_shape": list(sample["appearance"].shape),
        "camera_shape": list(sample["camera"].shape),
        "video_path": sample["video_path"],
        "all_frame_indices": sample["frame_indices"].tolist(),
        "context_frame_indices": sample["context_frame_indices"].tolist(),
        "metadata": sample["metadata"],
    }
    return f"""
  <section class="case">
    <h2>Case {case_id}</h2>
    <p><b>Caption:</b> {sample["caption"]}</p>
    <p><b>Context sampled frames:</b> {sample["context_frame_indices"].tolist()}</p>
    <div class="grid">
      {"".join(f'<figure><img src="{src}" /><figcaption>t={t}</figcaption></figure>' for src, t in zip(context_images, sample["context_frame_indices"].tolist()))}
    </div>
    <pre>{json.dumps(payload, indent=2, ensure_ascii=False)}</pre>
  </section>
"""


def build_report(samples: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    blocks = []
    for case_id, sample in enumerate(samples):
        summary.append(
            {
                "case_id": case_id,
                "video_path": sample["video_path"],
                "caption": sample["caption"],
                "context_frame_indices": sample["context_frame_indices"].tolist(),
            }
        )
        blocks.append(render_case_block(sample, case_id))

    with open(output_dir / "shape_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>phys_state_0601 context sampler viewer</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #f6f4ee; color: #222; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    .case {{ margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid #ddd; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
    .grid img {{ width: 100%; border: 1px solid #ccc; background: #fff; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 12px; color: #444; margin-top: 4px; }}
    pre {{ background: #fff; border: 1px solid #ddd; padding: 16px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>phys_state_0601 Context Sampler Viewer</h1>
  <p>当前可视化展示：context 不再固定为前 8 帧，而是从整段视频前 50% 时间范围内采样任意 8 帧。下面展示多个 case 的采样结果和 object boxes。</p>
  {"".join(blocks)}
</body>
</html>
"""
    html_path = output_dir / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--num-context-frames", type=int, default=8)
    parser.add_argument("--context-fraction", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/outputs/phys_state_dataset_viewer",
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    dataset = PhysStateEpisodeDataset(
        root=args.root,
        split=args.split,
        resolution=(144, 256),
        num_context_frames=args.num_context_frames,
        context_fraction=args.context_fraction,
        random_context_frames=True,
    )
    samples = [dataset[i] for i in range(args.start_index, min(len(dataset), args.start_index + args.num_cases))]
    output_dir = Path(args.output_dir)
    html_path = build_report(samples, output_dir)
    print(f"dataset report: {html_path}")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(output_dir), **handler_kwargs)

    with socketserver.TCPServer(("0.0.0.0", args.port), Handler) as httpd:
        print(f"serving report at http://0.0.0.0:{args.port}/index.html")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
