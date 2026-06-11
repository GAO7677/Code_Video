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

from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackAdapter
from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.track_supervision import align_tracks_to_boxes


def tensor_frame_to_pil(frame_chw: torch.Tensor) -> Image.Image:
    x = frame_chw.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).contiguous()
    return Image.fromarray(x.numpy())


def pil_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def track_inside_box(point_xy: torch.Tensor, box_xyxy: torch.Tensor, image_hw: tuple[int, int]) -> bool:
    width = image_hw[1]
    height = image_hw[0]
    x0 = float(box_xyxy[0].item()) * width
    y0 = float(box_xyxy[1].item()) * height
    x1 = float(box_xyxy[2].item()) * width
    y1 = float(box_xyxy[3].item()) * height
    x = float(point_xy[0].item())
    y = float(point_xy[1].item())
    return x0 <= x <= x1 and y0 <= y <= y1


def draw_case_frame(
    frame_chw: torch.Tensor,
    gt_boxes_k4: torch.Tensor,
    matched_gt_idx_k: torch.Tensor,
    tracks_xy_k2: torch.Tensor,
    vis_k: torch.Tensor,
) -> Image.Image:
    image = tensor_frame_to_pil(frame_chw)
    out = image.copy()
    draw = ImageDraw.Draw(out)
    colors = ["#d62828", "#f77f00", "#fcbf49", "#2a9d8f", "#277da1", "#6a4c93"]
    width, height = out.size

    for obj_idx, box in enumerate(gt_boxes_k4.tolist()):
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle([x0 * width, y0 * height, x1 * width, y1 * height], outline="#ffffff", width=2)
        draw.text((x0 * width + 2, y0 * height + 2), f"gt{obj_idx}", fill="#ffffff")

    for query_idx, point in enumerate(tracks_xy_k2.tolist()):
        color = colors[query_idx % len(colors)]
        x, y = point
        r = 5
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
        gt_idx = int(matched_gt_idx_k[query_idx].item())
        label = f"q{query_idx}->gt{gt_idx}"
        if float(vis_k[query_idx].item()) < 0.5:
            label += "(inv)"
        draw.text((x + 6, y - 6), label, fill=color)
    return out


def evaluate_sample(
    sample: dict,
    adapter: VGGTTrackAdapter,
    device: torch.device,
) -> dict:
    context_video = sample["context_video"].unsqueeze(0).to(device)
    context_boxes = sample["context_boxes"].unsqueeze(0).to(device)

    frames_bthwc = context_video.permute(0, 2, 3, 4, 1).float()
    frames_bthwc = (frames_bthwc + 1.0) / 2.0
    with torch.no_grad():
        vggt_out = adapter(frames_bthwc)

    tracks = vggt_out.tracks
    vis = vggt_out.visibility
    conf = vggt_out.confidence
    track_image_hw = vggt_out.image_hw

    scale_x = float(context_video.shape[-1]) / float(track_image_hw[1])
    scale_y = float(context_video.shape[-2]) / float(track_image_hw[0])
    tracks_native = tracks.clone()
    tracks_native[..., 0] *= scale_x
    tracks_native[..., 1] *= scale_y

    alignment = align_tracks_to_boxes(
        tracks=tracks_native,
        gt_boxes=context_boxes,
        image_hw=(context_video.shape[-2], context_video.shape[-1]),
    )

    valid_mask = alignment.matched_gt_valid > 0.5
    l1 = (tracks_native - alignment.matched_gt_centers).abs().sum(dim=-1)
    mean_center_l1 = float(l1[valid_mask].mean().item()) if valid_mask.any() else 0.0

    inside_hits = []
    for t in range(tracks_native.shape[1]):
        for q in range(tracks_native.shape[2]):
            if not bool(valid_mask[0, t, q].item()):
                continue
            gt_idx = int(alignment.matched_gt_indices[0, q].item())
            hit = track_inside_box(
                point_xy=tracks_native[0, t, q],
                box_xyxy=context_boxes[0, t, gt_idx],
                image_hw=(context_video.shape[-2], context_video.shape[-1]),
            )
            inside_hits.append(float(hit))
    inside_rate = float(sum(inside_hits) / len(inside_hits)) if inside_hits else 0.0

    frame_cards = []
    context_frames = sample["context_video"].permute(1, 0, 2, 3)
    for t in range(context_frames.shape[0]):
        img = draw_case_frame(
            frame_chw=context_frames[t],
            gt_boxes_k4=sample["context_boxes"][t],
            matched_gt_idx_k=alignment.matched_gt_indices[0].cpu(),
            tracks_xy_k2=tracks_native[0, t].cpu(),
            vis_k=vis[0, t].cpu(),
        )
        frame_cards.append(pil_to_data_url(img))

    return {
        "caption": sample["caption"],
        "video_path": sample["video_path"],
        "context_frame_indices": sample["context_frame_indices"].tolist(),
        "track_image_hw": list(track_image_hw),
        "vggt_used_model": bool(vggt_out.used_model),
        "shapes": {
            "context_video": list(context_video.shape),
            "context_boxes": list(context_boxes.shape),
            "vggt_query_points": list(vggt_out.query_points.shape),
            "tracks": list(tracks.shape),
            "tracks_native_xy": list(tracks_native.shape),
            "visibility": list(vis.shape),
            "confidence": list(conf.shape),
            "matched_gt_indices": list(alignment.matched_gt_indices.shape),
            "matched_gt_centers": list(alignment.matched_gt_centers.shape),
            "matched_gt_valid": list(alignment.matched_gt_valid.shape),
            "track_pair_cost": list(alignment.pair_cost.shape),
        },
        "metrics": {
            "mean_center_l1_px": mean_center_l1,
            "inside_box_rate": inside_rate,
            "valid_track_points": int(valid_mask.sum().item()),
        },
        "matched_gt_indices": alignment.matched_gt_indices[0].tolist(),
        "frame_images": frame_cards,
    }


def build_report(results: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "num_cases": len(results),
        "avg_mean_center_l1_px": sum(r["metrics"]["mean_center_l1_px"] for r in results) / max(len(results), 1),
        "avg_inside_box_rate": sum(r["metrics"]["inside_box_rate"] for r in results) / max(len(results), 1),
        "cases": [
            {
                "case_id": idx,
                "video_path": r["video_path"],
                "caption": r["caption"],
                "metrics": r["metrics"],
            }
            for idx, r in enumerate(results)
        ],
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    blocks = []
    for idx, result in enumerate(results):
        blocks.append(
            f"""
  <section class=\"case\">
    <h2>Case {idx}</h2>
    <p><b>Caption:</b> {result['caption']}</p>
    <p><b>Context frames:</b> {result['context_frame_indices']}</p>
    <p><b>Metrics:</b> mean_center_l1_px={result['metrics']['mean_center_l1_px']:.2f}, inside_box_rate={result['metrics']['inside_box_rate']:.3f}, valid_track_points={result['metrics']['valid_track_points']}</p>
    <p><b>Matched gt indices:</b> {result['matched_gt_indices']}</p>
    <div class=\"grid\">
      {''.join(f'<figure><img src="{src}" /><figcaption>t={t}</figcaption></figure>' for src, t in zip(result['frame_images'], result['context_frame_indices']))}
    </div>
    <pre>{json.dumps({'metrics': result['metrics'], 'shapes': result['shapes'], 'track_image_hw': result['track_image_hw'], 'vggt_used_model': result['vggt_used_model']}, indent=2, ensure_ascii=False)}</pre>
  </section>
"""
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\">
  <title>VGGT vs Box Supervision</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #f6f4ee; color: #222; }}
    .case {{ margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid #ddd; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
    .grid img {{ width: 100%; border: 1px solid #ccc; background: #fff; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 12px; color: #444; margin-top: 4px; }}
    pre {{ background: #fff; border: 1px solid #ddd; padding: 16px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>VGGT vs Box Supervision</h1>
  <p>当前页面验证的是：给定 context video，先用 VGGT 的 query-point tracking 产生 tracks，再和数据集 gt boxes 做匹配监督。白框是 gt box；彩色圆点是 VGGT track 点；标签 `q&lt;i&gt;-&gt;gt&lt;j&gt;` 表示该 query 当前匹配到哪个 gt object。</p>
  <p><b>Overall:</b> avg_mean_center_l1_px={summary['avg_mean_center_l1_px']:.2f}, avg_inside_box_rate={summary['avg_inside_box_rate']:.3f}, num_cases={summary['num_cases']}</p>
  {''.join(blocks)}
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
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/inspect_phys_state_vjepa_vggt.yaml",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/outputs/vggt_box_eval_viewer",
    )
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PhysStateEpisodeDataset(
        root=data_cfg["root"],
        split=args.split,
        resolution=tuple(data_cfg["resolution"]),
        num_context_frames=int(data_cfg["num_context_frames"]),
        context_fraction=float(data_cfg.get("context_fraction", 0.5)),
        random_context_frames=bool(data_cfg.get("random_context_frames", True)),
        seed=int(cfg.get("experiment", {}).get("seed", 42)),
    )

    adapter = VGGTTrackAdapter(
        model_path=model_cfg.get("vggt_model_path"),
        num_queries=int(model_cfg["object_num_queries"]),
        device=str(device),
        input_hw=tuple(model_cfg["vggt_input_hw"]),
    ).to(device)

    results = []
    for idx in range(args.start_index, min(len(dataset), args.start_index + args.num_cases)):
        sample = dataset[idx]
        results.append(evaluate_sample(sample, adapter, device))

    output_dir = Path(args.output_dir)
    html_path = build_report(results, output_dir)
    print(f"eval report: {html_path}")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(output_dir), **handler_kwargs)

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("0.0.0.0", args.port), Handler) as httpd:
        print(f"serving report at http://0.0.0.0:{args.port}/index.html")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
