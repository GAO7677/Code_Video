#!/usr/bin/env python3
"""Render several YTVIS-HQ validation cases from a trained recognition probe."""

from argparse import ArgumentParser
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from visualize_vjepa_xssc_downstream_one_train_step import (
    ROOT,
    add_header,
    checkpoint_load_summary,
    class_text,
    decode_video,
    draw_gt,
    fit_width,
    load_matching_checkpoint,
    render_output,
    set_seed,
    snapshot,
    to_device,
    write_h264,
)


DEFAULT_CONFIG = ROOT / (
    "upstream/config-randsfq/"
    "rsfq2_r_recogn-ytvis_hq-vjepa2_1_vitl16_256-video-"
    "slot512-step7000-bs8-steps5000-gpu7.py"
)
DEFAULT_CHECKPOINT = Path(
    "/data/gaoya/agent-data/checkpoints/"
    "xssc_vjepa2_1_downstream_recognition_step7000_bs8_steps5000_gpu7/"
    "rsfq2_r_recogn-ytvis_hq-vjepa2_1_vitl16_256-video-slot512-"
    "step7000-bs8-steps5000-gpu7/42/step-005000.pth"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/vjepa_xssc_downstream_val_viewer"
)
WANDB_URL = (
    "https://wandb.ai/875222004-gy/xssc_vjepa2_1_downstream/"
    "runs/6kl8tt53"
)
RUN_SUMMARY = {
    "state": "finished",
    "optimizer_steps": 5000,
    "runtime_seconds": 1653,
    "batch_size_per_gpu": 8,
    "effective_global_batch_size": 8,
    "source_xssc_checkpoint_step": 7000,
    "val_ce": 0.9417189359664916,
    "val_l1": 0.11765022575855257,
    "val_top1": 0.7499239444732666,
    "val_top3": 0.9242470264434814,
    "val_iou": 0.5148244500160217,
    "val_num_matches": 3287,
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--cfg-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/gaoya/dataset"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sample-indices", type=int, nargs="+", default=[0, 47, 94, 141, 188, 235]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=3.0)
    return parser.parse_args()


def box_iou(left, right):
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    intersection_lt = np.maximum(left[:2], right[:2])
    intersection_rb = np.minimum(left[2:], right[2:])
    intersection_wh = np.maximum(intersection_rb - intersection_lt, 0)
    intersection = float(np.prod(intersection_wh))
    left_area = float(np.prod(np.maximum(left[2:] - left[:2], 0)))
    right_area = float(np.prod(np.maximum(right[2:] - right[:2], 0)))
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def case_metrics(result):
    records = [record for frame in result["matches"] for record in frame]
    if not records:
        return {
            "num_matches": 0,
            "top1": None,
            "mean_iou": None,
            "mean_confidence": None,
        }
    return {
        "num_matches": len(records),
        "top1": float(
            np.mean(
                [record["pred_class"] == record["gt_class"] for record in records]
            )
        ),
        "mean_iou": float(
            np.mean(
                [
                    box_iou(
                        record["pred_box_ltrb_raw"], record["gt_box_ltrb"]
                    )
                    for record in records
                ]
            )
        ),
        "mean_confidence": float(
            np.mean([record["confidence"] for record in records])
        ),
    }


def render_case(output_dir, sample_index, batch_cpu, result, fps):
    case_dir = output_dir / f"case-{sample_index:04d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    raw_video = decode_video(batch_cpu["video"])
    label_indices = list(range(1, len(raw_video), 2))
    if len(label_indices) != len(result["segment"]):
        raise RuntimeError(
            f"tubelet alignment mismatch for sample {sample_index}: "
            f"raw={len(raw_video)}, labels={len(label_indices)}, "
            f"pred={len(result['segment'])}"
        )
    aligned = raw_video[label_indices]
    gt_segment = batch_cpu["segment"][0].cpu().numpy()
    gt_boxes = batch_cpu["bbox"][0].cpu().numpy()
    gt_classes = batch_cpu["clazz"][0].cpu().numpy()

    frames = []
    for time_index, raw_index in enumerate(label_indices):
        source = add_header(
            aligned[time_index],
            ["INPUT", f"raw frame {raw_index} · V-JEPA tubelet {time_index}"],
            color=(121, 192, 255),
        )
        gt = add_header(
            draw_gt(
                aligned[time_index],
                gt_segment[time_index],
                gt_boxes[time_index],
                gt_classes[time_index],
            ),
            ["GT", "true mask + LTRB box + class label"],
            color=(126, 231, 135),
        )
        pred_frame, pred_text = render_output(
            aligned[time_index],
            result["segment"][time_index],
            result["matches"][time_index],
            box_color=(255, 196, 107),
        )
        pred = add_header(
            pred_frame,
            ["PRED", pred_text],
            color=(255, 196, 107),
        )
        width = max(panel.shape[1] for panel in (source, gt, pred))
        frames.append(
            np.concatenate([fit_width(panel, width) for panel in (source, gt, pred)], axis=1)
        )

    video_file = case_dir / "comparison.mp4"
    write_h264(video_file, frames, fps)
    Image.fromarray(frames[0]).save(case_dir / "poster.jpg", quality=90)
    metrics = case_metrics(result)
    metadata = {
        "sample_index": sample_index,
        "raw_frame_count": len(raw_video),
        "tubelet_target_frame_count": len(label_indices),
        "target_raw_frame_indices": label_indices,
        "loss": result["loss"],
        "metrics": metrics,
        "matches_per_frame": [len(records) for records in result["matches"]],
        "detections": result["matches"],
        "video": f"case-{sample_index:04d}/comparison.mp4",
        "poster": f"case-{sample_index:04d}/poster.jpg",
    }
    (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def format_metric(value, percent=False):
    if value is None:
        return "—"
    return f"{value * 100:.2f}%" if percent else f"{value:.4f}"


def render_page(report):
    options = "".join(
        f'<option value="{index}">val case {case["sample_index"]}</option>'
        for index, case in enumerate(report["cases"])
    )
    case_payload = html.escape(json.dumps(report["cases"]), quote=False)
    run = report["run"]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V-JEPA xSSC downstream validation</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#0d1117;color:#e6edf3;font:14px system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:3;padding:13px 22px;border-bottom:1px solid #30363d;background:rgba(13,17,23,.96)}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px;margin:0 0 12px}} .sub{{color:#8b949e}}
main{{max-width:1500px;margin:auto;padding:18px}} .cards{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:16px}}
.card{{padding:14px;border:1px solid #30363d;border-radius:8px;background:#161b22;min-width:0}}
.metric{{font-size:22px;font-weight:700;color:#79c0ff}} .toolbar{{display:flex;gap:10px;align-items:center;margin-bottom:14px}}
select,button{{border:1px solid #388bfd;border-radius:7px;padding:9px 12px;color:white;background:#161b22}}
button{{background:#1f6feb;font-weight:650;cursor:pointer}} video{{display:block;width:100%;background:#000;border-radius:6px}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;margin-top:12px}} th,td{{padding:8px 10px;border-bottom:1px solid #30363d;text-align:right}} th:first-child,td:first-child{{text-align:left}}
a{{color:#79c0ff}} code{{color:#a5d6ff;overflow-wrap:anywhere}} .fixed{{position:fixed;right:20px;bottom:20px;z-index:5}}
@media(max-width:900px){{.cards{{grid-template-columns:1fr 1fr}} main{{padding:10px}}}}
</style></head><body>
<header><h1>V-JEPA xSSC 下游验证可视化</h1><div class="sub">最终 recognition checkpoint · 非因果 xSSC step-7000 · YTVIS-HQ val</div></header>
<main>
<section class="cards">
<div class="card"><div class="sub">Run</div><div class="metric">FINISHED</div><div>5000/5000 steps · 27m33s</div></div>
<div class="card"><div class="sub">Final val Top-1</div><div class="metric">{run['val_top1']*100:.2f}%</div><div>Top-3 {run['val_top3']*100:.2f}%</div></div>
<div class="card"><div class="sub">Final val IoU</div><div class="metric">{run['val_iou']:.4f}</div><div>{run['val_num_matches']} matches</div></div>
<div class="card"><div class="sub">Final val CE</div><div class="metric">{run['val_ce']:.4f}</div><div>classification</div></div>
<div class="card"><div class="sub">Final val L1</div><div class="metric">{run['val_l1']:.4f}</div><div>box regression</div></div>
</section>
<section class="card"><div class="toolbar"><label for="case-select">选择验证 case：</label><select id="case-select">{options}</select><a href="{html.escape(WANDB_URL)}" target="_blank" rel="noreferrer">打开 W&B run</a></div>
<h2 id="case-title"></h2><video id="viewer" controls muted loop autoplay playsinline></video>
<table><thead><tr><th>Case</th><th>CE</th><th>L1</th><th>匹配数</th><th>Top-1</th><th>Mean IoU</th><th>置信度</th><th>raw / target 帧</th></tr></thead><tbody id="case-row"></tbody></table>
<p class="sub">Pred 只展示官方 IoU&gt;0.1 匹配到 GT 的 slots；这不是对未匹配 slot 的检测阈值筛选。V-JEPA tubelet=2，因此目标对应 raw frame 1,3,5,…。</p>
<p class="sub">Final checkpoint: <code>{html.escape(report['checkpoint'])}</code></p></section>
</main><button class="fixed" id="replay">全部重新播放</button>
<script>
const cases={case_payload}; const select=document.getElementById('case-select'); const video=document.getElementById('viewer');
function f(v,p=false){{if(v===null)return '—';return p?(v*100).toFixed(2)+'%':Number(v).toFixed(4)}}
function show(){{const c=cases[Number(select.value)];document.getElementById('case-title').textContent='val case '+c.sample_index+' · 输入 / GT / Pred';video.src=c.video;video.poster=c.poster;video.play();document.getElementById('case-row').innerHTML=`<tr><td>${{c.sample_index}}</td><td>${{f(c.loss.ce)}}</td><td>${{f(c.loss.l1)}}</td><td>${{c.metrics.num_matches}}</td><td>${{f(c.metrics.top1,true)}}</td><td>${{f(c.metrics.mean_iou)}}</td><td>${{f(c.metrics.mean_confidence)}}</td><td>${{c.raw_frame_count}} / ${{c.tubelet_target_frame_count}}</td></tr>`}}
select.onchange=show;document.getElementById('replay').onclick=()=>{{video.currentTime=0;video.play()}};show();
</script></body></html>"""


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from object_centric_bench.learn import MetricWrap
    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    config_file = args.cfg_file.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(config_file)
    cfg.dataset_v.base_dir = args.data_dir.resolve()
    dataset = build_from_config(cfg.dataset_v)
    collate = build_from_config(cfg.collate_fn_v)
    for sample_index in args.sample_indices:
        if not 0 <= sample_index < len(dataset):
            raise IndexError(f"sample index {sample_index} outside [0, {len(dataset)})")

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    device = torch.device("cuda", 0)

    model = ModelWrap(build_from_config(cfg.model), cfg.model_imap, cfg.model_omap)
    load_report = load_matching_checkpoint(
        model,
        checkpoint,
        exclude_patterns=(),
        allowed_missing_patterns=cfg.checkpoint_allowed_missing,
        expected_source_variant=cfg.variant_name,
        expected_source_step=RUN_SUMMARY["optimizer_steps"],
    )
    model.freez(cfg.freez, verbose=False)
    model = model.to(device).eval()
    loss_fn = MetricWrap(**build_from_config(cfg.loss_fn_v)).to(device)
    amp_dtype = getattr(torch, cfg.amp_dtype)

    cases = []
    for ordinal, sample_index in enumerate(args.sample_indices):
        print(f"[val-case] {ordinal + 1}/{len(args.sample_indices)} index={sample_index}", flush=True)
        batch_cpu = collate([dataset[sample_index]])
        batch = to_device(batch_cpu, device)
        set_seed(args.seed + sample_index)
        result = snapshot(model, loss_fn, batch, amp_dtype)
        cases.append(render_case(output_dir, sample_index, batch_cpu, result, args.fps))

    report = {
        "wandb_url": WANDB_URL,
        "run": RUN_SUMMARY,
        "config_file": str(config_file),
        "checkpoint": str(checkpoint),
        "checkpoint_load": checkpoint_load_summary(load_report),
        "dataset_size": len(dataset),
        "seed_policy": "base seed 42 + validation sample index",
        "cases": cases,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "index.html").write_text(render_page(report))
    print(json.dumps({"output": str(output_dir), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
