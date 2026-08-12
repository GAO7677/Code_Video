#!/usr/bin/env python3
"""Compare two V-JEPA xSSC checkpoints on complete short test_5 videos."""

from argparse import ArgumentParser
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch

from infer_vjepa_xssc_video_slot_overlay import (
    SLOT_COLORS,
    add_header,
    add_legend,
    checkpoint_load_summary,
    fit_width,
    load_model,
    make_contact_sheet,
    normalize_frames,
    render_overlay,
    set_seed,
    upsample_labels,
)
from visualize_vjepa_xssc_test5_source_contact_sheets import (
    DEFAULT_TEST5,
    decode_and_sample,
    load_unique_cases,
)


ROOT = Path(__file__).resolve().parent
OLD_CONFIG = ROOT / (
    "upstream/config-randsfq/"
    "rsfq2_r-ytvis_hq-vjepa2_1_vitl16_256-video-slot512.py"
)
OLD_CHECKPOINT = Path(
    "/data/gaoya/agent-data/checkpoints/"
    "xssc_vjepa2_1_video_noncausal_ytvis_hq_bs64_steps10000/"
    "rsfq2_r-ytvis_hq-vjepa2_1_vitl16_256-video-slot512/42/step-010000.pth"
)
NEW_CONFIG = ROOT / (
    "upstream/config-randsfq/"
    "rsfq2_r-ytvis_hq-vjepa2_1_vitl16-ar10f-slot512-transfer10000-bs64.py"
)
NEW_CHECKPOINT = Path(
    "/data/gaoya/agent-data/checkpoints/"
    "xssc_vjepa2_1_video_noncausal_ytvis_hq_10f_ar_bs64_steps20000/"
    "rsfq2_r-ytvis_hq-vjepa2_1_vitl16-ar10f-slot512-transfer10000-bs64/"
    "42/step-013000.pth"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/vjepa_xssc_test5_step10000_vs_step13000"
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--test5-file", type=Path, default=DEFAULT_TEST5)
    parser.add_argument("--old-config", type=Path, default=OLD_CONFIG)
    parser.add_argument("--old-checkpoint", type=Path, default=OLD_CHECKPOINT)
    parser.add_argument("--new-config", type=Path, default=NEW_CONFIG)
    parser.add_argument("--new-checkpoint", type=Path, default=NEW_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-frame-count-lt", type=int, default=50)
    parser.add_argument("--resize-mode", choices=("center-crop", "padding"), default="padding")
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


@torch.inference_mode()
def infer_attention(model, video, device, amp_dtype):
    with torch.autocast("cuda", dtype=amp_dtype):
        output = model(batch={"video": video.to(device)})
    attention = output["attentd"]
    if attention.ndim != 5 or attention.shape[0] != 1:
        raise RuntimeError(f"expected [1,T,S,H,W], got {tuple(attention.shape)}")
    return attention[0].float().cpu().numpy()


def components(mask):
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            size = 0
            while stack:
                yy, xx = stack.pop()
                size += 1
                for y2, x2 in ((yy - 1, xx), (yy + 1, xx), (yy, xx - 1), (yy, xx + 1)):
                    if 0 <= y2 < height and 0 <= x2 < width and mask[y2, x2] and not seen[y2, x2]:
                        seen[y2, x2] = True
                        stack.append((y2, x2))
            sizes.append(size)
    return sizes


def optimal_remap(reference, candidate, slots):
    overlap = np.zeros((slots, slots), dtype=np.int64)
    for ref_slot in range(slots):
        for cand_slot in range(slots):
            overlap[ref_slot, cand_slot] = np.count_nonzero(
                (reference == ref_slot) & (candidate == cand_slot)
            )
    rows, cols = linear_sum_assignment(-overlap)
    mapping = np.arange(slots, dtype=np.uint8)
    for row, col in zip(rows, cols):
        mapping[col] = row
    return mapping[candidate], float(overlap[rows, cols].sum() / reference.size)


def partition_metrics(attention):
    # Decoder cross-attention is already positive attention mass, but it is
    # not normalized over slots after head aggregation. Convert it to a local
    # per-patch slot distribution without applying a second softmax.
    probabilities = np.clip(attention, 0.0, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True).clip(min=1e-12)
    labels = probabilities.argmax(axis=1).astype(np.uint8)
    slots = attention.shape[1]
    boundary_values = []
    component_values = []
    tiny_values = []
    effective_values = []
    dominant_values = []
    for label in labels:
        horizontal = np.count_nonzero(label[:, 1:] != label[:, :-1])
        vertical = np.count_nonzero(label[1:, :] != label[:-1, :])
        boundary_values.append((horizontal + vertical) / (label.shape[0] * (label.shape[1] - 1) + (label.shape[0] - 1) * label.shape[1]))
        counts = np.bincount(label.ravel(), minlength=slots).astype(np.float64)
        active = counts > 0
        area_prob = counts / counts.sum()
        effective_values.append(float(np.exp(-(area_prob[active] * np.log(area_prob[active])).sum())))
        dominant_values.append(float(area_prob.max()))
        slot_components = []
        tiny_pixels = 0
        for slot in np.flatnonzero(active):
            sizes = components(label == slot)
            slot_components.append(len(sizes))
            tiny_pixels += sum(size for size in sizes if size < 4)
        component_values.append(float(np.mean(slot_components)))
        tiny_values.append(tiny_pixels / label.size)
    temporal = []
    for left, right in zip(labels[:-1], labels[1:]):
        _, agreement = optimal_remap(left, right, slots)
        temporal.append(agreement)
    entropy = -(probabilities * np.log(probabilities.clip(min=1e-12))).sum(axis=1) / math.log(slots)
    return labels, {
        "mean_max_probability": float(probabilities.max(axis=1).mean()),
        "normalized_attention_entropy": float(entropy.mean()),
        "boundary_density": float(np.mean(boundary_values)),
        "components_per_active_slot": float(np.mean(component_values)),
        "tiny_component_pixel_ratio": float(np.mean(tiny_values)),
        "effective_slots": float(np.mean(effective_values)),
        "dominant_slot_area_ratio": float(np.mean(dominant_values)),
        "adjacent_tubelet_partition_agreement": float(np.mean(temporal)) if temporal else 1.0,
    }


def aggregate_metrics(cases, key):
    names = list(cases[0][key])
    return {name: float(np.mean([case[key][name] for case in cases])) for name in names}


def verdict(old, new):
    def relative_drop(key, threshold=0.05):
        return (old[key] - new[key]) / max(abs(old[key]), 1e-12) >= threshold

    favorable = [
        new["mean_max_probability"] - old["mean_max_probability"] >= 0.01,
        old["normalized_attention_entropy"] - new["normalized_attention_entropy"] >= 0.01,
        relative_drop("boundary_density"),
        relative_drop("components_per_active_slot"),
        relative_drop("tiny_component_pixel_ratio"),
        new["adjacent_tubelet_partition_agreement"] - old["adjacent_tubelet_partition_agreement"] >= 0.03,
    ]
    collapse = new["dominant_slot_area_ratio"] > 0.75 or (
        new["dominant_slot_area_ratio"] - old["dominant_slot_area_ratio"] > 0.15
    )
    count = sum(favorable)
    if count >= 4 and not collapse:
        return "这两例上的代理指标达到较一致改善，但仍需结合有 GT 的实例分割指标确认。"
    if count <= 2 or collapse:
        return "这两例上没有证据表明物体 slot 区分度明显改善；空间碎片仍然严重。"
    return "结果有改善也有退化，当前不能判定物体 slot 区分度明显变好。"


def render_case(case, frames, old_labels, new_labels, output_dir, columns, alpha, quality):
    panels = []
    aligned_new = []
    for tubelet in range(len(old_labels)):
        aligned, _ = optimal_remap(old_labels[tubelet], new_labels[tubelet], 7)
        aligned_new.append(aligned)
    old_per_frame = np.repeat(upsample_labels(old_labels, 256, 256), 2, axis=0)[: len(frames)]
    new_per_frame = np.repeat(upsample_labels(np.stack(aligned_new), 256, 256), 2, axis=0)[: len(frames)]
    for frame_index, (frame, old_label, new_label) in enumerate(zip(frames, old_per_frame, new_per_frame)):
        input_panel = add_header(frame, ["MODEL INPUT", f"source f{frame_index:03d} · complete consecutive input"], color=(121, 192, 255))
        old_panel = add_header(render_overlay(frame, old_label, alpha), ["OLD · STEP 10000", f"tubelet {frame_index // 2:02d} · local slot colors"], color=(255, 170, 120))
        new_panel = add_header(render_overlay(frame, new_label, alpha), ["LATEST · STEP 13000", f"tubelet {frame_index // 2:02d} · colors aligned to old"], color=(120, 230, 170))
        width = max(panel.shape[1] for panel in (input_panel, old_panel, new_panel))
        panels.append(add_legend(np.concatenate([fit_width(input_panel, width), fit_width(old_panel, width), fit_width(new_panel, width)], axis=1)))
    sheet, rows = make_contact_sheet(panels, columns)
    case_dir = output_dir / "cases" / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    sheet_file = case_dir / "comparison.webp"
    sheet.save(sheet_file, format="WEBP", quality=quality, method=6)
    return str(sheet_file.relative_to(output_dir)), rows, list(sheet.size)


def build_html(report):
    payload = json.dumps(report, separators=(",", ":"))
    metrics = [
        ("Soft max confidence", "mean_max_probability", "higher"),
        ("Soft normalized entropy", "normalized_attention_entropy", "lower"),
        ("Boundary density", "boundary_density", "lower"),
        ("Components / active slot", "components_per_active_slot", "lower"),
        ("Tiny-component pixel ratio", "tiny_component_pixel_ratio", "lower"),
        ("Adjacent-tubelet agreement", "adjacent_tubelet_partition_agreement", "higher"),
        ("Effective slots", "effective_slots", "context"),
        ("Dominant-slot area", "dominant_slot_area_ratio", "collapse check"),
    ]
    rows = "".join(
        f'<tr><td>{label}</td><td id="o-{key}"></td><td id="n-{key}"></td><td id="d-{key}"></td><td>{direction}</td></tr>'
        for label, key, direction in metrics
    )
    legend = "".join(f'<span><i style="background:rgb({r},{g},{b})"></i>S{s}</span>' for s, (r, g, b) in enumerate(SLOT_COLORS.tolist()))
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V-JEPA xSSC test_5 checkpoint comparison</title><style>
*{{box-sizing:border-box}}:root{{color-scheme:dark}}body{{margin:0;background:#0d1117;color:#e6edf3;font:14px system-ui,sans-serif}}header{{position:sticky;top:0;z-index:4;background:rgba(13,17,23,.97);border-bottom:1px solid #30363d}}.bar{{max-width:2400px;margin:auto;padding:12px 18px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}}h1{{font-size:19px;margin:0 auto 0 0}}select,button{{height:36px;border:1px solid #4b535d;border-radius:6px;background:#20262d;color:#f2f5f7;font:inherit}}select{{min-width:min(760px,75vw);padding:0 10px}}button{{padding:0 12px;cursor:pointer}}main{{max-width:2400px;margin:auto;padding:18px}}.grid{{display:grid;grid-template-columns:minmax(0,1fr) minmax(500px,.8fr);gap:14px;margin-bottom:14px}}.card{{border:1px solid #30363d;border-radius:8px;background:#161b22;padding:14px}}.meta{{display:grid;grid-template-columns:130px minmax(0,1fr);gap:7px 12px}}.key,.note{{color:#8b949e}}code{{color:#79c0ff;overflow-wrap:anywhere}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:right;padding:7px;border-bottom:1px solid #30363d}}th:first-child,td:first-child{{text-align:left}}.verdict{{font-size:16px;line-height:1.5;border-left:4px solid #d29922;padding:10px 12px;background:#1c1b16;margin-bottom:12px}}.legend{{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0}}.legend span{{display:flex;align-items:center;gap:5px}}.legend i{{width:14px;height:14px;border-radius:3px}}.sheet{{display:block;width:100%;height:auto;border:1px solid #30363d;border-radius:8px;background:#050607}}a{{color:#58a6ff}}@media(max-width:1050px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><header><div class="bar"><h1>V-JEPA xSSC · test_5 最新权重对比</h1><button onclick="location.href='../'">项目总览</button><button id="prev">‹</button><select id="case"></select><button id="next">›</button></div></header><main><div id="verdict" class="verdict"></div><div class="grid"><section class="card"><div id="meta" class="meta"></div></section><section class="card"><table><thead><tr><th>无 GT 代理指标</th><th>step-10000</th><th>step-13000</th><th>Δ new-old</th><th>方向</th></tr></thead><tbody>{rows}</tbody></table></section></div><div class="legend">{legend}</div><a id="sheetLink" target="_blank"><img id="sheet" class="sheet"></a><p class="note">每格从左到右：完整输入帧、旧 step-10000、最新 step-13000。新版本颜色按每个 tubelet 与旧分区做最优置换，仅便于视觉比较，不改变边界。上述指标没有使用物体 GT，只能衡量分区置信度、空间碎片和时间稳定性，不能替代 ARI-FG/mBO/mIoU。</p></main><script>
const D={payload}, keys={json.dumps([key for _, key, _ in metrics])};const sel=document.getElementById('case'),meta=document.getElementById('meta'),img=document.getElementById('sheet'),link=document.getElementById('sheetLink');D.cases.forEach((c,i)=>{{const o=document.createElement('option');o.value=i;o.textContent=`${{String(i+1).padStart(2,'0')}} | ${{c.case_id}} | ${{c.source_frame_count}}f`;sel.appendChild(o)}});function fmt(x){{return Number(x).toFixed(4)}}function show(){{const c=D.cases[Number(sel.value)];img.src=c.sheet;link.href=c.sheet;document.getElementById('verdict').textContent=D.assessment;meta.innerHTML=`<span class="key">Case</span><strong>${{c.case_id}}</strong><span class="key">Source</span><code>${{c.source_video}}</code><span class="key">Input</span><span>${{c.source_frame_count}}/${{c.source_frame_count}} consecutive frames · no sampling · padding 256²</span><span class="key">Old</span><code>step-10000 · 6f training · transition dt=3</code><span class="key">Latest</span><code>step-13000 · 10f training · transition dt=5</code>`;for(const k of keys){{document.getElementById('o-'+k).textContent=fmt(c.old_metrics[k]);document.getElementById('n-'+k).textContent=fmt(c.new_metrics[k]);document.getElementById('d-'+k).textContent=fmt(c.new_metrics[k]-c.old_metrics[k])}}}}sel.onchange=show;document.getElementById('prev').onclick=()=>{{sel.selectedIndex=(sel.selectedIndex-1+D.cases.length)%D.cases.length;show()}};document.getElementById('next').onclick=()=>{{sel.selectedIndex=(sel.selectedIndex+1)%D.cases.length;show()}};show();</script></body></html>'''


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    set_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    cases = load_unique_cases(args.test5_file.resolve())
    prepared = []
    for case in cases:
        frames, indices, count, shape, metadata = decode_and_sample(Path(case["source_video"]), 0, args.resize_mode, args.source_frame_count_lt)
        if frames is None:
            continue
        prepared.append({**case, "frames": frames, "indices": indices, "source_frame_count": count, "source_shape": shape, "source_fps": float(metadata.get("fps", 30.0))})
    if not prepared:
        raise RuntimeError("no matching source videos")

    results = {"old": {}, "new": {}}
    load_reports = {}
    for name, config, checkpoint in (("old", args.old_config, args.old_checkpoint), ("new", args.new_config, args.new_checkpoint)):
        cfg, model, load_report = load_model(config.resolve(), checkpoint.resolve(), device)
        load_reports[name] = checkpoint_load_summary(load_report)
        for case in prepared:
            video = normalize_frames(case["frames"])
            attention = infer_attention(model, video, device, getattr(torch, cfg.amp_dtype))
            labels, metrics = partition_metrics(attention)
            results[name][case["case_id"]] = {"labels": labels, "metrics": metrics, "attention_shape": list(attention.shape)}
        del model
        torch.cuda.empty_cache()

    rendered = []
    for case in prepared:
        case_id = case["case_id"]
        old_result, new_result = results["old"][case_id], results["new"][case_id]
        sheet, rows, size = render_case(case, case["frames"], old_result["labels"], new_result["labels"], output_dir, args.columns, args.alpha, args.quality)
        rendered.append({
            **{
                key: value
                for key, value in case.items()
                if key not in {"frames", "indices"}
            },
            "sampled_source_indices": case["indices"].tolist(),
            "old_metrics": old_result["metrics"],
            "new_metrics": new_result["metrics"],
            "old_attention_shape": old_result["attention_shape"],
            "new_attention_shape": new_result["attention_shape"],
            "sheet": sheet,
            "sheet_rows": rows,
            "sheet_size": size,
        })
    old_aggregate = aggregate_metrics(rendered, "old_metrics")
    new_aggregate = aggregate_metrics(rendered, "new_metrics")
    report = {
        "title": "V-JEPA xSSC test_5 step-10000 vs step-13000",
        "test5_file": str(args.test5_file.resolve()),
        "scope": "unique source videos with decoded frame count < 50",
        "old_config": str(args.old_config.resolve()),
        "old_checkpoint": str(args.old_checkpoint.resolve()),
        "new_config": str(args.new_config.resolve()),
        "new_checkpoint": str(args.new_checkpoint.resolve()),
        "checkpoint_load": load_reports,
        "old_aggregate": old_aggregate,
        "new_aggregate": new_aggregate,
        "assessment": verdict(old_aggregate, new_aggregate),
        "cases": rendered,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "index.html").write_text(build_html(report))
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "metric", "step10000", "step13000", "delta"])
        for case in rendered:
            for key in case["old_metrics"]:
                old_value, new_value = case["old_metrics"][key], case["new_metrics"][key]
                writer.writerow([case["case_id"], key, old_value, new_value, new_value - old_value])
    print(json.dumps({"output_dir": str(output_dir), "cases": len(rendered), "assessment": report["assessment"], "old_aggregate": old_aggregate, "new_aggregate": new_aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
