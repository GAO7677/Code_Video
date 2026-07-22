#!/usr/bin/env python3
"""All-slot frame overlay comparison across YTVIS-HQ, MOVi-C, and test_5 clips."""

import argparse
import html
import json
from pathlib import Path
import random
import re
import sys

import imageio.v3 as iio
import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parent
TRAIN_XSSC_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, str(TRAIN_XSSC_ROOT))
sys.path.insert(0, "/home/gaoya/Grounded-SAM-2-main")

from build_movi_c_amg_gtbox_xssc_slider import masks_to_boxes  # noqa: E402
from visualize_movi_c_sam2_amg import (  # noqa: E402
    DEFAULT_CHECKPOINT as DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_CONFIG as DEFAULT_SAM2_CONFIG,
    resolve_sam2_config_name,
    select_xssc_candidates,
)

IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 1, 3, 1, 1)
PALETTE = np.asarray(
    [
        [239, 68, 68],
        [59, 130, 246],
        [34, 197, 94],
        [250, 204, 21],
        [168, 85, 247],
        [6, 182, 212],
        [249, 115, 22],
        [236, 72, 153],
        [132, 204, 22],
        [20, 184, 166],
        [251, 146, 60],
    ],
    dtype=np.uint8,
)

YTVIS_CONFIG = ROOT / "upstream/config-randsfq/rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512.py"
MOVIC_CONFIG = ROOT / "upstream/config-randsfq/rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000.py"
YTVIS_CKPT_DIR = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC/"
    "restart_save1000_20260720T140029Z/rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512/42"
)
MOVIC_CKPT_DIR = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC/"
    "restart_save1000_20260720T140029Z/movi_c_transfer15000_b64_acc3_20260721T134713Z/"
    "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/42"
)
MOVIC_RUN_DIR = MOVIC_CKPT_DIR.parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/data/gaoya/dataset"))
    parser.add_argument("--test5-file", type=Path, default=Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/gaoya/agent-data/outputs/xssc_slot_overlay_ytvis_movic_test5_compare"))
    parser.add_argument("--num-cases-per-source", type=int, default=3)
    parser.add_argument("--num-ytvis-cases", type=int, default=None)
    parser.add_argument("--num-movic-cases", type=int, default=None)
    parser.add_argument("--num-test5-cases", type=int, default=None, help="Number of test_5 entries; use 0 for all entries.")
    parser.add_argument("--max-frames", type=int, default=6, help="Maximum frames per case; use 0 for full videos.")
    parser.add_argument("--external-resize-mode", choices=("crop", "padding"), default="crop")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--quality", type=int, default=88)
    parser.add_argument("--sam2-config", type=Path, default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--max-selected", type=int, default=11)
    parser.add_argument("--min-area-ratio", type=float, default=0.004)
    parser.add_argument("--max-area-ratio", type=float, default=0.35)
    parser.add_argument("--min-bbox-side", type=float, default=7.0)
    parser.add_argument("--background-area-ratio", type=float, default=0.06)
    parser.add_argument("--background-span-ratio", type=float, default=0.75)
    parser.add_argument("--border-area-ratio", type=float, default=0.025)
    parser.add_argument("--border-occupancy-ratio", type=float, default=0.18)
    parser.add_argument("--opposite-edge-area-ratio", type=float, default=0.04)
    parser.add_argument("--shadow-min-area-ratio", type=float, default=0.03)
    parser.add_argument("--shadow-max-luminance-ratio", type=float, default=0.55)
    parser.add_argument("--shadow-max-chromaticity-distance", type=float, default=0.10)
    parser.add_argument("--shadow-max-gradient-mean", type=float, default=20.0)
    parser.add_argument("--duplicate-iou", type=float, default=0.70)
    parser.add_argument("--duplicate-containment", type=float, default=0.85)
    parser.add_argument("--extra-movic-checkpoint", type=Path, default=None)
    parser.add_argument("--extra-movic-label", default=None)
    return parser.parse_args()


def resolve(path):
    return path if path.is_absolute() else (ROOT / path)


def decode_dataset_rgb(video):
    rgb = video.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return rgb.clamp(0, 255).round().to(torch.uint8)[0].permute(0, 2, 3, 1).numpy()


def normalize_rgb_frames(frames):
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float()[None]
    return (video - IMAGENET_MEAN) / IMAGENET_STD


def center_crop_square(frames):
    h, w = frames.shape[1:3]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frames[:, y0 : y0 + side, x0 : x0 + side]


def resize_frames(frames, size=256):
    out = []
    for frame in frames:
        out.append(np.asarray(Image.fromarray(frame).resize((size, size), Image.Resampling.BILINEAR)))
    return np.stack(out, axis=0)


def resize_pad_frames(frames, size=256):
    out = []
    for frame in frames:
        height, width = frame.shape[:2]
        scale = min(size / width, size / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = Image.fromarray(frame).resize(
            (new_width, new_height), Image.Resampling.BILINEAR
        )
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        canvas.paste(resized, ((size - new_width) // 2, (size - new_height) // 2))
        out.append(np.asarray(canvas))
    return np.stack(out, axis=0)


def read_video_clip(path, max_frames, resize_mode):
    frames = iio.imread(path, plugin="pyav")
    if frames.ndim == 3:
        frames = frames[None]
    if max_frames and max_frames > 0:
        frames = frames[:max_frames]
    frames = frames[:, :, :, :3]
    if resize_mode == "padding":
        frames = resize_pad_frames(frames)
    else:
        frames = resize_frames(center_crop_square(frames))
    return frames.astype(np.uint8), normalize_rgb_frames(frames.astype(np.uint8))


def load_checkpoint(model, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("m.encode_backbone.")]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch {checkpoint}: missing={missing}, unexpected={incompatible.unexpected_keys}")


def build_model(config_file, checkpoint, device):
    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    cfg = Config.fromfile(config_file)
    model = build_from_config(cfg.model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)
    model.freez(cfg.freez, verbose=False)
    load_checkpoint(model, checkpoint)
    return cfg, model.to(device).eval()


def build_dataset(config_file, data_dir):
    from object_centric_bench.util import Config, build_from_config

    cfg = Config.fromfile(config_file)
    cfg.dataset_v.base_dir = data_dir.resolve()
    return cfg, build_from_config(cfg.dataset_v), build_from_config(cfg.collate_fn_v)


def infer(model, batch, device, amp_dtype):
    with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
        out = model(batch={key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()})
    attentd = out["attentd"][0].detach().float().cpu()
    labels = attentd.argmax(dim=1).to(torch.uint8).numpy()
    return labels


def overlay(frame, labels):
    labels_full = labels.repeat(16, axis=0).repeat(16, axis=1)
    colors = PALETTE[labels_full % len(PALETTE)]
    result = (frame.astype(np.float32) * 0.43 + colors.astype(np.float32) * 0.57).round().clip(0, 255).astype(np.uint8)
    for pos in range(16, result.shape[0], 16):
        result[pos, :, :] = (result[pos, :, :].astype(np.float32) * 0.72).astype(np.uint8)
        result[:, pos, :] = (result[:, pos, :].astype(np.float32) * 0.72).astype(np.uint8)
    return result


def save_webp(path, array, quality):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="WEBP", quality=quality, method=4)


def key_to_text(key):
    return key.decode("utf-8", errors="replace") if isinstance(key, bytes) else str(key)


def sample_dataset_cases(name, config_file, data_dir, n, seed):
    cfg, dataset, collate = build_dataset(config_file, data_dir)
    indices = sorted(random.Random(seed).sample(range(len(dataset)), min(n, len(dataset))))
    cases = []
    for index in indices:
        item = dataset[index]
        batch = collate([item])
        frames = decode_dataset_rgb(batch["video"])
        cases.append(
            {
                "source": name,
                "case_id": f"{name}_{index:06d}",
                "source_key": key_to_text(dataset.keys[index]) if hasattr(dataset, "keys") else f"index={index}",
                "rgb": frames,
                "video": batch["video"],
                "bbox": batch.get("bbox"),
            }
        )
    return cases


def sample_test5_cases(test5_file, n, max_frames, resize_mode):
    cases = []
    lines = [line.strip() for line in test5_file.read_text().splitlines() if line.strip()]
    if n is not None and n > 0:
        lines = lines[:n]
    for position, line in enumerate(lines, start=1):
        json_path = Path(line.strip())
        payload = json.loads(json_path.read_text())
        rgb, video = read_video_clip(payload["source_video"], max_frames, resize_mode)
        cases.append(
            {
                "source": "test5",
                "case_id": f"test5_{position:03d}_{json_path.stem}",
                "source_key": payload["source_video"],
                "rgb": rgb,
                "video": video,
                "bbox": None,
            }
        )
    return cases


def build_sam2_generator(args, device):
    if not args.sam2_config.is_file() or not args.sam2_checkpoint.is_file():
        raise FileNotFoundError("SAM2 config or checkpoint is missing")
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    sam2 = build_sam2(
        resolve_sam2_config_name(args.sam2_config),
        str(args.sam2_checkpoint.resolve()),
        device=str(device),
        mode="eval",
    )
    return SAM2AutomaticMaskGenerator(sam2)


def make_amg_condition(case, generator, args, num_slots):
    image = case["rgb"][0]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        annotations = generator.generate(image)
    selected = select_xssc_candidates(
        annotations, image.shape[0] * image.shape[1], args, image=image
    )
    if selected:
        masks = np.stack([item["segmentation"].astype(bool) for item in selected], axis=0)
    else:
        masks = np.zeros((0, image.shape[0], image.shape[1]), dtype=bool)
    boxes = masks_to_boxes(masks, num_slots, int(case["video"].shape[1]))
    return (
        torch.from_numpy(boxes).float(),
        {
            "raw_mask_count": int(len(annotations)),
            "selected_mask_count": int(len(selected)),
            "selected_boxes_xywh": [
                [float(value) for value in item["bbox"]] for item in selected
            ],
        },
    )


def select_best_movi_checkpoint(checkpoints):
    log_file = MOVIC_RUN_DIR / "train.log"
    if not log_file.is_file():
        return checkpoints[-1], "train.log not found; best falls back to latest."
    text = log_file.read_text(errors="ignore")
    val_by_step = {
        int(step): {"recon": float(recon), "mbo": float(mbo)}
        for step, recon, mbo in re.findall(
            r"(\d+) \{'recon': ([0-9.eE+-]+), 'mbo': ([0-9.eE+-]+)\}", text
        )
    }
    scored = []
    for checkpoint in checkpoints:
        step = int(checkpoint.stem.split("-")[-1])
        if step in val_by_step:
            scored.append((val_by_step[step]["mbo"], step, checkpoint, val_by_step[step]))
    if not scored:
        return checkpoints[-1], "No checkpoint step has a parsed val mBO; best falls back to latest."
    mbo, step, checkpoint, metrics = max(scored, key=lambda row: row[0])
    return checkpoint, f"Best existing checkpoint by parsed val mBO: step={step}, mBO={mbo:.6f}, recon={metrics['recon']:.6f}."


def build_html(metadata):
    data = json.dumps(metadata, separators=(",", ":"))
    title = html.escape(metadata["title"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>
*{{box-sizing:border-box}}:root{{color-scheme:dark}}body{{margin:0;background:#101214;color:#f4f5f6;font:14px system-ui,sans-serif;letter-spacing:0}}header{{position:sticky;top:0;z-index:4;background:rgba(16,18,20,.98);border-bottom:1px solid #34383d}}.bar{{max-width:2200px;margin:auto;padding:10px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}}h1{{font-size:18px;margin:0 auto 0 0}}select,input,button{{height:34px;border:1px solid #4b525a;border-radius:5px;background:#202429;color:#f5f6f7;font:inherit}}select{{padding:0 28px 0 9px}}button{{width:34px;cursor:pointer}}#frameSlider{{min-width:220px;flex:0 1 420px;accent-color:#38bdf8}}#counter{{min-width:96px;color:#c4c9cf}}main{{max-width:2200px;margin:auto;padding:16px}}.meta{{display:flex;gap:16px;padding-bottom:14px;color:#aeb5bd;white-space:nowrap;overflow:auto}}.wrap{{overflow:auto}}.grid{{display:grid;grid-template-columns:repeat(var(--panel-count,5),minmax(230px,1fr));gap:12px;min-width:calc(var(--panel-count,5) * 242px)}}figure{{margin:0}}img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#050607;border:1px solid #34383d;border-radius:4px}}figcaption{{padding:7px 2px 0;min-height:50px;color:#c4c9cf}}strong{{display:block;color:#f3f4f6;margin-bottom:2px}}.note{{margin-top:14px;color:#aeb5bd;line-height:1.45}}.metric{{color:#7dd3fc;font-size:12px}}
</style></head><body><header><div class="bar"><h1>{title}</h1><select id="caseSelect"></select><button id="prev" title="Previous frame">&#8249;</button><button id="next" title="Next frame">&#8250;</button><input id="frameSlider" type="range" min="0" max="0" step="1" value="0"><span id="counter"></span></div></header><main><div id="meta" class="meta"></div><div class="wrap"><section id="grid" class="grid"></section></div><p class="note">MOVi-C checkpoints use bbox-conditioned initialization. In this page their boxes come from filtered SAM2 AMG frame-0 masks converted to pseudo boxes; YTVIS-HQ checkpoints are unconditioned.</p></main>
<script>
const DATA={data};const sel=document.getElementById('caseSelect');const slider=document.getElementById('frameSlider');const counter=document.getElementById('counter');const grid=document.getElementById('grid');const meta=document.getElementById('meta');let frame=0;
function pat(p,i){{return p.replace('{{frame}}',String(i).padStart(4,'0'))}}function cur(){{return DATA.cases[Number(sel.value)]}}function update(){{const c=cur();frame=Math.max(0,Math.min(frame,c.frames-1));slider.value=String(frame);counter.textContent=`${{frame+1}} / ${{c.frames}}`;Array.from(grid.querySelectorAll('img')).forEach((img,i)=>{{img.src=pat(img.dataset.pattern,frame)}})}}function render(){{const c=cur();frame=0;slider.max=String(c.frames-1);grid.style.setProperty('--panel-count',String(c.models.length+1));meta.innerHTML=`<span>${{c.source}}</span><span>${{c.case_id}}</span><span>AMG selected ${{c.amg.selected_mask_count}} / raw ${{c.amg.raw_mask_count}}</span><span>${{c.source_key}}</span>`;grid.innerHTML=[`<figure><img data-pattern="${{c.original_pattern}}"><figcaption><strong>Original</strong><span class="metric">effective 256 center crop</span></figcaption></figure>`,...c.models.map(m=>`<figure><img data-pattern="${{m.frame_pattern}}"><figcaption><strong>${{m.label}}</strong><span class="metric">${{m.slots}} slots | ${{m.condition}}</span></figcaption></figure>`)].join('');update()}}DATA.cases.forEach((c,i)=>{{const o=document.createElement('option');o.value=String(i);o.textContent=`${{String(i+1).padStart(2,'0')}} | ${{c.source}} | ${{c.case_id}}`;sel.appendChild(o)}});sel.addEventListener('change',render);slider.addEventListener('input',()=>{{frame=Number(slider.value);update()}});document.getElementById('prev').addEventListener('click',()=>{{frame--;update()}});document.getElementById('next').addEventListener('click',()=>{{frame++;update()}});render();
</script></body></html>"""


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    amp_dtype = getattr(torch, args.amp_dtype)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    movi_steps = sorted(MOVIC_CKPT_DIR.glob("step-*.pth"))
    latest_movi = movi_steps[-1]
    best_movi, best_policy = select_best_movi_checkpoint(movi_steps)
    specs = [
        ("ytvis_step4000", "ytvis", YTVIS_CONFIG, YTVIS_CKPT_DIR / "step-004000.pth"),
        ("ytvis_step15000", "ytvis", YTVIS_CONFIG, YTVIS_CKPT_DIR / "step-015000.pth"),
        (f"movi_latest_{latest_movi.stem[-6:]}", "movic", MOVIC_CONFIG, latest_movi),
        (f"movi_best_{best_movi.stem[-6:]}", "movic", MOVIC_CONFIG, best_movi),
    ]
    if args.extra_movic_checkpoint is not None:
        extra_checkpoint = args.extra_movic_checkpoint.resolve()
        extra_label = args.extra_movic_label or f"movi_extra_{extra_checkpoint.stem[-6:]}"
        specs.append((extra_label, "movic", MOVIC_CONFIG, extra_checkpoint))
    for _, _, _, ckpt in specs:
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)

    num_ytvis = args.num_cases_per_source if args.num_ytvis_cases is None else args.num_ytvis_cases
    num_movic = args.num_cases_per_source if args.num_movic_cases is None else args.num_movic_cases
    num_test5 = args.num_cases_per_source if args.num_test5_cases is None else args.num_test5_cases
    cases = []
    if num_ytvis:
        cases.extend(sample_dataset_cases("ytvis_hq_val", YTVIS_CONFIG, args.data_dir, num_ytvis, args.seed))
    if num_movic:
        cases.extend(sample_dataset_cases("movi_c_val", MOVIC_CONFIG, args.data_dir, num_movic, args.seed + 7))
    if num_test5 is not None:
        cases.extend(
            sample_test5_cases(
                args.test5_file,
                num_test5,
                args.max_frames,
                args.external_resize_mode,
            )
        )

    sam2_generator = build_sam2_generator(args, device)
    for position, case in enumerate(cases, start=1):
        amg_bbox, amg_meta = make_amg_condition(case, sam2_generator, args, 11)
        case["amg_bbox"] = amg_bbox
        case["amg_meta"] = amg_meta
        print(
            f"[amg] {position}/{len(cases)} {case['case_id']} "
            f"raw={amg_meta['raw_mask_count']} selected={amg_meta['selected_mask_count']}",
            flush=True,
        )
    del sam2_generator
    torch.cuda.empty_cache()

    rendered_cases = []
    models = {}
    for label, family, cfg_path, ckpt in specs:
        cfg, model = build_model(cfg_path, ckpt, device)
        models[label] = (family, cfg_path, model, int(cfg.max_num))
        print(f"[model] {label} slots={cfg.max_num} checkpoint={ckpt}", flush=True)

    for ci, case in enumerate(cases, start=1):
        case_dir = out_dir / "cases" / case["case_id"]
        rgb = case["rgb"] if args.max_frames <= 0 else case["rgb"][: args.max_frames]
        video = case["video"][:, : len(rgb)]
        for frame_id, frame in enumerate(rgb):
            save_webp(case_dir / "original" / f"{frame_id:04d}.webp", frame, args.quality)
        model_rows = []
        for label, family, cfg_path, model, slots in [(k, *v) for k, v in models.items()]:
            batch = {"video": video}
            if family == "movic":
                batch["bbox"] = case["amg_bbox"][:, : video.shape[1]]
            labels = infer(model, batch, device, amp_dtype)
            for frame_id, frame in enumerate(rgb):
                save_webp(case_dir / label / f"{frame_id:04d}.webp", overlay(frame, labels[frame_id]), args.quality)
            model_rows.append(
                {
                    "label": label,
                    "config": cfg_path.name,
                    "slots": int(labels.shape[1]) if labels.ndim == 4 else slots,
                    "condition": "AMG pseudo boxes" if family == "movic" else "unconditioned",
                    "frame_pattern": f"cases/{case['case_id']}/{label}/{{frame}}.webp",
                }
            )
            print(f"[infer] {ci}/{len(cases)} {case['case_id']} {label}", flush=True)
        rendered_cases.append(
            {
                "source": case["source"],
                "case_id": case["case_id"],
                "source_key": case["source_key"],
                "frames": len(rgb),
                "amg": case["amg_meta"],
                "original_pattern": f"cases/{case['case_id']}/original/{{frame}}.webp",
                "models": model_rows,
            }
        )

    metadata = {
        "title": "xSSC DINOv3 All-slot Overlay Comparison",
        "seed": args.seed,
        "max_frames": args.max_frames,
        "external_resize_mode": args.external_resize_mode,
        "checkpoints": [{"label": label, "config": str(cfg), "checkpoint": str(ckpt)} for label, _, cfg, ckpt in specs],
        "movi_best_policy": best_policy,
        "movi_condition": "Filtered SAM2 AMG masks on frame 0 are converted to pseudo boxes and passed as MOVi-C xSSC bbox condition.",
        "sam2_config": str(args.sam2_config.resolve()),
        "sam2_checkpoint": str(args.sam2_checkpoint.resolve()),
        "cases": rendered_cases,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out_dir / "index.html").write_text(build_html(metadata))
    (out_dir / "README.md").write_text(
        "# xSSC slot overlay comparison\n\n"
        "This folder compares all-slot patch assignments overlaid on the effective 256x256 input frames.\n\n"
        "- Sources: YTVIS-HQ val, MOVi-C val, and source videos from test_5.txt.\n"
        "- Checkpoints: YTVIS-HQ step-004000, YTVIS-HQ step-015000, MOVi-C latest, MOVi-C best.\n"
        f"- MOVi-C best policy: {best_policy}\n"
        "- MOVi-C checkpoints use bbox-conditioned initialization. Here the condition comes from filtered SAM2 AMG frame-0 masks converted to pseudo boxes.\n"
        "- YTVIS-HQ checkpoints are unconditioned and do not consume boxes.\n\n"
        f"- External video resize mode: {args.external_resize_mode}.\n\n"
        "Serve locally from this directory with:\n\n"
        "```bash\n"
        f"cd {out_dir} && python3 -m http.server 8897 --bind 0.0.0.0\n"
        "```\n"
    )
    print(json.dumps({"output_dir": str(out_dir), "index_html": str(out_dir / "index.html"), "cases": len(rendered_cases)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
