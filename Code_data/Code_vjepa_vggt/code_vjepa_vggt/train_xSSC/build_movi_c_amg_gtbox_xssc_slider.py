#!/usr/bin/env python3
"""Build a frame slider comparing AMG masks, GT boxes, and xSSC slots."""

from __future__ import annotations

import argparse
import gc
import html
import json
from pathlib import Path
import random
import re
import sys

import cv2
import numpy as np
import torch


TRAIN_XSSC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRAIN_XSSC_ROOT.parent
PACKAGE_PARENT = REPO_ROOT.parent
EXPERIMENT = TRAIN_XSSC_ROOT / "xssc_rsfq2_ytvis_dinov3_vitl16_256"
sys.path.insert(0, str(PACKAGE_PARENT))
sys.path.insert(0, str(TRAIN_XSSC_ROOT))
sys.path.insert(0, str(EXPERIMENT / "third_party/dinov3"))
sys.path.insert(0, str(EXPERIMENT / "upstream"))

from compare_movi_c_gt_vs_gdino_sam2 import (  # noqa: E402
    add_title,
    align_slot_labels,
    load_lightweight_checkpoint,
    slot_overlay,
)


DEFAULT_REPORT = Path(
    "/data/gaoya/agent-data/outputs/"
    "movi_c_gt_vs_gdino_sam2_fixed5_20260722"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/gaoya/dataset"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--webp-quality", type=int, default=92)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masks_to_boxes(masks, num_slots, num_frames):
    boxes = np.zeros((1, num_frames, num_slots, 4), dtype=np.float32)
    height, width = masks.shape[-2:]
    for slot_id, mask in enumerate(masks[:num_slots]):
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        box = np.asarray(
            [xs.min() / width, ys.min() / height, (xs.max() + 1) / width, (ys.max() + 1) / height],
            dtype=np.float32,
        )
        boxes[0, :, slot_id] = box
    return boxes


def decode_video(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"Could not decode {path}")
    return np.stack(frames)


def write_webp(path, image_rgb, quality):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path),
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_WEBP_QUALITY, int(quality)],
    )
    if not ok:
        raise RuntimeError(f"Could not write {path}")


def infer_amg_conditioned_labels(
    model,
    dataset,
    collate_fn,
    dataset_index,
    selected_masks,
    num_slots,
    device,
    amp_dtype,
):
    batch = collate_fn([dataset[dataset_index]])
    num_frames = int(batch["video"].shape[1])
    boxes = masks_to_boxes(selected_masks, num_slots, num_frames)
    set_seed(42 + dataset_index)
    with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
        output = model(
            batch={
                "video": batch["video"].to(device),
                "bbox": torch.from_numpy(boxes).to(device),
            }
        )
    attention = output["attentd"][0].detach().float().cpu().numpy()
    return attention.argmax(axis=1).astype(np.uint8), boxes


def build_slider_section(cases):
    case_buttons = []
    for position, case in enumerate(cases):
        active = " active" if position == 0 else ""
        case_buttons.append(
            f"<button type='button' class='frame-case-button{active}' "
            f"data-case='{position}'>case {case['dataset_index']:03d}</button>"
        )
    cases_json = html.escape(json.dumps(cases), quote=False)
    return f"""<!-- FRAME_SLIDER_START -->
<style>
#frame-slider{{border-bottom:1px solid #343a40;padding:0 0 24px;margin:0 0 24px}}
#frame-slider h1{{margin:0 0 8px}}#frame-slider .frame-note{{margin:0 0 16px;color:#b7bec8}}
.frame-controls{{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center;margin:14px 0}}
.frame-cases,.frame-branches{{display:flex;gap:0;flex-wrap:wrap;margin:10px 0}}
.frame-case-button,.frame-branch-button{{border:1px solid #4a515b;background:#171a1e;color:#d7dce2;padding:7px 11px;border-radius:0;cursor:pointer}}
.frame-case-button+button,.frame-branch-button+button{{border-left:0}}
.frame-case-button.active,.frame-branch-button.active{{background:#e7eaee;color:#111418}}
#frame-range{{width:100%;accent-color:#f59e0b}}#frame-number{{font-variant-numeric:tabular-nums;min-width:70px;text-align:right}}
#frame-image{{display:block;width:100%;height:auto;background:#000;aspect-ratio:3/1}}
.frame-panel-labels{{display:grid;grid-template-columns:repeat(3,1fr);gap:0;color:#aeb5bf;font-size:12px;margin-top:6px}}
.frame-panel-labels span{{text-align:center}}
@media(max-width:700px){{.frame-controls{{grid-template-columns:1fr auto}}.frame-controls label{{grid-column:1/-1}}}}
</style>
<section id="frame-slider">
  <h1>Frame comparison</h1>
  <p class="frame-note">AMG masks are generated independently per frame. The xSSC branch selector changes only the slot initializer; AMG-conditioned xSSC uses filtered frame-0 AMG boxes.</p>
  <div class="frame-cases" role="tablist" aria-label="MOVi-C case">{''.join(case_buttons)}</div>
  <div class="frame-branches" role="group" aria-label="xSSC conditioning">
    <button type="button" class="frame-branch-button active" data-branch="amg">AMG box</button>
    <button type="button" class="frame-branch-button" data-branch="gt">GT box</button>
    <button type="button" class="frame-branch-button" data-branch="gdino">GDINO+SAM2 box</button>
  </div>
  <div class="frame-controls">
    <label for="frame-range">Frame</label>
    <input id="frame-range" type="range" min="0" max="23" value="0" step="1">
    <output id="frame-number" for="frame-range">00 / 23</output>
  </div>
  <img id="frame-image" alt="AMG masks, GT boxes, and xSSC slot overlay">
  <div class="frame-panel-labels"><span>AMG instance masks</span><span>Dataset GT boxes</span><span>xSSC slot overlay</span></div>
</section>
<script type="application/json" id="frame-slider-data">{cases_json}</script>
<script>
(() => {{
  const cases = JSON.parse(document.getElementById('frame-slider-data').textContent);
  const range = document.getElementById('frame-range');
  const output = document.getElementById('frame-number');
  const image = document.getElementById('frame-image');
  let caseIndex = 0;
  let branch = 'amg';
  const render = () => {{
    const frame = Number(range.value);
    image.src = `${{cases[caseIndex].asset_dir}}/frame_${{String(frame).padStart(2, '0')}}_${{branch}}.webp`;
    output.value = `${{String(frame).padStart(2, '0')}} / 23`;
    for (const offset of [-1, 1]) {{
      const adjacent = frame + offset;
      if (adjacent >= 0 && adjacent <= 23) {{
        const preload = new Image();
        preload.src = `${{cases[caseIndex].asset_dir}}/frame_${{String(adjacent).padStart(2, '0')}}_${{branch}}.webp`;
      }}
    }}
  }};
  range.addEventListener('input', render);
  document.querySelectorAll('.frame-case-button').forEach(button => {{
    button.addEventListener('click', () => {{
      caseIndex = Number(button.dataset.case);
      document.querySelectorAll('.frame-case-button').forEach(item => item.classList.toggle('active', item === button));
      render();
    }});
  }});
  document.querySelectorAll('.frame-branch-button').forEach(button => {{
    button.addEventListener('click', () => {{
      branch = button.dataset.branch;
      document.querySelectorAll('.frame-branch-button').forEach(item => item.classList.toggle('active', item === button));
      render();
    }});
  }});
  render();
}})();
</script>
<!-- FRAME_SLIDER_END -->"""


def update_index(report_dir, cases):
    index_path = report_dir / "index.html"
    page = index_path.read_text()
    page = re.sub(
        r"<!-- FRAME_SLIDER_START -->.*?<!-- FRAME_SLIDER_END -->",
        "",
        page,
        flags=re.DOTALL,
    )
    section = build_slider_section(cases)
    if "<main>" not in page:
        raise RuntimeError(f"Cannot find <main> in {index_path}")
    index_path.write_text(page.replace("<main>", "<main>" + section, 1))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    report_dir = args.report_dir.resolve()
    comparison = json.loads((report_dir / "metrics.json").read_text())
    amg_report = json.loads((report_dir / "sam2_amg/metrics.json").read_text())
    amg_videos = json.loads(
        (report_dir / "sam2_amg/videos/metrics.json").read_text()
    )
    comparison_by_index = {
        int(case["dataset_index"]): case for case in comparison["cases"]
    }
    amg_by_index = {
        int(case["dataset_index"]): case for case in amg_report["cases"]
    }
    amg_video_by_index = {
        int(case["dataset_index"]): case for case in amg_videos["cases"]
    }

    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    config_file = Path(comparison["config"])
    cfg = Config.fromfile(config_file)
    cfg.dataset_v.base_dir = args.data_dir.resolve()
    dataset = build_from_config(cfg.dataset_v)
    collate_fn = build_from_config(cfg.collate_fn_v)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = ModelWrap(
        build_from_config(cfg.model), cfg.model_imap, cfg.model_omap
    ).to(device).eval()
    model.freez(cfg.freez, verbose=False)
    checkpoint_report = load_lightweight_checkpoint(
        model, Path(comparison["checkpoint"])
    )
    amp_dtype = getattr(torch, args.amp_dtype)

    assets_root = report_dir / "frame_slider"
    cases = []
    for position, dataset_index in enumerate(comparison["indices"], start=1):
        dataset_index = int(dataset_index)
        comparison_case = comparison_by_index[dataset_index]
        amg_case = amg_by_index[dataset_index]
        amg_video_case = amg_video_by_index[dataset_index]
        amg_arrays = np.load(report_dir / amg_case["arrays"])
        selected_masks = amg_arrays["selected_masks"].astype(bool)
        amg_labels_raw, amg_boxes = infer_amg_conditioned_labels(
            model,
            dataset,
            collate_fn,
            dataset_index,
            selected_masks,
            cfg.num_slots,
            device,
            amp_dtype,
        )
        comparison_arrays = np.load(report_dir / comparison_case["arrays"])
        gt_labels = comparison_arrays["gt_condition_slot_labels"]
        amg_labels, amg_color_mapping = align_slot_labels(
            gt_labels, amg_labels_raw, cfg.num_slots
        )

        sample = dataset[dataset_index]
        imagenet_mean = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
        imagenet_std = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
        normalized = sample["video"].permute(0, 2, 3, 1).numpy()
        video_rgb = (
            normalized * imagenet_std[None, None, None, :]
            + imagenet_mean[None, None, None, :]
        ).round().clip(0, 255).astype(np.uint8)
        amg_xssc_overlay = slot_overlay(video_rgb, amg_labels)

        comparison_video = decode_video(report_dir / comparison_case["video"])
        amg_video = decode_video(report_dir / amg_video_case["video"])
        if not (
            len(video_rgb) == len(comparison_video) == len(amg_video) == 24
        ):
            raise RuntimeError(f"Unexpected frame count for index {dataset_index}")

        case_dir = assets_root / f"case_{position:02d}_index_{dataset_index:03d}"
        for frame_index in range(24):
            amg_panel = amg_video[frame_index, :, 512:768]
            gt_box_panel = comparison_video[frame_index, :, 256:512]
            branch_panels = {
                "amg": add_title(
                    amg_xssc_overlay[frame_index], "xSSC slots | AMG frame-0 boxes"
                ),
                "gt": comparison_video[frame_index, :, 512:768],
                "gdino": comparison_video[frame_index, :, 1024:1280],
            }
            for branch, xssc_panel in branch_panels.items():
                frame = np.concatenate(
                    [amg_panel, gt_box_panel, xssc_panel], axis=1
                )
                write_webp(
                    case_dir / f"frame_{frame_index:02d}_{branch}.webp",
                    frame,
                    args.webp_quality,
                )

        np.savez_compressed(
            case_dir / "amg_xssc.npz",
            amg_boxes=amg_boxes[0],
            amg_condition_slot_labels_raw=amg_labels_raw,
            amg_condition_slot_labels_aligned=amg_labels,
        )
        cases.append(
            {
                "dataset_index": dataset_index,
                "asset_dir": str(case_dir.relative_to(report_dir)),
                "frames": 24,
                "amg_mask_count": int(len(selected_masks)),
                "amg_to_gt_slot_color_mapping": amg_color_mapping,
            }
        )
        print(
            f"[slider {position}/{len(comparison['indices'])}] "
            f"index={dataset_index} assets={case_dir}",
            flush=True,
        )

    payload = {
        "checkpoint": comparison["checkpoint"],
        "checkpoint_report": checkpoint_report,
        "cases": cases,
        "xssc_branches": {
            "amg": "filtered frame-0 AMG boxes",
            "gt": "dataset frame-0 GT boxes",
            "gdino": "frame-0 GDINO plus SAM2 boxes",
        },
        "color_alignment": "AMG and GDINO slot colors aligned to GT branch for display only",
    }
    assets_root.mkdir(parents=True, exist_ok=True)
    (assets_root / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
    update_index(report_dir, cases)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"report={report_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
