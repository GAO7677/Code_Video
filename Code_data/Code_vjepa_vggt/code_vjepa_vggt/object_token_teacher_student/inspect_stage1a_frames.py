#!/usr/bin/env python3
"""Frame-accurate Stage1A aux visualizer.

Fixes the frame/coord issues of the video overlay:
- The aux loss uses GT boxes/tracks GROUPED from the FULL (context+future) sequence
  to `latent_frames`, taking the LAST frame of each group -> true source indices
  are [(i+1)*group-1 for i in range(latent_frames)] over batch["video"] (NOT an
  even linspace over context_video).
- Boxes/tracks are normalized in full-frame [0,1] space -> de-normalize by the
  video H,W and overlay on the REAL video frame at the true index.
Outputs one PNG per (case, latent-frame) with GT (green) vs Pred (red) boxes +
track endpoints, plus the per-frame box L1 so visual and number line up.

Usage:
  python3 inspect_stage1a_frames.py --config <1A.yaml> --checkpoint <step.pt> \
      --indices 0 1 2 --output-dir <dir> --device cuda:0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from code_vjepa_vggt.inspect_cotracker_vggt_geometry import draw_box_rgb, draw_point_rgb, tensor_frame_to_uint8_hwc
from code_vjepa_vggt.object_token_teacher_student.runtime_stage1a_full_token import FullTokenTeacherTrainer
from code_vjepa_vggt.infer_context_video_wan import _load_trainable_state
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.masks import collate_video_batch

GT_COLOR = (40, 220, 40)
PRED_COLOR = (240, 60, 60)


def _load_aux_state(trainer: FullTokenTeacherTrainer, ckpt: Path) -> dict[str, Any]:
    state = _load_trainable_state(ckpt)
    lk = "object_pooler.latent_proj.weight"
    if lk in state and state[lk].dim() == 2:
        trainer.object_pooler._ensure_latent_proj(int(state[lk].shape[1]), trainer.device_obj)
    cur = trainer.state_dict()
    prefixes = ("object_pooler.", "object_aux_heads.", "object_adapter.")
    filt = {k: v for k, v in state.items()
            if k.startswith(prefixes) and k in cur and tuple(cur[k].shape) == tuple(v.shape)}
    trainer.load_state_dict(filt, strict=False)
    return {"loaded": len(filt), "ckpt_tensors": len(state)}


def _save_png(path: Path, frame_hwc_uint8: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame_hwc_uint8).save(path)


def _process_case(trainer, sample_index, output_dir, image_save):
    sample = trainer.dataset[int(sample_index)]
    batch = collate_video_batch([sample])
    with torch.no_grad():
        prep = trainer._prepare_stage1a_batch(batch)
    aux = prep.object_aux_out

    video = batch["video"][0]                      # [3, T_full, H, W] real frames
    full_T = int(video.shape[1])
    H, W = int(video.shape[-2]), int(video.shape[-1])

    gt_box = prep.gt_box_xyxy[0].float().cpu().numpy()        # [Lf, O, 4] normalized
    gt_box_valid = (prep.gt_box_valid[0].cpu().numpy() > 0.5)
    pred_box = aux.pred_box_xyxy[0].float().cpu().numpy()
    gt_trk = prep.gt_track_summary[0].float().cpu().numpy()   # [Lf, O, 4] last_xy+delta (norm)
    gt_trk_valid = (prep.gt_track_valid[0].cpu().numpy() > 0.5)
    pred_trk = aux.pred_track_summary[0].float().cpu().numpy()
    obj_valid = (prep.object_valid_mask[0].cpu().numpy() > 0.5)

    latent_frames = int(gt_box.shape[0])
    group = full_T // latent_frames
    # true source index = LAST frame of each group (matches _group_box_targets [:, :, -1])
    src_indices = [(i + 1) * group - 1 for i in range(latent_frames)]
    scale = np.array([W, H, W, H], dtype=np.float32)

    case = {"sample_index": int(sample_index), "video_path": sample.get("video_path"),
            "full_T": full_T, "latent_frames": latent_frames, "group": group,
            "src_indices": src_indices, "frames": []}
    for i, src in enumerate(src_indices):
        frame = tensor_frame_to_uint8_hwc(video[:, src]).copy()
        per_obj = []
        for o in range(gt_box.shape[1]):
            if not bool(obj_valid[o]):
                continue
            gb = gt_box[i, o] * scale
            pb = pred_box[i, o] * scale
            if bool(gt_box_valid[i, o]):
                draw_box_rgb(frame, gb, GT_COLOR, f"gt{o}")
            draw_box_rgb(frame, pb, PRED_COLOR, f"pr{o}")
            # track endpoints: last_xy (norm) -> px; start = last - delta
            if bool(gt_trk_valid[i, o]):
                g_last = np.array([gt_trk[i, o, 0] * (W - 1), gt_trk[i, o, 1] * (H - 1)])
                draw_point_rgb(frame, g_last, GT_COLOR, f"gt{o}")
            p_last = np.array([pred_trk[i, o, 0] * (W - 1), pred_trk[i, o, 1] * (H - 1)])
            draw_point_rgb(frame, p_last, PRED_COLOR, f"pr{o}")
            box_l1 = float(np.abs(gt_box[i, o] - pred_box[i, o]).mean()) if bool(gt_box_valid[i, o]) else None
            per_obj.append({"obj": o, "gt_box": [round(float(x), 4) for x in gt_box[i, o]],
                            "pred_box": [round(float(x), 4) for x in pred_box[i, o]],
                            "box_l1": None if box_l1 is None else round(box_l1, 4)})
        png = output_dir / f"case{sample_index:05d}_latent{i}_src{src}.png"
        _save_png(png, frame)
        case["frames"].append({"latent_idx": i, "src_frame": src, "png": png.name, "objects": per_obj})
    return case


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--indices", type=int, nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    cfg["data"]["num_workers"] = 0
    cfg["data"]["batch_size"] = 1
    cfg["data"]["random_context_frames"] = False
    cfg["model"]["init_wan_lora_from_checkpoint"] = None
    cfg.setdefault("logging", {})["use_wandb"] = False

    trainer = FullTokenTeacherTrainer(cfg, build_optimizer=False, device=args.device)
    torch.nn.Module.train(trainer, False)
    info = _load_aux_state(trainer, Path(args.checkpoint))
    print(f"[frames] loaded {info['loaded']}/{info['ckpt_tensors']} tensors", flush=True)

    out = Path(args.output_dir)
    cases = [_process_case(trainer, i, out, True) for i in args.indices]
    (out / "frames_metrics.json").write_text(
        json.dumps({"checkpoint": args.checkpoint, "cases": cases}, ensure_ascii=False, indent=1))
    print(f"[frames] wrote {sum(len(c['frames']) for c in cases)} PNGs to {out}", flush=True)
    print(f"[frames] metrics: {out / 'frames_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()

