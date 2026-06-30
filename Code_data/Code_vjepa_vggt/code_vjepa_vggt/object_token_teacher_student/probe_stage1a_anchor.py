#!/usr/bin/env python3
"""Probe: dump the per-frame `active_box_xyxy` anchor + `pred_box_xyxy` + gt box
that the Stage1A aux box head sees, to determine whether the anchor itself is
static across latent frames (root cause) or the head residual is the limiter.

Monkeypatches ObjectAuxHeads.forward to capture its `active_box_xyxy` argument.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from code_vjepa_vggt.object_token_teacher_student.runtime_stage1a_full_token import FullTokenTeacherTrainer
from code_vjepa_vggt.infer_context_video_wan import _load_trainable_state
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.masks import collate_video_batch


def _load_aux_state(trainer, ckpt: Path):
    state = _load_trainable_state(ckpt)
    lk = "object_pooler.latent_proj.weight"
    if lk in state and state[lk].dim() == 2:
        trainer.object_pooler._ensure_latent_proj(int(state[lk].shape[1]), trainer.device_obj)
    cur = trainer.state_dict()
    pref = ("object_pooler.", "object_aux_heads.", "object_adapter.")
    filt = {k: v for k, v in state.items()
            if k.startswith(pref) and k in cur and tuple(cur[k].shape) == tuple(v.shape)}
    trainer.load_state_dict(filt, strict=False)
    return len(filt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--index", type=int, default=1)
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
    n = _load_aux_state(trainer, Path(args.checkpoint))
    print(f"[probe] loaded {n} tensors", flush=True)

    captured = {}
    orig_forward = trainer.object_aux_heads.forward

    def wrapped(object_latent_tokens, active_track_summary, active_box_xyxy=None):
        captured["active_box_xyxy"] = None if active_box_xyxy is None else active_box_xyxy.detach().float().cpu().numpy()
        captured["active_track_summary"] = active_track_summary.detach().float().cpu().numpy()
        captured["latent_tokens"] = object_latent_tokens.detach().float().cpu().numpy()
        return orig_forward(object_latent_tokens, active_track_summary, active_box_xyxy)

    trainer.object_aux_heads.forward = wrapped

    sample = trainer.dataset[int(args.index)]
    batch = collate_video_batch([sample])
    with torch.no_grad():
        prep = trainer._prepare_stage1a_batch(batch)

    anchor = captured.get("active_box_xyxy")     # [B?, Lf, O, 4] or [Lf, O, 4]
    gt_box = prep.gt_box_xyxy[0].float().cpu().numpy()
    pred_box = prep.object_aux_out.pred_box_xyxy[0].float().cpu().numpy()
    tok = captured.get("latent_tokens")
    print(f"[probe] anchor shape={None if anchor is None else anchor.shape} "
          f"gt shape={gt_box.shape} pred shape={pred_box.shape} tok shape={None if tok is None else tok.shape}")

    # squeeze possible batch dim on anchor
    if anchor is not None and anchor.ndim == 4:
        anchor = anchor[0]
    if tok is not None and tok.ndim >= 3:
        tok_lf = tok[0] if tok.shape[0] == 1 else tok
    else:
        tok_lf = None

    Lf = gt_box.shape[0]
    obj = 0
    print(f"\n=== obj {obj}: per-latent-frame anchor / pred / gt ===")
    for i in range(Lf):
        a = None if anchor is None else [round(float(x), 4) for x in anchor[i, obj]]
        p = [round(float(x), 4) for x in pred_box[i, obj]]
        g = [round(float(x), 4) for x in gt_box[i, obj]]
        print(f"  f{i}: anchor={a}  pred={p}  gt={g}")
    # token variation across frames: are the per-frame latent tokens identical?
    if tok_lf is not None and tok_lf.ndim == 3:  # [Lf, O, D]
        d = tok_lf[:, obj]  # [Lf, D]
        diffs = [round(float(np.abs(d[i] - d[0]).mean()), 6) for i in range(d.shape[0])]
        print(f"\n[probe] obj{obj} latent-token mean|frame_i - frame_0|: {diffs}")
    # anchor variation
    if anchor is not None:
        av = [round(float(np.abs(anchor[i, obj] - anchor[0, obj]).mean()), 6) for i in range(Lf)]
        print(f"[probe] obj{obj} anchor mean|frame_i - frame_0|: {av}")


if __name__ == "__main__":
    main()
