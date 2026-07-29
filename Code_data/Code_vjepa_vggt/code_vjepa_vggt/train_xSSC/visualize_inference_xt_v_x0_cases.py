#!/usr/bin/env python3
"""Capture xt, CFG velocity, and x0 at an exact timestep on pure inference.

Run:
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  -m code_vjepa_vggt.train_xSSC.visualize_inference_xt_v_x0_cases \
  --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/configs/xt_v_x0_two_cases.json
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from diffsynth.utils.data import save_video

from code_vjepa_vggt.train0705 import (
    infer_stage1b_context_only_no_gt_box_v_newtrain0705 as infer_base,
)
from code_vjepa_vggt.train_xSSC import (
    infer_xssc_context_slots_dinov3 as infer_xssc,
)
from code_vjepa_vggt.train_xSSC import (
    visualize_training_xt_v_x0_cases as cases_base,
)
from code_vjepa_vggt.train_xSSC import (
    visualize_training_xt_v_x0_dinov3 as single,
)
from code_vjepa_vggt.utils.video_io import (
    preprocess_video_rgb_uint8,
    read_video_prefix,
    read_video_uniform,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parent
    / "configs"
    / "xt_v_x0_two_cases.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def read_frames(
    video_path: Path,
    count: int,
    sampling_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if sampling_mode == "prefix":
        frames, indices = read_video_prefix(video_path, count)
    elif sampling_mode == "uniform":
        frames, indices = read_video_uniform(video_path, count)
    else:
        raise ValueError(f"Unsupported sampling_mode={sampling_mode!r}")
    if len(indices) != count:
        raise RuntimeError(
            f"{video_path} yielded {len(indices)}/{count} frames"
        )
    return frames, indices


def preprocess_for_official_inference(
    frames: np.ndarray,
    inference: dict[str, Any],
    video: dict[str, Any],
) -> torch.Tensor:
    mode = str(inference["context_resize_mode"])
    if mode != "cover_crop":
        raise ValueError(
            "Pure inference currently requires the official "
            "context_resize_mode=cover_crop"
        )
    return preprocess_video_rgb_uint8(
        frames,
        (int(video["height"]), int(video["width"])),
        resize_mode="cover_crop",
        cover_crop_hw=(
            int(inference["input_cover_crop_height"]),
            int(inference["input_cover_crop_width"]),
        ),
    )


class ExactInferenceX0Probe:
    """Observe the official loop and evaluate an exact-t probe off-trajectory."""

    def __init__(
        self,
        *,
        pipe,
        target_timestep: float,
        cfg_scale: float,
        cfg_merge: bool,
    ) -> None:
        if cfg_merge:
            raise ValueError("Exact probe currently requires cfg_merge=false")
        self.pipe = pipe
        self.target_timestep = float(target_timestep)
        self.cfg_scale = float(cfg_scale)
        self.cfg_merge = bool(cfg_merge)
        self.original_model_fn = pipe.model_fn
        self.original_scheduler_step = pipe.scheduler.step
        self.current_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.capture: dict[str, Any] | None = None

    def install(self) -> None:
        self.pipe.model_fn = self.model_fn
        self.pipe.scheduler.step = self.scheduler_step

    def restore(self) -> None:
        self.pipe.model_fn = self.original_model_fn
        self.pipe.scheduler.step = self.original_scheduler_step

    def model_fn(self, *args, **kwargs):
        self.current_calls.append((args, dict(kwargs)))
        return self.original_model_fn(*args, **kwargs)

    def _prefix_length(self) -> int:
        if not self.current_calls:
            return 0
        kwargs = self.current_calls[0][1]
        raw_length = kwargs.get("num_clean_prefix_latents")
        if raw_length is not None:
            return int(raw_length)
        clean = kwargs.get("clean_prefix_latents")
        return 0 if clean is None else int(clean.shape[2])

    def _probe_model(
        self,
        *,
        xt_target: torch.Tensor,
        target_timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        expected_calls = 1 if self.cfg_scale == 1.0 else 2
        if len(self.current_calls) != expected_calls:
            raise RuntimeError(
                "Unexpected model_fn calls before scheduler.step: "
                f"got={len(self.current_calls)}, expected={expected_calls}"
            )
        outputs = []
        for args, saved_kwargs in self.current_calls:
            probe_kwargs = dict(saved_kwargs)
            probe_kwargs["latents"] = xt_target
            probe_kwargs["timestep"] = target_timestep
            outputs.append(self.original_model_fn(*args, **probe_kwargs))
        positive = outputs[0]
        if self.cfg_scale == 1.0:
            return positive, None, positive
        negative = outputs[1]
        cfg = negative + self.cfg_scale * (positive - negative)
        return positive, negative, cfg

    def scheduler_step(
        self,
        model_output: torch.Tensor,
        timestep,
        sample: torch.Tensor,
        *args,
        **kwargs,
    ):
        scheduler = self.pipe.scheduler
        timestep_value = float(
            timestep.detach().float().cpu().item()
            if isinstance(timestep, torch.Tensor)
            else timestep
        )
        current_index = int(
            torch.argmin(
                (scheduler.timesteps.float() - timestep_value).abs()
            ).item()
        )
        parent_index = int(
            torch.argmin(
                (
                    scheduler.timesteps.float()
                    - self.target_timestep
                ).abs()
            ).item()
        )
        if current_index == parent_index and self.capture is None:
            current_sigma = float(
                scheduler.sigmas[current_index].float().item()
            )
            target_sigma = (
                self.target_timestep
                / float(scheduler.num_train_timesteps)
            )
            prefix_length = self._prefix_length()
            xt_target = sample + model_output * (
                target_sigma - current_sigma
            )
            if prefix_length > 0:
                xt_target = xt_target.clone()
                xt_target[:, :, :prefix_length] = sample[
                    :, :, :prefix_length
                ]
            target_tensor = torch.tensor(
                [self.target_timestep],
                device=sample.device,
                dtype=self.pipe.torch_dtype,
            )
            positive, negative, cfg = self._probe_model(
                xt_target=xt_target,
                target_timestep=target_tensor,
            )
            x0_raw = xt_target - target_sigma * cfg
            x0_restored = x0_raw.clone()
            if prefix_length > 0:
                x0_restored[:, :, :prefix_length] = sample[
                    :, :, :prefix_length
                ]
            self.capture = {
                "requested_timestep": self.target_timestep,
                "requested_sigma": target_sigma,
                "parent_step_index": current_index,
                "parent_timestep": timestep_value,
                "parent_sigma": current_sigma,
                "interpolation_delta_sigma": (
                    target_sigma - current_sigma
                ),
                "num_clean_prefix_latents": prefix_length,
                "xt": xt_target.detach().cpu().to(torch.float16),
                "v_positive": positive.detach().cpu().to(torch.float16),
                "v_negative": (
                    None
                    if negative is None
                    else negative.detach().cpu().to(torch.float16)
                ),
                "v_cfg": cfg.detach().cpu().to(torch.float16),
                "x0_raw": x0_raw.detach().cpu().to(torch.float16),
                "x0_context_restored": x0_restored.detach()
                .cpu()
                .to(torch.float16),
            }
        result = self.original_scheduler_step(
            model_output,
            timestep,
            sample,
            *args,
            **kwargs,
        )
        self.current_calls.clear()
        return result


def configure_inference_model(model, config: dict[str, Any]) -> None:
    device = torch.device(str(config["runtime"]["device"]))
    model.to(device)
    model.pipe.to(device=device, dtype=model.pipe.torch_dtype)
    model.eval()
    model.pipe.dit.eval()
    model.aux_max_objects = model.xssc_num_slots
    model.object_adapter = nn.Identity()
    inference = config["pure_inference"]
    os.environ["XSSC_PREPROCESS_MODE"] = str(
        inference["xssc_preprocess_mode"]
    )
    os.environ["XSSC_SLOT_TEMPORAL_MODE"] = str(
        inference["xssc_slot_temporal_mode"]
    )
    os.environ["XSSC_SLOT_PERTURB"] = str(
        inference["xssc_slot_perturbation"]
    )


def run_case(
    *,
    model,
    case: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    inference = config["pure_inference"]
    video_config = config["video_preprocessing"]
    path = Path(case["video_path"]).expanduser().resolve()
    sampling_mode = str(inference["context_sampling_mode"])
    source_frames, source_indices = read_frames(
        path,
        int(video_config["num_frames"]),
        sampling_mode,
    )
    context_frames, context_indices = read_frames(
        path,
        int(video_config["num_context_frames"]),
        sampling_mode,
    )
    source_tensor = preprocess_for_official_inference(
        source_frames,
        inference,
        video_config,
    )
    context_tensor = preprocess_for_official_inference(
        context_frames,
        inference,
        video_config,
    )
    context_pil = infer_base._tensor_video_to_pil_list(context_tensor)
    with torch.no_grad():
        object_context, object_debug = infer_xssc._build_object_context(
            model,
            context_video_single=context_tensor,
            prompt=str(case["caption"]),
            video_path=str(path),
        )

    probe = ExactInferenceX0Probe(
        pipe=model.pipe,
        target_timestep=float(inference["capture_target_timestep"]),
        cfg_scale=float(inference["cfg_scale"]),
        cfg_merge=bool(inference["cfg_merge"]),
    )
    pipe_kwargs = {
        "prompt": str(case["caption"]),
        "negative_prompt": str(inference["negative_prompt"]),
        "context_video": context_pil,
        "seed": int(config["flow_matching"]["noise_seed"]),
        "tiled": bool(inference["tiled_vae"]),
        "tile_size": tuple(int(v) for v in inference["vae_tile_size"]),
        "tile_stride": tuple(
            int(v) for v in inference["vae_tile_stride"]
        ),
        "height": int(video_config["height"]),
        "width": int(video_config["width"]),
        "num_frames": int(video_config["num_frames"]),
        "num_inference_steps": int(
            inference["num_inference_steps"]
        ),
        "cfg_scale": float(inference["cfg_scale"]),
        "cfg_merge": bool(inference["cfg_merge"]),
        "sigma_shift": float(inference["sigma_shift"]),
        "object_context": object_context,
    }
    probe.install()
    try:
        with torch.no_grad():
            final_video = model.pipe(**pipe_kwargs)
    finally:
        probe.restore()
    if probe.capture is None:
        raise RuntimeError(
            f"Failed to capture t={inference['capture_target_timestep']}"
        )
    return {
        "case": case,
        "source_tensor": source_tensor,
        "source_frame_indices": source_indices,
        "context_frame_indices": context_indices,
        "object_debug": object_debug,
        "object_context_stats": single.tensor_stats(object_context),
        "capture": probe.capture,
        "final_video": final_video,
    }


def save_case(
    *,
    model,
    result: dict[str, Any],
    case_dir: Path,
    config: dict[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    output = config["pure_inference_output"]
    case_dir.mkdir(parents=True, exist_ok=True)
    capture = result["capture"]
    decoded = single.decode_latents(
        model.pipe,
        {
            "infer_xt": capture["xt"],
            "infer_x0": capture["x0_context_restored"],
        },
    )
    fps = int(output["fps"])
    quality = int(output["video_quality"])
    single.save_tensor_video(
        result["source_tensor"],
        case_dir / "source_video_49f.mp4",
        fps=fps,
        quality=quality,
    )
    single.save_tensor_video(
        decoded["infer_xt"],
        case_dir / "infer_xt_t832.mp4",
        fps=fps,
        quality=quality,
    )
    single.save_tensor_video(
        decoded["infer_x0"],
        case_dir / "infer_x0_t832.mp4",
        fps=fps,
        quality=quality,
    )
    save_video(
        result["final_video"],
        str(case_dir / "final_inference_40steps.mp4"),
        fps=fps,
        quality=quality,
    )
    tensor_payload = {
        name: value
        for name, value in capture.items()
        if isinstance(value, torch.Tensor) or value is None
    }
    if bool(output["save_latents"]):
        torch.save(tensor_payload, case_dir / "latents.pt")
    metadata = {
        "schema_version": 1,
        "case": result["case"],
        "checkpoint": str(checkpoint),
        "formula": "x0_pred = xt - sigma_t * v_cfg",
        "seed": int(config["flow_matching"]["noise_seed"]),
        "inference": config["pure_inference"],
        "source_frame_indices": result[
            "source_frame_indices"
        ].tolist(),
        "context_frame_indices": result[
            "context_frame_indices"
        ].tolist(),
        "capture": {
            name: (
                single.tensor_stats(value)
                if isinstance(value, torch.Tensor)
                else value
            )
            for name, value in capture.items()
            if name not in {
                "v_positive",
                "v_negative",
                "v_cfg",
                "x0_raw",
                "x0_context_restored",
                "xt",
            }
        },
        "capture_tensors": {
            name: single.tensor_stats(value)
            for name, value in capture.items()
            if isinstance(value, torch.Tensor)
        },
        "object_context": result["object_context_stats"],
        "object_debug": single.jsonable(result["object_debug"]),
        "decoder": {
            "method": "WanVideoVAE38.decode",
            "decoded_frames": {
                name: int(value.shape[2])
                for name, value in decoded.items()
            },
        },
    }
    (case_dir / "metadata.json").write_text(
        json.dumps(single.jsonable(metadata), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {
        "case_id": str(result["case"]["case_id"]),
        "dataset_source": str(
            result["case"].get("dataset_source", "configured")
        ),
        "caption": str(result["case"]["caption"]),
        "relative_dir": case_dir.name,
        "capture": metadata["capture"],
        "object_context": metadata["object_context"],
        "object_debug": metadata["object_debug"],
    }


def build_index(
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    training_root = Path(
        config["pure_inference_output"]["training_reference_root"]
    ).name
    sections = []
    for row_index, summary in enumerate(summaries):
        case_dir = html.escape(summary["relative_dir"])
        training_reference = (
            f"../{html.escape(training_root)}/{case_dir}/"
            "vae_predicted_x0.mp4"
        )
        cards = (
            (f"{case_dir}/source_video_49f.mp4", "输入视频 x0"),
            (training_reference, "训练加噪前向 x0(t=832)"),
            (f"{case_dir}/infer_xt_t832.mp4", "纯推理 xt(t=832)"),
            (f"{case_dir}/infer_x0_t832.mp4", "纯推理 x0(t=832)"),
            (
                f"{case_dir}/final_inference_40steps.mp4",
                "纯推理最终40步结果",
            ),
        )
        figures = "".join(
            "<figure><figcaption>"
            f"{html.escape(title)}</figcaption><video "
            f"preload='metadata' playsinline src='{src}' "
            f"data-row='{row_index}'></video></figure>"
            for src, title in cards
        )
        capture = summary["capture"]
        selected = summary["object_debug"].get(
            "xssc_amg_selected_counts",
            [],
        )
        sections.append(
            "<section>"
            f"<h2>{html.escape(summary['case_id'])}</h2>"
            f"<p>{html.escape(summary['dataset_source'])} · "
            f"target t={capture['requested_timestep']:.1f} · "
            f"parent t={capture['parent_timestep']:.4f} · "
            f"σ={capture['requested_sigma']:.6f} · "
            f"AMG={html.escape(str(selected))}</p>"
            f"<p class='caption'>{html.escape(summary['caption'])}</p>"
            f"<div class='controls'><button onclick='playRow({row_index},true)'>"
            "从头同步播放本行</button>"
            f"<button onclick='pauseRow({row_index})'>暂停本行</button>"
            f"<a href='{case_dir}/metadata.json'>metadata</a></div>"
            f"<div class='grid'>{figures}</div></section>"
        )
    inference = config["pure_inference"]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>xSSC纯推理 xt → v → x0 · t=832</title>
<style>
:root{{--bg:#f4f6f4;--paper:#fff;--ink:#202523;--muted:#65706a;--line:#c9d0cc;--accent:#176f62}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}header,main{{max-width:1800px;margin:auto;padding:18px 24px}}header{{border-bottom:1px solid var(--line)}}h1,h2,p{{margin:0}}h1{{font-size:25px}}h2{{font-size:18px;margin-bottom:3px}}header p,section>p{{color:var(--muted)}}section{{padding:20px 0;border-bottom:1px solid var(--line)}}.caption{{margin-top:3px}}.controls{{display:flex;gap:8px;align-items:center;margin:10px 0}}button{{border:1px solid #9aa59f;background:#fff;padding:7px 10px;font:inherit;cursor:pointer}}a{{color:var(--accent);font-weight:700;text-decoration:none}}.grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px}}figure{{margin:0;background:var(--paper);border:1px solid var(--line);padding:7px}}figcaption{{font-weight:700;margin-bottom:5px;min-height:40px}}video{{display:block;width:100%;aspect-ratio:16/9;background:#111}}code{{font-size:12px}}@media(max-width:1300px){{.grid{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>xSSC纯推理：xt → CFG v → x0</h1>
<p>seed={config['flow_matching']['noise_seed']} · {inference['num_inference_steps']}步 · CFG={inference['cfg_scale']} · target t={inference['capture_target_timestep']} · <a href="../index.html">返回总入口</a></p>
<p>官方40步最近节点为t=833.3333；页面中的t=832探针沿该Euler段插值后额外计算正/负CFG，不反馈到正式40步轨迹。</p></header>
<main>{''.join(sections)}</main><script>
const rows=i=>[...document.querySelectorAll(`video[data-row="${{i}}"]`)];
function playRow(i,restart){{rows(i).forEach(v=>{{if(restart)v.currentTime=0;v.play().catch(()=>{{}})}})}}
function pauseRow(i){{rows(i).forEach(v=>v.pause())}}
</script></body></html>"""


def main() -> None:
    cli = parse_args()
    config = cases_base.load_config(cli.config)
    output = config["pure_inference_output"]
    root = Path(output["root"]).expanduser().resolve()
    if root.is_dir() and any(root.iterdir()) and not bool(
        output["overwrite"]
    ):
        raise FileExistsError(
            f"Output is non-empty and overwrite=false: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "resolved_config.json").write_text(
        json.dumps(
            {
                key: value
                for key, value in config.items()
                if key != "_config_path"
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args = cases_base.build_training_args(config)
    model, checkpoint, _ = cases_base.load_model(args, config)
    configure_inference_model(model, config)

    summaries = []
    for case_index, raw_case in enumerate(config["cases"]):
        case = cases_base.require_mapping(
            raw_case,
            f"cases[{case_index}]",
        )
        print(
            f"[infer-xt-v-x0] {case_index + 1}/{len(config['cases'])} "
            f"{case['case_id']}",
            flush=True,
        )
        result = run_case(model=model, case=case, config=config)
        summaries.append(
            save_case(
                model=model,
                result=result,
                case_dir=root / cases_base.slug(str(case["case_id"])),
                config=config,
                checkpoint=checkpoint,
            )
        )
    (root / "summary.json").write_text(
        json.dumps(single.jsonable(summaries), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (root / "index.html").write_text(
        build_index(summaries, config),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "output": str(root),
            "cases": summaries,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
