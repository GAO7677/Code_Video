#!/usr/bin/env python3
"""Run config-driven training-style xt -> DiT v -> x0 diagnostics.

Run:
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  -m code_vjepa_vggt.train_xSSC.visualize_training_xt_v_x0_cases \
  --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/configs/xt_v_x0_two_cases.json
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

from code_vjepa_vggt import context_wan_v_newtrain as context_flow
from code_vjepa_vggt.train_xSSC import (
    train_xssc_context_slots_dinov3 as train,
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


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    config = require_mapping(
        json.loads(path.read_text(encoding="utf-8")),
        "config",
    )
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("Only schema_version=1 is supported")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("config.cases must be a non-empty list")
    case_ids = [str(require_mapping(case, "case").get("case_id", "")) for case in cases]
    if any(not value for value in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("Every case requires a unique non-empty case_id")
    config["_config_path"] = str(path)
    return config


def slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not result:
        raise ValueError(f"case_id cannot be converted to a path: {value!r}")
    return result


def build_training_args(config: dict[str, Any]) -> argparse.Namespace:
    model = require_mapping(config["model"], "model")
    arch = require_mapping(config["architecture"], "architecture")
    boxes = require_mapping(config["xssc_boxes"], "xssc_boxes")
    video = require_mapping(
        config["video_preprocessing"],
        "video_preprocessing",
    )
    runtime = require_mapping(config["runtime"], "runtime")

    parser = single.build_parser()
    args = parser.parse_args([])
    overrides = {
        "diffsynth_root": model["diffsynth_root"],
        "wan_root": model["wan_root"],
        "lora_checkpoint": model["wan_lora_checkpoint"],
        "xssc_root": model["xssc_root"],
        "xssc_config": model["xssc_config"],
        "xssc_checkpoint": model["xssc_checkpoint"],
        "dinov3_root": model["dinov3_root"],
        "dinov3_checkpoint": model["dinov3_checkpoint"],
        "xssc_input_size": int(arch["xssc_input_size"]),
        "xssc_max_time_steps": int(arch["xssc_max_time_steps"]),
        "xssc_box_source": boxes["source"],
        "xssc_box_cache_dir": boxes["cache_dir"],
        "xssc_filter_empty_amg": False,
        "object_lora_rank": int(arch["object_lora_rank"]),
        "object_lora_alpha": float(arch["object_lora_alpha"]),
        "object_lora_dropout": float(arch["object_lora_dropout"]),
        "xssc_slot_track_dropout": float(
            arch["xssc_slot_track_dropout"]
        ),
        "height": int(video["height"]),
        "width": int(video["width"]),
        "num_frames": int(video["num_frames"]),
        "fixed_num_context_frames": int(video["num_context_frames"]),
        "train_batch_size": 1,
        "no_context_ratio": 0.0,
        "lora_base_model": "dit",
        "lora_target_modules": arch["wan_lora_target_modules"],
        "lora_rank": int(arch["wan_lora_rank"]),
        "lora_alpha": int(arch["wan_lora_alpha"]),
        "extra_inputs": "input_image",
        "object_gate_init": float(arch["object_gate_init"]),
        "lambda_main": 1.0,
        "lambda_object_context_reg": 1.0e-4,
        "dataset_num_workers": 0,
        "diag_checkpoint": Path(model["wan_xssc_checkpoint"]),
        "diag_device": runtime["device"],
    }
    for name, value in overrides.items():
        setattr(args, name, value)
    return train.tvn.prepare_args(args)


def video_probe(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    result = {
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return result


def load_case_sample(
    case: dict[str, Any],
    video_config: dict[str, Any],
) -> dict[str, Any]:
    path = Path(case["video_path"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    num_frames = int(video_config["num_frames"])
    mode = str(video_config["sampling_mode"])
    if mode == "prefix":
        frames, source_indices = read_video_prefix(path, num_frames)
    elif mode == "uniform":
        frames, source_indices = read_video_uniform(path, num_frames)
    else:
        raise ValueError(f"Unsupported sampling_mode={mode!r}")
    if len(source_indices) != num_frames:
        raise RuntimeError(
            f"{path} yielded {len(source_indices)}/{num_frames} frames"
        )
    resize_mode = str(video_config["resize_mode"])
    if resize_mode not in {"stretch", "cover_crop"}:
        raise ValueError(f"Unsupported resize_mode={resize_mode!r}")
    target_hw = (
        int(video_config["height"]),
        int(video_config["width"]),
    )
    video = preprocess_video_rgb_uint8(
        frames,
        target_hw,
        value_range="minus_one_to_one",
        resize_mode=resize_mode,
    )
    num_context = int(video_config["num_context_frames"])
    if not 1 <= num_context <= num_frames:
        raise ValueError("num_context_frames must be within [1,num_frames]")
    local_indices = torch.arange(num_frames, dtype=torch.long)
    context_indices = torch.arange(num_context, dtype=torch.long)
    return {
        "video": video,
        "context_video": video[:, context_indices].contiguous(),
        "caption": str(case["caption"]),
        "video_path": str(path),
        "frame_indices": local_indices,
        "context_frame_indices": context_indices,
        "num_context_frames": num_context,
        "metadata": {
            "dataset_source": str(case.get("dataset_source", "configured")),
            "case_id": str(case["case_id"]),
            "source_video_path": str(path),
            "source_video": video_probe(path),
            "sampled_source_frame_indices": source_indices.tolist(),
            "sampling_mode": mode,
            "resize_mode": resize_mode,
            "target_resolution": [target_hw[1], target_hw[0]],
        },
    }


def timestep_from_config(pipe, flow: dict[str, Any]) -> tuple[int, torch.Tensor]:
    scheduler_steps = len(pipe.scheduler.timesteps)
    raw_index = flow.get("timestep_index")
    if raw_index is None:
        fraction = float(flow["timestep_fraction"])
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("timestep_fraction must be in [0,1]")
        index = int(round(fraction * (scheduler_steps - 1)))
    else:
        index = int(raw_index)
    if not 0 <= index < scheduler_steps:
        raise ValueError(
            f"timestep_index={index} outside [0,{scheduler_steps - 1}]"
        )
    timestep = pipe.scheduler.timesteps[index : index + 1].to(
        device=pipe.device,
        dtype=pipe.torch_dtype,
    )
    return index, timestep


def load_model(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[torch.nn.Module, Path, dict[str, Any]]:
    device = torch.device(str(config["runtime"]["device"]))
    model = train.build_model(args, SimpleNamespace(device=device))
    checkpoint = train.tvn._resolve_checkpoint_file(
        Path(config["model"]["wan_xssc_checkpoint"]).expanduser().resolve()
    )
    load_info = train.tvn._load_filtered_checkpoint_into_model(
        model,
        checkpoint,
        include_prefixes=("slot_norm.", "slot_projector.", "time_embedding."),
        include_substrings=(".object_cross_attn.", ".object_gate"),
    )
    expected = sum(
        1 for _, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if load_info["loaded_count"] != expected or load_info[
        "skipped_shape_mismatch"
    ]:
        raise RuntimeError(
            "Incomplete Wan+xSSC checkpoint: "
            f"loaded={load_info['loaded_count']}/{expected}, "
            f"shape_mismatch={len(load_info['skipped_shape_mismatch'])}"
        )
    model.to(device)
    mode = str(config["runtime"]["forward_mode"])
    if mode == "eval":
        model.eval()
    elif mode == "train":
        model.train()
    else:
        raise ValueError("runtime.forward_mode must be eval or train")
    return model, Path(checkpoint), load_info


def run_forward(
    *,
    model: torch.nn.Module,
    sample: dict[str, Any],
    case_index: int,
    checkpoint: Path,
    checkpoint_load: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    flow_config = require_mapping(
        config["flow_matching"],
        "flow_matching",
    )
    seed = int(flow_config["noise_seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    inputs_shared, inputs_posi = model._prepare_pipeline_sample(sample)[:2]
    context_video = inputs_shared["raw_sample"]["context_video"]
    if context_video.ndim == 4:
        context_video = context_video.unsqueeze(0)
    context_video = context_video.to(
        device=model.pipe.device,
        dtype=model.pipe.torch_dtype,
    )
    with torch.no_grad():
        object_context, slots = model._build_object_context(context_video)
    selected_counts = list(model._last_xssc_amg_selected_counts)
    require_nonempty = bool(
        config["xssc_boxes"]["require_nonempty_selected_masks"]
    )
    if require_nonempty and (
        not selected_counts or min(selected_counts) <= 0
    ):
        raise RuntimeError(
            f"{sample['metadata']['case_id']} has empty AMG masks: "
            f"{selected_counts}"
        )

    pipe = model.pipe
    input_latents = inputs_shared["input_latents"]
    timestep_index, timestep = timestep_from_config(pipe, flow_config)
    generator = torch.Generator(device=input_latents.device)
    generator.manual_seed(seed)
    noise = torch.randn(
        tuple(input_latents.shape),
        device=input_latents.device,
        dtype=input_latents.dtype,
        generator=generator,
    )
    (
        latent_xt,
        training_target,
        context_latent_indices,
        num_clean_prefix_latents,
        clean_prefix_latents,
    ) = single.apply_training_noise(
        pipe=pipe,
        inputs_shared=inputs_shared,
        input_latents=input_latents,
        noise=noise,
        timestep=timestep,
    )
    model_inputs = dict(inputs_shared)
    model_inputs["latents"] = latent_xt
    models = {
        name: getattr(pipe, name)
        for name in pipe.in_iteration_models
    }
    with torch.no_grad():
        velocity = pipe.model_fn(
            **models,
            **model_inputs,
            **inputs_posi,
            object_context=object_context,
            timestep=timestep,
        )
    predicted_x0_raw = context_flow._predict_x0_from_diffsynth_flow(
        scheduler=pipe.scheduler,
        latent_xt=latent_xt,
        model_output=velocity,
        timestep=timestep,
    )
    predicted_x0 = single.restore_condition_latents(
        latent=predicted_x0_raw,
        input_latents=input_latents,
        inputs_shared=inputs_shared,
        context_latent_indices=context_latent_indices,
        num_clean_prefix_latents=num_clean_prefix_latents,
        clean_prefix_latents=clean_prefix_latents,
    )
    prediction_slice, target_slice = single.supervised_slices(
        prediction=velocity,
        target=training_target,
        inputs_shared=inputs_shared,
        context_latent_indices=context_latent_indices,
        num_clean_prefix_latents=num_clean_prefix_latents,
    )
    supervised_v_mse = torch.nn.functional.mse_loss(
        prediction_slice.float(),
        target_slice.float(),
    )
    fully_noisy_xt = pipe.scheduler.add_noise(
        input_latents,
        noise,
        timestep,
    )
    oracle_x0 = context_flow._predict_x0_from_diffsynth_flow(
        scheduler=pipe.scheduler,
        latent_xt=fully_noisy_xt,
        model_output=training_target,
        timestep=timestep,
    )
    oracle_error = (
        oracle_x0.float() - input_latents.float()
    ).abs()
    sigma = context_flow._diffsynth_sigma_for_timestep(
        pipe.scheduler,
        timestep,
    )
    metadata = {
        "schema_version": 1,
        "formula": "x0_pred = xt - sigma_t * v_pred",
        "checkpoint": str(checkpoint),
        "checkpoint_load": checkpoint_load,
        "sample": {
            "requested_index": case_index,
            "actual_index": case_index,
            "empty_amg_resamples": 0,
            "dataset_source": sample["metadata"]["dataset_source"],
            "caption": sample["caption"],
            "video_path": sample["video_path"],
            "metadata": single.jsonable(sample["metadata"]),
            "frame_indices": single.jsonable(sample["frame_indices"]),
            "context_frame_indices": single.jsonable(
                sample["context_frame_indices"]
            ),
        },
        "flow": {
            "scheduler_steps": len(pipe.scheduler.timesteps),
            "timestep_index": timestep_index,
            "timestep_value": float(timestep.float().item()),
            "sigma": float(sigma.float().item()),
            "noise_seed": seed,
            "context_latent_indices": context_latent_indices,
            "num_clean_prefix_latents": num_clean_prefix_latents,
            "supervised_v_mse": float(supervised_v_mse.item()),
            "oracle_x0_mean_abs_error": float(
                oracle_error.mean().item()
            ),
            "oracle_x0_max_abs_error": float(oracle_error.max().item()),
        },
        "xssc": {
            "selected_amg_masks": selected_counts,
            "slots": single.tensor_stats(slots),
            "object_context": single.tensor_stats(object_context),
        },
        "tensors": {
            "input_x0": single.tensor_stats(input_latents),
            "noise": single.tensor_stats(noise),
            "training_xt": single.tensor_stats(latent_xt),
            "target_v": single.tensor_stats(training_target),
            "predicted_v": single.tensor_stats(velocity),
            "predicted_x0_raw": single.tensor_stats(predicted_x0_raw),
            "predicted_x0_context_restored": single.tensor_stats(
                predicted_x0
            ),
        },
    }
    tensors = {
        "input_x0": input_latents.detach().cpu().to(torch.float16),
        "noise": noise.detach().cpu().to(torch.float16),
        "training_xt": latent_xt.detach().cpu().to(torch.float16),
        "target_v": training_target.detach().cpu().to(torch.float16),
        "predicted_v": velocity.detach().cpu().to(torch.float16),
        "predicted_x0_raw": predicted_x0_raw.detach()
        .cpu()
        .to(torch.float16),
        "predicted_x0_context_restored": predicted_x0.detach()
        .cpu()
        .to(torch.float16),
    }
    return {
        "sample": sample,
        "metadata": metadata,
        "tensors": tensors,
    }


def save_case(
    *,
    model: torch.nn.Module,
    result: dict[str, Any],
    case_dir: Path,
    output_config: dict[str, Any],
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    tensors = result["tensors"]
    videos = single.decode_latents(
        model.pipe,
        {
            "ground_truth": tensors["input_x0"],
            "training_xt": tensors["training_xt"],
            "predicted_x0": tensors[
                "predicted_x0_context_restored"
            ],
        },
    )
    fps = int(output_config["fps"])
    quality = int(output_config["video_quality"])
    single.save_tensor_video(
        result["sample"]["video"],
        case_dir / "source_training_video.mp4",
        fps=fps,
        quality=quality,
    )
    for name, filename in (
        ("ground_truth", "vae_ground_truth_x0.mp4"),
        ("training_xt", "vae_training_xt.mp4"),
        ("predicted_x0", "vae_predicted_x0.mp4"),
    ):
        single.save_tensor_video(
            videos[name],
            case_dir / filename,
            fps=fps,
            quality=quality,
        )
    if bool(output_config["save_latents"]):
        torch.save(tensors, case_dir / "latents.pt")
    metadata = result["metadata"]
    metadata["decoder"] = {
        "method": "WanVideoVAE38.decode",
        "tiled": True,
        "tile_size": [30, 52],
        "tile_stride": [15, 26],
        "decoded_frames": {
            name: int(video.shape[2])
            for name, video in videos.items()
        },
    }
    (case_dir / "metadata.json").write_text(
        json.dumps(single.jsonable(metadata), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "index.html").write_text(
        single.build_page(metadata),
        encoding="utf-8",
    )
    return {
        "case_id": metadata["sample"]["metadata"]["case_id"],
        "relative_dir": case_dir.name,
        "dataset_source": metadata["sample"]["dataset_source"],
        "caption": metadata["sample"]["caption"],
        "flow": metadata["flow"],
        "xssc": metadata["xssc"],
    }


def build_index(
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    sections = []
    videos = (
        ("source_training_video.mp4", "训练输入"),
        ("vae_ground_truth_x0.mp4", "VAE(x0 GT)"),
        ("vae_training_xt.mp4", "VAE(xt)"),
        ("vae_predicted_x0.mp4", "VAE(x0 pred)"),
    )
    for row_index, summary in enumerate(summaries):
        base = html.escape(summary["relative_dir"])
        cells = "".join(
            "<figure><figcaption>"
            f"{html.escape(title)}</figcaption><video controls "
            f"preload='metadata' playsinline src='{base}/{filename}' "
            f"data-row='{row_index}'></video></figure>"
            for filename, title in videos
        )
        flow = summary["flow"]
        selected = summary["xssc"]["selected_amg_masks"]
        sections.append(
            "<section>"
            f"<h2>{html.escape(summary['case_id'])}</h2>"
            f"<p>{html.escape(summary['dataset_source'])} · "
            f"t={flow['timestep_value']:.1f} · σ={flow['sigma']:.4f} · "
            f"v-MSE={flow['supervised_v_mse']:.6f} · "
            f"AMG={html.escape(str(selected))}</p>"
            f"<p class='caption'>{html.escape(summary['caption'])}</p>"
            f"<div class='controls'><button onclick='playRow({row_index},true)'>"
            "从头同步播放本行</button>"
            f"<button onclick='pauseRow({row_index})'>暂停本行</button>"
            f"<a href='{base}/index.html'>详细数值</a></div>"
            f"<div class='grid'>{cells}</div></section>"
        )
    preprocessing = config["video_preprocessing"]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>xSSC训练 xt → v → x0 · 两个指定视频</title>
<style>
:root{{--bg:#f4f6f4;--paper:#fff;--ink:#202523;--muted:#65706a;--line:#c9d0cc;--accent:#176f62}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
header,main{{max-width:1720px;margin:auto;padding:18px 24px}}header{{border-bottom:1px solid var(--line)}}h1,h2,p{{margin:0}}h1{{font-size:25px}}h2{{font-size:18px;margin-bottom:3px}}header p,section>p{{color:var(--muted)}}section{{padding:20px 0;border-bottom:1px solid var(--line)}}.caption{{margin-top:3px}}.controls{{display:flex;gap:8px;align-items:center;margin:10px 0}}button{{border:1px solid #9aa59f;background:#fff;padding:7px 10px;font:inherit;cursor:pointer}}a{{color:var(--accent);font-weight:700;text-decoration:none}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}}figure{{margin:0;background:var(--paper);border:1px solid var(--line);padding:7px}}figcaption{{font-weight:700;margin-bottom:5px}}video{{display:block;width:100%;aspect-ratio:16/9;background:#111}}code{{font-size:12px}}@media(max-width:1100px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>xSSC训练单步：xt → DiT v → x0</h1>
<p>{len(summaries)}个指定视频 · {preprocessing['sampling_mode']} {preprocessing['num_frames']}帧 · 前{preprocessing['num_context_frames']}帧context · {preprocessing['resize_mode']}到{preprocessing['width']}×{preprocessing['height']} · <a href="../index.html">返回总入口</a></p>
<p>全部可变参数保存在 <code>resolved_config.json</code>；x0 pred是一轮训练前向的单噪声时刻估计，不是40步生成结果。</p></header>
<main>{''.join(sections)}</main><script>
const rows=i=>[...document.querySelectorAll(`video[data-row="${{i}}"]`)];
function playRow(i,restart){{rows(i).forEach(v=>{{if(restart)v.currentTime=0;v.play().catch(()=>{{}})}})}}
function pauseRow(i){{rows(i).forEach(v=>v.pause())}}
</script></body></html>"""


def main() -> None:
    cli = parse_args()
    config = load_config(cli.config)
    output_config = require_mapping(config["output"], "output")
    output_root = Path(output_config["root"]).expanduser().resolve()
    if (
        output_root.is_dir()
        and any(output_root.iterdir())
        and not bool(output_config["overwrite"])
    ):
        raise FileExistsError(
            f"Output is non-empty and output.overwrite=false: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.json").write_text(
        json.dumps(
            {key: value for key, value in config.items() if key != "_config_path"},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    args = build_training_args(config)
    model, checkpoint, load_info = load_model(args, config)
    video_config = require_mapping(
        config["video_preprocessing"],
        "video_preprocessing",
    )
    forward_results = []
    for case_index, raw_case in enumerate(config["cases"]):
        case = require_mapping(raw_case, f"cases[{case_index}]")
        sample = load_case_sample(case, video_config)
        print(
            f"[xt-v-x0] forward {case_index + 1}/{len(config['cases'])}: "
            f"{case['case_id']}",
            flush=True,
        )
        forward_results.append(
            run_forward(
                model=model,
                sample=sample,
                case_index=case_index,
                checkpoint=checkpoint,
                checkpoint_load=load_info,
                config=config,
            )
        )

    summaries = []
    for result in forward_results:
        case_id = result["sample"]["metadata"]["case_id"]
        print(f"[xt-v-x0] decode: {case_id}", flush=True)
        summaries.append(
            save_case(
                model=model,
                result=result,
                case_dir=output_root / slug(case_id),
                output_config=output_config,
            )
        )
    (output_root / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "index.html").write_text(
        build_index(summaries, config),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "config": config["_config_path"],
            "output": str(output_root),
            "cases": summaries,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
