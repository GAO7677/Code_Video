#!/usr/bin/env python3
"""Visualize one DINOv3 Wan+xSSC training forward as x0 -> xt -> v -> x0_pred.

Run:
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  -m code_vjepa_vggt.train_xSSC.visualize_training_xt_v_x0_dinov3
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from diffsynth.utils.data import save_video

from code_vjepa_vggt import context_wan_v_newtrain as context_flow
from code_vjepa_vggt.train_xSSC import train_xssc_context_slots_dinov3 as train


PROJ = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
XSSC_EXP_ROOT = PROJ / "code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256"
DEFAULT_WAN_CHECKPOINT = Path(
    "/data/gaoya/agent-data/checkpoints/train_xssc_context_slots_dinov3/"
    "formal_gpu01_20260722T143309Z/checkpoints/step-001500"
)
DEFAULT_XSSC_CHECKPOINT = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/"
    "dinov3_xSSC/restart_save1000_20260720T140029Z/"
    "movi_c_transfer15000_b64_acc3_20260721T134713Z/"
    "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/42/"
    "step-026000.pth"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/xssc-training-xt-v-x0"
)


def build_parser() -> argparse.ArgumentParser:
    parser = train.build_parser()
    parser.set_defaults(
        diffsynth_root=str(DIFFSYNTH_ROOT),
        wan_root="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B",
        xssc_root=str(XSSC_EXP_ROOT),
        xssc_config=str(
            XSSC_EXP_ROOT
            / "upstream/config-randsfq/"
            "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000.py"
        ),
        xssc_checkpoint=str(DEFAULT_XSSC_CHECKPOINT),
        dinov3_root=str(XSSC_EXP_ROOT / "third_party/dinov3"),
        dinov3_checkpoint=(
            "/data/gaoya/ckpt/facebook-dinov3-vitl16-pretrain-lvd1689m/"
            "model.safetensors"
        ),
        xssc_input_size=256,
        xssc_max_time_steps=64,
        xssc_box_source="amg",
        xssc_box_cache_dir=(
            "/data/gaoya/agent-data/cache/"
            "xssc_dinov3_context_amg_boxes_wan_train"
        ),
        xssc_filter_empty_amg=False,
        xssc_empty_amg_max_resample_attempts=20,
        object_lora_rank=32,
        object_lora_alpha=32.0,
        object_lora_dropout=0.05,
        xssc_slot_track_dropout=0.10,
        dataset_type="xssc_replay_mix",
        pybullet0713_root=(
            "/data/gaoya/AAA_test_video/Dataset_physV/"
            "0717pybullet_5000_vbenchtop5"
        ),
        pybullet0713_split="train",
        pybullet0713_sampling_strategy="prefix",
        kubric_root="/data/gaoya/dataset/nnsriram97-phyco_kubric",
        kubric_split="train",
        kubric_sampling_strategy="prefix",
        kubric_cache_root=(
            "/data/gaoya/agent-data/cache/kubric_no_gt_box_dataset"
        ),
        kubric_replay_index_num_frames=69,
        kubric_replay_index_num_context_frames=20,
        openvid_root=(
            "/data/gaoya/dataset/"
            "mvp-lab-OpenVidHD-0.4M-720p-48fps/train"
        ),
        mixture_pybullet_ratio=0.30,
        mixture_kubric_ratio=0.30,
        mixture_openvid_ratio=0.40,
        height=512,
        width=896,
        num_frames=49,
        fixed_num_context_frames=8,
        train_batch_size=1,
        no_context_ratio=0.0,
        lora_base_model="dit",
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=32,
        lora_alpha=32,
        lora_checkpoint=(
            "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/"
            "raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/"
            "step-000500/checkpoint.safetensors"
        ),
        extra_inputs="input_image",
        object_gate_init=0.1,
        lambda_main=1.0,
        lambda_object_context_reg=1.0e-4,
        dataset_num_workers=0,
    )
    group = parser.add_argument_group("xt_v_x0_diagnostic")
    group.add_argument(
        "--diag-checkpoint",
        type=Path,
        default=DEFAULT_WAN_CHECKPOINT,
        help="Wan+xSSC object-branch checkpoint directory or safetensors file.",
    )
    group.add_argument("--diag-sample-index", type=int, default=0)
    group.add_argument("--diag-noise-seed", type=int, default=42)
    group.add_argument(
        "--diag-timestep-fraction",
        type=float,
        default=0.5,
        help="Scheduler index fraction in [0,1]; ignored when timestep-index is set.",
    )
    group.add_argument("--diag-timestep-index", type=int)
    group.add_argument("--diag-device", default="cuda:0")
    group.add_argument("--diag-output", type=Path, default=DEFAULT_OUTPUT)
    group.add_argument("--diag-fps", type=int, default=30)
    group.add_argument("--diag-video-quality", type=int, default=8)
    group.add_argument("--diag-max-empty-amg-resamples", type=int, default=20)
    return parser


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(value).all().item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "abs_mean": float(value.abs().mean().item()),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def video_tensor_to_frames(video: torch.Tensor) -> list[np.ndarray]:
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5 or int(video.shape[0]) != 1:
        raise ValueError(f"Expected video [1,C,T,H,W], got {tuple(video.shape)}")
    frames = (
        video[0]
        .detach()
        .float()
        .clamp(-1.0, 1.0)
        .add(1.0)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .contiguous()
        .cpu()
        .numpy()
    )
    return [frame for frame in frames]


def save_tensor_video(
    video: torch.Tensor,
    path: Path,
    *,
    fps: int,
    quality: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_video(
        video_tensor_to_frames(video),
        str(path),
        fps=int(fps),
        quality=int(quality),
    )


def apply_training_noise(
    *,
    pipe,
    inputs_shared: dict[str, Any],
    input_latents: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int], int, torch.Tensor | None]:
    training_target = pipe.scheduler.training_target(
        input_latents,
        noise,
        timestep,
    )
    clean_prefix_latents = inputs_shared.get("clean_prefix_latents")
    num_clean_prefix_latents = context_flow.resolve_num_clean_prefix_latents(
        clean_prefix_latents=clean_prefix_latents,
        num_clean_prefix_latents=inputs_shared.get("num_clean_prefix_latents"),
    )
    context_latent_indices = (
        context_flow.resolve_context_latent_indices_from_frames(
            raw_frame_indices=inputs_shared.get("context_frame_indices"),
            raw_num_frames=inputs_shared.get("num_frames"),
            latent_length=input_latents.shape[2],
        )
    )
    if context_latent_indices:
        latent_xt = pipe.scheduler.add_noise(
            input_latents,
            noise,
            timestep,
        )
        latent_xt = context_flow.apply_clean_latents_at_indices(
            latent_xt,
            input_latents,
            context_latent_indices,
        )
    elif num_clean_prefix_latents > 0:
        latent_xt = input_latents.clone()
        latent_xt[:, :, num_clean_prefix_latents:] = pipe.scheduler.add_noise(
            input_latents[:, :, num_clean_prefix_latents:],
            noise[:, :, num_clean_prefix_latents:],
            timestep,
        )
        latent_xt = context_flow.apply_clean_prefix_to_latents(
            latent_xt,
            clean_prefix_latents,
        )
    else:
        latent_xt = pipe.scheduler.add_noise(
            input_latents,
            noise,
            timestep,
        )
        if "first_frame_latents" in inputs_shared:
            latent_xt[:, :, 0:1] = inputs_shared["first_frame_latents"]
    return (
        latent_xt,
        training_target,
        context_latent_indices,
        int(num_clean_prefix_latents),
        clean_prefix_latents,
    )


def restore_condition_latents(
    *,
    latent: torch.Tensor,
    input_latents: torch.Tensor,
    inputs_shared: dict[str, Any],
    context_latent_indices: list[int],
    num_clean_prefix_latents: int,
    clean_prefix_latents: torch.Tensor | None,
) -> torch.Tensor:
    output = latent.clone()
    if context_latent_indices:
        return context_flow.apply_clean_latents_at_indices(
            output,
            input_latents,
            context_latent_indices,
        )
    if num_clean_prefix_latents > 0:
        return context_flow.apply_clean_prefix_to_latents(
            output,
            clean_prefix_latents,
        )
    if "first_frame_latents" in inputs_shared:
        output[:, :, 0:1] = inputs_shared["first_frame_latents"]
    return output


def supervised_slices(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    inputs_shared: dict[str, Any],
    context_latent_indices: list[int],
    num_clean_prefix_latents: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if context_latent_indices:
        prediction = context_flow.slice_non_context_latents(
            prediction,
            latent_length=target.shape[2],
            context_latent_indices=context_latent_indices,
        )
        target = context_flow.slice_non_context_latents(
            target,
            latent_length=target.shape[2],
            context_latent_indices=context_latent_indices,
        )
    elif num_clean_prefix_latents > 0:
        prediction = prediction[:, :, num_clean_prefix_latents:]
        target = target[:, :, num_clean_prefix_latents:]
    elif "first_frame_latents" in inputs_shared:
        prediction = prediction[:, :, 1:]
        target = target[:, :, 1:]
    return prediction, target


def decode_latents(pipe, latents: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    pipe.load_models_to_device(["vae"])
    vae_dtype = next(pipe.vae.model.parameters()).dtype
    output = {}
    with torch.no_grad():
        for name, latent in latents.items():
            value = latent.to(device=pipe.device, dtype=vae_dtype)
            decoded = pipe.vae.decode_framewise(value, device=pipe.device)
            output[name] = decoded.clamp_(-1, 1).cpu()
    return output


def build_page(metadata: dict[str, Any]) -> str:
    sample = metadata["sample"]
    flow = metadata["flow"]
    cards = (
        ("source_training_video.mp4", "训练样本", "数据集送入训练前的49帧视频"),
        ("vae_ground_truth_x0.mp4", "VAE(x0 GT)", "真实clean latent经VAE解码"),
        ("vae_training_xt.mp4", "VAE(xt)", "按训练规则加噪，并保留clean context latent"),
        ("vae_predicted_x0.mp4", "VAE(x0 pred)", "DiT输出v后，由xt - sigma·v反推"),
    )
    figures = "".join(
        f"<figure><figcaption><strong>{html.escape(title)}</strong>"
        f"<span>{html.escape(note)}</span></figcaption>"
        f"<video controls preload='metadata' playsinline src='{path}'></video></figure>"
        for path, title, note in cards
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>xSSC 训练单步 xt → v → x0</title>
<style>
:root{{--bg:#f4f6f4;--paper:#fff;--ink:#202523;--muted:#66706b;--line:#c8cfcb;--accent:#176f62}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif;letter-spacing:0}}
header,main{{max-width:1500px;margin:auto;padding:18px 24px}}header{{border-bottom:1px solid var(--line)}}h1,h2,p{{margin:0}}h1{{font-size:25px}}
.top{{display:flex;justify-content:space-between;align-items:center;gap:15px}}a{{color:var(--accent);font-weight:700;text-decoration:none}}.meta{{color:var(--muted);margin-top:4px}}
.bar{{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}}button{{border:1px solid #97a39d;background:#fff;padding:7px 11px;font:inherit;cursor:pointer}}
.videos{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}figure{{margin:0;border:1px solid var(--line);background:var(--paper);padding:9px}}
figcaption{{display:flex;justify-content:space-between;gap:10px;margin-bottom:7px}}figcaption span{{color:var(--muted);font-size:12px}}video{{display:block;width:100%;background:#111;aspect-ratio:16/9}}
.facts{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));border:1px solid var(--line);background:#fff;margin-top:18px}}.fact{{padding:10px 12px;border-right:1px solid var(--line)}}.fact:last-child{{border-right:0}}.fact span{{display:block;color:var(--muted);font-size:12px}}.fact strong{{font-size:17px}}
.note{{margin-top:14px;color:var(--muted)}}code{{font-size:12px}}@media(max-width:850px){{.videos{{grid-template-columns:1fr}}.facts{{grid-template-columns:1fr 1fr}}}}
</style></head><body><header><div class="top"><h1>xSSC训练单步：xt → DiT v → x0</h1><a href="../index.html">返回总入口</a></div>
<p class="meta">{html.escape(str(sample["dataset_source"]))} · sample index {sample["requested_index"]} · noise seed {flow["noise_seed"]}</p></header>
<main><div class="bar"><button onclick="playAll(true)">从头同步播放</button><button onclick="playAll(false)">继续播放</button><button onclick="pauseAll()">暂停</button></div>
<div class="videos">{figures}</div>
<div class="facts"><div class="fact"><span>Scheduler index</span><strong>{flow["timestep_index"]}/{flow["scheduler_steps"] - 1}</strong></div>
<div class="fact"><span>Timestep / sigma</span><strong>{flow["timestep_value"]:.1f} / {flow["sigma"]:.4f}</strong></div>
<div class="fact"><span>训练监督 v-MSE</span><strong>{flow["supervised_v_mse"]:.6f}</strong></div>
<div class="fact"><span>公式校验 max error</span><strong>{flow["oracle_x0_max_abs_error"]:.2e}</strong></div></div>
<p class="note">预测x0只是一轮训练前向在单个噪声时刻的估计，不是完成40步去噪后的生成结果。Context对应的latent不参与监督，展示前已按训练语义恢复为clean latent。</p>
</main><script>
const videos=[...document.querySelectorAll("video")];
function playAll(restart){{videos.forEach(v=>{{if(restart)v.currentTime=0;v.play().catch(()=>{{}})}})}}
function pauseAll(){{videos.forEach(v=>v.pause())}}
</script></body></html>"""


def main() -> None:
    parser = build_parser()
    args = train.tvn.prepare_args(parser.parse_args())
    if not 0.0 <= float(args.diag_timestep_fraction) <= 1.0:
        parser.error("--diag-timestep-fraction must be in [0,1]")
    device = torch.device(args.diag_device)
    if device.type != "cuda":
        parser.error("This diagnostic requires a CUDA device")
    torch.manual_seed(int(args.diag_noise_seed))
    np.random.seed(int(args.diag_noise_seed))

    output = args.diag_output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = train.base.build_dataset(args)
    requested_index = int(args.diag_sample_index) % len(dataset)

    model = train.build_model(
        args,
        SimpleNamespace(device=device),
    )
    checkpoint = train.tvn._resolve_checkpoint_file(
        args.diag_checkpoint.expanduser().resolve()
    )
    load_info = train.tvn._load_filtered_checkpoint_into_model(
        model,
        checkpoint,
        include_prefixes=("slot_norm.", "slot_projector.", "time_embedding."),
        include_substrings=(".object_cross_attn.", ".object_gate"),
    )
    expected_count = sum(
        1 for _, parameter in model.named_parameters() if parameter.requires_grad
    )
    if (
        load_info["loaded_count"] != expected_count
        or load_info["skipped_shape_mismatch"]
    ):
        raise RuntimeError(
            "Incomplete Wan+xSSC checkpoint: "
            f"loaded={load_info['loaded_count']}/{expected_count}, "
            f"shape_mismatch={len(load_info['skipped_shape_mismatch'])}"
        )
    model.to(device)
    model.eval()
    model.set_empty_amg_resample_dataset(dataset)

    sample = dataset[requested_index]
    actual_index: int | None = requested_index
    empty_resamples = 0
    while True:
        prepared = model._prepare_pipeline_sample(sample)
        inputs_shared, inputs_posi = prepared[0], prepared[1]
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
        if selected_counts and min(selected_counts) > 0:
            break
        empty_resamples += 1
        if empty_resamples > int(args.diag_max_empty_amg_resamples):
            raise RuntimeError(
                "Could not find a non-empty AMG training sample after "
                f"{empty_resamples} replacements"
            )
        sample = model._sample_empty_amg_replacement()
        actual_index = None

    pipe = model.pipe
    input_latents = inputs_shared["input_latents"]
    scheduler_steps = len(pipe.scheduler.timesteps)
    if args.diag_timestep_index is None:
        timestep_index = int(
            round(float(args.diag_timestep_fraction) * (scheduler_steps - 1))
        )
    else:
        timestep_index = int(args.diag_timestep_index)
    if not 0 <= timestep_index < scheduler_steps:
        raise ValueError(
            f"timestep index {timestep_index} outside [0,{scheduler_steps - 1}]"
        )
    timestep = pipe.scheduler.timesteps[timestep_index : timestep_index + 1].to(
        device=pipe.device,
        dtype=pipe.torch_dtype,
    )
    generator = torch.Generator(device=input_latents.device)
    generator.manual_seed(int(args.diag_noise_seed))
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
    ) = apply_training_noise(
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
    predicted_x0 = restore_condition_latents(
        latent=predicted_x0_raw,
        input_latents=input_latents,
        inputs_shared=inputs_shared,
        context_latent_indices=context_latent_indices,
        num_clean_prefix_latents=num_clean_prefix_latents,
        clean_prefix_latents=clean_prefix_latents,
    )
    prediction_slice, target_slice = supervised_slices(
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
    oracle_error = (oracle_x0.float() - input_latents.float()).abs()
    sigma = context_flow._diffsynth_sigma_for_timestep(
        pipe.scheduler,
        timestep,
    )

    videos = decode_latents(
        pipe,
        {
            "ground_truth": input_latents,
            "training_xt": latent_xt,
            "predicted_x0": predicted_x0,
        },
    )
    source_video = sample["video"]
    save_tensor_video(
        source_video,
        output / "source_training_video.mp4",
        fps=args.diag_fps,
        quality=args.diag_video_quality,
    )
    for name, filename in (
        ("ground_truth", "vae_ground_truth_x0.mp4"),
        ("training_xt", "vae_training_xt.mp4"),
        ("predicted_x0", "vae_predicted_x0.mp4"),
    ):
        save_tensor_video(
            videos[name],
            output / filename,
            fps=args.diag_fps,
            quality=args.diag_video_quality,
        )

    torch.save(
        {
            "input_x0": input_latents.detach().cpu().to(torch.float16),
            "noise": noise.detach().cpu().to(torch.float16),
            "training_xt": latent_xt.detach().cpu().to(torch.float16),
            "target_v": training_target.detach().cpu().to(torch.float16),
            "predicted_v": velocity.detach().cpu().to(torch.float16),
            "predicted_x0_raw": predicted_x0_raw.detach().cpu().to(torch.float16),
            "predicted_x0_context_restored": predicted_x0.detach()
            .cpu()
            .to(torch.float16),
        },
        output / "latents.pt",
    )
    metadata = {
        "schema_version": 1,
        "formula": "x0_pred = xt - sigma_t * v_pred",
        "checkpoint": str(checkpoint),
        "checkpoint_load": load_info,
        "sample": {
            "requested_index": requested_index,
            "actual_index": actual_index,
            "empty_amg_resamples": empty_resamples,
            "dataset_source": sample.get("metadata", {}).get(
                "dataset_source",
                "unknown",
            ),
            "caption": str(sample.get("caption", "")),
            "video_path": str(sample.get("video_path", "")),
            "metadata": jsonable(sample.get("metadata", {})),
            "frame_indices": jsonable(sample.get("frame_indices")),
            "context_frame_indices": jsonable(
                sample.get("context_frame_indices")
            ),
        },
        "flow": {
            "scheduler_steps": scheduler_steps,
            "timestep_index": timestep_index,
            "timestep_value": float(timestep.detach().float().item()),
            "sigma": float(sigma.detach().float().item()),
            "noise_seed": int(args.diag_noise_seed),
            "context_latent_indices": context_latent_indices,
            "num_clean_prefix_latents": num_clean_prefix_latents,
            "supervised_v_mse": float(supervised_v_mse.detach().item()),
            "oracle_x0_mean_abs_error": float(oracle_error.mean().item()),
            "oracle_x0_max_abs_error": float(oracle_error.max().item()),
        },
        "xssc": {
            "selected_amg_masks": selected_counts,
            "slots": tensor_stats(slots),
            "object_context": tensor_stats(object_context),
        },
        "tensors": {
            "input_x0": tensor_stats(input_latents),
            "noise": tensor_stats(noise),
            "training_xt": tensor_stats(latent_xt),
            "target_v": tensor_stats(training_target),
            "predicted_v": tensor_stats(velocity),
            "predicted_x0_raw": tensor_stats(predicted_x0_raw),
            "predicted_x0_context_restored": tensor_stats(predicted_x0),
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(jsonable(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(
        build_page(metadata),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "output": str(output),
            "checkpoint": str(checkpoint),
            "dataset_source": metadata["sample"]["dataset_source"],
            "timestep_index": timestep_index,
            "timestep_value": metadata["flow"]["timestep_value"],
            "sigma": metadata["flow"]["sigma"],
            "supervised_v_mse": metadata["flow"]["supervised_v_mse"],
            "oracle_x0_max_abs_error": metadata["flow"][
                "oracle_x0_max_abs_error"
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
