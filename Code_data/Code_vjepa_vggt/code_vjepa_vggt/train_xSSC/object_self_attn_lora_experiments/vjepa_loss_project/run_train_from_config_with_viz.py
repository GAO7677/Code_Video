#!/usr/bin/env python3
"""Training launcher that visualizes v -> x0 at every denoise step.

Usage:
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/vjepa_loss_project/run_train_from_config_with_viz.py \
    /home/gaoya/.../formal_full_sa_no_object_gpu27.json
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import os
import shlex
import site
import sys
import traceback
from datetime import datetime, timezone
import http.server
import socketserver
import threading
from functools import partial
from pathlib import Path
from typing import Any


def _prefer_conda_site_packages() -> None:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    try:
        user_sites = site.getusersitepackages()
    except Exception:
        user_sites = []
    if isinstance(user_sites, str):
        user_sites = [user_sites]
    blocked = {str(Path(path).resolve()) for path in user_sites}
    sys.path[:] = [
        path
        for path in sys.path
        if str(Path(path or ".").resolve()) not in blocked
    ]
    site.ENABLE_USER_SITE = False


_prefer_conda_site_packages()

import torch


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TRAIN_XSSC_DIR = Path(__file__).resolve().parents[1]
if str(_TRAIN_XSSC_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_XSSC_DIR))

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from code_vjepa_vggt.train_xSSC.object_self_attn_lora_experiments import launch_from_config as _LaunchModule
    from code_vjepa_vggt.train_xSSC import visualize_training_xt_v_x0_dinov3 as _VisSingle
    from code_vjepa_vggt.train_xSSC.object_self_attn_lora_experiments import train_xssc_object_self_attn_lora as _ExpModule
    from code_vjepa_vggt.train_xSSC import train_xssc_context_slots as _ContextSlots


launch_from_config: _LaunchModule | None = None
_diffsynth_sigma_for_timestep = None
vis_single: _VisSingle | None = None
exp: _ExpModule | None = None
train_xssc_context_slots: _ContextSlots | None = None


def _bootstrap_launch_from_config() -> None:
    global launch_from_config
    if launch_from_config is None:
        from code_vjepa_vggt.train_xSSC.object_self_attn_lora_experiments import (
            launch_from_config as _launch_from_config,
        )
        launch_from_config = _launch_from_config


def _bootstrap_runtime_modules() -> None:
    global _diffsynth_sigma_for_timestep, vis_single, exp, train_xssc_context_slots
    if _diffsynth_sigma_for_timestep is not None:
        return
    from code_vjepa_vggt.context_wan_v_newtrain import _diffsynth_sigma_for_timestep as _sigma
    from code_vjepa_vggt.train_xSSC import (
        visualize_training_xt_v_x0_dinov3 as _vis_single,
    )
    from code_vjepa_vggt.train_xSSC.object_self_attn_lora_experiments import (
        launch_from_config as _launch_from_config,
        train_xssc_object_self_attn_lora as _exp,
    )
    from code_vjepa_vggt.train_xSSC import train_xssc_context_slots as _slots

    _diffsynth_sigma_for_timestep = _sigma
    vis_single = _vis_single
    launch_from_config = _launch_from_config
    exp = _exp
    train_xssc_context_slots = _slots


class _TinyVaeVideoDecoder:
    def __init__(self, checkpoint: Path, *, parallel: bool) -> None:
        self.checkpoint = checkpoint.expanduser().resolve()
        self.parallel = parallel
        self._model: Any | None = None

    def _load_model(self, pipe) -> Any:
        if self._model is not None:
            return self._model
        taehv_py = self.checkpoint.parent / "taehv.py"
        if not taehv_py.is_file():
            taehv_py = Path("/home/gaoya/Code_Video/taehv/taehv.py")
        if not taehv_py.is_file():
            raise FileNotFoundError(f"Cannot find taehv.py for tiny VAE: {taehv_py}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Cannot find tiny VAE checkpoint: {self.checkpoint}")

        spec = importlib.util.spec_from_file_location("_codex_taehv", str(taehv_py))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import tiny VAE module from {taehv_py}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.TAEHV(checkpoint_path=str(self.checkpoint)).eval()
        model = model.to(device=pipe.device, dtype=pipe.torch_dtype)
        self._model = model
        print(
            f"Tiny VAE decoder: {self.checkpoint} on {pipe.device} ({pipe.torch_dtype})",
            flush=True,
        )
        return model

    def decode(self, pipe, latents: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        model = self._load_model(pipe)
        output = {}
        with torch.no_grad():
            for name, latent in latents.items():
                value = latent.detach().to(device=pipe.device, dtype=pipe.torch_dtype)
                if value.ndim != 5:
                    raise ValueError(f"Expected latent [B,C,T,H,W], got {tuple(value.shape)}")
                value_ntchw = value.permute(0, 2, 1, 3, 4).contiguous()
                if int(value_ntchw.shape[2]) != int(model.latent_channels):
                    raise ValueError(
                        "Tiny VAE latent channel mismatch: "
                        f"got {value_ntchw.shape[2]}, expected {model.latent_channels} "
                        f"for {self.checkpoint.name}"
                    )
                decoded = model.decode_video(
                    value_ntchw,
                    parallel=self.parallel,
                    show_progress_bar=False,
                )
                output[name] = (
                    decoded.permute(0, 2, 1, 3, 4)
                    .contiguous()
                    .mul(2.0)
                    .sub(1.0)
                    .clamp_(-1.0, 1.0)
                    .cpu()
                )
        return output


class _VizIndex:
    def __init__(
        self,
        root: Path,
        keep_last: int,
        fps: int,
        quality: int,
        decoder: str,
        tiny_vae_checkpoint: Path,
        tiny_vae_parallel: bool,
    ) -> None:
        self.root = root
        self.keep_last = keep_last
        self.fps = fps
        self.quality = quality
        self.decoder = decoder
        self.tiny_vae = _TinyVaeVideoDecoder(
            tiny_vae_checkpoint,
            parallel=tiny_vae_parallel,
        )
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._step = 0

    def decode_latents(self, pipe, latents: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.decoder == "wan":
            return vis_single.decode_latents(pipe, latents)
        if self.decoder == "tiny-vae":
            return self.tiny_vae.decode(pipe, latents)
        raise ValueError(f"Unsupported decoder: {self.decoder}")

    def _record_dir(self, step: int) -> Path:
        return self.root / f"step_{step:07d}"

    def _extract_sample(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raw_sample = inputs.get("raw_sample", {})
        if not isinstance(raw_sample, dict):
            raw_sample = {}
        context_video = raw_sample.get("context_video")
        case_id = raw_sample.get("case_id")
        if case_id is None:
            video_path = raw_sample.get("video_path")
            if isinstance(video_path, (str, Path)):
                case_id = Path(video_path).stem
            else:
                case_id = "sample"
        return {
            "case_id": str(case_id),
            "dataset_source": str(raw_sample.get("dataset_source", "")),
            "video_path": str(raw_sample.get("video_path", "")),
            "context_frames": int(raw_sample.get("num_context_frames", 0)),
            "num_frames": int(raw_sample.get("num_frames", 0)),
            "context_video_present": bool(context_video is not None),
            "context_shape": list(context_video.shape)
            if torch.is_tensor(context_video)
            else None,
        }

    def _build_html(self) -> str:
        rows: list[str] = []
        for item in reversed(self.records):
            item_dir = item["step_dir"]
            videos = item["videos"]
            caption = html.escape(item.get("caption", ""))
            sample = item["sample"]
            sample_desc = html.escape(
                f"{sample['case_id']} · step={item['step']} · t={item['timestep']:.1f} · σ={item['sigma']:.4f} · loss={item['loss']:.6f} · fps={item.get('fps', self.fps)}"
            )
            source = html.escape(item.get("caption", ""))

            video_blocks: list[str] = []
            for title, name in videos:
                video_blocks.append(
                    f"""
                    <figure>
                      <figcaption>{html.escape(str(title))}</figcaption>
                      <video controls preload=\"metadata\" src=\"{item_dir}/{html.escape(str(name))}\" width=\"320\" height=\"180\"></video>
                    </figure>
                    """
                )

            rows.append(
                f"""
                <section class=\"case\">
                  <h2>{sample_desc}</h2>
                  <p>{caption or source}</p>
                  <p>{html.escape(sample['dataset_source'])} · {html.escape(sample['video_path'])}</p>
                  <div class=\"row\">{''.join(video_blocks)}</div>
                </section>
                """
            )

        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Train v->x0 (live)</title>
<style>
:root {{ --bg: #0f1319; --fg: #e7ecf3; --line: #2f3641; --card: #171c25; }}
body {{ margin: 0; padding: 18px; background: var(--bg); color: var(--fg); font: 14px/1.4 system-ui, -apple-system, Arial; }}
h1, h2 {{ margin: 4px 0; }}
.case {{ border: 1px solid var(--line); border-radius: 10px; padding: 12px; margin-bottom: 14px; background: var(--card); }}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 12px 0 16px; }}
.toolbar button {{ border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; background: #243044; color: var(--fg); font-weight: 700; cursor: pointer; }}
.toolbar button:hover {{ background: #30405c; }}
.row {{ display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; }}
figure {{ margin: 0; }}
figcaption {{ margin-bottom: 5px; }}
video {{ width: 100%; background: #000; border-radius: 8px; border: 1px solid var(--line); }}
#meta {{ color:#aeb4c3; margin: 0 0 12px 0; }}
code {{ color:#85c2ff; }}
@media (max-width: 1100px) {{ .row {{ grid-template-columns: repeat(2, minmax(220px, 1fr)); }} }}
@media (max-width: 700px) {{ .row {{ grid-template-columns: 1fr; }} }}
</style>
<script>
function refreshPage() {{
  location.reload();
}}
function replayAll() {{
  document.querySelectorAll("video").forEach((video) => {{
    video.pause();
    video.currentTime = 0;
    video.play().catch(() => {{}});
  }});
}}
</script>
</head>
<body>
<h1>Wan2.2 xSSC: 每次去噪 v → x0 可视化</h1>
<p id="meta">保存路径: <code>{html.escape(str(self.root))}</code>；decoder: <code>{html.escape(str(self.decoder))}</code>；每步保存1张，可滚动查看最近 {self.keep_last} 条。</p>
<div class="toolbar">
  <button type="button" onclick="refreshPage()">手动刷新</button>
  <button type="button" onclick="replayAll()">全部重新播放</button>
</div>
{''.join(rows)}
</body>
</html>"""

    def _write_json(self) -> None:
        payload = {
            "schema_version": 1,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "records": self.records,
        }
        (self.root / "records.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_index(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_json()
        (self.root / "index.html").write_text(self._build_html(), encoding="utf-8")

    def append(self, item: dict[str, Any]) -> None:
        with self._lock:
            self.records.append(item)
            if len(self.records) > self.keep_last:
                oldest = self.records.pop(0)
                oldest_dir = self.root / oldest["step_dir"]
                if oldest_dir.is_dir():
                    for child in oldest_dir.iterdir():
                        child.unlink()
                    oldest_dir.rmdir()
            self._write_json()
        self.write_index()

    def next_step(self) -> int:
        with self._lock:
            self._step += 1
            return self._step


def _serve_background(host: str, port: int, root: Path) -> tuple[socketserver.TCPServer, threading.Thread]:
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    class _ThreadingHTTPServer(
        socketserver.ThreadingTCPServer,
        http.server.HTTPServer,
    ):
        daemon_threads = True

    server = _ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _resolve_source_fps(inputs: dict[str, Any], fallback_fps: int) -> int:
    raw_sample = inputs.get("raw_sample", {})
    if not isinstance(raw_sample, dict):
        return int(fallback_fps)
    video_path = raw_sample.get("video_path")
    if not video_path:
        return int(fallback_fps)
    try:
        from decord import VideoReader

        fps = float(VideoReader(str(video_path)).get_avg_fps())
        if math.isfinite(fps) and fps > 0:
            return max(1, int(round(fps)))
    except Exception:
        pass
    return int(fallback_fps)


def _sample_high_noise_timestep(pipe, top_ratio: float) -> torch.Tensor:
    total = len(pipe.scheduler.timesteps)
    if total < 1:
        raise ValueError("scheduler.timesteps is empty")
    ratio = min(1.0, max(0.0, float(top_ratio)))
    top_k = max(1, min(total, int(math.ceil(total * ratio))))
    sigmas = pipe.scheduler.sigmas[:total].detach().float().cpu()
    candidate_indices = torch.topk(sigmas, k=top_k).indices
    chosen = int(candidate_indices[torch.randint(0, top_k, (1,)).item()].item())
    return pipe.scheduler.timesteps[chosen].to(
        dtype=pipe.torch_dtype,
        device=pipe.device,
    )


def _build_visualized_loss_record(*, pipe, step: int, loss: torch.Tensor, timestep: torch.Tensor, model_output: torch.Tensor, latent_xt: torch.Tensor, input_latents: torch.Tensor, inputs: dict[str, Any], viz: _VizIndex, context_latent_indices: list[int], num_clean_prefix_latents: int, clean_prefix_latents: torch.Tensor | None) -> None:
    sigma = float(_diffsynth_sigma_for_timestep(pipe.scheduler, timestep).item())
    timestep_value = float(timestep.detach().float().mean().item())
    if not torch.isfinite(loss).all():
        loss_value = float("nan")
    else:
        loss_value = float(loss.detach().float().mean().item())

    predicted_x0 = vis_single.context_flow._predict_x0_from_diffsynth_flow(
        scheduler=pipe.scheduler,
        latent_xt=latent_xt,
        model_output=model_output,
        timestep=timestep,
    )
    restored_x0 = vis_single.restore_condition_latents(
        latent=predicted_x0,
        input_latents=input_latents,
        inputs_shared=inputs,
        context_latent_indices=context_latent_indices,
        num_clean_prefix_latents=num_clean_prefix_latents,
        clean_prefix_latents=clean_prefix_latents,
    )

    step_root = viz._record_dir(step)
    step_root.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        videos = viz.decode_latents(
            pipe,
            {
                "ground_truth_x0": input_latents[:1].detach().to(
                    device=pipe.device,
                    dtype=pipe.torch_dtype,
                ),
                "training_xt": latent_xt[:1].detach().to(
                    device=pipe.device,
                    dtype=pipe.torch_dtype,
                ),
                "predicted_x0": restored_x0[:1].detach().to(
                    device=pipe.device,
                    dtype=pipe.torch_dtype,
                ),
            },
        )
    source_fps = _resolve_source_fps(inputs, viz.fps)
    vis_single.save_tensor_video(
        videos["ground_truth_x0"].cpu(),
        step_root / "gt_x0.mp4",
        fps=source_fps,
        quality=viz.quality,
    )
    vis_single.save_tensor_video(
        videos["training_xt"].cpu(),
        step_root / "x_t.mp4",
        fps=source_fps,
        quality=viz.quality,
    )
    vis_single.save_tensor_video(
        videos["predicted_x0"].cpu(),
        step_root / "pred_x0.mp4",
        fps=source_fps,
        quality=viz.quality,
    )

    sample = viz._extract_sample(inputs)
    viz.append(
        {
            "step": step,
            "loss": loss_value,
            "timestep": timestep_value,
            "sigma": sigma,
            "fps": source_fps,
            "step_dir": step_root.name,
            "caption": str(inputs.get("caption", "")),
            "sample": sample,
            "videos": [
                ("ground_truth_x0", "gt_x0.mp4"),
                ("x_t", "x_t.mp4"),
                ("pred_x0", "pred_x0.mp4"),
            ],
        }
    )


def _build_wrapper(
    viz: _VizIndex,
    viz_every_n_steps: int,
    *,
    viz_high_noise: bool,
    viz_high_noise_top_ratio: float,
):
    original = train_xssc_context_slots.flow_match_context_sft_loss

    def flow_match_context_sft_loss_with_visualization(pipe, **inputs):
        max_timestep_boundary = int(
            inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps)
        )
        min_timestep_boundary = int(
            inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps)
        )

        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = pipe.scheduler.timesteps[timestep_id].to(
            dtype=pipe.torch_dtype,
            device=pipe.device,
        )

        input_latents = inputs["input_latents"]
        clean_prefix_latents = inputs.get("clean_prefix_latents")
        num_clean_prefix_latents = vis_single.context_flow.resolve_num_clean_prefix_latents(
            clean_prefix_latents=clean_prefix_latents,
            num_clean_prefix_latents=inputs.get("num_clean_prefix_latents"),
        )
        context_latent_indices = vis_single.context_flow.resolve_context_latent_indices_from_frames(
            raw_frame_indices=inputs.get("context_frame_indices"),
            raw_num_frames=inputs.get("num_frames"),
            latent_length=input_latents.shape[2],
        )

        if num_clean_prefix_latents < 0 or num_clean_prefix_latents >= input_latents.shape[2]:
            raise ValueError(
                "num_clean_prefix_latents must be in [0, latent_length). "
                f"Got {num_clean_prefix_latents} for latent length {input_latents.shape[2]}."
            )
        if context_latent_indices and len(context_latent_indices) >= input_latents.shape[2]:
            raise ValueError(
                "context_latent_indices must leave at least one latent step for supervision. "
                f"Got {context_latent_indices} for latent length {input_latents.shape[2]}."
            )

        def forward_at_timestep(current_timestep: torch.Tensor, *, no_grad: bool):
            noise = torch.randn_like(input_latents)
            training_target = pipe.scheduler.training_target(
                input_latents,
                noise,
                current_timestep,
            )
            if context_latent_indices:
                latent_xt = pipe.scheduler.add_noise(
                    input_latents,
                    noise,
                    current_timestep,
                )
                latent_xt = vis_single.context_flow.apply_clean_latents_at_indices(
                    latent_xt,
                    input_latents,
                    context_latent_indices,
                )
            elif num_clean_prefix_latents > 0:
                latent_xt = input_latents.clone()
                latent_xt[:, :, num_clean_prefix_latents:] = pipe.scheduler.add_noise(
                    input_latents[:, :, num_clean_prefix_latents:],
                    noise[:, :, num_clean_prefix_latents:],
                    current_timestep,
                )
                latent_xt = vis_single.context_flow.apply_clean_prefix_to_latents(
                    latent_xt,
                    clean_prefix_latents,
                )
            else:
                latent_xt = pipe.scheduler.add_noise(input_latents, noise, current_timestep)
                if "first_frame_latents" in inputs:
                    latent_xt[:, :, 0:1] = inputs["first_frame_latents"]

            models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
            model_inputs = dict(inputs)
            model_inputs["latents"] = latent_xt
            if no_grad:
                with torch.no_grad():
                    noise_pred = pipe.model_fn(
                        **models,
                        **model_inputs,
                        timestep=current_timestep,
                    )
            else:
                noise_pred = pipe.model_fn(
                    **models,
                    **model_inputs,
                    timestep=current_timestep,
                )
            return latent_xt, training_target, noise_pred

        def loss_for(noise_pred: torch.Tensor, training_target: torch.Tensor, loss_timestep: torch.Tensor):
            if context_latent_indices:
                noise_pred_for_loss = vis_single.context_flow.slice_non_context_latents(
                    noise_pred,
                    latent_length=input_latents.shape[2],
                    context_latent_indices=context_latent_indices,
                )
                training_target_for_loss = vis_single.context_flow.slice_non_context_latents(
                    training_target,
                    latent_length=input_latents.shape[2],
                    context_latent_indices=context_latent_indices,
                )
            elif num_clean_prefix_latents > 0:
                noise_pred_for_loss = noise_pred[:, :, num_clean_prefix_latents:]
                training_target_for_loss = training_target[:, :, num_clean_prefix_latents:]
            elif "first_frame_latents" in inputs:
                noise_pred_for_loss = noise_pred[:, :, 1:]
                training_target_for_loss = training_target[:, :, 1:]
            else:
                noise_pred_for_loss = noise_pred
                training_target_for_loss = training_target

            loss = torch.nn.functional.mse_loss(
                noise_pred_for_loss.float(),
                training_target_for_loss.float(),
            )
            return loss * pipe.scheduler.training_weight(loss_timestep)

        latent_xt, training_target, noise_pred = forward_at_timestep(
            timestep,
            no_grad=False,
        )
        loss = loss_for(noise_pred, training_target, timestep)

        step = viz.next_step()
        if step % viz_every_n_steps == 0:
            try:
                viz_timestep = timestep
                viz_latent_xt = latent_xt
                viz_noise_pred = noise_pred
                viz_loss = loss
                if viz_high_noise:
                    viz_timestep = _sample_high_noise_timestep(
                        pipe,
                        viz_high_noise_top_ratio,
                    )
                    viz_latent_xt, viz_training_target, viz_noise_pred = forward_at_timestep(
                        viz_timestep,
                        no_grad=True,
                    )
                    viz_loss = loss_for(
                        viz_noise_pred,
                        viz_training_target,
                        viz_timestep,
                    )
                _build_visualized_loss_record(
                    pipe=pipe,
                    step=step,
                    loss=viz_loss,
                    timestep=viz_timestep,
                    model_output=viz_noise_pred,
                    latent_xt=viz_latent_xt,
                    input_latents=input_latents,
                    inputs=inputs,
                    viz=viz,
                    context_latent_indices=context_latent_indices,
                    num_clean_prefix_latents=int(num_clean_prefix_latents),
                    clean_prefix_latents=clean_prefix_latents,
                )
            except Exception:
                traceback.print_exc()

        return loss

    return flow_match_context_sft_loss_with_visualization, original


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="run config-driven training and visualize xt/v->x0 for each denoise step"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--viz-port", type=int, default=8765)
    parser.add_argument("--viz-host", default="0.0.0.0")
    parser.add_argument("--viz-root", type=Path, default=Path("/data/gaoya/agent-data/checkpoints/xssc_viz"))
    parser.add_argument("--viz-every-n-steps", type=int, default=1)
    parser.add_argument("--viz-keep-last", type=int, default=200)
    parser.add_argument("--viz-fps", type=int, default=6)
    parser.add_argument("--viz-quality", type=int, default=5)
    parser.add_argument(
        "--viz-decoder",
        choices=("wan", "tiny-vae"),
        default="wan",
        help="Decode visualization latents with the full Wan VAE or the downloaded tiny VAE.",
    )
    parser.add_argument(
        "--tiny-vae-checkpoint",
        type=Path,
        default=Path("/home/gaoya/Code_Video/taehv/taew2_2.pth"),
        help="Tiny VAE checkpoint. Wan2.2 5B should use taew2_2.pth.",
    )
    parser.add_argument(
        "--tiny-vae-parallel",
        action="store_true",
        help="Decode tiny VAE frames in parallel; faster but uses more VRAM.",
    )
    parser.add_argument(
        "--viz-high-noise",
        action="store_true",
        help="Visualize an extra high-noise forward pass without changing training loss.",
    )
    parser.add_argument(
        "--viz-high-noise-top-ratio",
        type=float,
        default=0.15,
        help="Sample visualization timesteps from the highest-sigma fraction.",
    )
    parser.add_argument(
        "--gpu-set",
        default=None,
        help="Override config launch.gpu_set, for example 2,3.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_worker_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="accelerate worker for v->x0 training visualization"
    )
    parser.add_argument("--as-worker", action="store_true")
    parser.add_argument("--viz-port", type=int, required=True)
    parser.add_argument("--viz-host", required=True)
    parser.add_argument("--viz-output", type=Path, required=True)
    parser.add_argument("--viz-every-n-steps", type=int, required=True)
    parser.add_argument("--viz-keep-last", type=int, required=True)
    parser.add_argument("--viz-fps", type=int, required=True)
    parser.add_argument("--viz-quality", type=int, required=True)
    parser.add_argument("--viz-decoder", choices=("wan", "tiny-vae"), required=True)
    parser.add_argument("--tiny-vae-checkpoint", type=Path, required=True)
    parser.add_argument("--tiny-vae-parallel", action="store_true")
    parser.add_argument("--viz-high-noise", action="store_true")
    parser.add_argument("--viz-high-noise-top-ratio", type=float, required=True)
    return parser.parse_known_args()


def _is_main_rank() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def _worker_main() -> None:
    args, train_args = parse_worker_args()
    if not args.as_worker:
        raise RuntimeError("worker mode requires --as-worker")
    if args.viz_every_n_steps < 1:
        raise ValueError("--viz-every-n-steps must be >= 1")
    if args.viz_port <= 0 or args.viz_port > 65535:
        raise ValueError("--viz-port must be in [1, 65535]")

    _bootstrap_runtime_modules()
    if (
        _diffsynth_sigma_for_timestep is None
        or vis_single is None
        or exp is None
        or train_xssc_context_slots is None
    ):
        raise RuntimeError("Failed to load runtime modules after PYTHONPATH setup")

    main_rank = _is_main_rank()
    server = None
    original_fn = None
    patched_loss_targets: list[tuple[Any, Any]] = []
    if main_rank:
        args.viz_output.mkdir(parents=True, exist_ok=True)
        viz = _VizIndex(
            root=args.viz_output,
            keep_last=args.viz_keep_last,
            fps=args.viz_fps,
            quality=args.viz_quality,
            decoder=args.viz_decoder,
            tiny_vae_checkpoint=args.tiny_vae_checkpoint,
            tiny_vae_parallel=bool(args.tiny_vae_parallel),
        )
        viz.write_index()
        server, _ = _serve_background(args.viz_host, args.viz_port, args.viz_output)
        print(
            f"Visualization: python -m http.server --directory {args.viz_output} "
            f"--bind {args.viz_host} {args.viz_port}",
            flush=True,
        )
        print(f"Open: http://localhost:{args.viz_port}/index.html", flush=True)
        flow_with_viz, original_fn = _build_wrapper(
            viz,
            args.viz_every_n_steps,
            viz_high_noise=bool(args.viz_high_noise),
            viz_high_noise_top_ratio=float(args.viz_high_noise_top_ratio),
        )
        target_modules = [
            train_xssc_context_slots,
            getattr(exp, "base", None),
            getattr(exp, "tvn", None),
            sys.modules.get("train_xssc_context_slots"),
            sys.modules.get("code_vjepa_vggt.train_xSSC.train_xssc_context_slots"),
            sys.modules.get("train_v_newtrain"),
            sys.modules.get("code_vjepa_vggt.train_v_newtrain"),
        ]
        seen_targets: set[int] = set()
        for module in target_modules:
            if module is None or id(module) in seen_targets:
                continue
            seen_targets.add(id(module))
            if hasattr(module, "flow_match_context_sft_loss"):
                patched_loss_targets.append(
                    (module, getattr(module, "flow_match_context_sft_loss"))
                )
                setattr(module, "flow_match_context_sft_loss", flow_with_viz)

    original_argv = sys.argv[:]
    sys.argv = [str(Path(__file__).name)] + train_args
    try:
        exp.main()
    finally:
        if main_rank:
            for module, original in patched_loss_targets:
                setattr(module, "flow_match_context_sft_loss", original)
        sys.argv = original_argv
        if server is not None:
            server.shutdown()
            server.server_close()


def main() -> None:
    if "--as-worker" in sys.argv:
        _worker_main()
        return

    args = parse_args()
    if args.viz_every_n_steps < 1:
        raise ValueError("--viz-every-n-steps must be >= 1")
    if args.viz_port <= 0 or args.viz_port > 65535:
        raise ValueError("--viz-port must be in [1, 65535]")

    _bootstrap_launch_from_config()
    if launch_from_config is None:
        raise RuntimeError("Failed to import launch_from_config")

    config_path = args.config.expanduser().resolve()
    raw_config, sources = launch_from_config.load_config(config_path)
    resolved = launch_from_config.validate_config(raw_config, config_path.parent)
    if args.gpu_set:
        gpu_ids = [item.strip() for item in str(args.gpu_set).split(",") if item.strip()]
        if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError(f"Invalid --gpu-set: {args.gpu_set}")
        if "4" in gpu_ids:
            raise ValueError("GPU 4 is prohibited by workspace rules")
        if len(gpu_ids) != int(resolved["launch"]["num_processes"]):
            raise ValueError(
                "--gpu-set must contain one GPU per configured process: "
                f"{len(gpu_ids)} vs {resolved['launch']['num_processes']}"
            )
        resolved["launch"]["gpu_set"] = ",".join(gpu_ids)
    run_tag = args.run_tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    output_dir = Path(resolved["paths"]["output_root"]) / resolved["experiment"]["name"] / run_tag
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=False)

    cache_root = Path(resolved["paths"]["cache_root"])
    cache_root.mkdir(parents=True, exist_ok=True)

    cache_dirs = {
        "HF_HOME": cache_root / "huggingface",
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root / "xdg",
    }
    for value in cache_dirs.values():
        value.mkdir(parents=True, exist_ok=True)
    if resolved["adaptation"]["enable_object_branch"]:
        Path(resolved["paths"]["xssc_box_cache_dir"]).mkdir(parents=True, exist_ok=True)

    # Mirror launch_from_config's environment setup so in-process launch can find modules.
    os.environ.update({key: str(value) for key, value in cache_dirs.items()})
    os.environ.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_VISIBLE_DEVICES": str(resolved["launch"]["gpu_set"]),
        }
    )
    if resolved["logging"].get("wandb_run_id"):
        os.environ["WANDB_RUN_ID"] = str(resolved["logging"]["wandb_run_id"])
        os.environ["WANDB_RESUME"] = str(resolved["logging"].get("wandb_resume", ""))
    project_root = Path(resolved["paths"]["project_root"]).resolve()
    diffsynth_root = Path(resolved["paths"]["diffsynth_root"]).resolve()
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            str(project_root),
            str(diffsynth_root),
            os.environ.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)
    train_xssc_dir = Path(__file__).resolve().parents[1]
    for path in (project_root, diffsynth_root, train_xssc_dir, Path(__file__).resolve().parents[4]):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    command = launch_from_config.build_command(resolved, output_dir)
    train_scripts = {
        str(launch_from_config.TRAIN_SCRIPT),
        str(launch_from_config.OFFICIAL_XSSC_OBJECT_ONLY_TRAIN_SCRIPT),
    }
    token_index = None
    for index, token in enumerate(command):
        if token in train_scripts:
            token_index = index
            break
    if token_index is None:
        raise RuntimeError(
            "cannot find training script token in generated command: "
            f"{sorted(train_scripts)}"
        )
    viz_output = (args.viz_root.expanduser().resolve() / resolved["experiment"]["name"]) / run_tag
    worker_prefix = [
        str(Path(__file__).resolve()),
        "--as-worker",
        "--viz-host",
        str(args.viz_host),
        "--viz-port",
        str(args.viz_port),
        "--viz-output",
        str(viz_output),
        "--viz-every-n-steps",
        str(args.viz_every_n_steps),
        "--viz-keep-last",
        str(args.viz_keep_last),
        "--viz-fps",
        str(args.viz_fps),
        "--viz-quality",
        str(args.viz_quality),
        "--viz-decoder",
        str(args.viz_decoder),
        "--tiny-vae-checkpoint",
        str(args.tiny_vae_checkpoint),
        "--viz-high-noise-top-ratio",
        str(args.viz_high_noise_top_ratio),
    ]
    if args.tiny_vae_parallel:
        worker_prefix.append("--tiny-vae-parallel")
    if args.viz_high_noise:
        worker_prefix.append("--viz-high-noise")
    command = command[:token_index] + worker_prefix + command[token_index + 1 :]

    if args.dry_run:
        print(json.dumps({
            "generated_output_dir": str(output_dir),
            "launch_command": command,
            "launch_command_text": shlex.join(command),
            "viz_root": str(viz_output),
        }, ensure_ascii=False, indent=2))
        return

    viz_output.mkdir(parents=True, exist_ok=False)
    (output_dir / "train_visualization_root.txt").write_text(
        str(viz_output), encoding="utf-8"
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps({"config_path": str(config_path), "resolved_config": resolved, "sources": sources}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update({key: str(value) for key, value in cache_dirs.items()})
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(resolved["launch"]["gpu_set"])
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if resolved["logging"].get("wandb_run_id"):
        env["WANDB_RUN_ID"] = str(resolved["logging"]["wandb_run_id"])
        env["WANDB_RESUME"] = str(resolved["logging"].get("wandb_resume", ""))
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(project_root),
            str(diffsynth_root),
            env.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)
    print(json.dumps({
        "experiment": resolved["experiment"]["name"],
        "gpu_set": resolved["launch"]["gpu_set"],
        "output_dir": str(output_dir),
        "viz_root": str(viz_output),
        "command": shlex.join(command),
    }, ensure_ascii=True, indent=2), flush=True)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
