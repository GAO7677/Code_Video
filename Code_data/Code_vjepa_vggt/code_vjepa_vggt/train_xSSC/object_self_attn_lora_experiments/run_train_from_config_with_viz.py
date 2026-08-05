#!/usr/bin/env python3
"""Training launcher that visualizes v -> x0 at every denoise step.

Usage:
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/run_train_from_config_with_viz.py \
    /home/gaoya/.../formal_full_sa_no_object_gpu27.json
"""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime, timezone
import http.server
import socketserver
import threading
from functools import partial
from pathlib import Path
import sys
from typing import Any

import torch


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from code_vjepa_vggt.train_xSSC.object_self_attn_lora_experiments import launch_from_config
from code_vjepa_vggt.context_wan_v_newtrain import _diffsynth_sigma_for_timestep
from code_vjepa_vggt.train_xSSC import visualize_training_xt_v_x0_dinov3 as vis_single
from code_vjepa_vggt.train_xSSC.object_self_attn_lora_experiments import train_xssc_object_self_attn_lora as exp
from code_vjepa_vggt.train_xSSC import train_xssc_context_slots

class _VizIndex:
    def __init__(self, root: Path, keep_last: int, fps: int, quality: int) -> None:
        self.root = root
        self.keep_last = keep_last
        self.fps = fps
        self.quality = quality
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._step = 0

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
        rows = []
        for item in reversed(self.records):
            item_dir = item["step_dir"]
            videos = item["videos"]
            caption = html.escape(item.get("caption", ""))
            sample = item["sample"]
            sample_desc = html.escape(
                f"{sample['case_id']} · step={item['step']} · t={item['timestep']:.1f} · σ={item['sigma']:.4f} · loss={item['loss']:.6f}"
            )
            source = html.escape(item.get("caption", ""))
            rows.append(
                f"""
                <section class=\"case\">
                  <h2>{sample_desc}</h2>
                  <p>{caption or source}</p>
                  <p>{sample['dataset_source']} · {sample['video_path']}</p>
                  <div class=\"row\">{"".join(
                    f"""
                    <figure>
                      <figcaption>{title}</figcaption>
                      <video controls preload=\"metadata\" src=\"{item_dir}/{name}\" width=\"320\" height=\"180\"></video>
                    </figure>
                    """
                    for title, name in videos
                )}</div>
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
setTimeout(()=>location.reload(), 2500);
</script>
</head>
<body>
<h1>Wan2.2 xSSC: 每次去噪 v → x0 可视化</h1>
<p id="meta">保存路径: <code>{html.escape(str(self.root))}</code>；每步保存1张，可滚动查看最近 {self.keep_last} 条。</p>
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
        (self.root / "index.html").write_text(self._build_html(), encoding="utf-8")

    def next_step(self) -> int:
        with self._lock:
            self._step += 1
            return self._step


def _serve_background(host: str, port: int, root: Path) -> tuple[socketserver.TCPServer, threading.Thread]:
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    server = socketserver.ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


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
        videos = vis_single.decode_latents(
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
    vis_single.save_tensor_video(
        videos["ground_truth_x0"].cpu(),
        step_root / "gt_x0.mp4",
        fps=viz.fps,
        quality=viz.quality,
    )
    vis_single.save_tensor_video(
        videos["training_xt"].cpu(),
        step_root / "x_t.mp4",
        fps=viz.fps,
        quality=viz.quality,
    )
    vis_single.save_tensor_video(
        videos["predicted_x0"].cpu(),
        step_root / "pred_x0.mp4",
        fps=viz.fps,
        quality=viz.quality,
    )

    sample = viz._extract_sample(inputs)
    viz.append(
        {
            "step": step,
            "loss": loss_value,
            "timestep": timestep_value,
            "sigma": sigma,
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


def _build_wrapper(viz: _VizIndex, viz_every_n_steps: int):
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
        noise = torch.randn_like(input_latents)
        training_target = pipe.scheduler.training_target(input_latents, noise, timestep)

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

        if context_latent_indices:
            latent_xt = pipe.scheduler.add_noise(input_latents, noise, timestep)
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
                timestep,
            )
            latent_xt = vis_single.context_flow.apply_clean_prefix_to_latents(
                latent_xt,
                clean_prefix_latents,
            )
        else:
            latent_xt = pipe.scheduler.add_noise(input_latents, noise, timestep)
            if "first_frame_latents" in inputs:
                latent_xt[:, :, 0:1] = inputs["first_frame_latents"]

        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        model_inputs = dict(inputs)
        model_inputs["latents"] = latent_xt
        noise_pred = pipe.model_fn(**models, **model_inputs, timestep=timestep)

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
        loss = loss * pipe.scheduler.training_weight(timestep)

        step = viz.next_step()
        if step % viz_every_n_steps == 0:
            try:
                _build_visualized_loss_record(
                    pipe=pipe,
                    step=step,
                    loss=loss,
                    timestep=timestep,
                    model_output=noise_pred,
                    latent_xt=latent_xt,
                    input_latents=input_latents,
                    inputs=inputs,
                    viz=viz,
                    context_latent_indices=context_latent_indices,
                    num_clean_prefix_latents=int(num_clean_prefix_latents),
                    clean_prefix_latents=clean_prefix_latents,
                )
            except Exception:
                pass

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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.viz_every_n_steps < 1:
        raise ValueError("--viz-every-n-steps must be >= 1")
    if args.viz_port <= 0 or args.viz_port > 65535:
        raise ValueError("--viz-port must be in [1, 65535]")

    config_path = args.config.expanduser().resolve()
    raw_config, sources = launch_from_config.load_config(config_path)
    resolved = launch_from_config.validate_config(raw_config, config_path.parent)
    run_tag = args.run_tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    output_dir = Path(resolved["paths"]["output_root"]) / resolved["experiment"]["name"] / run_tag
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
    for path in (project_root, diffsynth_root, Path(__file__).resolve().parents[4]):
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
    train_args = command[token_index + 1 :]

    if args.dry_run:
        print(json.dumps({
            "generated_output_dir": str(output_dir),
            "train_command": command,
            "train_args": train_args,
            "viz_root": str(
                args.viz_root.expanduser().resolve() / resolved["experiment"]["name"] / run_tag
            ),
        }, ensure_ascii=False, indent=2))
        return

    viz_output = (args.viz_root.expanduser().resolve() / resolved["experiment"]["name"]) / run_tag
    viz_output.mkdir(parents=True, exist_ok=False)
    (output_dir / "train_visualization_root.txt").write_text(
        str(viz_output), encoding="utf-8"
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps({"config_path": str(config_path), "resolved_config": resolved, "sources": sources}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    viz = _VizIndex(root=viz_output, keep_last=args.viz_keep_last, fps=args.viz_fps, quality=args.viz_quality)

    server, _ = _serve_background(args.viz_host, args.viz_port, viz_output)
    print(f"Visualization: python -m http.server --directory {viz_output} --bind {args.viz_host} {args.viz_port}")
    print(f"Open: http://localhost:{args.viz_port}/index.html")

    flow_with_viz, original_fn = _build_wrapper(viz, args.viz_every_n_steps)
    train_xssc_context_slots.flow_match_context_sft_loss = flow_with_viz

    original_argv = sys.argv[:]
    sys.argv = [str(Path(__file__).name)] + train_args
    try:
        exp.main()
    finally:
        train_xssc_context_slots.flow_match_context_sft_loss = original_fn
        sys.argv = original_argv
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
