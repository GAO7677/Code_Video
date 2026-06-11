from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from code_vjepa_vggt.adapters.jepa_adapter import JEPAPatchAdapter
from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackAdapter
from code_vjepa_vggt.data import BallBlockVideoDataset
from code_vjepa_vggt.models.context_fuser import ContextTokenFuser
from code_vjepa_vggt.models.object_tokens import ObjectTubeProjector
from code_vjepa_vggt.models.wan_context_model import WanContextVideoModel
from code_vjepa_vggt.utils.masks import broadcast_latent_mask, latent_frame_mask


@dataclass
class TrainerState:
    step: int = 0


class ContextVideoTrainer:
    def __init__(self, cfg: dict[str, Any], build_optimizer: bool = True) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_optimizer = build_optimizer

        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        opt_cfg = cfg["optimization"]

        self.bundle = WanContextVideoModel(
            ckpt_dir=model_cfg["wan_ckpt_dir"],
            task=model_cfg["wan_task"],
            device=str(self.device),
        )
        self.bundle.freeze_parts(
            freeze_vae=model_cfg["freeze_vae"],
            freeze_text_encoder=model_cfg["freeze_text_encoder"],
            freeze_dit=model_cfg["freeze_wan_dit"],
        )
        self.bundle.dit.train(mode=build_optimizer)

        cond_dim = int(model_cfg.get("cond_proj_dim", self.bundle.config.text_dim))
        if cond_dim != self.bundle.config.text_dim:
            raise ValueError(f"cond_proj_dim must match Wan text_dim={self.bundle.config.text_dim}, got {cond_dim}")

        self.jepa_adapter = JEPAPatchAdapter(
            ckpt_path=str(Path(model_cfg["je_pa_ckpt_dir"]) / "original" / "model.pth"),
            device=str(self.device),
            crop_size=int(model_cfg["jepa_input_size"]),
            num_frames=int(data_cfg["num_context_frames"]),
            patch_size=int(model_cfg["jepa_patch_size"]),
            tubelet_size=int(model_cfg["jepa_tubelet_size"]),
        ).to(self.device)
        self.vggt_adapter = VGGTTrackAdapter(
            model_path=model_cfg.get("vggt_model_path"),
            num_queries=int(model_cfg["object_num_queries"]),
            device=str(self.device),
            input_hw=tuple(model_cfg["vggt_input_hw"]),
        ).to(self.device)
        self.object_pooler = ObjectTubeProjector(
            jepa_dim=self.jepa_adapter.encoder.backbone.embed_dim,
            latent_dim=self.bundle.config.in_dim,
            out_dim=cond_dim,
            jepa_window_radius=int(model_cfg["jepa_window_radius"]),
            latent_window_radius=int(model_cfg["latent_window_radius"]),
        ).to(self.device)
        self.context_fuser = ContextTokenFuser(
            text_dim=cond_dim,
            max_context_len=self.bundle.config.text_len,
        ).to(self.device)

        self.dataset = BallBlockVideoDataset(
            root=data_cfg["root"],
            num_frames=data_cfg["num_frames"],
            num_context_frames=data_cfg["num_context_frames"],
            resolution=tuple(data_cfg["resolution"]),
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=data_cfg["batch_size"],
            shuffle=True,
            num_workers=data_cfg["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

        self.optimizer = None
        if build_optimizer:
            trainable_params = list(self.bundle.dit.parameters())
            trainable_params += list(self.context_fuser.parameters())
            trainable_params += list(self.object_pooler.parameters())
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=opt_cfg["lr"],
                weight_decay=opt_cfg["weight_decay"],
                betas=tuple(opt_cfg["betas"]),
                eps=opt_cfg["eps"],
            )
        self.state = TrainerState()

    def _encode_text(self, captions: list[str]) -> list[torch.Tensor]:
        with torch.no_grad():
            ctx = self.bundle.text_encoder(captions, self.device)
        return [u.to(self.device) for u in ctx]

    def _encode_video_latents(self, videos_bcthw: torch.Tensor) -> list[torch.Tensor]:
        videos_list = [u.to(self.device) for u in videos_bcthw]
        with torch.no_grad():
            zs = self.bundle.vae.encode(videos_list)
        return zs

    @staticmethod
    def _shape_list(tensors: list[torch.Tensor]) -> list[list[int]]:
        return [list(t.shape) for t in tensors]

    def _prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        videos = batch["video"].to(self.device)
        context_videos = batch["context_video"].to(self.device)
        captions = list(batch["caption"])
        num_context_frames = int(batch["num_context_frames"][0]) if torch.is_tensor(batch["num_context_frames"]) else int(batch["num_context_frames"])

        text_ctx = self._encode_text(captions)
        full_latents = self._encode_video_latents(videos)
        context_latents = self._encode_video_latents(context_videos)
        context_latent_batch = torch.stack(context_latents, dim=0)

        jepa_out = self.jepa_adapter(context_videos)
        frames_bthwc = context_videos.permute(0, 2, 3, 4, 1).float()
        frames_bthwc = (frames_bthwc + 1.0) / 2.0
        vggt_out = self.vggt_adapter(frames_bthwc)
        object_out = self.object_pooler(
            jepa_patch_tokens=jepa_out.patch_tokens,
            context_latents=context_latent_batch,
            tracks=vggt_out.tracks,
            visibility=vggt_out.visibility,
            confidence=vggt_out.confidence,
            track_image_hw=vggt_out.image_hw,
        )
        fused_context = self.context_fuser(text_ctx, object_out.object_tokens)

        debug = {
            "video": list(videos.shape),
            "context_video": list(context_videos.shape),
            "text_context": self._shape_list(text_ctx),
            "full_latents": self._shape_list(full_latents),
            "context_latents": self._shape_list(context_latents),
            "jepa_patch_tokens": list(jepa_out.patch_tokens.shape),
            "vggt_query_points": list(vggt_out.query_points.shape),
            "vggt_tracks": list(vggt_out.tracks.shape),
            "vggt_visibility": list(vggt_out.visibility.shape),
            "object_tokens": list(object_out.object_tokens.shape),
            "object_jepa_tokens": list(object_out.jepa_tokens.shape),
            "object_latent_tokens": list(object_out.latent_tokens.shape),
            "object_geom_tokens": list(object_out.geom_tokens.shape),
            "fused_context": self._shape_list(fused_context),
            "vggt_used_model": bool(vggt_out.used_model),
            "vggt_track_image_hw": list(vggt_out.image_hw),
            "video_path": batch["video_path"][0] if isinstance(batch["video_path"], list) else batch["video_path"],
            "frame_indices": batch["frame_indices"][0].tolist() if batch["frame_indices"].ndim == 2 else batch["frame_indices"].tolist(),
            "caption": captions[0] if captions else "",
        }
        return {
            "videos": videos,
            "context_videos": context_videos,
            "captions": captions,
            "num_context_frames": num_context_frames,
            "full_latents": full_latents,
            "context_latents": context_latents,
            "fused_context": fused_context,
            "debug": debug,
        }

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        if self.optimizer is None:
            raise RuntimeError("trainer was created without optimizer")

        prepared = self._prepare_batch(batch)
        videos = prepared["videos"]
        num_context_frames = prepared["num_context_frames"]
        full_latents = prepared["full_latents"]
        context_latents = prepared["context_latents"]
        fused_context = prepared["fused_context"]

        losses = []
        for i, latent_clean in enumerate(full_latents):
            context_clean = context_latents[i]
            noise = torch.randn_like(latent_clean)
            timestep_scalar = torch.randint(
                low=0,
                high=self.bundle.config.num_train_timesteps,
                size=(1,),
                device=self.device,
            ).float()
            t_norm = timestep_scalar / float(self.bundle.config.num_train_timesteps)
            x_t = (1.0 - t_norm) * latent_clean + t_norm * noise

            context_mask_t, future_mask_t = latent_frame_mask(
                num_video_frames=videos.shape[2],
                num_context_frames=num_context_frames,
                vae_stride_t=self.bundle.config.vae_stride[0],
                device=self.device,
            )
            context_mask = broadcast_latent_mask(context_mask_t, latent_clean)
            future_mask = broadcast_latent_mask(future_mask_t, latent_clean)
            x_t = context_mask * context_clean + (1.0 - context_mask) * x_t

            seq_len = x_t.shape[1] * x_t.shape[2] * x_t.shape[3] // (
                self.bundle.config.patch_size[1] * self.bundle.config.patch_size[2]
            )
            t_tokens = torch.full((1, seq_len), timestep_scalar.item(), device=self.device)
            pred = self.bundle.dit(
                [x_t],
                t=t_tokens,
                context=[fused_context[i]],
                seq_len=seq_len,
                y=None,
            )[0]

            target = noise
            denom = future_mask.sum().clamp_min(1.0)
            loss_main = ((pred - target) ** 2 * future_mask).sum() / denom
            losses.append(loss_main)

        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.state.step += 1

        return {"loss": float(loss.item())}

    @torch.no_grad()
    def inspect_one_batch(self) -> dict[str, Any]:
        batch = next(iter(self.loader))
        prepared = self._prepare_batch(batch)
        return prepared["debug"]

    @torch.no_grad()
    def write_inspection_report(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        debug = self.inspect_one_batch()

        json_path = output_path / "shape_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(debug, f, indent=2, ensure_ascii=False)

        video_src = Path(debug["video_path"])
        video_link = output_path / video_src.name
        if not video_link.exists():
            video_link.symlink_to(video_src)

        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Object-Centric Wan Context Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f7f5ef; color: #222; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    pre {{ background: #fff; border: 1px solid #ddd; padding: 16px; overflow-x: auto; white-space: pre-wrap; }}
    video {{ width: 100%; max-width: 720px; border: 1px solid #ccc; background: #000; }}
  </style>
</head>
<body>
  <h1>Object-Centric Wan Context Report</h1>
  <p>query-point tracks drive object anchors; JEPA local tubes and VAE local latents are pooled into object tokens.</p>
  <div class="grid">
    <div>
      <h2>Source Video</h2>
      <video controls src="./{video_src.name}"></video>
    </div>
    <div>
      <h2>Shape Report</h2>
      <pre>{json.dumps(debug, indent=2, ensure_ascii=False)}</pre>
    </div>
  </div>
</body>
</html>
"""
        html_path = output_path / "index.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        return html_path

    def train(self) -> None:
        max_steps = self.cfg["optimization"]["max_steps"]
        log_every = self.cfg["logging"]["log_every"]
        out_dir = Path(self.cfg["experiment"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        loader_iter = iter(self.loader)
        while self.state.step < max_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.loader)
                batch = next(loader_iter)

            metrics = self.train_step(batch)
            if self.state.step % log_every == 0:
                print(f"[step {self.state.step}] loss={metrics['loss']:.6f}")
