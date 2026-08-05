"""Train the existing Wan Full-SA LoRA with a frozen V-JEPA2 feature loss.

The original flow-matching loss remains authoritative. This module captures the
single DiT model output produced by that loss, reconstructs x0 in latent space,
decodes prediction and target with the same frozen Tiny VAE, and compares their
frozen V-JEPA2 token features. No video files are written or read in this path.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

import code_vjepa_vggt.context_wan_v_newtrain as context_wan
import train_xssc_object_self_attn_lora as core


def _import_module_from_root(module_name: str, root: str):
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Module root does not exist: {root_path}")
    root_text = str(root_path)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module(module_name)
    module_path = Path(module.__file__).resolve()
    if root_path != module_path.parent and root_path not in module_path.parents:
        raise ImportError(
            f"Imported {module_name!r} from {module_path}, expected it under {root_path}"
        )
    return module


def _checkpoint_encoder_state(payload) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported V-JEPA checkpoint object: {type(payload)!r}")
    state = None
    for key in ("ema_encoder", "target_encoder", "encoder", "model", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict) and candidate:
            state = candidate
            break
    if state is None and payload and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        state = payload
    if state is None:
        raise KeyError(
            "V-JEPA checkpoint has no ema_encoder/target_encoder/encoder/model/state_dict"
        )

    normalized = {}
    for key, value in state.items():
        name = str(key)
        while name.startswith("module."):
            name = name[len("module.") :]
        if name.startswith("backbone."):
            name = name[len("backbone.") :]
        normalized[name] = value
    return normalized


def _load_vjepa_encoder(repo: str, checkpoint: str, device: torch.device):
    backbones = _import_module_from_root("src.hub.backbones", repo)
    encoder, _ = backbones.vjepa2_1_vit_large_384(pretrained=False)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state = _checkpoint_encoder_state(payload)
    encoder.load_state_dict(state, strict=True)
    del state, payload
    encoder.requires_grad_(False)
    encoder.eval()
    encoder.to(device=device, dtype=torch.float32)
    return encoder


def _load_tiny_vae(
    root: str,
    checkpoint: str,
    device: torch.device,
    dtype: torch.dtype,
):
    taehv = _import_module_from_root("taehv", root)
    decoder = taehv.TAEHV(str(Path(checkpoint).expanduser().resolve()))
    decoder.requires_grad_(False)
    decoder.eval()
    decoder.to(device=device, dtype=dtype)
    return decoder


class VJEPAFeatureLossWanModule(core.DINOv3XSSCContextSlotsWanModule):
    """Existing training module plus differentiable Tiny-VAE/V-JEPA2 loss."""

    def __init__(
        self,
        *args,
        vjepa_loss_weight: float,
        vjepa_sigma_min: float,
        vjepa_sigma_max: float,
        vjepa_every_n_forwards: int,
        vjepa_num_frames: int,
        vjepa_input_size: int,
        vjepa_repo: str,
        vjepa_checkpoint: str,
        tiny_vae_root: str,
        tiny_vae_checkpoint: str,
        tiny_vae_parallel: bool,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vjepa_loss_weight = float(vjepa_loss_weight)
        self.vjepa_sigma_min = float(vjepa_sigma_min)
        self.vjepa_sigma_max = float(vjepa_sigma_max)
        self.vjepa_every_n_forwards = int(vjepa_every_n_forwards)
        self.vjepa_num_frames = int(vjepa_num_frames)
        self.vjepa_input_size = int(vjepa_input_size)
        self.tiny_vae_parallel = bool(tiny_vae_parallel)
        self._vjepa_forward_count = 0

        if self.vjepa_loss_weight <= 0.0:
            raise ValueError("vjepa_loss_weight must be positive")
        if not 0.0 <= self.vjepa_sigma_min < self.vjepa_sigma_max <= 1.0:
            raise ValueError("Invalid V-JEPA sigma range")
        if self.vjepa_every_n_forwards <= 0:
            raise ValueError("vjepa_every_n_forwards must be positive")
        if self.vjepa_num_frames <= 0 or self.vjepa_num_frames % 2:
            raise ValueError("vjepa_num_frames must be a positive even integer")
        if self.vjepa_input_size != 384:
            raise ValueError("V-JEPA2.1 ViT-L requires vjepa_input_size=384")

        model_device = self.pipe.dit.patch_embedding.weight.device
        tiny_dtype = self.pipe.torch_dtype
        tiny_vae = _load_tiny_vae(
            tiny_vae_root,
            tiny_vae_checkpoint,
            model_device,
            tiny_dtype,
        )
        vjepa_encoder = _load_vjepa_encoder(
            vjepa_repo,
            vjepa_checkpoint,
            model_device,
        )
        # These frozen auxiliaries are deliberately unregistered. DDP must not
        # broadcast them, and LoRA checkpoints must not traverse or save them.
        object.__setattr__(self, "_tiny_vae", tiny_vae)
        object.__setattr__(self, "_vjepa_encoder", vjepa_encoder)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "_tiny_vae"):
            self._tiny_vae.eval()
        if hasattr(self, "_vjepa_encoder"):
            self._vjepa_encoder.eval()
        return self

    def _decode_tiny_vae(self, latents: torch.Tensor) -> torch.Tensor:
        latent_ntchw = latents.permute(0, 2, 1, 3, 4).contiguous()
        device_type = latent_ntchw.device.type
        autocast_enabled = device_type == "cuda"
        with torch.autocast(
            device_type=device_type,
            dtype=self.pipe.torch_dtype,
            enabled=autocast_enabled,
        ):
            return self._tiny_vae.decode_video(
                latent_ntchw,
                parallel=self.tiny_vae_parallel,
                show_progress_bar=False,
            )

    def _preprocess_vjepa(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or int(video.shape[2]) != 3:
            raise ValueError(
                f"Tiny VAE video must be [B,T,3,H,W], got {tuple(video.shape)}"
            )
        time_steps = int(video.shape[1])
        frame_indices = torch.linspace(
            0,
            time_steps - 1,
            steps=self.vjepa_num_frames,
            device=video.device,
        ).round().to(torch.long)
        frames = video.index_select(1, frame_indices).float()
        batch, selected_frames, channels, height, width = frames.shape

        resize_short = round((256.0 / 224.0) * self.vjepa_input_size)
        scale = resize_short / min(height, width)
        resized_height = max(self.vjepa_input_size, round(height * scale))
        resized_width = max(self.vjepa_input_size, round(width * scale))
        frames = F.interpolate(
            frames.reshape(batch * selected_frames, channels, height, width),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        top = (resized_height - self.vjepa_input_size) // 2
        left = (resized_width - self.vjepa_input_size) // 2
        frames = frames[
            :,
            :,
            top : top + self.vjepa_input_size,
            left : left + self.vjepa_input_size,
        ]
        mean = frames.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = frames.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        return frames.view(
            batch,
            selected_frames,
            channels,
            self.vjepa_input_size,
            self.vjepa_input_size,
        ).permute(0, 2, 1, 3, 4).contiguous()

    def _encode_vjepa(self, video: torch.Tensor) -> torch.Tensor:
        self._vjepa_encoder.eval()
        with torch.autocast(device_type=video.device.type, enabled=False):
            features = self._vjepa_encoder(video.float())
        if not isinstance(features, torch.Tensor) or features.ndim < 2:
            raise TypeError(
                "V-JEPA encoder must return a tensor with a feature dimension, "
                f"got {type(features)!r}"
            )
        return features

    def _vjepa_feature_loss(
        self,
        pred_x0_latents: torch.Tensor,
        target_x0_latents: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            target_video = self._decode_tiny_vae(target_x0_latents)
            target_input = self._preprocess_vjepa(target_video)
            target_features = self._encode_vjepa(target_input).detach()

        pred_video = self._decode_tiny_vae(pred_x0_latents)
        pred_input = self._preprocess_vjepa(pred_video)
        pred_features = self._encode_vjepa(pred_input)
        if pred_features.shape != target_features.shape:
            raise RuntimeError(
                "V-JEPA feature shape mismatch: "
                f"pred={tuple(pred_features.shape)}, "
                f"target={tuple(target_features.shape)}"
            )

        pred_features = F.normalize(pred_features.float(), dim=-1)
        target_features = F.normalize(target_features.float(), dim=-1)
        # Mean token-wise squared L2 distance. This is normalized feature MSE
        # summed over channels, keeping the scale independent of feature width.
        return (pred_features - target_features).square().sum(dim=-1).mean()

    @staticmethod
    def _restore_condition_latents(
        pred_x0: torch.Tensor,
        target_x0: torch.Tensor,
        captured_inputs: dict,
    ) -> torch.Tensor:
        context_indices = context_wan.resolve_context_latent_indices_from_frames(
            raw_frame_indices=captured_inputs.get("context_frame_indices"),
            raw_num_frames=captured_inputs.get("num_frames"),
            latent_length=int(target_x0.shape[2]),
        )
        if context_indices:
            return context_wan.apply_clean_latents_at_indices(
                pred_x0,
                target_x0,
                context_indices,
            )

        prefix_length = context_wan.resolve_num_clean_prefix_latents(
            clean_prefix_latents=captured_inputs.get("clean_prefix_latents"),
            num_clean_prefix_latents=captured_inputs.get(
                "num_clean_prefix_latents"
            ),
        )
        if prefix_length > 0:
            pred_x0 = pred_x0.clone()
            pred_x0[:, :, :prefix_length] = target_x0[:, :, :prefix_length]
        elif "first_frame_latents" in captured_inputs:
            pred_x0 = pred_x0.clone()
            pred_x0[:, :, 0:1] = target_x0[:, :, 0:1]
        return pred_x0

    def forward(self, data, inputs=None):
        if isinstance(data, list):
            return super().forward(data, inputs=inputs)
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(
            inputs,
            self.pipe.device,
            self.pipe.torch_dtype,
        )
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss, metrics = self._compute_object_losses(
            self.pipe,
            inputs[0],
            inputs[1],
        )
        self.last_train_metrics = metrics
        return loss

    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        captured: list[dict] = []
        original_model_fn = pipe.model_fn

        def capture_model_fn(*args, **kwargs):
            output = original_model_fn(*args, **kwargs)
            captured.append(
                {
                    "model_output": output,
                    "latents": kwargs.get("latents"),
                    "timestep": kwargs.get("timestep"),
                    "inputs": kwargs,
                }
            )
            return output

        pipe.model_fn = capture_model_fn
        try:
            total, metrics = super()._compute_object_losses(
                pipe,
                inputs_shared,
                inputs_posi,
            )
        finally:
            pipe.model_fn = original_model_fn

        if len(captured) != 1:
            raise RuntimeError(
                "Expected exactly one DiT forward in flow-matching loss, "
                f"captured {len(captured)}"
            )
        record = captured[0]
        if not isinstance(record["latents"], torch.Tensor):
            raise RuntimeError("DiT forward did not receive latent x_t")
        if not isinstance(record["model_output"], torch.Tensor):
            raise RuntimeError("DiT forward did not return a tensor v prediction")

        sigma = context_wan._diffsynth_sigma_for_timestep(
            pipe.scheduler,
            record["timestep"],
        )
        sigma_value = float(sigma.detach().float().item())
        self._vjepa_forward_count += 1
        cadence_applies = (
            self._vjepa_forward_count % self.vjepa_every_n_forwards == 0
        )
        sigma_applies = self.vjepa_sigma_min <= sigma_value <= self.vjepa_sigma_max
        apply_vjepa = cadence_applies and sigma_applies

        metrics["train/vjepa_sigma"] = sigma_value
        metrics["train/vjepa_loss_applied"] = float(apply_vjepa)
        metrics["train/vjepa_loss_weight"] = self.vjepa_loss_weight
        if not apply_vjepa:
            metrics["train/loss_vjepa"] = 0.0
            return total, metrics

        sigma_tensor = sigma.to(
            device=record["latents"].device,
            dtype=record["latents"].dtype,
        )
        while sigma_tensor.ndim < record["latents"].ndim:
            sigma_tensor = sigma_tensor.unsqueeze(-1)
        pred_x0 = record["latents"] - sigma_tensor * record["model_output"]
        target_x0 = inputs_shared["input_latents"]
        pred_x0 = self._restore_condition_latents(
            pred_x0,
            target_x0,
            record["inputs"],
        )
        loss_vjepa = self._vjepa_feature_loss(pred_x0, target_x0)
        total = total + self.vjepa_loss_weight * loss_vjepa
        metrics["train/loss_vjepa"] = float(loss_vjepa.detach().item())
        metrics["train/loss_total"] = float(total.detach().item())
        return total, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = core.build_parser()
    parser.description += " Adds a frozen Tiny-VAE/V-JEPA2 feature MSE."
    group = parser.add_argument_group("vjepa_feature_loss")
    group.add_argument("--vjepa_loss_weight", type=float, required=True)
    group.add_argument("--vjepa_sigma_min", type=float, required=True)
    group.add_argument("--vjepa_sigma_max", type=float, required=True)
    group.add_argument("--vjepa_every_n_forwards", type=int, required=True)
    group.add_argument("--vjepa_num_frames", type=int, required=True)
    group.add_argument("--vjepa_input_size", type=int, default=384)
    group.add_argument("--vjepa_repo", required=True)
    group.add_argument("--vjepa_checkpoint", required=True)
    group.add_argument("--tiny_vae_root", required=True)
    group.add_argument("--tiny_vae_checkpoint", required=True)
    group.add_argument("--tiny_vae_parallel", action="store_true")
    return parser


def build_model(args: argparse.Namespace, accelerator):
    return core.build_model(
        args,
        accelerator,
        model_class=VJEPAFeatureLossWanModule,
        extra_model_kwargs={
            "vjepa_loss_weight": args.vjepa_loss_weight,
            "vjepa_sigma_min": args.vjepa_sigma_min,
            "vjepa_sigma_max": args.vjepa_sigma_max,
            "vjepa_every_n_forwards": args.vjepa_every_n_forwards,
            "vjepa_num_frames": args.vjepa_num_frames,
            "vjepa_input_size": args.vjepa_input_size,
            "vjepa_repo": args.vjepa_repo,
            "vjepa_checkpoint": args.vjepa_checkpoint,
            "tiny_vae_root": args.tiny_vae_root,
            "tiny_vae_checkpoint": args.tiny_vae_checkpoint,
            "tiny_vae_parallel": args.tiny_vae_parallel,
        },
    )


def log_stage_summary(accelerator, model, args: argparse.Namespace) -> None:
    core._log_stage_summary(accelerator, model, args)
    if accelerator.is_main_process:
        accelerator.print(
            "V-JEPA auxiliary loss: "
            f"weight={args.vjepa_loss_weight:g}, "
            f"sigma=[{args.vjepa_sigma_min:g}, {args.vjepa_sigma_max:g}], "
            f"every_n_forwards={args.vjepa_every_n_forwards}, "
            f"frames={args.vjepa_num_frames}, input={args.vjepa_input_size}, "
            f"tiny_vae_parallel={args.tiny_vae_parallel}"
        )


def main() -> None:
    core.main(
        build_parser_fn=build_parser,
        build_model_fn=build_model,
        log_stage_summary_fn=log_stage_summary,
    )


if __name__ == "__main__":
    main()
