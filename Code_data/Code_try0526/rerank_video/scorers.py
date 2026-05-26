from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .schemas import GeometryConfig, JEPAScoreConfig, LatentMotionConfig
from .video_utils import load_video_frames, resize_crop_frame, uniform_subsample_frames


DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/DiffSynth-Studio-main")
if str(DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_ROOT))


class LatentMotionScorer:
    def __init__(self, config: LatentMotionConfig) -> None:
        self.config = config
        self.pipe = None

    def _lazy_init(self) -> None:
        if self.pipe is not None:
            return
        from diffsynth import ModelConfig
        from diffsynth.pipelines.wan_video import WanVideoPipeline

        if self.config.vae_root is None:
            raise ValueError("Latent motion scorer requires vae_root")
        vae_candidates = [
            self.config.vae_root / "Wan2.2_VAE.pth",
            self.config.vae_root / "Wan2.1_VAE.pth",
        ]
        vae_path = None
        for path in vae_candidates:
            if path.is_file():
                vae_path = path
                break
        if vae_path is None:
            raise FileNotFoundError(f"No Wan VAE found under {self.config.vae_root}")
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=self.config.device,
            model_configs=[ModelConfig(path=str(vae_path))],
            tokenizer_config=None,
            redirect_common_files=False,
        )

    def _encode_latents(self, video_path: Path) -> torch.Tensor:
        self._lazy_init()
        frames = load_video_frames(video_path)
        frames = uniform_subsample_frames(frames, self.config.max_frames)
        pil_frames = [Image.fromarray(frame).convert("RGB") for frame in frames]
        tensor = self.pipe.preprocess_video(pil_frames)
        latents = self.pipe.vae.encode(
            tensor,
            device=self.config.device,
            tiled=True,
            tile_size=self.config.tile_size,
            tile_stride=self.config.tile_stride,
        )
        return latents.float()

    def score(self, video_path: Path) -> tuple[float, dict[str, Any]]:
        latents = self._encode_latents(video_path)
        diffs = latents[:, :, 1:] - latents[:, :, :-1]
        flat = diffs[0].permute(1, 0, 2, 3).reshape(diffs.shape[2], -1)
        if flat.shape[0] < 2:
            return 0.0, {"reason": "too_few_latent_steps"}

        magnitudes = torch.linalg.norm(flat, dim=1)
        magnitude_mean = float(magnitudes.mean().item())
        magnitude_std = float(magnitudes.std(unbiased=False).item())
        magnitude_cv = magnitude_std / max(magnitude_mean, 1e-6)

        flat_a = flat[:-1]
        flat_b = flat[1:]
        direction_cos = F.cosine_similarity(flat_a, flat_b, dim=1)
        direction_continuity = float(((direction_cos + 1.0) * 0.5).mean().item())

        accel = magnitudes[1:] - magnitudes[:-1]
        accel_abs_mean = float(accel.abs().mean().item())
        accel_smoothness = math.exp(-accel_abs_mean / max(magnitude_mean, 1e-6))

        low_motion_ratio = float((magnitudes < max(magnitude_mean * 0.15, 1e-6)).float().mean().item())
        motion_active = max(0.0, 1.0 - low_motion_ratio)

        mag_np = magnitudes.cpu().numpy().astype(np.float64)
        centered = mag_np - float(mag_np.mean())
        spectrum = np.fft.rfft(centered)
        power = np.abs(spectrum) ** 2
        if power.size <= 1 or float(power[1:].sum()) <= 0:
            spectral_stability = 1.0
            high_freq_ratio = 0.0
        else:
            split = max(1, power.shape[0] // 3)
            high_freq_ratio = float(power[-split:].sum() / max(power[1:].sum(), 1e-8))
            spectral_stability = max(0.0, 1.0 - high_freq_ratio)

        magnitude_consistency = math.exp(-magnitude_cv)
        final_score = float(
            0.20 * motion_active
            + 0.20 * magnitude_consistency
            + 0.25 * direction_continuity
            + 0.20 * accel_smoothness
            + 0.15 * spectral_stability
        )
        details = {
            "magnitude_mean": magnitude_mean,
            "magnitude_std": magnitude_std,
            "magnitude_cv": magnitude_cv,
            "direction_continuity": direction_continuity,
            "accel_abs_mean": accel_abs_mean,
            "accel_smoothness": accel_smoothness,
            "low_motion_ratio": low_motion_ratio,
            "motion_active": motion_active,
            "high_freq_ratio": high_freq_ratio,
            "spectral_stability": spectral_stability,
            "magnitude_consistency": magnitude_consistency,
        }
        return final_score, details


class GeometryProxyScorer:
    def __init__(self, config: GeometryConfig) -> None:
        self.config = config

    @staticmethod
    def _grayscale(frame: np.ndarray) -> np.ndarray:
        frame_f = np.asarray(frame, dtype=np.float32)
        return (0.299 * frame_f[..., 0] + 0.587 * frame_f[..., 1] + 0.114 * frame_f[..., 2]).astype(np.float32)

    def _largest_motion_component(self, anchor_gray: np.ndarray, frame_gray: np.ndarray) -> tuple[np.ndarray | None, dict[str, float]]:
        diff = np.abs(frame_gray - anchor_gray)
        motion = (diff >= self.config.diff_threshold).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion, connectivity=8)
        best_index = -1
        best_area = 0
        for label_index in range(1, num_labels):
            area = int(stats[label_index, cv2.CC_STAT_AREA])
            if area >= self.config.min_component_area and area > best_area:
                best_area = area
                best_index = label_index
        if best_index < 0:
            return None, {"area": 0.0}
        mask = labels == best_index
        x = int(stats[best_index, cv2.CC_STAT_LEFT])
        y = int(stats[best_index, cv2.CC_STAT_TOP])
        w = int(stats[best_index, cv2.CC_STAT_WIDTH])
        h = int(stats[best_index, cv2.CC_STAT_HEIGHT])
        return mask, {
            "area": float(best_area),
            "x": float(x),
            "y": float(y),
            "w": float(w),
            "h": float(h),
            "cx": float(centroids[best_index][0]),
            "cy": float(centroids[best_index][1]),
        }

    def score(self, *, context_video_path: Path, candidate_video_path: Path) -> tuple[float, dict[str, Any]]:
        context_frames = load_video_frames(context_video_path)
        candidate_frames = uniform_subsample_frames(load_video_frames(candidate_video_path), self.config.max_frames)
        if not context_frames or not candidate_frames:
            return 0.0, {"reason": "missing_frames"}
        anchor_gray = self._grayscale(context_frames[-1])
        tracks: list[dict[str, float]] = []
        for frame in candidate_frames:
            frame_gray = self._grayscale(frame)
            mask, stats = self._largest_motion_component(anchor_gray, frame_gray)
            if mask is None:
                continue
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                continue
            pseudo_depth = 1.0 - float(frame_gray[mask].mean() / 255.0)
            bbox_area = max(float(stats["w"] * stats["h"]), 1.0)
            mask_area = max(float(stats["area"]), 1.0)
            tracks.append(
                {
                    **stats,
                    "pseudo_depth": pseudo_depth,
                    "mask_area": mask_area,
                    "bbox_area": bbox_area,
                    "height_px": float(stats["h"]),
                    "width_px": float(stats["w"]),
                    "compactness": mask_area / bbox_area,
                }
            )

        if len(tracks) < 3:
            return 0.0, {"reason": "too_few_motion_tracks", "track_count": len(tracks)}

        heights = np.asarray([item["height_px"] for item in tracks], dtype=np.float64)
        depths = np.asarray([item["pseudo_depth"] for item in tracks], dtype=np.float64)
        centroids = np.asarray([[item["cx"], item["cy"]] for item in tracks], dtype=np.float64)
        compactness = np.asarray([item["compactness"] for item in tracks], dtype=np.float64)
        scale_depth_product = heights * np.clip(depths, 1e-3, None)
        scale_depth_rmse = float(
            np.sqrt(np.mean((scale_depth_product - scale_depth_product.mean()) ** 2))
            / max(abs(scale_depth_product.mean()), 1e-6)
        )
        scale_depth_score = math.exp(-scale_depth_rmse)

        velocities = centroids[1:] - centroids[:-1]
        vel_norm = np.linalg.norm(velocities, axis=1)
        if velocities.shape[0] >= 2:
            accelerations = velocities[1:] - velocities[:-1]
            accel_norm = np.linalg.norm(accelerations, axis=1)
            traj_smoothness = math.exp(-float(accel_norm.mean()) / max(float(vel_norm.mean()), 1e-6))
            dir_cos = []
            for prev_vec, next_vec in zip(velocities[:-1], velocities[1:]):
                denom = np.linalg.norm(prev_vec) * np.linalg.norm(next_vec)
                if denom <= 1e-6:
                    continue
                dir_cos.append(float(np.dot(prev_vec, next_vec) / denom))
            direction_consistency = (float(np.mean(dir_cos)) + 1.0) * 0.5 if dir_cos else 0.5
        else:
            traj_smoothness = 0.5
            direction_consistency = 0.5
        trajectory_score = 0.65 * traj_smoothness + 0.35 * direction_consistency

        compactness_cv = float(compactness.std() / max(compactness.mean(), 1e-6))
        width_height_ratio = np.asarray(
            [item["width_px"] / max(item["height_px"], 1e-6) for item in tracks],
            dtype=np.float64,
        )
        ratio_cv = float(width_height_ratio.std() / max(width_height_ratio.mean(), 1e-6))
        rigidity_score = 0.5 * math.exp(-compactness_cv) + 0.5 * math.exp(-ratio_cv)

        final_score = float(0.38 * scale_depth_score + 0.34 * trajectory_score + 0.28 * rigidity_score)
        details = {
            "track_count": len(tracks),
            "scale_depth_rmse": scale_depth_rmse,
            "scale_depth_score": scale_depth_score,
            "traj_smoothness": traj_smoothness,
            "direction_consistency": direction_consistency,
            "trajectory_score": trajectory_score,
            "compactness_cv": compactness_cv,
            "ratio_cv": ratio_cv,
            "rigidity_score": rigidity_score,
        }
        return final_score, details


def _normalize_clip_uint8_to_imagenet(frames: list[np.ndarray], crop_size: int) -> torch.Tensor:
    import torchvision.transforms.functional as TF

    short_side_size = int(crop_size * 256 / 224)
    tensor_frames = []
    for frame in frames:
        image = Image.fromarray(frame).convert("RGB")
        width, height = image.size
        if height <= width:
            new_h = short_side_size
            new_w = int(round(width * short_side_size / height))
        else:
            new_w = short_side_size
            new_h = int(round(height * short_side_size / width))
        image = image.resize((new_w, new_h), resample=Image.BILINEAR)
        left = max(0, (new_w - crop_size) // 2)
        top = max(0, (new_h - crop_size) // 2)
        image = image.crop((left, top, left + crop_size, top + crop_size))
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tensor_frames.append(tensor)
    clip = torch.stack(tensor_frames, dim=1)
    return clip.unsqueeze(0)


class _VJEPA2Backend:
    def __init__(self, config: JEPAScoreConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.crop_size = int(config.crop_size)
        self.context_encoder = None
        self.target_encoder = None
        self.predictor = None
        self.tubelet_size = 2
        self.grid_size = self.crop_size // 16
        self._load_models()

    def _load_models(self) -> None:
        if self.config.vjepa_checkpoint is None or self.config.vjepa_repo_root is None:
            raise ValueError("vjepa2 backend requires vjepa_checkpoint and vjepa_repo_root")
        repo_root = self.config.vjepa_repo_root
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import src.models.predictor as vit_predictor
        import src.models.vision_transformer as vit_encoder

        ckpt = torch.load(self.config.vjepa_checkpoint, map_location="cpu")
        enc_kwargs = dict(
            img_size=(self.crop_size, self.crop_size),
            patch_size=16,
            num_frames=self.config.max_frames,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
        )
        self.context_encoder = vit_encoder.vit_giant_xformers(**enc_kwargs)
        self.target_encoder = vit_encoder.vit_giant_xformers(**enc_kwargs)
        self.predictor = vit_predictor.vit_predictor(
            img_size=(self.crop_size, self.crop_size),
            patch_size=16,
            num_frames=self.config.max_frames,
            tubelet_size=2,
            use_mask_tokens=True,
            embed_dim=self.context_encoder.embed_dim,
            predictor_embed_dim=384,
            depth=12,
            num_heads=12,
            num_mask_tokens=10,
            use_rope=True,
            uniform_power=False,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
        )

        def clean(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            out = {}
            for key, value in state_dict.items():
                out[key.replace("module.", "").replace("backbone.", "")] = value
            return out

        self.context_encoder.load_state_dict(clean(ckpt["encoder"]), strict=False)
        self.target_encoder.load_state_dict(clean(ckpt["target_encoder"]), strict=False)
        self.predictor.load_state_dict(clean(ckpt["predictor"]), strict=False)
        self.context_encoder.eval().to(self.device)
        self.target_encoder.eval().to(self.device)
        self.predictor.eval().to(self.device)

    @torch.no_grad()
    def score(self, context_frames: list[np.ndarray], future_frames: list[np.ndarray]) -> tuple[float, dict[str, Any]]:
        context_frames = uniform_subsample_frames(context_frames, self.config.context_frames)
        future_frames = uniform_subsample_frames(future_frames, self.config.future_frames)
        if len(context_frames) < 2 or len(future_frames) < 2:
            return 0.0, {"reason": "too_few_frames"}
        context_clip = _normalize_clip_uint8_to_imagenet(context_frames, self.crop_size).to(self.device)
        future_clip = _normalize_clip_uint8_to_imagenet(future_frames, self.crop_size).to(self.device)

        context_tokens = self.context_encoder(context_clip)
        future_tokens = self.target_encoder(future_clip)
        future_token_count = future_tokens.shape[1]
        context_positions = torch.arange(context_tokens.shape[1], device=self.device).unsqueeze(0)
        target_positions = torch.arange(future_token_count, device=self.device).unsqueeze(0) + context_tokens.shape[1]
        predicted_tokens = self.predictor(context_tokens, masks_x=context_positions, masks_y=target_positions)
        predicted_tokens = predicted_tokens[:, -future_token_count:, :]

        pred_vec = predicted_tokens.mean(dim=1)
        future_vec = future_tokens.mean(dim=1)
        predictive_alignment = float(F.cosine_similarity(pred_vec, future_vec, dim=1).mean().item())

        context_mid = len(context_frames) // 2
        tail_clip = _normalize_clip_uint8_to_imagenet(context_frames[context_mid:], self.crop_size).to(self.device)
        future_head_clip = _normalize_clip_uint8_to_imagenet(future_frames[: max(2, len(future_frames) // 2)], self.crop_size).to(self.device)
        tail_vec = self.target_encoder(tail_clip).mean(dim=1)
        future_head_vec = self.target_encoder(future_head_clip).mean(dim=1)
        continuity = float(F.cosine_similarity(tail_vec, future_head_vec, dim=1).mean().item())

        smoothness_values: list[float] = []
        for start in range(0, max(len(future_frames) - 7, 1), max(len(future_frames) // 4, 1)):
            window = future_frames[start : start + min(8, len(future_frames) - start)]
            if len(window) < 2:
                continue
            clip = _normalize_clip_uint8_to_imagenet(window, self.crop_size).to(self.device)
            smoothness_values.append(float(self.target_encoder(clip).mean(dim=1).squeeze(0).norm().item()))
        if len(smoothness_values) >= 2:
            smoothness_cv = float(np.std(smoothness_values) / max(np.mean(smoothness_values), 1e-6))
            temporal_smoothness = math.exp(-smoothness_cv)
        else:
            temporal_smoothness = 0.5

        score = float(0.5 * ((predictive_alignment + 1.0) * 0.5) + 0.3 * ((continuity + 1.0) * 0.5) + 0.2 * temporal_smoothness)
        return score, {
            "predictive_alignment": predictive_alignment,
            "continuity": continuity,
            "temporal_smoothness": temporal_smoothness,
        }


class _VideoMAEBackend:
    def __init__(self, config: JEPAScoreConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        model_id = config.videomae_model_id or "MCG-NJU/videomae-base"
        from transformers import VideoMAEImageProcessor, VideoMAEModel

        self.processor = VideoMAEImageProcessor.from_pretrained(model_id)
        self.model = VideoMAEModel.from_pretrained(model_id).eval().to(self.device)

    @torch.no_grad()
    def _embed(self, frames: list[np.ndarray]) -> torch.Tensor:
        inputs = self.processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state.mean(dim=1)

    @torch.no_grad()
    def score(self, context_frames: list[np.ndarray], future_frames: list[np.ndarray]) -> tuple[float, dict[str, Any]]:
        context_frames = uniform_subsample_frames(context_frames, self.config.context_frames)
        future_frames = uniform_subsample_frames(future_frames, self.config.future_frames)
        if len(context_frames) < 4 or len(future_frames) < 2:
            return 0.0, {"reason": "too_few_frames"}
        split = max(2, len(context_frames) // 2)
        c1 = self._embed(context_frames[:split])
        c2 = self._embed(context_frames[split:])
        future = self._embed(future_frames)
        predicted = c2 + (c2 - c1)
        predictive_alignment = float(F.cosine_similarity(predicted, future, dim=1).mean().item())
        continuity = float(F.cosine_similarity(c2, future, dim=1).mean().item())
        score = float(0.6 * ((predictive_alignment + 1.0) * 0.5) + 0.4 * ((continuity + 1.0) * 0.5))
        return score, {
            "predictive_alignment": predictive_alignment,
            "continuity": continuity,
        }


class JEPAPredictiveScorer:
    def __init__(self, config: JEPAScoreConfig) -> None:
        self.config = config
        if config.backend == "vjepa2":
            self.backend = _VJEPA2Backend(config)
        elif config.backend == "videomae":
            self.backend = _VideoMAEBackend(config)
        else:
            raise ValueError(f"Unsupported JEPA backend: {config.backend}")

    def score(self, *, context_video_path: Path, candidate_video_path: Path) -> tuple[float, dict[str, Any]]:
        context_frames = uniform_subsample_frames(load_video_frames(context_video_path), self.config.max_frames)
        future_frames = uniform_subsample_frames(load_video_frames(candidate_video_path), self.config.max_frames)
        return self.backend.score(context_frames, future_frames)

