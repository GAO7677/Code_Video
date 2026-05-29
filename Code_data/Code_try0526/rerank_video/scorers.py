from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
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
        self._pdi_modules_loaded = False
        self._sam_wrapper_cls = None
        self._track_wrapper_cls = None
        self._projection_judge_cls = None
        self._audit_scale_consistency = None
        self._audit_rigidity_stability = None

    @staticmethod
    def _grayscale(frame: np.ndarray) -> np.ndarray:
        frame_f = np.asarray(frame, dtype=np.float32)
        return (0.299 * frame_f[..., 0] + 0.587 * frame_f[..., 1] + 0.114 * frame_f[..., 2]).astype(np.float32)

    def _lazy_load_pdi_modules(self) -> None:
        if self._pdi_modules_loaded:
            return
        if self.config.pdi_repo_root is None:
            raise ValueError("GeometryConfig.pdi_repo_root is required for backend='sam_depth'")
        src_root = self.config.pdi_repo_root / "src"
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        from pdi_eval.perception.sam_wrapper import Sam2Wrapper
        from pdi_eval.perception.track_wrapper import TrackWrapper
        from pdi_eval.geometry.projection import ProjectionJudge
        from pdi_eval.evaluator.scale_audit import audit_scale_consistency
        from pdi_eval.evaluator.volume_audit import audit_rigidity_stability

        self._sam_wrapper_cls = Sam2Wrapper
        self._track_wrapper_cls = TrackWrapper
        self._projection_judge_cls = ProjectionJudge
        self._audit_scale_consistency = audit_scale_consistency
        self._audit_rigidity_stability = audit_rigidity_stability
        self._pdi_modules_loaded = True

    def _cache_root(self) -> Path:
        if getattr(self.config, "cache_root", None) is not None:
            root = Path(self.config.cache_root)
        else:
            root = Path("/tmp/try0526_geometry_proxy_cache")
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _video_id(video_path: Path) -> str:
        resolved = video_path.expanduser().resolve()
        digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
        return f"{resolved.stem}_{digest}"

    def _extract_depth_anything(self, video_path: Path) -> list[np.ndarray]:
        if self.config.depth_anything_repo_root is None or self.config.depth_anything_ckpt is None:
            raise ValueError("depth_anything_repo_root and depth_anything_ckpt are required for backend='sam_depth'")
        cache_dir = self._cache_root() / self._video_id(video_path) / "depth_anything"
        depth_files = sorted(cache_dir.glob("*.npy"))
        if not depth_files:
            frames_dir = cache_dir.parent / "frames"
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            frames = load_video_frames(video_path)
            frames_dir.mkdir(parents=True, exist_ok=True)
            for index, frame in enumerate(frames):
                image = Image.fromarray(frame).convert("RGB")
                image.save(frames_dir / f"{index:06d}.jpg")
            cmd = [
                sys.executable,
                str(self.config.depth_anything_repo_root / "run_videos.py"),
                "--img-path",
                str(frames_dir),
                "--outdir",
                str(cache_dir),
                "--encoder",
                "vitl",
                "--load-from",
                str(self.config.depth_anything_ckpt),
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.config.depth_anything_repo_root) + os.pathsep + env.get("PYTHONPATH", "")
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=str(self.config.depth_anything_repo_root), env=env)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Depth-Anything failed for {video_path}\nstdout:\n{completed.stdout[-2000:]}\nstderr:\n{completed.stderr[-2000:]}"
                )
            depth_files = sorted(cache_dir.glob("*.npy"))
        return [np.load(path).astype(np.float32) for path in depth_files]

    @staticmethod
    def _vp_inside_bbox(vp_xy: tuple[float, float], masks: np.ndarray, margin_ratio: float = 0.1) -> bool:
        if masks is None or len(masks) == 0:
            return False
        combined = np.any(masks[: min(5, len(masks))], axis=0)
        ys, xs = np.where(combined)
        if len(xs) == 0:
            return False
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        mx = (x_max - x_min) * margin_ratio
        my = (y_max - y_min) * margin_ratio
        return (x_min - mx <= vp_xy[0] <= x_max + mx) and (y_min - my <= vp_xy[1] <= y_max + my)

    def _compute_vp_error(
        self,
        *,
        tracks_use: np.ndarray,
        bg_tracks: np.ndarray,
        frames: list[np.ndarray],
        masks_use: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[float, dict[str, Any]]:
        self._lazy_load_pdi_modules()
        cam_cx = width / 2.0
        cam_cy = height / 2.0
        proj = self._projection_judge_cls(cx=cam_cx, cy=cam_cy)
        fg_tracks_ntd = tracks_use.transpose(1, 0, 2)
        bg_tracks_ntd = bg_tracks.transpose(1, 0, 2) if bg_tracks.ndim == 3 and bg_tracks.shape[0] > 0 else None
        lsd_frames = np.asarray(frames[: min(3, len(frames))], dtype=np.uint8) if frames else None
        lsd_masks = masks_use[: len(lsd_frames)] if lsd_frames is not None else None
        global_vp, fg_vp, bg_vp = proj.estimate_vanishing_point_v2(
            fg_tracks=fg_tracks_ntd,
            bg_tracks=bg_tracks_ntd,
            frames=lsd_frames,
            masks=lsd_masks,
        )
        fg_degenerate = fg_vp == (cam_cx, cam_cy)
        fg_in_bbox = self._vp_inside_bbox(fg_vp, masks_use)
        if fg_degenerate or fg_in_bbox:
            vp = bg_vp
        else:
            vp = fg_vp
        fg_dir = np.array([vp[0] - cam_cx, vp[1] - cam_cy], dtype=np.float64)
        bg_dir = np.array([bg_vp[0] - cam_cx, bg_vp[1] - cam_cy], dtype=np.float64)
        fg_norm = float(np.linalg.norm(fg_dir))
        bg_norm = float(np.linalg.norm(bg_dir))
        fg_offscreen = vp[0] < 0 or vp[0] > width or vp[1] < 0 or vp[1] > height
        if fg_norm < 5.0 or bg_norm < 5.0 or fg_offscreen:
            eps_vp = 0.0
        else:
            cos_sim = float(np.dot(fg_dir, bg_dir)) / max(fg_norm * bg_norm, 1e-8)
            eps_vp = (1.0 - float(np.clip(cos_sim, -1.0, 1.0))) / 2.0
        return eps_vp, {
            "global_vp": [float(global_vp[0]), float(global_vp[1])],
            "fg_vp": [float(fg_vp[0]), float(fg_vp[1])],
            "bg_vp": [float(bg_vp[0]), float(bg_vp[1])],
            "selected_vp": [float(vp[0]), float(vp[1])],
            "fg_in_bbox": bool(fg_in_bbox),
            "fg_degenerate": bool(fg_degenerate),
        }

    def _score_sam_depth(
        self,
        *,
        candidate_video_path: Path,
        target_object: str | None,
    ) -> tuple[float, dict[str, Any]]:
        if not target_object:
            return 0.0, {"reason": "sam_depth_backend_requires_target_object"}
        self._lazy_load_pdi_modules()
        if self.config.sam_ckpt is None or self.config.sam_cfg is None or self.config.tracker_ckpt is None:
            return 0.0, {"reason": "missing_sam_or_tracker_checkpoints"}

        os.environ.setdefault("PDI_FLORENCE_MODEL_ID", "/data/gaoya/ckpt/microsoft-Florence-2-base")
        sam = self._sam_wrapper_cls(checkpoint=str(self.config.sam_ckpt), config=str(self.config.sam_cfg), device=self.config.device)
        res_2d = sam.infer(str(candidate_video_path), text_query=target_object)
        masks = np.asarray(res_2d.masks).astype(bool)
        if masks.ndim != 3 or len(masks) < 3:
            return 0.0, {"reason": "too_few_masks"}

        mask_sizes = masks.reshape(len(masks), -1).sum(axis=1)
        if float(np.median(mask_sizes)) < float(self.config.min_mask_pixels):
            return 0.0, {"reason": "mask_too_small", "median_mask_pixels": float(np.median(mask_sizes))}

        tracker = self._track_wrapper_cls(checkpoint=str(self.config.tracker_ckpt), device=self.config.device)
        res_tracks_raw = tracker.infer(str(candidate_video_path), initial_mask=masks[0].astype(np.uint8))

        tracks = np.asarray(res_tracks_raw.tracks_2d)
        visibility = np.asarray(res_tracks_raw.confidence)
        bg_tracks = np.asarray(res_tracks_raw.metadata.get("bg_tracks", np.empty((0, 0, 2))))

        depth_maps = self._extract_depth_anything(candidate_video_path)
        frames = load_video_frames(candidate_video_path)

        T_use = min(len(masks), len(depth_maps), len(tracks), len(frames))
        masks_use = masks[:T_use]
        tracks_use = tracks[:T_use]
        visibility_use = visibility[:T_use]
        bg_tracks_use = bg_tracks[:T_use] if bg_tracks.ndim == 3 and bg_tracks.shape[0] >= T_use else np.empty((0, 0, 2))
        frames_use = frames[:T_use]
        depth_use = depth_maps[:T_use]
        h_seq = np.asarray(res_2d.h_pixel[:T_use], dtype=np.float64)

        z_seq = []
        for depth_map, mask in zip(depth_use, masks_use):
            if depth_map.shape != mask.shape:
                mask_rs = cv2.resize(mask.astype(np.uint8), (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            else:
                mask_rs = mask
            if np.any(mask_rs):
                z_seq.append(float(np.median(depth_map[mask_rs])))
            else:
                z_seq.append(float(np.median(depth_map)))
        z_seq = np.asarray(z_seq, dtype=np.float64)

        # Depth-Anything can behave like depth or inverse-depth depending on the
        # release/checkpoint. Evaluate both orientations and keep the one with
        # the smaller scale residual so the proxy score is not sign-sensitive.
        z_seq_norm = z_seq / max(float(z_seq[0]), 1e-6)
        direct_history = np.asarray(self._audit_scale_consistency(h_seq, z_seq_norm), dtype=np.float64)
        direct_error = float(direct_history.mean()) if direct_history.size > 0 else 0.0
        inverse_history = np.asarray(self._audit_scale_consistency(h_seq, 1.0 / np.maximum(z_seq_norm, 1e-6)), dtype=np.float64)
        inverse_error = float(inverse_history.mean()) if inverse_history.size > 0 else 0.0
        if inverse_error < direct_error:
            scale_history = inverse_history
            scale_error = inverse_error
            scale_orientation = "inverse_depth"
        else:
            scale_history = direct_history
            scale_error = direct_error
            scale_orientation = "depth"

        rigidity_error, rigidity_history = self._audit_rigidity_stability(tracks_use, h_seq)
        rigidity_error = float(rigidity_error)

        vp_error, vp_details = self._compute_vp_error(
            tracks_use=tracks_use,
            bg_tracks=bg_tracks_use,
            frames=frames_use,
            masks_use=masks_use,
            width=frames_use[0].shape[1],
            height=frames_use[0].shape[0],
        )

        scale_score = math.exp(-scale_error)
        rigidity_score = math.exp(-rigidity_error)
        vp_score = math.exp(-2.0 * vp_error)
        proxy_error_total = float(0.40 * scale_error + 0.35 * rigidity_error + 0.25 * vp_error)
        proxy_total = float(math.exp(-proxy_error_total))

        return proxy_total, {
            "backend": "sam_depth",
            "track_count": int(tracks_use.shape[1]) if tracks_use.ndim == 3 else 0,
            "median_mask_pixels": float(np.median(mask_sizes[:T_use])),
            "scale_error": scale_error,
            "rigidity_error": rigidity_error,
            "vp_error": float(vp_error),
            "scale_score": scale_score,
            "rigidity_score": rigidity_score,
            "vp_score": vp_score,
            "proxy_error_total": proxy_error_total,
            "proxy_total": proxy_total,
            "scale_orientation": scale_orientation,
            "scale_history_mean": scale_error,
            "rigidity_history_mean": float(np.mean(rigidity_history)) if np.size(rigidity_history) > 0 else 0.0,
            **vp_details,
        }

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

    def _score_with_anchor_frame(
        self,
        *,
        anchor_frame: np.ndarray,
        candidate_video_path: Path,
    ) -> tuple[float, dict[str, Any]]:
        candidate_frames = uniform_subsample_frames(load_video_frames(candidate_video_path), self.config.max_frames)
        if anchor_frame.size == 0 or not candidate_frames:
            return 0.0, {"reason": "missing_frames"}
        anchor_gray = self._grayscale(anchor_frame)
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

    def score(
        self,
        *,
        context_video_path: Path,
        candidate_video_path: Path,
        target_object: str | None = None,
    ) -> tuple[float, dict[str, Any]]:
        if self.config.backend == "sam_depth":
            if not target_object:
                raise ValueError(
                    "GeometryProxyScorer backend='sam_depth' requires target_object. "
                    "Use score_from_anchor_image(...) or pass target_object explicitly."
                )
            return self._score_sam_depth(
                candidate_video_path=candidate_video_path,
                target_object=target_object,
            )
        context_frames = load_video_frames(context_video_path)
        if not context_frames:
            return 0.0, {"reason": "missing_frames"}
        return self._score_with_anchor_frame(
            anchor_frame=context_frames[-1],
            candidate_video_path=candidate_video_path,
        )

    def score_from_anchor_image(
        self,
        *,
        anchor_image_path: Path,
        candidate_video_path: Path,
        target_object: str | None = None,
    ) -> tuple[float, dict[str, Any]]:
        if self.config.backend == "sam_depth":
            return self._score_sam_depth(candidate_video_path=candidate_video_path, target_object=target_object)
        anchor_image = Image.open(anchor_image_path).convert("RGB")
        candidate_frames = uniform_subsample_frames(load_video_frames(candidate_video_path), self.config.max_frames)
        if not candidate_frames:
            return 0.0, {"reason": "missing_frames"}
        target_height, target_width = candidate_frames[0].shape[:2]
        anchor_image = anchor_image.resize((target_width, target_height), Image.Resampling.BILINEAR)
        anchor_frame = np.asarray(anchor_image, dtype=np.uint8)
        return self._score_with_anchor_frame(
            anchor_frame=anchor_frame,
            candidate_video_path=candidate_video_path,
        )


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
        dtype_name = os.environ.get("TRY0526_VJEPA_DTYPE", "float32").lower()
        if dtype_name == "bfloat16":
            self.model_dtype = torch.bfloat16
        elif dtype_name == "float16":
            self.model_dtype = torch.float16
        else:
            # V-JEPA2.1 large has shown illegal memory access in mixed precision on this machine.
            # Default to float32 so the proxy metric is stable and reproducible.
            self.model_dtype = torch.float32
        self.context_encoder = None
        self.target_encoder = None
        self.predictor = None
        self.model_variant = "vjepa2"
        self._load_models()

    def _load_models(self) -> None:
        if self.config.vjepa_checkpoint is None or self.config.vjepa_repo_root is None:
            raise ValueError("vjepa2 backend requires vjepa_checkpoint and vjepa_repo_root")
        repo_root = self.config.vjepa_repo_root
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        ckpt = torch.load(self.config.vjepa_checkpoint, map_location="cpu")
        if "ema_encoder" in ckpt or self.config.vjepa_model_name.startswith("vjepa2_1_"):
            self._load_vjepa21_models(ckpt)
        else:
            self._load_vjepa2_models(ckpt)
        self.context_encoder.eval().to(device=self.device, dtype=self.model_dtype)
        self.target_encoder.eval().to(device=self.device, dtype=self.model_dtype)
        self.predictor.eval().to(device=self.device, dtype=self.model_dtype)

    @staticmethod
    def _clean(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        for key, value in state_dict.items():
            out[key.replace("module.", "").replace("backbone.", "")] = value
        return out

    @staticmethod
    def _align_prediction_to_target(
        predicted_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        pred_dim = int(predicted_tokens.shape[-1])
        target_dim = int(target_tokens.shape[-1])
        if pred_dim == target_dim:
            return predicted_tokens, target_tokens, {
                "mode": "identity",
                "prediction_feature_dim": pred_dim,
                "target_feature_dim": target_dim,
            }

        if pred_dim > target_dim:
            aligned = predicted_tokens[..., -target_dim:]
            return aligned, target_tokens, {
                "mode": "take_prediction_tail",
                "prediction_feature_dim": pred_dim,
                "target_feature_dim": target_dim,
                "aligned_feature_dim": target_dim,
            }

        aligned_target = target_tokens[..., -pred_dim:]
        return predicted_tokens, aligned_target, {
            "mode": "take_target_tail",
            "prediction_feature_dim": pred_dim,
            "target_feature_dim": target_dim,
            "aligned_feature_dim": pred_dim,
        }

    def _infer_future_token_layout(self, token_count: int, frame_count: int) -> tuple[int, int]:
        spatial_tokens = (self.crop_size // 16) * (self.crop_size // 16)
        tubelet_size = 2
        temporal_tokens = max(frame_count // tubelet_size, 1)
        if temporal_tokens * spatial_tokens == token_count:
            return temporal_tokens, spatial_tokens
        if token_count % spatial_tokens == 0:
            return max(token_count // spatial_tokens, 1), spatial_tokens
        return 1, token_count

    def _temporal_pool_tokens(self, tokens: torch.Tensor, frame_count: int) -> tuple[torch.Tensor, dict[str, int]]:
        temporal_tokens, spatial_tokens = self._infer_future_token_layout(int(tokens.shape[1]), frame_count)
        pooled = tokens.reshape(tokens.shape[0], temporal_tokens, spatial_tokens, tokens.shape[-1]).mean(dim=2)
        return pooled, {
            "temporal_token_count": int(temporal_tokens),
            "spatial_token_count": int(spatial_tokens),
        }

    @staticmethod
    def _gram_margin_l1(left: torch.Tensor, right: torch.Tensor, margin: float = 0.1) -> tuple[float, float]:
        left_norm = F.normalize(left.float(), dim=-1)
        right_norm = F.normalize(right.float(), dim=-1)
        left_gram = left_norm @ left_norm.transpose(1, 2)
        right_gram = right_norm @ right_norm.transpose(1, 2)
        diff = (left_gram - right_gram).abs()
        raw_error = float(diff.mean().item())
        margin_error = float(torch.clamp(diff - margin, min=0.0).mean().item())
        return raw_error, margin_error

    @staticmethod
    def _gram_l1(left: torch.Tensor, right: torch.Tensor) -> float:
        left_norm = F.normalize(left.float(), dim=-1)
        right_norm = F.normalize(right.float(), dim=-1)
        left_gram = left_norm @ left_norm.transpose(1, 2)
        right_gram = right_norm @ right_norm.transpose(1, 2)
        return float((left_gram - right_gram).abs().mean().item())

    @staticmethod
    def _profile_l1(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-6) -> float:
        left_profile = torch.linalg.vector_norm(left.float(), dim=-1)
        right_profile = torch.linalg.vector_norm(right.float(), dim=-1)
        left_profile = left_profile / left_profile.mean(dim=1, keepdim=True).clamp_min(eps)
        right_profile = right_profile / right_profile.mean(dim=1, keepdim=True).clamp_min(eps)
        return float((left_profile - right_profile).abs().mean().item())

    def _load_vjepa2_models(self, ckpt: dict[str, Any]) -> None:
        import src.models.predictor as vit_predictor
        import src.models.vision_transformer as vit_encoder

        self.model_variant = "vjepa2"
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
        context_key = "encoder" if "encoder" in ckpt else "target_encoder"
        target_key = "target_encoder" if "target_encoder" in ckpt else context_key
        self.context_encoder.load_state_dict(self._clean(ckpt[context_key]), strict=False)
        self.target_encoder.load_state_dict(self._clean(ckpt[target_key]), strict=False)
        self.predictor.load_state_dict(self._clean(ckpt["predictor"]), strict=False)

    def _load_vjepa21_models(self, ckpt: dict[str, Any]) -> None:
        import app.vjepa_2_1.models.predictor as vit_predictor
        import app.vjepa_2_1.models.vision_transformer as vit_encoder

        model_name = self.config.vjepa_model_name or "vjepa2_1_vit_large_384"
        arch_name_map = {
            "vjepa2_1_vit_base_384": "vit_base",
            "vjepa2_1_vit_large_384": "vit_large",
            "vjepa2_1_vit_giant_384": "vit_giant_xformers",
            "vjepa2_1_vit_gigantic_384": "vit_gigantic_xformers",
        }
        predictor_depth_map = {
            "vjepa2_1_vit_base_384": 12,
            "vjepa2_1_vit_large_384": 12,
            "vjepa2_1_vit_giant_384": 24,
            "vjepa2_1_vit_gigantic_384": 24,
        }
        predictor_mask_map = {
            "vjepa2_1_vit_base_384": 8,
            "vjepa2_1_vit_large_384": 8,
            "vjepa2_1_vit_giant_384": 8,
            "vjepa2_1_vit_gigantic_384": 8,
        }
        arch_name = arch_name_map.get(model_name, "vit_large")
        predictor_depth = predictor_depth_map.get(model_name, 12)
        predictor_num_mask_tokens = predictor_mask_map.get(model_name, 8)
        self.model_variant = model_name

        enc_kwargs = dict(
            patch_size=16,
            img_size=(self.crop_size, self.crop_size),
            num_frames=self.config.max_frames,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )
        self.context_encoder = vit_encoder.__dict__[arch_name](**enc_kwargs)
        self.target_encoder = vit_encoder.__dict__[arch_name](**enc_kwargs)
        self.predictor = vit_predictor.vit_predictor(
            img_size=(self.crop_size, self.crop_size),
            patch_size=16,
            use_mask_tokens=True,
            embed_dim=self.context_encoder.embed_dim,
            predictor_embed_dim=384,
            teacher_embed_dim=1664,
            num_frames=self.config.max_frames,
            tubelet_size=2,
            depth=predictor_depth,
            num_heads=12,
            num_mask_tokens=predictor_num_mask_tokens,
            use_rope=True,
            uniform_power=False,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
            n_output_distillation=1,
            return_all_tokens=True,
            img_temporal_dim_size=1,
        )
        self.context_encoder.load_state_dict(self._clean(ckpt["encoder"]), strict=True)
        self.target_encoder.load_state_dict(self._clean(ckpt["ema_encoder"]), strict=True)
        self.predictor.load_state_dict(self._clean(ckpt["predictor"]), strict=True)

    @torch.no_grad()
    def score(self, context_frames: list[np.ndarray], future_frames: list[np.ndarray]) -> tuple[float, dict[str, Any]]:
        context_frames = uniform_subsample_frames(context_frames, self.config.context_frames)
        future_frames = uniform_subsample_frames(future_frames, self.config.future_frames)
        if len(context_frames) < 2 or len(future_frames) < 2:
            return 0.0, {"reason": "too_few_frames"}
        context_clip = _normalize_clip_uint8_to_imagenet(context_frames, self.crop_size).to(self.device, dtype=self.model_dtype)
        future_clip = _normalize_clip_uint8_to_imagenet(future_frames, self.crop_size).to(self.device, dtype=self.model_dtype)

        autocast_enabled = self.device.type == "cuda" and self.model_dtype != torch.float32
        with torch.autocast(device_type=self.device.type, dtype=self.model_dtype, enabled=autocast_enabled):
            context_tokens = self.context_encoder(context_clip)
            future_tokens = self.target_encoder(future_clip)
            future_token_count = future_tokens.shape[1]
            context_positions = torch.arange(context_tokens.shape[1], device=self.device).unsqueeze(0)
            target_positions = torch.arange(future_token_count, device=self.device).unsqueeze(0) + context_tokens.shape[1]
            predicted_out = self.predictor(context_tokens, masks_x=context_positions, masks_y=target_positions)
            predicted_context_tokens = None
            if isinstance(predicted_out, tuple):
                predicted_tokens, predicted_context_tokens = predicted_out
            else:
                predicted_tokens = predicted_out

            predicted_tokens_aligned, future_tokens_aligned, feature_alignment = self._align_prediction_to_target(
                predicted_tokens,
                future_tokens,
            )

        predicted_tokens_aligned = predicted_tokens_aligned.float()
        future_tokens_aligned = future_tokens_aligned.float()
        predictive_alignment = float(
            F.cosine_similarity(predicted_tokens_aligned, future_tokens_aligned, dim=-1).mean().item()
        )
        predictive_l2 = float(torch.mean((predicted_tokens_aligned - future_tokens_aligned) ** 2).item())

        # Primary proxy metrics should not depend on the ad-hoc feature-dimension truncation above.
        # Pool predicted and target futures in their native feature spaces, then compare only
        # dimension-invariant temporal structures and motion profiles.
        predicted_temporal_native, predicted_layout = self._temporal_pool_tokens(predicted_tokens.float(), len(future_frames))
        future_temporal_native, future_layout = self._temporal_pool_tokens(future_tokens.float(), len(future_frames))

        predicted_temporal_aligned, _ = self._temporal_pool_tokens(predicted_tokens_aligned, len(future_frames))
        future_temporal_aligned, _ = self._temporal_pool_tokens(future_tokens_aligned, len(future_frames))
        time_cosine = float(F.cosine_similarity(predicted_temporal_aligned, future_temporal_aligned, dim=-1).mean().item())

        temporal_relation_raw_error = self._gram_l1(predicted_temporal_native, future_temporal_native)
        temporal_relation_error = temporal_relation_raw_error

        if predicted_temporal_native.shape[1] >= 2 and future_temporal_native.shape[1] >= 2:
            pred_delta_native = predicted_temporal_native[:, 1:] - predicted_temporal_native[:, :-1]
            future_delta_native = future_temporal_native[:, 1:] - future_temporal_native[:, :-1]
            delta_relation_raw_error = self._gram_l1(pred_delta_native, future_delta_native)
            delta_profile_error = self._profile_l1(pred_delta_native, future_delta_native)
        else:
            delta_relation_raw_error = 1.0
            delta_profile_error = 1.0

        if predicted_temporal_aligned.shape[1] >= 2:
            pred_delta_aligned = predicted_temporal_aligned[:, 1:] - predicted_temporal_aligned[:, :-1]
            future_delta_aligned = future_temporal_aligned[:, 1:] - future_temporal_aligned[:, :-1]
            delta_cosine = float(F.cosine_similarity(pred_delta_aligned, future_delta_aligned, dim=-1).mean().item())
            delta_l2 = float(torch.mean((pred_delta_aligned - future_delta_aligned) ** 2).item())
        else:
            delta_cosine = 0.0
            delta_l2 = 1.0

        # Keep a hidden compatibility score for old plumbing, but make it a simple monotonic
        # transform of the new raw errors instead of a hand-tuned weighted aggregate.
        score = float(math.exp(-(temporal_relation_raw_error + delta_relation_raw_error + delta_profile_error)))
        return score, {
            "backend": self.model_variant,
            "model_dtype": str(self.model_dtype).replace("torch.", ""),
            "score_version": "physalign_temporal_v2_raw_error_triplet",
            "predictive_alignment": predictive_alignment,
            "predictive_l2": predictive_l2,
            "time_cosine": time_cosine,
            "temporal_relation_raw_error": temporal_relation_raw_error,
            "temporal_relation_error": temporal_relation_error,
            "temporal_relation_score": None,
            "delta_relation_raw_error": delta_relation_raw_error,
            "delta_profile_error": delta_profile_error,
            "delta_cosine": delta_cosine,
            "delta_l2": delta_l2,
            "delta_score": None,
            "context_token_count": int(context_tokens.shape[1]),
            "future_token_count": int(future_token_count),
            "prediction_token_dim": int(predicted_tokens.shape[-1]),
            "future_token_dim": int(future_tokens.shape[-1]),
            "predictor_returned_context": predicted_context_tokens is not None,
            "feature_alignment": feature_alignment,
            **predicted_layout,
            "future_temporal_token_count": future_layout.get("temporal_token_count"),
            "future_spatial_token_count": future_layout.get("spatial_token_count"),
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

    def score_from_anchor_image(
        self,
        *,
        anchor_image_path: Path,
        candidate_video_path: Path,
    ) -> tuple[float, dict[str, Any]]:
        candidate_frames = uniform_subsample_frames(load_video_frames(candidate_video_path), self.config.max_frames)
        if not candidate_frames:
            return 0.0, {"reason": "missing_frames"}
        anchor_image = Image.open(anchor_image_path).convert("RGB")
        target_height, target_width = candidate_frames[0].shape[:2]
        anchor_image = anchor_image.resize((target_width, target_height), Image.Resampling.BILINEAR)
        anchor_frame = np.asarray(anchor_image, dtype=np.uint8)
        context_frames = [anchor_frame.copy() for _ in range(max(int(self.config.context_repeat_frames), 2))]
        score, details = self.backend.score(context_frames, candidate_frames)
        details = {"context_mode": "repeat_anchor_frame", "context_repeat_frames": len(context_frames), **details}
        return score, details
