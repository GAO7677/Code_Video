from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from lib.data_utils import (
    actual_future_positions,
    constant_velocity_rollout,
    load_rgb_clip,
    temporal_union_bboxes,
    token_indices_from_bboxes,
)


@dataclass
class FeatureOutput:
    vector: torch.Tensor
    tokens: torch.Tensor | None = None
    grid_size: int | None = None
    crop_size: int | None = None
    tubelet_size: int | None = None


def _as_device(device: str) -> torch.device:
    return torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


def _normalize_clip_uint8_to_imagenet(frames: list[np.ndarray], crop_size: int) -> torch.Tensor:
    import torchvision.transforms.functional as TF
    from PIL import Image

    short_side_size = int(crop_size * 256 / 224)
    tensor_frames = []
    for frame in frames:
        image = Image.fromarray(frame)
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


def cosine_score(left: torch.Tensor, right: torch.Tensor) -> float:
    left = F.normalize(left.float(), dim=-1)
    right = F.normalize(right.float(), dim=-1)
    return float(torch.sum(left * right, dim=-1).item())


def _mean_tokens(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.mean(dim=0)


def _pool_tokens_with_candidate_boxes(
    *,
    tokens: torch.Tensor,
    candidate: dict[str, Any],
    pooling: str,
    crop_size: int,
    tubelet_size: int,
    grid_size: int,
) -> torch.Tensor:
    participants = candidate.get("participants", [])
    if not participants:
        return _mean_tokens(tokens)

    bbox_map = temporal_union_bboxes(
        candidate["sample_dir"],
        candidate["frame_indices"],
        object_indices=participants,
        tubelet_size=tubelet_size,
        crop_size=crop_size,
    )
    token_map = token_indices_from_bboxes(bbox_map, grid_size=grid_size, crop_size=crop_size)

    pooled = []
    for obj_idx in participants[:2]:
        indices = token_map.get(obj_idx)
        if indices is None or indices.size == 0:
            pooled.append(torch.zeros(tokens.shape[-1], device=tokens.device, dtype=tokens.dtype))
            continue
        pooled.append(tokens[torch.from_numpy(indices).to(tokens.device)].mean(dim=0))

    if pooling == "object":
        return pooled[0]
    if len(pooled) == 1:
        pooled.append(torch.zeros_like(pooled[0]))
    return torch.cat(pooled[:2], dim=-1)


class RetrievalBackend(abc.ABC):
    name: str

    def __init__(self, *, device: str = "auto", crop_size: int = 384) -> None:
        self.device = _as_device(device)
        self.crop_size = crop_size

    @abc.abstractmethod
    def encode_query(self, query_row: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def encode_candidate(self, candidate: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def score(
        self,
        *,
        query_row: dict[str, Any],
        candidate: dict[str, Any],
        query_state: Any,
        candidate_state: Any,
        pooling: str,
    ) -> float:
        raise NotImplementedError


class RandomBackend(RetrievalBackend):
    name = "random"

    def encode_query(self, query_row: dict[str, Any]) -> None:
        return None

    def encode_candidate(self, candidate: dict[str, Any]) -> None:
        return None

    def score(self, *, query_row: dict[str, Any], candidate: dict[str, Any], query_state: Any, candidate_state: Any, pooling: str) -> float:
        seed = hash((query_row["query_id"], candidate["candidate_id"], self.name)) & 0xFFFFFFFF
        generator = np.random.default_rng(seed)
        return float(generator.random())


class StateExtrapolationBackend(RetrievalBackend):
    name = "state_extrap"

    def encode_query(self, query_row: dict[str, Any]) -> np.ndarray:
        context_frames = query_row["context"]["frame_indices"]
        return constant_velocity_rollout(
            query_row["source_sample_dir"],
            context_frames,
            future_width=query_row["future_width"],
            object_indices=query_row["participants"],
        )

    def encode_candidate(self, candidate: dict[str, Any]) -> np.ndarray:
        return actual_future_positions(
            candidate["sample_dir"],
            candidate["frame_indices"],
            object_indices=candidate["participants"],
        )

    def score(self, *, query_row: dict[str, Any], candidate: dict[str, Any], query_state: np.ndarray, candidate_state: np.ndarray, pooling: str) -> float:
        query_positions = query_state
        cand_positions = candidate_state
        if cand_positions.shape[1] < query_positions.shape[1]:
            pad = np.zeros((cand_positions.shape[0], query_positions.shape[1] - cand_positions.shape[1], 3), dtype=np.float32)
            cand_positions = np.concatenate([cand_positions, pad], axis=1)
        mse = float(np.mean((query_positions - cand_positions[:, : query_positions.shape[1]]) ** 2))
        return -mse


class VJEPA2PredictorBackend(RetrievalBackend):
    name = "vjepa_predictor"

    def __init__(self, *, checkpoint: str, device: str = "auto", crop_size: int = 384, model_frames: int = 64) -> None:
        super().__init__(device=device, crop_size=crop_size)
        self.checkpoint = checkpoint
        self.model_frames = model_frames
        self.tubelet_size = 2
        self.grid_size = crop_size // 16
        self.context_encoder, self.target_encoder, self.predictor = self._load_models()

    def _load_models(self):
        import sys

        repo_root = Path("/home/gaoya/Code_Video/vjepa2-main")
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        import torch
        import src.models.predictor as vit_predictor
        import src.models.vision_transformer as vit_encoder

        ckpt = torch.load(self.checkpoint, map_location="cpu")

        enc_kwargs = dict(
            img_size=(self.crop_size, self.crop_size),
            patch_size=16,
            num_frames=self.model_frames,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
        )
        context_encoder = vit_encoder.vit_giant_xformers(**enc_kwargs)
        target_encoder = vit_encoder.vit_giant_xformers(**enc_kwargs)

        predictor = vit_predictor.vit_predictor(
            img_size=(self.crop_size, self.crop_size),
            patch_size=16,
            num_frames=self.model_frames,
            tubelet_size=2,
            use_mask_tokens=True,
            embed_dim=context_encoder.embed_dim,
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
                key = key.replace("module.", "").replace("backbone.", "")
                out[key] = value
            return out

        context_encoder.load_state_dict(clean(ckpt["encoder"]), strict=False)
        target_encoder.load_state_dict(clean(ckpt["target_encoder"]), strict=False)
        predictor.load_state_dict(clean(ckpt["predictor"]), strict=False)

        context_encoder.eval().to(self.device)
        target_encoder.eval().to(self.device)
        predictor.eval().to(self.device)
        return context_encoder, target_encoder, predictor

    def _load_clip(self, sample_dir: str, frame_indices: list[int]) -> torch.Tensor:
        frames = load_rgb_clip(Path(sample_dir) / "rgb", frame_indices)
        return _normalize_clip_uint8_to_imagenet(frames, crop_size=self.crop_size).to(self.device)

    @torch.no_grad()
    def encode_query(self, query_row: dict[str, Any]) -> FeatureOutput:
        clip = self._load_clip(query_row["context"]["sample_dir"], query_row["context"]["frame_indices"])
        context_tokens = self.context_encoder(clip)

        horizon = int(query_row["horizon"])
        future_width = int(query_row["future_width"])
        if horizon % self.tubelet_size != 0:
            raise ValueError(f"horizon must be divisible by tubelet size {self.tubelet_size}")
        if future_width % self.tubelet_size != 0:
            raise ValueError(f"future width must be divisible by tubelet size {self.tubelet_size}")

        num_pred_tokens = (future_width // self.tubelet_size) * self.grid_size * self.grid_size
        skip_tokens = (horizon // self.tubelet_size) * self.grid_size * self.grid_size
        context_positions = torch.arange(context_tokens.shape[1], device=self.device).unsqueeze(0)
        target_positions = torch.arange(num_pred_tokens, device=self.device).unsqueeze(0) + context_tokens.shape[1] + skip_tokens

        predicted = self.predictor(context_tokens, masks_x=context_positions, masks_y=target_positions)
        tokens = predicted[:, -num_pred_tokens:, :]
        return FeatureOutput(
            vector=_mean_tokens(tokens.squeeze(0)),
            tokens=tokens.squeeze(0),
            grid_size=self.grid_size,
            crop_size=self.crop_size,
            tubelet_size=self.tubelet_size,
        )

    @torch.no_grad()
    def encode_candidate(self, candidate: dict[str, Any]) -> FeatureOutput:
        clip = self._load_clip(candidate["sample_dir"], candidate["frame_indices"])
        tokens = self.target_encoder(clip)
        return FeatureOutput(
            vector=_mean_tokens(tokens.squeeze(0)),
            tokens=tokens.squeeze(0),
            grid_size=self.grid_size,
            crop_size=self.crop_size,
            tubelet_size=self.tubelet_size,
        )

    def score(self, *, query_row: dict[str, Any], candidate: dict[str, Any], query_state: FeatureOutput, candidate_state: FeatureOutput, pooling: str) -> float:
        if pooling == "global":
            return cosine_score(query_state.vector, candidate_state.vector)
        query_vec = _pool_tokens_with_candidate_boxes(
            tokens=query_state.tokens,
            candidate=candidate,
            pooling=pooling,
            crop_size=self.crop_size,
            tubelet_size=self.tubelet_size,
            grid_size=self.grid_size,
        )
        cand_vec = _pool_tokens_with_candidate_boxes(
            tokens=candidate_state.tokens,
            candidate=candidate,
            pooling=pooling,
            crop_size=self.crop_size,
            tubelet_size=self.tubelet_size,
            grid_size=self.grid_size,
        )
        return cosine_score(query_vec, cand_vec)


class VJEPA2ContextBackend(VJEPA2PredictorBackend):
    name = "vjepa_context"

    @torch.no_grad()
    def encode_query(self, query_row: dict[str, Any]) -> FeatureOutput:
        clip = self._load_clip(query_row["context"]["sample_dir"], query_row["context"]["frame_indices"])
        tokens = self.context_encoder(clip)
        return FeatureOutput(
            vector=_mean_tokens(tokens.squeeze(0)),
            tokens=tokens.squeeze(0),
            grid_size=self.grid_size,
            crop_size=self.crop_size,
            tubelet_size=self.tubelet_size,
        )

    def score(self, *, query_row: dict[str, Any], candidate: dict[str, Any], query_state: FeatureOutput, candidate_state: FeatureOutput, pooling: str) -> float:
        if pooling == "global":
            return cosine_score(query_state.vector, candidate_state.vector)
        query_candidate_like = {
            "sample_dir": query_row["context"]["sample_dir"],
            "frame_indices": query_row["context"]["frame_indices"],
            "participants": query_row["participants"],
        }
        query_vec = _pool_tokens_with_candidate_boxes(
            tokens=query_state.tokens,
            candidate=query_candidate_like,
            pooling=pooling,
            crop_size=self.crop_size,
            tubelet_size=self.tubelet_size,
            grid_size=self.grid_size,
        )
        cand_vec = _pool_tokens_with_candidate_boxes(
            tokens=candidate_state.tokens,
            candidate=candidate,
            pooling=pooling,
            crop_size=self.crop_size,
            tubelet_size=self.tubelet_size,
            grid_size=self.grid_size,
        )
        return cosine_score(query_vec, cand_vec)


class HFVideoMAEBackend(RetrievalBackend):
    name = "videomae"

    def __init__(self, *, model_id: str, device: str = "auto", crop_size: int = 224) -> None:
        super().__init__(device=device, crop_size=crop_size)
        from transformers import VideoMAEImageProcessor, VideoMAEModel

        self.processor = VideoMAEImageProcessor.from_pretrained(model_id)
        self.model = VideoMAEModel.from_pretrained(model_id).to(self.device).eval()

    @torch.no_grad()
    def _encode(self, clip_frames: list[np.ndarray]) -> FeatureOutput:
        inputs = self.processor(clip_frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        outputs = self.model(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state[:, 1:, :]
        return FeatureOutput(vector=_mean_tokens(tokens.squeeze(0)), tokens=tokens.squeeze(0))

    def encode_query(self, query_row: dict[str, Any]) -> FeatureOutput:
        return self._encode(load_rgb_clip(Path(query_row["context"]["sample_dir"]) / "rgb", query_row["context"]["frame_indices"]))

    def encode_candidate(self, candidate: dict[str, Any]) -> FeatureOutput:
        return self._encode(load_rgb_clip(Path(candidate["sample_dir"]) / "rgb", candidate["frame_indices"]))

    def score(self, *, query_row: dict[str, Any], candidate: dict[str, Any], query_state: FeatureOutput, candidate_state: FeatureOutput, pooling: str) -> float:
        return cosine_score(query_state.vector, candidate_state.vector)


class HFImageAverageBackend(RetrievalBackend):
    hf_name: str

    def __init__(self, *, model_id: str, device: str = "auto", crop_size: int = 224) -> None:
        super().__init__(device=device, crop_size=crop_size)
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.model_id = model_id

    @torch.no_grad()
    def _encode_frames(self, frames: list[np.ndarray]) -> FeatureOutput:
        inputs = self.processor(images=frames, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state
        if hidden.ndim != 3:
            raise ValueError(f"{self.model_id} did not return last_hidden_state with shape [B, N, D]")
        vectors = hidden[:, 0, :]
        return FeatureOutput(vector=vectors.mean(dim=0), tokens=None)

    def encode_query(self, query_row: dict[str, Any]) -> FeatureOutput:
        frames = load_rgb_clip(Path(query_row["context"]["sample_dir"]) / "rgb", query_row["context"]["frame_indices"])
        return self._encode_frames(frames)

    def encode_candidate(self, candidate: dict[str, Any]) -> FeatureOutput:
        frames = load_rgb_clip(Path(candidate["sample_dir"]) / "rgb", candidate["frame_indices"])
        return self._encode_frames(frames)

    def score(self, *, query_row: dict[str, Any], candidate: dict[str, Any], query_state: FeatureOutput, candidate_state: FeatureOutput, pooling: str) -> float:
        return cosine_score(query_state.vector, candidate_state.vector)


class DinoBackend(HFImageAverageBackend):
    name = "dino"


class ClipBackend(HFImageAverageBackend):
    name = "clip"


def build_backend(args: Any) -> RetrievalBackend:
    if args.backend == "random":
        return RandomBackend(device=args.device, crop_size=args.crop_size)
    if args.backend == "state_extrap":
        return StateExtrapolationBackend(device=args.device, crop_size=args.crop_size)
    if args.backend == "vjepa_predictor":
        return VJEPA2PredictorBackend(
            checkpoint=args.vjepa_checkpoint,
            device=args.device,
            crop_size=args.crop_size,
        )
    if args.backend == "vjepa_context":
        return VJEPA2ContextBackend(
            checkpoint=args.vjepa_checkpoint,
            device=args.device,
            crop_size=args.crop_size,
        )
    if args.backend == "videomae":
        if not args.videomae_model_id:
            raise ValueError("videomae backend requires --videomae-model-id")
        return HFVideoMAEBackend(model_id=args.videomae_model_id, device=args.device, crop_size=args.crop_size)
    if args.backend == "dino":
        model_id = args.dino_model_id or "facebook/dinov2-base"
        return DinoBackend(model_id=model_id, device=args.device, crop_size=args.crop_size)
    if args.backend == "clip":
        model_id = args.clip_model_id or "openai/clip-vit-base-patch32"
        return ClipBackend(model_id=model_id, device=args.device, crop_size=args.crop_size)
    raise ValueError(f"Unsupported backend: {args.backend}")
