from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


VJEPA2_REPO = Path("/home/gaoya/Code_Video/vjepa2-main")
CACHE_DIR = Path("/data/gaoya/agent-data/cache/torch/hub/checkpoints")
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

_MODEL_ALIASES = {
    "vith": "vjepa2_vit_huge",
    "vitg": "vjepa2_vit_giant",
    "vitg384": "vjepa2_vit_giant_384",
}

_DEFAULT_CHECKPOINTS = {
    "vith": "https://dl.fbaipublicfiles.com/vjepa2/vith.pt",
    "vitg384": Path("/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth"),
    "vitg": Path("/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth"),
}


def add_vjepa_repo_to_path(repo_root: Path = VJEPA2_REPO) -> None:
    repo_root = repo_root.expanduser().resolve()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _load_state_dict(checkpoint_path: str | Path) -> dict:
    if isinstance(checkpoint_path, Path):
        return torch.load(str(checkpoint_path), map_location="cpu")
    checkpoint_text = str(checkpoint_path)
    if checkpoint_text.startswith(("http://", "https://")):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return torch.hub.load_state_dict_from_url(
            checkpoint_text,
            map_location="cpu",
            model_dir=str(CACHE_DIR),
        )
    return torch.load(checkpoint_text, map_location="cpu")


def _scalar_attr(value) -> int:
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("Expected a non-empty tuple/list attribute")
        return int(value[0])
    return int(value)


def load_vjepa2_models(
    model_name: str = "vitg",
    *,
    repo_root: Path = VJEPA2_REPO,
    local_torchhub: bool = True,
    checkpoint_path: str | Path | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, int]:
    model_key = _MODEL_ALIASES.get(model_name)
    if model_key is None:
        raise ValueError(f"Unsupported V-JEPA 2 model: {model_name}")

    repo_root = repo_root.expanduser().resolve()
    if checkpoint_path:
        checkpoint_text = str(checkpoint_path)
        if checkpoint_text.startswith(("http://", "https://")):
            checkpoint_path = checkpoint_text
        else:
            checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    else:
        checkpoint_path = None
    if local_torchhub:
        encoder, predictor = torch.hub.load(
            str(repo_root),
            model_key,
            source="local",
            pretrained=checkpoint_path is None,
        )
    else:
        encoder, predictor = torch.hub.load(
            "facebookresearch/vjepa2",
            model_key,
            pretrained=checkpoint_path is None,
        )
    target_encoder = copy.deepcopy(encoder)

    if checkpoint_path is None:
        checkpoint_path = _DEFAULT_CHECKPOINTS.get(model_name)

    if checkpoint_path is not None:
        if isinstance(checkpoint_path, Path) and not checkpoint_path.is_file():
            raise FileNotFoundError(f"V-JEPA checkpoint not found: {checkpoint_path}")

        add_vjepa_repo_to_path(repo_root)
        from src.hub.backbones import _clean_backbone_key

        state_dict = _load_state_dict(checkpoint_path)
        target_key = "target_encoder" if "target_encoder" in state_dict else "ema_encoder"
        encoder_key = "encoder" if "encoder" in state_dict else target_key
        if target_key not in state_dict:
            raise KeyError(
                f"Checkpoint {checkpoint_path} does not contain target_encoder or ema_encoder"
            )
        if "predictor" not in state_dict:
            raise KeyError(f"Checkpoint {checkpoint_path} does not contain predictor")

        encoder_state_dict = _clean_backbone_key(state_dict[encoder_key])
        target_state_dict = _clean_backbone_key(state_dict[target_key])
        predictor_state_dict = _clean_backbone_key(state_dict["predictor"])

        encoder.load_state_dict(encoder_state_dict, strict=False)
        target_encoder.load_state_dict(target_state_dict, strict=False)
        predictor.load_state_dict(predictor_state_dict, strict=False)

    img_size = 384 if "384" in model_name else 256
    return encoder, target_encoder, predictor, img_size


def prepare_video_for_vjepa(video_btchw: torch.Tensor, img_size: int) -> torch.Tensor:
    if video_btchw.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(video_btchw.shape)}")
    if video_btchw.shape[1] != 3:
        raise ValueError(f"Expected RGB input, got channel dim {video_btchw.shape[1]}")

    bsz, _, frames, _, _ = video_btchw.shape
    x = video_btchw.float().clamp(-1.0, 1.0)
    x = (x + 1.0) * 0.5

    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1)

    x = x.permute(0, 2, 1, 3, 4).reshape(bsz * frames, 3, x.shape[-2], x.shape[-1])
    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    x = x.view(bsz, frames, 3, img_size, img_size).permute(0, 2, 1, 3, 4).contiguous()
    x = (x - mean) / std
    return x


def generate_causal_masks(
    *,
    batch_size: int,
    img_size: int,
    frames_per_clip: int,
    encoder: torch.nn.Module,
    context_frames: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    patch_size = _scalar_attr(encoder.patch_size)
    tubelet_size = _scalar_attr(encoder.tubelet_size)
    grid_size = img_size // patch_size
    grid_depth = frames_per_clip // tubelet_size
    context_depth = context_frames // tubelet_size
    future_depth = grid_depth - context_depth
    if future_depth <= 0:
        raise ValueError(
            f"context_frames={context_frames} is too large for frames_per_clip={frames_per_clip}"
        )

    n_context = int(grid_size**2 * context_depth)
    n_target = int(grid_size**2 * future_depth)

    ctxt_positions = torch.arange(n_context, device=device).unsqueeze(0).repeat(batch_size, 1)
    tgt_positions = torch.arange(n_target, device=device).unsqueeze(0).repeat(batch_size, 1)
    tgt_positions = tgt_positions + n_context
    return [ctxt_positions], [tgt_positions]


def _window_video(video_btchw: torch.Tensor, window_size: int, stride: int) -> torch.Tensor:
    bsz, channels, _, height, width = video_btchw.shape
    pieces = video_btchw.unfold(2, window_size, stride)
    pieces = pieces.permute(0, 2, 5, 1, 3, 4).contiguous()
    pieces = pieces.view(-1, window_size, channels, height, width)
    return pieces.permute(0, 2, 1, 3, 4).contiguous()


def _token_grid_shape(*, img_size: int, encoder: torch.nn.Module, future_frames: int) -> tuple[int, int, int]:
    patch_size = _scalar_attr(encoder.patch_size)
    tubelet_size = _scalar_attr(encoder.tubelet_size)
    grid_h = int(img_size // patch_size)
    grid_w = int(img_size // patch_size)
    future_depth = max(1, int(future_frames // max(1, tubelet_size)))
    return future_depth, grid_h, grid_w


def _motion_weights_for_future_tokens(
    future_motion_mask_thw: torch.Tensor | None,
    *,
    token_shape: tuple[int, int, int],
    motion_mask_mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if future_motion_mask_thw is None:
        return None
    if future_motion_mask_thw.ndim == 3:
        future_motion_mask_thw = future_motion_mask_thw.unsqueeze(0)
    if future_motion_mask_thw.ndim != 4:
        raise ValueError(
            "future_motion_mask_thw must be [T,H,W] or [B,T,H,W], "
            f"got {tuple(future_motion_mask_thw.shape)}"
        )
    if motion_mask_mode not in {"per_frame", "temporal_union"}:
        raise ValueError(f"Unsupported motion_mask_mode: {motion_mask_mode}")

    weights = future_motion_mask_thw.to(device=device, dtype=dtype)
    if motion_mask_mode == "temporal_union":
        union_hw = (weights > 0.5).any(dim=1, keepdim=True).to(dtype=dtype)
        weights = union_hw.expand(-1, weights.shape[1], -1, -1).contiguous()

    future_depth, grid_h, grid_w = token_shape
    weights = F.interpolate(
        weights.unsqueeze(1),
        size=(future_depth, grid_h, grid_w),
        mode="trilinear",
        align_corners=False,
    ).squeeze(1)
    return weights.clamp_(0.0, 1.0)


def _masked_token_surprise_mean(
    token_surprise: torch.Tensor,
    *,
    future_motion_mask_thw: torch.Tensor | None,
    motion_mask_mode: str,
    token_shape: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if token_surprise.ndim == 1:
        token_surprise = token_surprise.unsqueeze(0)
    if token_surprise.ndim != 2:
        raise ValueError(f"Expected token_surprise [B,N] or [N], got {tuple(token_surprise.shape)}")

    future_depth, grid_h, grid_w = token_shape
    expected_tokens = future_depth * grid_h * grid_w
    if int(token_surprise.shape[-1]) != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} tokens for future window, got {int(token_surprise.shape[-1])}"
        )

    token_surprise = token_surprise.view(token_surprise.shape[0], future_depth, grid_h, grid_w)
    weights = _motion_weights_for_future_tokens(
        future_motion_mask_thw,
        token_shape=token_shape,
        motion_mask_mode=motion_mask_mode,
        device=device,
        dtype=dtype,
    )
    if weights is None or float(weights.sum().item()) <= 1.0e-6:
        return token_surprise.mean()
    if weights.shape[0] == 1 and token_surprise.shape[0] > 1:
        weights = weights.expand(token_surprise.shape[0], -1, -1, -1)
    if weights.shape[0] != token_surprise.shape[0]:
        raise ValueError(
            "motion mask batch does not match token batch: "
            f"weights={tuple(weights.shape)} tokens={tuple(token_surprise.shape)}"
        )
    return (token_surprise * weights).sum() / weights.sum().clamp_min(1.0e-6)


def compute_masked_predictive_surprise(
    video_btchw: torch.Tensor,
    *,
    encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    img_size: int,
    window_size: int = 16,
    context_frames: int = 8,
    stride: int = 4,
    reduction: str = "mean",
    future_motion_mask_thw: torch.Tensor | None = None,
    motion_mask_mode: str = "per_frame",
) -> torch.Tensor:
    add_vjepa_repo_to_path()
    from src.masks.utils import apply_masks

    if reduction not in {"mean", "max"}:
        raise ValueError(f"Unsupported reduction: {reduction}")
    if video_btchw.shape[2] < window_size:
        raise ValueError(
            f"Video has only {video_btchw.shape[2]} frames, but window_size={window_size}"
        )

    model_dtype = next(encoder.parameters()).dtype
    x = prepare_video_for_vjepa(video_btchw, img_size=img_size).to(dtype=model_dtype)
    pieces = _window_video(x, window_size=window_size, stride=stride)
    losses: list[torch.Tensor] = []

    for chunk_id in range(pieces.shape[0]):
        chunk = pieces[chunk_id : chunk_id + 1]
        masks_enc, masks_pred = generate_causal_masks(
            batch_size=chunk.shape[0],
            img_size=img_size,
            frames_per_clip=window_size,
            encoder=encoder,
            context_frames=context_frames,
            device=chunk.device,
        )

        target_tokens = target_encoder(chunk)
        if isinstance(target_tokens, (list, tuple)):
            if len(target_tokens) == 0:
                raise RuntimeError("V-JEPA target encoder returned an empty output list")
            target_tokens = target_tokens[-1]
        if not isinstance(target_tokens, torch.Tensor):
            raise RuntimeError(f"Unexpected V-JEPA target encoder output type: {type(target_tokens)!r}")
        target_tokens = F.layer_norm(target_tokens, (target_tokens.shape[-1],))
        context_tokens = encoder(chunk, masks_enc)
        context_tokens = predictor(context_tokens, masks_enc, masks_pred)
        context_tokens = F.layer_norm(context_tokens, (context_tokens.shape[-1],))

        masked_target = apply_masks(target_tokens, masks_pred, concat=False)
        token_surprise = 1.0 - F.cosine_similarity(context_tokens, masked_target[0], dim=-1)
        token_shape = _token_grid_shape(
            img_size=img_size,
            encoder=encoder,
            future_frames=window_size - context_frames,
        )
        surprise = _masked_token_surprise_mean(
            token_surprise,
            future_motion_mask_thw=future_motion_mask_thw,
            motion_mask_mode=motion_mask_mode,
            token_shape=token_shape,
            device=video_btchw.device,
            dtype=video_btchw.dtype,
        )
        losses.append(surprise)

    loss_stack = torch.stack(losses)
    if reduction == "mean":
        return loss_stack.mean()
    return loss_stack.max()


def _resample_frames(video_btchw: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Uniformly resample the temporal axis of [B,C,T,H,W] to exactly num_frames."""
    total = video_btchw.shape[2]
    if total == num_frames:
        return video_btchw
    idx = torch.linspace(0, total - 1, steps=num_frames, device=video_btchw.device)
    idx = idx.round().long().clamp_(0, total - 1)
    return video_btchw.index_select(2, idx)


def build_context_future_clip(
    *,
    context_btchw: torch.Tensor,
    future_btchw: torch.Tensor,
    window_size: int,
    context_frames: int,
) -> torch.Tensor:
    """Assemble a single [B,C,window_size,H,W] clip = [context | future].

    The context portion is detached (fixed real conditioning); the future portion
    keeps its graph so gradients flow only to the generated frames.
    """
    future_frames = window_size - context_frames
    if future_frames <= 0:
        raise ValueError(
            f"window_size={window_size} must exceed context_frames={context_frames}"
        )
    if context_btchw.shape[-2:] != future_btchw.shape[-2:]:
        future_btchw = F.interpolate(
            future_btchw.reshape(-1, *future_btchw.shape[-3:]),
            size=context_btchw.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).view(future_btchw.shape[0], future_btchw.shape[1], future_btchw.shape[2], *context_btchw.shape[-2:])

    ctx = _resample_frames(context_btchw, context_frames).detach()
    fut = _resample_frames(future_btchw, future_frames)
    return torch.cat([ctx, fut], dim=2)


def compute_context_anchored_alignment(
    clip_btchw: torch.Tensor,
    *,
    encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    img_size: int,
    window_size: int = 16,
    context_frames: int = 8,
    predicted_future_ref: torch.Tensor | None = None,
    future_motion_mask_thw: torch.Tensor | None = None,
    motion_mask_mode: str = "per_frame",
) -> torch.Tensor:
    """Energy = feature-space mismatch between the generated future frames and the
    future that V-JEPA's predictor forecasts from the *real* context.

    Unlike ``compute_masked_predictive_surprise`` (which windows the generation and
    only measures self-consistency), this anchors the target to the actual
    conditioning video, so minimizing it pushes the generated motion toward the
    physically-grounded continuation V-JEPA expects.

    ``clip_btchw`` is a single [B,3,window_size,H,W] clip = [real_context | generated].
    If ``predicted_future_ref`` is provided it is used as the fixed target and the
    encoder+predictor pass is skipped (the prediction depends only on the fixed
    context, so it can be precomputed once per generation).
    """
    add_vjepa_repo_to_path()
    from src.masks.utils import apply_masks

    if clip_btchw.shape[2] != window_size:
        raise ValueError(
            f"Expected clip with {window_size} frames, got {clip_btchw.shape[2]}"
        )

    model_dtype = next(target_encoder.parameters()).dtype
    x = prepare_video_for_vjepa(clip_btchw, img_size=img_size).to(dtype=model_dtype)

    masks_enc, masks_pred = generate_causal_masks(
        batch_size=x.shape[0],
        img_size=img_size,
        frames_per_clip=window_size,
        encoder=target_encoder,
        context_frames=context_frames,
        device=x.device,
    )

    if predicted_future_ref is None:
        context_tokens = encoder(x, masks_enc)
        predicted = predictor(context_tokens, masks_enc, masks_pred)
        predicted = F.layer_norm(predicted, (predicted.shape[-1],))
    else:
        predicted = predicted_future_ref.to(device=x.device, dtype=x.dtype)

    target_tokens = target_encoder(x)
    if isinstance(target_tokens, (list, tuple)):
        target_tokens = target_tokens[-1]
    target_tokens = F.layer_norm(target_tokens, (target_tokens.shape[-1],))
    masked_target = apply_masks(target_tokens, masks_pred, concat=False)[0]

    token_surprise = 1.0 - F.cosine_similarity(predicted, masked_target, dim=-1)
    token_shape = _token_grid_shape(
        img_size=img_size,
        encoder=target_encoder,
        future_frames=window_size - context_frames,
    )
    return _masked_token_surprise_mean(
        token_surprise,
        future_motion_mask_thw=future_motion_mask_thw,
        motion_mask_mode=motion_mask_mode,
        token_shape=token_shape,
        device=clip_btchw.device,
        dtype=clip_btchw.dtype,
    )


def precompute_future_prediction(
    clip_btchw: torch.Tensor,
    *,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    img_size: int,
    window_size: int = 16,
    context_frames: int = 8,
) -> torch.Tensor:
    """Run encoder+predictor once on a [context | placeholder] clip to obtain the
    fixed future-feature target. Only the context portion affects the result."""
    add_vjepa_repo_to_path()

    model_dtype = next(encoder.parameters()).dtype
    x = prepare_video_for_vjepa(clip_btchw, img_size=img_size).to(dtype=model_dtype)
    masks_enc, masks_pred = generate_causal_masks(
        batch_size=x.shape[0],
        img_size=img_size,
        frames_per_clip=window_size,
        encoder=encoder,
        context_frames=context_frames,
        device=x.device,
    )
    context_tokens = encoder(x, masks_enc)
    predicted = predictor(context_tokens, masks_enc, masks_pred)
    predicted = F.layer_norm(predicted, (predicted.shape[-1],))
    return predicted.detach()


class VJEPASurpriseEnergy:
    def __init__(
        self,
        model_name: str = "vitg",
        *,
        device: torch.device | str = "cuda",
        repo_root: Path = VJEPA2_REPO,
        local_torchhub: bool = True,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        encoder, target_encoder, predictor, img_size = load_vjepa2_models(
            model_name=model_name,
            repo_root=repo_root,
            local_torchhub=local_torchhub,
            checkpoint_path=checkpoint_path,
        )
        self.encoder = freeze_module(encoder.to(self.device))
        self.target_encoder = freeze_module(target_encoder.to(self.device))
        self.predictor = freeze_module(predictor.to(self.device))
        self.img_size = img_size
        self.model_name = model_name
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None

    def __call__(
        self,
        video_btchw: torch.Tensor,
        *,
        window_size: int = 16,
        context_frames: int = 8,
        stride: int = 4,
        reduction: str = "mean",
        future_motion_mask_thw: torch.Tensor | None = None,
        motion_mask_mode: str = "per_frame",
    ) -> torch.Tensor:
        return compute_masked_predictive_surprise(
            video_btchw.to(self.device),
            encoder=self.encoder,
            target_encoder=self.target_encoder,
            predictor=self.predictor,
            img_size=self.img_size,
            window_size=window_size,
            context_frames=context_frames,
            stride=stride,
            reduction=reduction,
            future_motion_mask_thw=future_motion_mask_thw.to(self.device) if future_motion_mask_thw is not None else None,
            motion_mask_mode=motion_mask_mode,
        )

    def context_anchored(
        self,
        clip_btchw: torch.Tensor,
        *,
        window_size: int = 16,
        context_frames: int = 8,
        predicted_future_ref: torch.Tensor | None = None,
        future_motion_mask_thw: torch.Tensor | None = None,
        motion_mask_mode: str = "per_frame",
    ) -> torch.Tensor:
        return compute_context_anchored_alignment(
            clip_btchw.to(self.device),
            encoder=self.encoder,
            target_encoder=self.target_encoder,
            predictor=self.predictor,
            img_size=self.img_size,
            window_size=window_size,
            context_frames=context_frames,
            predicted_future_ref=predicted_future_ref,
            future_motion_mask_thw=future_motion_mask_thw.to(self.device) if future_motion_mask_thw is not None else None,
            motion_mask_mode=motion_mask_mode,
        )

    def precompute_future_prediction(
        self,
        clip_btchw: torch.Tensor,
        *,
        window_size: int = 16,
        context_frames: int = 8,
    ) -> torch.Tensor:
        return precompute_future_prediction(
            clip_btchw.to(self.device),
            encoder=self.encoder,
            predictor=self.predictor,
            img_size=self.img_size,
            window_size=window_size,
            context_frames=context_frames,
        )
