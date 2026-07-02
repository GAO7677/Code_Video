from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


VJEPA2_REPO = Path("/home/gaoya/Code_Video/vjepa2-main")
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

_MODEL_ALIASES = {
    "vith": "vjepa2_vit_huge",
    "vitg": "vjepa2_vit_giant",
    "vitg384": "vjepa2_vit_giant_384",
}

_DEFAULT_CHECKPOINTS = {
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
    checkpoint_path = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
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
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"V-JEPA checkpoint not found: {checkpoint_path}")

        add_vjepa_repo_to_path(repo_root)
        from src.hub.backbones import _clean_backbone_key

        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
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
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return ctxt_positions, tgt_positions


def _window_video(video_btchw: torch.Tensor, window_size: int, stride: int) -> torch.Tensor:
    bsz, channels, _, height, width = video_btchw.shape
    pieces = video_btchw.unfold(2, window_size, stride)
    pieces = pieces.permute(0, 2, 5, 1, 3, 4).contiguous()
    pieces = pieces.view(-1, window_size, channels, height, width)
    return pieces.permute(0, 2, 1, 3, 4).contiguous()


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
        if isinstance(target_tokens, torch.Tensor):
            target_token_list = [target_tokens]
        else:
            target_token_list = list(target_tokens)
        target_tokens = torch.stack(
            [F.layer_norm(tokens, (tokens.shape[-1],)) for tokens in target_token_list]
        )
        context_tokens = encoder(chunk, masks_enc)
        context_tokens = predictor(context_tokens, masks_enc, masks_pred)
        context_tokens = F.layer_norm(context_tokens, (context_tokens.shape[-1],))

        masked_target = apply_masks(target_tokens, masks_pred, concat=False)
        surprise = 1.0 - F.cosine_similarity(context_tokens, masked_target[0], dim=-1).mean()
        losses.append(surprise)

    loss_stack = torch.stack(losses)
    if reduction == "mean":
        return loss_stack.mean()
    return loss_stack.max()


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
        )
