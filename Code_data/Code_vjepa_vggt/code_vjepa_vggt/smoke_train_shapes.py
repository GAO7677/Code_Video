from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackAdapter
from code_vjepa_vggt.adapters.jepa_adapter import JEPAPatchAdapter
from code_vjepa_vggt.data import PhysStateEpisodeDataset
from code_vjepa_vggt.models.context_fuser import ContextTokenFuser
from code_vjepa_vggt.models.object_tokens import ObjectTubeProjector
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.masks import broadcast_latent_mask, expand_context_latents_to_full, latent_frame_mask
from code_vjepa_vggt.utils.paths import ensure_upstream_paths
from code_vjepa_vggt.utils.track_supervision import align_tracks_to_boxes, track_box_l1_loss

ensure_upstream_paths()


def shape_of_list(xs: list[torch.Tensor]) -> list[list[int]]:
    return [list(x.shape) for x in xs]


def _ensure_fake_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_wan_components():
    wan_root = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main/wan")
    modules_root = wan_root / "modules"
    configs_root = wan_root / "configs"

    _ensure_fake_package("wan", wan_root)
    _ensure_fake_package("wan.modules", modules_root)
    _ensure_fake_package("wan.configs", configs_root)

    _load_module("wan.modules.tokenizers", modules_root / "tokenizers.py")
    t5_mod = _load_module("wan.modules.t5", modules_root / "t5.py")
    vae_mod = _load_module("wan.modules.vae2_2", modules_root / "vae2_2.py")
    _load_module("wan.configs.shared_config", configs_root / "shared_config.py")
    cfg_mod = _load_module("wan.configs.wan_ti2v_5B", configs_root / "wan_ti2v_5B.py")

    return t5_mod.T5EncoderModel, vae_mod.Wan2_2_VAE, cfg_mod.ti2v_5B


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/inspect_phys_state_vjepa_vggt.yaml",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--output",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/outputs/smoke_train_shapes.json",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    T5EncoderModel, Wan2_2_VAE, wan_cfg = load_wan_components()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PhysStateEpisodeDataset(
        root=data_cfg["root"],
        split=data_cfg["split"],
        resolution=tuple(data_cfg["resolution"]),
        num_context_frames=int(data_cfg["num_context_frames"]),
        context_fraction=float(data_cfg.get("context_fraction", 0.5)),
        random_context_frames=bool(data_cfg.get("random_context_frames", True)),
        seed=int(cfg.get("experiment", {}).get("seed", 42)),
    )
    sample = dataset[args.index]

    video = sample["video"].unsqueeze(0).to(device)
    context_video = sample["context_video"].unsqueeze(0).to(device)
    caption = [sample["caption"]]
    context_boxes = sample["context_boxes"].unsqueeze(0).to(device)

    text_encoder = T5EncoderModel(
        text_len=wan_cfg.text_len,
        dtype=wan_cfg.t5_dtype,
        device=torch.device("cpu"),
        checkpoint_path=f"{model_cfg['wan_ckpt_dir']}/{wan_cfg.t5_checkpoint}",
        tokenizer_path=f"{model_cfg['wan_ckpt_dir']}/{wan_cfg.t5_tokenizer}",
        shard_fn=None,
    )
    with torch.no_grad():
        text_ctx = [u.to(device) for u in text_encoder(caption, torch.device("cpu"))]

    vae = Wan2_2_VAE(
        vae_pth=f"{model_cfg['wan_ckpt_dir']}/{wan_cfg.vae_checkpoint}",
        device=device,
    )
    with torch.no_grad():
        full_latents = vae.encode([video[0]])
        context_latents = vae.encode([context_video[0]])
    context_latent_batch = torch.stack(context_latents, dim=0)

    jepa = JEPAPatchAdapter(
        ckpt_path=str(Path(model_cfg["je_pa_ckpt_dir"]) / "original" / "model.pth"),
        device=str(device),
        crop_size=int(model_cfg["jepa_input_size"]),
        num_frames=int(data_cfg["num_context_frames"]),
        patch_size=int(model_cfg["jepa_patch_size"]),
        tubelet_size=int(model_cfg["jepa_tubelet_size"]),
    ).to(device)
    with torch.no_grad():
        jepa_out = jepa(context_video)

    frames_bthwc = context_video.permute(0, 2, 3, 4, 1).float()
    frames_bthwc = (frames_bthwc + 1.0) / 2.0
    vggt = VGGTTrackAdapter(
        model_path=model_cfg.get("vggt_model_path"),
        num_queries=int(model_cfg["object_num_queries"]),
        device=str(device),
        input_hw=tuple(model_cfg["vggt_input_hw"]),
    ).to(device)
    with torch.no_grad():
        vggt_out = vggt(frames_bthwc)
    tracks = vggt_out.tracks
    vis = vggt_out.visibility
    conf = vggt_out.confidence
    track_image_hw = vggt_out.image_hw

    scale_x = float(context_video.shape[-1]) / float(track_image_hw[1])
    scale_y = float(context_video.shape[-2]) / float(track_image_hw[0])
    tracks_native = tracks.clone()
    tracks_native[..., 0] *= scale_x
    tracks_native[..., 1] *= scale_y
    track_alignment = align_tracks_to_boxes(
        tracks=tracks_native,
        gt_boxes=context_boxes,
        image_hw=(context_video.shape[-2], context_video.shape[-1]),
    )
    track_box_loss = track_box_l1_loss(
        tracks=tracks_native,
        matched_gt_centers=track_alignment.matched_gt_centers,
        matched_gt_valid=track_alignment.matched_gt_valid,
    )

    latent_dim = int(getattr(wan_cfg, "in_dim", 16))
    object_pooler = ObjectTubeProjector(
        jepa_dim=jepa.encoder.backbone.embed_dim,
        latent_dim=latent_dim,
        out_dim=int(model_cfg["cond_proj_dim"]),
        jepa_window_radius=int(model_cfg["jepa_window_radius"]),
        latent_window_radius=int(model_cfg["latent_window_radius"]),
    ).to(device)
    with torch.no_grad():
        object_out = object_pooler(
            jepa_patch_tokens=jepa_out.patch_tokens,
            context_latents=context_latent_batch,
            tracks=tracks,
            visibility=vis,
            confidence=conf,
            track_image_hw=track_image_hw,
        )

    fuser = ContextTokenFuser(
        text_dim=int(model_cfg["cond_proj_dim"]),
        max_context_len=wan_cfg.text_len,
    ).to(device)
    with torch.no_grad():
        fused_context = fuser(text_ctx, object_out.object_tokens)

    latent_clean = full_latents[0]
    context_clean = context_latents[0]
    noise = torch.randn_like(latent_clean)
    timestep_scalar = torch.tensor([123.0], device=device)
    t_norm = timestep_scalar / float(wan_cfg.num_train_timesteps)
    x_t_noisy = (1.0 - t_norm) * latent_clean + t_norm * noise
    context_mask_t, future_mask_t = latent_frame_mask(
        num_video_frames=video.shape[2],
        num_context_frames=context_video.shape[2],
        vae_stride_t=wan_cfg.vae_stride[0],
        device=device,
    )
    context_mask = broadcast_latent_mask(context_mask_t, latent_clean)
    future_mask = broadcast_latent_mask(future_mask_t, latent_clean)
    context_clean_full = expand_context_latents_to_full(context_clean, latent_clean)
    x_t = context_mask * context_clean_full + (1.0 - context_mask) * x_t_noisy
    seq_len = x_t.shape[1] * x_t.shape[2] * x_t.shape[3] // (wan_cfg.patch_size[1] * wan_cfg.patch_size[2])
    t_tokens = torch.full((1, seq_len), timestep_scalar.item(), device=device)

    report = {
        "status": "partial_train_smoke_ok",
        "note": "Smoke test ran through dataset sampling, text encode, VAE encode, JEPA encode, object token pooling, and latent/noise masking. Wan DiT forward + loss backward are not run here because the current environment blocks flash_attention-backed WanModel execution.",
        "sample_index": args.index,
        "caption": sample["caption"],
        "video_path": sample["video_path"],
        "context_frame_indices": sample["context_frame_indices"].tolist(),
        "track_box_l1_loss": float(track_box_loss.item()),
        "shapes": {
            "video": list(video.shape),
            "context_video": list(context_video.shape),
            "context_boxes": list(context_boxes.shape),
            "text_context": shape_of_list(text_ctx),
            "full_latents": shape_of_list(full_latents),
            "context_latents": shape_of_list(context_latents),
            "jepa_patch_tokens": list(jepa_out.patch_tokens.shape),
            "vggt_query_points": list(vggt_out.query_points.shape),
            "tracks": list(tracks.shape),
            "tracks_native_xy": list(tracks_native.shape),
            "visibility": list(vis.shape),
            "confidence": list(conf.shape),
            "track_image_hw": list(track_image_hw),
            "matched_gt_indices": list(track_alignment.matched_gt_indices.shape),
            "matched_gt_centers": list(track_alignment.matched_gt_centers.shape),
            "matched_gt_valid": list(track_alignment.matched_gt_valid.shape),
            "track_pair_cost": list(track_alignment.pair_cost.shape),
            "object_jepa_tokens": list(object_out.jepa_tokens.shape),
            "object_latent_tokens": list(object_out.latent_tokens.shape),
            "object_geom_tokens": list(object_out.geom_tokens.shape),
            "object_tokens": list(object_out.object_tokens.shape),
            "fused_context": shape_of_list(fused_context),
            "latent_clean": list(latent_clean.shape),
            "noise": list(noise.shape),
            "x_t_noisy": list(x_t_noisy.shape),
            "context_mask_t": list(context_mask_t.shape),
            "future_mask_t": list(future_mask_t.shape),
            "context_mask": list(context_mask.shape),
            "future_mask": list(future_mask.shape),
            "context_clean_full": list(context_clean_full.shape),
            "x_t_after_context_restore": list(x_t.shape),
            "t_tokens": list(t_tokens.shape),
            "seq_len": seq_len,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
