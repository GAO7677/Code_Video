from __future__ import annotations

import torch


def latent_frame_mask(
    num_video_frames: int,
    num_context_frames: int,
    vae_stride_t: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    lat_t = (num_video_frames - 1) // vae_stride_t + 1
    context_lat_t = max(1, (num_context_frames - 1) // vae_stride_t + 1)
    context_mask = torch.zeros(lat_t, dtype=torch.float32, device=device)
    context_mask[:context_lat_t] = 1.0
    future_mask = 1.0 - context_mask
    return context_mask, future_mask


def broadcast_latent_mask(mask_t: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    # latents: [C, T, H, W]
    return mask_t.view(1, -1, 1, 1).to(dtype=latents.dtype, device=latents.device)


def expand_context_latents_to_full(
    context_latents: torch.Tensor,
    full_latents: torch.Tensor,
) -> torch.Tensor:
    # context_latents/full_latents: [C, T, H, W]
    out = torch.zeros_like(full_latents)
    copy_t = min(context_latents.shape[1], full_latents.shape[1])
    out[:, :copy_t] = context_latents[:, :copy_t]
    return out


def collate_video_batch(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("cannot collate an empty batch")

    batch: dict[str, object] = {}
    keys = samples[0].keys()
    for key in keys:
        values = [sample[key] for sample in samples]
        first = values[0]
        if isinstance(first, torch.Tensor):
            if key == "context_video":
                max_t = max(int(v.shape[1]) for v in values)
                padded = []
                for value in values:
                    if value.shape[1] == max_t:
                        padded.append(value)
                        continue
                    pad = value[:, -1:].expand(-1, max_t - value.shape[1], -1, -1).contiguous()
                    padded.append(torch.cat([value, pad], dim=1))
                batch[key] = torch.stack(padded, dim=0)
            elif key in {"context_boxes", "context_states"}:
                max_t = max(int(v.shape[0]) for v in values)
                padded = []
                for value in values:
                    if value.shape[0] == max_t:
                        padded.append(value)
                        continue
                    pad_shape = (max_t - value.shape[0],) + tuple(value.shape[1:])
                    pad = value.new_zeros(pad_shape)
                    padded.append(torch.cat([value, pad], dim=0))
                batch[key] = torch.stack(padded, dim=0)
            else:
                batch[key] = torch.stack(values, dim=0)
        elif isinstance(first, (int, float)):
            batch[key] = torch.tensor(values)
        else:
            batch[key] = values
    return batch
