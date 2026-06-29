from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PredictorLossOutput:
    loss_total: torch.Tensor
    loss_token: torch.Tensor
    loss_track: torch.Tensor
    loss_box: torch.Tensor


def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    diff = (pred - target).abs()
    if valid_mask is None:
        return diff.mean()
    weights = valid_mask.to(dtype=pred.dtype, device=pred.device)
    while weights.ndim < diff.ndim:
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(diff)
    denom = weights.sum().clamp_min(1.0)
    return (diff * weights).sum() / denom


def compute_predictor_losses(
    *,
    pred_future_tokens: torch.Tensor,
    teacher_future_tokens: torch.Tensor,
    pred_future_track_summary: torch.Tensor,
    gt_future_track_summary: torch.Tensor,
    gt_future_track_valid: torch.Tensor,
    pred_future_box_xyxy: torch.Tensor,
    gt_future_box_xyxy: torch.Tensor,
    gt_future_box_valid: torch.Tensor,
    lambda_token: float,
    lambda_track: float,
    lambda_box: float,
) -> PredictorLossOutput:
    token_valid = gt_future_box_valid
    token_diff = (torch.nan_to_num(pred_future_tokens, nan=0.0, posinf=0.0, neginf=0.0) - torch.nan_to_num(
        teacher_future_tokens,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )).abs()
    token_weights = token_valid.to(dtype=pred_future_tokens.dtype, device=pred_future_tokens.device)
    while token_weights.ndim < token_diff.ndim:
        token_weights = token_weights.unsqueeze(-1)
    token_weights = token_weights.expand_as(token_diff)
    token_denom = token_weights.sum().clamp_min(1.0)
    loss_token = (token_diff * token_weights).sum() / token_denom
    loss_track = masked_l1_loss(
        pred_future_track_summary,
        gt_future_track_summary,
        gt_future_track_valid,
    )
    loss_box = masked_l1_loss(
        pred_future_box_xyxy,
        gt_future_box_xyxy,
        gt_future_box_valid,
    )
    total = (
        float(lambda_token) * loss_token
        + float(lambda_track) * loss_track
        + float(lambda_box) * loss_box
    )
    return PredictorLossOutput(
        loss_total=total,
        loss_token=loss_token,
        loss_track=loss_track,
        loss_box=loss_box,
    )
