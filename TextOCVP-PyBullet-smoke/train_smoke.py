#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

HERE = Path(__file__).resolve().parent
TEXTOCVP_ROOT = Path("/home/gaoya/Code_Video/TextOCVP-master")
sys.path.insert(0, str(TEXTOCVP_ROOT / "src"))
sys.path.insert(0, str(HERE))

from models.Predictors.predictor_wrapper import PredictorWrapper  # noqa: E402
from models.Predictors.text_cond_OCVP import TextOCVP_T5  # noqa: E402
from models.SAVi import SAVi  # noqa: E402
from pybullet_dataset import PyBulletTextOCVPDataset  # noqa: E402
from transformers import T5EncoderModel, T5Tokenizer  # noqa: E402


LOCAL_T5 = Path("/data/gaoya/agent-data/cache/textocvp/t5-small")


class LocalTextOCVP_T5(TextOCVP_T5):
    def _instantiate_text_encoder(self) -> None:
        self.text_encoder = T5EncoderModel.from_pretrained(LOCAL_T5)
        self.text_encoder.requires_grad_(False)
        self.t5_token_dim = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dataset-limit", type=int, default=32)
    parser.add_argument("--decomp-steps", type=int, default=6)
    parser.add_argument("--predictor-steps", type=int, default=3)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=112)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--num-slots", type=int, default=6)
    parser.add_argument("--slot-dim", type=int, default=128)
    return parser.parse_args()


def build_savi(args: argparse.Namespace) -> SAVi:
    resolution = [args.height, args.width]
    return SAVi(
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
        num_iterations_first=3,
        num_iterations=1,
        in_channels=3,
        mlp_hidden=256,
        mlp_encoder_dim=128,
        initializer="LearnedRandom",
        transition_module={
            "model_name": "TransformerBlock",
            "num_heads": 4,
            "mlp_size": 512,
        },
        encoder=copy.deepcopy(
            {
                "encoder_name": "ConvEncoder",
                "encoder_params": {
                    "num_channels": [32, 32, 32, 32],
                    "kernel_size": 5,
                    "resolution": resolution,
                    "downsample_encoder": False,
                    "downsample": 2,
                },
            }
        ),
        decoder=copy.deepcopy(
            {
                "decoder_name": "ConvDecoder",
                "decoder_params": {
                    "num_channels": [64, 64, 64, 64],
                    "kernel_size": 5,
                    "resolution": resolution,
                    "downsample_decoder": False,
                    "upsample": 1,
                },
            }
        ),
    )


def build_predictor(slot_dim: int) -> PredictorWrapper:
    exp_params = {
        "model": {"model_params": {"slot_dim": slot_dim}},
        "predictor": {
            "predictor_name": "TextOCVP_T5",
            "predictor_params": {},
        },
        "prediction_params": {
            "num_context": 1,
            "num_preds": 9,
            "teacher_force": False,
            "input_buffer_size": 10,
        },
    }
    predictor = LocalTextOCVP_T5(
        slot_dim=slot_dim,
        predictor_params={
            "token_dim": 512,
            "n_heads": 8,
            "hidden_dim": 1024,
            "num_layers": 2,
            "residual": True,
            "input_buffer_size": 10,
        },
        fusion_params={"num_heads": 8, "head_dim": 64, "mlp_size": 1024},
        text_encoder_params={},
    )
    return PredictorWrapper(exp_params=exp_params, predictor=predictor)


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def save_decomposition_visualization(
    video: torch.Tensor,
    output: dict[str, torch.Tensor],
    output_dir: Path,
) -> None:
    count = min(5, video.shape[1])
    rows = [video[0, :count].cpu(), output["recons_imgs"][0, :count].cpu()]
    masks = output["masks"][0, 0].repeat(1, 3, 1, 1).cpu()
    rows.append(masks)
    grid = make_grid(torch.cat(rows, dim=0), nrow=max(count, masks.shape[0]), padding=2)
    save_image(grid, output_dir / "decomposition_grid.png")


def save_prediction_visualization(
    target: torch.Tensor,
    prediction: torch.Tensor,
    output_dir: Path,
) -> None:
    count = min(9, target.shape[1])
    grid = make_grid(
        torch.cat([target[0, :count].cpu(), prediction[0, :count].cpu()], dim=0),
        nrow=count,
        padding=2,
    )
    save_image(grid, output_dir / "prediction_grid.png")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke")
    device = torch.device(f"cuda:{args.gpu}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PyBulletTextOCVPDataset(
        args.dataset_root,
        "train",
        num_frames=args.num_frames,
        image_hw=(args.height, args.width),
        limit=args.dataset_limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    savi = build_savi(args).to(device)
    savi_optimizer = torch.optim.Adam(savi.parameters(), lr=1.0e-4)
    iterator = iter(loader)
    decomp_losses: list[float] = []
    print(f"[dataset] samples={len(dataset)} frames={args.num_frames} hw={args.height}x{args.width}")
    print(f"[savi] trainable={sum(p.numel() for p in savi.parameters() if p.requires_grad):,}")

    last_batch = None
    last_output = None
    savi.train()
    for step in range(1, args.decomp_steps + 1):
        batch, iterator = next_batch(iterator, loader)
        video = batch["video"].to(device, non_blocking=True)
        output = savi(x=video, num_imgs=video.shape[1], decode=True)
        loss = F.mse_loss(output["recons_imgs"].clamp(0.0, 1.0), video)
        savi_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(savi.parameters(), 0.05)
        savi_optimizer.step()
        decomp_losses.append(float(loss.detach()))
        print(
            f"[decomp {step:03d}] loss={loss.item():.6f} grad={float(grad_norm):.6f} "
            f"video={tuple(video.shape)} slots={tuple(output['slot_history'].shape)} "
            f"masks={tuple(output['masks'].shape)}"
        )
        last_batch, last_output = video.detach(), output

    assert last_batch is not None and last_output is not None
    save_decomposition_visualization(last_batch, last_output, args.output_dir)

    for parameter in savi.parameters():
        parameter.requires_grad_(False)
    savi.eval()
    tokenizer = T5Tokenizer.from_pretrained(LOCAL_T5)
    predictor = build_predictor(args.slot_dim).to(device)
    predictor.train()
    predictor_optimizer = torch.optim.Adam(
        [p for p in predictor.parameters() if p.requires_grad], lr=1.0e-4
    )
    predictor_losses: list[float] = []
    iterator = iter(loader)
    last_target = None
    last_prediction = None
    print(
        f"[predictor] trainable={sum(p.numel() for p in predictor.parameters() if p.requires_grad):,}"
    )

    for step in range(1, args.predictor_steps + 1):
        batch, iterator = next_batch(iterator, loader)
        video = batch["video"].to(device, non_blocking=True)
        tokenized = tokenizer(
            list(batch["caption"]),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            decomposed = savi(x=video, num_imgs=video.shape[1], decode=False)
            target_slots = decomposed["slot_history"]
        pred_slots = predictor(
            target_slots,
            caption_tokens=tokenized["input_ids"].to(device),
            attn_masks=tokenized["attention_mask"].to(device),
        )
        decoded = savi(
            mode="decode",
            slots=pred_slots.reshape(-1, args.num_slots, args.slot_dim),
        )
        pred_frames = decoded["recons_imgs"].view(
            video.shape[0], 9, 3, args.height, args.width
        )
        target_frames = video[:, 1:10]
        loss_img = F.mse_loss(pred_frames.clamp(0.0, 1.0), target_frames)
        loss_slot = F.mse_loss(pred_slots, target_slots[:, 1:10])
        loss = loss_img + loss_slot
        predictor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), 0.05)
        predictor_optimizer.step()
        predictor_losses.append(float(loss.detach()))
        print(
            f"[predictor {step:03d}] loss={loss.item():.6f} img={loss_img.item():.6f} "
            f"slot={loss_slot.item():.6f} grad={float(grad_norm):.6f} "
            f"pred_slots={tuple(pred_slots.shape)} pred_frames={tuple(pred_frames.shape)}"
        )
        last_target, last_prediction = target_frames.detach(), pred_frames.detach()

    assert last_target is not None and last_prediction is not None
    save_prediction_visualization(last_target, last_prediction, args.output_dir)

    checkpoint = {
        "savi": savi.state_dict(),
        "predictor": predictor.state_dict(),
        "args": vars(args),
        "decomp_losses": decomp_losses,
        "predictor_losses": predictor_losses,
    }
    torch.save(checkpoint, args.output_dir / "smoke_checkpoint.pt")
    report = {
        "status": "passed",
        "dataset_samples": len(dataset),
        "video_shape": list(last_batch.shape),
        "slot_shape": list(last_output["slot_history"].shape),
        "mask_shape": list(last_output["masks"].shape),
        "decomposition_losses": decomp_losses,
        "predictor_losses": predictor_losses,
        "checkpoint": str(args.output_dir / "smoke_checkpoint.pt"),
    }
    (args.output_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(f"[passed] output={args.output_dir}")


if __name__ == "__main__":
    main()
