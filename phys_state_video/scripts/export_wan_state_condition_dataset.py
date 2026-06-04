from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.dataset import NpzPredictorDataset, collate_predictor_episodes
from phys_state_video.predictor_wan_state import WanStateLatentPredictor, WanStateLatentPredictorConfig
from phys_state_video.predictor_wan_state_v2 import WanStateLatentPredictorV2, WanStateLatentPredictorV2Config
from phys_state_video.utils import detach_to_cpu_numpy, require_torch
from phys_state_video.wan_bridge import WanLatentExtractor
from phys_state_video.wan_state_v2_helpers import (
    MockLatentExtractor,
    compute_future_latent_steps,
    resample_camera_to_latent_steps,
)

torch = require_torch()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export phys_state_video episodes into Wan-compatible state_condition bundles."
    )
    parser.add_argument("--episodes", required=True, help="Episode .npz file or directory.")
    parser.add_argument("--output", required=True, help="Output directory for Wan state-condition bundles.")
    parser.add_argument(
        "--future-state-source",
        default="ground_truth",
        choices=["ground_truth", "wan_predictor"],
        help=(
            "Source of future condition. "
            "`ground_truth` exports `predicted_states=future_states`; "
            "`wan_predictor` exports predictor `state_tokens` and `predicted_states`."
        ),
    )
    parser.add_argument(
        "--predictor",
        default=None,
        help="Wan-state predictor checkpoint. Required when --future-state-source=wan_predictor.",
    )
    parser.add_argument(
        "--wan-ckpt-dir",
        default=None,
        help="Wan checkpoint directory used by WanLatentExtractor. Required for wan_predictor export.",
    )
    parser.add_argument("--wan-repo-root", default="/home/gaoya/Code_Video/Wan2.2-main")
    parser.add_argument("--wan-task", default="i2v-A14B")
    parser.add_argument(
        "--predictor-latent-source",
        default="auto",
        choices=["auto", "mock", "wan"],
        help="How to build latents when exporting from a predictor checkpoint.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0, help="If > 0, export only the first N episodes.")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, map_location):
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def load_wan_state_predictor(checkpoint_path: str, device: str):
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    predictor_version = checkpoint.get("predictor_version")
    if predictor_version == "wan_state_v2_latent_time":
        model = WanStateLatentPredictorV2(WanStateLatentPredictorV2Config(**config)).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model, checkpoint
    if predictor_version != "wan_state_v1" and "latent_channels" not in config:
        raise ValueError(
            f"checkpoint does not look like WanStateLatentPredictor: {checkpoint_path} "
            f"(predictor_version={predictor_version!r}, config_keys={sorted(config.keys())})"
        )
    model = WanStateLatentPredictor(WanStateLatentPredictorConfig(**config)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def build_predictor_latent_extractor(args, predictor_ckpt):
    predictor_version = predictor_ckpt.get("predictor_version", "wan_state_v1")
    latent_source = args.predictor_latent_source
    if latent_source == "auto":
        latent_source = predictor_ckpt.get("latent_source", "wan" if predictor_version == "wan_state_v1" else "mock")
    if latent_source == "mock":
        return MockLatentExtractor(
            latent_channels=int(predictor_ckpt.get("mock_latent_channels") or predictor_ckpt["config"]["latent_channels"]),
            latent_height=int(predictor_ckpt.get("mock_latent_height") or 8),
            latent_width=int(predictor_ckpt.get("mock_latent_width") or 8),
            device=args.device,
        )
    if args.wan_ckpt_dir is None:
        raise ValueError("--wan-ckpt-dir is required when predictor latent source is Wan")
    return WanLatentExtractor(
        ckpt_dir=args.wan_ckpt_dir,
        wan_repo_root=args.wan_repo_root,
        task=args.wan_task,
        device=args.device,
    )


def save_frame_png(frame_chw: np.ndarray, path: Path) -> None:
    frame = np.clip(frame_chw, 0.0, 1.0)
    frame = (frame.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    Image.fromarray(frame).save(path)


def build_ground_truth_state_condition(batch) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    future_states = detach_to_cpu_numpy(batch["future_states"][0]).astype(np.float32)
    return {
        "predicted_states": future_states,
    }, {
        "future_condition_kind": "ground_truth_predicted_states",
        "predicted_states_shape": list(future_states.shape),
    }


def build_predictor_state_condition(
    batch,
    predictor,
    predictor_ckpt,
    latent_extractor,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    predictor_version = predictor_ckpt.get("predictor_version", "wan_state_v1")
    with torch.no_grad():
        context_frames = batch["context_frames"].to(device)
        if predictor_version == "wan_state_v2_latent_time":
            context_latents = latent_extractor.encode_context_frames_raw(context_frames)
            context_latent_steps = int(context_latents.shape[1])
            future_latent_steps = compute_future_latent_steps(
                context_steps=context_frames.shape[1],
                future_steps=batch["future_states"].shape[1],
                temporal_stride=latent_extractor.temporal_stride,
            )
            camera_latent = resample_camera_to_latent_steps(batch["camera"].to(device), context_latent_steps)
            outputs = predictor(
                context_latents=context_latents,
                camera=camera_latent,
                prompt_token_ids=batch["prompt_token_ids"].to(device),
                prompt_token_mask=batch["prompt_token_mask"].to(device),
                future_latent_steps=future_latent_steps,
                num_objects=batch["context_states"].shape[2],
            )
        else:
            context_latents = latent_extractor.encode_context_frames(context_frames)
            outputs = predictor(
                context_latents,
                batch["camera"].to(device),
                prompt_token_ids=batch["prompt_token_ids"].to(device),
                prompt_token_mask=batch["prompt_token_mask"].to(device),
                future_steps=batch["future_states"].shape[1],
                num_objects=batch["context_states"].shape[2],
            )

    state_tokens = detach_to_cpu_numpy(outputs["state_tokens"][0]).astype(np.float32)
    memory_tokens = detach_to_cpu_numpy(outputs["memory_tokens"][0]).astype(np.float32)
    condition_maps = detach_to_cpu_numpy(outputs["condition_maps"][0]).astype(np.float32)
    predicted_states = detach_to_cpu_numpy(outputs["future_state_predictions"][0]).astype(np.float32)
    context_state_predictions = detach_to_cpu_numpy(outputs["context_state_predictions"][0]).astype(np.float32)
    meta = {
        "state_tokens": state_tokens,
        "memory_tokens": memory_tokens,
        "condition_maps": condition_maps,
        "predicted_states": predicted_states,
        "context_state_predictions": context_state_predictions,
        "future_state_maps": detach_to_cpu_numpy(outputs["future_state_maps"][0]).astype(np.float32),
        "future_object_slots": detach_to_cpu_numpy(outputs["future_object_slots"][0]).astype(np.float32),
        "context_object_slots": detach_to_cpu_numpy(outputs["context_object_slots"][0]).astype(np.float32),
    }
    condition_meta = {
        "future_condition_kind": "wan_predictor_state_tokens",
        "predictor_version": predictor_version,
        "state_tokens_shape": list(state_tokens.shape),
        "memory_tokens_shape": list(memory_tokens.shape),
        "condition_maps_shape": list(condition_maps.shape),
        "predicted_states_shape": list(predicted_states.shape),
        "context_state_predictions_shape": list(context_state_predictions.shape),
        "future_state_maps_shape": list(meta["future_state_maps"].shape),
        "future_object_slots_shape": list(meta["future_object_slots"].shape),
        "context_object_slots_shape": list(meta["context_object_slots"].shape),
    }
    if predictor_version == "wan_state_v2_latent_time":
        condition_meta["context_latent_steps"] = int(context_latents.shape[1])
        condition_meta["future_latent_steps"] = int(state_tokens.shape[0])
        condition_meta["temporal_stride"] = int(latent_extractor.temporal_stride)
        condition_meta["predictor_latent_source"] = predictor_ckpt.get("latent_source", "mock")
    return meta, condition_meta


def main():
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.future_state_source == "wan_predictor" and (args.predictor is None or args.wan_ckpt_dir is None):
        if args.predictor_latent_source == "wan":
            raise ValueError("--predictor and --wan-ckpt-dir are required when --future-state-source=wan_predictor")
        if args.predictor is None:
            raise ValueError("--predictor is required when --future-state-source=wan_predictor")

    dataset = NpzPredictorDataset(args.episodes)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = None
    predictor_ckpt = None
    latent_extractor = None
    if args.future_state_source == "wan_predictor":
        predictor, predictor_ckpt = load_wan_state_predictor(args.predictor, args.device)
        latent_extractor = build_predictor_latent_extractor(args, predictor_ckpt)

    records = []
    limit = args.limit if args.limit > 0 else len(dataset)
    for index in range(min(limit, len(dataset))):
        sample = dataset[index]
        batch = collate_predictor_episodes([sample])
        sample_path = dataset.files[index]
        sample_id = sample_path.stem
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        if args.future_state_source == "wan_predictor":
            state_condition, condition_meta = build_predictor_state_condition(
                batch,
                predictor=predictor,
                predictor_ckpt=predictor_ckpt,
                latent_extractor=latent_extractor,
                device=args.device,
            )
        else:
            state_condition, condition_meta = build_ground_truth_state_condition(batch)

        image_path = sample_dir / "input_image.png"
        state_condition_path = sample_dir / "state_condition.npz"
        meta_path = sample_dir / "meta.json"
        prompt_path = sample_dir / "prompt.txt"

        context_frame0 = detach_to_cpu_numpy(batch["context_frames"][0, 0]).astype(np.float32)
        save_frame_png(context_frame0, image_path)
        np.savez_compressed(state_condition_path, **state_condition)

        meta = {
            "sample_id": sample_id,
            "episode_path": str(sample_path),
            "prompt": batch["prompts"][0],
            "future_state_source": args.future_state_source,
            "context_frames_shape": list(batch["context_frames"][0].shape),
            "context_states_shape": list(batch["context_states"][0].shape),
            "future_states_shape": list(batch["future_states"][0].shape),
            "camera_shape": list(batch["camera"][0].shape),
            **condition_meta,
        }
        if predictor_ckpt is not None:
            meta["predictor_checkpoint"] = str(args.predictor)
            meta["predictor_version"] = predictor_ckpt.get("predictor_version", "wan_state_v1")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        prompt_path.write_text(batch["prompts"][0], encoding="utf-8")

        record = {
            "sample_id": sample_id,
            "episode_path": str(sample_path),
            "image_path": str(image_path),
            "state_condition_path": str(state_condition_path),
            "meta_path": str(meta_path),
            "prompt": batch["prompts"][0],
            "future_state_source": args.future_state_source,
        }
        records.append(record)

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "num_samples": len(records),
        "episodes_root": str(Path(args.episodes)),
        "output_dir": str(output_dir),
        "future_state_source": args.future_state_source,
        "manifest_path": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
