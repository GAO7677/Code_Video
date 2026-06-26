from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from PIL import Image


WAN_REPO = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main_copy")


def add_repo_to_path() -> None:
    repo = str(WAN_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def patch_wan_from_pretrained() -> None:
    from diffusers.models.modeling_utils import ModelMixin
    from wann.modules.model import WanModel

    def _wan_from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("low_cpu_mem_usage", False)
        return ModelMixin.from_pretrained.__func__(cls, *args, **kwargs)

    WanModel.from_pretrained = classmethod(_wan_from_pretrained)


def load_pipeline(manifest: dict, device_id: int):
    add_repo_to_path()
    patch_wan_from_pretrained()

    from wann.configs import WAN_CONFIGS
    from wann.textimage2video import WanTI2V

    task = manifest["task"]
    cfg = WAN_CONFIGS[task]
    wan_args = manifest["wan_args"]
    pipeline = WanTI2V(
        config=cfg,
        checkpoint_dir=manifest["ckpt_dir"],
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=wan_args["t5_cpu"],
        convert_model_dtype=wan_args["convert_model_dtype"],
    )
    return pipeline, cfg


def generate_case(pipeline, cfg, case: dict, manifest: dict, overwrite: bool) -> Path:
    add_repo_to_path()
    from wann.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS
    from wann.utils.utils import save_video

    video_path = Path(case["video_path"]).expanduser().resolve()
    if video_path.exists() and not overwrite:
        logging.info("Skip existing case %s -> %s", case["case_id"], video_path)
        return video_path

    video_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = case["prompt"]
    image = Image.open(case["image_path"]).convert("RGB")
    wan_args = manifest["wan_args"]
    size_key = wan_args["size"]

    logging.info("Generating %s", case["case_id"])
    video = pipeline.generate(
        prompt,
        img=image,
        size=SIZE_CONFIGS[size_key],
        max_area=MAX_AREA_CONFIGS[size_key],
        frame_num=wan_args["frame_num"],
        shift=wan_args["sample_shift"],
        sample_solver="unipc",
        sampling_steps=wan_args["sample_steps"],
        guide_scale=wan_args["sample_guide_scale"],
        seed=case["seed"],
        offload_model=wan_args["offload_model"],
    )
    save_video(
        tensor=video[None],
        save_file=str(video_path),
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    return video_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Wan TI2V baseline videos from a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"]
    if args.limit is not None:
        cases = cases[: args.limit]

    pipeline, cfg = load_pipeline(manifest=manifest, device_id=0)
    written = []
    for case in cases:
        video_path = generate_case(
            pipeline=pipeline,
            cfg=cfg,
            case=case,
            manifest=manifest,
            overwrite=args.overwrite,
        )
        written.append(str(video_path))

    print(json.dumps({"num_cases": len(written), "videos": written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
