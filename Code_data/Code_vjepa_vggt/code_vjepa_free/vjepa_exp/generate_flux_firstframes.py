from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import FluxKontextPipeline, FluxPipeline


"""
Examples

Generate first-frame images from `input_image_prompt` / legacy `input_image_caption` / `firstframe_caption`:
CUDA_VISIBLE_DEVICES=5 \
/home/gaoya/miniconda3/envs/flux/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_exp/generate_flux_firstframes.py \
    --manifest /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42/manifest.json \
    --output-root /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42_flux_firstframes \
    --cuda-visible-devices 5

Generate and also write a new manifest with updated `input_image` fields:
CUDA_VISIBLE_DEVICES=5 \
/home/gaoya/miniconda3/envs/flux/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_exp/generate_flux_firstframes.py \
    --manifest /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42/manifest.json \
    --output-root /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42_flux_firstframes \
    --json-root /data/gaoya/AAA_test_video/0626vjepa_free/testjsons \
    --write-manifest /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42_flux_firstframes/manifest_with_firstframes.json \
    --cuda-visible-devices 5

This script does not fix the random seed for image generation. It also writes one
metadata json per sample under `--json-root`. The per-sample json schema uses:
`input_image`, `input_image_prompt`, `input_image_model`,
`input_image_negative_prompt`, and `input_video_prompt`.
"""

DEFAULT_MODEL_ROOT = Path("/data/luoyang/ckpt/pretrained/models--black-forest-labs--FLUX.1-Kontext-dev")
DEFAULT_JSON_ROOT = Path("/data/gaoya/AAA_test_video/0626vjepa_free/testjsons")
DEFAULT_NEGATIVE_PROMPT = ""
WIDE_FRAME_SUFFIX = (
    " Wide shot from a medium distance. Show the full scene context and keep all key objects fully visible "
    "inside the frame. Leave some surrounding space around the subjects. Avoid close-up or zoomed-in composition."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate first-frame images from manifest input_image_prompt/firstframe_caption using local FLUX T2I."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, default=DEFAULT_JSON_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--write-manifest", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(manifest_path: Path, payload: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def build_pipe(model_root: Path):
    model_root = model_root.expanduser().resolve()
    model_name = model_root.name.lower()
    pipeline_cls = FluxKontextPipeline if "kontext" in model_name else FluxPipeline
    pipe = pipeline_cls.from_pretrained(str(model_root), torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    return pipe


def resolve_output_image_path(output_root: Path, case_id: str) -> Path:
    return output_root / case_id / "firstframe_flux.png"


def resolve_output_json_path(json_root: Path, case_id: str) -> Path:
    return json_root / f"{case_id}.json"


def get_case_prompt(case: dict) -> str:
    prompt = case.get("input_image_prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    prompt = case.get("input_image_caption")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    prompt = case.get("firstframe_caption")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    raise ValueError(
        f"missing input_image_prompt/input_image_caption/firstframe_caption for case {case.get('case_id')}"
    )


def adapt_prompt_for_wide_framing(prompt: str) -> str:
    text = str(prompt).strip()
    if not text:
        return text
    lowered = text.lower()
    if "avoid close-up" in lowered or "zoomed-in composition" in lowered:
        return text
    return text + WIDE_FRAME_SUFFIX


def get_video_prompt(case: dict) -> str:
    for key in ("input_video_prompt", "video_prompt", "prompt"):
        value = case.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    json_root = args.json_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    json_root.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    cases = manifest["cases"]
    if args.limit is not None:
        cases = cases[: args.limit]

    pipe = build_pipe(args.model_root)

    updated_cases = []
    for case in manifest["cases"]:
        case_copy = dict(case)
        updated_cases.append(case_copy)

    case_lookup = {str(case["case_id"]): case for case in updated_cases}

    for case in cases:
        case_id = str(case["case_id"])
        prompt = adapt_prompt_for_wide_framing(get_case_prompt(case))
        video_prompt = get_video_prompt(case)

        output_path = resolve_output_image_path(output_root, case_id)
        json_path = resolve_output_json_path(json_root, case_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.force:
            print(f"[skip] {case_id}")
            case_lookup[case_id]["input_image"] = str(output_path)
            case_lookup[case_id]["input_image_prompt"] = str(prompt)
            case_lookup[case_id]["input_image_model"] = str(args.model_root.expanduser().resolve())
            case_lookup[case_id]["input_image_negative_prompt"] = str(args.negative_prompt)
            case_lookup[case_id]["input_video_prompt"] = str(video_prompt)
            case_lookup[case_id]["firstframe_json"] = str(json_path)
            continue

        print(f"[case] {case_id} -> {output_path}")
        image = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            width=int(args.width),
            height=int(args.height),
            guidance_scale=float(args.guidance_scale),
            num_inference_steps=int(args.steps),
        ).images[0]
        image.save(output_path)
        case_lookup[case_id]["input_image"] = str(output_path)
        case_lookup[case_id]["input_image_prompt"] = str(prompt)
        case_lookup[case_id]["input_image_model"] = str(args.model_root.expanduser().resolve())
        case_lookup[case_id]["input_image_negative_prompt"] = str(args.negative_prompt)
        case_lookup[case_id]["input_video_prompt"] = str(video_prompt)
        case_lookup[case_id]["firstframe_json"] = str(json_path)

        sample_json = {
            "case_id": case_id,
            "source_video": str(case.get("source_video", "")),
            "input_video_prompt": str(video_prompt),
            "input_image_prompt": str(prompt),
            "input_image_model": str(args.model_root.expanduser().resolve()),
            "input_image_negative_prompt": str(args.negative_prompt),
            "input_image": str(output_path),
            "run_dir": str(case.get("run_dir", "")),
        }
        save_manifest(json_path, sample_json)
        print(f"[done] {case_id} -> {output_path}")

    if args.write_manifest is not None:
        new_manifest = dict(manifest)
        new_manifest["cases"] = updated_cases
        new_manifest["output_root"] = str(output_root)
        save_manifest(args.write_manifest.expanduser().resolve(), new_manifest)
        print(f"[manifest] {args.write_manifest.expanduser().resolve()}")


if __name__ == "__main__":
    main()
