'''
Run command example:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2,3 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/inspect_kubric_infer_forward_ctx_sweep.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name train_stage1b_kubric0708_step1000_ctx_sweep \
  --output-root /data/gaoya/agent-data/outputs/kubric_infer_forward_ctx_sweep \
  --inference-devices cuda:0,cuda:1 \
  --num-inference-steps 40 \
  --num-frames 49 \
  --ctx-values 1,4,8,12,16,20 \
  --limit 2
'''
from __future__ import annotations

import argparse
import gc
import html
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _read_cli_arg_value(argv: list[str], names: tuple[str, ...], default: str | None = None) -> str | None:
    for name in names:
        if name not in argv:
            continue
        index = argv.index(name)
        if index + 1 < len(argv):
            return argv[index + 1]
    return default


_DEFAULT_DIFFSYNTH_ROOT_STR = "/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main"
_SELECTED_DIFFSYNTH_ROOT = _read_cli_arg_value(
    sys.argv,
    ("--diffsynth-root", "--diffsynth_root"),
    os.environ.get("DIFFSYNTH_ROOT", _DEFAULT_DIFFSYNTH_ROOT_STR),
)
if _SELECTED_DIFFSYNTH_ROOT:
    os.environ["DIFFSYNTH_ROOT"] = _SELECTED_DIFFSYNTH_ROOT
    if _SELECTED_DIFFSYNTH_ROOT not in sys.path:
        sys.path.insert(0, _SELECTED_DIFFSYNTH_ROOT)

from code_vjepa_vggt import batch_infer_v_newtrain_from_jsonl as core
from code_vjepa_vggt.AAAinfer.utils.named_paths import resolve_output_root
from code_vjepa_vggt.train0705 import infer_stage1b_context_only_no_gt_box_v_newtrain0705 as infer0705
from code_vjepa_vggt.train0705 import normalize_ti2v_t2v_result_jsons as normjson
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    inspect_kubric_actual_train_forward_aux_overlay as actualinspect,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    inspect_kubric_train_forward_aux_overlay as inspectmod,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as batchmod,
)
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8
from diffsynth.utils.data import save_video


DEFAULT_OUTPUT_BASE = "/data/gaoya/agent-data/outputs/kubric_infer_forward_ctx_sweep"


def _parse_ctx_values(raw_value: str) -> list[int]:
    values = []
    for item in str(raw_value).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"ctx value must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("ctx-values is empty")
    deduped: list[int] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _sanitize_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in str(value))
    return cleaned.strip("._") or "sample"


def _resolve_existing_path(value: object, *, base_dir: Path | None = None) -> Path | None:
    resolved = normjson.resolve_existing_path(value)
    if resolved is not None:
        return resolved
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve() if candidate.exists() else None
    if base_dir is None:
        return None
    joined = (base_dir / candidate).resolve()
    return joined if joined.exists() else None


def _resolve_input_image(payload: dict[str, object], json_path: Path) -> Path | None:
    base_dir = json_path.parent
    input_image = _resolve_existing_path(payload.get("input_image"), base_dir=base_dir)
    if input_image is not None:
        return input_image
    return _resolve_existing_path(payload.get("first_frame_path"), base_dir=base_dir)


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "cuda out of memory" in message or "out of memory" in message


def _render_prompt_text_image(prompt: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    wrapped = textwrap.wrap(prompt or "(empty prompt)", width=72) or ["(empty prompt)"]
    line_height = 18
    width = 1200
    height = max(160, 40 + line_height * len(wrapped))
    canvas = Image.new("RGB", (width, height), color=(250, 247, 240))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(210, 201, 184), width=2)
    draw.text((24, 18), "Prompt", fill=(41, 50, 65), font=font)
    y = 48
    for line in wrapped:
        draw.text((24, y), line, fill=(31, 31, 31), font=font)
        y += line_height
    canvas.save(output_path, format="PNG")
    return output_path


def _save_input_image_copy(input_image_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(input_image_path).convert("RGB")
    image.save(output_path, format="PNG")
    return output_path


def _make_case_sample(
    *,
    context_video_single: torch.Tensor,
    caption: str,
    source_video_path: Path,
    frame_indices: np.ndarray,
    sample_key: str,
    input_json_path: Path,
) -> dict[str, Any]:
    frame_indices_t = torch.as_tensor(frame_indices, dtype=torch.long)
    return {
        "context_video": context_video_single,
        "num_context_frames": int(context_video_single.shape[1]),
        "caption": str(caption),
        "video_path": str(source_video_path),
        "context_frame_indices": frame_indices_t,
        "metadata": {
            "sample_key": str(sample_key),
            "input_json": str(input_json_path),
            "source_video_path": str(source_video_path),
        },
    }


def _run_generation(
    *,
    model,
    prompt: str,
    context_pil: list[Image.Image],
    object_context: torch.Tensor | None,
    num_frames: int,
    sampling_steps: int,
    cfg_scale: float,
    seed: int,
    height: int,
    width: int,
) -> tuple[Any, dict[str, object]]:
    pipe = model.pipe
    final_object_context = object_context
    ablation_debug: dict[str, object] = {
        "mode": str(getattr(model, "_object_context_ablation_mode", "none")),
        "applied": False,
    }
    if bool(getattr(model, "enable_object_branch", False)):
        final_object_context, ablation_debug = infer0705._apply_object_context_ablation(
            object_context,
            mode=str(getattr(model, "_object_context_ablation_mode", "none")),
            random_seed=getattr(model, "_object_context_random_seed", None),
            random_scale=float(getattr(model, "_object_context_random_scale", 1.0)),
        )

    pipe_kwargs = dict(
        prompt=str(prompt),
        negative_prompt="",
        context_video=context_pil,
        seed=int(seed),
        tiled=True,
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        num_inference_steps=int(sampling_steps),
        cfg_scale=float(cfg_scale),
    )
    if bool(getattr(model, "enable_object_branch", False)):
        pipe_kwargs["object_context"] = final_object_context

    pipe.dit.eval()
    with torch.no_grad():
        video = pipe(**pipe_kwargs)
    return video, ablation_debug


def _write_case_page(case_dir: Path, result: dict[str, Any]) -> None:
    input_image_block = ""
    if result.get("input_image_png"):
        input_image_block = f"""
      <figure>
        <img src="{html.escape(str(result['input_image_png']))}" />
        <figcaption>Optional input image</figcaption>
      </figure>
"""

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Kubric Inference Forward Ctx Sweep</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f2eee5;
      --panel: #fffdf9;
      --line: #d8d0c5;
      --text: #1f1f1f;
      --muted: #615a52;
      --link: #0d5ea8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      font-family: sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top right, #efe4ce 0, transparent 25%),
        linear-gradient(180deg, #f6f1e8 0%, #f1ece4 100%);
    }}
    .page {{ max-width: 1850px; margin: 0 auto; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(320px, 1fr)); gap: 16px; }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      background: #fff;
    }}
    img, video {{ display: block; width: 100%; background: #000; }}
    figcaption {{ padding: 10px 12px; font-size: 13px; color: var(--muted); border-top: 1px solid var(--line); }}
    pre {{
      margin: 16px 0 0;
      padding: 14px;
      overflow-x: auto;
      border-radius: 10px;
      background: #faf7f0;
      border: 1px solid var(--line);
      white-space: pre-wrap;
    }}
    a {{ color: var(--link); }}
    @media (max-width: 1300px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Kubric Inference Forward Context Sweep</h1>
    <p><b>Sample key:</b> {html.escape(str(result["sample_key"]))}</p>
    <p><b>Input json:</b> {html.escape(str(result["input_json"]))}</p>
    <p><b>Source video:</b> {html.escape(str(result["source_video"]))}</p>
    <p><b>Requested / effective context frames:</b> {int(result["requested_context_frames"])} / {int(result["effective_context_frames"])}</p>
    <p><b>Context frame indices:</b> {html.escape(str(result["frame_indices"]))}</p>
    <p><b>Prompt:</b> {html.escape(str(result["input_caption"]))}</p>
    <div class="grid">
      <figure>
        <video controls preload="none" playsinline src="{html.escape(str(result['generated_video_mp4']))}"></video>
        <figcaption>Generated inference video</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{html.escape(str(result['context_video_mp4']))}"></video>
        <figcaption>Actual context video sent to pipe()</figcaption>
      </figure>
      <figure>
        <img src="{html.escape(str(result['context_sheet_jpg']))}" />
        <figcaption>Context contact sheet</figcaption>
      </figure>
      <figure>
        <img src="{html.escape(str(result['prompt_text_png']))}" />
        <figcaption>Prompt text</figcaption>
      </figure>
      <figure>
        <img src="{html.escape(str(result['prompt_preview_png']))}" />
        <figcaption>Prompt frame plus prompt boxes and sampled query points</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{html.escape(str(result['input_overlay_video']))}"></video>
        <figcaption>Input overlay before pipe(): query points and CoTracker tracks</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{html.escape(str(result['box_overlay_video']))}"></video>
        <figcaption>Aux predicted boxes vs reference boxes</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{html.escape(str(result['track_overlay_video']))}"></video>
        <figcaption>Aux predicted track summaries vs reference track summaries</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{html.escape(str(result['source_full_video_mp4']))}"></video>
        <figcaption>Original source full video</figcaption>
      </figure>
{input_image_block}    </div>
    <h2>Metrics</h2>
    <pre>{html.escape(json.dumps(result["metrics"], ensure_ascii=False, indent=2))}</pre>
    <h2>Metadata</h2>
    <pre>{html.escape(json.dumps(result["metadata"], ensure_ascii=False, indent=2))}</pre>
  </div>
</body>
</html>
"""
    (case_dir / "index.html").write_text(html_text, encoding="utf-8")


def _write_summary_page(output_dir: Path, results: list[dict[str, Any]]) -> None:
    cards: list[str] = []
    for result in results:
        rel_dir = result["relative_dir"]
        cards.append(
            f"""
<section class="case-card">
  <div class="case-header">
    <div>
      <h2>{html.escape(str(result["sample_key"]))} / ctx{int(result["requested_context_frames"]):02d}</h2>
      <p class="meta"><b>effective:</b> {int(result["effective_context_frames"])}</p>
      <p class="meta"><b>frame_indices:</b> {html.escape(str(result["frame_indices"]))}</p>
      <p class="caption">{html.escape(str(result["input_caption"]))}</p>
    </div>
    <div class="actions">
      <a href="{html.escape(rel_dir)}/index.html">open case report</a>
    </div>
  </div>
  <div class="media-grid">
    <figure>
      <video controls preload="none" playsinline src="{html.escape(rel_dir)}/{html.escape(str(result['generated_video_mp4']))}"></video>
      <figcaption>Generated inference video</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(rel_dir)}/{html.escape(str(result['context_video_mp4']))}"></video>
      <figcaption>Context video</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(rel_dir)}/{html.escape(str(result['input_overlay_video']))}"></video>
      <figcaption>Input overlay</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(rel_dir)}/{html.escape(str(result['box_overlay_video']))}"></video>
      <figcaption>Aux box overlay</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(rel_dir)}/{html.escape(str(result['track_overlay_video']))}"></video>
      <figcaption>Aux track overlay</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(rel_dir)}/{html.escape(str(result['prompt_text_png']))}" />
      <figcaption>Prompt text</figcaption>
    </figure>
  </div>
</section>
"""
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Kubric Inference Forward Context Sweep</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f2eee5;
      --panel: #fffdf9;
      --line: #d8d0c5;
      --text: #1f1f1f;
      --muted: #615a52;
      --link: #0d5ea8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      font-family: sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top right, #efe4ce 0, transparent 25%),
        linear-gradient(180deg, #f6f1e8 0%, #f1ece4 100%);
    }}
    .page {{ max-width: 1850px; margin: 0 auto; }}
    .case-list {{ display: grid; gap: 20px; }}
    .case-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
    }}
    .case-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 16px;
    }}
    .case-header h2 {{ margin: 0 0 8px; font-size: 22px; line-height: 1.25; }}
    .meta {{ margin: 4px 0; color: var(--muted); word-break: break-word; }}
    .caption {{ margin: 12px 0 0; line-height: 1.6; }}
    .actions a {{ color: var(--link); text-decoration: none; font-weight: 600; }}
    .actions a:hover {{ text-decoration: underline; }}
    .media-grid {{ display: grid; grid-template-columns: repeat(3, minmax(320px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: #ffffff; }}
    img, video {{ display: block; width: 100%; background: #000; }}
    figcaption {{ padding: 10px 12px; font-size: 13px; color: var(--muted); border-top: 1px solid var(--line); }}
    @media (max-width: 1300px) {{ .media-grid {{ grid-template-columns: 1fr; }} .case-header {{ flex-direction: column; }} }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Kubric Inference Forward Context Sweep</h1>
    <p>Total reports: {len(results)}.</p>
    <div class="case-list">
      {''.join(cards)}
    </div>
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep multiple context lengths for the Kubric stage1b context-only no-GT-box "
            "inference path, and export forward-debug overlays plus the actual generated video."
        )
    )
    parser.add_argument("--weights-root", type=Path, required=True, help="step-* dir containing checkpoint.safetensors")
    parser.add_argument("--input-json-list-path", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--ctx-values", type=str, default="1,4,8,12,16,20")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--aux-device", type=str, default=None)
    parser.add_argument(
        "--inference-devices",
        type=str,
        default=None,
        help="Optional two-device layout like cuda:0,cuda:1. First is main inference device, second is aux device.",
    )
    parser.add_argument("--wan-root", type=Path, default=batchmod.DEFAULT_WAN_ROOT)
    parser.add_argument("--diffsynth-root", type=Path, default=batchmod.DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument("--lora-checkpoint", type=Path, default=batchmod.DEFAULT_BASE_LORA)
    parser.add_argument("--stage1a-init-from", type=Path, default=batchmod.DEFAULT_STAGE1A)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--context-frames", type=int, default=20)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--disable-object-branch", action="store_true")
    parser.add_argument("--object-num-queries", type=int, default=8)
    parser.add_argument("--aux-max-objects", type=int, default=4)
    parser.add_argument("--object-pooler-latent-dim", type=int, default=16)
    parser.add_argument("--cond-proj-dim", type=int, default=4096)
    parser.add_argument("--jepa-window-radius", type=int, default=1)
    parser.add_argument("--latent-window-radius", type=int, default=1)
    parser.add_argument("--object-gate-init", type=float, default=0.1)
    parser.add_argument("--jepa-ckpt-path", default="/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth")
    parser.add_argument("--jepa-input-size", type=int, default=384)
    parser.add_argument("--jepa-patch-size", type=int, default=16)
    parser.add_argument("--jepa-tubelet-size", type=int, default=2)
    parser.add_argument("--cotracker-checkpoint", default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")
    parser.add_argument("--cotracker-input-h", type=int, default=384)
    parser.add_argument("--cotracker-input-w", type=int, default=512)
    parser.add_argument("--cotracker-window-len", type=int, default=60)
    parser.add_argument("--vggt-model-path", default="/data/gaoya/ckpt/facebook-VGGT-1B")
    parser.add_argument("--vggt-input-h", type=int, default=420)
    parser.add_argument("--vggt-input-w", type=int, default=728)
    parser.add_argument("--vggt-cache-root", default=None)
    parser.add_argument("--grounding-device", default=None)
    parser.add_argument("--sam2-segment-len", type=int, default=8)
    parser.add_argument("--grounding-proposal-source", default="gdino_only")
    parser.add_argument("--grounding-motion-score-ratio", type=float, default=0.15)
    parser.add_argument(
        "--grounding-text-prompt",
        default="box . cube . block . cylinder . capsule . sphere . ball .",
    )
    parser.add_argument("--grounding-extra-prompt-terms", default="")
    parser.add_argument("--grounding-disable-caption-terms", action="store_true", default=True)
    parser.add_argument("--grounding-gdino-box-threshold", type=float, default=0.20)
    parser.add_argument("--grounding-gdino-text-threshold", type=float, default=0.15)
    parser.add_argument("--grounding-prompt-frame-mode", default="first")
    parser.add_argument("--grounding-track-dedupe-iou-threshold", type=float, default=0.75)
    parser.add_argument("--grounding-container-suppress-ratio-threshold", type=float, default=0.95)
    parser.add_argument("--grounding-container-suppress-min-contained", type=int, default=2)
    parser.add_argument("--grounding-container-suppress-min-area-ratio", type=float, default=1.5)
    parser.add_argument("--grounding-container-suppress-small-iou-threshold", type=float, default=0.7)
    parser.add_argument("--initialize-model-on-cpu", action="store_true")
    infer0705.add_vjepa_cli_args(parser)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--object-context-ablation",
        choices=["none", "zero", "random"],
        default="none",
        help="Replace the final object_context fed into Wan DiT for ablation.",
    )
    parser.add_argument(
        "--object-context-random-seed",
        type=int,
        default=None,
        help="Optional RNG seed used when --object-context-ablation=random.",
    )
    parser.add_argument(
        "--object-context-random-scale",
        type=float,
        default=1.0,
        help="Std multiplier used when --object-context-ablation=random.",
    )
    return parser.parse_args()


def main() -> None:
    batchmod._install_kubric_runtime_hooks()
    cli_args = parse_args()
    if bool(cli_args.disable_object_branch):
        raise ValueError("inspect_kubric_infer_forward_ctx_sweep.py expects the object branch to be enabled")

    infer0705.apply_vjepa_preset_if_requested(cli_args)
    weights_root = cli_args.weights_root.expanduser().resolve()
    input_json_list_path = cli_args.input_json_list_path.expanduser().resolve()
    model_name = str(cli_args.model_name).strip()
    ctx_values = _parse_ctx_values(cli_args.ctx_values)
    output_root = resolve_output_root(
        explicit_output_root=cli_args.output_root,
        base_output_root=DEFAULT_OUTPUT_BASE,
        model_name=model_name,
    )

    if not weights_root.exists():
        raise FileNotFoundError(f"weights-root not found: {weights_root}")

    cli_args.device, cli_args.aux_device = batchmod._resolve_runtime_devices(cli_args)
    torch.manual_seed(int(cli_args.seed))
    np.random.seed(int(cli_args.seed))

    json_paths = core._read_list_file(input_json_list_path)
    if cli_args.limit is not None:
        json_paths = json_paths[: max(0, int(cli_args.limit))]

    output_root.mkdir(parents=True, exist_ok=True)
    step_output_dir = output_root / weights_root.name
    step_output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "input_json_list_path": str(input_json_list_path),
        "weights_root": str(weights_root),
        "num_items": len(json_paths),
        "ctx_values": ctx_values,
        "num_inference_steps": int(cli_args.num_inference_steps),
        "cfg_scale": float(cli_args.cfg_scale),
        "seed": int(cli_args.seed),
        "height": int(cli_args.height),
        "width": int(cli_args.width),
        "num_frames": int(cli_args.num_frames),
        "sampling_mode": str(cli_args.sampling_mode),
        "device": str(cli_args.device),
        "aux_device": cli_args.aux_device,
        "inference_devices": cli_args.inference_devices,
        "object_context_ablation": {
            "mode": str(cli_args.object_context_ablation),
            "random_seed": cli_args.object_context_random_seed,
            "random_scale": float(cli_args.object_context_random_scale),
        },
        "vjepa": infer0705.summarize_vjepa_args(cli_args),
    }
    (output_root / "batch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    runtime_args = batchmod._build_runtime_args(cli_args, weights_root, step_output_dir)
    model, _, _ = infer0705._build_runtime_model(runtime_args)
    model.to(torch.device(cli_args.device))
    model.eval()
    model.pipe.dit.eval()
    model._object_context_ablation_mode = str(cli_args.object_context_ablation)
    model._object_context_random_seed = cli_args.object_context_random_seed
    model._object_context_random_scale = float(cli_args.object_context_random_scale)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    method_name = batchmod._build_method_name_from_checkpoint_dir(weights_root)

    for input_json_path in json_paths:
        payload = core._load_input_json(input_json_path)
        source_video = batchmod._resolve_source_video(payload, input_json_path)
        input_caption = core._ensure_str_field(payload, "input_caption", input_json_path)
        input_image_path = _resolve_input_image(payload, input_json_path)
        sample_stem = _sanitize_name(input_json_path.stem)
        sample_root = step_output_dir / sample_stem
        sample_root.mkdir(parents=True, exist_ok=True)
        source_video_path = Path(source_video).expanduser().resolve()

        for requested_ctx in ctx_values:
            case_dir = sample_root / f"ctx{int(requested_ctx):02d}"
            result_json_path = case_dir / "result.json"

            if result_json_path.exists() and not (cli_args.force or cli_args.overwrite):
                with result_json_path.open("r", encoding="utf-8") as handle:
                    cached = json.load(handle)
                cached["relative_dir"] = str(case_dir.relative_to(step_output_dir))
                results.append(cached)
                print(f"[skip] {sample_stem} ctx{int(requested_ctx):02d}")
                continue

            case_dir.mkdir(parents=True, exist_ok=True)
            print(f"[run] {sample_stem} ctx{int(requested_ctx):02d}")
            try:
                frames, frame_indices = batchmod._load_context_video_for_mode(
                    video_path=source_video_path,
                    target_context_frames=int(requested_ctx),
                    sampling_mode=str(cli_args.sampling_mode),
                )
                effective_context_frames = int(frames.shape[0])
                context_video_single = preprocess_video_rgb_uint8(
                    frames,
                    (int(cli_args.height), int(cli_args.width)),
                )
                context_pil = infer0705._tensor_video_to_pil_list(context_video_single)
                sample = _make_case_sample(
                    context_video_single=context_video_single,
                    caption=input_caption,
                    source_video_path=source_video_path,
                    frame_indices=frame_indices,
                    sample_key=sample_stem,
                    input_json_path=input_json_path,
                )

                with torch.no_grad():
                    debug = actualinspect._run_forward_debug(model, sample)

                video, ablation_debug = _run_generation(
                    model=model,
                    prompt=input_caption,
                    context_pil=context_pil,
                    object_context=debug["object_context"],
                    num_frames=int(cli_args.num_frames),
                    sampling_steps=int(cli_args.num_inference_steps),
                    cfg_scale=float(cli_args.cfg_scale),
                    seed=int(cli_args.seed),
                    height=int(cli_args.height),
                    width=int(cli_args.width),
                )

                image_hw = (int(context_video_single.shape[-2]), int(context_video_single.shape[-1]))
                grounding_sample = model.viewer_grounding.build_sample(
                    frames_tchw_01=((context_video_single.permute(1, 0, 2, 3).float() + 1.0) / 2.0).cpu().numpy(),
                    caption=str(input_caption),
                    image_hw=image_hw,
                )

                context_video_browser = inspectmod._write_tensor_video(
                    case_dir / "context_video.mp4",
                    context_video_single,
                    fps=int(cli_args.fps),
                )
                source_full_video_browser = inspectmod._export_browser_video(
                    source_video_path,
                    case_dir / "source_full_video.browser.mp4",
                )
                context_sheet_path = case_dir / "context_sheet.jpg"
                batchmod._save_context_contact_sheet(context_pil=context_pil, output_path=context_sheet_path)

                prompt_text_path = _render_prompt_text_image(
                    input_caption,
                    case_dir / "prompt_text.png",
                )
                prompt_txt_path = case_dir / "prompt.txt"
                prompt_txt_path.write_text(input_caption + "\n", encoding="utf-8")

                valid_queries = inspectmod._valid_query_count(debug["object_valid_mask"], model.object_num_queries)
                valid_queries = max(valid_queries, 0)
                query_points_prior = debug["query_points_prior"][0].detach().float().cpu().numpy()
                cotracker_tracks = debug["cotracker_out"].tracks[0].detach().float().cpu().numpy()
                cotracker_visibility = debug["cotracker_out"].visibility[0].detach().float().cpu().numpy()
                valid_queries_px = query_points_prior[:valid_queries]
                valid_tracks = cotracker_tracks[:, :valid_queries]
                valid_visibility = cotracker_visibility[:, :valid_queries]
                query_owner = [
                    obj_idx
                    for obj_idx in range(int((debug["object_valid_mask"][0] > 0.5).sum().item()))
                    for _ in range(int(model.object_num_queries))
                ]

                prompt_preview = inspectmod._render_prompt_preview(
                    context_video=context_video_single,
                    grounding_sample=grounding_sample,
                    valid_queries_px=valid_queries_px,
                    query_owner=query_owner,
                )
                prompt_preview_path = case_dir / "prompt_preview.png"
                inspectmod._write_rgb_png(prompt_preview_path, prompt_preview)

                input_overlay_video = inspectmod.render_track_overlay(
                    context_video=context_video_single,
                    object_tracks=getattr(grounding_sample, "object_tracks", []),
                    prompt_frame_idx=int(getattr(grounding_sample, "prompt_frame_idx", 0)),
                    query_points_px_k2=valid_queries_px.astype("float32"),
                    query_owner=query_owner,
                    tracks_tk2=valid_tracks.astype("float32"),
                    visibility_tk=valid_visibility.astype("float32"),
                    color_rgb=inspectmod.INPUT_TRACK_COLOR,
                    prefix="trk",
                )
                input_overlay_raw = case_dir / "input_prepipe_overlay.mp4"
                inspectmod.write_mp4(input_overlay_raw, input_overlay_video, fps=int(cli_args.fps))
                input_overlay_browser = inspectmod._ensure_browser_video(input_overlay_raw)

                ref_box_xyxy = debug["object_out"].active_box_xyxy[0].detach().float().cpu().numpy()
                pred_box_xyxy = debug["object_aux_out"].pred_box_xyxy[0].detach().float().cpu().numpy()
                ref_track_summary = (
                    debug["object_out"].active_track_summary[0, ..., :4].detach().float().cpu().numpy()
                )
                pred_track_summary = debug["object_aux_out"].pred_track_summary[0].detach().float().cpu().numpy()
                latent_valid_mask = debug["latent_valid_mask"]

                box_overlay_video = inspectmod._render_ref_pred_box_overlay(
                    context_video=context_video_single,
                    ref_box_xyxy=ref_box_xyxy,
                    pred_box_xyxy=pred_box_xyxy,
                    valid_mask=latent_valid_mask,
                    image_hw=image_hw,
                )
                box_overlay_raw = case_dir / "aux_pred_box_overlay.mp4"
                inspectmod.write_mp4(box_overlay_raw, box_overlay_video, fps=int(cli_args.fps))
                box_overlay_browser = inspectmod._ensure_browser_video(box_overlay_raw)

                track_overlay_video = inspectmod._render_ref_pred_track_overlay(
                    context_video=context_video_single,
                    ref_track_summary=ref_track_summary,
                    pred_track_summary=pred_track_summary,
                    valid_mask=latent_valid_mask,
                    image_hw=image_hw,
                )
                track_overlay_raw = case_dir / "aux_pred_track_overlay.mp4"
                inspectmod.write_mp4(track_overlay_raw, track_overlay_video, fps=int(cli_args.fps))
                track_overlay_browser = inspectmod._ensure_browser_video(track_overlay_raw)

                generated_video_path = case_dir / "generated_video.mp4"
                save_video(video, str(generated_video_path), fps=int(cli_args.fps), quality=int(cli_args.quality))
                generated_video_browser = inspectmod._ensure_browser_video(generated_video_path)

                input_image_png = None
                if input_image_path is not None:
                    input_image_png = _save_input_image_copy(
                        input_image_path,
                        case_dir / "input_image.png",
                    )

                result = {
                    "method": method_name,
                    "sample_key": sample_stem,
                    "input_json": str(input_json_path),
                    "source_video": str(source_video_path),
                    "input_caption": str(input_caption),
                    "input_image": None if input_image_path is None else str(input_image_path),
                    "requested_context_frames": int(requested_ctx),
                    "effective_context_frames": int(effective_context_frames),
                    "frame_indices": frame_indices.tolist(),
                    "seed": int(cli_args.seed),
                    "step": int(cli_args.num_inference_steps),
                    "guidance": float(cli_args.cfg_scale),
                    "ckpt": str(weights_root),
                    "sampling_mode": str(cli_args.sampling_mode),
                    "generated_video_mp4": generated_video_browser.name,
                    "context_video_mp4": context_video_browser.name,
                    "source_full_video_mp4": source_full_video_browser.name,
                    "context_sheet_jpg": context_sheet_path.name,
                    "prompt_text_png": prompt_text_path.name,
                    "prompt_preview_png": prompt_preview_path.name,
                    "prompt_txt": prompt_txt_path.name,
                    "input_overlay_video": input_overlay_browser.name,
                    "box_overlay_video": box_overlay_browser.name,
                    "track_overlay_video": track_overlay_browser.name,
                    "input_image_png": None if input_image_png is None else input_image_png.name,
                    "metrics": debug["metrics"],
                    "metadata": {
                        "model_device": str(model.pipe.device),
                        "aux_device": cli_args.aux_device,
                        "object_context_ablation": ablation_debug,
                        "model_args": {
                            "height": int(cli_args.height),
                            "width": int(cli_args.width),
                            "num_frames": int(cli_args.num_frames),
                            "enable_object_branch": bool(getattr(model, "enable_object_branch", False)),
                            "lora_checkpoint": str(cli_args.lora_checkpoint),
                            "stage1a_init_from": str(cli_args.stage1a_init_from),
                        },
                        "vjepa": infer0705.summarize_vjepa_args(cli_args),
                    },
                }
                result["relative_dir"] = str(case_dir.relative_to(step_output_dir))
                result_json_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                _write_case_page(case_dir, result)
                results.append(result)

                del video
                del debug
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                error_payload = {
                    "sample_key": sample_stem,
                    "requested_context_frames": int(requested_ctx),
                    "error": str(exc),
                }
                failures.append(error_payload)
                (case_dir / "error.json").write_text(
                    json.dumps(error_payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"[error] {sample_stem} ctx{int(requested_ctx):02d}: {exc}")
                if _is_cuda_oom(exc):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise SystemExit(86) from exc

    summary_payload = {
        "weights_root": str(weights_root),
        "output_root": str(output_root),
        "step": weights_root.name,
        "num_requested_cases": len(json_paths) * len(ctx_values),
        "num_success": len(results),
        "num_failed": len(failures),
        "ctx_values": ctx_values,
        "results": results,
        "failures": failures,
    }
    (step_output_dir / "results.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "weights_root": str(weights_root),
                "output_root": str(output_root),
                "step": weights_root.name,
                "num_requested_cases": len(json_paths) * len(ctx_values),
                "num_success": len(results),
                "num_failed": len(failures),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_summary_page(step_output_dir, results)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
