from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image

from .scorers import GeometryProxyScorer
from .schemas import GeometryConfig
from .video_utils import ensure_dir, pil_list_to_numpy, save_video_frames, write_json


DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/DiffSynth-Studio-main")
if str(DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_ROOT))


DEFAULT_CASES = [
    {
        "case_id": "curved_car_turn",
        "category": "Curved_Motion",
        "filename": "car-turn.mp4",
        "prompt": "A low-angle tracking shot follows a sports car drifting through a sharp corner on a professional race track, with curb lines and tire marks shifting dynamically in the background.",
        "target_object": "sports car",
    },
    {
        "case_id": "dynamic_bus",
        "category": "Dynamic_Tracking",
        "filename": "bus.mp4",
        "prompt": "A moving tracking shot follows a city bus driving forward through an urban street, with strong parallax from roadside structures and background traffic.",
        "target_object": "bus",
    },
]


@dataclass
class PDICase:
    case_id: str
    category: str
    filename: str
    prompt: str
    target_object: str


@dataclass
class EvalResult:
    case_id: str
    provider: str
    prompt: str
    target_object: str
    gt_video_path: Path
    first_frame_path: Path
    video_path: Path
    geometry_score: float
    geometry_details: dict[str, Any]


def default_cases() -> list[PDICase]:
    return [PDICase(**item) for item in DEFAULT_CASES]


def _run_command(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def ensure_gt_video(case: PDICase, gt_root: Path) -> Path:
    category_dir = ensure_dir(gt_root / case.category)
    output_path = category_dir / case.filename
    if output_path.is_file():
        return output_path
    url = f"https://huggingface.co/datasets/AnteaWu/PDI-Dataset/resolve/main/GT/{case.category}/{case.filename}"
    _run_command(["wget", "-O", str(output_path), url])
    return output_path


def extract_first_frame_image(video_path: Path, image_path: Path) -> Path:
    if image_path.is_file():
        return image_path
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read first frame from {video_path}")
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ensure_dir(image_path.parent)
    image.save(image_path)
    return image_path


def _build_model_configs_for_wan(wan_root: Path):
    from diffsynth import ModelConfig

    dit_shards = [
        wan_root / "diffusion_pytorch_model-00001-of-00003.safetensors",
        wan_root / "diffusion_pytorch_model-00002-of-00003.safetensors",
        wan_root / "diffusion_pytorch_model-00003-of-00003.safetensors",
    ]
    return [
        ModelConfig(path=[str(path) for path in dit_shards]),
        ModelConfig(path=str(wan_root / "models_t5_umt5-xxl-enc-bf16.pth")),
        ModelConfig(path=str(wan_root / "Wan2.2_VAE.pth")),
    ]


def _find_tokenizer_path(root: Path) -> Path:
    for path in [root / "google" / "umt5-xxl", root / "umt5-xxl"]:
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Tokenizer directory not found under {root}")


class WanTI2VRunner:
    def __init__(self, model_root: Path, device: str) -> None:
        from diffsynth import ModelConfig
        from diffsynth.pipelines.wan_video import WanVideoPipeline

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=_build_model_configs_for_wan(model_root),
            tokenizer_config=ModelConfig(path=str(_find_tokenizer_path(model_root))),
        )

    def generate(
        self,
        *,
        first_frame_path: Path,
        prompt: str,
        output_path: Path,
        seed: int,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        num_inference_steps: int,
        cfg_scale: float,
        quality: int,
    ) -> Path:
        image = Image.open(first_frame_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        with torch.no_grad():
            video = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                input_image=image,
                height=height,
                width=width,
                num_frames=num_frames,
                cfg_scale=cfg_scale,
                num_inference_steps=num_inference_steps,
            )
        save_video_frames(output_path, pil_list_to_numpy(video[:num_frames]), fps=fps, quality=quality)
        return output_path


class VaceTI2VRunner:
    def __init__(self, model_root: Path, device: str) -> None:
        from diffsynth import ModelConfig
        from diffsynth.pipelines.wan_video import WanVideoPipeline

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[
                ModelConfig(path=str(model_root / "diffusion_pytorch_model.safetensors")),
                ModelConfig(path=str(model_root / "models_t5_umt5-xxl-enc-bf16.pth")),
                ModelConfig(path=str(model_root / "Wan2.1_VAE.pth")),
            ],
            tokenizer_config=ModelConfig(
                path=str(model_root / "google" / "umt5-xxl"),
                skip_download=True,
            ),
            redirect_common_files=False,
        )

    def generate(
        self,
        *,
        first_frame_path: Path,
        prompt: str,
        output_path: Path,
        seed: int,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        num_inference_steps: int,
        cfg_scale: float,
        quality: int,
    ) -> Path:
        first_frame = Image.open(first_frame_path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        placeholder = Image.new("RGB", (width, height), (128, 128, 128))
        mask_black = Image.new("RGB", (width, height), (0, 0, 0))
        mask_white = Image.new("RGB", (width, height), (255, 255, 255))
        video_input = [first_frame] + [placeholder.copy() for _ in range(max(num_frames - 1, 0))]
        video_mask = [mask_black] + [mask_white.copy() for _ in range(max(num_frames - 1, 0))]
        with torch.no_grad():
            video = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                vace_video=video_input,
                vace_video_mask=video_mask,
                height=height,
                width=width,
                num_frames=num_frames,
                seed=seed,
                cfg_scale=cfg_scale,
                num_inference_steps=num_inference_steps,
                tiled=True,
            )
        save_video_frames(output_path, pil_list_to_numpy(video[:num_frames]), fps=fps, quality=quality)
        return output_path


def _render_html(results: list[EvalResult], output_path: Path) -> None:
    grouped: dict[str, list[EvalResult]] = {}
    for result in results:
        grouped.setdefault(result.case_id, []).append(result)

    cards = []
    for case_id, items in grouped.items():
        provider_order = {"gt": 0, "wan": 1, "vace": 2}
        items = sorted(items, key=lambda item: provider_order.get(item.provider, 99))
        header = items[0]
        provider_blocks = []
        for item in items:
            provider_blocks.append(
                f"""
                <div class="provider-card">
                  <h3>{item.provider}</h3>
                  <video controls preload="metadata" src="{os.path.relpath(item.video_path, output_path.parent)}"></video>
                  <div class="metric">geometry_proxy: <strong>{item.geometry_score:.4f}</strong></div>
                  <pre>{json.dumps(item.geometry_details, ensure_ascii=False, indent=2)}</pre>
                </div>
                """
            )
        cards.append(
            f"""
            <section class="case-card">
              <div class="case-head">
                <div>
                  <h2>{case_id}</h2>
                  <div class="meta">{header.target_object} | {header.prompt}</div>
                </div>
                <div class="ref-grid">
                  <div>
                    <div class="label">GT first frame</div>
                    <img src="{os.path.relpath(header.first_frame_path, output_path.parent)}" alt="{case_id} first frame" />
                  </div>
                </div>
              </div>
              <div class="providers">
                {''.join(provider_blocks)}
              </div>
            </section>
            """
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PDI Proxy Eval</title>
  <style>
    :root {{
      --bg: #f2efe8;
      --panel: #fffaf1;
      --line: #cbbda7;
      --text: #1f1c18;
      --muted: #6d6559;
      --accent: #a24d2c;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(162,77,44,0.12), transparent 28%),
        linear-gradient(180deg, #f7f1e7 0%, var(--bg) 100%);
      color: var(--text);
    }}
    .page {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      letter-spacing: -0.03em;
    }}
    .sub {{
      color: var(--muted);
      margin-bottom: 24px;
    }}
    .case-card {{
      border: 1px solid var(--line);
      background: rgba(255,250,241,0.92);
      border-radius: 22px;
      padding: 20px;
      margin-bottom: 22px;
      box-shadow: 0 18px 60px rgba(60,40,20,0.08);
    }}
    .case-head {{
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 18px;
      align-items: start;
      margin-bottom: 20px;
    }}
    .meta {{
      color: var(--muted);
      line-height: 1.45;
      font-size: 14px;
    }}
    .ref-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}
    .providers {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    .provider-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: #fffdf8;
    }}
    .label {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }}
    video, img {{
      width: 100%;
      border-radius: 12px;
      display: block;
      background: #ddd3c5;
    }}
    pre {{
      overflow: auto;
      font-size: 12px;
      background: #f6efe4;
      border-radius: 12px;
      padding: 10px;
      color: #3b3329;
    }}
    .metric {{
      margin: 10px 0 8px;
      font-size: 15px;
    }}
    strong {{
      color: var(--accent);
    }}
    @media (max-width: 980px) {{
      .case-head, .providers, .ref-grid {{
        grid-template-columns: 1fr;
      }}
      .page {{
        padding: 16px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>PDI Proxy Eval</h1>
    <div class="sub">Wan native TI2V vs VACE native TI2V-firstframe on selected PDI-style cases. Scores are local geometry-proxy scores, not official PDI-Bench outputs.</div>
    {''.join(cards)}
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def run_eval(
    *,
    output_root: Path,
    tmp_root: Path,
    cases: list[PDICase],
    wan_root: Path,
    vace_root: Path,
    device: str,
    width: int,
    height: int,
    fps: int,
    num_frames: int,
    num_inference_steps: int,
    cfg_scale: float,
    quality: int,
    negative_prompt: str,
    seed: int,
) -> dict[str, Any]:
    run_dir = ensure_dir(output_root)
    gt_root = ensure_dir(tmp_root / "gt_videos")
    ref_root = ensure_dir(run_dir / "reference")
    generated_root = ensure_dir(run_dir / "generated")
    report_root = ensure_dir(run_dir / "report")

    scorer = GeometryProxyScorer(GeometryConfig())
    results: list[EvalResult] = []

    prepared_cases: list[tuple[int, PDICase, Path, Path]] = []
    for case_index, case in enumerate(cases):
        gt_video_path = ensure_gt_video(case, gt_root)
        first_frame_path = extract_first_frame_image(gt_video_path, ref_root / case.case_id / "first_frame.png")
        geometry_score, geometry_details = scorer.score_from_anchor_image(
            anchor_image_path=first_frame_path,
            candidate_video_path=gt_video_path,
        )
        results.append(
            EvalResult(
                case_id=case.case_id,
                provider="gt",
                prompt=case.prompt,
                target_object=case.target_object,
                gt_video_path=gt_video_path,
                first_frame_path=first_frame_path,
                video_path=gt_video_path,
                geometry_score=geometry_score,
                geometry_details=geometry_details,
            )
        )
        prepared_cases.append((case_index, case, gt_video_path, first_frame_path))

    provider_specs = [
        ("wan", WanTI2VRunner, wan_root, 0),
        ("vace", VaceTI2VRunner, vace_root, 1),
    ]
    for provider_name, runner_cls, model_root, seed_offset in provider_specs:
        runner = runner_cls(model_root, device=device)
        try:
            for case_index, case, gt_video_path, first_frame_path in prepared_cases:
                output_path = generated_root / case.case_id / provider_name / f"{provider_name}.mp4"
                ensure_dir(output_path.parent)
                runner.generate(
                    first_frame_path=first_frame_path,
                    prompt=case.prompt,
                    output_path=output_path,
                    seed=seed + case_index * 10 + seed_offset,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=fps,
                    num_inference_steps=num_inference_steps,
                    cfg_scale=cfg_scale,
                    quality=quality,
                )
                geometry_score, geometry_details = scorer.score_from_anchor_image(
                    anchor_image_path=first_frame_path,
                    candidate_video_path=output_path,
                )
                results.append(
                    EvalResult(
                        case_id=case.case_id,
                        provider=provider_name,
                        prompt=case.prompt,
                        target_object=case.target_object,
                        gt_video_path=gt_video_path,
                        first_frame_path=first_frame_path,
                        video_path=output_path,
                        geometry_score=geometry_score,
                        geometry_details=geometry_details,
                    )
                )
        finally:
            del runner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "cases": [asdict(case) for case in cases],
        "results": [
            {
                **asdict(item),
                "gt_video_path": str(item.gt_video_path),
                "first_frame_path": str(item.first_frame_path),
                "video_path": str(item.video_path),
            }
            for item in results
        ],
    }
    write_json(report_root / "summary.json", summary)
    _render_html(results, report_root / "index.html")
    return {
        "run_dir": str(run_dir),
        "html_path": str(report_root / "index.html"),
        "summary_path": str(report_root / "summary.json"),
        "result_count": len(results),
    }
