#!/usr/bin/env python3
"""Generate one GT/pred-x0/xSSC-overlay diagnostic for one xSSC backend."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_DIR.parent
TRAIN_XSSC_ROOT = EXPERIMENT_ROOT.parent
PACKAGE_ROOT = EXPERIMENT_ROOT.parents[2]
DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
for _path in (
    PROJECT_DIR,
    EXPERIMENT_ROOT,
    TRAIN_XSSC_ROOT,
    PACKAGE_ROOT,
    DIFFSYNTH_ROOT,
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import av
import torch
import torch.nn.functional as F

import launch_from_config as launcher
import train_xssc_context_slots as dataset_module
import train_xssc_object_self_attn_lora as core
import train_xssc_object_self_attn_lora_xssc_loss as trainer


DEFAULT_OUTPUT = Path("/data/gaoya/agent-data/outputs/xssc_loss_diagnostics")
PALETTE = np.asarray(
    [
        (231, 76, 60),
        (52, 152, 219),
        (46, 204, 113),
        (241, 196, 15),
        (155, 89, 182),
        (230, 126, 34),
        (26, 188, 156),
        (236, 112, 99),
        (52, 73, 94),
        (149, 165, 166),
        (255, 105, 180),
    ],
    dtype=np.float32,
)


class _Accelerator:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.is_main_process = True

    @staticmethod
    def print(*args, **kwargs) -> None:
        print(*args, **kwargs)


def _parse_trainer_args(config_path: Path, width: int, height: int):
    raw, _ = launcher.load_config(config_path)
    config = launcher.validate_config(raw, config_path.resolve().parent)
    command = launcher.build_command(config, DEFAULT_OUTPUT / "unused")
    script_index = command.index(str(launcher.XSSC_LOSS_TRAIN_SCRIPT))
    args = trainer.build_parser().parse_args(command[script_index + 1 :])
    args.height = int(height)
    args.width = int(width)
    args.train_batch_size = 1
    # Use a deterministic PyBullet case for a backend-to-backend comparison.
    args.mixture_pybullet_ratio = 1.0
    args.mixture_kubric_ratio = 0.0
    args.mixture_openvid_ratio = 0.0
    args.xssc_loss_gradient_diagnostics_every_n_forwards = 1_000_000
    return core.tvn.prepare_args(args), config


def _to_uint8_video(video: torch.Tensor) -> np.ndarray:
    item = video[0].detach().float().clamp(-1.0, 1.0)
    item = ((item + 1.0) * 127.5).round().to(torch.uint8)
    return item.permute(1, 2, 3, 0).cpu().numpy()


def _attention_overlay(
    video: torch.Tensor,
    attention: torch.Tensor,
    valid_slots: torch.Tensor,
    alpha: float = 0.48,
) -> np.ndarray:
    base = _to_uint8_video(video).astype(np.float32)
    weights = attention[0].detach().float()
    valid = valid_slots[0].to(device=weights.device, dtype=weights.dtype)
    weights = weights * valid[None, :, None, None]
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    color_table = torch.as_tensor(
        PALETTE[: weights.shape[1]],
        device=weights.device,
        dtype=weights.dtype,
    )
    colors = torch.einsum("tshw,sc->tchw", weights, color_table)
    colors = F.interpolate(
        colors,
        size=(base.shape[1], base.shape[2]),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1).cpu().numpy()
    overlay = (1.0 - alpha) * base + alpha * colors
    return np.clip(np.rint(overlay), 0, 255).astype(np.uint8)


def _write_mp4(path: Path, frames: np.ndarray, fps: int = 15) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = int(frames.shape[2])
    stream.height = int(frames.shape[1])
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}
    for image in frames:
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _jsonable_sample_metadata(sample: dict) -> dict:
    result = {}
    for key, value in sample.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, Path):
            result[key] = str(value)
    return result


def _render_dashboard(output_root: Path) -> Path:
    records = []
    for metadata_path in sorted(output_root.glob("*/metadata.json")):
        records.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    cards = []
    for record in records:
        backend = html.escape(record["backend"])
        folder = html.escape(record["folder"])
        metrics = record["metrics"]
        metric_rows = "".join(
            f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
            for key, value in metrics.items()
        )
        videos = []
        for filename, title in (
            ("gt.mp4", "GT video"),
            ("pred_x0.mp4", "Predicted x0 (Wan VAE)"),
            ("gt_xssc_overlay.mp4", "GT xSSC slot overlay"),
            ("pred_xssc_overlay.mp4", "Pred x0 xSSC slot overlay"),
        ):
            videos.append(
                "<figure><figcaption>"
                + html.escape(title)
                + f'</figcaption><video controls loop muted playsinline src="{folder}/{filename}"></video></figure>'
            )
        cards.append(
            f"""
            <section class="backend-card">
              <div class="heading"><div><h2>{backend}</h2>
              <p>{html.escape(record['checkpoint'])}</p></div>
              <span>{record['slot_count']} slots × {record['slot_dim']} dims</span></div>
              <div class="video-grid">{''.join(videos)}</div>
              <details><summary>Diagnostic metrics and configuration</summary>
              <table>{metric_rows}</table></details>
            </section>
            """
        )
    if not cards:
        cards.append("<p>No diagnostic result has been generated yet.</p>")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>xSSC loss diagnostics</title><style>
:root{{--bg:#0b1020;--panel:#141b2d;--ink:#edf3ff;--muted:#9cabca;--line:#2c3855;--accent:#58d6c7}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px 24px 100px}} h1{{margin:0 0 6px;font-size:30px}} .intro{{color:var(--muted);margin:0 0 26px}}
.backend-card{{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:20px;margin:0 0 24px;box-shadow:0 10px 30px #0004}}
.heading{{display:flex;gap:20px;justify-content:space-between;align-items:flex-start}} h2{{margin:0;font-size:22px}} .heading p{{color:var(--muted);margin:4px 0 12px;word-break:break-all}} .heading span{{white-space:nowrap;color:var(--accent)}}
.video-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}} figure{{margin:0}} figcaption{{font-weight:650;margin:0 0 7px}} video{{display:block;width:100%;background:#000;border-radius:9px;border:1px solid var(--line)}}
details{{margin-top:16px}} summary{{cursor:pointer;color:var(--accent)}} table{{border-collapse:collapse;margin-top:10px;width:100%}} th,td{{padding:7px 9px;border:1px solid var(--line);text-align:left}} th{{width:42%;color:var(--muted)}}
#replay{{position:fixed;right:26px;bottom:24px;border:0;border-radius:999px;background:var(--accent);color:#06221f;font-weight:750;padding:13px 20px;box-shadow:0 7px 24px #0008;cursor:pointer}}
@media(max-width:850px){{.video-grid{{grid-template-columns:1fr}}.heading{{display:block}}}}
</style></head><body><main><h1>Full-SA + No-Object: frozen xSSC loss diagnostics</h1>
<p class="intro">Same Wan2.2 + merged OpenVid LoRA prediction path. Loss/overlays use future frames 8–48; colored regions are xSSC encoder slot-attention assignments.</p>
{''.join(cards)}</main><button id="replay">Replay all</button><script>
document.getElementById('replay').onclick=()=>{{document.querySelectorAll('video').forEach(v=>{{v.currentTime=0;v.play().catch(()=>{{}})}})}};
</script></body></html>"""
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "index.html"
    index_path.write_text(document, encoding="utf-8")
    return index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The diagnostic requires one CUDA GPU")
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)

    train_args, config = _parse_trainer_args(cli.config, cli.width, cli.height)
    dataset = dataset_module.build_dataset(train_args)
    sample = dataset[int(cli.sample_index) % len(dataset)]
    accelerator = _Accelerator(torch.device("cuda:0"))
    model = trainer.build_model(train_args, accelerator)
    model.train()
    # Backend construction consumes different amounts of RNG state. Reset here
    # so both diagnostics use the same flow-matching timestep/noise and therefore
    # the same Wan pred-x0 path before their frozen xSSC encoders diverge.
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)

    captured = {}
    original_xssc_feature_loss = model._xssc_feature_loss

    def collect_visuals(pred_x0_latents, target_x0_latents, **_kwargs):
        result = original_xssc_feature_loss(
            pred_x0_latents,
            target_x0_latents,
            return_visuals=True,
        )
        captured["visuals"] = result[2]
        return result

    model._xssc_feature_loss = collect_visuals
    with torch.no_grad():
        loss = model(sample)
    visuals = captured.get("visuals")
    if visuals is None:
        raise RuntimeError("Diagnostic forward did not capture xSSC visuals")

    backend = train_args.xssc_loss_backend
    output_dir = cli.output_root.resolve() / backend
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_mp4(output_dir / "gt.mp4", _to_uint8_video(visuals["target_video"]), cli.fps)
    _write_mp4(output_dir / "pred_x0.mp4", _to_uint8_video(visuals["pred_video"]), cli.fps)
    _write_mp4(
        output_dir / "gt_xssc_overlay.mp4",
        _attention_overlay(
            visuals["target_video"],
            visuals["target_attention"],
            visuals["valid_slots"],
        ),
        cli.fps,
    )
    _write_mp4(
        output_dir / "pred_xssc_overlay.mp4",
        _attention_overlay(
            visuals["pred_video"],
            visuals["pred_attention"],
            visuals["valid_slots"],
        ),
        cli.fps,
    )
    metrics = dict(model.last_train_metrics)
    metrics.update(
        {
            "diagnostic_loss_tensor": float(loss.detach().item()),
            "resolution": f"{cli.height}x{cli.width}",
            "frames": int(visuals["target_video"].shape[2]),
            "fps": cli.fps,
            "sample_index": cli.sample_index,
            "sample": _jsonable_sample_metadata(sample),
            "note": "visual-only no_grad forward; training configs remain 512x896",
        }
    )
    metadata = {
        "backend": backend,
        "folder": backend,
        "checkpoint": config["paths"]["xssc_checkpoint"],
        "slot_count": model.xssc_loss_num_slots,
        "slot_dim": model.xssc_loss_slot_dim,
        "metrics": metrics,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    index_path = _render_dashboard(cli.output_root.resolve())
    print(
        json.dumps(
            {
                "backend": backend,
                "output_dir": str(output_dir),
                "dashboard": str(index_path),
                "metrics": metrics,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
