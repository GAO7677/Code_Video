#!/usr/bin/env python3
"""Compare pure xSSC slot-track de-duplication across checkpoints.

This script intentionally does not load Wan.  It only runs:

    training case -> xSSC preprocessing -> optional AMG boxes
    -> frozen xSSC slots -> slot-track similarity -> dedup/merge

Each checkpoint is processed in a fresh subprocess so DINOv2/DINOv3 variants
can use their own ``object_centric_bench`` package without import-cache leaks.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
TRAIN_XSSC_ROOT = ROOT.parent
PROJECT_ROOT = TRAIN_XSSC_ROOT.parent
PACKAGE_PARENT = PROJECT_ROOT.parent
DEFAULT_TRAIN_CONFIG = ROOT / "configs/formal_full_sa_slot_dedup_merge_gpu67.json"
DEFAULT_RESTART_ROOT = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/"
    "dinov3_xSSC/restart_save1000_20260720T140029Z"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/xssc_slot_dedup_weight_compare"
)
DEFAULT_CASE_INDICES = "808,58755,142643"
PYTHON_BIN = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--restart-root", type=Path, default=DEFAULT_RESTART_ROOT)
    parser.add_argument("--official-root", type=Path, default=Path("/data/gaoya/ckpt/xSSC"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--indices", default=DEFAULT_CASE_INDICES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default="3")
    parser.add_argument("--dedup-threshold", type=float, default=0.94)
    parser.add_argument("--dedup-min-keep", type=int, default=3)
    parser.add_argument("--dedup-mode", choices=["merge", "mask"], default="merge")
    parser.add_argument("--metric", choices=["mean_frame_cosine", "pooled_cosine"], default="mean_frame_cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker-spec", type=Path, default=None)
    return parser.parse_args()


def natural_step(path: Path) -> int:
    match = re.search(r"step-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(root: Path) -> Path | None:
    checkpoints = sorted(root.glob("step-*.pth"), key=natural_step)
    return checkpoints[-1] if checkpoints else None


def discover_specs(restart_root: Path, official_root: Path) -> list[dict[str, Any]]:
    vitl_root = TRAIN_XSSC_ROOT / "xssc_rsfq2_ytvis_dinov3_vitl16_256"
    vits_root = TRAIN_XSSC_ROOT / "xssc_rsfq2_movic_dinov3_vits16_official_dims"
    specs: list[dict[str, Any]] = []

    ytvis_dir = restart_root / "rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512/42"
    ckpt = latest_checkpoint(ytvis_dir)
    if ckpt is not None:
        specs.append(
            {
                "name": f"dinov3_vitl_ytvis_hq_slot512_{ckpt.stem}",
                "short_name": f"ViT-L YTVIS {ckpt.stem}",
                "family": "dinov3",
                "xssc_root": str(vitl_root),
                "xssc_config": str(
                    vitl_root
                    / "upstream/config-randsfq/rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512.py"
                ),
                "xssc_checkpoint": str(ckpt),
                "dinov3_root": str(vitl_root / "third_party/dinov3"),
                "dinov3_checkpoint": "/data/gaoya/ckpt/facebook-dinov3-vitl16-pretrain-lvd1689m/model.safetensors",
            }
        )

    movic_dirs = sorted(restart_root.glob("movi_c_transfer*/rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/42"))
    seen_checkpoints: set[str] = set()
    for movic_dir in movic_dirs:
        ckpt = latest_checkpoint(movic_dir)
        if ckpt is None or str(ckpt) in seen_checkpoints:
            continue
        seen_checkpoints.add(str(ckpt))
        run_name = movic_dir.parents[1].name
        specs.append(
            {
                "name": f"dinov3_vitl_movic_transfer_{run_name}_{ckpt.stem}",
                "short_name": f"ViT-L MOVi-C {ckpt.stem}",
                "family": "dinov3",
                "xssc_root": str(vitl_root),
                "xssc_config": str(
                    vitl_root
                    / "upstream/config-randsfq/rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000.py"
                ),
                "xssc_checkpoint": str(ckpt),
                "dinov3_root": str(vitl_root / "third_party/dinov3"),
                "dinov3_checkpoint": "/data/gaoya/ckpt/facebook-dinov3-vitl16-pretrain-lvd1689m/model.safetensors",
            }
        )

    vits_dir = (
        restart_root
        / "ytvis_hq_dinov3_vits16_official_dims_b192_acc1_20260723T125549Z"
        / "rsfq2_r-ytvis_hq-dinov3_vits16_256-official_dims/42"
    )
    ckpt = latest_checkpoint(vits_dir)
    if ckpt is not None:
        specs.append(
            {
                "name": f"dinov3_vits_ytvis_hq_official_dims_{ckpt.stem}",
                "short_name": f"ViT-S YTVIS official-dims {ckpt.stem}",
                "family": "dinov3",
                "xssc_root": str(vits_root),
                "xssc_config": str(
                    vits_root
                    / "upstream/config-randsfq/rsfq2_r-ytvis_hq-dinov3_vits16_256-official_dims.py"
                ),
                "xssc_checkpoint": str(ckpt),
                "dinov3_root": str(vits_root / "third_party/dinov3"),
                "dinov3_checkpoint": "/data/gaoya/ckpt/facebook-dinov3-vits16-pretrain-lvd1689m/model.safetensors",
            }
        )

    for ckpt in sorted((official_root / "rsfq2_r-ytvis").glob("*.pth")):
        specs.append(
            {
                "name": f"official_dinov2_r_ytvis_{ckpt.stem}",
                "short_name": f"Official DINOv2 YTVIS {ckpt.stem}",
                "family": "dinov2",
                "xssc_root": "/home/gaoya/Code_Video/xSSC-main",
                "xssc_config": "/home/gaoya/Code_Video/xSSC-main/config-randsfq/rsfq2_r-ytvis.py",
                "xssc_checkpoint": str(ckpt),
            }
        )
    return specs


def worker_command(args: argparse.Namespace, spec_path: Path, model_dir: Path) -> list[str]:
    return [
        str(PYTHON_BIN),
        str(Path(__file__).resolve()),
        "--worker-spec",
        str(spec_path),
        "--train-config",
        str(args.train_config),
        "--output-dir",
        str(model_dir),
        "--indices",
        str(args.indices),
        "--device",
        str(args.device),
        "--dedup-threshold",
        str(args.dedup_threshold),
        "--dedup-min-keep",
        str(args.dedup_min_keep),
        "--dedup-mode",
        str(args.dedup_mode),
        "--metric",
        str(args.metric),
        "--seed",
        str(args.seed),
    ]


def run_orchestrator(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = discover_specs(args.restart_root.expanduser().resolve(), args.official_root.expanduser().resolve())
    if not specs:
        raise RuntimeError("No xSSC checkpoints discovered")

    spec_dir = output_dir / "_specs"
    model_root = output_dir / "models"
    spec_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(PROJECT_ROOT.parent),
            str(PROJECT_ROOT),
            str(TRAIN_XSSC_ROOT),
            str(ROOT),
            "/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main",
            env.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)

    worker_results = []
    for spec in specs:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec["name"])
        spec_path = spec_dir / f"{safe_name}.json"
        spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        model_dir = model_root / safe_name
        if model_dir.exists() and args.force:
            subprocess.run(["rm", "-rf", str(model_dir)], check=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        command = worker_command(args, spec_path, model_dir)
        log_path = model_dir / "worker.log"
        print(f"[run] {spec['short_name']} -> {model_dir}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                command,
                env=env,
                cwd=str(ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            log.write(proc.stdout)
        if proc.returncode != 0:
            print(proc.stdout, flush=True)
            raise RuntimeError(f"Worker failed for {spec['name']}; see {log_path}")
        metadata_path = model_dir / "metadata.json"
        worker_results.append(json.loads(metadata_path.read_text(encoding="utf-8")))

    build_compare_page(output_dir, worker_results, args)
    print(f"viewer={output_dir / 'index.html'}", flush=True)


def build_compare_page(output_dir: Path, results: list[dict], args: argparse.Namespace) -> None:
    all_cases: dict[int, list[dict]] = {}
    models = []
    for result in results:
        models.append(result["model"])
        for case in result["cases"]:
            all_cases.setdefault(int(case["index"]), []).append(
                {
                    "model": result["model"],
                    "case": case,
                    "base": str(Path("models") / result["model"]["safe_name"]),
                }
            )

    model_rows = []
    for model in models:
        model_rows.append(
            "<tr>"
            f"<td>{html.escape(model['short_name'])}</td>"
            f"<td>{html.escape(model['family'])}</td>"
            f"<td>{html.escape(str(model['slot_shape']))}</td>"
            f"<td>{html.escape(model['initializer'])}</td>"
            f"<td><code>{html.escape(model['checkpoint'])}</code></td>"
            "</tr>"
        )

    case_sections = []
    for case_index, entries in sorted(all_cases.items()):
        source = entries[0]["case"].get("source", "unknown")
        cards = []
        for entry in entries:
            model = entry["model"]
            case = entry["case"]
            base = Path(entry["base"])
            merge_lines = []
            for group in case["merge_groups"]:
                if group["duplicates"]:
                    merge_lines.append(
                        f"rep {group['representative']} <= {group['members']}"
                    )
            merge_text = "<br>".join(html.escape(item) for item in merge_lines) or "no merge"
            stats = case["dedup_stats"]
            cards.append(
                f"""
                <article class="card">
                  <h3>{html.escape(model['short_name'])}</h3>
                  <div class="small">
                    retained {case['retained_slots']}/{case['num_slots']} |
                    dup {stats['duplicate_fraction_mean']:.3f} |
                    mean offdiag {stats['mean_offdiag_similarity']:.3f}
                  </div>
                  <div class="merge">{merge_text}</div>
                  <div class="imgs">
                    <figure><img src="{base / case['before_heatmap']}" loading="lazy"><figcaption>before</figcaption></figure>
                    <figure><img src="{base / case['active_heatmap']}" loading="lazy"><figcaption>active after</figcaption></figure>
                  </div>
                  <div class="vids">
                    <figure><video src="{base / case['before_overlay']}" controls muted preload="metadata"></video><figcaption>slot overlay before</figcaption></figure>
                    <figure><video src="{base / case['after_overlay']}" controls muted preload="metadata"></video><figcaption>slot overlay after</figcaption></figure>
                  </div>
                </article>
                """
            )
        case_sections.append(
            f"""
            <section class="case">
              <h2>case index {case_index} | {html.escape(str(source))}</h2>
              <div class="grid">{''.join(cards)}</div>
            </section>
            """
        )

    report = {
        "indices": args.indices,
        "threshold": args.dedup_threshold,
        "metric": args.metric,
        "mode": args.dedup_mode,
        "min_keep": args.dedup_min_keep,
        "num_models": len(models),
    }
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>xSSC Slot Dedup Weight Compare</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:#101214; color:#eef2f7; font:13px system-ui,sans-serif; letter-spacing:0; }}
    header {{ position:sticky; top:0; z-index:5; padding:12px 16px; background:#16191c; border-bottom:1px solid #333b44; }}
    h1 {{ margin:0 0 6px; font-size:20px; }}
    h2 {{ margin:0 0 12px; font-size:17px; }}
    h3 {{ margin:0 0 6px; font-size:14px; }}
    main {{ max-width:2200px; margin:0 auto; padding:16px; }}
    code {{ color:#d5f5ff; }}
    .summary {{ color:#bdc7d1; }}
    table {{ width:100%; border-collapse:collapse; margin:12px 0 18px; }}
    th,td {{ border:1px solid #303942; padding:6px 8px; text-align:left; vertical-align:top; }}
    th {{ background:#192027; }}
    td {{ background:#12171c; color:#cbd5df; }}
    .case {{ padding:18px 0 28px; border-top:1px solid #30363d; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:12px; }}
    .card {{ border:1px solid #333b44; background:#14191e; padding:10px; border-radius:8px; }}
    .small {{ color:#b8c1cb; margin-bottom:6px; }}
    .merge {{ color:#d7dee7; min-height:34px; line-height:1.35; margin-bottom:8px; }}
    .imgs,.vids {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }}
    figure {{ margin:0; min-width:0; }}
    img,video {{ display:block; width:100%; background:#000; border:1px solid #303942; }}
    figcaption {{ padding:4px 1px; color:#aeb8c2; font-size:11px; }}
  </style>
</head>
<body>
  <header>
    <h1>xSSC slot-track de-duplication: latest checkpoints + official weights</h1>
    <div class="summary">{html.escape(json.dumps(report, ensure_ascii=False))}</div>
  </header>
  <main>
    <h2>Compared checkpoints</h2>
    <table>
      <thead><tr><th>method</th><th>family</th><th>slot shape</th><th>initializer</th><th>checkpoint</th></tr></thead>
      <tbody>{''.join(model_rows)}</tbody>
    </table>
    {''.join(case_sections)}
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        json.dumps({"report": report, "models": models, "results": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_worker(args: argparse.Namespace) -> None:
    # Heavy imports stay in worker mode so orchestrator remains clean.
    import cv2
    import imageio_ffmpeg
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    for item in (PACKAGE_PARENT, PROJECT_ROOT, TRAIN_XSSC_ROOT, ROOT):
        text = str(item)
        if text not in sys.path:
            sys.path.insert(0, text)
    import launch_slot_dedup_from_config as config_launcher
    import train_xssc_object_self_attn_lora as object_train
    import train_xssc_object_self_attn_lora_slot_dedup as dedup_train

    spec = json.loads(args.worker_spec.read_text(encoding="utf-8"))
    spec["safe_name"] = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec["name"])
    output_dir = args.output_dir.expanduser().resolve()
    asset_root = output_dir / "assets"
    asset_root.mkdir(parents=True, exist_ok=True)

    train_args = load_dataset_args(config_launcher, object_train, dedup_train, args.train_config, output_dir)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dataset = object_train.base.build_dataset(train_args)
    indices = [int(item) for item in str(args.indices).replace(",", " ").split()]
    xssc, slot_dim, num_slots, initializer = load_xssc_variant(spec, device)
    cases = []
    for position, index in enumerate(indices, start=1):
        sample = dataset[index]
        payload = process_case(
            cv2=cv2,
            imageio_ffmpeg=imageio_ffmpeg,
            plt=plt,
            np=np,
            torch=torch,
            F=F,
            object_train=object_train,
            dedup_train=dedup_train,
            sample=sample,
            index=index,
            position=position,
            xssc=xssc,
            num_slots=num_slots,
            slot_dim=slot_dim,
            initializer=initializer,
            device=device,
            args=args,
            output_dir=output_dir,
            asset_root=asset_root,
        )
        cases.append(payload)
        print(
            f"[{spec['short_name']}] case={index} retained={payload['retained_slots']}/{num_slots}",
            flush=True,
        )

    model = {
        "name": spec["name"],
        "safe_name": spec["safe_name"],
        "short_name": spec["short_name"],
        "family": spec["family"],
        "checkpoint": spec["xssc_checkpoint"],
        "config": spec["xssc_config"],
        "root": spec["xssc_root"],
        "slot_shape": [8, num_slots, slot_dim],
        "initializer": initializer,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps({"model": model, "cases": cases}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_dataset_args(config_launcher, object_train, dedup_train, config_path: Path, output_dir: Path):
    raw, _ = config_launcher.base.load_config(config_path)
    config = config_launcher.validate_config(raw, config_path.expanduser().resolve().parent)
    command = config_launcher.build_command(config, output_dir / "_unused")
    script_index = next(index for index, token in enumerate(command) if str(token).endswith(".py"))
    train_argv = [str(item) for item in command[script_index + 1 :]]
    return object_train.tvn.prepare_args(dedup_train.build_parser().parse_args(train_argv))


def config_initializer(config_path: Path) -> str:
    text = config_path.read_text(encoding="utf-8")
    if re.search(r"initializ\s*=\s*dict\(type=MLP", text):
        return "bbox_mlp"
    return "normal_shared"


def load_xssc_variant(spec: dict, device):
    import torch

    xssc_root = Path(spec["xssc_root"]).expanduser().resolve()
    config_path = Path(spec["xssc_config"]).expanduser().resolve()
    checkpoint_path = Path(spec["xssc_checkpoint"]).expanduser().resolve()
    initializer = config_initializer(config_path)
    if spec["family"] == "dinov3":
        upstream = xssc_root / "upstream"
        dinov3_root = Path(spec["dinov3_root"]).expanduser().resolve()
        for path in (upstream, dinov3_root):
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)
        old_root = os.environ.get("DINOV3_ROOT")
        old_ckpt = os.environ.get("DINOV3_CHECKPOINT")
        os.environ["DINOV3_ROOT"] = str(dinov3_root)
        os.environ["DINOV3_CHECKPOINT"] = str(Path(spec["dinov3_checkpoint"]).expanduser().resolve())
        try:
            from object_centric_bench.util import Config, build_from_config

            cfg = Config.fromfile(config_path)
            model = build_from_config(cfg.model)
        finally:
            if old_root is None:
                os.environ.pop("DINOV3_ROOT", None)
            else:
                os.environ["DINOV3_ROOT"] = old_root
            if old_ckpt is None:
                os.environ.pop("DINOV3_CHECKPOINT", None)
            else:
                os.environ["DINOV3_CHECKPOINT"] = old_ckpt
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        if state and all(str(key).startswith("m.") for key in state):
            state = {str(key)[2:]: value for key, value in state.items()}
        incompatible = model.load_state_dict(state, strict=False)
        bad_missing = [key for key in incompatible.missing_keys if not key.startswith("encode_backbone.")]
        if bad_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"DINOv3 xSSC checkpoint mismatch: missing={bad_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        slot_dim = int(cfg.emb_dim)
        num_slots = int(cfg.max_num)
    else:
        text = str(xssc_root)
        if text not in sys.path:
            sys.path.insert(0, text)
        import timm
        from object_centric_bench.util import Config, build_from_config

        cfg = Config.fromfile(config_path)
        original_create_model = timm.create_model

        def create_model_offline(*create_args, **kwargs):
            kwargs["pretrained"] = False
            return original_create_model(*create_args, **kwargs)

        timm.create_model = create_model_offline
        try:
            model = build_from_config(cfg.model)
        finally:
            timm.create_model = original_create_model
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        if state and all(str(key).startswith("m.") for key in state):
            state = {str(key)[2:]: value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        slot_dim = int(cfg.emb_dim)
        num_slots = int(cfg.max_num)
    model.decode = None
    model.requires_grad_(False)
    model.eval()
    model.to(device=device)
    return model, slot_dim, num_slots, initializer


def process_case(
    *,
    cv2,
    imageio_ffmpeg,
    plt,
    np,
    torch,
    F,
    object_train,
    dedup_train,
    sample: dict,
    index: int,
    position: int,
    xssc,
    num_slots: int,
    slot_dim: int,
    initializer: str,
    device,
    args: argparse.Namespace,
    output_dir: Path,
    asset_root: Path,
) -> dict:
    context_video = sample["context_video"]
    torch.manual_seed(int(args.seed) + int(index))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed) + int(index))
    raw_rgb = to_uint8_video(np, torch, context_video)
    xssc_video = preprocess_xssc(torch, F, object_train, context_video, 256).to(device=device)
    boxes = None
    selected_boxes = 0
    if initializer == "bbox_mlp":
        builder = object_train.AMGBoxBuilder(
            sam2_config=object_train.DEFAULT_SAM2_CONFIG,
            sam2_checkpoint=object_train.DEFAULT_SAM2_CHECKPOINT,
            cache_dir="/data/gaoya/agent-data/cache/xssc_slot_dedup_weight_compare_amg",
            filter_args=object_train._amg_filter_args_from_args(
                SimpleNamespaceFromDefaults(object_train)
            ),
        )
        boxes = builder(xssc_video, num_slots)
        selected_boxes = int(builder.last_selected_counts[0]) if builder.last_selected_counts else 0
    slots, attention = extract_slots_generic(torch, xssc, xssc_video, boxes, initializer)
    similarity = dedup_train.compute_slot_track_similarity(slots, metric=args.metric)
    groups = dedup_train._connected_components_from_similarity(
        similarity[0],
        threshold=float(args.dedup_threshold),
        min_keep=int(args.dedup_min_keep),
    )
    deduped_slots, keep_mask, dedup_stats = dedup_train.deduplicate_xssc_slot_tracks(
        slots,
        mode=args.dedup_mode,
        threshold=float(args.dedup_threshold),
        similarity_metric=args.metric,
        min_keep=int(args.dedup_min_keep),
    )
    after_similarity = dedup_train.compute_slot_track_similarity(deduped_slots, metric=args.metric)
    keep_np = keep_mask[0].detach().cpu().numpy().astype(bool)
    active_similarity = after_similarity[0].detach().cpu().numpy()[keep_np][:, keep_np]
    if active_similarity.size == 0:
        active_similarity = np.zeros((1, 1), dtype=np.float32)

    case_dir = asset_root / f"case_{position:02d}_index_{index:06d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    before_heatmap = case_dir / "slot_similarity_before.png"
    active_heatmap = case_dir / "slot_similarity_after_active.png"
    save_heatmap(plt, np, before_heatmap, similarity[0].detach().cpu().numpy(), "before dedup", groups, args.dedup_threshold)
    save_heatmap(plt, np, active_heatmap, active_similarity, "active after dedup", [], args.dedup_threshold)
    before_overlay = case_dir / "slot_overlay_before.mp4"
    after_overlay = case_dir / "slot_overlay_after_merge.mp4"
    write_video(cv2, imageio_ffmpeg, before_overlay, slot_overlay_video(cv2, np, raw_rgb, attention), 3.0)
    write_video(
        cv2,
        imageio_ffmpeg,
        after_overlay,
        slot_overlay_video(cv2, np, raw_rgb, attention, groups=groups, mode=args.dedup_mode),
        3.0,
    )
    metadata = dict(sample.get("metadata", {}))
    return {
        "index": int(index),
        "source": str(metadata.get("dataset_source", "unknown")),
        "num_slots": int(num_slots),
        "slot_dim": int(slot_dim),
        "selected_boxes": int(selected_boxes),
        "retained_slots": int(keep_mask[0].sum().item()),
        "keep_mask": keep_np.astype(int).tolist(),
        "merge_groups": groups_to_payload(groups, similarity[0].detach().cpu().numpy()),
        "dedup_stats": dedup_stats,
        "before_heatmap": str(before_heatmap.relative_to(output_dir)),
        "active_heatmap": str(active_heatmap.relative_to(output_dir)),
        "before_overlay": str(before_overlay.relative_to(output_dir)),
        "after_overlay": str(after_overlay.relative_to(output_dir)),
    }


class SimpleNamespaceFromDefaults:
    def __init__(self, object_train):
        defaults = {
            "xssc_amg_max_selected": 11,
            "xssc_amg_min_area_ratio": 0.004,
            "xssc_amg_max_area_ratio": 0.35,
            "xssc_amg_min_bbox_side": 7.0,
            "xssc_amg_background_area_ratio": 0.06,
            "xssc_amg_background_span_ratio": 0.75,
            "xssc_amg_border_area_ratio": 0.025,
            "xssc_amg_border_occupancy_ratio": 0.18,
            "xssc_amg_opposite_edge_area_ratio": 0.04,
            "xssc_amg_shadow_min_area_ratio": 0.03,
            "xssc_amg_shadow_max_luminance_ratio": 0.55,
            "xssc_amg_shadow_max_chromaticity_distance": 0.1,
            "xssc_amg_shadow_max_gradient_mean": 20.0,
            "xssc_amg_duplicate_iou": 0.7,
            "xssc_amg_duplicate_containment": 0.85,
        }
        self.__dict__.update(defaults)


def to_uint8_video(np, torch, context_video):
    frames = context_video.permute(1, 2, 3, 0).float()
    frames = (frames + 1.0).mul(127.5).round().clamp(0, 255)
    return frames.to(torch.uint8).cpu().numpy()


def preprocess_xssc(torch, F, object_train, context_video, input_size: int):
    frames = context_video.unsqueeze(0).permute(0, 2, 1, 3, 4).float()
    batch, time_steps, channels, height, width = frames.shape
    crop_size = min(int(height), int(width))
    top = (int(height) - crop_size) // 2
    left = (int(width) - crop_size) // 2
    frames = frames[..., top : top + crop_size, left : left + crop_size]
    frames = F.interpolate(
        frames.reshape(batch * time_steps, channels, crop_size, crop_size),
        size=(input_size, input_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    frames = (frames + 1.0).mul(127.5).clamp(0.0, 255.0)
    mean = frames.new_tensor(object_train.base.XSSC_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = frames.new_tensor(object_train.base.XSSC_IMAGENET_STD).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    return frames.view(batch, time_steps, channels, input_size, input_size)


def extract_slots_generic(torch, model, video, boxes, initializer: str):
    model.eval()
    batch, time_steps, _, _, _ = video.shape
    flat_video = video.flatten(0, 1)
    with torch.inference_mode(), torch.autocast(
        device_type=flat_video.device.type,
        dtype=torch.bfloat16,
        enabled=flat_video.device.type == "cuda",
    ):
        feature = model.encode_backbone(flat_video).detach()
        encoded = feature.permute(0, 2, 3, 1)
        encoded = model.encode_posit_embed(encoded).flatten(1, 2)
        encoded = model.encode_project(encoded)
        encoded = encoded.view(batch, time_steps, encoded.shape[1], encoded.shape[2])
        if boxes is not None:
            boxes = boxes.to(device=encoded.device, dtype=encoded.dtype)
        slots = None
        attentions = []
        for frame_id in range(time_steps):
            if frame_id == 0:
                if initializer == "bbox_mlp":
                    if boxes is None:
                        raise RuntimeError("bbox_mlp initializer requires boxes")
                    query = model.initializ(boxes[:, 0])
                else:
                    query = model.initializ(batch)
            else:
                query = model.transit(slots, encoded[:, : frame_id + 1])
            num_iter = None if frame_id == 0 else 1
            current_slots, current_attention = model.aggregat(
                encoded[:, frame_id],
                query,
                num_iter=num_iter,
            )
            current_slots = current_slots[:, None]
            slots = current_slots if slots is None else torch.cat((slots, current_slots), dim=1)
            attentions.append(current_attention)
        attention = torch.stack(attentions, dim=1)
        patch_side = int(round(attention.shape[-1] ** 0.5))
        attention = attention.view(batch, time_steps, attention.shape[2], patch_side, patch_side)
    return slots, attention


def groups_to_payload(groups, similarity):
    payload = []
    for group in groups:
        pairs = []
        for i, left in enumerate(group):
            for right in group[i + 1 :]:
                pairs.append(
                    {
                        "pair": [int(left), int(right)],
                        "similarity": float(similarity[int(left), int(right)]),
                    }
                )
        payload.append(
            {
                "representative": int(group[0]),
                "members": [int(item) for item in group],
                "duplicates": [int(item) for item in group[1:]],
                "pairs": pairs,
            }
        )
    return payload


def save_heatmap(plt, np, path: Path, matrix, title: str, groups, threshold: float):
    fig_size = max(4.2, min(7.2, matrix.shape[0] * 0.62))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=150)
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            color = "white" if abs(value) > 0.65 else "black"
            weight = "bold" if row != col and value >= threshold else "normal"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=6, color=color, fontweight=weight)
    for group in groups:
        if len(group) <= 1:
            continue
        for left in group:
            for right in group:
                ax.add_patch(
                    plt.Rectangle((right - 0.5, left - 0.5), 1, 1, fill=False, edgecolor="#111827", linewidth=1.1)
                )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def slot_overlay_video(cv2, np, frames_rgb, attention, *, groups=None, mode="before"):
    palette = np.asarray(
        [
            [239, 68, 68],
            [59, 130, 246],
            [34, 197, 94],
            [250, 204, 21],
            [168, 85, 247],
            [6, 182, 212],
            [249, 115, 22],
            [236, 72, 153],
            [132, 204, 22],
            [20, 184, 166],
            [244, 114, 182],
            [148, 163, 184],
            [251, 146, 60],
            [45, 212, 191],
            [192, 132, 252],
        ],
        dtype=np.uint8,
    )
    labels = attention[0].float().cpu().numpy().argmax(axis=1).astype(np.int32)
    remap = {slot: slot for slot in range(palette.shape[0])}
    hidden = set()
    if groups is not None:
        for group in groups:
            rep = int(group[0])
            for item in group[1:]:
                if mode == "mask":
                    hidden.add(int(item))
                else:
                    remap[int(item)] = rep
    output = []
    for frame_id, frame in enumerate(frames_rgb):
        label_small = labels[min(frame_id, labels.shape[0] - 1)].copy()
        original_small = label_small.copy()
        for source, target in remap.items():
            if source != target:
                label_small[label_small == source] = target
        label_map = cv2.resize(
            label_small.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        color = palette[label_map % len(palette)].copy()
        if hidden:
            hidden_mask = np.zeros_like(original_small, dtype=bool)
            for slot_id in hidden:
                hidden_mask |= original_small == slot_id
            hidden_mask = cv2.resize(
                hidden_mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            color[hidden_mask] = np.asarray([150, 150, 150], dtype=np.uint8)
        output.append(cv2.addWeighted(frame, 0.58, color, 0.42, 0.0))
    return np.stack(output, axis=0)


def write_video(cv2, imageio_ffmpeg, path: Path, frames_rgb, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_rgb = frames_rgb.astype("uint8")
    height, width = frames_rgb.shape[1:3]
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(path),
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = proc.communicate(frames_rgb.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker_spec is not None:
        from types import SimpleNamespace

        run_worker(parsed)
    else:
        run_orchestrator(parsed)
