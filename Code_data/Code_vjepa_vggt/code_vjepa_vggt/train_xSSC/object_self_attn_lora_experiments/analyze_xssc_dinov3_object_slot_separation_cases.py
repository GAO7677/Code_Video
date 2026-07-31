#!/usr/bin/env python3
"""Run the object/slot separation analysis with latest DINOv3 xSSC weights.

This mirrors ``analyze_xssc_object_slot_separation_cases.py`` on the same JSON
source-video list, but each DINOv3 xSSC variant runs in a fresh subprocess so
the variant-local ``object_centric_bench`` package cannot leak across models.
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
PYTHON_BIN = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")

DEFAULT_RESTART_ROOT = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/"
    "dinov3_xSSC/restart_save1000_20260720T140029Z"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_slot_separation_cases_dinov3_latest"
)
DEFAULT_OFFICIAL_REFERENCE = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_slot_separation_cases"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="append", default=None)
    parser.add_argument("--restart-root", type=Path, default=DEFAULT_RESTART_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--official-reference-dir", type=Path, default=DEFAULT_OFFICIAL_REFERENCE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default="3")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--xssc-input-size", type=int, default=256)
    parser.add_argument("--xssc-batch-size", type=int, default=16)
    parser.add_argument("--raft-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--skip-raft", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker-spec", type=Path, default=None)
    return parser.parse_args()


def natural_step(path: Path) -> int:
    match = re.search(r"step-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(root: Path) -> Path | None:
    checkpoints = sorted(root.glob("step-*.pth"), key=natural_step)
    return checkpoints[-1] if checkpoints else None


def discover_dinov3_specs(restart_root: Path) -> list[dict[str, Any]]:
    vitl_root = TRAIN_XSSC_ROOT / "xssc_rsfq2_ytvis_dinov3_vitl16_256"
    vits_root = TRAIN_XSSC_ROOT / "xssc_rsfq2_movic_dinov3_vits16_official_dims"
    specs: list[dict[str, Any]] = []

    ytvis_dir = restart_root / "rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512/42"
    ckpt = latest_checkpoint(ytvis_dir)
    if ckpt is not None:
        specs.append(
            {
                "name": f"dinov3_vitl_ytvis_hq_slot512_{ckpt.stem}",
                "short_name": f"DINOv3 ViT-L YTVIS-HQ slot512 {ckpt.stem}",
                "family": "dinov3",
                "variant": "vitl_ytvis_hq_slot512",
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

    for movic_dir in sorted(
        restart_root.glob("movi_c_transfer*/rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/42")
    ):
        ckpt = latest_checkpoint(movic_dir)
        if ckpt is None:
            continue
        run_name = movic_dir.parents[1].name
        specs.append(
            {
                "name": f"dinov3_vitl_movic_transfer_{run_name}_{ckpt.stem}",
                "short_name": f"DINOv3 ViT-L MOVi-C transfer {ckpt.stem}",
                "family": "dinov3",
                "variant": "vitl_movic_slot512_bbox_mlp",
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
                "short_name": f"DINOv3 ViT-S YTVIS-HQ official-dims {ckpt.stem}",
                "family": "dinov3",
                "variant": "vits_ytvis_hq_official_dims",
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
    return specs


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def worker_command(args: argparse.Namespace, spec_path: Path, model_dir: Path) -> list[str]:
    return [
        str(PYTHON_BIN),
        str(Path(__file__).resolve()),
        "--worker-spec",
        str(spec_path),
        "--output-dir",
        str(model_dir),
        "--device",
        str(args.device),
        "--num-frames",
        str(args.num_frames),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--xssc-input-size",
        str(args.xssc_input_size),
        "--xssc-batch-size",
        str(args.xssc_batch_size),
        "--raft-iters",
        str(args.raft_iters),
        "--seed",
        str(args.seed),
        "--max-cases",
        str(args.max_cases),
        *sum((["--json", item] for item in (args.json or [])), []),
        *(["--skip-raft"] if args.skip_raft else []),
    ]


def run_orchestrator(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and args.force:
        subprocess.run(["rm", "-rf", str(output_dir)], check=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_dir = output_dir / "_specs"
    model_root = output_dir / "models"
    spec_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    specs = discover_dinov3_specs(args.restart_root.expanduser().resolve())
    if not specs:
        raise RuntimeError(f"No DINOv3 xSSC checkpoints found under {args.restart_root}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(PACKAGE_PARENT),
            str(PROJECT_ROOT),
            str(TRAIN_XSSC_ROOT),
            str(ROOT),
            "/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main",
            env.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)

    results = []
    for spec in specs:
        spec["safe_name"] = safe_name(spec["name"])
        spec_path = spec_dir / f"{spec['safe_name']}.json"
        spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        model_dir = model_root / spec["safe_name"]
        if model_dir.exists() and args.force:
            subprocess.run(["rm", "-rf", str(model_dir)], check=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        log_path = model_dir / "worker.log"
        command = worker_command(args, spec_path, model_dir)
        print(f"[run] {spec['short_name']} -> {model_dir}", flush=True)
        print(f"[cmd] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} {shlex.join(command)}", flush=True)
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log_path.write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            print(proc.stdout, flush=True)
            raise RuntimeError(f"worker failed for {spec['name']}; see {log_path}")
        results.append(json.loads((model_dir / "metadata.json").read_text(encoding="utf-8")))

    metadata = build_report(output_dir, results, args)
    build_html(output_dir, metadata)
    print(f"viewer={output_dir / 'index.html'}", flush=True)


def summarize_official_reference(reference_dir: Path) -> dict[str, Any] | None:
    metadata_path = reference_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    rows = metadata.get("verdict_summary", [])
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("level", "unknown"))] = counts.get(str(row.get("level", "unknown")), 0) + 1
    metrics = ["residual_track_cos", "d_adj_spearman", "d_pair_spearman", "centroid_distance"]
    means = {}
    for metric in metrics:
        values = [float(row[metric]) for row in rows if isinstance(row.get(metric), (int, float))]
        means[metric] = sum(values) / len(values) if values else None
    return {
        "dir": str(reference_dir),
        "index": str(reference_dir / "index.html"),
        "num_rows": len(rows),
        "level_counts": counts,
        "mean_metrics": means,
    }


def build_report(output_dir: Path, results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for result in results:
        model = result["model"]
        for case in result["cases"]:
            verdict = case["record"]["verdict"]
            metrics = verdict.get("metrics", {})
            rows.append(
                {
                    "case_id": case["case_id"],
                    "source_video": case["source_video"],
                    "model_name": model["name"],
                    "model_short_name": model["short_name"],
                    "level": verdict["level"],
                    "top_pair": verdict.get("top_pair", []),
                    "selected_boxes": case.get("selected_boxes", 0),
                    "num_slots": model["num_slots"],
                    "slot_dim": model["slot_dim"],
                    "initializer": model["initializer"],
                    "residual_track_cos": float(metrics.get("residual_track_cos", float("nan"))),
                    "d_adj_spearman": float(metrics.get("d_adj_spearman", float("nan"))),
                    "d_pair_spearman": float(metrics.get("d_pair_spearman", float("nan"))),
                    "centroid_distance": float(metrics.get("centroid_distance", float("nan"))),
                }
            )
    model_summaries = []
    for result in results:
        model = result["model"]
        model_rows = [row for row in rows if row["model_name"] == model["name"]]
        counts: dict[str, int] = {}
        for row in model_rows:
            counts[row["level"]] = counts.get(row["level"], 0) + 1
        means = {}
        for metric in ("residual_track_cos", "d_adj_spearman", "d_pair_spearman", "centroid_distance"):
            values = [float(row[metric]) for row in model_rows if str(row[metric]) != "nan"]
            means[metric] = sum(values) / len(values) if values else None
        model_summaries.append({"model": model, "level_counts": counts, "mean_metrics": means})

    metadata = {
        "args": {
            "restart_root": str(args.restart_root),
            "num_frames": args.num_frames,
            "preprocess": f"source -> cover_crop {args.height}x{args.width} -> center square -> {args.xssc_input_size}x{args.xssc_input_size}",
            "raft_iters": args.raft_iters,
            "skip_raft": bool(args.skip_raft),
        },
        "results": results,
        "verdict_summary": rows,
        "model_summaries": model_summaries,
        "official_reference": summarize_official_reference(args.official_reference_dir.expanduser().resolve()),
        "note": (
            "DINOv3 source-video analysis reuses the same no-GT evidence labels as "
            "analyze_xssc_object_slot_separation_cases.py. MOVi-C bbox_mlp weights use "
            "AMG pseudo boxes from the xSSC 256x256 first frame; YTVIS variants use NormalShared initialization."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def build_html(output_dir: Path, metadata: dict[str, Any]) -> None:
    model_rows = []
    for item in metadata["model_summaries"]:
        model = item["model"]
        means = item["mean_metrics"]
        model_rows.append(
            "<tr>"
            f"<td>{html.escape(model['short_name'])}</td>"
            f"<td>{html.escape(model['family'])}<br><span class='small'>{html.escape(model.get('variant', model['family']))}</span></td>"
            f"<td>{html.escape(str([model['num_frames'], model['num_slots'], model['slot_dim']]))}</td>"
            f"<td>{html.escape(model['initializer'])}</td>"
            f"<td>{html.escape(str(item['level_counts']))}</td>"
            f"<td>{means['residual_track_cos']:.3f}</td>"
            f"<td>{means['d_adj_spearman']:.3f}</td>"
            f"<td>{means['d_pair_spearman']:.3f}</td>"
            f"<td>{means['centroid_distance']:.3f}</td>"
            f"<td><code>{html.escape(model['checkpoint'])}</code></td>"
            "</tr>"
        )

    summary_rows = []
    for row in metadata["verdict_summary"]:
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(row['case_id'])}</td>"
            f"<td>{html.escape(row['model_short_name'])}</td>"
            f"<td>{html.escape(row['level'])}</td>"
            f"<td>{html.escape(str(row['top_pair']))}</td>"
            f"<td>{int(row['selected_boxes'])}</td>"
            f"<td>{row['residual_track_cos']:.3f}</td>"
            f"<td>{row['d_adj_spearman']:.3f}</td>"
            f"<td>{row['d_pair_spearman']:.3f}</td>"
            f"<td>{row['centroid_distance']:.3f}</td>"
            "</tr>"
        )

    cases_by_id: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for result in metadata["results"]:
        model = result["model"]
        for case in result["cases"]:
            cases_by_id.setdefault(case["case_id"], []).append((model, case))

    case_sections = []
    for case_id, entries in cases_by_id.items():
        first_case = entries[0][1]
        cards = []
        for model, case in entries:
            record = case["record"]
            verdict = record["verdict"]
            metrics = verdict.get("metrics", {})
            model_base = Path("models") / model["safe_name"] / case_id
            assets = record["assets"]
            cards.append(
                f"""
                <article class="model {html.escape(verdict['level'])}">
                  <h3>{html.escape(model['short_name'])} | {html.escape(verdict['level'])}</h3>
                  <p class="small">shape=[{model['num_frames']},{model['num_slots']},{model['slot_dim']}] init={html.escape(model['initializer'])} selected_boxes={case.get('selected_boxes', 0)}</p>
                  <p class="small">top_pair={html.escape(str(verdict.get('top_pair', [])))} residual={metrics.get('residual_track_cos', float('nan')):.3f} D_adj={metrics.get('d_adj_spearman', float('nan')):.3f} D_pair={metrics.get('d_pair_spearman', float('nan')):.3f} centroid={metrics.get('centroid_distance', float('nan')):.3f}</p>
                  <div class="videos">
                    <figure><video src="{model_base / assets['all_slot_overlay']}" controls muted preload="metadata"></video><figcaption>all-slot hard assignment overlay</figcaption></figure>
                    <figure><video src="{model_base / assets['per_slot_grid_overlay']}" controls muted preload="metadata"></video><figcaption>per-slot overlay grid</figcaption></figure>
                  </div>
                  <div class="plots">
                    <figure><img src="{model_base / assets['slot_dynamics_curves']}" loading="lazy"><figcaption>D_adj / RAFT slot-flow / centroid speed</figcaption></figure>
                    <figure><img src="{model_base / assets['residual_track_cos']}" loading="lazy"><figcaption>feature residual track similarity</figcaption></figure>
                    <figure><img src="{model_base / assets['d_adj_corr']}" loading="lazy"><figcaption>D_adj similarity</figcaption></figure>
                    <figure><img src="{model_base / assets['d_pair_corr']}" loading="lazy"><figcaption>D_pair similarity</figcaption></figure>
                    <figure><img src="{model_base / assets['centroid_distance']}" loading="lazy"><figcaption>slot centroid distance</figcaption></figure>
                    <figure><img src="{model_base / assets['d_pair_matrices']}" loading="lazy"><figcaption>D_pair matrices</figcaption></figure>
                  </div>
                </article>
                """
            )
        case_sections.append(
            f"""
            <section class="case">
              <h2>{html.escape(case_id)}</h2>
              <p class="small"><b>source:</b> {html.escape(first_case['source_video'])}</p>
              <p class="small"><b>caption:</b> {html.escape(first_case.get('caption', ''))}</p>
              <div class="models">{''.join(cards)}</div>
            </section>
            """
        )

    official = metadata.get("official_reference")
    official_html = ""
    if official:
        official_html = (
            "<p class='small'>Official DINOv2 reference page from the previous run: "
            f"<code>{html.escape(official['index'])}</code> | "
            f"counts={html.escape(str(official['level_counts']))} | "
            f"means={html.escape(str(official['mean_metrics']))}</p>"
        )

    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DINOv3 xSSC Slot Object/Motion Separation</title>
  <style>
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:#101214; color:#eef2f7; font:13px system-ui,sans-serif; letter-spacing:0; }}
    header {{ position:sticky; top:0; z-index:10; padding:12px 16px; background:#15191d; border-bottom:1px solid #303942; }}
    main {{ max-width:2300px; margin:0 auto; padding:16px; }}
    h1 {{ margin:0 0 6px; font-size:20px; }}
    h2 {{ margin:0 0 8px; font-size:17px; }}
    h3 {{ margin:0 0 6px; font-size:14px; }}
    code {{ color:#d5f5ff; }}
    .small {{ color:#bdc7d1; overflow-wrap:anywhere; }}
    .case {{ padding:18px 0 30px; border-top:1px solid #303942; }}
    .models {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(620px,1fr)); gap:12px; }}
    .model {{ border:1px solid #333b44; background:#14191e; padding:10px; border-radius:8px; }}
    .model.strong {{ border-color:#42d392; }}
    .model.partial {{ border-color:#f7c948; }}
    .model.merge-risk,.model.weak {{ border-color:#ff6b6b; }}
    .videos,.plots {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }}
    figure {{ margin:0; min-width:0; }}
    img,video {{ display:block; width:100%; background:#000; border:1px solid #303942; }}
    figcaption {{ color:#aeb8c2; font-size:11px; padding:4px 1px; }}
    table {{ width:100%; border-collapse:collapse; margin:8px 0 16px; }}
    th,td {{ border:1px solid #303942; padding:5px 7px; text-align:left; vertical-align:top; }}
    th {{ background:#192027; }}
    td {{ background:#12171c; color:#cbd5df; }}
    @media(max-width:900px) {{ .models,.videos,.plots {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>DINOv3 xSSC slot object/motion separation</h1>
    <div class="small">{html.escape(metadata['args']['preprocess'])}; no GT masks are used for verdicts.</div>
  </header>
  <main>
    {official_html}
    <h2>Model Summary</h2>
    <table>
      <thead><tr><th>method</th><th>family</th><th>slot shape</th><th>initializer</th><th>level counts</th><th>mean feature cos</th><th>mean D_adj</th><th>mean D_pair</th><th>mean centroid</th><th>checkpoint</th></tr></thead>
      <tbody>{''.join(model_rows)}</tbody>
    </table>
    <h2>Verdict Summary</h2>
    <table>
      <thead><tr><th>case</th><th>method</th><th>level</th><th>top pair</th><th>AMG boxes</th><th>feature cos</th><th>D_adj</th><th>D_pair</th><th>centroid</th></tr></thead>
      <tbody>{''.join(summary_rows)}</tbody>
    </table>
    {''.join(case_sections)}
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(text, encoding="utf-8")


def run_worker(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np
    import torch

    for item in (PACKAGE_PARENT, PROJECT_ROOT, TRAIN_XSSC_ROOT, ROOT):
        text = str(item)
        if text not in sys.path:
            sys.path.insert(0, text)
    import analyze_official_xssc_dynamics_raft as base
    import analyze_xssc_object_slot_separation_cases as objsep
    import run_xssc_slot_dedup_weight_compare as compare

    spec = json.loads(args.worker_spec.read_text(encoding="utf-8"))
    spec["safe_name"] = safe_name(spec["name"])
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, slot_dim, num_slots, initializer = compare.load_xssc_variant(spec, device)
    raft = None if args.skip_raft else base.build_raft(device, args.raft_iters)
    json_paths = args.json if args.json else objsep.DEFAULT_JSONS
    cases, duplicates = objsep.read_cases(json_paths, args.max_cases)

    model_records = []
    for case_position, case in enumerate(cases, start=1):
        case_dir = output_dir / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        video_tensor, _, frame_indices = objsep.read_source_video(
            Path(case["source_video"]),
            args.num_frames,
            args.height,
            args.width,
        )
        normalized, rgb = base.preprocess_video_for_xssc(video_tensor, args.xssc_input_size)
        base.write_video(case_dir / "xssc_input_49f.mp4", rgb, fps=8.0)
        flow = None if raft is None else base.compute_raft_flow(raft, rgb, device, args.raft_iters)
        boxes = None
        selected_boxes = 0
        if initializer == "bbox_mlp":
            boxes, selected_boxes = build_amg_boxes(normalized[None].to(device), num_slots)
        seed = int(args.seed) + case_position * 1000 + natural_step(Path(spec["xssc_checkpoint"]))
        slots, attention = extract_variant_slots(
            model,
            normalized,
            device=device,
            seed=seed,
            batch_size=args.xssc_batch_size,
            initializer=initializer,
            boxes=boxes,
        )
        record = objsep.render_case_weight(
            case_dir,
            "xssc",
            rgb,
            slots.numpy().astype(np.float32),
            attention.numpy().astype(np.float32),
            flow,
        )
        record["checkpoint"] = spec["xssc_checkpoint"]
        record["initializer"] = initializer
        record["selected_boxes"] = int(selected_boxes)
        model_records.append(
            {
                **case,
                "frame_indices": [int(v) for v in frame_indices.tolist()],
                "selected_boxes": int(selected_boxes),
                "record": record,
            }
        )
        verdict = record["verdict"]
        print(
            f"[case] {case_position}/{len(cases)} {case['case_id']} "
            f"level={verdict['level']} pair={verdict.get('top_pair', [])} "
            f"boxes={selected_boxes}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata = {
        "model": {
            "name": spec["name"],
            "safe_name": spec["safe_name"],
            "short_name": spec["short_name"],
            "family": spec["family"],
            "variant": spec.get("variant", spec["family"]),
            "checkpoint": spec["xssc_checkpoint"],
            "config": spec["xssc_config"],
            "root": spec["xssc_root"],
            "dinov3_checkpoint": spec["dinov3_checkpoint"],
            "initializer": initializer,
            "num_frames": int(args.num_frames),
            "num_slots": int(num_slots),
            "slot_dim": int(slot_dim),
        },
        "cases": model_records,
        "duplicates_skipped": duplicates,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_amg_boxes(video: Any, num_slots: int) -> tuple[Any, int]:
    import train_xssc_object_self_attn_lora as object_train
    import run_xssc_slot_dedup_weight_compare as compare

    defaults = compare.SimpleNamespaceFromDefaults(object_train)
    builder = object_train.AMGBoxBuilder(
        sam2_config=object_train.DEFAULT_SAM2_CONFIG,
        sam2_checkpoint=object_train.DEFAULT_SAM2_CHECKPOINT,
        cache_dir="/data/gaoya/agent-data/cache/xssc_object_slot_separation_dinov3_latest_amg",
        filter_args=object_train._amg_filter_args_from_args(defaults),
    )
    boxes = builder(video, num_slots)
    selected = int(builder.last_selected_counts[0]) if builder.last_selected_counts else 0
    return boxes, selected


def extract_variant_slots(
    model: Any,
    normalized: Any,
    *,
    device: Any,
    seed: int,
    batch_size: int,
    initializer: str,
    boxes: Any | None,
) -> tuple[Any, Any]:
    import torch

    model.eval()
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    features = []
    with torch.inference_mode():
        for start in range(0, len(normalized), batch_size):
            batch = normalized[start : start + batch_size].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                feature = model.encode_backbone(batch).detach()
            features.append(feature.to(device=device, dtype=torch.bfloat16))
        feature = torch.cat(features, dim=0)

        encoded_parts = []
        for start in range(0, len(feature), batch_size):
            current = feature[start : start + batch_size]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                encoded = current.permute(0, 2, 3, 1)
                encoded = model.encode_posit_embed(encoded).flatten(1, 2)
                encoded = model.encode_project(encoded)
            encoded_parts.append(encoded)
        encoded_all = torch.cat(encoded_parts, dim=0)[None]

        if boxes is not None:
            boxes = boxes.to(device=device, dtype=encoded_all.dtype)
        slots = None
        slot_parts = []
        attn_parts = []
        for frame_id in range(encoded_all.shape[1]):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if frame_id == 0:
                    if initializer == "bbox_mlp":
                        if boxes is None:
                            raise RuntimeError("bbox_mlp initializer requires AMG pseudo boxes")
                        query = model.initializ(boxes[:, 0])
                    else:
                        query = model.initializ(1)
                else:
                    query = model.transit(slots, encoded_all[:, : frame_id + 1])
                current_slots, current_attention = model.aggregat(
                    encoded_all[:, frame_id],
                    query,
                    num_iter=None if frame_id == 0 else 1,
                )
            slots = current_slots[:, None] if slots is None else torch.cat((slots, current_slots[:, None]), dim=1)
            slot_parts.append(current_slots[0].detach().float().cpu())
            attn_parts.append(current_attention[0].detach().float().cpu())

    slot_tensor = torch.stack(slot_parts, dim=0)
    attention = torch.stack(attn_parts, dim=0)
    patch_side = int(round(attention.shape[-1] ** 0.5))
    attention = attention.view(attention.shape[0], attention.shape[1], patch_side, patch_side)
    return slot_tensor, attention


def main() -> None:
    args = parse_args()
    if args.worker_spec is not None:
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
