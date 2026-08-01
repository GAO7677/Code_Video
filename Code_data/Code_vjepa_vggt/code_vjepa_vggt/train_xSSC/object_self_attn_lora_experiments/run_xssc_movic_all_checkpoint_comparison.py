#!/usr/bin/env python3
"""Evaluate every MOVi-C DINOv3 xSSC checkpoint and rank slot separation."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
PYTHON_BIN = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
EVALUATOR = ROOT / "analyze_xssc_dinov3_object_slot_separation_cases.py"
DEFAULT_CHECKPOINT_DIR = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC/"
    "restart_save1000_20260720T140029Z/"
    "movi_c_transfer15000_b64_acc3_20260721T134713Z/"
    "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/42"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/"
    "xssc_object_slot_separation_cases_dinov3_latest/all_checkpoints"
)
METRICS = {
    "residual_track_cos": "lower",
    "d_adj_spearman": "lower",
    "d_pair_spearman": "lower",
    "centroid_distance": "higher",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gpus", default="2,3,7")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--xssc-input-size", type=int, default=256)
    parser.add_argument("--xssc-batch-size", type=int, default=16)
    parser.add_argument("--raft-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"step-(\d+)\.pth", path.name)
    if match is None:
        raise ValueError(f"Unexpected checkpoint name: {path.name}")
    return int(match.group(1))


def parse_gpus(text: str) -> list[int]:
    gpus = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    if 4 in gpus:
        raise ValueError("GPU 4 is disabled by workspace policy")
    if len(gpus) != len(set(gpus)):
        raise ValueError(f"Duplicate GPU ids: {gpus}")
    return gpus


def atomic_write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_worker_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        device="cuda:0",
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        xssc_input_size=args.xssc_input_size,
        xssc_batch_size=args.xssc_batch_size,
        raft_iters=args.raft_iters,
        seed=args.seed,
        max_cases=args.max_cases,
        json=None,
        skip_raft=False,
    )


def build_tasks(args: argparse.Namespace, evaluator: Any) -> list[dict[str, Any]]:
    checkpoints = sorted(args.checkpoint_dir.resolve().glob("step-*.pth"), key=checkpoint_step)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {args.checkpoint_dir}")
    tasks = []
    for checkpoint in checkpoints:
        specs = evaluator.discover_dinov3_specs(
            evaluator.DEFAULT_RESTART_ROOT.resolve(),
            checkpoint,
        )
        spec = next(item for item in specs if item.get("variant") == "vitl_movic_slot512_bbox_mlp")
        spec["safe_name"] = evaluator.safe_name(spec["name"])
        spec["step"] = checkpoint_step(checkpoint)
        tasks.append(spec)
    return tasks


def run_gpu_queue(
    gpu: int,
    tasks: list[dict[str, Any]],
    args: argparse.Namespace,
    evaluator: Any,
    status: dict[str, Any],
    status_path: Path,
    status_lock: threading.Lock,
) -> list[str]:
    failures = []
    worker_args = make_worker_args(args)
    output_dir = args.output_dir.resolve()
    for spec in tasks:
        model_dir = output_dir / "models" / spec["safe_name"]
        metadata_path = model_dir / "metadata.json"
        if metadata_path.is_file() and not args.force:
            with status_lock:
                status["checkpoints"][str(spec["step"])] = {"state": "reused", "gpu": gpu}
                atomic_write_json(status_path, status)
            print(f"[reuse][gpu{gpu}] step-{spec['step']:06d}", flush=True)
            continue
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        spec_path = output_dir / "specs" / f"{spec['safe_name']}.json"
        atomic_write_json(spec_path, spec)
        command = evaluator.worker_command(worker_args, spec_path, model_dir)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(evaluator.PACKAGE_PARENT),
                str(evaluator.PROJECT_ROOT),
                str(evaluator.TRAIN_XSSC_ROOT),
                str(evaluator.ROOT),
                "/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main",
                env.get("PYTHONPATH", ""),
            ]
        ).rstrip(os.pathsep)
        with status_lock:
            status["checkpoints"][str(spec["step"])] = {"state": "running", "gpu": gpu}
            atomic_write_json(status_path, status)
        print(f"[run][gpu{gpu}] step-{spec['step']:06d}", flush=True)
        log_path = model_dir / "worker.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                command,
                cwd=str(ROOT),
                env=env,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        state = "complete" if proc.returncode == 0 and metadata_path.is_file() else "failed"
        with status_lock:
            status["checkpoints"][str(spec["step"])] = {
                "state": state,
                "gpu": gpu,
                "log": str(log_path),
            }
            atomic_write_json(status_path, status)
        print(f"[{state}][gpu{gpu}] step-{spec['step']:06d}", flush=True)
        if state == "failed":
            failures.append(f"step-{spec['step']:06d}")
    return failures


def percentile_utilities(values: dict[str, float], direction: str) -> dict[str, float]:
    transformed = {key: value if direction == "higher" else -value for key, value in values.items()}
    if len(transformed) == 1:
        return {next(iter(transformed)): 1.0}
    output = {}
    all_values = list(transformed.values())
    denominator = len(all_values) - 1
    for key, value in transformed.items():
        lower = sum(other < value for other in all_values)
        equal = sum(other == value for other in all_values)
        output[key] = (lower + 0.5 * (equal - 1)) / denominator
    return output


def aggregate_results(
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
    evaluator: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_dir = args.output_dir.resolve()
    results = []
    for spec in tasks:
        metadata_path = output_dir / "models" / spec["safe_name"] / "metadata.json"
        if metadata_path.is_file():
            results.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    if len(results) != len(tasks):
        raise RuntimeError(f"Only {len(results)}/{len(tasks)} checkpoint results are complete")

    report_args = argparse.Namespace(
        restart_root=evaluator.DEFAULT_RESTART_ROOT,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        xssc_input_size=args.xssc_input_size,
        raft_iters=args.raft_iters,
        skip_raft=False,
        official_reference_dir=evaluator.DEFAULT_OFFICIAL_REFERENCE,
    )
    metadata = evaluator.build_report(output_dir, results, report_args)

    rows_by_model_case: dict[str, dict[str, dict[str, Any]]] = {}
    model_by_name = {}
    for result in results:
        model = result["model"]
        model_by_name[model["name"]] = model
        rows_by_model_case[model["name"]] = {}
        for case in result["cases"]:
            metrics = case["record"]["verdict"]["metrics"]
            rows_by_model_case[model["name"]][case["case_id"]] = {
                "level": case["record"]["verdict"]["level"],
                **{metric: float(metrics[metric]) for metric in METRICS},
            }

    model_names = list(rows_by_model_case)
    case_ids = sorted(next(iter(rows_by_model_case.values())))
    per_model_case_scores = {name: {case_id: [] for case_id in case_ids} for name in model_names}
    for case_id in case_ids:
        for metric, direction in METRICS.items():
            values = {name: rows_by_model_case[name][case_id][metric] for name in model_names}
            utilities = percentile_utilities(values, direction)
            for name, utility in utilities.items():
                per_model_case_scores[name][case_id].append(utility)

    rng = np.random.default_rng(args.seed)
    ranking = []
    for name in model_names:
        model = model_by_name[name]
        case_scores = np.asarray(
            [np.mean(per_model_case_scores[name][case_id]) for case_id in case_ids],
            dtype=np.float64,
        )
        bootstrap = np.empty(args.bootstrap_samples, dtype=np.float64)
        for index in range(args.bootstrap_samples):
            bootstrap[index] = rng.choice(case_scores, size=len(case_scores), replace=True).mean()
        levels = [rows_by_model_case[name][case_id]["level"] for case_id in case_ids]
        means = {
            metric: float(np.mean([rows_by_model_case[name][case_id][metric] for case_id in case_ids]))
            for metric in METRICS
        }
        ranking.append(
            {
                "step": checkpoint_step(Path(model["checkpoint"])),
                "model_name": name,
                "safe_name": model["safe_name"],
                "checkpoint": model["checkpoint"],
                "score": float(case_scores.mean()),
                "score_ci95_low": float(np.quantile(bootstrap, 0.025)),
                "score_ci95_high": float(np.quantile(bootstrap, 0.975)),
                "strong": levels.count("strong"),
                "partial": levels.count("partial"),
                "merge_risk": levels.count("merge-risk") + levels.count("weak"),
                **means,
            }
        )
    score_matrix = np.asarray(
        [
            [np.mean(per_model_case_scores[name][case_id]) for case_id in case_ids]
            for name in model_names
        ],
        dtype=np.float64,
    )
    joint_rng = np.random.default_rng(args.seed + 1)
    bootstrap_win_counts = np.zeros(len(model_names), dtype=np.int64)
    for _ in range(args.bootstrap_samples):
        sampled_cases = joint_rng.integers(0, len(case_ids), size=len(case_ids))
        winner = int(score_matrix[:, sampled_cases].mean(axis=1).argmax())
        bootstrap_win_counts[winner] += 1
    loo_win_counts = np.zeros(len(model_names), dtype=np.int64)
    for held_out in range(len(case_ids)):
        keep = [index for index in range(len(case_ids)) if index != held_out]
        winner = int(score_matrix[:, keep].mean(axis=1).argmax())
        loo_win_counts[winner] += 1
    item_by_name = {item["model_name"]: item for item in ranking}
    for model_index, name in enumerate(model_names):
        item_by_name[name]["bootstrap_win_probability"] = float(
            bootstrap_win_counts[model_index] / args.bootstrap_samples
        )
        item_by_name[name]["loo_win_count"] = int(loo_win_counts[model_index])

    score_order = sorted(ranking, key=lambda item: (-item["score"], -item["strong"], item["step"]))
    for index, item in enumerate(score_order, start=1):
        item["score_rank"] = index
        item["rank"] = index
    verdict_order = sorted(
        ranking,
        key=lambda item: (
            -item["strong"],
            item["merge_risk"],
            -item["score"],
            item["step"],
        )
    )
    for index, item in enumerate(verdict_order, start=1):
        item["verdict_rank"] = index
    ranking = score_order
    metadata["selection"] = {
        "best_checkpoint": ranking[0],
        "best_continuous_score": score_order[0],
        "best_verdict_count": verdict_order[0],
        "recommendation_rule": (
            "Recommend the highest continuous rank-aggregation score. Report the hard-verdict "
            "leader separately because threshold crossings and top-pair changes are discontinuous."
        ),
        "score_definition": (
            "For each source case and metric, checkpoints receive a [0,1] percentile utility. "
            "Lower is better for residual_track_cos, d_adj_spearman, and d_pair_spearman; "
            "higher is better for centroid_distance. The final score is the equal-weight mean "
            "over four metrics and all cases. CI95 bootstraps source cases."
        ),
        "ranking": ranking,
    }
    atomic_write_json(output_dir / "metadata.json", metadata)
    atomic_write_json(output_dir / "checkpoint_ranking.json", metadata["selection"])
    write_ranking_csv(output_dir / "checkpoint_ranking.csv", ranking)
    plot_curves(output_dir / "checkpoint_curves.png", ranking)
    build_html(output_dir, metadata, rows_by_model_case)
    return metadata, ranking


def write_ranking_csv(path: Path, ranking: list[dict[str, Any]]) -> None:
    fields = [
        "rank", "verdict_rank", "step", "score", "score_ci95_low", "score_ci95_high",
        "bootstrap_win_probability", "loo_win_count",
        "strong", "partial", "merge_risk", *METRICS.keys(), "checkpoint",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranking)


def plot_curves(path: Path, ranking: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(ranking, key=lambda item: item["step"])
    steps = [item["step"] for item in ordered]
    best_step = ranking[0]["step"]
    panels = [
        ("score", "Rank aggregation score (higher better)"),
        ("residual_track_cos", "Residual track cosine (lower better)"),
        ("d_adj_spearman", "D_adj correlation (lower better)"),
        ("d_pair_spearman", "D_pair correlation (lower better)"),
        ("centroid_distance", "Centroid distance (higher better)"),
        ("strong", "Strong verdict count (higher better)"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, panels):
        axis.plot(steps, [item[metric] for item in ordered], marker="o", markersize=3, linewidth=1.2)
        axis.axvline(best_step, color="#d62728", linestyle="--", linewidth=1, label=f"best {best_step}")
        axis.set_title(title)
        axis.set_xlabel("checkpoint step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def build_html(
    output_dir: Path,
    metadata: dict[str, Any],
    rows_by_model_case: dict[str, dict[str, dict[str, Any]]],
) -> None:
    ranking = metadata["selection"]["ranking"]
    best = ranking[0]
    table_rows = []
    for item in ranking:
        class_name = "best" if item["rank"] == 1 else ""
        table_rows.append(
            f"<tr class='{class_name}'><td>{item['rank']}</td><td>{item['verdict_rank']}</td><td>{item['step']}</td>"
            f"<td>{item['score']:.4f}</td><td>[{item['score_ci95_low']:.4f}, {item['score_ci95_high']:.4f}]</td>"
            f"<td>{item['bootstrap_win_probability']:.3f}</td><td>{item['loo_win_count']}/9</td>"
            f"<td>{item['strong']}</td><td>{item['partial']}</td><td>{item['merge_risk']}</td>"
            f"<td>{item['residual_track_cos']:.4f}</td><td>{item['d_adj_spearman']:.4f}</td>"
            f"<td>{item['d_pair_spearman']:.4f}</td><td>{item['centroid_distance']:.4f}</td></tr>"
        )

    result_by_name = {result["model"]["name"]: result for result in metadata["results"]}
    item_by_step = {item["step"]: item for item in ranking}
    compare_steps = []
    for step in (best["step"], 39000, 50000):
        if step in item_by_step and step not in compare_steps:
            compare_steps.append(step)
    case_ids = sorted(next(iter(rows_by_model_case.values())))
    comparison_sections = []
    for case_id in case_ids:
        cards = []
        for step in compare_steps:
            item = item_by_step[step]
            model_base = Path("models") / item["safe_name"] / case_id / "xssc"
            case_row = rows_by_model_case[item["model_name"]][case_id]
            cards.append(
                f"<figure><h4>step {step} | {html.escape(case_row['level'])}</h4>"
                f"<video src='{model_base / 'all_slot_overlay.mp4'}' controls muted preload='metadata'></video>"
                f"<figcaption>res={case_row['residual_track_cos']:.3f}, D_adj={case_row['d_adj_spearman']:.3f}, "
                f"D_pair={case_row['d_pair_spearman']:.3f}, centroid={case_row['centroid_distance']:.3f}</figcaption></figure>"
            )
        comparison_sections.append(
            f"<section><h3>{html.escape(case_id)}</h3><div class='comparison'>{''.join(cards)}</div></section>"
        )

    viewer_data = {}
    for item in ranking:
        result = result_by_name[item["model_name"]]
        viewer_data[str(item["step"])] = {
            case["case_id"]: {
                "base": str(Path("models") / item["safe_name"] / case["case_id"] / "xssc"),
                "level": case["record"]["verdict"]["level"],
            }
            for case in result["cases"]
        }
    step_options = "".join(f"<option value='{item['step']}'>step {item['step']}</option>" for item in ranking)
    case_options = "".join(f"<option value='{html.escape(case_id)}'>{html.escape(case_id)}</option>" for case_id in case_ids)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>xSSC MOVi-C checkpoint comparison</title><style>
*{{box-sizing:border-box}} body{{margin:0;background:#101214;color:#edf2f7;font:13px system-ui,sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:5;padding:12px 18px;background:#161a1e;border-bottom:1px solid #38414a}}
main{{max-width:1800px;margin:auto;padding:16px}} h1{{font-size:21px;margin:0 0 5px}} h2{{font-size:18px;margin:22px 0 8px}} h3{{font-size:14px;margin:15px 0 6px}} h4{{font-size:13px;margin:0 0 5px}}
.muted{{color:#b5c0ca}} .summary{{display:grid;grid-template-columns:minmax(300px,560px) 1fr;gap:16px;align-items:start}}
img,video{{display:block;width:100%;background:#000;border:1px solid #343d45}} figure{{margin:0;min-width:0}} figcaption{{color:#aeb9c3;padding:4px 0;font-size:11px}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:5px 7px;border:1px solid #343d45;text-align:right;white-space:nowrap}} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}} th{{background:#1b2229}} td{{background:#13181d}} tr.best td{{background:#17362b;color:#dcfff0}}
.comparison{{display:grid;grid-template-columns:repeat({len(compare_steps)},minmax(0,1fr));gap:10px}} .viewer-controls{{display:flex;gap:10px;margin-bottom:10px}} select{{background:#181e24;color:#eef;border:1px solid #46515c;padding:7px}}
.viewer-videos,.viewer-plots{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}} .viewer-plots{{margin-top:10px;grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:900px){{.summary,.comparison,.viewer-videos,.viewer-plots{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>xSSC MOVi-C all-checkpoint comparison</h1>
<div class="muted">35 checkpoints, fixed 9 cases / AMG boxes / seed / preprocessing / RAFT. Recommended: step {best['step']} (score {best['score']:.4f}).</div></header><main>
<div class="summary"><div><h2>Selection</h2><p>{html.escape(metadata['selection']['score_definition'])}</p>
<p>{html.escape(metadata['selection']['recommendation_rule'])}</p>
<p><b>Best checkpoint:</b> <code>{html.escape(best['checkpoint'])}</code></p></div><img src="checkpoint_curves.png"></div>
<h2>Ranking</h2><div style="overflow:auto"><table><thead><tr><th>score rank</th><th>verdict rank</th><th>step</th><th>score</th><th>case-bootstrap CI95</th><th>bootstrap win</th><th>LOO wins</th><th>strong</th><th>partial</th><th>merge-risk</th><th>residual cos</th><th>D_adj</th><th>D_pair</th><th>centroid</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<h2>Checkpoint Viewer</h2><div class="viewer-controls"><select id="step">{step_options}</select><select id="case">{case_options}</select></div>
<div class="viewer-videos"><video id="allSlot" controls muted preload="metadata"></video><video id="perSlot" controls muted preload="metadata"></video></div>
<div class="viewer-plots"><img id="curves"><img id="residual"><img id="dadj"><img id="dpair"><img id="centroid"><img id="matrices"></div>
<h2>Best vs step 39000 / 50000</h2>{''.join(comparison_sections)}
</main><script>
const data={json.dumps(viewer_data, ensure_ascii=False)}; const step=document.getElementById('step'); const caseSel=document.getElementById('case');
function refresh(){{const item=data[step.value][caseSel.value]; const base=item.base; document.getElementById('allSlot').src=base+'/all_slot_overlay.mp4'; document.getElementById('perSlot').src=base+'/per_slot_grid_overlay.mp4'; document.getElementById('curves').src=base+'/slot_dynamics_curves.png'; document.getElementById('residual').src=base+'/residual_track_cos.png'; document.getElementById('dadj').src=base+'/d_adj_corr.png'; document.getElementById('dpair').src=base+'/d_pair_corr.png'; document.getElementById('centroid').src=base+'/centroid_distance.png'; document.getElementById('matrices').src=base+'/d_pair_matrices.png';}}
step.addEventListener('change',refresh); caseSel.addEventListener('change',refresh); refresh();
</script></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    gpus = parse_gpus(args.gpus)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import analyze_xssc_dinov3_object_slot_separation_cases as evaluator

    tasks = build_tasks(args, evaluator)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)
    (args.output_dir / "specs").mkdir(exist_ok=True)
    status_path = args.output_dir / "status.json"
    status = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "output_dir": str(args.output_dir),
        "gpus": gpus,
        "total": len(tasks),
        "checkpoints": {},
    }
    if status_path.is_file():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        status["checkpoints"] = previous.get("checkpoints", {})
    atomic_write_json(status_path, status)
    manifest = {
        "checkpoint_count": len(tasks),
        "steps": [task["step"] for task in tasks],
        "fixed_seed_policy": "seed + case_position * 1000; checkpoint step excluded",
        "args": vars(args),
    }
    manifest["args"] = {key: str(value) if isinstance(value, Path) else value for key, value in manifest["args"].items()}
    atomic_write_json(args.output_dir / "run_manifest.json", manifest)

    queues = [[] for _ in gpus]
    for index, task in enumerate(tasks):
        queues[index % len(gpus)].append(task)
    status_lock = threading.Lock()
    failures = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(run_gpu_queue, gpu, queue, args, evaluator, status, status_path, status_lock)
            for gpu, queue in zip(gpus, queues)
        ]
        for future in as_completed(futures):
            failures.extend(future.result())
    if failures:
        raise RuntimeError(f"Checkpoint workers failed: {failures}")
    metadata, ranking = aggregate_results(args, tasks, evaluator)
    print(f"[best] step-{ranking[0]['step']:06d} score={ranking[0]['score']:.6f}", flush=True)
    print(f"[viewer] {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
