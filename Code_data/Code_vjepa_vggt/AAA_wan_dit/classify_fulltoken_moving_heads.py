#!/usr/bin/env python3
"""Classify Wan attention heads from full-token and moving-trajectory evidence."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


MODELS = ("wan_lora", "xssc", "physrvg")
MODEL_NAMES = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
ROLES = ("S", "T", "P", "C", "G")
ROLE_NAMES = {
    "S": "空间局部候选",
    "T": "运动轨迹候选",
    "P": "固定位置候选",
    "C": "上下文候选",
    "G": "全局候选",
    "M": "混合/不稳定",
}
ROLE_NAMES_EN = {
    "S": "Spatial",
    "T": "Trajectory",
    "P": "Position",
    "C": "Context",
    "G": "Global",
    "M": "Mixed",
}
ROLE_COLORS = {
    "S": "#00796b",
    "T": "#d1495b",
    "P": "#6a4c93",
    "C": "#e19c24",
    "G": "#3d6fb6",
    "M": "#9a9a96",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=Path(
            "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/capture"
        ),
    )
    parser.add_argument(
        "--query-root",
        type=Path,
        default=Path(
            "/data/gaoya/agent-data/outputs/wan_dit_paired_query_50seeds/query_maps"
        ),
    )
    parser.add_argument(
        "--pass1-root",
        type=Path,
        default=Path(
            "/data/gaoya/agent-data/outputs/wan_dit_paired_query_50seeds/pass1"
        ),
    )
    parser.add_argument("--seed", type=int, default=851)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-blocks", type=int, default=30)
    parser.add_argument("--top-per-role", type=int, default=2)
    return parser.parse_args()


def _rank(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="stable")
    ranks = np.empty_like(flat, dtype=np.float64)
    ranks[order] = np.arange(len(flat), dtype=np.float64)
    if len(flat) > 1:
        ranks /= float(len(flat) - 1)
    return ranks.reshape(values.shape).astype(np.float32)


def _find_video(root: Path, case: str) -> Path:
    matches = [
        path for path in root.rglob("*.mp4")
        if path.stem == case and "_runtime" not in path.parts
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one video for {case} under {root}, got {matches}")
    return matches[0]


def _load_sample(
    capture_root: Path,
    model: str,
    case: str,
    expected_blocks: int,
) -> dict[str, Any]:
    files = sorted(
        (capture_root / model).glob(
            f"block*/matrices/{model}/{case}/block*_fulltoken_moving.npz"
        )
    )
    if len(files) != expected_blocks:
        raise RuntimeError(
            f"{model}/{case}: found {len(files)} blocks, expected {expected_blocks}"
        )
    blocks = []
    for path in files:
        data = np.load(path, allow_pickle=False)
        block = int(path.name.split("_", 1)[0].replace("block", ""))
        blocks.append(
            {
                "block": block,
                "path": path,
                "steps": data["steps_one_based"].astype(int),
                "full_names": data["full_feature_names"].astype(str).tolist(),
                "object_names": data["object_feature_names"].astype(str).tolist(),
                "full": data["full_features"].astype(np.float32),
                "object_by_time": data[
                    "object_features_by_query_time"
                ].astype(np.float32),
            }
        )
    blocks.sort(key=lambda item: item["block"])
    steps = blocks[0]["steps"]
    if any(not np.array_equal(item["steps"], steps) for item in blocks):
        raise RuntimeError(f"{model}/{case}: inconsistent denoise steps")
    full = np.stack([item["full"] for item in blocks], axis=1)
    object_by_time = np.stack(
        [item["object_by_time"] for item in blocks], axis=1
    )
    obj = object_by_time.mean(axis=3)
    object_names = blocks[0]["object_names"]
    for name in ("context_enrichment", "history_bias"):
        feature_index = object_names.index(name)
        obj[..., feature_index] = object_by_time[
            ..., 2:, feature_index
        ].mean(axis=3)
    return {
        "model": model,
        "case": case,
        "steps": steps,
        "paths": [item["path"] for item in blocks],
        "full_names": blocks[0]["full_names"],
        "object_names": blocks[0]["object_names"],
        "full": full,
        "object": obj,
    }


def _classify(sample: dict[str, Any]) -> dict[str, Any]:
    full_index = {name: index for index, name in enumerate(sample["full_names"])}
    object_index = {
        name: index for index, name in enumerate(sample["object_names"])
    }
    full = sample["full"]
    obj = sample["object"]
    score_steps = []
    for step_index in range(full.shape[0]):
        f = full[step_index]
        o = obj[step_index]
        score_steps.append(
            np.stack(
                (
                    0.55 * _rank(f[..., full_index["local_enrichment"]])
                    + 0.45 * _rank(f[..., full_index["same_frame_mass"]]),
                    0.55 * _rank(o[..., object_index["trajectory_selectivity_log2"]])
                    + 0.25 * _rank(o[..., object_index["trajectory_enrichment"]])
                    + 0.20 * _rank(o[..., object_index["mean_time_distance"]]),
                    0.75 * _rank(o[..., object_index["fixed_position_enrichment"]])
                    + 0.25 * _rank(f[..., full_index["aligned_enrichment"]]),
                    0.55 * _rank(o[..., object_index["context_enrichment"]])
                    + 0.25 * _rank(f[..., full_index["context_enrichment"]])
                    + 0.20 * _rank(o[..., object_index["history_bias"]]),
                    0.60 * _rank(f[..., full_index["entropy"]])
                    + 0.25 * _rank(f[..., full_index["mean_time_distance"]])
                    + 0.15 * _rank(-f[..., full_index["same_frame_mass"]]),
                ),
                axis=-1,
            )
        )
    score_steps_array = np.stack(score_steps, axis=0)
    mean_scores = score_steps_array.mean(axis=0)
    winner = mean_scores.argmax(axis=-1)
    sorted_scores = np.sort(mean_scores, axis=-1)
    margin = sorted_scores[..., -1] - sorted_scores[..., -2]
    step_winner = score_steps_array.argmax(axis=-1)
    consistency = np.mean(step_winner == winner[None, ...], axis=0)
    labels = np.asarray(ROLES, dtype="<U1")[winner]
    labels[(margin < 0.08) | (consistency < 0.75)] = "M"
    return {
        **sample,
        "score_steps": score_steps_array,
        "scores": mean_scores,
        "labels": labels,
        "margin": margin,
        "consistency": consistency,
        "full_mean": full.mean(axis=0),
        "object_mean": obj.mean(axis=0),
    }


def _render_role_grid(item: dict[str, Any], output: Path) -> None:
    labels = item["labels"]
    role_order = (*ROLES, "M")
    role_to_int = {role: index for index, role in enumerate(role_order)}
    values = np.vectorize(role_to_int.get)(labels)
    fig, axis = plt.subplots(figsize=(13.5, 7.0), constrained_layout=True)
    axis.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap([ROLE_COLORS[role] for role in role_order]),
        vmin=-0.5,
        vmax=len(role_order) - 0.5,
    )
    axis.set_xlabel("Head index")
    axis.set_ylabel("DiT block")
    axis.set_xticks(np.arange(labels.shape[1]))
    axis.set_yticks(np.arange(labels.shape[0]))
    axis.set_title(
        f"{MODEL_NAMES[item['model']]} | {item['case']} | 4-step stable role"
    )
    handles = [
        plt.Line2D(
            [0], [0], marker="s", linestyle="", markersize=9,
            color=ROLE_COLORS[role], label=f"{role}: {ROLE_NAMES_EN[role]}"
        )
        for role in role_order
    ]
    axis.legend(
        handles=handles,
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        frameon=False,
    )
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _top_heads(item: dict[str, Any], count: int) -> dict[str, list[tuple[int, int]]]:
    output = {}
    for role_index, role in enumerate(ROLES):
        score = item["scores"][..., role_index].copy()
        stable = item["consistency"] >= 0.75
        score[~stable] = -np.inf
        indices = np.argsort(score.reshape(-1))[::-1][:count]
        output[role] = [
            (int(index // score.shape[1]), int(index % score.shape[1]))
            for index in indices
        ]
    return output


def _render_head_evidence(
    item: dict[str, Any],
    *,
    block: int,
    head: int,
    output: Path,
) -> None:
    path = item["paths"][block]
    data = np.load(path, allow_pickle=False)
    steps = data["steps_one_based"].astype(int)
    temporal = data["temporal_matrix"][:, head]
    trajectory = data["trajectory_enrichment"][:, head]
    null_mean = 0.5 * (
        data["shift_enrichment"][:, head] + data["shuffle_enrichment"][:, head]
    )
    selectivity = np.log2(
        np.maximum(trajectory, 1.0e-8) / np.maximum(null_mean, 1.0e-8)
    )
    fig, axes = plt.subplots(
        len(steps), 2, figsize=(10.5, 3.0 * len(steps)), constrained_layout=True
    )
    for row, step in enumerate(steps):
        image0 = axes[row, 0].imshow(
            temporal[row], vmin=0.0, vmax=max(0.25, float(np.nanmax(temporal[row]))),
            cmap="viridis", aspect="equal"
        )
        axes[row, 0].set_title(f"step {step}: all-token temporal mass")
        image1 = axes[row, 1].imshow(
            selectivity[row], vmin=-2.0, vmax=2.0, cmap="coolwarm", aspect="equal"
        )
        axes[row, 1].set_title(f"step {step}: log2 trajectory/null")
        for axis in axes[row]:
            axis.set_xlabel("key latent time")
            axis.set_ylabel("query latent time")
        fig.colorbar(image0, ax=axes[row, 0], fraction=0.046)
        fig.colorbar(image1, ax=axes[row, 1], fraction=0.046)
    fig.suptitle(
        f"{MODEL_NAMES[item['model']]} | block {block:02d}, head {head:02d}",
        fontsize=13,
    )
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _metric(item: dict[str, Any], block: int, head: int, kind: str, name: str) -> float:
    names = item[f"{kind}_names"]
    values = item[f"{kind}_mean"]
    return float(values[block, head, names.index(name)])


def _copy_media(
    *,
    output_dir: Path,
    query_root: Path,
    pass1_root: Path,
    seed: int,
    model: str,
    case: str,
) -> tuple[str, str, dict[str, Any]]:
    query_path = query_root / model / f"seed-{seed:06d}" / "query_map.json"
    query_payload = json.loads(query_path.read_text(encoding="utf-8"))["cases"][case]
    media = output_dir / "media" / model / case
    media.mkdir(parents=True, exist_ok=True)
    preview = media / "query_preview.jpg"
    shutil.copy2(query_payload["preview"], preview)
    source_video = _find_video(
        pass1_root / model / f"seed-{seed:06d}" / "generated" / model,
        case,
    )
    video = media / "generated.mp4"
    shutil.copy2(source_video, video)
    return (
        str(video.relative_to(output_dir)),
        str(preview.relative_to(output_dir)),
        query_payload,
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        Path(line.strip()).stem
        for line in (
            Path(__file__).resolve().parent / "fulltoken_moving_pilot_cases.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    classified = []
    for model in MODELS:
        for case in cases:
            classified.append(
                _classify(
                    _load_sample(
                        args.capture_root.expanduser().resolve(),
                        model,
                        case,
                        args.expected_blocks,
                    )
                )
            )

    report: dict[str, Any] = {
        "method": {
            "ranking_scope": "within each model/case/denoise-step over all 30x24 heads",
            "aggregation": "mean role score over steps 5,15,25,35",
            "mixed_rule": "top-two margin < 0.08 or winner consistency < 0.75",
            "roles": ROLE_NAMES,
        },
        "samples": [],
    }
    sections = []
    for item in classified:
        model = item["model"]
        case = item["case"]
        slug = f"{model}__{case}"
        role_grid = output_dir / f"{slug}__roles.png"
        _render_role_grid(item, role_grid)
        top = _top_heads(item, int(args.top_per_role))
        evidence_rows = []
        top_payload = {}
        for role in ROLES:
            top_payload[role] = []
            for block, head in top[role]:
                evidence = output_dir / (
                    f"{slug}__{role}_b{block:02d}_h{head:02d}.png"
                )
                if not evidence.is_file():
                    _render_head_evidence(
                        item, block=block, head=head, output=evidence
                    )
                row = {
                    "role": role,
                    "block": block,
                    "head": head,
                    "score": float(item["scores"][block, head, ROLES.index(role)]),
                    "margin": float(item["margin"][block, head]),
                    "consistency": float(item["consistency"][block, head]),
                    "local_enrichment": _metric(
                        item, block, head, "full", "local_enrichment"
                    ),
                    "trajectory_enrichment": _metric(
                        item, block, head, "object", "trajectory_enrichment"
                    ),
                    "trajectory_selectivity_log2": _metric(
                        item, block, head, "object", "trajectory_selectivity_log2"
                    ),
                    "fixed_position_enrichment": _metric(
                        item, block, head, "object", "fixed_position_enrichment"
                    ),
                    "context_enrichment": _metric(
                        item, block, head, "object", "context_enrichment"
                    ),
                    "entropy": _metric(item, block, head, "full", "entropy"),
                    "evidence_image": evidence.name,
                }
                top_payload[role].append(row)
                evidence_rows.append(row)
        video, preview, query_payload = _copy_media(
            output_dir=output_dir,
            query_root=args.query_root.expanduser().resolve(),
            pass1_root=args.pass1_root.expanduser().resolve(),
            seed=int(args.seed),
            model=model,
            case=case,
        )
        counts = Counter(item["labels"].reshape(-1).tolist())
        report["samples"].append(
            {
                "model": model,
                "case": case,
                "role_counts": dict(counts),
                "top_heads": top_payload,
                "video": video,
                "preview": preview,
                "prompt": query_payload.get("prompt", ""),
                "track_quality": query_payload.get("track_quality", {}),
            }
        )
        table_rows = "".join(
            "<tr>"
            f"<td><span class='role role-{row['role']}'>{row['role']}</span></td>"
            f"<td>B{row['block']:02d} H{row['head']:02d}</td>"
            f"<td>{row['score']:.3f}</td><td>{row['consistency']:.2f}</td>"
            f"<td>{row['local_enrichment']:.2f}</td>"
            f"<td>{row['trajectory_selectivity_log2']:.2f}</td>"
            f"<td>{row['fixed_position_enrichment']:.2f}</td>"
            f"<td>{row['context_enrichment']:.2f}</td>"
            f"<td>{row['entropy']:.3f}</td>"
            f"<td><a href='{html.escape(row['evidence_image'])}'>矩阵</a></td>"
            "</tr>"
            for row in evidence_rows
        )
        count_text = " · ".join(
            f"{role} {counts.get(role, 0)}" for role in (*ROLES, "M")
        )
        evidence_figures = "".join(
            "<figure>"
            f"<a href='{html.escape(top_payload[role][0]['evidence_image'])}'>"
            f"<img loading='lazy' src='{html.escape(top_payload[role][0]['evidence_image'])}'></a>"
            f"<figcaption>{role} · B{top_payload[role][0]['block']:02d} "
            f"H{top_payload[role][0]['head']:02d}</figcaption></figure>"
            for role in ROLES
        )
        sections.append(
            f"""<section>
<h2>{html.escape(MODEL_NAMES[model])} · {html.escape(case)}</h2>
<p class="prompt">{html.escape(str(query_payload.get("prompt", "")))}</p>
<div class="media-row">
  <figure><video controls preload="metadata" src="{html.escape(video)}"></video>
  <figcaption>seed 851 确定性 pass-1 生成视频</figcaption></figure>
  <figure><img src="{html.escape(preview)}"><figcaption>运动目标 query 轨迹与 token</figcaption></figure>
</div>
<p class="counts">{html.escape(count_text)}</p>
<a href="{html.escape(role_grid.name)}"><img class="role-grid" loading="lazy" src="{html.escape(role_grid.name)}"></a>
<div class="evidence-row">{evidence_figures}</div>
<div class="table-wrap"><table>
<thead><tr><th>类</th><th>Head</th><th>分数</th><th>4步一致率</th>
<th>局部富集</th><th>轨迹/null log2</th><th>固定位置富集</th>
<th>ctx富集</th><th>熵</th><th>证据</th></tr></thead>
<tbody>{table_rows}</tbody></table></div>
</section>"""
        )

    report_path = output_dir / "classification.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>全 token 时间矩阵与运动轨迹 Head 分类 Pilot</title>
<style>
body{{margin:0;background:#f7f7f5;color:#242622;font:14px Arial,sans-serif;overflow-x:hidden}}
main{{max-width:1500px;margin:auto;padding:18px}} h1,h2{{letter-spacing:0}}
h1{{font-size:25px;margin:0 0 10px}} h2{{font-size:18px;overflow-wrap:anywhere}}
.method{{padding:10px 0;border-top:1px solid #bbb;border-bottom:1px solid #bbb;line-height:1.65}}
section{{padding:18px 0;border-bottom:2px solid #555}} .prompt{{max-width:1100px}}
.media-row{{display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:1200px}}
.evidence-row{{display:grid;grid-template-columns:repeat(5,minmax(210px,1fr));gap:9px;margin:12px 0}}
figure{{margin:0;min-width:0}} video,figure img{{display:block;width:100%;max-height:420px;object-fit:contain;background:#111}}
figcaption{{padding-top:5px;color:#555}} .role-grid{{width:100%;height:auto;display:block;background:#fff}}
.counts{{font-weight:700}} .table-wrap{{overflow:auto}} table{{border-collapse:collapse;width:100%;background:#fff}}
th,td{{padding:5px 7px;border:1px solid #d1d1cc;text-align:right;white-space:nowrap}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
.role{{display:inline-block;color:#fff;font-weight:700;padding:2px 7px;border-radius:3px}}
{''.join(f'.role-{role}{{background:{ROLE_COLORS[role]}}}' for role in ROLES)}
a{{color:#075e54}} @media(max-width:1100px){{.evidence-row{{grid-template-columns:repeat(2,minmax(210px,1fr))}}}}
@media(max-width:800px){{.media-row,.evidence-row{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>全 token 时间矩阵 + moving-object 轨迹富集 Head 分类 Pilot</h1>
<div class="method">
<b>判断流程：</b>每个 Q token 对全部 5824 个 K token 做精确 softmax；先汇总为 13×13 时间矩阵，
再对运动轨迹、半网格平移 null、时间循环打乱 null 计算条件富集。每个模型、case、去噪步内，
在 30×24 个 head 间做百分位评分；S/T/P/C/G 分别表示空间局部、运动轨迹、固定位置、
8 帧上下文和全局候选。四个去噪步取均值；第一、第二名差值小于 0.08，或四步一致率低于 0.75，
标为 M（混合/不稳定）。这些是<b>相对功能候选</b>，不是因果结论。
</div>
{''.join(sections)}
</main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"[fulltoken-moving] wrote {output_dir / 'index.html'}")
    print(f"[fulltoken-moving] wrote {report_path}")


if __name__ == "__main__":
    main()
