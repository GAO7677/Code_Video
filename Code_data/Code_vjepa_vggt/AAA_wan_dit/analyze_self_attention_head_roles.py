#!/usr/bin/env python3
"""Classify Wan Block self-attention heads from pooled all-token matrices."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


TAG_NAMES = {
    "S": "帧内空间",
    "T": "跨帧时间对齐",
    "C": "上下文/首帧锚定",
    "G": "全局弥散",
    "P": "相对位置/混合路由",
}

ROLE_NAMES = {
    "S": "帧内空间 head",
    "T": "时间对齐 head",
    "C": "上下文锚定 head",
    "G": "全局聚合 head",
    "P": "相对位置/混合路由 head",
    "ST": "时空局部/跟踪 head",
    "TC": "时间传播+上下文锚定 head",
    "CG": "全局上下文 head",
    "STC": "时空跟踪+上下文锚定 head",
}


def _metrics(
    attention: np.ndarray,
    *,
    token_count: int,
    temporal_tokens: int,
) -> np.ndarray:
    heads, bins, key_bins = attention.shape
    if bins != key_bins:
        raise ValueError(f"expected square matrices, got {attention.shape}")
    spatial_tokens = token_count // temporal_tokens
    centers = np.minimum(
        ((np.arange(bins) + 0.5) * token_count / bins).astype(np.int64),
        token_count - 1,
    )
    frame = centers // spatial_tokens
    spatial = centers % spatial_tokens
    same_frame = frame[:, None] == frame[None, :]
    cross_frame = ~same_frame

    # One display bin covers about token_count / bins original tokens.
    spatial_tolerance = max(1, int(math.ceil(token_count / bins)))
    near_spatial = (
        np.abs(spatial[:, None] - spatial[None, :]) <= spatial_tolerance
    )
    aligned_cross_time = cross_frame & near_spatial
    local_same_frame = same_frame & near_spatial
    frame_distance = np.abs(frame[:, None] - frame[None, :])
    past_frame = frame[None, :] < frame[:, None]
    future_frame = frame[None, :] > frame[:, None]

    cross_fraction = aligned_cross_time.sum(1) / np.maximum(
        cross_frame.sum(1), 1
    )
    local_fraction = local_same_frame.sum(1) / np.maximum(
        same_frame.sum(1), 1
    )

    probability = np.clip(attention, 1.0e-30, None)
    same_mass = (attention * same_frame).sum(2)
    cross_mass = 1.0 - same_mass
    aligned_mass = (attention * aligned_cross_time).sum(2)
    local_mass = (attention * local_same_frame).sum(2)
    aligned_enrichment = (
        aligned_mass
        / np.maximum(cross_mass, 1.0e-8)
        / cross_fraction[None, :]
    )
    local_enrichment = (
        local_mass
        / np.maximum(same_mass, 1.0e-8)
        / local_fraction[None, :]
    )
    first_frame = frame == 0

    return np.stack(
        [
            -(probability * np.log(probability)).sum(2).mean(1)
            / math.log(bins),
            same_mass.mean(1),
            attention[:, :, first_frame].sum(2).mean(1),
            (attention * frame_distance).sum(2).mean(1),
            aligned_mass.mean(1),
            np.nanmean(aligned_enrichment, axis=1),
            local_mass.mean(1),
            np.nanmean(local_enrichment, axis=1),
            (attention * past_frame).sum(2).mean(1),
            (attention * future_frame).sum(2).mean(1),
        ],
        axis=1,
    )


def _tags(row: np.ndarray) -> str:
    entropy, same_frame, first_frame, _, aligned_mass, aligned_enrichment, local_mass, _ = row[:8]
    tags = []
    if same_frame >= 0.20 or local_mass >= 0.18:
        tags.append("S")
    if aligned_mass >= 0.40 and aligned_enrichment >= 10.0:
        tags.append("T")
    if first_frame >= 0.23:
        tags.append("C")
    if entropy >= 0.75:
        tags.append("G")
    return "".join(tags) or "P"


def _role_name(tags: str) -> str:
    if tags in ROLE_NAMES:
        return ROLE_NAMES[tags]
    return "+".join(TAG_NAMES[tag] for tag in tags)


def _confidence(model_tags: list[str], overall_tags: str) -> str:
    counts = Counter(model_tags)
    if len(counts) == 1:
        return "高"
    common = set(model_tags[0])
    for tags in model_tags[1:]:
        common &= set(tags)
    if set(overall_tags) <= common:
        return "高"
    if counts.most_common(1)[0][1] >= 2:
        return "中"
    if common and set(overall_tags) & common:
        return "中"
    return "低"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = sorted(root.glob("*/*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"no summaries found under {root}")
    first_summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    temporal_tokens, grid_h, grid_w = (
        int(value) for value in first_summary["latent_grid"]
    )
    token_count = temporal_tokens * grid_h * grid_w

    by_model: dict[str, list[np.ndarray]] = {}
    all_samples: list[np.ndarray] = []
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model = str(summary["model"])
        for step in summary["steps"]:
            matrix_path = (
                summary_path.parent
                / str(step["directory"])
                / str(step["matrix_npz"])
            )
            with np.load(matrix_path) as arrays:
                metrics = _metrics(
                    arrays["key_mass"],
                    token_count=token_count,
                    temporal_tokens=temporal_tokens,
                )
            by_model.setdefault(model, []).append(metrics)
            all_samples.append(metrics)

    overall = np.stack(all_samples).mean(0)
    model_means = {
        model: np.stack(samples).mean(0)
        for model, samples in sorted(by_model.items())
    }
    records = []
    for head, row in enumerate(overall):
        overall_tags = _tags(row)
        model_tags = {
            model: _tags(values[head]) for model, values in model_means.items()
        }
        records.append(
            {
                "head": head,
                "role": _role_name(overall_tags),
                "tags": overall_tags,
                "confidence": _confidence(list(model_tags.values()), overall_tags),
                "model_tags": model_tags,
                "entropy": float(row[0]),
                "same_frame_mass": float(row[1]),
                "first_frame_mass": float(row[2]),
                "mean_frame_distance": float(row[3]),
                "aligned_cross_time_mass": float(row[4]),
                "aligned_cross_time_enrichment": float(row[5]),
                "local_same_frame_mass": float(row[6]),
                "local_same_frame_enrichment": float(row[7]),
                "past_frame_mass": float(row[8]),
                "future_frame_mass": float(row[9]),
                "history_bias": float(row[8] - row[9]),
            }
        )

    json_path = output_dir / "block17_head_roles.json"
    json_path.write_text(
        json.dumps(
            {
                "root": str(root),
                "latent_grid": [temporal_tokens, grid_h, grid_w],
                "num_samples": len(all_samples),
                "models": sorted(model_means),
                "thresholds": {
                    "spatial": "same_frame_mass >= 0.20 or local_same_frame_mass >= 0.18",
                    "temporal": "aligned_cross_time_mass >= 0.40 and enrichment >= 10",
                    "context": "first_frame_mass >= 0.23",
                    "global": "normalized_entropy >= 0.75",
                },
                "heads": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = output_dir / "block17_head_roles.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                key
                for key in records[0]
                if key != "model_tags"
            ]
            + [f"{model}_tags" for model in model_means],
        )
        writer.writeheader()
        for record in records:
            row = {key: value for key, value in record.items() if key != "model_tags"}
            row.update(
                {
                    f"{model}_tags": record["model_tags"][model]
                    for model in model_means
                }
            )
            writer.writerow(row)

    rows = []
    for record in records:
        evidence = (
            f"帧内 {record['same_frame_mass']:.1%}; "
            f"跨帧同位 {record['aligned_cross_time_mass']:.1%}; "
            f"首帧 {record['first_frame_mass']:.1%}; "
            f"历史偏置 {record['history_bias']:+.1%}; "
            f"熵 {record['entropy']:.3f}"
        )
        model_text = ", ".join(
            f"{model}={tags}"
            for model, tags in record["model_tags"].items()
        )
        rows.append(
            f"| {record['head']:02d} | {record['role']} | "
            f"{record['confidence']} | {evidence} | {model_text} |"
        )

    markdown = f"""# Block 17 Self-Attention Head Roles

基于 `{len(all_samples)}` 份矩阵（三个模型、五个 case、四个去噪步）统计。
输入 token 网格为 `{temporal_tokens}x{grid_h}x{grid_w}={token_count}`，保存矩阵为
512x512 连续 token 池化。角色允许多标签。

标签：`S` 帧内空间，`T` 跨帧近似同位置，`C` 首帧/上下文锚定，
`G` 高熵全局聚合，`P` 相对位置或未被前四类覆盖的混合路由。
“历史偏置”定义为指向过去帧的质量减去指向未来帧的质量。

| Head | 主要角色 | 稳定性 | 核心证据 | 分模型标签 |
|---:|---|:---:|---|---|
{chr(10).join(rows)}

## 判定阈值

- 空间：帧内质量 >= 20%，或帧内近邻质量 >= 18%。
- 时间：跨帧近似同空间位置质量 >= 40%，且相对均匀注意力富集 >= 10 倍。
- 上下文：平均指向首个 latent 帧的质量 >= 23%。
- 全局：归一化注意力熵 >= 0.75。

## 限制

这些指标来自 512x512 池化矩阵，不是原始 5824x5824 矩阵。“同空间位置”
按池化 bin 中心、允许约一个 bin 的 token 误差近似计算。因此标签适合用于筛选
和提出消融假设，不应被当成严格的神经语义证明。
"""
    markdown_path = output_dir / "block17_head_roles.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "markdown": str(markdown_path),
                "csv": str(csv_path),
                "json": str(json_path),
                "heads": len(records),
                "samples": len(all_samples),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
