#!/usr/bin/env python3
"""Build paired four-model baseline statistics inside the shared gallery."""

from __future__ import annotations

import csv
import html
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


GALLERY = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/head-role-dose-control-pilot"
)
OPENVID = "openvid_lora_step10000"
METRICS = (
    ("physics_iq_with_context", "Physics-IQ ctx"),
    ("physics_iq_without_context", "Physics-IQ noctx"),
    ("pmf_with_context", "PMF ctx"),
    ("pmf_without_context", "PMF noctx"),
)


def atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main() -> None:
    manifest = json.loads((GALLERY / "manifest.json").read_text(encoding="utf-8"))
    labels = manifest["model_labels"]
    baselines = [
        record
        for record in manifest["records"]
        if record.get("kind") == "baseline" and int(record.get("seed", -1)) == 851
    ]
    values: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for record in baselines:
        for metric, _ in METRICS:
            value = record.get("metrics", {}).get(metric)
            if isinstance(value, (int, float)):
                values[str(record["model"])][str(record["case_id"])][metric] = float(
                    value
                )

    models = list(manifest["models"])
    if OPENVID not in models:
        raise RuntimeError("OpenVid model is missing from the shared manifest")
    cases = sorted(str(case["id"]) for case in manifest["cases"])
    for model in models:
        if set(values[model]) != set(cases):
            raise RuntimeError(f"{model} has {len(values[model])}/{len(cases)} baselines")
        for case in cases:
            missing = [metric for metric, _ in METRICS if metric not in values[model][case]]
            if missing:
                raise RuntimeError(f"{model}/{case} missing {missing}")

    output = GALLERY / "openvid-baseline-comparison"
    output.mkdir(parents=True, exist_ok=True)
    mean_rows: list[dict[str, Any]] = []
    for model in models:
        row: dict[str, Any] = {"model": model, "label": labels[model]}
        for metric, _ in METRICS:
            row[metric] = float(
                np.mean([values[model][case][metric] for case in cases])
            )
        mean_rows.append(row)

    paired_rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        if model == OPENVID:
            continue
        for metric_index, (metric, _) in enumerate(METRICS):
            deltas = np.array(
                [
                    values[OPENVID][case][metric] - values[model][case][metric]
                    for case in cases
                ],
                dtype=float,
            )
            low, high = bootstrap(deltas, 20260731 + 10 * model_index + metric_index)
            paired_rows.append(
                {
                    "reference_model": model,
                    "reference_label": labels[model],
                    "metric": metric,
                    "mean_delta": float(deltas.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "openvid_wins": int(np.count_nonzero(deltas > 0)),
                    "ties": int(np.count_nonzero(deltas == 0)),
                    "reference_wins": int(np.count_nonzero(deltas < 0)),
                    "num_cases": len(deltas),
                }
            )

    with (output / "baseline_means.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mean_rows[0]))
        writer.writeheader()
        writer.writerows(mean_rows)
    with (output / "openvid_paired_deltas.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    atomic_text(
        output / "analysis.json",
        json.dumps(
            {
                "models": models,
                "model_labels": labels,
                "metrics": [metric for metric, _ in METRICS],
                "num_cases": len(cases),
                "baseline_means": mean_rows,
                "openvid_paired_deltas": paired_rows,
                "delta_semantics": "OpenVid minus reference; positive is better",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )

    mean_table = table(
        ["模型", *[label for _, label in METRICS]],
        [
            [row["label"], *[f"{row[metric]:.4f}" for metric, _ in METRICS]]
            for row in mean_rows
        ],
    )
    delta_table = table(
        ["参考模型", "指标", "OpenVid-参考", "95% CI", "胜/平/负"],
        [
            [
                row["reference_label"],
                dict(METRICS)[row["metric"]],
                f"{row['mean_delta']:+.4f}",
                f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}]",
                f"{row['openvid_wins']}/{row['ties']}/{row['reference_wins']}",
            ]
            for row in paired_rows
        ],
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>四模型 Baseline 对比</title><style>
:root{{--bg:#101315;--panel:#181d20;--line:#3a4247;--text:#eef1f2;--muted:#a8b0b5;--accent:#58bda8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:13px/1.5 system-ui,sans-serif}}
header,main{{padding:14px 18px}}header{{border-bottom:1px solid var(--line)}}h1,h2,p{{margin:0}}h2{{margin-top:24px}}
a{{color:var(--accent)}}.note{{color:var(--muted);margin-top:6px}}table{{margin-top:9px;width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{padding:6px 8px;border:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#242a2e}}
</style></head><body><header><h1>四模型 Baseline 对比 · seed 851</h1>
<p class="note">20 个相同 source case；四项当前指标均为越高越好。Paired 差值为 OpenVid - 参考模型，正值表示 OpenVid 更好。</p>
<p><a href="../cases/">返回逐 Case 视频与消融指标</a></p></header><main>
<h2>Baseline 均值</h2>{mean_table}
<h2>OpenVid 配对差值</h2>{delta_table}
<p class="note">胜/平/负按逐 case 比较；置信区间由 case bootstrap 10,000 次计算。</p>
</main></body></html>"""
    atomic_text(output / "index.html", page)
    print(
        f"[openvid-baseline-comparison] models={len(models)} cases={len(cases)} "
        f"output={output / 'index.html'}"
    )


if __name__ == "__main__":
    main()
