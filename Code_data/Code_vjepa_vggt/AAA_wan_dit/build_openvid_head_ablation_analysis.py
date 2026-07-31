#!/usr/bin/env python3
"""Build paired OpenVid-LoRA head-ablation curves against its own baseline."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy import stats

font_manager.fontManager.addfont(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_openvid_lora_head34/"
    "seed851/generation/openvid_lora_step10000/seed-000851"
)
SUBSETS = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_openvid_lora_head34/"
    "configs/openvid_lora_head34_subsets.json"
)
OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/"
    "head-role-dose-control-pilot/openvid-head-ablation-analysis"
)
METRICS = (
    ("physics_iq_with_context", "Physics-IQ ctx"),
    ("physics_iq_without_context", "Physics-IQ noctx"),
    ("pmf_with_context", "PMF ctx"),
    ("pmf_without_context", "PMF noctx"),
)
STAGES = ((0, 10), (10, 20), (0, 40))
STAGE_LABEL = {(0, 10): "0-10", (10, 20): "10-20", (0, 40): "0-40"}
FAMILY_LABEL = {
    "matched": "固定数量对照",
    "local_dominant": "Local-dominant × 深度",
    "same_dominant": "Same-frame-dominant × 深度",
}
COLORS = ("#176B87", "#B33A3A", "#537A2C", "#7A5195")
CONFIG_RE = re.compile(r"^(?P<subset>.+)_steps(?P<start>\d{2})_(?P<end>\d{2})$")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_scores(result_root: Path) -> dict[str, dict[str, float]]:
    result = {}
    for path in sorted(result_root.glob("*.json")):
        payload = read_json(path)
        result[path.stem] = {
            metric: float(payload[metric]["score"]) for metric, _ in METRICS
        }
    if len(result) != 20:
        raise ValueError(f"{result_root}: expected 20 cases, got {len(result)}")
    return result


def classify(subset: str) -> tuple[str, str]:
    fixed = {
        "S_local_k32_r00_exactblock": ("matched", "Local top-32"),
        "S_same_k32_r00_exactblock": ("matched", "Same-frame top-32"),
        "S_local_same_union_k64_r00_exactblock": (
            "matched",
            "Local ∪ Same-frame (64)",
        ),
    }
    if subset in fixed:
        return fixed[subset]
    for prefix, family, label in (
        ("S_local_dominant_", "local_dominant", "Local-dominant"),
        ("S_same_frame_dominant_", "same_dominant", "Same-frame-dominant"),
    ):
        if subset.startswith(prefix):
            suffix = subset.removeprefix(prefix)
            suffix_label = {
                "all": "all",
                "depth_early": "early B00-09",
                "depth_middle": "middle B10-19",
                "depth_late": "late B20-29",
            }[suffix]
            return family, f"{label} {suffix_label}"
    raise ValueError(f"Unknown subset: {subset}")


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    return tuple(np.quantile(values[indices].mean(axis=1), (0.025, 0.975)))


def adjust_bh(rows: list[dict], metric: str) -> None:
    ordered = sorted(rows, key=lambda row: row[f"{metric}_p"])
    running = 1.0
    for rank in range(len(ordered), 0, -1):
        row = ordered[rank - 1]
        running = min(running, row[f"{metric}_p"] * len(ordered) / rank)
        row[f"{metric}_q"] = running


def save_summary(rows: list[dict]) -> None:
    fields = [
        "config",
        "subset",
        "family",
        "series",
        "stage",
        "heads",
        "mean_relative_drop_pct",
    ]
    for metric, _ in METRICS:
        fields += [
            f"{metric}_mean",
            f"{metric}_baseline",
            f"{metric}_delta",
            f"{metric}_ci_low",
            f"{metric}_ci_high",
            f"{metric}_p",
            f"{metric}_q",
            f"{metric}_wins",
        ]
    with (OUTPUT / "paired_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(metric: str, label: str, rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), sharey=True)
    for axis, family in zip(axes, FAMILY_LABEL, strict=True):
        family_rows = [row for row in rows if row["family"] == family]
        series = list(dict.fromkeys(row["series"] for row in family_rows))
        for index, name in enumerate(series):
            by_stage = {
                (row["start"], row["end"]): row
                for row in family_rows
                if row["series"] == name
            }
            values = np.asarray(
                [by_stage[stage][f"{metric}_delta"] for stage in STAGES]
            )
            low = np.asarray(
                [by_stage[stage][f"{metric}_ci_low"] for stage in STAGES]
            )
            high = np.asarray(
                [by_stage[stage][f"{metric}_ci_high"] for stage in STAGES]
            )
            axis.errorbar(
                range(3),
                values,
                yerr=(values - low, high - values),
                marker="o",
                capsize=3,
                linewidth=1.8,
                color=COLORS[index],
                label=name,
            )
        axis.axhline(0, color="#20262E", linewidth=1.2, linestyle="--")
        axis.set_title(FAMILY_LABEL[family], fontsize=11)
        axis.set_xticks(range(3), [STAGE_LABEL[stage] for stage in STAGES])
        axis.set_xlabel("去噪步区间")
        axis.grid(axis="y", alpha=0.22)
        axis.legend(fontsize=7.5, frameon=False)
    axes[0].set_ylabel(f"Δ {label}（消融 − baseline）")
    fig.suptitle(f"{label}：OpenVid LoRA head 消融", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{metric}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_ranking(rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: row["mean_relative_drop_pct"])
    labels = [f"{row['series']} | {row['stage']}" for row in ordered]
    values = [row["mean_relative_drop_pct"] for row in ordered]
    colors = ["#B33A3A" if value > 0 else "#176B87" for value in values]
    fig, axis = plt.subplots(figsize=(11.5, 9))
    axis.barh(range(len(rows)), values, color=colors)
    axis.axvline(0, color="#20262E", linewidth=1.2)
    axis.set_yticks(range(len(rows)), labels, fontsize=7.5)
    axis.set_xlabel("四指标平均相对下降率（%，正值表示消融后更差）")
    axis.set_title("OpenVid LoRA：33 组 head 消融综合排序")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "mean_relative_drop.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def metric_cell(row: dict, metric: str) -> str:
    delta = row[f"{metric}_delta"]
    low, high = row[f"{metric}_ci_low"], row[f"{metric}_ci_high"]
    tone = "bad" if delta < 0 else "good"
    return (
        f'<td><span class="{tone}">{delta:+.3f}</span>'
        f"<small>[{low:+.3f}, {high:+.3f}]</small></td>"
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    subset_data = read_json(SUBSETS)["subsets"]
    baseline = load_scores(ROOT / "baseline" / "results")
    cases = sorted(baseline)
    baseline_means = {
        metric: float(np.mean([baseline[case][metric] for case in cases]))
        for metric, _ in METRICS
    }
    rows, case_rows = [], []
    for config_path in sorted(path for path in ROOT.iterdir() if path.is_dir()):
        if config_path.name == "baseline":
            continue
        match = CONFIG_RE.fullmatch(config_path.name)
        if not match:
            raise ValueError(f"Unexpected config: {config_path.name}")
        subset = match.group("subset")
        start, end = int(match.group("start")), int(match.group("end"))
        family, series = classify(subset)
        current = load_scores(config_path / "results")
        if sorted(current) != cases:
            raise ValueError(f"{config_path}: case set differs from baseline")
        row = {
            "config": config_path.name,
            "subset": subset,
            "family": family,
            "series": series,
            "stage": STAGE_LABEL[(start, end)],
            "start": start,
            "end": end,
            "heads": int(subset_data[subset]["k"]),
        }
        relative_drops = []
        for metric_index, (metric, _) in enumerate(METRICS):
            base = np.asarray([baseline[case][metric] for case in cases])
            value = np.asarray([current[case][metric] for case in cases])
            delta = value - base
            low, high = bootstrap_ci(
                delta, 851_000 + len(rows) * len(METRICS) + metric_index
            )
            row.update(
                {
                    f"{metric}_mean": float(value.mean()),
                    f"{metric}_baseline": baseline_means[metric],
                    f"{metric}_delta": float(delta.mean()),
                    f"{metric}_ci_low": float(low),
                    f"{metric}_ci_high": float(high),
                    f"{metric}_p": float(stats.ttest_rel(value, base).pvalue),
                    f"{metric}_wins": int(np.count_nonzero(delta > 0)),
                }
            )
            relative_drops.append(
                100 * (baseline_means[metric] - value.mean())
                / abs(baseline_means[metric])
            )
            case_rows += [
                {
                    "config": config_path.name,
                    "case": case,
                    "metric": metric,
                    "baseline": float(base_value),
                    "ablation": float(value_now),
                    "delta": float(delta_now),
                }
                for case, base_value, value_now, delta_now in zip(
                    cases, base, value, delta, strict=True
                )
            ]
        row["mean_relative_drop_pct"] = float(np.mean(relative_drops))
        rows.append(row)
    if len(rows) != 33:
        raise ValueError(f"Expected 33 ablations, got {len(rows)}")
    for metric, _ in METRICS:
        adjust_bh(rows, metric)
    save_summary(rows)
    with (OUTPUT / "paired_case_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, case_rows[0])
        writer.writeheader()
        writer.writerows(case_rows)
    for metric, label in METRICS:
        plot_metric(metric, label, rows)
    plot_ranking(rows)

    findings = []
    for metric, label in METRICS:
        worst = min(rows, key=lambda row: row[f"{metric}_delta"])
        best = max(rows, key=lambda row: row[f"{metric}_delta"])
        findings.append(
            {
                "metric": metric,
                "label": label,
                "worst": worst["config"],
                "worst_delta": worst[f"{metric}_delta"],
                "best": best["config"],
                "best_delta": best[f"{metric}_delta"],
            }
        )
    analysis = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "cases": len(cases),
        "ablations": len(rows),
        "baseline_means": baseline_means,
        "most_harmful_composite": max(
            rows, key=lambda row: row["mean_relative_drop_pct"]
        )["config"],
        "least_harmful_composite": min(
            rows, key=lambda row: row["mean_relative_drop_pct"]
        )["config"],
        "metric_findings": findings,
    }
    (OUTPUT / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    baseline_cells = "".join(
        f"<td>{baseline_means[metric]:.3f}</td>" for metric, _ in METRICS
    )
    finding_cards = "".join(
        f"<article><h3>{item['label']}</h3>"
        f"<p>最大下降：<b>{item['worst']}</b> ({item['worst_delta']:+.3f})</p>"
        f"<p>最大上升：<b>{item['best']}</b> ({item['best_delta']:+.3f})</p>"
        "</article>"
        for item in findings
    )
    table_rows = "".join(
        "<tr>"
        f"<td>{row['series']}</td><td>{row['stage']}</td><td>{row['heads']}</td>"
        f"<td>{row['mean_relative_drop_pct']:+.2f}%</td>"
        + "".join(metric_cell(row, metric) for metric, _ in METRICS)
        + "</tr>"
        for row in sorted(rows, key=lambda item: -item["mean_relative_drop_pct"])
    )
    curves = "".join(
        f'<section><h2>{label}</h2><img src="{metric}.png" alt="{label}"></section>'
        for metric, label in METRICS
    )
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenVid LoRA Head 消融分析</title><style>
:root{{--ink:#20262e;--muted:#64717d;--line:#d8dde2;--paper:#f7f8f9;
--good:#176b87;--bad:#b33a3a}}*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);
font:14px/1.55 Arial,sans-serif}}header{{border-bottom:1px solid var(--line);
background:var(--paper)}}.shell,main{{max-width:1500px;margin:auto;padding:18px 24px}}
h1{{margin:0;font-size:26px}}h2{{font-size:19px}}h3{{font-size:15px;margin:0 0 6px}}
p{{margin:4px 0}}a{{color:#176b87;margin-right:16px}}.note{{border-left:4px solid
#c06c23;background:#fff8ed;padding:10px 14px;margin:12px 0 18px}}.cards{{display:grid;
grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:14px 0 22px}}article{{
border:1px solid var(--line);border-radius:6px;padding:11px}}section{{margin-bottom:28px}}
img{{width:100%;border:1px solid var(--line)}}.table-wrap{{overflow:auto;
border:1px solid var(--line)}}table{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{padding:6px 8px;border-bottom:1px solid #e7eaed;text-align:right;white-space:nowrap}}
th{{background:#eef1f3;position:sticky;top:0}}th:first-child,td:first-child{{text-align:left}}
small{{display:block;color:var(--muted);font-size:10px}}.good{{color:var(--good);
font-weight:700}}.bad{{color:var(--bad);font-weight:700}}.muted{{color:var(--muted)}}
@media(max-width:900px){{.cards{{grid-template-columns:1fr 1fr}}}}</style></head>
<body><header><div class="shell"><h1>OpenVid LoRA Head 消融分析</h1>
<p>seed 851 · 20 个配对 case · 33 个消融配置 · {updated}</p>
<nav><a href="../cases/">视频 case 页</a><a href="../openvid-baseline-comparison/">
四模型 baseline</a><a href="paired_summary.csv">汇总 CSV</a>
<a href="paired_case_metrics.csv">逐 case CSV</a></nav></div></header><main>
<div class="note"><b>读图：</b>纵轴为“消融分数 − OpenVid LoRA baseline 分数”。
虚线 0 表示不变，低于 0 表示消融后变差；误差线为同一批 20 个 case 的 bootstrap
95% 区间。固定 32-head 实验适合比较类别；100/59-head 与深度子集还受 head
数量影响，不能只凭绝对变化判断类别重要性。</div>
<section><h2>当前结论</h2><div class="note">
<p><b>最一致的负向干预：</b>在去噪步 0-10 消融 late B20-29 的 41 个
Local-dominant heads，四项指标均下降；其中两项 PMF 的 95% 区间完全低于 0，
Physics-IQ 区间仍跨 0。</p>
<p><b>固定 32-head 公平比较：</b>Local top-32 与 Same-frame top-32
在三个时段都没有形成稳定的四指标下降，因此目前不能说某一类 top-32 head
整体比另一类更关键。</p>
<p><b>去噪阶段：</b>Local-dominant late heads 的负向影响集中在 0-10；
Local-dominant all 在 10-20 的 PMF 降幅大于 0-10。0-40 是四倍干预剂量，
只适合观察累积效应，不能直接和十步窗口比较强弱。</p>
<p><b>指标冲突：</b>全程消融 Local-dominant all 时 Physics-IQ 上升而 PMF
显著下降；全程消融 Same-frame-dominant all 时 Physics-IQ 大幅上升而 PMF
近似不变。这说明 Physics-IQ 单视角近似分数可能奖励更静态或像素重叠更高的结果，
不能把其上升单独解释为物理质量改善。</p></div></section>
<h2>OpenVid LoRA baseline</h2><div class="table-wrap"><table><thead><tr><th>模型</th>
{''.join(f'<th>{label}</th>' for _, label in METRICS)}</tr></thead><tbody>
<tr><td>OpenVid LoRA baseline</td>{baseline_cells}</tr></tbody></table></div>
<div class="cards">{finding_cards}</div><section><h2>综合排序</h2>
<img src="mean_relative_drop.png" alt="relative drop"><p class="muted">
综合值是四项指标相对下降百分比的简单平均，只用于导航，结论以原始曲线为准。</p>
</section>{curves}<section><h2>完整配对结果</h2><div class="table-wrap"><table>
<thead><tr><th>Head 子集</th><th>去噪步</th><th>Heads</th><th>平均下降率</th>
{''.join(f'<th>Δ {label}<small>95% CI</small></th>' for _, label in METRICS)}
</tr></thead><tbody>{table_rows}</tbody></table></div></section></main></body></html>"""
    (OUTPUT / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
