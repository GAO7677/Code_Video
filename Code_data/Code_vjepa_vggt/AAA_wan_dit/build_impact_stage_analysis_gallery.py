#!/usr/bin/env python3
"""Build paired Impact statistics and same-seed qualitative comparisons."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ANALYSIS_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_stc_motion_analysis"
)
DEFAULT_GALLERY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery"
)
MODELS = ("wan_lora", "xssc", "physrvg")
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
STAGES = ((0, 5), (5, 10), (0, 10), (10, 20), (20, 30))
ROLES = ("S", "T", "ST")
ROLE_LABELS = {"S": "只消融 S", "T": "只消融 T", "ST": "联合消融 S+T"}
ROLE_CHART_LABELS = {"S": "S-only", "T": "T-only", "ST": "S+T"}
ROLE_COLORS = {"S": "#22876f", "T": "#d18d19", "ST": "#3478b8"}
EFFECT_LABELS = {
    "s_minus_t": "S - T",
    "add_t_given_s": "在消融 S 后再消融 T",
    "add_s_given_t": "在消融 T 后再消融 S",
}
EFFECT_CHART_LABELS = {
    "s_minus_t": "S - T",
    "add_t_given_s": "Add T given S ablated",
    "add_s_given_t": "Add S given T ablated",
}
EFFECT_COLORS = {
    "s_minus_t": "#6d747c",
    "add_t_given_s": "#d18d19",
    "add_s_given_t": "#22876f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--gallery-root", type=Path, default=DEFAULT_GALLERY_ROOT)
    parser.add_argument("--output-name", default="impact-stage-analysis")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    return parser.parse_args()


def bootstrap_ci(
    values: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_impacts(
    frame: pd.DataFrame,
    model: str,
    start: int,
    end: int,
) -> pd.DataFrame:
    selected = frame[
        (frame["model"] == model)
        & (frame["denoise_start"] == start)
        & (frame["denoise_end"] == end)
        & frame["role"].isin(ROLES)
        & ~frame["tracking_failure"].astype(bool)
    ]
    return (
        selected.pivot(index="seed", columns="role", values="impact_score")
        .dropna(subset=list(ROLES))
        .sort_index()
    )


def compute_statistics(
    frame: pd.DataFrame,
    samples: int,
) -> tuple[pd.DataFrame, dict[tuple[str, int, int], pd.DataFrame]]:
    rng = np.random.default_rng(42)
    records: list[dict[str, Any]] = []
    paired: dict[tuple[str, int, int], pd.DataFrame] = {}
    for model in MODELS:
        for start, end in STAGES:
            wide = paired_impacts(frame, model, start, end)
            paired[(model, start, end)] = wide
            record: dict[str, Any] = {
                "model": model,
                "stage": f"[{start},{end})",
                "denoise_start": start,
                "denoise_end": end,
                "n_paired_seeds": len(wide),
            }
            for role in ROLES:
                values = wide[role].to_numpy(float)
                low, high = bootstrap_ci(values, samples, rng)
                record[f"{role}_mean"] = float(values.mean())
                record[f"{role}_ci_low"] = low
                record[f"{role}_ci_high"] = high
            effects = {
                "s_minus_t": wide["S"] - wide["T"],
                "add_t_given_s": wide["ST"] - wide["S"],
                "add_s_given_t": wide["ST"] - wide["T"],
            }
            for name, values in effects.items():
                array = values.to_numpy(float)
                low, high = bootstrap_ci(array, samples, rng)
                record[f"{name}_mean"] = float(array.mean())
                record[f"{name}_ci_low"] = low
                record[f"{name}_ci_high"] = high
                record[f"{name}_positive_rate"] = float((array > 0).mean())
            records.append(record)
    return pd.DataFrame(records), paired


def confidence_text(mean: float, low: float, high: float) -> str:
    if low > 0:
        return "方向稳定为正"
    if high < 0:
        return "方向稳定为负"
    if abs(mean) < 0.03:
        return "接近零且区间跨零"
    return "存在趋势，但区间跨零"


def interpretation(row: pd.Series) -> str:
    direct = float(row["s_minus_t_mean"])
    if abs(direct) < 0.03:
        dominant = "S 与 T 的单独影响接近"
    elif direct > 0:
        dominant = "只消融 S 的轨迹影响更大"
    else:
        dominant = "只消融 T 的轨迹影响更大"
    add_t = float(row["add_t_given_s_mean"])
    add_s = float(row["add_s_given_t_mean"])
    if abs(add_t) < 0.03 and abs(add_s) < 0.03:
        interaction = "联合消融几乎没有额外变化，表现为冗余或饱和"
    elif add_t > 0 and add_s > 0:
        interaction = "加入第二类 head 后 Impact 继续增大，表现为互补"
    elif add_t < 0 and add_s < 0:
        interaction = "加入第二类 head 后反而更接近 baseline，表现为抵消"
    elif add_t > 0 >= add_s:
        interaction = "T 加在 S 上增加影响，但 S 加在 T 上产生抵消"
    else:
        interaction = "S 加在 T 上增加影响，但 T 加在 S 上产生抵消"
    direct_confidence = confidence_text(
        float(row["s_minus_t_mean"]),
        float(row["s_minus_t_ci_low"]),
        float(row["s_minus_t_ci_high"]),
    )
    add_t_confidence = confidence_text(
        float(row["add_t_given_s_mean"]),
        float(row["add_t_given_s_ci_low"]),
        float(row["add_t_given_s_ci_high"]),
    )
    add_s_confidence = confidence_text(
        float(row["add_s_given_t_mean"]),
        float(row["add_s_given_t_ci_low"]),
        float(row["add_s_given_t_ci_high"]),
    )
    return (
        f"{dominant}；{interaction}。证据：S-T {direct_confidence}，"
        f"消S后再消T {add_t_confidence}，消T后再消S {add_s_confidence}。"
    )


def representative_seed(wide: pd.DataFrame) -> int:
    columns = ["S", "T", "ST"]
    values = wide[columns].to_numpy(float)
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-6] = 1.0
    distances = np.square((values - center) / scale).mean(axis=1)
    return int(wide.index[int(np.argmin(distances))])


def replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.resolve())


def save_impact_chart(stats: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    x = np.arange(len(STAGES))
    width = 0.24
    for axis, model in zip(axes, MODELS):
        subset = stats[stats["model"] == model].set_index("stage")
        for index, role in enumerate(ROLES):
            labels = [f"[{start},{end})" for start, end in STAGES]
            means = np.array([subset.loc[label, f"{role}_mean"] for label in labels])
            lows = np.array([subset.loc[label, f"{role}_ci_low"] for label in labels])
            highs = np.array([subset.loc[label, f"{role}_ci_high"] for label in labels])
            axis.bar(
                x + (index - 1) * width,
                means,
                width,
                color=ROLE_COLORS[role],
                label=ROLE_CHART_LABELS[role],
            )
            axis.errorbar(
                x + (index - 1) * width,
                means,
                yerr=np.vstack((means - lows, highs - means)),
                fmt="none",
                ecolor="#222",
                elinewidth=0.8,
                capsize=2,
            )
        axis.set_title(MODEL_LABELS[model])
        axis.set_ylabel("Impact")
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(x, [f"[{a},{b})" for a, b in STAGES])
    axes[-1].set_xlabel("Denoising step range")
    axes[0].legend(ncol=3, frameon=False, loc="upper right")
    fig.suptitle("Paired Impact relative to the same baseline (mean and 95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_effect_chart(stats: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    x = np.arange(len(STAGES))
    offsets = (-0.12, 0.0, 0.12)
    effect_names = ("s_minus_t", "add_t_given_s", "add_s_given_t")
    for axis, model in zip(axes, MODELS):
        subset = stats[stats["model"] == model].set_index("stage")
        for offset, name in zip(offsets, effect_names):
            labels = [f"[{start},{end})" for start, end in STAGES]
            means = np.array([subset.loc[label, f"{name}_mean"] for label in labels])
            lows = np.array([subset.loc[label, f"{name}_ci_low"] for label in labels])
            highs = np.array([subset.loc[label, f"{name}_ci_high"] for label in labels])
            axis.errorbar(
                x + offset,
                means,
                yerr=np.vstack((means - lows, highs - means)),
                fmt="o-",
                color=EFFECT_COLORS[name],
                linewidth=1.4,
                markersize=4,
                capsize=3,
                label=EFFECT_CHART_LABELS[name],
            )
        axis.axhline(0, color="#333", linewidth=1)
        axis.set_title(MODEL_LABELS[model])
        axis.set_ylabel("Paired Impact difference")
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(x, [f"[{a},{b})" for a, b in STAGES])
    axes[-1].set_xlabel("Denoising step range")
    axes[0].legend(ncol=3, frameon=False, loc="upper right")
    fig.suptitle("S/T dominance and conditional effects (mean and 95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fmt_effect(mean: float, low: float, high: float) -> str:
    return f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]"


def build_page(
    inventory: dict[str, Any],
    frame: pd.DataFrame,
    stats: pd.DataFrame,
    paired: dict[tuple[str, int, int], pd.DataFrame],
    output_dir: Path,
) -> list[dict[str, Any]]:
    entries = {entry["entry_id"]: entry for entry in inventory["entries"]}
    media_dir = output_dir / "media"
    manifest: list[dict[str, Any]] = []
    stage_sections = []
    for start, end in STAGES:
        rows = []
        videos = []
        for model in MODELS:
            stat = stats[
                (stats["model"] == model)
                & (stats["denoise_start"] == start)
                & (stats["denoise_end"] == end)
            ].iloc[0]
            rows.append(
                "<tr>"
                f"<th>{MODEL_LABELS[model]}</th>"
                f"<td>{int(stat['n_paired_seeds'])}</td>"
                f"<td>{stat['S_mean']:.3f}</td><td>{stat['T_mean']:.3f}</td>"
                f"<td>{stat['ST_mean']:.3f}</td>"
                f"<td>{fmt_effect(stat['s_minus_t_mean'], stat['s_minus_t_ci_low'], stat['s_minus_t_ci_high'])}</td>"
                f"<td>{fmt_effect(stat['add_t_given_s_mean'], stat['add_t_given_s_ci_low'], stat['add_t_given_s_ci_high'])}</td>"
                f"<td>{fmt_effect(stat['add_s_given_t_mean'], stat['add_s_given_t_ci_low'], stat['add_s_given_t_ci_high'])}</td>"
                "</tr>"
            )
            wide = paired[(model, start, end)]
            seed = representative_seed(wide)
            baseline_row = frame[
                (frame["model"] == model)
                & (frame["seed"] == seed)
                & (frame["variant"] == "baseline")
            ].iloc[0]
            selected_rows = {"baseline": baseline_row}
            for role in ROLES:
                selected_rows[role] = frame[
                    (frame["model"] == model)
                    & (frame["seed"] == seed)
                    & (frame["role"] == role)
                    & (frame["denoise_start"] == start)
                    & (frame["denoise_end"] == end)
                ].iloc[0]
            cells = []
            record = {
                "model": model,
                "stage": [start, end],
                "representative_seed": seed,
                "interpretation": interpretation(stat),
                "videos": [],
            }
            for key, label in (
                ("baseline", "Baseline"),
                ("S", "只消融 S"),
                ("T", "只消融 T"),
                ("ST", "联合消融 S+T"),
            ):
                row = selected_rows[key]
                entry = entries[row["entry_id"]]
                source = Path(entry["source"]["path"])
                filename = (
                    f"{model}__steps{start:02d}_{end:02d}"
                    f"__seed-{seed:06d}__{key}.mp4"
                )
                link = media_dir / filename
                replace_symlink(link, source)
                impact = float(row["impact_score"])
                cells.append(
                    '<article class="video-cell">'
                    f"<h4>{html.escape(label)}</h4>"
                    f"<p>{html.escape(str(row['variant']))}</p>"
                    f'<video controls preload="metadata" src="media/{filename}"></video>'
                    f'<div class="score">Impact <strong>{impact:.3f}</strong></div>'
                    "</article>"
                )
                record["videos"].append(
                    {
                        "kind": key,
                        "variant": str(row["variant"]),
                        "impact": impact,
                        "source": str(source),
                        "media": f"media/{filename}",
                    }
                )
            videos.append(
                '<div class="model-videos">'
                f'<div class="model-head"><h3>{MODEL_LABELS[model]}</h3>'
                f"<span>代表 seed {seed}</span><p>{html.escape(interpretation(stat))}</p></div>"
                f'<div class="video-grid">{"".join(cells)}</div></div>'
            )
            manifest.append(record)
        stage_sections.append(
            f'<section id="stage-{start}-{end}"><h2>去噪阶段 [{start},{end})</h2>'
            '<div class="table-wrap"><table><thead><tr><th>模型</th><th>配对 seeds</th>'
            "<th>S</th><th>T</th><th>S+T</th><th>S-T</th>"
            "<th>消S后再消T</th><th>消T后再消S</th></tr></thead>"
            f'<tbody>{"".join(rows)}</tbody></table></div>{"".join(videos)}</section>'
        )

    case = inventory["case"]
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>S/T 分阶段运动轨迹 Impact 分析</title>
<style>
:root{{--bg:#f3f5f7;--panel:#fff;--line:#d7dce2;--text:#20252b;--muted:#66717d;--accent:#1769aa}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Arial,sans-serif}}
.toolbar{{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--line);padding:10px 16px;display:flex;align-items:center;gap:9px}}
.toolbar h1{{font-size:18px;margin:0 10px 0 0}}button,select{{border:1px solid #aeb6c0;background:#fff;border-radius:5px;padding:7px 10px}}
button{{cursor:pointer}}main{{max-width:1580px;margin:auto;padding:18px}}.scope,.definition{{background:#fff;border-left:3px solid var(--accent);padding:10px 12px;margin-bottom:14px}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:24px}}.charts img{{width:100%;background:#fff;border:1px solid var(--line)}}
section{{margin:0 0 34px}}h2{{font-size:20px;margin:0 0 10px}}.table-wrap{{overflow-x:auto;background:#fff;border:1px solid var(--line)}}
table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:7px 9px;border-bottom:1px solid #e7eaee;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.model-videos{{margin-top:14px}}.model-head{{display:grid;grid-template-columns:auto auto 1fr;align-items:baseline;gap:10px;margin-bottom:6px}}
.model-head h3{{font-size:16px;margin:0}}.model-head span,.model-head p,.video-cell p{{color:var(--muted)}}.model-head p{{margin:0}}
.video-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}}
.video-cell{{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:8px}}.video-cell h4{{font-size:14px;margin:0}}
.video-cell p{{height:20px;margin:2px 0 5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
video{{display:block;width:100%;aspect-ratio:16/9;background:#111}}.score{{display:flex;justify-content:space-between;padding-top:6px}}
.muted{{color:var(--muted)}}@media(max-width:900px){{.charts{{grid-template-columns:1fr}}.video-grid{{grid-template-columns:1fr 1fr}}.model-head{{grid-template-columns:1fr}}}}
</style></head><body>
<div class="toolbar"><h1>S/T 分阶段 Impact 分析</h1>
<select id="stage"><option value="">选择阶段</option>{''.join(f'<option value="stage-{a}-{b}">[{a},{b})</option>' for a,b in STAGES)}</select>
<button id="play">播放全部</button><button id="replay">从头播放</button><button id="pause">暂停全部</button></div>
<main>
<div class="scope"><strong>证据范围：</strong>{html.escape(str(case.get('title') or case.get('id')))}；6 个 seeds，3 个模型。统计只保留同模型、同 seed、同阶段且 S/T/S+T 轨迹均有效的配对四元组。</div>
<div class="definition"><strong>唯一指标 Impact：</strong>同 seed 消融视频相对 baseline 的四项归一化运动误差经 log1p 后取均值。数值越大表示运动轨迹改变越强，不表示质量越好。表中括号为配对 bootstrap 95% CI；“消S后再消T”=Impact(S+T)-Impact(S)，另一列同理。</div>
<div class="definition"><strong>总体结论：</strong>最早期 [0,5) 由 S 主导，PhysRVG 的方向最稳定；[5,10) 开始模型分化，xSSC 偏 T、PhysRVG 偏 S、Wan 接近冗余；[10,20) Wan/xSSC 中 T 单独影响更强且 S+T 出现抵消，而 PhysRVG 仍呈互补；[20,30) S 再次占主导。由此更适合把 S 理解为运动结构约束，把 T 理解为模型相关的中期动态调节，而不是把两者视为可线性相加的独立模块。</div>
<div class="charts"><img src="impact_by_stage.png" alt="Impact 分阶段柱状图"><img src="paired_effects.png" alt="配对条件增量图"></div>
{''.join(stage_sections)}
<p class="muted">原始统计：<a href="paired_impact_statistics.csv">paired_impact_statistics.csv</a> · 页面选择：<a href="manifest.json">manifest.json</a></p>
</main><script>
const videos=[...document.querySelectorAll('video')];
document.getElementById('play').onclick=()=>videos.forEach(v=>v.play());
document.getElementById('replay').onclick=()=>videos.forEach(v=>{{v.currentTime=0;v.play()}});
document.getElementById('pause').onclick=()=>videos.forEach(v=>v.pause());
document.getElementById('stage').onchange=e=>{{if(e.target.value)document.getElementById(e.target.value).scrollIntoView()}};
</script></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    analysis_root = args.analysis_root.expanduser().resolve()
    output_dir = (
        args.gallery_root.expanduser().resolve() / "multiseed" / args.output_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((analysis_root / "inventory.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(analysis_root / "results" / "per_video_metrics.csv")
    stats, paired = compute_statistics(frame, args.bootstrap_samples)
    stats.to_csv(output_dir / "paired_impact_statistics.csv", index=False)
    save_impact_chart(stats, output_dir / "impact_by_stage.png")
    save_effect_chart(stats, output_dir / "paired_effects.png")
    manifest = build_page(inventory, frame, stats, paired, output_dir)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
