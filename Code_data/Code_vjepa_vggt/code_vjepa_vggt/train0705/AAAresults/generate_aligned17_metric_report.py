from __future__ import annotations

import csv
import html
import re
from pathlib import Path


INPUT_CSV = Path(
    "/data/gaoya/AAA_test_video/0623/test/report/v2v/groups/"
    "train_stage1b_diffsynth_native0705_0705_step007000_17/method_summary.csv"
)
OUTPUT_DIR = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/AAAresults"
)
CHART_DIR = OUTPUT_DIR / "charts"
OUTPUT_MD = OUTPUT_DIR / "train_stage1b_diffsynth_native0705_0705_aligned17_metrics.md"

METHOD_NAME_MAP = {
    "wan2p2_ti2v5B_aligned17": "wan2p2_base",
    "openvid_mixed_ctx24_384x672_lora_step-010000": "openvid_lora_10000",
    "raw_phys_state_wan_lora_continue_576x1024_f24_step-000500": "0613pybullet_lora_000500",
}

METRICS = [
    {
        "column": "wmreward_surprise_mean",
        "label": "WMReward Surprise",
        "direction": "lower",
        "arrow": "↓",
        "file_stem": "wmreward_surprise",
    },
    {
        "column": "physics_iq_score_mean",
        "label": "Physics-IQ Approx",
        "direction": "higher",
        "arrow": "↑",
        "file_stem": "physics_iq",
    },
    {
        "column": "videophy2_score_mean",
        "label": "VideoPhy2-PC",
        "direction": "higher",
        "arrow": "↑",
        "file_stem": "videophy2",
    },
    {
        "column": "phyground_general_avg_mean",
        "label": "PhyGround",
        "direction": "higher",
        "arrow": "↑",
        "file_stem": "phyground",
    },
    {
        "column": "cosmos_reason1_score_mean",
        "label": "Cosmos-Reason1",
        "direction": "higher",
        "arrow": "↑",
        "file_stem": "cosmos_reason1",
    },
]


def normalize_method_name(raw_method: str) -> str | None:
    if raw_method in METHOD_NAME_MAP:
        return METHOD_NAME_MAP[raw_method]

    match = re.search(r"(step-\d+)$", raw_method)
    if match:
        return match.group(1)
    return None


def method_sort_key(name: str) -> tuple[int, int]:
    prefix_order = {
        "wan2p2_base": (0, 0),
        "openvid_lora_10000": (1, 0),
        "0613pybullet_lora_000500": (2, 0),
    }
    if name in prefix_order:
        return prefix_order[name]
    match = re.fullmatch(r"step-(\d+)", name)
    if match:
        return (3, int(match.group(1)))
    return (99, 0)


def fmt(value: float) -> str:
    return f"{value:.6f}"


def is_best_value(value: float, best_value: float, tolerance: float = 1e-12) -> bool:
    return abs(value - best_value) <= tolerance


def format_metric_cell(value: float, best_value: float) -> str:
    text = fmt(value)
    if is_best_value(value, best_value):
        return f"**{text}** ★最佳"
    return text


def read_rows() -> list[dict[str, str]]:
    with INPUT_CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


def escape(text: str) -> str:
    return html.escape(text, quote=True)


def build_svg(
    method_names: list[str],
    metric_values: list[float],
    metric_label: str,
    arrow: str,
    best_index: int,
    output_path: Path,
) -> None:
    width = 1800
    height = 900
    margin_left = 110
    margin_right = 40
    margin_top = 80
    margin_bottom = 290
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    value_min = min(metric_values)
    value_max = max(metric_values)
    span = value_max - value_min
    pad = span * 0.08 if span > 0 else max(abs(value_min) * 0.05, 1.0)
    y_min = value_min - pad
    y_max = value_max + pad

    def x_pos(index: int) -> float:
        if len(method_names) == 1:
            return margin_left + plot_width / 2.0
        return margin_left + plot_width * index / (len(method_names) - 1)

    def y_pos(value: float) -> float:
        if y_max == y_min:
            return margin_top + plot_height / 2.0
        ratio = (value - y_min) / (y_max - y_min)
        return margin_top + plot_height * (1.0 - ratio)

    grid_lines = []
    tick_labels = []
    tick_count = 5
    for tick_index in range(tick_count + 1):
        ratio = tick_index / tick_count
        y = margin_top + plot_height * (1.0 - ratio)
        tick_value = y_min + (y_max - y_min) * ratio
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" '
            f'y2="{y:.2f}" stroke="#d9dde3" stroke-width="1"/>'
        )
        tick_labels.append(
            f'<text x="{margin_left - 12}" y="{y + 5:.2f}" text-anchor="end" '
            f'font-size="20" fill="#384152">{tick_value:.4f}</text>'
        )

    points = [(x_pos(i), y_pos(v)) for i, v in enumerate(metric_values)]
    polyline_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)

    point_elems = []
    value_elems = []
    for index, ((x, y), value) in enumerate(zip(points, metric_values)):
        fill = "#c62828" if index == best_index else "#1565c0"
        radius = 7 if index == best_index else 5
        point_elems.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}" />'
        )
        value_elems.append(
            f'<text x="{x:.2f}" y="{y - 12:.2f}" text-anchor="middle" '
            f'font-size="16" fill="#1f2937">{value:.4f}</text>'
        )

    x_label_elems = []
    x_tick_elems = []
    axis_y = margin_top + plot_height
    for index, method_name in enumerate(method_names):
        x = x_pos(index)
        x_tick_elems.append(
            f'<line x1="{x:.2f}" y1="{axis_y}" x2="{x:.2f}" y2="{axis_y + 8}" '
            f'stroke="#4b5563" stroke-width="1.2"/>'
        )
        x_label_elems.append(
            f'<text transform="translate({x:.2f},{axis_y + 28:.2f}) rotate(45)" '
            f'text-anchor="start" font-size="18" fill="#111827">{escape(method_name)}</text>'
        )

    title = f"{metric_label} ({arrow})"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{width / 2:.2f}" y="42" text-anchor="middle" font-size="32" fill="#111827">{escape(title)}</text>
<text x="{width / 2:.2f}" y="68" text-anchor="middle" font-size="18" fill="#6b7280">Red point marks the best value under the metric direction.</text>
{''.join(grid_lines)}
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{axis_y}" stroke="#4b5563" stroke-width="2"/>
<line x1="{margin_left}" y1="{axis_y}" x2="{width - margin_right}" y2="{axis_y}" stroke="#4b5563" stroke-width="2"/>
{''.join(tick_labels)}
{''.join(x_tick_elems)}
<polyline fill="none" stroke="#1565c0" stroke-width="3" points="{polyline_points}"/>
{''.join(point_elems)}
{''.join(value_elems)}
{''.join(x_label_elems)}
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_rows()
    normalized_rows = {}
    for row in rows:
        normalized_name = normalize_method_name(row["method"])
        if normalized_name is None:
            continue
        normalized_rows[normalized_name] = row

    ordered_names = sorted(normalized_rows, key=method_sort_key)
    ordered_rows = [normalized_rows[name] for name in ordered_names]

    list_paths = {row["matched_list_path"] for row in ordered_rows}
    dataset_sizes = {row["dataset_size"] for row in ordered_rows}
    if len(list_paths) != 1 or len(dataset_sizes) != 1:
        raise RuntimeError("Expected one shared dataset path and one shared dataset size.")

    dataset_path = next(iter(list_paths))
    dataset_size = next(iter(dataset_sizes))

    for metric in METRICS:
        values = [float(row[metric["column"]]) for row in ordered_rows]
        if metric["direction"] == "lower":
            best_index = min(range(len(values)), key=values.__getitem__)
        else:
            best_index = max(range(len(values)), key=values.__getitem__)
        build_svg(
            method_names=ordered_names,
            metric_values=values,
            metric_label=metric["label"],
            arrow=metric["arrow"],
            best_index=best_index,
            output_path=CHART_DIR / f"{metric['file_stem']}.svg",
        )

    best_metric_values = {}
    for metric in METRICS:
        values = [float(row[metric["column"]]) for row in ordered_rows]
        if metric["direction"] == "lower":
            best_metric_values[metric["column"]] = min(values)
        else:
            best_metric_values[metric["column"]] = max(values)

    lines = [
        "# train_stage1b_diffsynth_native0705_0705 aligned17 指标报告",
        "",
        "## 基本信息",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| 测试集路径 | `{dataset_path}` |",
        f"| 样本数量 | `{dataset_size}` |",
        f"| 汇总 CSV | `{INPUT_CSV}` |",
        f"| 方法顺序 | `{', '.join(ordered_names)}` |",
        "",
        "## 指标方向",
        "",
        "- `WMReward Surprise`: 越低越好",
        "- `Physics-IQ Approx`: 越高越好",
        "- `VideoPhy2-PC`: 越高越好",
        "- `PhyGround`: 越高越好",
        "- `Cosmos-Reason1`: 越高越好",
        "",
        "## 指标表格",
        "",
        "说明：表格中带 `★最佳` 的数值表示该指标当前最优结果；若有并列最优，会同时标出。",
        "",
        "| 方法 | 数量 | WMReward Surprise (↓) | Physics-IQ Approx (↑) | VideoPhy2-PC (↑) | PhyGround (↑) | Cosmos-Reason1 (↑) | 测试集路径 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for name, row in zip(ordered_names, ordered_rows):
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    row["num_cases"],
                    format_metric_cell(
                        float(row["wmreward_surprise_mean"]),
                        best_metric_values["wmreward_surprise_mean"],
                    ),
                    format_metric_cell(
                        float(row["physics_iq_score_mean"]),
                        best_metric_values["physics_iq_score_mean"],
                    ),
                    format_metric_cell(
                        float(row["videophy2_score_mean"]),
                        best_metric_values["videophy2_score_mean"],
                    ),
                    format_metric_cell(
                        float(row["phyground_general_avg_mean"]),
                        best_metric_values["phyground_general_avg_mean"],
                    ),
                    format_metric_cell(
                        float(row["cosmos_reason1_score_mean"]),
                        best_metric_values["cosmos_reason1_score_mean"],
                    ),
                    f"`{dataset_path}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 指标可视化折线图",
            "",
        ]
    )

    for metric in METRICS:
        lines.extend(
            [
                f"### {metric['label']} ({metric['arrow']})",
                "",
                f"![{metric['label']}](charts/{metric['file_stem']}.svg)",
                "",
            ]
        )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")
    for metric in METRICS:
        chart_path = CHART_DIR / f"{metric['file_stem']}.svg"
        print(f"Wrote {chart_path}")


if __name__ == "__main__":
    main()
