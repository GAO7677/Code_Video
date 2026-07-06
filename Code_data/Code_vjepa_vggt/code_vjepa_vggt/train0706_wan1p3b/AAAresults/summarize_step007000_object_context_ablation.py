from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


TESTSET_PATH = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
OUTPUT_MD = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/"
    "code_vjepa_vggt/train0706_wan1p3b/AAAresults/"
    "step007000_object_context_ablation_metrics.md"
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    result_dir: Path


METHODS = [
    MethodSpec(
        name="baseline",
        result_dir=Path(
            "/data/gaoya/AAA_test_video/0623/test/v2v/"
            "train_stage1b_diffsynth_native0705_0705/step-007000"
        ),
    ),
    MethodSpec(
        name="object_context_zero",
        result_dir=Path(
            "/data/gaoya/AAA_test_video/0623/test/v2v/"
            "train_stage1b_diffsynth_native0705_0705_object_context_zero/step-007000"
        ),
    ),
    MethodSpec(
        name="object_context_random",
        result_dir=Path(
            "/data/gaoya/AAA_test_video/0623/test/v2v/"
            "train_stage1b_diffsynth_native0705_0705_object_context_random/step-007000"
        ),
    ),
]


METRICS = [
    ("wmreward_surprise", "WMReward Surprise (↓)", ("wmreward", "surprise"), "lower"),
    ("physics_iq_score", "Physics-IQ Approx (↑)", ("physics_iq", "physics_iq_score"), "higher"),
    ("videophy2_score", "VideoPhy2-PC (↑)", ("videophy2", "score"), "higher"),
    ("phyground_general_avg", "PhyGround (↑)", ("phyground", "general_avg"), "higher"),
    ("cosmos_reason1_score", "Cosmos-Reason1 (↑)", ("cosmos_reason1", "score"), "higher"),
]


def load_case_payloads(result_dir: Path) -> list[dict]:
    payloads = []
    required_metric_keys = [key_path[0] for _, _, key_path, _ in METRICS]
    for path in sorted(result_dir.glob("*.json")):
        if path.name in {"result.json", "summary.json"}:
            continue
        if path.name.startswith("eval_summary_"):
            continue
        payload = json.loads(path.read_text())
        if all(metric_key in payload for metric_key in required_metric_keys):
            payloads.append(payload)
    return payloads


def get_nested_value(payload: dict, keys: tuple[str, ...]) -> float:
    current = payload
    for key in keys:
        current = current[key]
    return float(current)


def format_metric_cell(value: float, best_value: float, direction: str) -> str:
    is_best = abs(value - best_value) <= 1e-12
    text = f"{value:.6f}"
    if is_best:
        return f"**{text}**"
    return text


def compute_best_value(values: list[float], direction: str) -> float:
    if direction == "lower":
        return min(values)
    if direction == "higher":
        return max(values)
    raise ValueError(f"Unsupported direction: {direction}")


def main() -> None:
    rows = []
    for method in METHODS:
        payloads = load_case_payloads(method.result_dir)
        if not payloads:
            raise RuntimeError(f"No case json found under {method.result_dir}")

        row = {
            "method": method.name,
            "result_dir": str(method.result_dir),
            "num_cases": len(payloads),
        }
        for metric_key, _label, key_path, _direction in METRICS:
            values = [get_nested_value(payload, key_path) for payload in payloads]
            row[metric_key] = mean(values)
        rows.append(row)

    best_values = {}
    for metric_key, _label, _key_path, direction in METRICS:
        best_values[metric_key] = compute_best_value(
            [float(row[metric_key]) for row in rows],
            direction,
        )

    lines = [
        "# step-007000 object_context ablation 指标对比",
        "",
        "## 基本信息",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| 测试集路径 | `{TESTSET_PATH}` |",
        f"| 方法数量 | `{len(rows)}` |",
        f"| 每组样本数 | `{', '.join(str(row['num_cases']) for row in rows)}` |",
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
        "说明：加粗表示该指标在这三组里的当前最优值。",
        "",
        "| 方法 | 数量 | WMReward Surprise (↓) | Physics-IQ Approx (↑) | VideoPhy2-PC (↑) | PhyGround (↑) | Cosmos-Reason1 (↑) | 结果目录 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    str(row["num_cases"]),
                    format_metric_cell(
                        float(row["wmreward_surprise"]),
                        best_values["wmreward_surprise"],
                        "lower",
                    ),
                    format_metric_cell(
                        float(row["physics_iq_score"]),
                        best_values["physics_iq_score"],
                        "higher",
                    ),
                    format_metric_cell(
                        float(row["videophy2_score"]),
                        best_values["videophy2_score"],
                        "higher",
                    ),
                    format_metric_cell(
                        float(row["phyground_general_avg"]),
                        best_values["phyground_general_avg"],
                        "higher",
                    ),
                    format_metric_cell(
                        float(row["cosmos_reason1_score"]),
                        best_values["cosmos_reason1_score"],
                        "higher",
                    ),
                    f"`{row['result_dir']}`",
                ]
            )
            + " |"
        )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
