#!/usr/bin/env python3
"""Render aggregate pilot plots and a cautious machine-generated conclusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summarize_head_role_dose_control import PRIMARY_METRICS


MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
METRIC_LABELS = {
    "physics_iq_with_context": "Physics-IQ ctx",
    "pmf_with_context": "PMF ctx",
    "wmreward_surprise": "WMReward surprise",
    "videophy2_pc": "VideoPhy2 PC",
    "cosmos_reason1": "Cosmos-Reason1",
}
MATCH_LABELS = {
    "approx_depth": "k=8 approximate depth match",
    "exact_block": "k=5 exact same-block match",
}
ROLE_COLORS = {"S": "#3c8dbc", "T": "#d97943", "C": "#4ca66b"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def plot_model_match(frame: pd.DataFrame, model: str, matching: str, output: Path) -> None:
    selected = frame[(frame.model == model) & (frame.matching == matching)]
    stages = sorted(
        {
            (int(row.denoise_start), int(row.denoise_end))
            for row in selected.itertuples()
        }
    )
    fig, axes = plt.subplots(
        1, len(PRIMARY_METRICS), figsize=(18, 3.8), squeeze=False
    )
    x = np.arange(len(stages), dtype=float)
    for axis, metric in zip(axes[0], PRIMARY_METRICS):
        for role, offset in zip(("S", "T", "C"), (-0.22, 0.0, 0.22)):
            rows = selected[selected.role == role].set_index(
                ["denoise_start", "denoise_end"]
            )
            means, low, high = [], [], []
            for stage in stages:
                row = rows.loc[stage]
                means.append(float(row[f"{metric}_harm_mean"]))
                low.append(float(row[f"{metric}_harm_ci95_low"]))
                high.append(float(row[f"{metric}_harm_ci95_high"]))
            means_array = np.asarray(means)
            error = np.vstack((means_array - low, np.asarray(high) - means_array))
            axis.errorbar(
                x + offset,
                means_array,
                yerr=error,
                marker="o",
                capsize=3,
                linewidth=1.4,
                color=ROLE_COLORS[role],
                label=role,
            )
        axis.axhline(0, color="#555", linewidth=0.8)
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xticks(x, [f"{a}-{b}" for a, b in stages])
        axis.set_xlabel("denoise steps")
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("harm vs matched baseline (positive = worse)")
    axes[0, -1].legend(frameon=False)
    fig.suptitle(
        f"{MODEL_LABELS[model]} · {MATCH_LABELS[matching]}", fontweight="bold"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def conclusions(frame: pd.DataFrame, contrasts: pd.DataFrame) -> str:
    lines = [
        "# Pilot Result Interpretation",
        "",
        "这里的 harm 为正表示消融后指标相对同模型、同 seed、同 source case 的",
        "baseline 变差。下面只报告五项预先指定的主要指标，并将 k=8 近似深度匹配",
        "与 k=5 完全同 block 匹配分开。",
        "",
    ]
    for model in MODEL_LABELS:
        lines.extend((f"## {MODEL_LABELS[model]}", ""))
        for matching in MATCH_LABELS:
            subset = frame[
                (frame.model == model) & (frame.matching == matching)
            ]
            if subset.empty:
                continue
            lines.append(f"### {MATCH_LABELS[matching]}")
            stages = sorted(
                {
                    (int(row.denoise_start), int(row.denoise_end))
                    for row in subset.itertuples()
                }
            )
            for start, end in stages:
                stage = subset[
                    (subset.denoise_start == start)
                    & (subset.denoise_end == end)
                ]
                winners = []
                for metric in PRIMARY_METRICS:
                    valid = stage.dropna(subset=[f"{metric}_harm_mean"])
                    if valid.empty:
                        continue
                    row = valid.loc[valid[f"{metric}_harm_mean"].idxmax()]
                    winners.append(
                        f"{METRIC_LABELS[metric]}={row.role} "
                        f"({row[f'{metric}_harm_mean']:.4g}, "
                        f"CI [{row[f'{metric}_harm_ci95_low']:.4g}, "
                        f"{row[f'{metric}_harm_ci95_high']:.4g}])"
                    )
                lines.append(
                    f"- 去噪 {start}-{end}：各指标最大 harm 为 "
                    + "；".join(winners)
                    + "。"
                )
            lines.append("")
    strong = contrasts[
        contrasts.metric.isin(PRIMARY_METRICS)
        & ((contrasts.ci95_low > 0) | (contrasts.ci95_high < 0))
    ]
    lines.extend(
        (
            "## Matched Role Contrasts",
            "",
            "CI 不跨 0 的角色差值才列在这里；正值表示 contrast 左侧角色的消融",
            "比右侧造成更大 harm。这仍是探索性证据，不等同于独立验证集上的确认性结论。",
        )
    )
    if strong.empty:
        lines.append("- 五项主要指标中没有角色差值的 case-bootstrap 95% CI 排除 0。")
    else:
        for row in strong.itertuples():
            lines.append(
                f"- {MODEL_LABELS[row.model]} / {MATCH_LABELS[row.matching]} / "
                f"{int(row.denoise_start)}-{int(row.denoise_end)} / "
                f"{METRIC_LABELS[row.metric]}：{row.contrast}="
                f"{row.harm_difference_mean:.4g}，CI "
                f"[{row.ci95_low:.4g}, {row.ci95_high:.4g}]。"
            )
    lines.extend(
        (
            "",
            "## Limits",
            "",
            "- 这是数量和深度控制 pilot；尚未做输出能量匹配。",
            "- 当前 20 个 case 参与过 head 分类，后续需要冻结规则后在 held-out case 上复验。",
            "- 指标之间可能存在语义冲突，结论必须与页面中的同 case 视频配对检查结合。",
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    analysis = root / "analysis"
    frame = pd.read_csv(analysis / "role_harm_case_bootstrap.csv")
    contrasts = pd.read_csv(analysis / "matched_role_contrasts.csv")
    plot_root = analysis / "plots"
    for model in MODEL_LABELS:
        for matching in MATCH_LABELS:
            plot_model_match(
                frame,
                model,
                matching,
                plot_root / f"{model}_{matching}_primary_harm.png",
            )
    (analysis / "conclusions.md").write_text(
        conclusions(frame, contrasts), encoding="utf-8"
    )
    print(
        f"[dose-report] plots={len(MODEL_LABELS) * len(MATCH_LABELS)} "
        f"output={analysis}"
    )


if __name__ == "__main__":
    main()
