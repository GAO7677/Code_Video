#!/usr/bin/env python3
"""Audit AAAevalphysiq1 result provenance and plot 67-case metric curves."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "AAAevalphysiq1.txt"
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/"
    "xssc_object_self_attn_lora_visualizations/aaaevalphysiq1-metrics"
)
EXPECTED_CASES = 67


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    path: tuple[str, ...]
    direction: str = "higher"


METRICS = (
    Metric("physics_iq_with_context", "Physics-IQ ctx", ("physics_iq_with_context", "score")),
    Metric("physics_iq_without_context", "Physics-IQ no ctx", ("physics_iq_without_context", "score")),
    Metric("pmf_with_context", "PMF ctx", ("pmf_with_context", "score")),
    Metric("pmf_without_context", "PMF no ctx", ("pmf_without_context", "score")),
    Metric("wmreward", "WMReward surprise", ("wmreward", "surprise"), "lower"),
    Metric("videophy2_sa", "VideoPhy2 SA", ("videophy2", "sa_score")),
    Metric("videophy2_pc", "VideoPhy2 PC", ("videophy2", "pc_score")),
    Metric("videophy2_joint", "VideoPhy2 joint pass", ("videophy2", "joint_pass")),
    Metric("videophy2_pc_raw", "VideoPhy2 PC raw", ("videophy2", "pc_raw_score")),
    Metric("cosmos_reason1", "Cosmos-Reason1", ("cosmos_reason1", "score")),
    Metric("vbench_subject_consistency", "VBench subject", ("vbench_subject_consistency", "score")),
    Metric("vbench_background_consistency", "VBench background", ("vbench_background_consistency", "score")),
    Metric("vbench_temporal_flickering", "VBench flickering", ("vbench_temporal_flickering", "score")),
    Metric("vbench_motion_smoothness", "VBench smoothness", ("vbench_motion_smoothness", "score")),
    Metric("vbench_dynamic_degree", "VBench dynamic", ("vbench_dynamic_degree", "score")),
    Metric("vbench_aesthetic_quality", "VBench aesthetic", ("vbench_aesthetic_quality", "score")),
    Metric("vbench_imaging_quality", "VBench imaging", ("vbench_imaging_quality", "score")),
)


FAMILIES: dict[str, dict[str, str]] = {
    "dinov2_full": {
        "label": "DINOv2 xSSC, full 8-frame slots",
        "short": "DINOv2 full",
        "color": "#0072B2",
        "xssc_checkpoint": "/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/42-0130.pth",
        "xssc_config": "/home/gaoya/Code_Video/xSSC-main/config-randsfq/rsfq2_r-ytvis.py",
        "xssc_training_data": "YTVIS 2022 train (ytvis_2022/train.lmdb)",
        "xssc_shape": "[B,8,7,256] -> [B,56,3072]",
        "wan_training_data": "30% 0717 PyBullet + 30% PhyCo Kubric + 40% OpenVidHD",
    },
    "dinov2_pooled": {
        "label": "DINOv2 xSSC, mean-pooled context slots",
        "short": "DINOv2 pooled",
        "color": "#009E73",
        "xssc_checkpoint": "/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/42-0130.pth",
        "xssc_config": "/home/gaoya/Code_Video/xSSC-main/config-randsfq/rsfq2_r-ytvis.py",
        "xssc_training_data": "YTVIS 2022 train (ytvis_2022/train.lmdb)",
        "xssc_shape": "[B,8,7,256] -> mean(T) -> [B,7,3072]",
        "wan_training_data": "30% 0717 PyBullet + 30% PhyCo Kubric + 40% OpenVidHD",
    },
    "dinov3_full": {
        "label": "DINOv3 xSSC MOVi-C transfer, full 8-frame slots",
        "short": "DINOv3 MOVi-C",
        "color": "#D55E00",
        "xssc_checkpoint": (
            "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC/"
            "restart_save1000_20260720T140029Z/movi_c_transfer15000_b64_acc3_20260721T134713Z/"
            "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/42/step-026000.pth"
        ),
        "xssc_config": (
            "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/"
            "xssc_rsfq2_ytvis_dinov3_vitl16_256/upstream/config-randsfq/"
            "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000.py"
        ),
        "xssc_training_data": (
            "DINOv3 YTVIS-HQ step-015000 initialization, then MOVi-C train continuation "
            "to xSSC step-026000"
        ),
        "xssc_shape": "[B,8,11,512] -> [B,88,3072]",
        "wan_training_data": "30% 0717 PyBullet + 30% PhyCo Kubric + 40% OpenVidHD",
    },
}

DATASET_ROOTS = (
    ("30% 0717 PyBullet", "/data/gaoya/AAA_test_video/Dataset_physV/0717pybullet_5000_vbenchtop5"),
    ("30% PhyCo Kubric", "/data/gaoya/dataset/nnsriram97-phyco_kubric"),
    ("40% OpenVidHD", "/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train"),
)


@dataclass(frozen=True)
class Result:
    result_dir: Path
    family: str
    step: int
    prompt: str

    @property
    def series_key(self) -> str:
        return f"{self.family}:{self.prompt}"

    @property
    def label(self) -> str:
        return f"{FAMILIES[self.family]['short']} / {self.prompt}"


@dataclass(frozen=True)
class Stat:
    result: Result
    metric: Metric
    count: int
    mean: float | None
    ci95: float | None


SERIES_STYLES = {
    "dinov2_full:default": {"marker": "o", "linestyle": "-", "connect": True},
    "dinov2_pooled:default": {"marker": "s", "linestyle": "-", "connect": True},
    "dinov3_full:custom": {"marker": "D", "linestyle": "-", "connect": True},
    "dinov2_full:custom": {"marker": "X", "linestyle": "None", "connect": False},
    "dinov2_full:none": {"marker": "P", "linestyle": "None", "connect": False},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def classify(path: Path) -> Result:
    name = path.name
    step_match = re.search(r"step-(\d+)", name)
    if step_match is None:
        raise ValueError(f"No checkpoint step in result directory: {path}")
    if name.startswith("xssc_randomcrop_pooled"):
        family = "dinov2_pooled"
    elif name.startswith("dinov3_xssc_wan"):
        family = "dinov3_full"
    elif name.startswith("formal_mix49"):
        family = "dinov2_full"
    else:
        raise ValueError(f"Unknown result family: {path}")
    if "negpromptNone" in name:
        prompt = "none"
    elif "customprompt" in name:
        prompt = "custom"
    elif "defaultnegprompt" in name:
        prompt = "default"
    else:
        raise ValueError(f"Unknown prompt setting: {path}")
    return Result(path.resolve(), family, int(step_match.group(1)), prompt)


def load_results(path: Path) -> list[Result]:
    results = [
        classify(Path(line.strip()).expanduser())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    for result in results:
        if not result.result_dir.is_dir():
            raise FileNotFoundError(result.result_dir)
    return results


def nested_number(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def case_payloads(result_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(result_dir.glob("*.json")):
        if path.name.startswith("eval_summary_") or path.name in {
            "summary.json",
            "batch_manifest.json",
            "eval_summary.json",
        }:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("input_json"), str):
            payloads.append(payload)
    return payloads


def compute_stats(results: list[Result]) -> list[Stat]:
    stats: list[Stat] = []
    for result in results:
        payloads = case_payloads(result.result_dir)
        for metric in METRICS:
            values = [
                value
                for payload in payloads
                if (value := nested_number(payload, metric.path)) is not None
            ]
            mean = float(np.mean(values)) if values else None
            ci95 = (
                float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else None
            )
            stats.append(Stat(result, metric, len(values), mean, ci95))
    return stats


def write_stats(path: Path, stats: list[Stat]) -> None:
    fields = (
        "family", "family_label", "training_step", "prompt", "metric", "direction",
        "count", "expected_count", "complete", "mean", "ci95", "result_dir",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stat in stats:
            writer.writerow({
                "family": stat.result.family,
                "family_label": FAMILIES[stat.result.family]["label"],
                "training_step": stat.result.step,
                "prompt": stat.result.prompt,
                "metric": stat.metric.key,
                "direction": stat.metric.direction,
                "count": stat.count,
                "expected_count": EXPECTED_CASES,
                "complete": stat.count == EXPECTED_CASES,
                "mean": "" if stat.mean is None else f"{stat.mean:.8f}",
                "ci95": "" if stat.ci95 is None else f"{stat.ci95:.8f}",
                "result_dir": stat.result.result_dir,
            })


def write_provenance(path: Path, results: list[Result]) -> None:
    fields = (
        "result_dir", "family", "training_step", "prompt", "xssc_checkpoint",
        "xssc_config", "xssc_training_data", "wan_training_data", "xssc_shape",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            family = FAMILIES[result.family]
            writer.writerow({
                "result_dir": result.result_dir,
                "family": family["label"],
                "training_step": result.step,
                "prompt": result.prompt,
                "xssc_checkpoint": family["xssc_checkpoint"],
                "xssc_config": family["xssc_config"],
                "xssc_training_data": family["xssc_training_data"],
                "wan_training_data": family["wan_training_data"],
                "xssc_shape": family["xssc_shape"],
            })


def stat_lookup(stats: list[Stat]) -> dict[tuple[Path, str], Stat]:
    return {(stat.result.result_dir, stat.metric.key): stat for stat in stats}


def plot_curves(path: Path, results: list[Result], stats: list[Stat]) -> None:
    lookup = stat_lookup(stats)
    fig, axes = plt.subplots(6, 3, figsize=(17, 25))
    series = sorted({result.series_key for result in results})
    for metric, ax in zip(METRICS, axes.flat):
        for series_key in series:
            points = sorted(
                (result for result in results if result.series_key == series_key),
                key=lambda item: item.step,
            )
            values = [lookup[(point.result_dir, metric.key)] for point in points]
            valid = [(point, stat) for point, stat in zip(points, values) if stat.mean is not None]
            if not valid:
                continue
            style = SERIES_STYLES[series_key]
            family = FAMILIES[valid[0][0].family]
            xs = [point.step for point, _ in valid]
            ys = [stat.mean for _, stat in valid]
            es = [stat.ci95 or 0.0 for _, stat in valid]
            label = valid[0][0].label
            ax.errorbar(
                xs, ys, yerr=es, label=label, color=family["color"],
                marker=style["marker"], linestyle=style["linestyle"], linewidth=2,
                markersize=6, capsize=2.5, alpha=0.95,
            )
        ax.set_title(f"{metric.label} ({'higher' if metric.direction == 'higher' else 'lower'} is better)")
        ax.set_xlabel("Wan object-branch training step")
        ax.set_xticks([1000, 1500, 2000])
        ax.grid(True, alpha=0.22)
    for ax in axes.flat[len(METRICS):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.94, bottom=0.03, hspace=0.43, wspace=0.13)
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.972),
        ncol=3, frameon=False,
    )
    fig.suptitle(
        "AAAevalphysiq1: 67-case metric means with 95% CI",
        fontsize=17,
        y=0.992,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def esc(value: Any) -> str:
    return html.escape(str(value))


def write_html(path: Path, results: list[Result], stats: list[Stat]) -> None:
    lookup = stat_lookup(stats)
    family_rows = []
    for key, family in FAMILIES.items():
        family_rows.append(
            "<tr>"
            f"<td><b>{esc(family['short'])}</b></td>"
            f"<td><code>{esc(family['xssc_checkpoint'])}</code></td>"
            f"<td>{esc(family['xssc_training_data'])}</td>"
            f"<td>{esc(family['wan_training_data'])}</td>"
            f"<td><code>{esc(family['xssc_shape'])}</code></td>"
            "</tr>"
        )
    result_rows = []
    for index, result in enumerate(results, 1):
        counts = [lookup[(result.result_dir, metric.key)].count for metric in METRICS]
        result_rows.append(
            "<tr>"
            f"<td>{index}</td><td>{esc(FAMILIES[result.family]['short'])}</td>"
            f"<td>{result.step}</td><td>{esc(result.prompt)}</td>"
            f"<td>{min(counts)}/{EXPECTED_CASES} to {max(counts)}/{EXPECTED_CASES}</td>"
            f"<td><code>{esc(result.result_dir)}</code></td></tr>"
        )
    metric_header = "".join(f"<th>{esc(metric.label)}</th>" for metric in METRICS)
    metric_rows = []
    for result in results:
        cells = []
        for metric in METRICS:
            stat = lookup[(result.result_dir, metric.key)]
            value = "-" if stat.mean is None else f"{stat.mean:.4f}"
            cells.append(f"<td title='n={stat.count}'>{value}</td>")
        metric_rows.append(
            f"<tr><td>{esc(FAMILIES[result.family]['short'])}</td>"
            f"<td>{result.step}</td><td>{esc(result.prompt)}</td>{''.join(cells)}</tr>"
        )
    dataset_rows = "".join(
        f"<tr><td>{esc(name)}</td><td><code>{esc(root)}</code></td></tr>"
        for name, root in DATASET_ROOTS
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AAAevalphysiq1 指标与来源</title>
<style>
:root{{--ink:#172126;--muted:#607077;--line:#ccd5d8;--paper:#f7f9f8;--accent:#006d77}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.55 Arial,sans-serif}}
main{{max-width:1580px;margin:auto;padding:30px}}h1{{font-size:28px;margin:0 0 8px}}h2{{margin:28px 0 10px;font-size:19px}}
.note{{border-left:4px solid var(--accent);padding:10px 14px;background:#fff;margin:12px 0}}
.scroll{{overflow:auto;border:1px solid var(--line);background:#fff}}table{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{padding:7px 9px;border-bottom:1px solid #e3e8ea;text-align:left;vertical-align:top;white-space:nowrap}}th{{position:sticky;top:0;background:#eaf0f1}}
code{{font:11px/1.45 monospace;white-space:normal;word-break:break-all}}img{{width:100%;background:#fff;border:1px solid var(--line)}}
a{{color:#005f73}}.files a{{margin-right:18px}}
</style></head><body><main>
<h1>AAAevalphysiq1：xSSC 来源与指标曲线</h1>
<p>共读取 {len(results)} 个结果目录，每项指标按 case JSON 在 67 个 PhysicIQ case 上求算术平均；误差棒为 95% CI。</p>
<div class="note"><b>比较边界：</b>蓝线和绿线均使用 default negative prompt，可以直接比较训练 step；橙线使用 custom prompt。DINOv2 full 的 custom/none 两个点只用于 prompt 消融，不应并入 default 曲线解释训练趋势。</div>
<h2>三类 xSSC / Wan 训练来源</h2><div class="scroll"><table><thead><tr><th>分支</th><th>冻结 xSSC 权重</th><th>xSSC 自身训练数据</th><th>Wan 条件分支训练数据</th><th>推理形状</th></tr></thead><tbody>{''.join(family_rows)}</tbody></table></div>
<h2>Wan 三数据集的实际路径</h2><div class="scroll"><table><tbody>{dataset_rows}</tbody></table></div>
<h2>指标曲线</h2><img src="metric_curves.png" alt="metric curves">
<h2>结果目录映射</h2><div class="scroll"><table><thead><tr><th>#</th><th>分支</th><th>step</th><th>prompt</th><th>指标覆盖</th><th>结果目录</th></tr></thead><tbody>{''.join(result_rows)}</tbody></table></div>
<h2>67-case 平均指标</h2><div class="scroll"><table><thead><tr><th>分支</th><th>step</th><th>prompt</th>{metric_header}</tr></thead><tbody>{''.join(metric_rows)}</tbody></table></div>
<p class="files"><a href="metric_means.csv">指标 CSV</a><a href="result_provenance.csv">来源 CSV</a></p>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = load_results(args.input)
    stats = compute_stats(results)
    write_stats(args.output / "metric_means.csv", stats)
    write_provenance(args.output / "result_provenance.csv", results)
    plot_curves(args.output / "metric_curves.png", results, stats)
    write_html(args.output / "index.html", results, stats)
    incomplete = [stat for stat in stats if stat.count != EXPECTED_CASES]
    print(f"results={len(results)} metrics={len(METRICS)} incomplete={len(incomplete)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
