#!/usr/bin/env python3
"""Build static, case-grouped comparison pages for the three-model experiment."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import cv2
import numpy as np


OUTPUTS = Path("/data/gaoya/agent-data/outputs")
DEFAULT_DATA = OUTPUTS / "three_model_denoising_trajectory_comparison"
DEFAULT_OUTPUT = OUTPUTS / "three_model_case_comparison_simple"
MODEL_ROOTS = {
    "stage1b": OUTPUTS / "stage1b_kubric_step004000_sam2_regions_steps40",
    "lora": OUTPUTS / "wan_openvid_0613pybullet_lora_step000500_sam2_regions_steps40",
    "baseline": OUTPUTS / "wan22_ti2v_5b_baseline_sam2_regions_steps40",
}
MODEL_LABELS = {
    "stage1b": "Stage1b step-004000",
    "lora": "LoRA step-000500",
    "baseline": "Wan2.2 baseline",
}
MODEL_ORDER = ("stage1b", "lora", "baseline")
TRACK_KEY = "qk/L23/S039"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def link_assets(output: Path, name: str, target: Path) -> None:
    link = output / name
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"asset path exists and is not a symlink: {link}")
    os.symlink(target.resolve(), link, target_is_directory=True)


def point_colors(count: int) -> list[tuple[int, int, int]]:
    colors = []
    for index in range(count):
        hue = int(round(179 * index / max(count, 1)))
        value = cv2.cvtColor(
            np.uint8([[[hue, 210, 245]]]), cv2.COLOR_HSV2BGR
        )[0, 0]
        colors.append(tuple(int(channel) for channel in value))
    return colors


def draw_dashed_line(
    canvas: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    dash: float = 9.0,
) -> None:
    delta = end.astype(np.float32) - start.astype(np.float32)
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return
    direction = delta / length
    position = 0.0
    while position < length:
        segment_end = min(position + dash, length)
        p0 = tuple(np.rint(start + direction * position).astype(int))
        p1 = tuple(np.rint(start + direction * segment_end).astype(int))
        cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)
        position += dash * 1.75


def read_final_frame(video_path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_count - 1, 0))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"cannot read final frame: {video_path}")
    return frame


def render_trajectory_image(
    frame: np.ndarray,
    predictions: np.ndarray,
    gt_tracks: np.ndarray,
    visibility: np.ndarray,
    anchors: np.ndarray,
    query_latent_index: int,
    label: str,
    output_path: Path,
) -> None:
    canvas = cv2.addWeighted(frame, 0.78, np.zeros_like(frame), 0.22, 0)
    colors = point_colors(predictions.shape[1])
    for point_index, color in enumerate(colors):
        for frame_index in range(5, len(gt_tracks)):
            if not (visibility[frame_index - 1, point_index] and visibility[frame_index, point_index]):
                continue
            p0 = tuple(np.rint(gt_tracks[frame_index - 1, point_index]).astype(int))
            p1 = tuple(np.rint(gt_tracks[frame_index, point_index]).astype(int))
            cv2.line(canvas, p0, p1, (15, 20, 18), 4, cv2.LINE_AA)
            cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
        for latent_index in range(query_latent_index + 1, len(anchors)):
            p0 = predictions[latent_index - 1, point_index]
            p1 = predictions[latent_index, point_index]
            draw_dashed_line(canvas, p0, p1, (12, 16, 14), 5)
            draw_dashed_line(canvas, p0, p1, color, 2)
        if visibility[-1, point_index]:
            gt_point = tuple(np.rint(gt_tracks[-1, point_index]).astype(int))
            cv2.circle(canvas, gt_point, 6, (10, 14, 12), -1, cv2.LINE_AA)
            cv2.circle(canvas, gt_point, 4, color, -1, cv2.LINE_AA)
        pred_point = tuple(np.rint(predictions[-1, point_index]).astype(int))
        cv2.rectangle(
            canvas,
            (pred_point[0] - 6, pred_point[1] - 6),
            (pred_point[0] + 6, pred_point[1] + 6),
            (10, 14, 12),
            4,
        )
        cv2.rectangle(
            canvas,
            (pred_point[0] - 5, pred_point[1] - 5),
            (pred_point[0] + 5, pred_point[1] + 5),
            color,
            2,
        )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (16, 22, 19), -1)
    cv2.putText(
        canvas,
        label,
        (13, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (245, 242, 229),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "solid/circle: CoTracker   dashed/square: Q/K argmax",
        (13, canvas.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (245, 242, 229),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"failed to write {output_path}")


def scoped_metric(case: dict, model_name: str, scope: str) -> dict | None:
    model = next(item for item in case["models"] if item["name"] == model_name)
    rows = []
    for region_name, metric in model["metrics"][TRACK_KEY].items():
        selected = region_name.startswith("object_") if scope == "objects" else region_name == scope
        if selected and int(metric["comparisons"]) > 0:
            rows.append(metric)
    count = sum(int(row["comparisons"]) for row in rows)
    if count == 0:
        return None
    result = {"comparisons": count}
    for key in ("pck32", "mean_error_px", "motion_error_px"):
        usable = [row for row in rows if row.get(key) is not None]
        denominator = sum(int(row["comparisons"]) for row in usable)
        result[key] = (
            sum(float(row[key]) * int(row["comparisons"]) for row in usable) / denominator
            if denominator
            else None
        )
    return result


def aggregate(cases: list[dict], model_name: str, scope: str) -> dict:
    rows = [scoped_metric(case, model_name, scope) for case in cases]
    rows = [row for row in rows if row is not None]
    count = sum(int(row["comparisons"]) for row in rows)
    return {
        "valid_cases": len(rows),
        "comparisons": count,
        "pck32": sum(row["pck32"] * row["comparisons"] for row in rows) / count,
        "mean_error_px": sum(row["mean_error_px"] * row["comparisons"] for row in rows) / count,
        "motion_error_px": sum(row["motion_error_px"] * row["comparisons"] for row in rows) / count,
    }


def paired_bootstrap(cases: list[dict], first: str, second: str) -> dict:
    differences = []
    for case in cases:
        left = scoped_metric(case, first, "objects")
        right = scoped_metric(case, second, "objects")
        if left is not None and right is not None:
            differences.append(left["pck32"] - right["pck32"])
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(123)
    samples = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return {
        "pair": f"{first}-{second}",
        "cases": int(len(values)),
        "mean_pp": float(values.mean()),
        "median_pp": float(np.median(values)),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "wins": int((values > 0).sum()),
        "ties": int((values == 0).sum()),
        "losses": int((values < 0).sum()),
    }


STYLE = """
:root{--paper:#eee9dc;--ink:#17201d;--card:#fffdf7;--line:#c6bdab;--rust:#b3442c;--green:#176654;--muted:#66716b}*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#edd7c5 0,#eee9dc 35%,#dbe7df 100%) fixed;color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(1500px,calc(100% - 28px));margin:auto;padding:28px 0 64px}h1,h2,h3{font-family:Georgia,"Noto Serif CJK SC",serif;margin:0}h1{font-size:clamp(38px,6vw,78px);line-height:.94;letter-spacing:-.045em}.eyebrow{color:var(--rust);font-size:12px;font-weight:900;letter-spacing:.14em;text-transform:uppercase}.lead{max-width:1050px;color:var(--muted);line-height:1.6}.panel{background:#fffdf7ed;border:1px solid var(--line);padding:16px;box-shadow:0 14px 36px #21302717}.models{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.four{grid-template-columns:repeat(4,1fr)}.model{min-width:0}.model h3{font-size:18px;margin:0 0 8px}.model video,.model img{display:block;width:100%;background:#0d1411;aspect-ratio:7/4;object-fit:contain}.metric{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-top:7px}.metric div{background:#edf0e9;padding:8px}.metric span{display:block;color:var(--muted);font-size:9px;text-transform:uppercase}.metric b{font:700 19px/1.1 Georgia}.region{margin-top:14px}.region h2{margin-bottom:9px}.nav{display:flex;justify-content:space-between;gap:12px;margin:14px 0}.nav a,.case-link{color:var(--green);font-weight:800}.summary{overflow:auto;margin-top:18px}table{width:100%;border-collapse:collapse;background:var(--card)}th,td{border:1px solid var(--line);padding:8px;text-align:right;font-size:12px}th:first-child,td:first-child{text-align:left}.cases{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:16px}.case-link{display:grid;grid-template-columns:1fr repeat(3,70px);gap:8px;background:var(--card);border:1px solid var(--line);padding:10px;text-decoration:none}.case-link span:not(:first-child){text-align:right}.note{font-size:12px;color:var(--muted);line-height:1.55}.conclusion{border-left:5px solid var(--rust);margin-top:14px}.prompt{color:var(--muted)}@media(max-width:1000px){.models,.four,.cases{grid-template-columns:1fr}.case-link{grid-template-columns:1fr repeat(3,60px)}}
"""


def metric_cards(metric: dict | None) -> str:
    if metric is None:
        return '<div class="metric"><div><span>Valid</span><b>NA</b></div></div>'
    return (
        '<div class="metric">'
        f'<div><span>PCK@32</span><b>{metric["pck32"]:.1f}%</b></div>'
        f'<div><span>Error</span><b>{metric["mean_error_px"]:.1f}px</b></div>'
        f'<div><span>Motion</span><b>{metric["motion_error_px"]:.1f}px</b></div>'
        "</div>"
    )


def source_url(path: str) -> str:
    resolved = Path(path).resolve()
    return "/" + resolved.relative_to("/data/gaoya").as_posix()


def build_case_page(
    case: dict,
    output: Path,
    previous_case: str | None,
    next_case: str | None,
) -> None:
    case_key = case["case_key"]
    manifest = json.loads(
        (MODEL_ROOTS["stage1b"] / "cases" / case_key / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    navigation = '<a href="../index.html">All cases</a>'
    if previous_case:
        navigation += f'<a href="{previous_case}.html">Previous</a>'
    if next_case:
        navigation += f'<a href="{next_case}.html">Next</a>'
    source = source_url(manifest["context_video"])
    video_cards = [
        f'<article class="model"><h3>Source / GT</h3><video controls preload="metadata" src="{source}"></video></article>'
    ]
    for model_name in MODEL_ORDER:
        video = f"../{model_name}/cases/{case_key}/generated.mp4"
        poster = f"../{model_name}/cases/{case_key}/query_points.png"
        video_cards.append(
            f'<article class="model"><h3>{MODEL_LABELS[model_name]}</h3>'
            f'<video controls preload="metadata" poster="{poster}" src="{video}"></video></article>'
        )
    region_sections = []
    for region in case["regions"]:
        region_name = region["region_name"]
        cards = []
        for model_name in MODEL_ORDER:
            model = next(item for item in case["models"] if item["name"] == model_name)
            metric = model["metrics"][TRACK_KEY][region_name]
            image = f"../assets/{case_key}/{model_name}/{region_name}.jpg"
            cards.append(
                f'<article class="model"><h3>{MODEL_LABELS[model_name]}</h3>'
                f'<img loading="lazy" src="{image}" alt="{model_name} {region_name} trajectory">'
                f'{metric_cards(metric if metric["comparisons"] else None)}</article>'
            )
        phrase = region.get("region_phrase") or "non-object area"
        region_sections.append(
            f'<section class="panel region"><h2>{html.escape(region_name)} · {html.escape(phrase)}</h2>'
            f'<div class="models">{"".join(cards)}</div></section>'
        )
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{case_key}</title><style>{STYLE}</style></head><body><main><div class="nav">{navigation}</div><header><div class="eyebrow">Same source case · fixed Q/K L23 S39</div><h1>{html.escape(case_key)}</h1><p class="prompt">{html.escape(case["prompt"])}</p></header><section class="panel"><div class="models four">{"".join(video_cards)}</div></section>{"".join(region_sections)}<p class="note">所有模型使用 source frame 4 的同一组 SAM2 query 点。轨迹图背景是各自生成视频的最后一帧；实线圆点为该视频上的 CoTracker，虚线方点为 Q/K 最大相关 token。PCK 和误差仅统计可见未来 anchor。</p></main></body></html>'''
    (output / "cases" / f"{case_key}.html").write_text(page, encoding="utf-8")


def build_report(cases: list[dict], summary: dict, paired: list[dict], output: Path) -> None:
    lines = [
        "# 三模型跨帧 token 对应实验结果",
        "",
        "固定比较配置：query 为 source/context frame 4；Q/K matching；Transformer L23；denoising S39；latent anchors 为 `[0,4,8,12,16,20,24]`。",
        "",
        "## 汇总结果",
        "",
        "| 区域 | 模型 | PCK@32 | Mean error | Motion error | 有效 case | 比较数 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scope in ("objects", "background"):
        for model_name in MODEL_ORDER:
            row = summary[scope][model_name]
            lines.append(
                f"| {scope} | {MODEL_LABELS[model_name]} | {row['pck32']:.1f}% | "
                f"{row['mean_error_px']:.1f}px | {row['motion_error_px']:.1f}px | "
                f"{row['valid_cases']} | {row['comparisons']} |"
            )
    lines.extend(
        [
            "",
            "## 配对 source-case 差值",
            "",
            "| 对比 | case | PCK 差值 | bootstrap 95% CI | 胜/平/负 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {row['pair']} | {row['cases']} | {row['mean_pp']:.2f} pp | "
            f"[{row['ci95'][0]:.2f}, {row['ci95'][1]:.2f}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "1. Stage1b 与 LoRA 的物体 Q/K 对应准确率均约 96%，明显高于原始 baseline 的 83.9%；配对 source-case 平均优势分别为 12.17 和 11.17 个百分点。",
            "2. Stage1b 与 LoRA 之间没有可靠胜负：差值约 1.01 个百分点，95% bootstrap CI 为 [-0.72, 2.89]，跨过 0。",
            "3. Stage1b 的物体运动误差最低（6.5px），LoRA 为 9.3px，baseline 为 24.8px；说明训练后的模型不仅 top-1 位置更准，轨迹位移也更连续。",
            "4. 对应能力主要在最后去噪步突然形成，而非平滑出现；Q/K 的有效信息集中在中层 L5/L23，首层与末层明显较弱。",
            "5. baseline 的有效物体比较数更少（2597 vs. 3353/3293），可能表示物体更易消失、遮挡或变得难以跟踪；因为 PCK 排除了不可见点，baseline 的 83.9% 反而可能偏乐观。",
            "6. 本实验只证明模型内部存在更可靠的跨帧对应，不能单独证明生成视频物理合理；仍需将轨迹异常与碰撞、速度突变和几何约束联合评估。",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output = args.output_dir.resolve()
    (output / "cases").mkdir(parents=True, exist_ok=True)
    (output / "assets").mkdir(parents=True, exist_ok=True)
    for model_name, root in MODEL_ROOTS.items():
        link_assets(output, model_name, root)

    case_paths = sorted((data_root / "cases").glob("case_*.json"))
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in case_paths]
    if len(cases) != 50:
        raise RuntimeError(f"expected 50 cases, got {len(cases)}")

    for case in cases:
        case_key = case["case_key"]
        for model in case["models"]:
            model_name = model["name"]
            frame = read_final_frame(MODEL_ROOTS[model_name] / "cases" / case_key / "generated.mp4")
            anchors = np.asarray(model["anchors"], dtype=np.int64)
            for region in case["regions"]:
                region_name = region["region_name"]
                render_trajectory_image(
                    frame,
                    np.asarray(model["predictions"][TRACK_KEY][region_name], dtype=np.float32),
                    np.asarray(model["gt_tracks"][region_name], dtype=np.float32),
                    np.asarray(model["visibility"][region_name], dtype=bool),
                    anchors,
                    int(model["query_latent_index"]),
                    f"{MODEL_LABELS[model_name]} | {region_name} | Q/K L23 S39",
                    output / "assets" / case_key / model_name / f"{region_name}.jpg",
                )

    summary = {
        scope: {model_name: aggregate(cases, model_name, scope) for model_name in MODEL_ORDER}
        for scope in ("objects", "background")
    }
    paired = [
        paired_bootstrap(cases, "stage1b", "baseline"),
        paired_bootstrap(cases, "lora", "baseline"),
        paired_bootstrap(cases, "stage1b", "lora"),
    ]
    for index, case in enumerate(cases):
        build_case_page(
            case,
            output,
            cases[index - 1]["case_key"] if index else None,
            cases[index + 1]["case_key"] if index + 1 < len(cases) else None,
        )

    rows = []
    for case in cases:
        values = []
        for model_name in MODEL_ORDER:
            metric = scoped_metric(case, model_name, "objects")
            values.append("NA" if metric is None else f"{metric['pck32']:.1f}")
        rows.append(
            f'<a class="case-link" href="cases/{case["case_key"]}.html">'
            f'<span>{html.escape(case["case_key"])}</span>'
            f'<span>{values[0]}</span><span>{values[1]}</span><span>{values[2]}</span></a>'
        )
    table_rows = []
    for scope in ("objects", "background"):
        for model_name in MODEL_ORDER:
            row = summary[scope][model_name]
            table_rows.append(
                f'<tr><td>{scope}</td><td>{MODEL_LABELS[model_name]}</td>'
                f'<td>{row["pck32"]:.1f}%</td><td>{row["mean_error_px"]:.1f}px</td>'
                f'<td>{row["motion_error_px"]:.1f}px</td><td>{row["comparisons"]}</td></tr>'
            )
    index_html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Three-model case comparison</title><style>{STYLE}</style></head><body><main><header><div class="eyebrow">Simple comparison · Q/K L23 S39 · query frame 4</div><h1>Same Case,<br>Three Models</h1><p class="lead">每个 case 页面将 Source/GT、Stage1b、LoRA 和原始 Wan2.2 baseline 放在同一行；下方按 object A、object B、background 展示固定配置的轨迹和指标。</p></header><section class="panel conclusion"><h2>直接结论</h2><p>Stage1b 与 LoRA 的物体 PCK@32 均约 96%，baseline 为 83.9%。Stage1b 的物体 motion error 最低（6.5px），LoRA 为 9.3px，baseline 为 24.8px。Stage1b 与 LoRA 的配对差值置信区间跨过 0，因此二者没有可靠胜负，但都稳定优于 baseline。</p></section><section class="summary"><table><tr><th>Region</th><th>Model</th><th>PCK@32</th><th>Error</th><th>Motion</th><th>Comparisons</th></tr>{"".join(table_rows)}</table></section><h2 style="margin-top:22px">Cases · object PCK@32</h2><div class="case-link"><span>Case</span><span>Stage1b</span><span>LoRA</span><span>Base</span></div><div class="cases">{"".join(rows)}</div><p class="note">圆/实线为 CoTracker 伪 GT，方框/虚线为 Q/K argmax。完整分析见 <a href="RESULTS.md">RESULTS.md</a>。</p></main></body></html>'''
    (output / "index.html").write_text(index_html, encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps({"protocol": TRACK_KEY, "summary": summary, "paired": paired}, indent=2) + "\n",
        encoding="utf-8",
    )
    build_report(cases, summary, paired, output)
    print(f"built {len(cases)} static case pages: {output / 'index.html'}")


if __name__ == "__main__":
    main()
