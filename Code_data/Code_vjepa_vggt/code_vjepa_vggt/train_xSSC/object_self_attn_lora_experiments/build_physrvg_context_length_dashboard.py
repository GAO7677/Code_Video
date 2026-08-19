#!/usr/bin/env python3
"""Build the independent PhysRVG context-length gallery and metric pages."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_HUB_ROOT = Path("/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub")

METRICS = (
    ("physics_iq_with_context", "Physics-IQ · 含条件帧", ("physics_iq_with_context", "score"), "higher"),
    ("physics_iq_without_context", "Physics-IQ · 去条件帧", ("physics_iq_without_context", "score"), "higher"),
    ("pmf_with_context", "PMF · 含条件帧", ("pmf_with_context", "score"), "higher"),
    ("pmf_without_context", "PMF · 去条件帧", ("pmf_without_context", "score"), "higher"),
    ("wmreward", "WMReward surprise", ("wmreward", "surprise"), "lower"),
    ("vbench_subject_consistency", "VBench · 主体一致性", ("vbench_subject_consistency", "score"), "higher"),
    ("vbench_background_consistency", "VBench · 背景一致性", ("vbench_background_consistency", "score"), "higher"),
    ("vbench_temporal_flickering", "VBench · 时序闪烁", ("vbench_temporal_flickering", "score"), "higher"),
    ("vbench_motion_smoothness", "VBench · 运动平滑", ("vbench_motion_smoothness", "score"), "higher"),
    ("vbench_dynamic_degree", "VBench · 动态程度", ("vbench_dynamic_degree", "score"), "higher"),
    ("vbench_aesthetic_quality", "VBench · 美学质量", ("vbench_aesthetic_quality", "score"), "higher"),
    ("vbench_imaging_quality", "VBench · 成像质量", ("vbench_imaging_quality", "score"), "higher"),
    ("videophy2_sa", "VideoPhy2 · SA", ("videophy2", "sa_score"), "higher"),
    ("videophy2_pc", "VideoPhy2 · PC", ("videophy2", "pc_score"), "higher"),
    ("videophy2_joint", "VideoPhy2 · Joint", ("videophy2", "joint_rate"), "higher"),
    ("cosmos_reason1", "Cosmos-Reason1", ("cosmos_reason1", "score"), "higher"),
)

COLORS = {
    1: "#9b4dca",
    2: "#3f78b5",
    4: "#168a77",
    5: "#d28b28",
    6: "#d05a48",
    8: "#315c87",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--hub-root", type=Path, default=DEFAULT_HUB_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def link_directory(source: Path, destination: Path) -> None:
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(f"refusing to replace non-symlink directory: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(source, target_is_directory=True)
    os.replace(temporary, destination)


def link_file(source: Path, destination: Path) -> None:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(f"refusing to replace existing file: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(source)
    os.replace(temporary, destination)


def nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = payload
    for part in path:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_inputs(path: Path) -> list[Path]:
    return [
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def read_records(sweep_root: Path, dataset: str) -> list[dict[str, Any]]:
    manifest = load_json(sweep_root / "sweep_manifest.json")
    records = [
        row
        for row in manifest.get("records", [])
        if row.get("dataset") == dataset
    ]
    return sorted(records, key=lambda row: int(row["context_frames"]))


def case_payloads(records: list[dict[str, Any]], inputs: list[Path]) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        context_length = int(record["context_frames"])
        root = Path(record["result_root"]).resolve()
        cases: dict[str, dict[str, Any]] = {}
        for input_path in inputs:
            stem = input_path.stem
            result_json = root / f"{stem}.json"
            video = root / f"{stem}.mp4"
            if not result_json.is_file() or not video.is_file():
                continue
            try:
                payload = load_json(result_json)
            except Exception:
                continue
            cases[stem] = {
                "stem": stem,
                "payload": payload,
                "video": video,
                "input": input_path,
            }
        result[context_length] = cases
    return result


def make_media_links(
    page_root: Path,
    records: list[dict[str, Any]],
    cases_by_length: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, str]:
    media_root = page_root / "media"
    media_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        context_length = int(record["context_frames"])
        link_directory(
            Path(record["result_root"]),
            media_root / f"ctx{context_length:02d}",
        )
    case_media: dict[str, str] = {}
    for cases in cases_by_length.values():
        for stem, item in cases.items():
            source = item["payload"].get("input", {}).get("video_path")
            if not isinstance(source, str):
                continue
            source_path = Path(source).expanduser().resolve()
            destination = media_root / "_source" / stem / "context_video.mp4"
            if source_path.is_file():
                link_file(source_path, destination)
                case_media[stem] = destination.relative_to(page_root).as_posix()
    return case_media


def metric_means(cases: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for key, _, path, _ in METRICS:
        values = [
            value
            for item in cases.values()
            if (value := nested_value(item["payload"], path)) is not None
        ]
        output[key] = mean(values) if values else None
    return output


def paired_deltas(
    current: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]]
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for key, _, path, _ in METRICS:
        values = []
        for stem in sorted(set(current) & set(baseline)):
            current_value = nested_value(current[stem]["payload"], path)
            baseline_value = nested_value(baseline[stem]["payload"], path)
            if current_value is not None and baseline_value is not None:
                values.append(current_value - baseline_value)
        output[key] = mean(values) if values else None
    return output


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def video_card(
    context_length: int,
    item: dict[str, Any] | None,
    page_root: Path,
) -> str:
    color = COLORS.get(context_length, "#315c87")
    label = f"ctx={context_length:02d}"
    if item is None:
        return f'<div class="video-card missing" style="--accent:{color}"><b>{label}</b><span>未生成</span></div>'
    video_path = item["video"]
    # Each context result directory is linked as media/ctxXX.
    url = f"media/ctx{context_length:02d}/{video_path.name}"
    payload = item["payload"]
    effective = payload.get("inference", {}).get("effective_context_frames", context_length)
    return (
        f'<div class="video-card" style="--accent:{color}">'
        f'<div class="card-head"><b>{html.escape(label)}</b><span>{html.escape(str(effective))} raw frames</span></div>'
        f'<video controls preload="metadata" src="{html.escape(url, quote=True)}"></video>'
        f'<a href="{html.escape(url, quote=True)}" target="_blank">打开视频</a></div>'
    )


def build_case_page(
    *,
    dataset: str,
    records: list[dict[str, Any]],
    inputs: list[Path],
    cases_by_length: dict[int, dict[str, dict[str, Any]]],
    source_media: dict[str, str],
) -> str:
    lengths = [int(record["context_frames"]) for record in records]
    total = len(inputs)
    rows = []
    for input_path in inputs:
        stem = input_path.stem
        any_item = next(
            (cases[stem] for cases in cases_by_length.values() if stem in cases), None
        )
        caption = ""
        if any_item:
            caption = str(any_item["payload"].get("input_caption", ""))
        source = source_media.get(stem, "")
        source_html = (
            f'<video controls preload="metadata" src="{html.escape(source, quote=True)}"></video>'
            if source
            else "<span>源 context 不可用</span>"
        )
        cards = "".join(
            video_card(length, cases_by_length.get(length, {}).get(stem), Path("."))
            for length in lengths
        )
        rows.append(
            f'<article class="case" data-search="{html.escape((stem + " " + caption).lower(), quote=True)}">'
            f'<div class="case-title"><span>{html.escape(stem)}</span><button onclick="this.closest(\'.case\').classList.toggle(\'open\')">展开/收起</button></div>'
            f'<p>{html.escape(caption)}</p><div class="source"><div><b>输入 context · 8f</b>{source_html}</div></div>'
            f'<div class="cards">{cards}</div></article>'
        )
    model = "PHYRVG-PhysRVG finetuned DiT · LoRA OFF · reference"
    return page_shell(
        title=f"{model} · {dataset} context 长度",
        subtitle=f"{total} cases · {', '.join(f'ctx={length:02d}' for length in lengths)} · 仅改变输入 context 前缀长度",
        body=(
            '<div class="toolbar"><input id="filter" placeholder="筛选 case 名称或 prompt..." oninput="filterCases()">'
            '<button onclick="replayAll()">全部重新播放</button></div>'
            '<p class="note">视频不循环播放；按钮会将当前页面所有视频回到首帧并依次启动。</p>'
            + "".join(rows)
        ),
        active="cases",
    )


def build_metrics_page(
    *,
    dataset: str,
    records: list[dict[str, Any]],
    cases_by_length: dict[int, dict[str, dict[str, Any]]],
) -> str:
    lengths = [int(record["context_frames"]) for record in records]
    baseline = cases_by_length.get(8, {})
    means = {length: metric_means(cases_by_length.get(length, {})) for length in lengths}
    deltas = {length: paired_deltas(cases_by_length.get(length, {}), baseline) for length in lengths}
    header = "".join(
        f'<th style="border-top:4px solid {COLORS.get(length, "#315c87")}">ctx={length:02d}</th>'
        for length in lengths
    )
    mean_rows = []
    delta_rows = []
    for key, label, _, _ in METRICS:
        mean_rows.append(
            f"<tr><td>{html.escape(label)}</td>"
            + "".join(f"<td>{fmt(means[length].get(key))}</td>" for length in lengths)
            + "</tr>"
        )
        delta_rows.append(
            f"<tr><td>{html.escape(label)}</td>"
            + "".join(f"<td>{fmt(deltas[length].get(key))}</td>" for length in lengths)
            + "</tr>"
        )
    count_row = "<tr><td>有效 case 数</td>" + "".join(
        f"<td>{len(cases_by_length.get(length, {}))}</td>" for length in lengths
    ) + "</tr>"
    body = (
        '<p class="note">主比较为同一 case 相对于 ctx=8 的 paired delta；正负号均按原始指标方向展示，WMReward 仍保留原始 surprise 数值。</p>'
        '<h2>平均指标</h2><div class="table-wrap"><table><thead><tr><th>指标</th>'
        + header
        + "</tr></thead><tbody>"
        + count_row
        + "".join(mean_rows)
        + "</tbody></table></div>"
        '<h2>相对 ctx=8 的成对变化</h2><div class="table-wrap"><table><thead><tr><th>指标</th>'
        + header
        + "</tr></thead><tbody>"
        + "".join(delta_rows)
        + "</tbody></table></div>"
    )
    return page_shell(
        title=f"PHYRVG reference · {dataset} context 指标",
        subtitle=f"{len(cases_by_length.get(8, {}))} 个 ctx=8 baseline cases · 其他配置保持页面原生设置",
        body=body,
        active="metrics",
    )


def page_shell(*, title: str, subtitle: str, body: str, active: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--paper:#f4f0e5;--ink:#18231f;--muted:#68756f;--green:#173d34;--line:#d5ccba;--gold:#d29a36;--red:#b94f35}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:linear-gradient(145deg,#e5eee7,var(--paper) 45%,#e9e1d1);font-family:"Avenir Next","Noto Sans CJK SC",sans-serif}}
header{{padding:32px clamp(18px,4vw,60px);color:#f8f4e9;background:linear-gradient(120deg,#173d34,#102b25);border-bottom:4px solid var(--gold)}}
header a{{color:#f0d394;text-decoration:none;font:12px monospace}}h1{{margin:18px 0 8px;font:400 clamp(32px,5vw,62px)/1.03 Georgia,serif}}header p{{margin:0;color:#c9d7d1;line-height:1.6}}
nav{{display:flex;gap:14px;flex-wrap:wrap;padding:12px clamp(18px,4vw,60px);background:#fffdf7e8;border-bottom:1px solid var(--line)}}nav a{{color:var(--green);font:700 12px monospace}}
main{{max-width:1500px;margin:auto;padding:22px clamp(12px,3vw,38px) 60px}}.toolbar{{display:flex;gap:12px;align-items:center;margin-bottom:10px}}input{{flex:1;max-width:600px;padding:11px 13px;border:1px solid var(--line);border-radius:9px;background:#fffdf7;font-size:14px}}button{{border:0;border-radius:8px;padding:10px 14px;background:var(--red);color:#fff;cursor:pointer}}.note{{color:var(--muted);font-size:13px;line-height:1.6}}
.case{{margin:18px 0;padding:18px;border:1px solid var(--line);border-radius:15px;background:#fffdf7e8;box-shadow:0 10px 30px #30382f16}}.case-title{{display:flex;justify-content:space-between;gap:12px;font:700 15px monospace}}.case p{{color:var(--muted);line-height:1.5;margin:10px 0 14px}}.source{{display:flex;max-width:420px;margin-bottom:15px}}.source>div{{width:100%;padding:10px;border-left:4px solid #777;background:#f1ecdf}}.source video{{display:block;width:100%;margin-top:8px;border-radius:7px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}}.video-card{{padding:9px;border-top:4px solid var(--accent);border-radius:8px;background:#f1ecdf;min-width:0}}.video-card.missing{{min-height:80px;color:var(--muted)}}.card-head{{display:flex;justify-content:space-between;gap:4px;font-size:12px;margin-bottom:7px}}.card-head span{{color:var(--muted);font-size:10px}}.video-card video{{display:block;width:100%;aspect-ratio:16/9;background:#111;border-radius:5px}}.video-card a{{display:block;margin-top:6px;color:var(--green);font-size:11px}}.case:not(.open) .cards{{display:grid}}.case:not(.open){{}}
h2{{margin-top:28px;color:var(--green);font:400 28px Georgia,serif}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:10px;background:#fffdf7}}table{{width:100%;border-collapse:collapse;min-width:760px;font-size:12px}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left;position:sticky;left:0;background:#fffdf7}}th{{background:var(--green);color:#fff9ec}}tr:nth-child(even) td{{background:#f6f1e7}}tr:nth-child(even) td:first-child{{background:#f6f1e7}}
@media(max-width:700px){{.cards{{grid-template-columns:repeat(2,minmax(0,1fr))}}.source{{max-width:none}}}}
</style></head><body>
<header><a href="../">返回 8844 总览</a><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></header>
<nav><a href="../physrvg-context-length-test5/">test5 case</a><a href="../physrvg-context-length-physiciq/">Physics-IQ case</a><a href="../physrvg-context-length-metrics/">平均指标</a><a href="../test5-average-metrics/">原 test5 指标</a><a href="../physiciq-average-metrics/">原 Physics-IQ 指标</a></nav>
<main>{body}</main>
<script>
function filterCases(){{const q=document.getElementById('filter').value.toLowerCase();document.querySelectorAll('.case').forEach(x=>x.style.display=x.dataset.search.includes(q)?'':'none')}}
function replayAll(){{const videos=[...document.querySelectorAll('video')];videos.forEach(v=>{{v.pause();v.currentTime=0}});let i=0;function next(){{if(i>=videos.length)return;const v=videos[i++];v.play().catch(()=>{{}});v.onended=next}}next()}}
</script></body></html>"""


def build_root_entry(hub_root: Path, *, sweep_root: Path) -> None:
    root_index = hub_root / "index.html"
    if not root_index.is_file():
        return
    marker_start = "<!-- physrvg-context-length-entry:start -->"
    marker_end = "<!-- physrvg-context-length-entry:end -->"
    entry = f"""{marker_start}
<section style="margin:24px 0;padding:20px;border:1px solid #d5ccba;border-radius:14px;background:#fffdf7e8">
<h2 style="margin:0 0 8px;color:#173d34">PHYRVG · Context length sweep</h2>
<p style="color:#68756f">固定 LoRA OFF reference，只改变 context video 前缀长度；独立 case 页面和平均指标页面。</p>
<p><a href="physrvg-context-length-test5/">test5 case</a>　<a href="physrvg-context-length-physiciq/">Physics-IQ case</a>　<a href="physrvg-context-length-metrics/">平均指标</a></p>
</section>
{marker_end}"""
    page = root_index.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(marker_start) + r".*?" + re.escape(marker_end), re.DOTALL
    )
    if pattern.search(page):
        page = pattern.sub(entry, page)
    else:
        page = page.replace("</body>", entry + "\n</body>")
    write_text(root_index, page)


def build_dataset(
    *,
    sweep_root: Path,
    hub_root: Path,
    dataset: str,
    input_list: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, dict[str, Any]]]]:
    records = read_records(sweep_root, dataset)
    inputs = read_inputs(input_list)
    cases_by_length = case_payloads(records, inputs)
    page_name = f"physrvg-context-length-{dataset}"
    page_root = hub_root / page_name
    source_media = make_media_links(page_root, records, cases_by_length)
    write_text(
        page_root / "index.html",
        build_case_page(
            dataset=dataset,
            records=records,
            inputs=inputs,
            cases_by_length=cases_by_length,
            source_media=source_media,
        ),
    )
    return {
        "dataset": dataset,
        "records": records,
        "inputs": len(inputs),
        "cases_by_length": {str(k): len(v) for k, v in cases_by_length.items()},
        "page": page_name,
    }, cases_by_length


def main() -> None:
    args = parse_args()
    sweep_root = args.sweep_root.expanduser().resolve()
    hub_root = args.hub_root.expanduser().resolve()
    manifest_path = sweep_root / "sweep_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    hub_root.mkdir(parents=True, exist_ok=True)
    test5_list = Path(
        "/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt"
    )
    physiciq_list = Path(
        "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt"
    )
    summaries = []
    all_cases: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for dataset, input_list in (("test5", test5_list), ("physiciq", physiciq_list)):
        if not input_list.is_file():
            continue
        summary, cases = build_dataset(
            sweep_root=sweep_root,
            hub_root=hub_root,
            dataset=dataset,
            input_list=input_list,
        )
        summaries.append(summary)
        all_cases[dataset] = cases

    metrics_body = ""
    for dataset, input_list in (("test5", test5_list), ("physiciq", physiciq_list)):
        if dataset not in all_cases:
            continue
        records = read_records(sweep_root, dataset)
        metrics_body += (
            f'<section><h2>{html.escape(dataset)}</h2>'
            + build_metrics_page(
                dataset=dataset,
                records=records,
                cases_by_length=all_cases[dataset],
            ).split("<main>", 1)[1].rsplit("</main>", 1)[0]
            + "</section>"
        )
    write_text(
        hub_root / "physrvg-context-length-metrics" / "index.html",
        page_shell(
            title="PHYRVG reference · Context length metrics",
            subtitle="test5 与 Physics-IQ 分开统计；ctx=8 为现有 reference baseline",
            body=metrics_body,
            active="metrics",
        ),
    )
    build_root_entry(hub_root, sweep_root=sweep_root)
    write_text(
        sweep_root / "dashboard_manifest.json",
        json.dumps(
            {
                "sweep_root": str(sweep_root),
                "hub_root": str(hub_root),
                "pages": [
                    "physrvg-context-length-test5/",
                    "physrvg-context-length-physiciq/",
                    "physrvg-context-length-metrics/",
                ],
                "datasets": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    print(hub_root / "physrvg-context-length-metrics" / "index.html")


if __name__ == "__main__":
    main()
