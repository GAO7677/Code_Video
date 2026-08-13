#!/usr/bin/env python3
"""Read-only comparison board for latent guidance and direct attention control."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont

from AAA_my_test.wan_context_point_guidance.attention_audit_visualization import (
    ensure_generated_frame_attention_overlays,
)


CASE = "0613pybullet_sample_001460_w002"
TARGET = "object_A"
SEED = 47326
ANCHORS = tuple(range(0, 49, 4))
ATTENTION_STEPS = tuple(range(5, 41, 5))
GROUPS = ("top100", "bottom100", "random100")
GROUP_LABELS = {
    "top100": "Top100 · high-PCK",
    "bottom100": "Bottom100 · low-PCK",
    "random100": "Random100 · layer-matched",
}
DIRECTIONS = ("context_to_future", "future_to_context", "bidirectional")
DIRECTION_LABELS = {
    "context_to_future": "Context Query → Future Key",
    "future_to_context": "Future Query → Context Key",
    "bidirectional": "Bidirectional",
}
BASE = Path(
    "/data/gaoya/agent-data/outputs/wan_context_point_guidance_head_compare"
)
LATENT_ROOT = BASE / "attention_audit_v3" / "firstframe_ti2v"
DIRECT_ROOT = BASE / "direct_attention_tv_v1" / "firstframe_ti2v"
CACHE_ROOT = Path(
    "/data/gaoya/agent-data/cache/gt_stc_guidance_method_comparison"
)
CONTACT_COLUMNS = 7
CONTACT_TILE_WIDTH = 224
CONTACT_LABEL_HEIGHT = 24


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _source_video() -> Path | None:
    """Resolve the fixed source video from either experiment's baseline manifest."""
    for method in ("direct", "latent"):
        directory = _directory(method, "baseline", "context_to_future")
        if directory is None:
            continue
        value = _json(directory / "manifest.json").get("source_video")
        if isinstance(value, str):
            candidate = Path(value)
            if _ready(candidate):
                return candidate
    return None


def _video_frame_count(video: Path | None) -> int | None:
    if video is None or not _ready(video):
        return None
    capture = cv2.VideoCapture(str(video))
    try:
        value = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return value if value > 0 else None


def _directory(method: str, group: str, direction: str) -> Path | None:
    if method not in {"latent", "direct"}:
        return None
    if group == "baseline":
        variant = "baseline"
    elif group not in GROUPS:
        return None
    elif method == "latent":
        if direction != "context_to_future":
            return None
        variant = f"{group}__{TARGET}"
    else:
        if direction not in DIRECTIONS:
            return None
        variant = f"{group}__{direction}__{TARGET}"
    root = LATENT_ROOT if method == "latent" else DIRECT_ROOT
    return root / "generations" / CASE / f"seed_{SEED:05d}" / variant


def _target_metric(directory: Path | None) -> dict[str, Any] | None:
    if directory is None:
        return None
    for row in _json(directory / "trajectory_metrics.json").get("metrics", []):
        if str(row.get("target")) == TARGET:
            return row
    return None


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
    ]
    return sum(values) / len(values) if values else None


def _delta(value: Any, baseline: Any) -> float | None:
    if not isinstance(value, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    return float(value) - float(baseline)


def _variant(method: str, group: str, direction: str) -> dict[str, Any]:
    directory = _directory(method, group, direction)
    if directory is None:
        raise ValueError(f"invalid comparison variant: {method}/{group}/{direction}")
    video = directory / "generated.mp4"
    complete = _ready(directory / "complete.json") and _ready(video)
    metric = _target_metric(directory)
    baseline = _target_metric(_directory(method, "baseline", "context_to_future"))
    manifest = _json(directory / "manifest.json")
    audit = manifest.get("audit", []) if isinstance(manifest.get("audit"), list) else []
    attention = []
    if group != "baseline":
        for step in ATTENTION_STEPS:
            step_dir = directory / "attention_audit" / f"step_{step:02d}"
            attention.append(
                {
                    "step": step,
                    "ready": complete
                    and _ready(step_dir / "complete.json")
                    and _ready(step_dir / "raw_attention_maps.npz")
                    and _ready(step_dir / "metrics.json"),
                }
            )
    intervention = {
        "latent_update_rms": _mean(audit, "actual_mutable_update_rms"),
        "attention_tv": _mean(audit, "mean_actual_tv"),
        "correspondence_change": (
            _mean(audit, "loss_change")
            if method == "latent"
            else _mean(audit, "mean_target_ce_change")
        ),
        "av_delta_rms": _mean(audit, "mean_av_delta_rms"),
    }
    return {
        "method": method,
        "group": group,
        "direction": direction,
        "name": directory.name,
        "label": "Baseline" if group == "baseline" else GROUP_LABELS[group],
        "complete": complete,
        "frame_ready": complete,
        "metric_ready": metric is not None,
        "metric": metric,
        "delta": {
            key: _delta(
                None if metric is None else metric.get(key),
                None if baseline is None else baseline.get(key),
            )
            for key in (
                "ade_d0",
                "fde_d0",
                "pck_10pct_d0",
                "future_track_loss_score_0_100",
            )
        },
        "intervention": intervention,
        "attention": attention,
    }


def catalog() -> dict[str, Any]:
    source_video = _source_video()
    baselines = [
        _variant("latent", "baseline", "context_to_future"),
        _variant("direct", "baseline", "context_to_future"),
    ]
    lanes = [
        {
            "method": "latent",
            "method_label": "旧实验 · latent guidance",
            "direction": "context_to_future",
            "direction_label": DIRECTION_LABELS["context_to_future"],
            "variants": [
                _variant("latent", group, "context_to_future") for group in GROUPS
            ],
        }
    ]
    lanes.extend(
        {
            "method": "direct",
            "method_label": "新实验 · direct attention",
            "direction": direction,
            "direction_label": DIRECTION_LABELS[direction],
            "variants": [_variant("direct", group, direction) for group in GROUPS],
        }
        for direction in DIRECTIONS
    )
    all_variants = baselines + [v for lane in lanes for v in lane["variants"]]
    return {
        "case": CASE,
        "target": TARGET,
        "seed": SEED,
        "anchors": list(ANCHORS),
        "source": {
            "ready": source_video is not None,
            "frame_count": _video_frame_count(source_video),
            "label": "Source video · GT / pseudo-GT reference",
        },
        "attention_steps": list(ATTENTION_STEPS),
        "baselines": baselines,
        "lanes": lanes,
        "summary": {
            "planned": len(all_variants),
            "complete": sum(int(row["complete"]) for row in all_variants),
            "metrics": sum(int(row["metric_ready"]) for row in all_variants),
            "attention_ready": sum(
                int(audit["ready"])
                for row in all_variants
                for audit in row["attention"]
            ),
            "attention_total": sum(
                len(row["attention"]) for row in all_variants
            ),
        },
        "mechanisms": [
            {
                "method": "latent",
                "label": "Latent guidance",
                "operator": "x′ₛ = xₛ − η · normalized(∂Lcorr/∂xₛ)",
                "budget": "future latent update RMS = 0.01 / step",
                "meaning": "检验 selected heads 给出的梯度方向能否引导生成。",
            },
            {
                "method": "direct",
                "label": "Direct attention",
                "operator": "A′ = A + λ(T − A); O′ = A′V",
                "budget": "attention total variation = 0.10 / selected row",
                "meaning": "检验强制建立对应关系本身是否因果改变生成轨迹。",
            },
        ],
        "definitions": [
            {
                "metric": "GT Center-ADE / D0",
                "calculation": "所有未来共同可见 CoTracker 点相对 source GT 的逐帧平均距离，再除以首帧对象尺度 D0。",
                "direction": "越小越接近 GT；ΔADE < 0 表示相对同方法 Baseline 改善。",
            },
            {
                "metric": "GT Center-FDE / D0",
                "calculation": "最后一个 latent anchor 的共同可见点平均距离除以 D0。",
                "direction": "越小越好；只描述最终落点。",
            },
            {
                "metric": "PCK@10%D0",
                "calculation": "误差不超过 0.1D0 的共同可见点比例。",
                "direction": "越大越好。",
            },
            {
                "metric": "Track Loss",
                "calculation": "source 可追踪 future anchors 中，候选视频失去共同有效点的比例 ×100。",
                "direction": "越小越好；轨迹门控失败时优先读这个指标。",
            },
        ],
        "caveat": (
            "两种预算处于不同坐标系：latent RMS 0.01 与 attention TV 0.10 "
            "不能直接比较数值大小；页面比较的是最终轨迹结果与 head-group 排序。"
        ),
    }


def _ensure_anchor_frame(
    video: Path, method: str, group: str, direction: str, latent: int
) -> Path:
    if latent not in range(13):
        raise ValueError("latent anchor must lie in 0..12")
    output = CACHE_ROOT / "frames" / method / group / direction / f"R{latent:02d}.jpg"
    if _ready(output) and output.stat().st_mtime >= video.stat().st_mtime:
        return output
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, ANCHORS[latent])
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"could not decode F{ANCHORS[latent]:02d} from {video}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.jpg")
    Image.fromarray(rgb).save(
        temporary, format="JPEG", quality=90, optimize=True, subsampling=0
    )
    temporary.replace(output)
    return output


def _contact_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _ensure_contact_sheet(
    video: Path,
    output: Path,
    *,
    max_frames: int | None = None,
    mark_latent_anchors: bool = False,
) -> Path:
    """Decode frames and cache one numbered, seven-column temporal sheet."""
    if _ready(output) and output.stat().st_mtime >= video.stat().st_mtime:
        return output

    capture = cv2.VideoCapture(str(video))
    frames: list[Image.Image] = []
    try:
        while max_frames is None or len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"could not decode frames from {video}")

    source_width, source_height = frames[0].size
    tile_height = max(1, round(CONTACT_TILE_WIDTH * source_height / source_width))
    cell_height = tile_height + CONTACT_LABEL_HEIGHT
    rows = math.ceil(len(frames) / CONTACT_COLUMNS)
    canvas = Image.new(
        "RGB",
        (CONTACT_COLUMNS * CONTACT_TILE_WIDTH, rows * cell_height),
        color="#142633",
    )
    draw = ImageDraw.Draw(canvas)
    font = _contact_font(13)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    anchor_to_latent = (
        {frame: latent for latent, frame in enumerate(ANCHORS)}
        if mark_latent_anchors
        else {}
    )
    for frame_index, frame in enumerate(frames):
        column = frame_index % CONTACT_COLUMNS
        row = frame_index // CONTACT_COLUMNS
        x = column * CONTACT_TILE_WIDTH
        y = row * cell_height
        thumbnail = frame.resize((CONTACT_TILE_WIDTH, tile_height), resampling)
        canvas.paste(thumbnail, (x, y))
        latent = anchor_to_latent.get(frame_index)
        if latent is not None:
            draw.rectangle(
                (x + 1, y + 1, x + CONTACT_TILE_WIDTH - 2, y + tile_height - 2),
                outline="#f18a5b",
                width=4,
            )
        draw.rectangle(
            (x, y + tile_height, x + CONTACT_TILE_WIDTH, y + cell_height),
            fill="#142633" if latent is None else "#7f3828",
        )
        label = f"F{frame_index:02d}"
        if latent is not None:
            label += f"  ·  R{latent:02d}"
        draw.text(
            (x + 7, y + tile_height + 4),
            label,
            fill="#f4f8f6",
            font=font,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.jpg")
    canvas.save(temporary, format="JPEG", quality=88, optimize=True, subsampling=0)
    temporary.replace(output)
    return output


def asset(
    kind: str,
    method: str,
    group: str,
    direction: str,
    latent: str,
    step: str,
) -> Path | None:
    if kind == "source_contact_sheet":
        video = _source_video()
        if video is None:
            return None
        try:
            return _ensure_contact_sheet(
                video, CACHE_ROOT / "contact_sheets" / "source" / "sheet_v3.jpg"
            )
        except (OSError, RuntimeError, ValueError):
            return None
    try:
        latent_index = int(latent)
    except (TypeError, ValueError):
        return None
    directory = _directory(method, group, direction)
    if directory is None:
        return None
    video = directory / "generated.mp4"
    if not (_ready(directory / "complete.json") and _ready(video)):
        return None
    if kind == "contact_sheet":
        try:
            return _ensure_contact_sheet(
                video,
                CACHE_ROOT
                / "contact_sheets"
                / method
                / group
                / direction
                / "sheet_v3.jpg",
                max_frames=49,
                mark_latent_anchors=True,
            )
        except (OSError, RuntimeError, ValueError):
            return None
    if kind == "frame":
        try:
            return _ensure_anchor_frame(
                video, method, group, direction, latent_index
            )
        except (OSError, RuntimeError, ValueError):
            return None
    if kind != "attention" or group == "baseline":
        return None
    try:
        step_index = int(step)
    except (TypeError, ValueError):
        return None
    if step_index not in ATTENTION_STEPS or latent_index not in range(13):
        return None
    step_directory = directory / "attention_audit" / f"step_{step_index:02d}"
    if not (
        _ready(step_directory / "complete.json")
        and _ready(step_directory / "raw_attention_maps.npz")
        and _ready(step_directory / "metrics.json")
    ):
        return None
    try:
        output = ensure_generated_frame_attention_overlays(video, step_directory)
    except (OSError, RuntimeError, ValueError, KeyError):
        return None
    frame = output / f"R{latent_index:02d}.jpg"
    return frame if _ready(frame) else None


def page() -> str:
    return r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Latent Guidance × Direct Attention</title>
<style>
:root{--ink:#17232d;--paper:#edf1ef;--panel:#f8faf8;--rule:#bdc8c5;--latent:#255f91;--direct:#df6b3d;--good:#177d6c;--bad:#b54646;--quiet:#60726f;--shadow:0 16px 38px #162d3420}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:linear-gradient(90deg,#dce4e1 1px,transparent 1px),var(--paper);background-size:42px 100%;font:15px/1.5 "Avenir Next","Segoe UI",sans-serif}a{color:var(--latent)}button,select,input{font:inherit}header{padding:30px clamp(18px,5vw,76px) 26px;border-bottom:1px solid var(--rule);background:#f2f5f3ee}.eyebrow,.mono{font:700 11px/1.3 "SFMono-Regular",Consolas,monospace;letter-spacing:.11em;text-transform:uppercase}.eyebrow{color:var(--direct);margin-top:17px}h1{max-width:1080px;margin:7px 0 12px;font:800 clamp(38px,6vw,78px)/.93 "Arial Narrow","Avenir Next Condensed",sans-serif;letter-spacing:-.045em}.lead{max-width:940px;color:#40534f;font-size:17px}.thesis{display:grid;grid-template-columns:1fr 1fr;max-width:980px;margin-top:23px;border:1px solid var(--rule);background:var(--panel);box-shadow:var(--shadow)}.thesis div{padding:15px 17px;border-top:6px solid var(--latent)}.thesis div+div{border-left:1px solid var(--rule);border-top-color:var(--direct)}.thesis b{display:block;font:750 18px "Arial Narrow",sans-serif}.thesis code{font:12px/1.6 "SFMono-Regular",monospace;color:var(--quiet)}main{max-width:1880px;margin:auto;padding:24px clamp(14px,4vw,62px) 78px}.section{margin-top:18px;padding:19px;border:1px solid var(--rule);background:var(--panel);box-shadow:var(--shadow)}.section h2{margin:0;font:750 28px "Arial Narrow",sans-serif}.summary{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px}.stat{padding:14px;border-left:5px solid var(--latent);background:#e9f0f4}.stat:nth-child(2){border-color:var(--direct)}.stat b{display:block;font:800 28px "Arial Narrow",sans-serif}.toolbar{display:flex;gap:13px;align-items:end;flex-wrap:wrap;margin:15px 0}.toolbar label{font-weight:700}.toolbar select,.toolbar button{display:block;margin-top:5px;padding:8px 11px;border:1px solid #8fa19c;background:#fff;color:var(--ink)}.toolbar button{cursor:pointer;background:var(--ink);color:#fff}.toolbar input[type=range]{display:block;width:min(560px,76vw);margin-top:9px;accent-color:var(--direct)}.anchor-ruler{display:grid;grid-template-columns:repeat(13,1fr);gap:2px;margin:11px 0 18px}.anchor-ruler i{height:28px;display:grid;place-items:center;border:1px solid #aebdb8;background:#dbe4e1;color:#5a6d68;font:700 9px monospace;font-style:normal}.anchor-ruler i.active{background:var(--ink);color:#fff;border-color:var(--ink);box-shadow:inset 0 -5px var(--direct)}.mechanisms{display:grid;grid-template-columns:1fr 1fr;gap:12px}.mechanism{padding:16px;border-top:6px solid var(--latent);background:#edf3f6}.mechanism.direct{border-color:var(--direct);background:#fbf0eb}.mechanism h3{margin:0;font:750 23px "Arial Narrow",sans-serif}.equation{margin:10px 0;padding:10px;background:#fff;border:1px solid var(--rule);font:13px "SFMono-Regular",monospace}.warning{margin-top:12px;padding:12px 15px;border-left:6px solid #d3a12c;background:#fff7de;color:#645225}.baseline-grid{display:grid;grid-template-columns:1fr 1fr;gap:11px}.lane{margin-top:14px;border:1px solid var(--rule)}.lane-head{display:grid;grid-template-columns:minmax(230px,1fr) 2fr;gap:13px;padding:12px 14px;background:#e8eff3;border-left:7px solid var(--latent)}.lane.direct .lane-head{background:#f8ebe5;border-left-color:var(--direct)}.lane-head h3{margin:0;font:750 21px "Arial Narrow",sans-serif}.lane-head p{margin:3px 0;color:var(--quiet)}.variant-grid{display:grid;grid-template-columns:repeat(3,minmax(270px,1fr));gap:10px;padding:10px}.card{border:1px solid var(--rule);background:#fff;min-width:0}.frame,.pending{width:100%;aspect-ratio:16/9;display:block;background:#152634}.frame{object-fit:cover}.pending{display:grid;place-items:center;padding:20px;color:#f4c57d;text-align:center;font:700 12px "SFMono-Regular",monospace;letter-spacing:.08em}.caption{padding:11px}.caption>b{display:block;font-size:16px}.tag{display:block;color:var(--quiet);font:700 10px monospace;text-transform:uppercase}.metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:5px;margin-top:9px;font:11px/1.4 monospace;color:#435651}.good{color:var(--good)}.bad{color:var(--bad)}.attention-shell{padding:17px;background:#142633;color:#e9f2f0;border:1px solid #214254}.attention-shell h2{color:#fff}.attention-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.attention-card{background:#f7faf8;color:var(--ink);border-top:7px solid var(--latent)}.attention-card.direct{border-top-color:var(--direct)}.attention-card img{width:100%;display:block}.attention-note{color:#adc0bc}.definitions{overflow:auto}table{width:100%;min-width:760px;border-collapse:collapse}th,td{padding:10px;text-align:left;vertical-align:top;border-bottom:1px solid #d4ddda}th{color:var(--quiet);font:700 10px monospace;text-transform:uppercase}.footer{margin-top:25px;color:var(--quiet)}button:focus-visible,select:focus-visible,input:focus-visible,a:focus-visible{outline:3px solid #f1a177;outline-offset:2px}@media(max-width:900px){.thesis,.mechanisms,.baseline-grid,.attention-grid{grid-template-columns:1fr}.thesis div+div{border-left:0;border-top:6px solid var(--direct)}.variant-grid{grid-template-columns:repeat(3,80vw);overflow:auto}.lane-head{grid-template-columns:1fr}.summary{grid-template-columns:1fr 1fr}}@media(max-width:560px){h1{font-size:42px}.summary{grid-template-columns:1fr}.section{padding:13px}}@media(prefers-reduced-motion:no-preference){.section{animation:arrive .3s ease both}@keyframes arrive{from{opacity:0;transform:translateY(7px)}}}
</style><style>
.contact-legend{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:11px 0 16px;color:var(--quiet)}.contact-legend b{color:var(--ink)}.anchor-key{display:inline-block;width:18px;height:12px;margin-right:6px;border:3px solid var(--direct);vertical-align:-2px}.source-board{border-top:7px solid #213f4a;background:#fff}.sheet-link{display:block;background:#142633;overflow:auto}.sheet{width:100%;height:auto;display:block}.sheet-note{padding:9px 12px;background:#e9efed;color:var(--quiet);font:700 10px/1.5 "SFMono-Regular",monospace;text-transform:uppercase}.variant-grid .sheet{min-width:0}.variant-grid .card{align-self:start}.attention-shell .toolbar input[type=range]{width:min(440px,72vw)}
</style></head><body>
<header><a href="/">← 返回 8092 总入口</a> · <a href="/gt-stc-guidance-results?v=6">原 GT-STC 页面</a><div class="eyebrow">One trajectory · two intervention coordinate systems</div><h1>同一条轨迹，<br>两种施力位置。</h1><p class="lead">固定同一个 case、object、seed 和 latest3350 head groups，比较“用 attention loss 推动 latent”与“直接移动 attention probability”。原视频保留全部原始帧，每个生成结果把完整 49 帧拼成 7×7 静态时序板，便于直接检查轨迹、消失和外观变化。</p><div class="thesis"><div><b>旧实验 · 推动 latent</b><code>x′ₛ = xₛ − η · normalized(∂Lcorr/∂xₛ)</code></div><div><b>新实验 · 改写 attention</b><code>A′ = A + λ(T − A) · O′ = A′V</code></div></div></header>
<main><div id="summary" class="summary"></div><section class="section"><h2>先对齐实验口径</h2><div id="mechanisms" class="mechanisms"></div><div id="warning" class="warning"></div></section><section class="section"><h2>原视频 · 完整原始帧参考</h2><div class="toolbar"><button id="refresh">刷新落盘状态</button><span id="updated" class="mono">读取中</span></div><div class="contact-legend"><span><b>阅读顺序：</b>从左到右、从上到下，保留 source 的全部原始帧</span><span>原始帧编号不冒充生成 latent anchor</span><span>点击拼图可打开原尺寸</span></div><div id="source"></div></section><section class="section"><h2>两种实验 · 完整 49 帧结果</h2><div class="contact-legend"><span>每张图使用相同的 7×7 帧序 F00–F48</span><span><i class="anchor-key"></i>橙框 = latent anchor（R00–R12）</span><span>先纵向看时序，再横向比较 head group 与施力方式</span></div><h3>各自 Baseline</h3><div id="baselines" class="baseline-grid"></div><div id="lanes"></div></section><section class="section attention-shell"><h2>PRE / POST attention microscope</h2><p class="attention-note">上面的生成结果展示全部 49 帧；这里再选择一个 latent anchor 和 denoising step，读取各自实验的 PRE/POST attention overlay。旧实验固定 Context→Future；新实验方向可切换。</p><div class="toolbar"><label>Latent anchor<input id="anchor" type="range" min="0" max="12" value="12"><span id="anchorText" class="mono"></span></label><label>Head group<select id="group"><option value="top100">Top100</option><option value="bottom100">Bottom100</option><option value="random100">Random100</option></select></label><label>新实验方向<select id="direction"><option value="context_to_future">Context → Future</option><option value="future_to_context">Future → Context</option><option value="bidirectional">Bidirectional</option></select></label><label>Denoising step<select id="step"></select></label></div><div id="ruler" class="anchor-ruler"></div><div id="attention" class="attention-grid"></div></section><section class="section"><h2>指标定义</h2><div class="definitions"><table><thead><tr><th>指标</th><th>计算</th><th>判读</th></tr></thead><tbody id="defs"></tbody></table></div></section><p class="footer">页面每 30 秒重新读取 JSON；尚未完成的视频、CoTracker 指标、拼图或 attention overlay 保留明确 Pending。全页只提供静态帧图像，不嵌入视频播放器。</p></main>
<script>
const api='/api/gt-stc-guidance-method-comparison',E=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),F=(v,n=3)=>v==null?'N/A':Number(v).toFixed(n);let D;const $=id=>document.getElementById(id);
function asset(kind,row,extra=''){const q=new URLSearchParams({kind,method:row.method,group:row.group,direction:row.direction,latent:$('anchor').value});if(extra)for(const [k,v] of Object.entries(extra))q.set(k,v);return `${api}/asset?${q}`}
function sourceAsset(){return `${api}/asset?kind=source_contact_sheet`}
function metric(row){if(!row.metric_ready)return `<div class="metrics"><span class="bad">指标 Pending</span><span>${row.complete?'等待 CoTracker':'等待生成'}</span></div>`;const m=row.metric,d=row.delta||{},good=d.ade_d0!=null&&d.ade_d0<0;return `<div class="metrics"><span>Gate ${m.quality_pass?'PASS':'FAIL'}</span><span>ADE/D0 ${F(m.ade_d0)}</span><span class="${good?'good':'bad'}">ΔADE ${F(d.ade_d0)}</span><span>FDE/D0 ${F(m.fde_d0)}</span><span>ΔFDE ${F(d.fde_d0)}</span><span>PCK10 ${F(m.pck_10pct_d0)}</span><span>TrackLoss ${F(m.future_track_loss_score_0_100,1)}</span></div>`}
function card(row,methodLabel=''){const url=asset('contact_sheet',row),image=row.frame_ready?`<a class="sheet-link" href="${url}" target="_blank" rel="noopener"><img class="sheet" loading="lazy" src="${url}" alt="${E(row.label)} complete 49-frame contact sheet"></a>`:`<div class="pending">PENDING<br>${row.complete?'CONTACT SHEET':'GENERATION'}</div>`;return `<article class="card">${image}<div class="sheet-note">49 RGB frames · 7×7 · F00–F48</div><div class="caption"><b>${E(row.label)}</b><span class="tag">${E(methodLabel)} · complete temporal board</span>${metric(row)}</div></article>`}
function sourceCard(){if(!D.source.ready)return `<div class="pending">PENDING · SOURCE VIDEO</div>`;const url=sourceAsset();return `<article class="card source-board"><a class="sheet-link" href="${url}" target="_blank" rel="noopener"><img class="sheet" src="${url}" alt="Source video complete contact sheet"></a><div class="sheet-note">${E(D.source.label)} · all ${D.source.frame_count??'?'} raw frames · 7 columns</div></article>`}
function render(){if(!D)return;const a=Number($('anchor').value);$('anchorText').textContent=` R${String(a).padStart(2,'0')} · F${String(D.anchors[a]).padStart(2,'0')}`;$('ruler').innerHTML=D.anchors.map((f,i)=>`<i class="${i===a?'active':''}">R${String(i).padStart(2,'0')}<br>F${String(f).padStart(2,'0')}</i>`).join('');$('source').innerHTML=sourceCard();$('baselines').innerHTML=D.baselines.map((v,i)=>card(v,i?'Direct baseline':'Latent baseline')).join('');$('lanes').innerHTML=D.lanes.map(l=>`<section class="lane ${l.method}"><div class="lane-head"><div><span class="mono">${E(l.method_label)}</span><h3>${E(l.direction_label)}</h3></div><p>${l.method==='latent'?'固定 latent RMS；attention 通过重新前向间接改变。':'固定 attention TV；直接替换 selected Query/head 的 A·V 输出。'}</p></div><div class="variant-grid">${l.variants.map(v=>card(v,l.method_label)).join('')}</div></section>`).join('');renderAttention()}
function find(method,group,direction){for(const l of D.lanes)if(l.method===method&&l.direction===direction){return l.variants.find(v=>v.group===group)}return null}
function attentionCard(row,label,klass=''){const step=Number($('step').value),ready=row&&row.attention.some(x=>x.step===step&&x.ready),image=ready?`<img loading="lazy" src="${asset('attention',row,{step})}" alt="${E(label)} PRE POST attention overlay">`:`<div class="pending">PENDING · STEP ${step}<br>${row&&row.complete?'ATTENTION OVERLAY':'GENERATION'}</div>`;const iv=row?.intervention||{};return `<article class="card attention-card ${klass}">${image}<div class="caption"><b>${E(label)}</b><span class="tag">R${String($('anchor').value).padStart(2,'0')} · step ${step}</span><div class="metrics"><span>latent RMS ${F(iv.latent_update_rms,4)}</span><span>attention TV ${F(iv.attention_tv,4)}</span><span>Δ correspondence ${F(iv.correspondence_change,4)}</span><span>Δ A·V RMS ${F(iv.av_delta_rms,4)}</span></div></div></article>`}
function renderAttention(){if(!D)return;const g=$('group').value,dir=$('direction').value,old=find('latent',g,'context_to_future'),direct=find('direct',g,dir);$('attention').innerHTML=attentionCard(old,`旧 · ${g} · Context→Future`)+attentionCard(direct,`新 · ${g} · ${D.lanes.find(x=>x.method==='direct'&&x.direction===dir).direction_label}`,'direct')}
async function load(){const keep={anchor:$('anchor').value,group:$('group').value,direction:$('direction').value,step:$('step').value};D=await fetch(`${api}/catalog?x=${Date.now()}`).then(r=>r.json());$('step').innerHTML=D.attention_steps.map(x=>`<option value="${x}">${x}</option>`).join('');for(const [k,v] of Object.entries(keep))if(v&&$(k))$(k).value=v;$('summary').innerHTML=`<div class="stat"><span class="mono">Static outputs</span><b>${D.summary.complete}/${D.summary.planned}</b><small>两个 Baseline + 12 个实验槽</small></div><div class="stat"><span class="mono">Trajectory metrics</span><b>${D.summary.metrics}/${D.summary.planned}</b><small>source-GT CoTracker</small></div><div class="stat"><span class="mono">Attention audits</span><b>${D.summary.attention_ready}/${D.summary.attention_total}</b><small>step 5…40</small></div><div class="stat"><span class="mono">Frozen unit</span><b>1×1×1</b><small>case × object × seed</small></div>`;$('mechanisms').innerHTML=D.mechanisms.map(x=>`<article class="mechanism ${x.method}"><span class="mono">${E(x.method)}</span><h3>${E(x.label)}</h3><div class="equation">${E(x.operator)}</div><b>${E(x.budget)}</b><p>${E(x.meaning)}</p></article>`).join('');$('warning').textContent=D.caveat;$('defs').innerHTML=D.definitions.map(x=>`<tr><td><b>${E(x.metric)}</b></td><td>${E(x.calculation)}</td><td>${E(x.direction)}</td></tr>`).join('');$('updated').textContent=`更新 ${new Date().toLocaleTimeString()}`;render()}
$('anchor').oninput=render;$('group').onchange=renderAttention;$('direction').onchange=renderAttention;$('step').onchange=renderAttention;$('refresh').onclick=load;load();setInterval(load,30000);
</script></body></html>'''
