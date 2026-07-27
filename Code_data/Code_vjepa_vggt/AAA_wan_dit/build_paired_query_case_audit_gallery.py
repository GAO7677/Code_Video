#!/usr/bin/env python3
"""Build a Chinese case-audit gallery for the paired-query experiment."""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle

from analyze_multiblock_ball_query_heads import ROLE_LABELS, _role_scores
from moving_query_attention import FEATURE_NAMES

_CJK_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if _CJK_FONT.is_file():
    font_manager.fontManager.addfont(_CJK_FONT)
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


MODELS = ("wan_lora", "xssc", "physrvg")
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
PROTOCOLS = ("moving", "anchor_t2")
PROTOCOL_LABELS = {
    "moving": "Moving query",
    "anchor_t2": "Anchor-t2 query",
}
ROLES = tuple(ROLE_LABELS)
ROLE_COLORS = ("#2368a2", "#c8464f", "#228b65", "#d58a22", "#68727d")
ROLE_ZH = {
    "S": "帧内空间",
    "T": "目标轨迹",
    "P": "固定位置时间对齐",
    "C": "首帧/历史上下文",
    "G": "全局聚合",
}
FEATURE_ZH = {
    "entropy": "注意力熵",
    "same_frame_mass": "同帧质量",
    "local_mass": "目标邻域质量",
    "first_frame_mass": "首帧质量",
    "history_bias": "历史偏置",
    "mean_time_distance": "平均时间距离",
    "aligned_enrichment": "固定位置富集",
    "cross_ball_enrichment": "跨帧目标轨迹富集",
}


@dataclass
class ModelCaseResult:
    model: str
    case: str
    generated_video: str
    query_overlay: str
    query_preview: str
    query_chart: str
    role_chart: str
    focus_chart: str
    target_phrase: str
    detector_score: float
    prompt_frame: int
    track_score: float
    valid_ratio: float
    unresolved_frames: int
    fallback_frames: int
    repaired_frames: int
    query_counts: list[int]
    anchor_valid: bool
    md5_verified: bool
    role_agreement: float
    valid_role_cells: int
    moving_focus_role: str
    anchor_focus_role: str
    moving_focus_margin: float
    anchor_focus_margin: float
    focus_feature_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/data/gaoya/agent-data/outputs/wan_dit_paired_query_50seeds"
        ),
    )
    parser.add_argument(
        "--input-list",
        type=Path,
        default=Path(
            "/data/gaoya/agent-data/outputs/wan_dit_paired_query_50seeds/"
            "input_lists/test5_unique20.txt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data/gaoya/agent-data/outputs/wan_dit_paired_query_50seeds/"
            "_case_audit_zh"
        ),
    )
    parser.add_argument("--display-seed", type=int, default=851)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--focus-block", type=int, default=0)
    parser.add_argument("--focus-head", type=int, default=10)
    return parser.parse_args()


def _deduplicated_paths(path: Path) -> list[Path]:
    paths: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        current = Path(line.strip()).expanduser().resolve()
        if current not in paths:
            paths.append(current)
    return paths


def _safe_name(text: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in text
    )


def _ensure_symlink(source: Path, destination: Path) -> str:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return destination.as_posix()
        destination.unlink()
    elif destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    os.symlink(source, destination)
    return destination.as_posix()


def _ensure_browser_video(source: Path, destination: Path) -> str:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.is_file()
        and not destination.is_symlink()
        and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns
    ):
        return destination.as_posix()
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    temporary = destination.with_name(destination.stem + ".tmp.webm")
    if temporary.exists():
        temporary.unlink()
    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"VP80"),
        fps,
        (width, height),
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError(f"cannot transcode browser video: {source}")
    frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        frame_count += 1
    capture.release()
    writer.release()
    if frame_count == 0 or not temporary.is_file():
        raise RuntimeError(f"browser transcode produced no frames: {source}")
    temporary.replace(destination)
    return destination.as_posix()


def _relative(path: Path, root: Path) -> str:
    return path.absolute().relative_to(root.absolute()).as_posix()


def _winner_detector_score(item: dict[str, Any]) -> float:
    tracks = item.get("candidate_tracks") or []
    winner = int(item.get("winner_candidate_index", 0))
    if 0 <= winner < len(tracks):
        return float(tracks[winner].get("detector_score", float("nan")))
    return float("nan")


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float64)
    return order.astype(np.float64) / float(len(values) - 1)


def _classify_protocol(
    arrays: np.lib.npyio.NpzFile,
    protocol: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = {
        feature: arrays[f"{protocol}__{feature}"].astype(np.float64).mean(axis=0)
        for feature in FEATURE_NAMES
    }
    scores = _role_scores(features)
    matrix = np.stack([scores[role] for role in ROLES], axis=1)
    order = np.argsort(matrix, axis=1)
    primary = order[:, -1]
    margin = (
        np.take_along_axis(matrix, order[:, -1:], axis=1)[:, 0]
        - np.take_along_axis(matrix, order[:, -2:-1], axis=1)[:, 0]
    )
    return primary, margin, matrix


def _load_case_role_data(
    capture_root: Path,
    model: str,
    case: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], bool]:
    labels = {
        protocol: np.full((30, 24), -1, dtype=np.int16)
        for protocol in PROTOCOLS
    }
    margins = {
        protocol: np.full((30, 24), np.nan, dtype=np.float64)
        for protocol in PROTOCOLS
    }
    anchor_valid = True
    for block in range(30):
        path = (
            capture_root
            / f"block{block:02d}"
            / "matrices"
            / model
            / case
            / f"block{block:02d}_paired_query_features.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as arrays:
            current_anchor_valid = bool(arrays["anchor_t2_valid"])
            anchor_valid = anchor_valid and current_anchor_valid
            primary, margin, _ = _classify_protocol(arrays, "moving")
            labels["moving"][block] = primary
            margins["moving"][block] = margin
            if current_anchor_valid:
                primary, margin, _ = _classify_protocol(arrays, "anchor_t2")
                labels["anchor_t2"][block] = primary
                margins["anchor_t2"][block] = margin
    return labels, margins, anchor_valid


def _plot_query_protocol(
    output: Path,
    counts: list[int],
    *,
    case: str,
    model: str,
) -> None:
    times = np.arange(13)
    anchor = np.zeros(13, dtype=np.int16)
    if len(counts) > 2:
        anchor[2] = counts[2]
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9.4, 4.2),
        dpi=145,
        sharex=True,
        constrained_layout=True,
    )
    axes[0].bar(times, counts, color="#2477b3", width=0.72)
    axes[0].set_ylabel("Q token 数")
    axes[0].set_title("Moving query：每个 latent 时间使用目标当前位置")
    axes[1].bar(times, anchor, color="#c8464f", width=0.72)
    axes[1].axvline(2, color="#333333", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Q token 数")
    axes[1].set_xlabel("latent 时间 t；对应生成视频帧约为 4t")
    axes[1].set_title("Anchor-t2 query：只使用 t=2（视频第8帧）的目标 token")
    axes[1].set_xticks(times)
    for axis in axes:
        axis.set_ylim(0, max(max(counts, default=1) + 1, 3))
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"{MODEL_LABELS[model]} · {case}", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_role_map(
    output: Path,
    labels: dict[str, np.ndarray],
    *,
    case: str,
    model: str,
    focus_block: int,
    focus_head: int,
) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.4, 6.2),
        dpi=145,
        constrained_layout=True,
    )
    cmap = ListedColormap(ROLE_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(ROLES) + 0.5), len(ROLES))
    for axis, protocol in zip(axes[:2], PROTOCOLS):
        values = labels[protocol].astype(np.float64)
        values[values < 0] = np.nan
        axis.imshow(
            values,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )
        axis.add_patch(
            Rectangle(
                (focus_head - 0.5, focus_block - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#101820",
                linewidth=1.8,
            )
        )
        axis.set_title(PROTOCOL_LABELS[protocol])
        axis.set_xlabel("Head")
        axis.set_ylabel("Block")
        axis.set_xticks(np.arange(0, 24, 2))
        axis.set_yticks(np.arange(0, 30, 2))
    valid = (labels["moving"] >= 0) & (labels["anchor_t2"] >= 0)
    agreement = np.full((30, 24), np.nan, dtype=np.float64)
    agreement[valid] = (
        labels["moving"][valid] == labels["anchor_t2"][valid]
    ).astype(np.float64)
    axes[2].imshow(
        agreement,
        cmap=ListedColormap(("#d85b57", "#2f8f68")),
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    axes[2].add_patch(
        Rectangle(
            (focus_head - 0.5, focus_block - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#101820",
            linewidth=1.8,
        )
    )
    axes[2].set_title("两种 query 的角色是否一致")
    axes[2].set_xlabel("Head")
    axes[2].set_ylabel("Block")
    axes[2].set_xticks(np.arange(0, 24, 2))
    axes[2].set_yticks(np.arange(0, 30, 2))
    role_legend = [
        Patch(facecolor=color, label=f"{role}: {ROLE_ZH[role]}")
        for role, color in zip(ROLES, ROLE_COLORS)
    ]
    agreement_legend = [
        Patch(facecolor="#2f8f68", label="角色一致"),
        Patch(facecolor="#d85b57", label="角色变化"),
    ]
    figure.legend(
        handles=role_legend + agreement_legend,
        loc="upper center",
        ncol=7,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.03),
    )
    figure.suptitle(
        f"{MODEL_LABELS[model]} · {case} · 四个去噪步聚合角色",
        fontsize=13,
        y=1.08,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _focus_payload(
    path: Path,
    focus_head: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, str],
    dict[str, float],
    list[int],
    bool,
]:
    feature_percentiles: dict[str, np.ndarray] = {}
    role_scores: dict[str, np.ndarray] = {}
    aggregate_roles: dict[str, str] = {}
    aggregate_margins: dict[str, float] = {}
    with np.load(path) as arrays:
        steps = [int(value) for value in arrays["steps_one_based"]]
        anchor_valid = bool(arrays["anchor_t2_valid"])
        for protocol in PROTOCOLS:
            if protocol == "anchor_t2" and not anchor_valid:
                feature_percentiles[protocol] = np.full(
                    (len(FEATURE_NAMES), len(steps)), np.nan
                )
                role_scores[protocol] = np.full((len(ROLES), len(steps)), np.nan)
                aggregate_roles[protocol] = "-"
                aggregate_margins[protocol] = float("nan")
                continue
            per_feature = []
            per_step_roles = []
            for feature in FEATURE_NAMES:
                values = arrays[f"{protocol}__{feature}"].astype(np.float64)
                per_feature.append(
                    np.asarray(
                        [_rank01(row)[focus_head] for row in values],
                        dtype=np.float64,
                    )
                )
            for step_index in range(len(steps)):
                step_features = {
                    feature: arrays[f"{protocol}__{feature}"][step_index].astype(
                        np.float64
                    )
                    for feature in FEATURE_NAMES
                }
                scores = _role_scores(step_features)
                per_step_roles.append(
                    [float(scores[role][focus_head]) for role in ROLES]
                )
            feature_percentiles[protocol] = np.stack(per_feature, axis=0)
            role_scores[protocol] = np.asarray(per_step_roles).T
            primary, margin, _ = _classify_protocol(arrays, protocol)
            aggregate_roles[protocol] = ROLES[int(primary[focus_head])]
            aggregate_margins[protocol] = float(margin[focus_head])
    return (
        feature_percentiles,
        role_scores,
        aggregate_roles,
        aggregate_margins,
        steps,
        anchor_valid,
    )


def _plot_focus_evidence(
    output: Path,
    feature_percentiles: dict[str, np.ndarray],
    role_scores: dict[str, np.ndarray],
    steps: list[int],
    *,
    case: str,
    model: str,
    focus_block: int,
    focus_head: int,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.4, 10.2),
        dpi=145,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.35, 1.0)},
    )
    feature_labels = [FEATURE_ZH[name] for name in FEATURE_NAMES]
    for column, protocol in enumerate(PROTOCOLS):
        image = axes[0, column].imshow(
            feature_percentiles[protocol],
            cmap="viridis",
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
        )
        axes[0, column].set_title(
            f"{PROTOCOL_LABELS[protocol]}：8项证据的 head 百分位"
        )
        axes[0, column].set_yticks(np.arange(len(feature_labels)), feature_labels)
        axes[0, column].set_xticks(np.arange(len(steps)), steps)
        axes[0, column].set_xlabel("去噪步")
        figure.colorbar(image, ax=axes[0, column], fraction=0.035, pad=0.02)

        image = axes[1, column].imshow(
            role_scores[protocol],
            cmap="magma",
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
        )
        axes[1, column].set_title(
            f"{PROTOCOL_LABELS[protocol]}：由证据公式得到的角色分数"
        )
        axes[1, column].set_yticks(
            np.arange(len(ROLES)),
            [f"{role} · {ROLE_ZH[role]}" for role in ROLES],
        )
        axes[1, column].set_xticks(np.arange(len(steps)), steps)
        axes[1, column].set_xlabel("去噪步")
        figure.colorbar(image, ax=axes[1, column], fraction=0.035, pad=0.02)
    figure.suptitle(
        f"{MODEL_LABELS[model]} · Block {focus_block:02d} / Head {focus_head:02d}"
        f" · {case}",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _focus_feature_rows(
    feature_percentiles: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for index, feature in enumerate(FEATURE_NAMES):
        rows.append(
            {
                "feature": feature,
                "label": FEATURE_ZH[feature],
                "moving": float(np.nanmean(feature_percentiles["moving"][index])),
                "anchor": float(
                    np.nanmean(feature_percentiles["anchor_t2"][index])
                ),
            }
        )
    return rows


def _model_case_result(
    *,
    root: Path,
    output: Path,
    seed: int,
    model: str,
    case: str,
    focus_block: int,
    focus_head: int,
) -> ModelCaseResult:
    seed_name = f"seed-{seed:06d}"
    query_map_path = root / "query_maps" / model / seed_name / "query_map.json"
    query_payload = json.loads(query_map_path.read_text(encoding="utf-8"))
    item = query_payload["cases"][case]
    state_path = root / "state" / model / f"{seed_name}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "complete":
        raise RuntimeError(f"incomplete paired-query state: {state_path}")

    asset_root = output / "assets" / _safe_name(case) / model
    generated_destination = asset_root / "generated.mp4"
    legacy_overlay = asset_root / "query_overlay.mp4"
    if legacy_overlay.is_symlink() or legacy_overlay.exists():
        legacy_overlay.unlink()
    overlay_destination = asset_root / "query_overlay.webm"
    preview_destination = asset_root / "query_preview.jpg"
    _ensure_symlink(Path(item["generated_video"]), generated_destination)
    _ensure_browser_video(Path(item["overlay_video"]), overlay_destination)
    _ensure_symlink(Path(item["preview"]), preview_destination)

    charts = output / "charts" / _safe_name(case) / model
    query_chart = charts / "query_protocol.png"
    role_chart = charts / "role_map.png"
    focus_chart = charts / "focus_evidence.png"
    counts = [int(value) for value in item["query_tokens_per_time"]]
    _plot_query_protocol(query_chart, counts, case=case, model=model)

    capture_root = root / "capture" / model / seed_name
    labels, margins, anchor_valid = _load_case_role_data(
        capture_root,
        model,
        case,
    )
    _plot_role_map(
        role_chart,
        labels,
        case=case,
        model=model,
        focus_block=focus_block,
        focus_head=focus_head,
    )
    focus_npz = (
        capture_root
        / f"block{focus_block:02d}"
        / "matrices"
        / model
        / case
        / f"block{focus_block:02d}_paired_query_features.npz"
    )
    (
        feature_percentiles,
        role_scores,
        aggregate_roles,
        aggregate_margins,
        steps,
        focus_anchor_valid,
    ) = _focus_payload(focus_npz, focus_head)
    if focus_anchor_valid != anchor_valid:
        raise RuntimeError(f"inconsistent anchor validity: {focus_npz}")
    _plot_focus_evidence(
        focus_chart,
        feature_percentiles,
        role_scores,
        steps,
        case=case,
        model=model,
        focus_block=focus_block,
        focus_head=focus_head,
    )
    valid = (labels["moving"] >= 0) & (labels["anchor_t2"] >= 0)
    valid_cells = int(valid.sum())
    agreement = (
        float(
            np.mean(labels["moving"][valid] == labels["anchor_t2"][valid])
        )
        if valid_cells
        else float("nan")
    )
    quality = item.get("track_quality") or {}
    return ModelCaseResult(
        model=model,
        case=case,
        generated_video=_relative(generated_destination, output),
        query_overlay=_relative(overlay_destination, output),
        query_preview=_relative(preview_destination, output),
        query_chart=_relative(query_chart, output),
        role_chart=_relative(role_chart, output),
        focus_chart=_relative(focus_chart, output),
        target_phrase=str(item.get("target_phrase") or "-"),
        detector_score=_winner_detector_score(item),
        prompt_frame=int(item.get("prompt_frame_idx", -1)),
        track_score=float(quality.get("score", float("nan"))),
        valid_ratio=float(quality.get("valid_ratio", float("nan"))),
        unresolved_frames=len(item.get("unresolved_frames") or []),
        fallback_frames=int(item.get("fallback_frame_count", 0)),
        repaired_frames=int(item.get("cotracker_repair_frame_count", 0)),
        query_counts=counts,
        anchor_valid=anchor_valid,
        md5_verified=bool(state.get("md5_verified")),
        role_agreement=agreement,
        valid_role_cells=valid_cells,
        moving_focus_role=aggregate_roles["moving"],
        anchor_focus_role=aggregate_roles["anchor_t2"],
        moving_focus_margin=aggregate_margins["moving"],
        anchor_focus_margin=aggregate_margins["anchor_t2"],
        focus_feature_rows=_focus_feature_rows(feature_percentiles),
    )


def _input_assets(
    payload: dict[str, Any],
    destination: Path,
    output: Path,
) -> dict[str, str | None]:
    candidates = {
        "context": payload.get("input_video") or payload.get("input_video_randomf"),
        "source": payload.get("source_video"),
    }
    assets: dict[str, str | None] = {}
    for label, raw in candidates.items():
        if not raw:
            assets[label] = None
            continue
        source = Path(str(raw)).expanduser()
        if not source.is_file():
            assets[label] = None
            continue
        target = destination / f"{label}.webm"
        _ensure_browser_video(source, target)
        assets[label] = _relative(target, output)
    return assets


def _video(path: str | None, label: str) -> str:
    if not path:
        return (
            "<div class='missing-media'>"
            f"{html.escape(label)}文件不存在，页面未伪造替代内容。"
            "</div>"
        )
    return (
        f"<video controls muted loop preload='metadata' "
        f"src='{html.escape(path)}' aria-label='{html.escape(label)}'></video>"
    )


def _metric(value: float, digits: int = 3) -> str:
    return "-" if not np.isfinite(value) else f"{value:.{digits}f}"


def _feature_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        moving = float(row["moving"])
        anchor = float(row["anchor"])
        if not np.isfinite(anchor):
            best = "moving"
        elif moving > anchor:
            best = "moving"
        elif anchor > moving:
            best = "anchor"
        else:
            best = ""
        body.append(
            "<tr>"
            f"<th>{html.escape(str(row['label']))}</th>"
            f"<td class='{'best' if best == 'moving' else ''}'>{moving:.3f}</td>"
            f"<td class='{'best' if best == 'anchor' else ''}'>"
            f"{_metric(anchor)}</td>"
            "</tr>"
        )
    return (
        "<table class='feature-table'><thead><tr>"
        "<th>证据百分位（四步均值）</th><th>Moving</th><th>Anchor-t2</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _model_generation_card(result: ModelCaseResult) -> str:
    return f"""
<article class="model-item">
  <div class="model-heading">
    <h4>{MODEL_LABELS[result.model]}</h4>
    <span class="status {'ok' if result.md5_verified else 'warn'}">
      {'MD5重放一致' if result.md5_verified else '未通过MD5重放核验'}
    </span>
  </div>
  {_video(result.generated_video, MODEL_LABELS[result.model] + ' 生成视频')}
</article>
"""


def _model_localization_card(result: ModelCaseResult) -> str:
    return f"""
<article class="model-item">
  <div class="model-heading"><h4>{MODEL_LABELS[result.model]}</h4>
    <span class="status {'ok' if result.unresolved_frames == 0 else 'warn'}">
      未解析帧 {result.unresolved_frames}
    </span>
  </div>
  {_video(result.query_overlay, MODEL_LABELS[result.model] + ' 目标定位叠加')}
  <dl class="facts">
    <div><dt>检测类别</dt><dd>{html.escape(result.target_phrase)}</dd></div>
    <div><dt>检测分数</dt><dd>{_metric(result.detector_score)}</dd></div>
    <div><dt>提示帧</dt><dd>{result.prompt_frame}</dd></div>
    <div><dt>轨迹质量</dt><dd>{_metric(result.track_score)}</dd></div>
    <div><dt>有效帧比例</dt><dd>{_metric(result.valid_ratio)}</dd></div>
    <div><dt>回退 / 修复</dt><dd>{result.fallback_frames} / {result.repaired_frames}</dd></div>
  </dl>
  <img class="preview" src="{html.escape(result.query_preview)}"
       loading="lazy" alt="{html.escape(MODEL_LABELS[result.model])} query token 接触表">
</article>
"""


def _model_query_card(result: ModelCaseResult) -> str:
    minimum = min(result.query_counts) if result.query_counts else 0
    maximum = max(result.query_counts) if result.query_counts else 0
    anchor_count = result.query_counts[2] if len(result.query_counts) > 2 else 0
    return f"""
<article class="model-item">
  <div class="model-heading"><h4>{MODEL_LABELS[result.model]}</h4>
    <span class="status {'ok' if result.anchor_valid else 'warn'}">
      {'t=2 anchor有效' if result.anchor_valid else 't=2 anchor无效'}
    </span>
  </div>
  <img class="chart" src="{html.escape(result.query_chart)}" loading="lazy"
       alt="{html.escape(MODEL_LABELS[result.model])} paired query token 数量">
  <p class="result-line">Moving 每时刻选取 {minimum}–{maximum} 个 token；
     Anchor-t2 使用 {anchor_count} 个 token。</p>
</article>
"""


def _model_role_card(
    result: ModelCaseResult,
    focus_block: int,
    focus_head: int,
) -> str:
    moving_role = ROLE_ZH.get(result.moving_focus_role, "-")
    anchor_role = ROLE_ZH.get(result.anchor_focus_role, "-")
    return f"""
<article class="model-evidence">
  <div class="model-heading"><h4>{MODEL_LABELS[result.model]}</h4>
    <span class="status {'ok' if result.role_agreement >= 0.6 else 'warn'}">
      全部 block/head 角色一致率 {_metric(result.role_agreement * 100, 1)}%
    </span>
  </div>
  <div class="evidence-grid">
    <figure>
      <img class="chart" src="{html.escape(result.role_chart)}" loading="lazy"
           alt="{html.escape(MODEL_LABELS[result.model])} 角色热力图">
      <figcaption>30×24 个 block/head 的角色分类；黑框为
        Block {focus_block:02d} / Head {focus_head:02d}。</figcaption>
    </figure>
    <figure>
      <img class="chart" src="{html.escape(result.focus_chart)}" loading="lazy"
           alt="{html.escape(MODEL_LABELS[result.model])} 重点 head 判断证据">
      <figcaption>重点 head 在去噪步5、15、25、35的证据百分位和角色分数。</figcaption>
    </figure>
  </div>
  <div class="verdict">
    <strong>Block {focus_block:02d} / Head {focus_head:02d}：</strong>
    Moving 判为 <b>{result.moving_focus_role} · {moving_role}</b>
    （margin={_metric(result.moving_focus_margin)}）；
    Anchor-t2 判为 <b>{result.anchor_focus_role} · {anchor_role}</b>
    （margin={_metric(result.anchor_focus_margin)}）。
  </div>
  {_feature_table(result.focus_feature_rows)}
</article>
"""


def _case_html(
    *,
    index: int,
    path: Path,
    payload: dict[str, Any],
    input_assets: dict[str, str | None],
    results: list[ModelCaseResult],
    focus_block: int,
    focus_head: int,
) -> str:
    case = path.stem
    caption = str(payload.get("input_caption") or payload.get("prompt") or "-")
    generation = "".join(_model_generation_card(result) for result in results)
    localization = "".join(
        _model_localization_card(result) for result in results
    )
    queries = "".join(_model_query_card(result) for result in results)
    roles = "".join(
        _model_role_card(result, focus_block, focus_head) for result in results
    )
    return f"""
<main class="case-panel" id="case-{index}" data-case="{html.escape(case)}"
      {'hidden' if index else ''}>
  <section class="case-head">
    <div class="inner">
      <p class="eyebrow">随机案例 {index + 1:02d}</p>
      <h2>{html.escape(case)}</h2>
      <p class="caption">{html.escape(caption)}</p>
    </div>
  </section>

  <section class="step-band">
    <div class="inner">
      <div class="step-title"><span>0</span><div>
        <h3>输入与参照视频</h3>
        <p><b>判断依据：</b>三种方法必须读取同一个8帧上下文和同一文本条件。
        Source仅作为结果参照，不进入 paired-query 的注意力计算。</p>
      </div></div>
      <div class="input-grid">
        <figure>{_video(input_assets.get('context'), case + ' 8帧上下文')}
          <figcaption>实际输入的8帧上下文</figcaption></figure>
        <figure>{_video(input_assets.get('source'), case + ' source视频')}
          <figcaption>Source / GT参照视频</figcaption></figure>
      </div>
    </div>
  </section>

  <section class="step-band alt">
    <div class="inner">
      <div class="step-title"><span>1</span><div>
        <h3>生成并确定性重放</h3>
        <p><b>判断依据：</b>Pass 1先生成视频；Pass 2使用相同模型、case、seed和推理配置重放，
        最终视频MD5必须一致，才能确认后续Q/K来自被定位的同一个生成样本。</p>
      </div></div>
      <div class="model-grid">{generation}</div>
    </div>
  </section>

  <section class="step-band">
    <div class="inner">
      <div class="step-title"><span>2</span><div>
        <h3>在各模型输出上定位目标</h3>
        <p><b>判断依据：</b>GroundingDINO在帧0、8、24、40、48产生候选框；
        SAM2传播完整49帧mask，按检测分数、有效率、运动路径和形状稳定性选择轨迹，
        仅对有两端约束的内部缺口使用CoTracker修复。红色为mask，绿色格为最终query token。</p>
      </div></div>
      <div class="model-grid">{localization}</div>
    </div>
  </section>

  <section class="step-band alt">
    <div class="inner">
      <div class="step-title"><span>3</span><div>
        <h3>从mask映射到两套配对 query</h3>
        <p><b>判断依据：</b>取生成帧0、4、…、48对应13个latent时间，
        将mask面积池化到16×28网格；重叠比例≥0.10的格子按重叠度排序，每时刻最多8个。
        Moving使用每个时刻的目标token；Anchor-t2只使用t=2（第8帧）token。
        二者没有融合，也没有把anchor复制到其他时间。</p>
      </div></div>
      <div class="model-grid">{queries}</div>
    </div>
  </section>

  <section class="step-band">
    <div class="inner">
      <div class="step-title"><span>4</span><div>
        <h3>同一组Q/K上的精确注意力证据</h3>
        <p><b>判断依据：</b>在每个DiT block的24个head上，对选中的Q计算
        softmax(QKᵀ/√d)，归一化范围覆盖全部13×16×28=5824个K。
        Moving在各有效时间分别平均目标Q后再跨时间平均；Anchor只统计t=2。
        保存8项紧凑特征，不保存完整注意力矩阵。</p>
      </div></div>
      <div class="formula-strip">
        <span>同帧/局部质量</span><span>首帧质量</span><span>历史偏置</span>
        <span>时间距离</span><span>固定位置富集</span><span>目标轨迹富集</span>
        <span>注意力熵</span>
      </div>
    </div>
  </section>

  <section class="step-band alt role-band">
    <div class="inner">
      <div class="step-title"><span>5</span><div>
        <h3>角色判断与 paired 对照</h3>
        <p><b>判断依据：</b>先在同一block的24个head内将特征转成百分位：
        S=同帧与局部；T=70%轨迹富集+30%时间距离；P=固定位置富集；
        C=首帧与历史偏置；G=注意力熵。最高分为角色，前两名差值为margin。
        红色角色变化说明结论依赖query取法，不等于该head对生成具有因果贡献。</p>
      </div></div>
      <div class="role-stack">{roles}</div>
    </div>
  </section>
</main>
"""


def _page_html(
    *,
    cases: list[Path],
    case_sections: list[str],
    display_seed: int,
    random_seed: int,
    focus_block: int,
    focus_head: int,
) -> str:
    tabs = "".join(
        f"<button class='case-tab {'active' if index == 0 else ''}' "
        f"data-target='case-{index}' aria-controls='case-{index}' "
        f"aria-selected='{'true' if index == 0 else 'false'}'>"
        f"{index + 1:02d} · {html.escape(path.stem)}</button>"
        for index, path in enumerate(cases)
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paired-query 案例审计</title>
<style>
:root {{
  color-scheme: light;
  --ink:#20262d; --muted:#66717d; --line:#d8dde3; --paper:#ffffff;
  --band:#f3f5f7; --accent:#1f6b8f; --good:#20745a; --warn:#a34f2e;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:14px/1.55 "Noto Sans CJK SC","Microsoft YaHei",system-ui,sans-serif;
  color:var(--ink); background:var(--paper); letter-spacing:0; }}
header {{ border-bottom:1px solid var(--line); background:#fff; }}
.inner {{ width:min(1480px, calc(100% - 40px)); margin:0 auto; }}
.top {{ padding:22px 0 18px; display:flex; align-items:flex-end; justify-content:space-between; gap:24px; }}
h1,h2,h3,h4,p {{ margin-top:0; }}
h1 {{ margin-bottom:4px; font-size:25px; font-weight:720; }}
.subtitle {{ margin:0; color:var(--muted); max-width:900px; }}
.run-facts {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }}
.run-facts span,.status {{ border:1px solid var(--line); background:#f7f8f9; padding:4px 8px;
  border-radius:4px; white-space:nowrap; font-size:12px; }}
.tabs-wrap {{ position:sticky; top:0; z-index:10; background:rgba(255,255,255,.97);
  border-bottom:1px solid var(--line); overflow-x:auto; }}
.tabs {{ width:max-content; min-width:100%; padding:8px max(20px, calc((100vw - 1480px)/2)); display:flex; gap:6px; }}
.case-tab {{ border:1px solid var(--line); background:#fff; color:var(--ink); padding:7px 10px;
  border-radius:4px; cursor:pointer; max-width:290px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.case-tab.active {{ border-color:var(--accent); color:#fff; background:var(--accent); }}
.case-head {{ padding:26px 0 18px; border-bottom:1px solid var(--line); }}
.eyebrow {{ color:var(--accent); font-size:12px; font-weight:700; margin-bottom:5px; }}
h2 {{ font-size:21px; margin-bottom:5px; overflow-wrap:anywhere; }}
.caption {{ color:var(--muted); margin-bottom:0; }}
.step-band {{ padding:28px 0 34px; border-bottom:1px solid var(--line); background:#fff; }}
.step-band.alt {{ background:var(--band); }}
.step-title {{ display:grid; grid-template-columns:34px minmax(0,1fr); gap:12px; margin-bottom:18px; }}
.step-title>span {{ width:30px; height:30px; display:grid; place-items:center; border-radius:50%;
  background:var(--ink); color:#fff; font-weight:700; }}
h3 {{ font-size:18px; margin-bottom:5px; }}
.step-title p {{ margin:0; color:#4f5a65; max-width:1220px; }}
.input-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
.model-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; align-items:start; }}
.model-item,.model-evidence {{ background:#fff; border:1px solid var(--line); border-radius:6px; overflow:hidden; }}
.alt .model-item,.alt .model-evidence {{ background:#fff; }}
.model-heading {{ min-height:48px; padding:10px 12px; display:flex; align-items:center;
  justify-content:space-between; gap:8px; border-bottom:1px solid var(--line); }}
.model-heading h4 {{ font-size:15px; margin:0; }}
.status.ok {{ color:var(--good); border-color:#9cc9b9; background:#f0f8f5; }}
.status.warn {{ color:var(--warn); border-color:#dcb8a7; background:#fff6f1; }}
video {{ display:block; width:100%; aspect-ratio:16/9; object-fit:contain; background:#111; }}
figure {{ margin:0; min-width:0; }}
figcaption {{ color:var(--muted); padding:7px 10px 9px; font-size:12px; }}
.input-grid figure {{ border:1px solid var(--line); background:#fff; border-radius:6px; overflow:hidden; }}
.preview,.chart {{ display:block; width:100%; height:auto; background:#fff; }}
.preview {{ border-top:1px solid var(--line); }}
.facts {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); margin:0; padding:10px 12px; gap:7px 12px; }}
.facts div {{ min-width:0; }}
.facts dt {{ color:var(--muted); font-size:11px; }}
.facts dd {{ margin:1px 0 0; font-weight:650; overflow-wrap:anywhere; }}
.result-line {{ margin:0; padding:10px 12px 12px; color:#4f5a65; border-top:1px solid var(--line); }}
.formula-strip {{ display:grid; grid-template-columns:repeat(7,minmax(0,1fr)); gap:7px; }}
.formula-strip span {{ border-left:3px solid var(--accent); background:#eef3f6; padding:9px 8px; min-width:0; }}
.role-stack {{ display:grid; gap:18px; }}
.model-evidence {{ overflow:hidden; }}
.evidence-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1px; background:var(--line); }}
.evidence-grid figure {{ background:#fff; }}
.verdict {{ border-top:1px solid var(--line); border-bottom:1px solid var(--line); padding:11px 12px; }}
.feature-table {{ width:100%; border-collapse:collapse; font-size:12px; }}
.feature-table th,.feature-table td {{ border-bottom:1px solid #e7eaed; padding:6px 10px; text-align:right; }}
.feature-table th:first-child {{ text-align:left; font-weight:500; }}
.feature-table thead th {{ color:var(--muted); background:#fafbfc; font-weight:650; }}
.feature-table td.best {{ color:var(--good); font-weight:750; background:#eef8f3; }}
.missing-media {{ aspect-ratio:16/9; display:grid; place-items:center; padding:20px; background:#eceff2; color:var(--muted); }}
footer {{ padding:25px 0 35px; color:var(--muted); background:#fff; }}
@media (max-width:1000px) {{
  .model-grid {{ grid-template-columns:1fr; }}
  .formula-strip {{ grid-template-columns:repeat(3,minmax(0,1fr)); }}
  .top {{ align-items:flex-start; flex-direction:column; }}
  .run-facts {{ justify-content:flex-start; }}
}}
@media (max-width:720px) {{
  .inner {{ width:min(100% - 20px,1480px); }}
  .input-grid,.evidence-grid {{ grid-template-columns:1fr; }}
  .formula-strip {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
  .facts {{ grid-template-columns:1fr 1fr; }}
  h1 {{ font-size:21px; }}
  .step-band {{ padding:22px 0 26px; }}
}}
</style>
</head>
<body>
<header><div class="inner top">
  <div><h1>Paired-query 案例审计</h1>
    <p class="subtitle">逐步核查目标定位、latent token选择、同Q/K配对计算与head角色判断。
    本页展示当前共同完成seed的案例证据，不代替最终50-seed统计。</p></div>
  <div class="run-facts">
    <span>展示 seed：{display_seed:06d}</span>
    <span>案例抽样种子：{random_seed}</span>
    <span>随机案例：{len(cases)}/20</span>
    <span>重点：Block {focus_block:02d} · Head {focus_head:02d}</span>
  </div>
</div></header>
<nav class="tabs-wrap" aria-label="案例选择"><div class="tabs">{tabs}</div></nav>
{''.join(case_sections)}
<footer><div class="inner">
  结论边界：目标mask来自自动检测与跟踪；角色是基于同block内24个head相对排名的描述标签。
  paired设计控制了视频与Q/K差异，但两种协议的Q数量和时间覆盖不同，因此不能单独证明因果贡献。
</div></footer>
<script>
const tabs = [...document.querySelectorAll('.case-tab')];
const panels = [...document.querySelectorAll('.case-panel')];
tabs.forEach(tab => tab.addEventListener('click', () => {{
  tabs.forEach(item => {{
    const active = item === tab;
    item.classList.toggle('active', active);
    item.setAttribute('aria-selected', active ? 'true' : 'false');
  }});
  panels.forEach(panel => panel.hidden = panel.id !== tab.dataset.target);
  window.scrollTo({{top: document.querySelector('.tabs-wrap').offsetTop, behavior:'auto'}});
}}));
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    input_list = args.input_list.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not (0 <= args.focus_block < 30):
        raise ValueError("focus block must be in [0, 29]")
    if not (0 <= args.focus_head < 24):
        raise ValueError("focus head must be in [0, 23]")
    paths = _deduplicated_paths(input_list)
    if args.sample_count <= 0 or args.sample_count > len(paths):
        raise ValueError("sample count is outside the available case count")
    cases = random.Random(args.random_seed).sample(
        sorted(paths, key=lambda path: path.stem),
        args.sample_count,
    )
    output.mkdir(parents=True, exist_ok=True)
    sections = []
    manifest_cases = []
    for index, path in enumerate(cases):
        payload = json.loads(path.read_text(encoding="utf-8"))
        case = path.stem
        print(f"[paired-audit] {index + 1}/{len(cases)} {case}", flush=True)
        input_assets = _input_assets(
            payload,
            output / "assets" / _safe_name(case) / "input",
            output,
        )
        results = [
            _model_case_result(
                root=root,
                output=output,
                seed=args.display_seed,
                model=model,
                case=case,
                focus_block=args.focus_block,
                focus_head=args.focus_head,
            )
            for model in MODELS
        ]
        sections.append(
            _case_html(
                index=index,
                path=path,
                payload=payload,
                input_assets=input_assets,
                results=results,
                focus_block=args.focus_block,
                focus_head=args.focus_head,
            )
        )
        manifest_cases.append(
            {
                "case": case,
                "input_json": str(path),
                "models": [
                    {
                        "model": result.model,
                        "role_agreement": result.role_agreement,
                        "moving_focus_role": result.moving_focus_role,
                        "anchor_focus_role": result.anchor_focus_role,
                        "anchor_valid": result.anchor_valid,
                    }
                    for result in results
                ],
            }
        )
    page = _page_html(
        cases=cases,
        case_sections=sections,
        display_seed=args.display_seed,
        random_seed=args.random_seed,
        focus_block=args.focus_block,
        focus_head=args.focus_head,
    )
    (output / "index.html").write_text(page, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "root": str(root),
        "display_seed": args.display_seed,
        "random_seed": args.random_seed,
        "sample_count": args.sample_count,
        "focus_block": args.focus_block,
        "focus_head": args.focus_head,
        "cases": manifest_cases,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output / "index.html")


if __name__ == "__main__":
    main()
