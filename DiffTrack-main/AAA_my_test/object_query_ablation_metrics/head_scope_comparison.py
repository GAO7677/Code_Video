"""Strict Top100 / Bottom100 / All720 comparison for the port-8092 viewer."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable


OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/object_query_ablation_metrics")
FAST_ROOT = OUTPUT_ROOT / "head_scope_baseline_fast"
TRAJECTORY_ROOT = OUTPUT_ROOT / "head_scope_trajectory"
VIDEO_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1"
)
REQUEST_ROOT = VIDEO_ROOT.parent

SCOPES = ("top100", "bottom100", "all720")
SCOPE_LABELS = {
    "top100": "Top100 PCK Heads",
    "bottom100": "Bottom100 PCK Heads",
    "all720": "All720 Heads",
}
SCOPE_COUNTS = {"top100": 100, "bottom100": 100, "all720": 720}
MODE_ORDER = (
    "self_only",
    "self_same",
    "self_future",
    "self_past",
    "incoming_only",
    "incoming_same",
    "incoming_future",
    "incoming_past",
    "outgoing_only",
    "outgoing_same",
    "outgoing_future",
    "outgoing_past",
)


def _mode_metadata(mask_mode: str) -> dict[str, str]:
    if mask_mode.startswith("self_"):
        operator, flow = "M1", "R K/V → R Query"
    elif mask_mode.startswith("incoming_"):
        operator, flow = "M2", "C K/V → R Query"
    else:
        operator, flow = "M3", "R K/V → C Query"
    suffix = mask_mode.rsplit("_", 1)[-1]
    temporal = {
        "only": "All-time",
        "same": "Same",
        "future": "Future",
        "past": "Past",
    }.get(suffix, suffix)
    return {
        "operator": operator,
        "temporal": temporal,
        "label": f"{operator}-{temporal}",
        "flow": flow,
    }


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _fast_category(name: str) -> Callable[[dict[str, Any], str], float | None]:
    return lambda record, _region: _finite(
        record.get("metrics", {}).get("category_scores_0_100", {}).get(name)
    )


def _trajectory_object(
    name: str, *, require_quality: bool = False
) -> Callable[[dict[str, Any], str], float | None]:
    def extract(record: dict[str, Any], region: str) -> float | None:
        values = record.get("metrics", {}).get("objects", {}).get(region, {})
        if require_quality and values.get("quality_pass") is not True:
            return None
        return _finite(values.get(name))

    return extract


def _survival_object(name: str) -> Callable[[dict[str, Any], str], float | None]:
    return lambda record, region: _finite(
        record.get("metrics", {}).get("objects", {}).get(region, {}).get(name)
    )


METRICS: tuple[dict[str, Any], ...] = (
    {
        "id": "target_center_ade",
        "label": "Target Center-ADE / D0",
        "category": "轨迹",
        "source": "trajectory",
        "unit": "D0",
        "definition": "目标中心相对同 seed Baseline 的逐帧平均距离，除以 F00 bbox 对角线；仅统计三种 scope 均通过轨迹质量门的单元。",
        "extract": _trajectory_object("center_ade_norm", require_quality=True),
    },
    {
        "id": "target_center_fde",
        "label": "Target Center-FDE / D0",
        "category": "轨迹",
        "source": "trajectory",
        "unit": "D0",
        "definition": "目标在最后共同有效帧相对 Baseline 的中心距离；三种 scope 都必须通过质量门。",
        "extract": _trajectory_object("center_fde_norm", require_quality=True),
    },
    {
        "id": "target_velocity_error",
        "label": "Velocity Vector Error / D0",
        "category": "轨迹",
        "source": "trajectory",
        "unit": "D0/frame",
        "definition": "目标四帧差分速度向量相对 Baseline 的误差，除以 D0；三种 scope 都必须通过质量门。",
        "extract": _trajectory_object(
            "velocity_vector_error_norm_per_frame", require_quality=True
        ),
    },
    {
        "id": "target_point_ade",
        "label": "Target Point-ADE / D0",
        "category": "轨迹",
        "source": "trajectory",
        "unit": "D0",
        "definition": "共同可见 CoTracker 表面点相对 Baseline 的平均距离，除以 D0；三种 scope 都必须通过质量门。",
        "extract": _trajectory_object("point_ade_norm", require_quality=True),
    },
    {
        "id": "target_track_loss",
        "label": "Target Track Loss",
        "category": "可跟踪性",
        "source": "trajectory",
        "unit": "/100",
        "definition": "目标轨迹不可用比例的 0–100 分数；越大表示跟踪/对象连续性破坏越强，所有记录都保留。",
        "extract": _trajectory_object("track_loss_score_0_100"),
    },
    {
        "id": "target_local",
        "label": "Target-local Appearance",
        "category": "物体外观代理",
        "source": "fast",
        "unit": "/100",
        "definition": "冻结目标 ROI 内候选与 Baseline 的像素差异代理；会同时响应位移和外观变化，不是纯轨迹。",
        "extract": _fast_category("target_local"),
    },
    {
        "id": "temporal_appearance",
        "label": "Temporal Appearance Proxy",
        "category": "物体外观代理",
        "source": "fast",
        "unit": "/100",
        "definition": "目标 ROI 相邻帧差分相对 Baseline 的差异代理；外观闪烁和运动都会增大它。",
        "extract": _fast_category("temporal_appearance"),
    },
    {
        "id": "target_disappearance",
        "label": "Target Disappearance",
        "category": "对象存活",
        "source": "survival",
        "unit": "/100",
        "definition": "结合 DINO 身份与面积门控的目标消失分数；越大表示对象不再保持原身份/尺度。",
        "extract": _survival_object("disappearance_score_0_100"),
    },
    {
        "id": "target_identity_failure",
        "label": "Identity Failure Rate",
        "category": "对象存活",
        "source": "survival",
        "unit": "rate",
        "definition": "目标逐帧 DINO 身份相似度未通过校准阈值的比例。",
        "extract": _survival_object("identity_failure_rate"),
    },
    {
        "id": "target_area_failure",
        "label": "Area Failure Rate",
        "category": "对象存活",
        "source": "survival",
        "unit": "rate",
        "definition": "目标 mask 面积相对 Baseline 超出校准范围的帧比例。",
        "extract": _survival_object("area_failure_rate"),
    },
    {
        "id": "target_empty_mask",
        "label": "Empty-mask Rate",
        "category": "对象存活",
        "source": "survival",
        "unit": "rate",
        "definition": "SAM2 目标 mask 为空的帧比例；只表示字面消失，不覆盖身份替换。",
        "extract": _survival_object("empty_mask_rate"),
    },
    {
        "id": "outside_spillover",
        "label": "Outside-object Spillover",
        "category": "背景/传播",
        "source": "fast",
        "unit": "/100",
        "definition": "排除对象区域后的候选/Baseline 像素差异；包含阴影与接触区变化，不是纯静态背景。",
        "extract": _fast_category("outside_spillover"),
    },
    {
        "id": "global_appearance",
        "label": "Global Appearance",
        "category": "全局外观",
        "source": "fast",
        "unit": "/100",
        "definition": "全帧 SSIM/MAE 组合的可见差异代理；静态背景会稀释对象变化。",
        "extract": _fast_category("global_appearance"),
    },
)
METRIC_LOOKUP = {row["id"]: row for row in METRICS}


def _load_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _ranking_tag(record: dict[str, Any]) -> str:
    explicit = str(record.get("ranking_tag") or "")
    if explicit:
        return explicit
    variant_id = str(record.get("variant_id") or "")
    return "s039r2735" if variant_id.endswith("_s039r2735") else ""


RecordKey = tuple[str, int, str, str, str, str]


def _load_index(root: Path, filename: str) -> tuple[dict[RecordKey, dict[str, Any]], int]:
    index: dict[RecordKey, dict[str, Any]] = {}
    report_count = 0
    for path in sorted(root.glob(f"*/seed_*/{filename}")):
        payload = _load_payload(path)
        if payload is None:
            continue
        report_count += 1
        case, seed = str(payload.get("case") or ""), int(payload.get("seed", -1))
        for record in payload.get("records", []):
            if (
                not isinstance(record, dict)
                or record.get("target_scope") != "single_object"
                or not record.get("region")
                or record.get("mask_mode") not in MODE_ORDER
                or record.get("head_scope") not in SCOPES
            ):
                continue
            key = (
                case,
                seed,
                str(record["region"]),
                str(record["mask_mode"]),
                str(record["head_scope"]),
                _ranking_tag(record),
            )
            index[key] = record
    return index, report_count


def _case_balanced_mean(rows: list[tuple[str, int, float]]) -> float | None:
    by_case_seed: dict[tuple[str, int], list[float]] = defaultdict(list)
    for case, seed, value in rows:
        by_case_seed[(case, seed)].append(value)
    by_case: dict[str, list[float]] = defaultdict(list)
    for (case, _seed), values in by_case_seed.items():
        by_case[case].append(math.fsum(values) / len(values))
    if not by_case:
        return None
    case_means = [math.fsum(values) / len(values) for values in by_case.values()]
    return math.fsum(case_means) / len(case_means)


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 8)


def _fold_ratio(numerator: float | None, denominator: float | None) -> float | str | None:
    """Return a display-safe fold ratio without hiding a zero denominator."""
    if numerator is None or denominator is None:
        return None
    if abs(denominator) <= 1e-12:
        return None if abs(numerator) <= 1e-12 else "∞"
    return _round(numerator / denominator)


def _region_phrases() -> dict[tuple[str, str], str]:
    phrases: dict[tuple[str, str], str] = {}
    for path in sorted(REQUEST_ROOT.glob("cases*.json")):
        payload = _load_payload(path) or {}
        for sample in payload.get("samples", []):
            case = str(sample.get("case") or "")
            for region in sample.get("regions", []):
                name = str(region.get("region_name") or "")
                phrase = str(region.get("region_phrase") or "")
                if case and name and phrase:
                    phrases[(case, name)] = phrase
    return phrases


def _record_path(record: dict[str, Any] | None, *names: str) -> Path | None:
    if not record:
        return None
    for name in names:
        value = record.get(name)
        if value:
            path = Path(str(value))
            if path.is_file() and path.stat().st_size > 0:
                return path
    return None


def _metric_value(
    metric: dict[str, Any],
    indexes: dict[str, dict[RecordKey, dict[str, Any]]],
    key: RecordKey,
) -> float | None:
    record = indexes[metric["source"]].get(key)
    return metric["extract"](record, key[2]) if record else None


def _metric_analysis(
    metric: dict[str, Any],
    cohort: str,
    indexes: dict[str, dict[RecordKey, dict[str, Any]]],
    phrases: dict[tuple[str, str], str],
) -> dict[str, Any]:
    source_index = indexes[metric["source"]]
    grouped: dict[tuple[str, int, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for (case, seed, region, mode, scope, tag), record in source_index.items():
        if tag == cohort:
            grouped[(case, seed, region, mode, tag)][scope] = record

    complete: list[
        tuple[tuple[str, int, str, str, str], dict[str, float], dict[str, dict[str, Any]]]
    ] = []
    for unit, records in grouped.items():
        if not all(scope in records for scope in SCOPES):
            continue
        values = {
            scope: metric["extract"](records[scope], unit[2]) for scope in SCOPES
        }
        if all(value is not None for value in values.values()):
            complete.append((unit, values, records))

    wins: Counter[str] = Counter()
    ties = 0
    scope_values: dict[str, list[tuple[str, int, float]]] = {
        scope: [] for scope in SCOPES
    }
    for (case, seed, _region, _mode, _tag), values, _records in complete:
        for scope, value in values.items():
            scope_values[scope].append((case, seed, float(value)))
        maximum = max(values.values())
        winners = [
            scope for scope, value in values.items() if abs(value - maximum) <= 1e-12
        ]
        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            ties += 1

    scope_summary = []
    for scope in SCOPES:
        rows = scope_values[scope]
        raw_values = [value for _case, _seed, value in rows]
        scope_summary.append(
            {
                "head_scope": scope,
                "label": SCOPE_LABELS[scope],
                "head_count": SCOPE_COUNTS[scope],
                "case_balanced_mean": _round(_case_balanced_mean(rows)),
                "median": _round(median(raw_values) if raw_values else None),
                "wins": wins[scope],
                "win_rate": _round(wins[scope] / len(complete) if complete else None),
                "units": len(rows),
            }
        )
    summary_means = {
        row["head_scope"]: row["case_balanced_mean"] for row in scope_summary
    }
    valid_summary_means = [
        value for value in summary_means.values() if value is not None
    ]
    weakest_summary_mean = min(valid_summary_means) if valid_summary_means else None
    strongest_summary_mean = max(valid_summary_means) if valid_summary_means else None
    for row in scope_summary:
        value = row["case_balanced_mean"]
        row["fold_vs_top100"] = _fold_ratio(value, summary_means.get("top100"))
        row["fold_vs_weakest"] = _fold_ratio(value, weakest_summary_mean)
        row["max_min_fold"] = _fold_ratio(
            strongest_summary_mean, weakest_summary_mean
        )

    ranking_groups: dict[
        tuple[str, str, str], list[tuple[str, int, float]]
    ] = defaultdict(list)
    for (case, seed, region, mode, _tag), values, _records in complete:
        for scope, value in values.items():
            ranking_groups[(scope, mode, region)].append((case, seed, value))
    rankings = []
    for (scope, mode, region), rows in ranking_groups.items():
        metadata = _mode_metadata(mode)
        values = [value for _case, _seed, value in rows]
        rankings.append(
            {
                "head_scope": scope,
                "head_scope_label": SCOPE_LABELS[scope],
                "head_count": SCOPE_COUNTS[scope],
                "mask_mode": mode,
                **metadata,
                "target": region,
                "case_balanced_mean": _round(_case_balanced_mean(rows)),
                "median": _round(median(values)),
                "units": len(rows),
                "cases": len({case for case, _seed, _value in rows}),
                "case_seeds": len({(case, seed) for case, seed, _value in rows}),
            }
        )
    ranking_triplets: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rankings:
        ranking_triplets[(row["mask_mode"], row["target"])][row["head_scope"]] = row
    for scope_rows in ranking_triplets.values():
        means = {
            scope: row["case_balanced_mean"] for scope, row in scope_rows.items()
        }
        valid_means = [value for value in means.values() if value is not None]
        weakest_mean = min(valid_means) if valid_means else None
        strongest_mean = max(valid_means) if valid_means else None
        for row in scope_rows.values():
            value = row["case_balanced_mean"]
            row["fold_vs_top100"] = _fold_ratio(value, means.get("top100"))
            row["fold_vs_weakest"] = _fold_ratio(value, weakest_mean)
            row["max_min_fold"] = _fold_ratio(strongest_mean, weakest_mean)
    rankings.sort(
        key=lambda row: (
            -(
                row["case_balanced_mean"]
                if row["case_balanced_mean"] is not None
                else -math.inf
            ),
            SCOPES.index(row["head_scope"]),
            MODE_ORDER.index(row["mask_mode"]),
            row["target"],
        )
    )
    for rank, row in enumerate(rankings, 1):
        row["rank"] = rank

    representatives = []
    for (case, seed, region, mode, tag), values, records in complete:
        maximum = max(values.values())
        minimum = min(values.values())
        top100_value = values.get("top100")
        scope_rows = []
        all_generated = True
        for scope in SCOPES:
            key = (case, seed, region, mode, scope, tag)
            source_record = records[scope]
            variant_id = str(source_record.get("variant_id") or "")
            fast_record = indexes["fast"].get(key)
            trajectory_record = indexes["trajectory"].get(key)
            survival_record = indexes["survival"].get(key)
            generated = _record_path(
                trajectory_record, "video_path"
            ) or _record_path(fast_record, "path") or _record_path(
                survival_record, "video_path"
            )
            trajectory_overlay = _record_path(trajectory_record, "overlay_path")
            survival_overlay = _record_path(survival_record, "overlay_path")
            all_generated = all_generated and generated is not None
            companion = {
                row["id"]: _round(_metric_value(row, indexes, key))
                for row in METRICS
            }
            scope_rows.append(
                {
                    "head_scope": scope,
                    "label": SCOPE_LABELS[scope],
                    "head_count": SCOPE_COUNTS[scope],
                    "value": _round(values[scope]),
                    "fold_vs_top100": _fold_ratio(values[scope], top100_value),
                    "fold_vs_weakest": _fold_ratio(values[scope], minimum),
                    "variant_id": variant_id,
                    "views": {
                        "generated": generated is not None,
                        "trajectory": trajectory_overlay is not None,
                        "survival": survival_overlay is not None,
                    },
                    "metrics": companion,
                }
            )
        if not all_generated:
            continue
        metadata = _mode_metadata(mode)
        representatives.append(
            {
                "case": case,
                "seed": seed,
                "target": region,
                "target_phrase": phrases.get((case, region), ""),
                "mask_mode": mode,
                **metadata,
                "ranking_tag": tag,
                "spread": _round(maximum - minimum),
                "max_min_fold": _fold_ratio(maximum, minimum),
                "max_scope": max(values, key=values.get),
                "min_scope": min(values, key=values.get),
                "scopes": scope_rows,
            }
        )
    representatives.sort(
        key=lambda row: (
            -(row["spread"] or 0.0),
            row["case"],
            row["seed"],
            row["target"],
            MODE_ORDER.index(row["mask_mode"]),
        )
    )

    return {
        "coverage": {
            "complete_triplets": len(complete),
            "cases": len({unit[0] for unit, _values, _records in complete}),
            "case_seeds": len(
                {(unit[0], unit[1]) for unit, _values, _records in complete}
            ),
            "targets": len(
                {(unit[0], unit[2]) for unit, _values, _records in complete}
            ),
            "ties": ties,
        },
        "scope_summary": scope_summary,
        "rankings": rankings,
        "representatives": representatives[:30],
    }


def catalog() -> dict[str, Any]:
    """Return live, strictly paired head-scope rankings and representatives."""
    fast, fast_reports = _load_index(FAST_ROOT, "report.json")
    trajectory, trajectory_reports = _load_index(TRAJECTORY_ROOT, "report.json")
    survival, survival_reports = _load_index(
        TRAJECTORY_ROOT, "object_survival_report.json"
    )
    indexes = {"fast": fast, "trajectory": trajectory, "survival": survival}
    phrases = _region_phrases()
    available_tags = {
        key[-1] for index in indexes.values() for key in index if key[-1] in {"", "s039r2735"}
    }
    cohort_order = [tag for tag in ("s039r2735", "") if tag in available_tags]
    cohorts = []
    for tag in cohort_order:
        cohorts.append(
            {
                "id": tag or "pilot",
                "ranking_tag": tag,
                "label": (
                    "Latest S039 ranking · s039r2735"
                    if tag
                    else "Legacy pilot · 001460 / seed 47326"
                ),
                "metrics": {
                    metric["id"]: _metric_analysis(metric, tag, indexes, phrases)
                    for metric in METRICS
                },
            }
        )
    return {
        "ready": bool(cohorts),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "For each metric, only case×seed×target×M/time×ranking-tag units "
            "with finite Top100, Bottom100 and All720 values enter comparison. "
            "Scope means average within case-seed and then give every case equal weight. "
            "Missing values are never filled with zero."
        ),
        "effect_direction": "Every displayed metric is oriented so larger means a stronger deviation/failure relative to the same-seed Baseline; it does not necessarily mean worse physical correctness.",
        "scopes": [
            {
                "id": scope,
                "label": SCOPE_LABELS[scope],
                "head_count": SCOPE_COUNTS[scope],
            }
            for scope in SCOPES
        ],
        "metrics": [
            {key: value for key, value in metric.items() if key != "extract"}
            for metric in METRICS
        ],
        "report_inventory": {
            "fast_reports": fast_reports,
            "trajectory_reports": trajectory_reports,
            "survival_reports": survival_reports,
            "fast_records": len(fast),
            "trajectory_records": len(trajectory),
            "survival_records": len(survival),
        },
        "cohorts": cohorts,
    }


def asset(case: str, seed: int, variant_id: str, view: str) -> Path | None:
    """Resolve one exact cataloged MP4 without accepting arbitrary paths."""
    if (
        not case
        or Path(case).name != case
        or not variant_id
        or Path(variant_id).name != variant_id
        or view not in {"generated", "trajectory", "survival"}
    ):
        return None
    report_specs = (
        (TRAJECTORY_ROOT / case / f"seed_{seed:05d}" / "report.json", "trajectory"),
        (
            TRAJECTORY_ROOT
            / case
            / f"seed_{seed:05d}"
            / "object_survival_report.json",
            "survival",
        ),
        (FAST_ROOT / case / f"seed_{seed:05d}" / "report.json", "fast"),
    )
    records: dict[str, dict[str, Any]] = {}
    for path, source in report_specs:
        payload = _load_payload(path) or {}
        record = next(
            (
                row
                for row in payload.get("records", [])
                if isinstance(row, dict) and row.get("variant_id") == variant_id
            ),
            None,
        )
        if record:
            records[source] = record
    if view == "generated":
        candidate = _record_path(records.get("trajectory"), "video_path")
        candidate = candidate or _record_path(records.get("fast"), "path")
        candidate = candidate or _record_path(records.get("survival"), "video_path")
        allowed_root = VIDEO_ROOT.resolve()
    elif view == "trajectory":
        candidate = _record_path(records.get("trajectory"), "overlay_path")
        allowed_root = TRAJECTORY_ROOT.resolve()
    else:
        candidate = _record_path(records.get("survival"), "overlay_path")
        allowed_root = TRAJECTORY_ROOT.resolve()
    if candidate is None or candidate.suffix.lower() != ".mp4":
        return None
    resolved = candidate.resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError:
        return None
    return resolved


def page() -> str:
    return PAGE


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Top100 / Bottom100 / All720 Head-Scope 对比</title><style>
:root{--paper:#ebe5d7;--ink:#17241f;--deep:#173f36;--rust:#a44630;--gold:#d29b36;--line:#b9ad98;--card:#fffaf0;--muted:#706a60;--top:#22776a;--bottom:#aa4d38;--all:#6c54ad}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 0 0,#d65f3c36,transparent 34rem),radial-gradient(circle at 100% 0,#27897935,transparent 38rem),var(--paper);font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif}header{position:sticky;top:0;z-index:10;padding:14px 20px;background:#ebe5d7f3;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}a{color:var(--deep);font-weight:800}h1,h2,h3{font-family:Georgia,"Noto Serif CJK SC",serif}h1{font-size:clamp(31px,4.5vw,62px);line-height:.96;margin:5px 0}.lead{max-width:1300px;margin:7px 0;line-height:1.45}.tools{display:flex;flex-wrap:wrap;gap:8px;align-items:end}.tools label{font-size:12px;font-weight:900}.tools select,.tools button{display:block;margin-top:3px;padding:8px 10px;border:1px solid var(--line);background:var(--card);font-weight:800}.status{font:12px ui-monospace,monospace;padding:8px}main{width:min(100% - 18px,2200px);margin:auto;padding:18px 0 70px}.note,.section{margin:14px 0;padding:14px;border:1px solid var(--line);border-radius:16px;background:#fffaf0e9;box-shadow:0 12px 32px #5b493018}.note{border-left:7px solid var(--gold);line-height:1.55}.coverage{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}.coverage article{padding:10px;border:1px solid var(--line);background:#fff}.coverage b,.coverage span{display:block}.coverage b{font:900 25px ui-monospace,monospace;color:var(--deep)}.scope-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.scope{padding:13px;border:1px solid var(--line);background:#fff}.scope[data-scope=top100]{border-top:6px solid var(--top)}.scope[data-scope=bottom100]{border-top:6px solid var(--bottom)}.scope[data-scope=all720]{border-top:6px solid var(--all)}.scope strong{font-size:19px}.scope .value{font:900 31px ui-monospace,monospace;margin:8px 0}.scope dl{display:grid;grid-template-columns:1fr auto;margin:0;font-size:12px}.scope dt,.scope dd{margin:0;padding:3px}.scope dd{font-family:ui-monospace,monospace}.table-scroll{overflow:auto;border:1px solid var(--line);background:#fff;max-height:62vh}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:12px}th,td{padding:8px 10px;border-bottom:1px solid #ddd2c0;text-align:center;white-space:nowrap}thead th{position:sticky;top:0;background:var(--deep);color:#fff;z-index:1}tbody tr:first-child{background:#fff1cf}tbody tr:hover{background:#edf5ef}.low-support{color:#8d4c38}.gallery{display:grid;gap:15px}.rep{padding:13px;border:1px solid var(--line);background:#fff}.rep-head{display:flex;gap:12px;justify-content:space-between;align-items:start;flex-wrap:wrap}.rep-head h3{margin:0}.rep-meta{color:var(--muted);font:12px ui-monospace,monospace}.video-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin-top:10px}figure{margin:0;border:1px solid #d3c7b3;background:#fffaf2;padding:7px;min-width:0}video{display:block;width:100%;aspect-ratio:1280/704;background:#111}figcaption{padding:7px 2px 1px}.scope-name{display:flex;justify-content:space-between;gap:8px;font-weight:900}.metric-strip{display:grid;grid-template-columns:1fr auto;gap:3px 8px;margin-top:6px;font:11px ui-monospace,monospace}.metric-strip span:nth-child(2n){text-align:right}.definition-table td:nth-child(3){text-align:left;white-space:normal;min-width:420px}.empty{padding:35px;text-align:center;color:var(--muted)}.pill{display:inline-block;padding:3px 7px;border-radius:99px;background:#eee5d4;font:11px ui-monospace,monospace}.winner{background:#dcefe5;color:#175b49}.warn{color:#8d4c38;font-weight:900}@media(max-width:1050px){header{position:static}.coverage{grid-template-columns:repeat(2,1fr)}.scope-grid,.video-grid{grid-template-columns:1fr}}@media(max-width:600px){.coverage{grid-template-columns:1fr}}
</style></head><body><header><a href="/">返回总入口</a> · <a href="/object-query-m123-temporal-batch?v=1">返回 M1/M2/M3 批次页</a><h1>Top100 / Bottom100 / All720<br>严格配对影响排序</h1><p class="lead">同一 case、seed、Object、M/时间变体和 Head 排名快照内，只比较三种 scope 都有指标的单元。所有值越大都表示相对同 seed Baseline 的干预效应更强，不等于物理质量更差。</p><div class="tools"><label>Head 排名 cohort<select id="cohort"></select></label><label>指标<select id="metric"></select></label><label>消融<select id="operator"><option value="all">M1 + M2 + M3</option><option value="M1">M1</option><option value="M2">M2</option><option value="M3">M3</option></select></label><label>Object<select id="target"><option value="all">全部 Object</option></select></label><label>视频层<select id="view"><option value="generated">生成视频</option><option value="trajectory">轨迹 Overlay</option><option value="survival">对象存活 Overlay</option></select></label><button id="refresh">刷新数据</button><button id="replay">同步重播当前视频</button><span id="status" class="status">读取中</span></div></header><main><section class="note" id="method"></section><section class="section"><h2>当前指标样本量与 Head-Scope 总体对比</h2><div id="coverage" class="coverage"></div><div id="scopeSummary" class="scope-grid"></div></section><section class="section"><h2>完整影响排序</h2><p id="rankingNote"></p><div class="table-scroll"><table><thead><tr><th>#</th><th>Head scope</th><th>消融</th><th>信息流</th><th>Object</th><th>Case-balanced mean</th><th>vs Top100</th><th>vs 最弱组</th><th>三组 Max/Min</th><th>Median</th><th>N</th><th>Cases / Case-seeds</th></tr></thead><tbody id="ranking"></tbody></table></div></section><section class="section"><h2>代表性三列视频</h2><p>按当前指标的 <code>max(scope)−min(scope)</code> 从大到小选取；每个板块固定同一 case/seed/object/消融，只改变 Head scope。视频进入视口附近时才加载。</p><div id="gallery" class="gallery"></div></section><section class="section"><h2>指标定义</h2><div class="table-scroll"><table class="definition-table"><thead><tr><th>类别</th><th>指标</th><th>严格计算含义</th><th>单位</th><th>方向</th></tr></thead><tbody id="definitions"></tbody></table></div></section></main><script>
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let data=null,observer=null;const scopeOrder={top100:0,bottom100:1,all720:2};
function num(v,d=4){return Number.isFinite(Number(v))?Number(v).toFixed(d).replace(/0+$/,'').replace(/\.$/,''):'—'}
function fold(v){if(v===null||v===undefined||v==='')return '—';if(v==='∞')return '∞';return Number.isFinite(Number(v))?`×${num(v,2)}`:'—'}
function cohort(){return data.cohorts.find(x=>x.id===$('cohort').value)||data.cohorts[0]}function metricDef(){return data.metrics.find(x=>x.id===$('metric').value)||data.metrics[0]}function analysis(){return cohort().metrics[metricDef().id]}
function fillSelect(node,rows,value,label){const old=node.value;node.innerHTML=rows.map(x=>`<option value="${esc(value(x))}">${esc(label(x))}</option>`).join('');if([...node.options].some(x=>x.value===old))node.value=old}
function renderDefinitions(){$('definitions').innerHTML=data.metrics.map(m=>`<tr><td>${esc(m.category)}</td><td><b>${esc(m.label)}</b><br><code>${esc(m.id)}</code></td><td>${esc(m.definition)}</td><td>${esc(m.unit)}</td><td>越大＝相对 Baseline 影响越强</td></tr>`).join('')}
function filtered(rows){const op=$('operator').value,target=$('target').value;return rows.filter(r=>(op==='all'||r.operator===op)&&(target==='all'||r.target===target))}
function targets(){const values=[...new Set(Object.values(cohort().metrics).flatMap(x=>x.rankings.map(r=>r.target)))].sort();const old=$('target').value;$('target').innerHTML='<option value="all">全部 Object</option>'+values.map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('');if(values.includes(old))$('target').value=old}
function renderSummary(){const a=analysis(),m=metricDef(),c=a.coverage,inventory=data.report_inventory;$('method').innerHTML=`<b>统计口径：</b>${esc(data.method)}<br><b>倍数定义：</b>同一消融配置和 Object 下，当前 Head group 的 case-balanced mean ÷ Top100 或三组最小 mean；三组差距为 Max/Min。分母≤1e-12 时，非零/零记为 ∞，零/零记为 —；分母接近零时倍数对微小数值非常敏感，应与绝对值共同解读。<br><b>当前 cohort：</b>${esc(cohort().label)}。<b>${esc(m.label)}</b> 的三-scope 完整单元为 <b>${c.complete_triplets}</b>，覆盖 ${c.cases} cases / ${c.case_seeds} case-seeds；质量门失败会使轨迹距离指标的单元整体退出，但 Track Loss/消失指标仍保留。<br><span class="warn">当前指标回填仍在进行，刷新后样本量和排序可能增加；不要把低支持度行解释为跨 case 稳定结论。</span>`;$('coverage').innerHTML=[['完整三-Scope单元',c.complete_triplets],['Cases',c.cases],['Case-seeds',c.case_seeds],['Target slots',c.targets],['并列最大',c.ties]].map(x=>`<article><b>${x[1]}</b><span>${x[0]}</span></article>`).join('');const best=Math.max(...a.scope_summary.map(x=>Number(x.case_balanced_mean)));$('scopeSummary').innerHTML=a.scope_summary.map(x=>`<article class="scope" data-scope="${x.head_scope}"><strong>${esc(x.label)}</strong><div class="value">${num(x.case_balanced_mean)}</div><dl><dt>Case-balanced mean</dt><dd>${num(x.case_balanced_mean)}</dd><dt>vs Top100</dt><dd>${fold(x.fold_vs_top100)}</dd><dt>vs 最弱组</dt><dd>${fold(x.fold_vs_weakest)}</dd><dt>三组 Max/Min</dt><dd>${fold(x.max_min_fold)}</dd><dt>Raw median</dt><dd>${num(x.median)}</dd><dt>逐单元最大</dt><dd>${x.wins}/${x.units} · ${num(100*x.win_rate,1)}%</dd></dl>${Math.abs(Number(x.case_balanced_mean)-best)<1e-12?'<span class="pill winner">当前总体最大</span>':''}</article>`).join('');$('status').textContent=`${m.label} · ${c.complete_triplets} triplets · reports ${inventory.fast_reports}/${inventory.trajectory_reports}/${inventory.survival_reports}`}
function renderRanking(){const m=metricDef(),rows=filtered(analysis().rankings);$('rankingNote').textContent=`${m.label}（${m.unit}）：同一配置先在 seed 内汇总，再对 case 等权；倍数只在完全相同的 M/时间 × Object 内比较 Head group；N=三种 scope 均有效的严格配对单元。`; $('ranking').innerHTML=rows.length?rows.map((r,i)=>`<tr class="${r.cases<2?'low-support':''}"><td>${i+1}</td><td><b>${esc(r.head_scope_label)}</b></td><td>${esc(r.label)}<br><code>${esc(r.mask_mode)}</code></td><td>${esc(r.flow)}</td><td>${esc(r.target)}</td><td><b>${num(r.case_balanced_mean)}</b></td><td><b>${fold(r.fold_vs_top100)}</b></td><td>${fold(r.fold_vs_weakest)}</td><td>${fold(r.max_min_fold)}</td><td>${num(r.median)}</td><td>${r.units}</td><td>${r.cases} / ${r.case_seeds}${r.cases<2?' · 单 case':''}</td></tr>`).join(''):'<tr><td colspan="12">当前筛选无完整记录</td></tr>'}
function assetUrl(rep,row){const u=new URL('/api/object-query-head-scope-comparison/video',location.origin);u.searchParams.set('case',rep.case);u.searchParams.set('seed',rep.seed);u.searchParams.set('variant_id',row.variant_id);u.searchParams.set('view',$('view').value);return u}
function video(rep,row){const view=$('view').value;if(!row.views[view])return '<div class="empty">该视图尚未生成</div>';return `<video controls muted playsinline preload="none" data-src="${esc(assetUrl(rep,row))}"></video>`}
function arm(){if(observer)observer.disconnect();observer=new IntersectionObserver(entries=>entries.forEach(e=>{if(!e.isIntersecting)return;const v=e.target;if(!v.src){v.src=v.dataset.src;v.load()}observer.unobserve(v)}),{rootMargin:'700px'});document.querySelectorAll('video[data-src]').forEach(v=>observer.observe(v))}
function renderGallery(){const m=metricDef(),rows=filtered(analysis().representatives).slice(0,6);$('gallery').innerHTML=rows.length?rows.map((r,i)=>`<article class="rep"><div class="rep-head"><h3>#${i+1} · ${esc(r.case)} · seed ${r.seed}<br>${esc(r.label)} · ${esc(r.target)}${r.target_phrase?' / '+esc(r.target_phrase):''}</h3><div class="rep-meta">spread ${num(r.spread)} ${esc(m.unit)} · Max/Min ${fold(r.max_min_fold)}<br>MAX ${esc(r.max_scope)} · MIN ${esc(r.min_scope)}</div></div><div class="video-grid">${[...r.scopes].sort((a,b)=>scopeOrder[a.head_scope]-scopeOrder[b.head_scope]).map(s=>`<figure>${video(r,s)}<figcaption><div class="scope-name"><span>${esc(s.label)}</span><span>${num(s.value)} ${esc(m.unit)}</span></div><div class="metric-strip"><span>当前指标 vs Top100</span><span>${fold(s.fold_vs_top100)}</span><span>当前指标 vs 最弱组</span><span>${fold(s.fold_vs_weakest)}</span><span>Center-ADE/D0</span><span>${num(s.metrics.target_center_ade)}</span><span>Track Loss</span><span>${num(s.metrics.target_track_loss)}</span><span>Target-local</span><span>${num(s.metrics.target_local)}</span><span>Disappearance</span><span>${num(s.metrics.target_disappearance)}</span><span>Outside spillover</span><span>${num(s.metrics.outside_spillover)}</span></div></figcaption></figure>`).join('')}</div></article>`).join(''):'<div class="empty">当前筛选没有三种视频和指标都齐全的代表单元</div>';arm()}
function render(){targets();renderSummary();renderRanking();renderGallery();const u=new URL(location.href);u.searchParams.set('cohort',$('cohort').value);u.searchParams.set('metric',$('metric').value);history.replaceState(null,'',u)}
async function load(){const q=new URL(location.href).searchParams;$('status').textContent='读取严格配对指标…';data=await fetch('/api/object-query-head-scope-comparison/catalog',{cache:'no-store'}).then(r=>{if(!r.ok)throw new Error(`${r.status} ${r.statusText}`);return r.json()});if(!data.ready)throw new Error('暂无完整三-Scope配对');fillSelect($('cohort'),data.cohorts,x=>x.id,x=>x.label);fillSelect($('metric'),data.metrics,x=>x.id,x=>`${x.category} · ${x.label}`);if(q.get('cohort')&&data.cohorts.some(x=>x.id===q.get('cohort')))$('cohort').value=q.get('cohort');if(q.get('metric')&&data.metrics.some(x=>x.id===q.get('metric')))$('metric').value=q.get('metric');renderDefinitions();render()}
$('cohort').addEventListener('change',render);$('metric').addEventListener('change',render);$('operator').addEventListener('change',()=>{renderRanking();renderGallery()});$('target').addEventListener('change',()=>{renderRanking();renderGallery()});$('view').addEventListener('change',renderGallery);$('refresh').addEventListener('click',()=>load().catch(showError));$('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{if(!v.src){v.src=v.dataset.src;v.load()}const play=()=>{v.currentTime=0;v.play().catch(()=>{})};v.readyState?play():v.addEventListener('loadedmetadata',play,{once:true})}));function showError(e){$('status').textContent=`加载失败：${e.message}`}$('status').textContent='初始化…';load().catch(showError);
</script></body></html>'''
