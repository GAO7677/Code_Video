#!/usr/bin/env python3
"""Build the local static dashboard for Physics-IQ-Verified P0 comparisons.

The dashboard intentionally uses symbolic links for video assets.  It never copies
the 198-video submissions, so regenerating the page is cheap and the source media
remains in the benchmark workspace.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/physicsiq-verified-strict-dashboard"
)
INPUT_JSONS = Path(
    "/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/inputs/bpp/jsons"
)
CONDITIONING = Path(
    "/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/inputs/bpp/conditioning/24FPS"
)
GROUND_TRUTH = Path(
    "/data/gaoya/dataset/Anates-Labs-Research-Physics-IQ-Verified/split-videos/testing/24FPS"
)
XSSC_SUBMISSION = Path(
    "/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/generated_videos_5s/"
    "full_sa_no_object_gpu67_resume_step6-step-002000-36890878a58d-bpp-run_01"
)
XSSC_RAW = Path(
    "/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/raw/"
    "full_sa_no_object_gpu67_resume_step6-step-002000-36890878a58d-bpp-run_01"
)
PHYSRVG_72F_SUBMISSION = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub/"
    "physicsiq-verified-standard/assets/physrvg"
)
PHYSRVG_FULL_SA_SUBMISSION = Path(
    "/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/generated_videos_5s/"
    "physrvg-full-sa-vjepa-step000500-bpp-run_01"
)
PHYSRVG_FULL_SA_RAW = Path(
    "/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/raw/"
    "physrvg-full-sa-vjepa-step000500-bpp-run_01"
)
RESULTS_DIR = Path(
    "/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/evaluation/"
    "physics-IQ-benchmark-verified/results"
)
XSSC_CSV = RESULTS_DIR / (
    "full_sa_no_object_gpu67_resume_step6-step-002000-36890878a58d-bpp-run_01.csv"
)
XSSC_METRICS = RESULTS_DIR / (
    "full_sa_no_object_gpu67_resume_step6-step-002000-36890878a58d-bpp-run_01_metrics.json"
)
PHYSRVG_FULL_SA_CSV = RESULTS_DIR / "physrvg-full-sa-vjepa-step000500-bpp-run_01.csv"
PHYSRVG_FULL_SA_METRICS = (
    RESULTS_DIR / "physrvg-full-sa-vjepa-step000500-bpp-run_01_metrics.json"
)

VIEW_ORDER = {"left": 0, "center": 1, "right": 2}
NAME_RE = re.compile(r"^(\d{4})_perspective-(left|center|right)_(.+)\.mp4$")
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Dashboard directory (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory is unavailable: {path}")


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file is unavailable: {path}")


def ensure_directory_link(destination: Path, target: Path) -> None:
    """Create a stable directory symlink without overwriting user-owned files."""
    require_directory(target, f"asset target for {destination.name}")
    if destination.is_symlink():
        if destination.resolve() == target.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(
            f"Refusing to replace non-symlink asset directory: {destination}"
        )
    destination.symlink_to(target, target_is_directory=True)


def relative_asset(assets: Path, folder: str, filename: str) -> str | None:
    if (assets / folder / filename).is_file():
        return f"assets/{folder}/{filename}"
    return None


def clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def parse_float_list(value: str | list[float]) -> list[float]:
    if isinstance(value, list):
        values = value
    else:
        try:
            values = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid official list metric: {value[:80]!r}") from exc
    if not isinstance(values, list) or not values:
        raise ValueError("Official list metric is empty or not a list")
    # The official `physiq.calculate_iq_score.parse_list_of_floats` rounds every
    # parsed list element to four decimals before aggregation.  Mirror that
    # exactly so a mean over our case-view records reproduces the official JSON.
    parsed = [round(float(item), 4) for item in values]
    if not all(value == value for value in parsed):
        raise ValueError("Official list metric contains NaN")
    return parsed


def rounded(value: float, digits: int = 8) -> float:
    return round(float(value), digits)


def parse_view_metrics(row: dict[str, str], view: str) -> dict[str, Any]:
    """Match IQTable.compute_scores_scenario_by_view for one case and perspective."""
    suffix = f"perspective-{view}"
    spatial = float(row[f"spatial_iou_v1_{suffix}"])
    weighted_spatial = float(row[f"weighted_spatial_iou_v1_{suffix}"])
    spatiotemporal = fmean(parse_float_list(row[f"spatiotemporal_iou_v1_{suffix}"]))
    mse = fmean(parse_float_list(row[f"v1_mse_{suffix}"]))
    variance_spatial = float(row[f"variance_spatial_{suffix}"])
    variance_weighted = float(row[f"variance_weighted_spatial_{suffix}"])
    variance_spatiotemporal = fmean(
        parse_float_list(row[f"variance_spatiotemporal_iou_{suffix}"])
    )
    variance_mse = fmean(parse_float_list(row[f"variance_mse_{suffix}"]))

    components = {
        "spatial": {
            "raw": rounded(spatial),
            "variance": rounded(variance_spatial),
            "score": rounded(clip(spatial / (variance_spatial + EPS))),
        },
        "spatiotemporal": {
            "raw": rounded(spatiotemporal),
            "variance": rounded(variance_spatiotemporal),
            "score": rounded(clip(spatiotemporal / (variance_spatiotemporal + EPS))),
        },
        "weighted_spatial": {
            "raw": rounded(weighted_spatial),
            "variance": rounded(variance_weighted),
            "score": rounded(clip(weighted_spatial / (variance_weighted + EPS))),
        },
        "mse": {
            "raw": rounded(mse),
            "variance": rounded(variance_mse),
            "score": rounded(clip((mse / (variance_mse + EPS)) ** -1)),
        },
    }
    final = fmean(component["score"] for component in components.values())
    return {
        "components": components,
        "view_verified": rounded(final * 100, 4),
        "formula": "official IQTable per-view clipped arithmetic mean",
    }


def load_official_case_metrics(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    require_file(path, "official CSV")
    by_event: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 66:
        raise ValueError(f"Expected 66 official scenario rows in {path}, found {len(rows)}")
    for row in rows:
        scenario = row["scenario"]
        event = Path(scenario).stem
        by_event[event] = {
            view: parse_view_metrics(row, view) for view in VIEW_ORDER
        }
    return by_event


def score_summary_from_metrics(path: Path) -> dict[str, float]:
    require_file(path, "official metrics JSON")
    source = json.loads(path.read_text(encoding="utf-8"))
    return {
        "verified": rounded(source["final_score_view"] * 100, 4),
        "original": rounded(float(source["final_score_origround"]), 4),
        "spatial": rounded(source["score_spatial_view"] * 100, 4),
        "spatiotemporal": rounded(source["score_spatiotemporal_view"] * 100, 4),
        "weighted_spatial": rounded(source["score_weighted_spatial_view"] * 100, 4),
        "mse": rounded(source["score_mse_view"] * 100, 4),
    }


def title_for_event(event: str) -> str:
    return event.removeprefix("trimmed-").replace("-", " ")


def build_cases(
    assets: Path,
    case_metrics: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for json_path in sorted(INPUT_JSONS.glob("*.json")):
        item = json.loads(json_path.read_text(encoding="utf-8"))
        output_name = item["generated_video_name"]
        match = NAME_RE.match(output_name)
        if not match:
            raise ValueError(f"Unexpected generated video name: {output_name}")
        case_id, view, event_with_ext = match.groups()
        event = event_with_ext.removesuffix(".mp4")
        context_name = Path(item["source_video"]).name
        gt_name = context_name.replace("conditioning-videos", "testing-videos", 1)
        view_metrics = {
            method: metrics[event][view]
            for method, metrics in case_metrics.items()
            if event in metrics and view in metrics[event]
        }
        groups[event].append(
            {
                "id": case_id,
                "view": view,
                "prompt": item["input_caption"],
                "benchmark_scenario": item["benchmark_scenario"],
                "videos": {
                    "context": relative_asset(assets, "context", context_name),
                    "ground_truth": relative_asset(assets, "ground_truth", gt_name),
                    "xssc_step2000": relative_asset(assets, "xssc_step2000", output_name),
                    "xssc_step2000_raw": relative_asset(
                        assets, "xssc_step2000_raw", output_name
                    ),
                    "physrvg_72f": relative_asset(assets, "physrvg_72f", output_name),
                    "physrvg_full_sa": relative_asset(
                        assets, "physrvg_full_sa", output_name
                    ),
                    "physrvg_full_sa_raw": relative_asset(
                        assets, "physrvg_full_sa_raw", output_name
                    ),
                },
                "metrics": view_metrics,
            }
        )

    cases: list[dict[str, Any]] = []
    for event, views in groups.items():
        views.sort(key=lambda item: VIEW_ORDER[item["view"]])
        cases.append(
            {
                "event": event,
                "title": title_for_event(event),
                "first_id": min(int(item["id"]) for item in views),
                "views": views,
            }
        )
    cases.sort(key=lambda item: item["first_id"])
    if len(cases) != 66 or sum(len(case["views"]) for case in cases) != 198:
        raise ValueError("P0 input JSONs do not form the expected 66 cases / 198 views")
    return cases


def count_available(cases: list[dict[str, Any]], method: str) -> int:
    return sum(
        view["videos"].get(method) is not None
        for case in cases
        for view in case["views"]
    )


def copy_page_template(output: Path) -> None:
    template = SCRIPT_DIR / "strict_dashboard.html"
    require_file(template, "dashboard HTML template")
    target = output / "index.html"
    target.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")


def build_payload(output: Path) -> dict[str, Any]:
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    asset_targets = {
        "context": CONDITIONING,
        "ground_truth": GROUND_TRUTH,
        "xssc_step2000": XSSC_SUBMISSION,
        "xssc_step2000_raw": XSSC_RAW,
        "physrvg_72f": PHYSRVG_72F_SUBMISSION,
        "physrvg_full_sa": PHYSRVG_FULL_SA_SUBMISSION,
        "physrvg_full_sa_raw": PHYSRVG_FULL_SA_RAW,
    }
    for name, target in asset_targets.items():
        ensure_directory_link(assets / name, target)

    xssc_case_metrics = load_official_case_metrics(XSSC_CSV)
    full_sa_case_metrics = load_official_case_metrics(PHYSRVG_FULL_SA_CSV)
    cases = build_cases(
        assets,
        {
            "xssc_step2000": xssc_case_metrics,
            "physrvg_full_sa": full_sa_case_metrics,
        },
    )

    scoreboard = [
        {
            "key": "physrvg_72f",
            "label": "PhysRVG-72f-adapted",
            "short_label": "PhysRVG-72f",
            "family": "PhysRVG",
            "run": "physrvg-72f-xssc-aligned-bpp-run_01",
            "scores": {
                "verified": 39.9116,
                "original": 41.86,
                "spatial": 40.8648,
                "spatiotemporal": 51.2991,
                "weighted_spatial": 33.8715,
                "mse": 33.6110,
            },
            "video_status": "198/198 mounted locally",
            "case_metric_status": "Official per-case CSV remains on SSH 118; this page shows only its recorded global subscores.",
            "source": "P0 results registry / official aggregate recorded 2026-08-20 UTC",
            "color": "amber",
        },
        {
            "key": "physrvg_full_sa",
            "label": "PhysRVG Full-SA VJEPA",
            "short_label": "Full-SA VJEPA",
            "family": "PhysRVG",
            "run": "physrvg-full-sa-vjepa-step000500-bpp-run_01",
            "scores": score_summary_from_metrics(PHYSRVG_FULL_SA_METRICS),
            "video_status": "198/198 mounted locally",
            "case_metric_status": "Official CSV mounted locally; per-view components are shown below each video.",
            "source": "Local official metrics JSON + CSV",
            "color": "violet",
        },
        {
            "key": "xssc_step2000",
            "label": "xSSC Full-SA no-object",
            "short_label": "xSSC step-2000",
            "family": "xSSC",
            "run": "full_sa_no_object_gpu67_resume_step6-step-002000-36890878a58d-bpp-run_01",
            "scores": score_summary_from_metrics(XSSC_METRICS),
            "video_status": "198/198 mounted locally",
            "case_metric_status": "Official CSV mounted locally; per-view components are shown below each video.",
            "source": "Local official metrics JSON + CSV",
            "color": "cyan",
        },
        {
            "key": "xssc_dinov3",
            "label": "xSSC xSSC-loss DINOv3 MOVi-C",
            "short_label": "xSSC DINOv3",
            "family": "xSSC",
            "run": "full_sa_no_object_xssc_loss_dinov3_movic_step50000-step-000500-2c970f718bcf-bpp-run_01",
            "scores": {
                "verified": 33.2976,
                "original": 34.45,
                "spatial": 30.4234,
                "spatiotemporal": 50.4199,
                "weighted_spatial": 23.0123,
                "mse": 29.3349,
            },
            "video_status": "Not mounted on this host",
            "case_metric_status": "Official CSV and media remain on SSH 118; only recorded global scores are shown.",
            "source": "P0 results registry / official aggregate recorded 2026-08-12 UTC",
            "color": "rose",
        },
    ]
    scoreboard.sort(key=lambda item: item["scores"]["verified"], reverse=True)
    for rank, entry in enumerate(scoreboard, start=1):
        entry["rank"] = rank

    metric_defs = {
        "spatial": "SP · spatial IoU (v1), higher is better",
        "spatiotemporal": "ST · mean per-frame IoU (v1), higher is better",
        "weighted_spatial": "WS · weighted spatial IoU (v1), higher is better",
        "mse": "MSE · mean per-frame error (v1), lower is better",
        "view_verified": "Case-view composite · official per-view clipped arithmetic mean ×100",
    }
    asset_counts = {name: count_available(cases, name) for name in asset_targets}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "title": "Physics-IQ Verified · P0 Strict Comparison",
        "protocol": {
            "benchmark": "Physics-IQ-Verified",
            "comparison_scope": "P0 only — all four methods use the shared BPP V2V protocol.",
            "prompt": "Best-practice prompt (BPP)",
            "condition": "72 frames · 24 FPS · 3 seconds · V2V",
            "inference": "512×896 · 40 steps · guidance 5 · seed 42 · 189 raw frames",
            "submission": "Drop the 69-frame clean prefix; score 120 generated frames · 24 FPS · 5 seconds",
            "evaluator": "Official physiq/run_physics_iq.py + aggregate_runs_from_csvs.py --score-type verified",
        },
        "metric_definitions": metric_defs,
        "scoreboard": scoreboard,
        "availability": {
            "cases": len(cases),
            "views": sum(len(case["views"]) for case in cases),
            "assets": asset_counts,
            "per_view_metric_methods": ["xssc_step2000", "physrvg_full_sa"],
            "note": "The P0 scoreboard includes all four completed runs. Per-view metrics are shown only where the official CSV is mounted locally; media cards never invent a missing asset.",
        },
        "methods": [
            {
                "key": "context",
                "label": "Conditioning input",
                "detail": "72f · 24 FPS · 3s V2V context",
                "kind": "reference",
                "color": "slate",
            },
            {
                "key": "ground_truth",
                "label": "Ground truth",
                "detail": "Official testing video · 24 FPS",
                "kind": "reference",
                "color": "green",
            },
            {
                "key": "xssc_step2000",
                "label": "xSSC Full-SA no-object",
                "detail": "P0 submission · 120 generated frames · Verified 33.80",
                "kind": "model",
                "color": "cyan",
            },
            {
                "key": "xssc_dinov3",
                "label": "xSSC xSSC-loss DINOv3",
                "detail": "P0 run · global score only; media not mounted locally",
                "kind": "model",
                "color": "rose",
                "unavailable": True,
            },
            {
                "key": "physrvg_72f",
                "label": "PhysRVG-72f-adapted",
                "detail": "P0 submission · 120 generated frames · Verified 39.91",
                "kind": "model",
                "color": "amber",
            },
            {
                "key": "physrvg_full_sa",
                "label": "PhysRVG Full-SA VJEPA",
                "detail": "P0 submission · 120 generated frames · Verified 38.90",
                "kind": "model",
                "color": "violet",
            },
        ],
        "raw_methods": [
            {
                "key": "xssc_step2000_raw",
                "label": "xSSC raw trace",
                "detail": "189f · condition + prediction",
                "color": "cyan",
            },
            {
                "key": "physrvg_full_sa_raw",
                "label": "Full-SA VJEPA raw trace",
                "detail": "189f · condition + prediction",
                "color": "violet",
            },
        ],
        "cases": cases,
    }
    return payload


def main() -> None:
    args = parse_args()
    require_directory(INPUT_JSONS, "P0 input JSON")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = build_payload(output)
    copy_page_template(output)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "cases": payload["availability"]["cases"],
                "views": payload["availability"]["views"],
                "assets": payload["availability"]["assets"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
