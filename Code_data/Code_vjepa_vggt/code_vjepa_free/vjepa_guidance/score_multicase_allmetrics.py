#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


LOWER_IS_BETTER = {
    "official_pdi",
    "scale_component",
    "traj_component",
    "epsilon_rigidity",
    "vp_component",
    "wmreward_surprise",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score multiple method directories across multiple cases using the "
            "single-case physv_eval entry points, with full metric coverage."
        )
    )
    parser.add_argument(
        "--method-dir",
        action="append",
        required=True,
        help="Method spec in the form label=/abs/path/to/generated_videos_dir.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--baseline-label", type=str, default="baseline")
    parser.add_argument("--skip-wmreward", action="store_true")
    parser.add_argument("--skip-pdi", action="store_true")
    parser.add_argument("--skip-proxy", action="store_true")
    parser.add_argument("--skip-videophy2", action="store_true")
    parser.add_argument("--skip-phyground", action="store_true")
    parser.add_argument("--skip-cosmos", action="store_true")
    parser.add_argument("--skip-physics-iq", action="store_true")
    parser.add_argument("--skip-pmf", action="store_true")
    parser.add_argument("--videophy2-task", default="pc", choices=["sa", "pc", "rule"])
    parser.add_argument("--videophy2-device", default="cuda")
    parser.add_argument("--videophy2-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--videophy2-num-frames", type=int, default=32)
    parser.add_argument("--proxy-device", default="cuda")
    parser.add_argument("--phyground-general-only", action="store_true")
    parser.add_argument("--pmf-device", default="cpu")
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--pdi-python-bin", default=None)
    parser.add_argument("--pdi-cuda-visible-devices", default=None)
    parser.add_argument("--pdi-timeout-seconds", type=float, default=None)
    parser.add_argument("--wmreward-cuda-visible-devices", default=None)
    return parser.parse_args()


def parse_method_dirs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --method-dir spec (expected label=path): {spec}")
        label, path_str = spec.split("=", 1)
        label = label.strip()
        path = Path(path_str).expanduser().resolve()
        if not label:
            raise ValueError(f"Empty method label in spec: {spec}")
        if not path.is_dir():
            raise FileNotFoundError(f"Method directory not found: {path}")
        parsed.append((label, path))
    return parsed


def load_case_sidecar(sidecar_path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    input_json_path = sidecar.get("input_json")
    input_payload = None
    if isinstance(input_json_path, str) and input_json_path:
        input_json = Path(input_json_path).expanduser().resolve()
        if input_json.is_file():
            input_payload = json.loads(input_json.read_text(encoding="utf-8"))
            input_payload["_json_path"] = str(input_json)
    return sidecar, input_payload


def _first_str(payload: dict[str, Any] | None, keys: tuple[str, ...]) -> str | None:
    if payload is None:
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def collect_records(method_dirs: list[tuple[str, Path]], limit_cases: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for method_label, method_dir in method_dirs:
        local_videos = {path.stem: path for path in sorted(method_dir.glob("*.mp4"))}
        sidecars = {
            path.stem: path
            for path in sorted(method_dir.glob("*.json"))
            if path.name != "batch_manifest.json"
        }
        case_ids = sorted(set(local_videos) | set(sidecars))
        if limit_cases is not None:
            case_ids = case_ids[:limit_cases]
        for case_id in case_ids:
            local_video_path = local_videos.get(case_id)
            sidecar_path = sidecars.get(case_id)
            sidecar, input_payload = ({}, None) if sidecar_path is None else load_case_sidecar(sidecar_path)
            resolved_video_path = None
            referenced_output = sidecar.get("output_video")
            if isinstance(referenced_output, str) and referenced_output:
                candidate = Path(referenced_output).expanduser().resolve()
                if candidate.is_file():
                    resolved_video_path = candidate
            if resolved_video_path is None:
                resolved_video_path = local_video_path
            if resolved_video_path is None:
                print(
                    f"[warn] skip {method_label}/{case_id}: no local mp4 and output_video is missing or invalid",
                    flush=True,
                )
                continue
            prompt = _first_str(input_payload, ("input_caption", "prompt", "caption", "description"))
            if prompt is None:
                prompt = _first_str(sidecar, ("input_caption", "prompt", "caption", "description"))
            records.append(
                {
                    "method": method_label,
                    "method_dir": str(method_dir),
                    "case_id": case_id,
                    "video_path": str(resolved_video_path),
                    "local_video_path": str(local_video_path) if local_video_path is not None else None,
                    "sidecar_path": str(sidecar_path) if sidecar_path is not None else None,
                    "input_json": sidecar.get("input_json"),
                    "input_payload": input_payload,
                    "source_video": _first_str(input_payload, ("source_video", "source_video_path", "source")),
                    "context_video": _first_str(
                        input_payload,
                        ("input_video", "context_video", "context_video_path", "input_video_randomf"),
                    ),
                    "prompt": prompt,
                    "referenced_output_video": sidecar.get("output_video"),
                }
            )
    return records


def mean_or_none(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _metric_value(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


SUMMARY_METRICS = (
    "official_pdi",
    "scale_component",
    "traj_component",
    "epsilon_rigidity",
    "vp_component",
    "wmreward_surprise",
    "wmreward_similarity",
    "proxy_score",
    "videophy2_score",
    "phyground_general_avg",
    "phyground_physical_avg",
    "cosmos_reason1_score",
    "physics_iq_score",
    "pmf_score",
)


def build_output_payload(
    *,
    scored_rows: list[dict[str, Any]],
    method_dirs: list[tuple[str, Path]],
    baseline_label: str,
) -> dict[str, Any]:
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        by_case[row["case_id"]][row["method"]] = row
        by_method[row["method"]].append(row)

    summary_by_method: dict[str, dict[str, Any]] = {}
    for method, rows in sorted(by_method.items()):
        overlap_rows: list[dict[str, Any]] = []
        for row in rows:
            base = by_case.get(row["case_id"], {}).get(baseline_label)
            if base is not None and method != baseline_label:
                for metric_name in SUMMARY_METRICS:
                    current = _metric_value(row, metric_name)
                    base_value = _metric_value(base, metric_name)
                    delta_key = f"delta_{metric_name}_vs_baseline"
                    row[delta_key] = None if current is None or base_value is None else current - base_value
                overlap_rows.append(row)
            elif method == baseline_label:
                for metric_name in SUMMARY_METRICS:
                    delta_key = f"delta_{metric_name}_vs_baseline"
                    row[delta_key] = 0.0 if _metric_value(row, metric_name) is not None else None

        summary: dict[str, Any] = {
            "num_cases": len(rows),
            "num_overlap_with_baseline": len(overlap_rows) if method != baseline_label else len(rows),
        }
        for metric_name in SUMMARY_METRICS:
            summary[f"mean_{metric_name}"] = mean_or_none([_metric_value(row, metric_name) for row in rows])
            summary[f"mean_delta_{metric_name}_vs_baseline"] = mean_or_none(
                [_metric_value(row, f"delta_{metric_name}_vs_baseline") for row in overlap_rows]
            )
        summary_by_method[method] = summary

    ranking = sorted(
        summary_by_method.items(),
        key=lambda item: (
            item[1].get("mean_delta_wmreward_surprise_vs_baseline")
            if item[1].get("mean_delta_wmreward_surprise_vs_baseline") is not None
            else float("inf")
        ),
    )
    return {
        "baseline_label": baseline_label,
        "method_dirs": {label: str(path) for label, path in method_dirs},
        "summary_metrics": list(SUMMARY_METRICS),
        "rows": scored_rows,
        "summary_by_method": summary_by_method,
        "ranking_by_mean_delta_wmreward_surprise": [
            {"method": method, **summary} for method, summary in ranking
        ],
    }


def write_output_snapshot(out_json: Path, payload: dict[str, Any]) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_json.with_suffix(out_json.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(out_json)


def write_markdown_summary(out_md: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Multi-case Metric Summary",
        "",
        f"Baseline label: `{payload['baseline_label']}`",
        "",
        "| Method | Cases | ΔWMReward Surprise | ΔPhysics-IQ | ΔVideoPhy2 | ΔCosmos | ΔPMF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["ranking_by_mean_delta_wmreward_surprise"]:
        lines.append(
            "| {method} | {num_cases} | {d_wmr} | {d_piq} | {d_vp2} | {d_cosmos} | {d_pmf} |".format(
                method=row["method"],
                num_cases=row.get("num_cases", 0),
                d_wmr=_fmt_md(row.get("mean_delta_wmreward_surprise_vs_baseline")),
                d_piq=_fmt_md(row.get("mean_delta_physics_iq_score_vs_baseline")),
                d_vp2=_fmt_md(row.get("mean_delta_videophy2_score_vs_baseline")),
                d_cosmos=_fmt_md(row.get("mean_delta_cosmos_reason1_score_vs_baseline")),
                d_pmf=_fmt_md(row.get("mean_delta_pmf_score_vs_baseline")),
            )
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_md(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _log_metric_status(case_label: str, metric_name: str, *, ok: bool, detail: str | None = None) -> None:
    status = "ok" if ok else "error"
    suffix = f" :: {detail}" if detail else ""
    print(f"  [{status}] {case_label} :: {metric_name}{suffix}", flush=True)


def main() -> None:
    args = parse_args()
    method_dirs = parse_method_dirs(args.method_dir)
    records = collect_records(method_dirs, limit_cases=args.limit_cases)
    if not records:
        raise SystemExit("No videos found in method directories")

    pdi_score_case = None
    pdi_runner = None
    wmreward_score_case = None
    wmreward_runner = None
    proxy_score_case = None
    proxy_runner = None
    physics_iq_score_case = None
    videophy2_score_case = None
    videophy2_runner = None
    phyground_score_case = None
    phyground_runner = None
    cosmos_score_case = None
    cosmos_runner = None
    pmf_score_case = None

    if not args.skip_pdi:
        from physv_eval.single_case.pdi import score_case as pdi_score_case
        from physv_eval.official_pdi import OfficialPDIRunner

        pdi_runner = OfficialPDIRunner(
            python_bin=args.pdi_python_bin,
            cuda_visible_devices=args.pdi_cuda_visible_devices,
            timeout_seconds=args.pdi_timeout_seconds,
        )
    if not args.skip_wmreward:
        from physv_eval.single_case.wmreward import score_case as wmreward_score_case
        from physv_eval.wmreward_official import WMRewardRunner

        wmreward_runner = WMRewardRunner(cuda_visible_devices=args.wmreward_cuda_visible_devices)
    if not args.skip_proxy:
        from physv_eval.single_case.proxy import score_case as proxy_score_case
        from physv_eval.proxy_runner import ProxyRunner

        proxy_runner = ProxyRunner(device=args.proxy_device)
    if not args.skip_physics_iq:
        from physv_eval.single_case.physics_iq import score_case as physics_iq_score_case
    if not args.skip_phyground:
        from physv_eval.single_case.phyground import score_case as phyground_score_case
        from physv_eval.phyground_official import OfficialPhyGroundRunner

        phyground_runner = OfficialPhyGroundRunner()
    if not args.skip_videophy2:
        from physv_eval.single_case.videophy2 import score_case as videophy2_score_case
        from physv_eval.videophy2_auto import VideoPhy2Runner

        videophy2_runner = VideoPhy2Runner(
            device=args.videophy2_device,
            dtype=args.videophy2_dtype,
            num_frames=args.videophy2_num_frames,
        )
    if not args.skip_cosmos:
        from physv_eval.single_case.cosmos_reason1 import score_case as cosmos_score_case
        from physv_eval.cosmos_reason1_official import OfficialCosmosReason1Runner

        cosmos_runner = OfficialCosmosReason1Runner()
    if not args.skip_pmf:
        from physv_eval.single_case.pmf import score_case as pmf_score_case

    scored_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(records, start=1):
        out = dict(row)
        video_path = Path(row["video_path"])
        prompt = row.get("prompt")
        input_payload = row.get("input_payload")
        source_video = row.get("source_video")
        context_video = row.get("context_video")
        print(f"[{idx}/{len(records)}] {row['method']} :: {row['case_id']}", flush=True)

        if pdi_score_case is not None and pdi_runner is not None:
            try:
                pdi = pdi_score_case(str(video_path), text_query=prompt, runner=pdi_runner)
                out["pdi"] = pdi
                out["official_pdi"] = pdi.get("pdi_score")
                out["scale_component"] = pdi.get("scale_component")
                out["traj_component"] = pdi.get("traj_component")
                out["epsilon_rigidity"] = pdi.get("epsilon_rigidity")
                out["vp_component"] = pdi.get("vp_component")
                _log_metric_status(f"{row['method']}::{row['case_id']}", "pdi", ok=True, detail=f"pdi={out['official_pdi']}")
            except Exception as exc:
                out["pdi_error"] = repr(exc)
                _log_metric_status(f"{row['method']}::{row['case_id']}", "pdi", ok=False, detail=repr(exc))

        if wmreward_score_case is not None and wmreward_runner is not None:
            try:
                wm = wmreward_score_case(str(video_path), runner=wmreward_runner)
                out["wmreward"] = wm
                out["wmreward_surprise"] = wm.get("surprise")
                out["wmreward_similarity"] = wm.get("similarity")
                _log_metric_status(
                    f"{row['method']}::{row['case_id']}",
                    "wmreward",
                    ok=True,
                    detail=f"surprise={out['wmreward_surprise']}",
                )
            except Exception as exc:
                out["wmreward_error"] = repr(exc)
                _log_metric_status(f"{row['method']}::{row['case_id']}", "wmreward", ok=False, detail=repr(exc))

        if proxy_score_case is not None and proxy_runner is not None and context_video is not None:
            try:
                proxy = proxy_score_case(str(video_path), context_video_path=str(context_video), runner=proxy_runner)
                out["proxy"] = proxy
                out["proxy_score"] = None if proxy is None else proxy.get("score")
                _log_metric_status(
                    f"{row['method']}::{row['case_id']}",
                    "proxy",
                    ok=True,
                    detail=f"score={out['proxy_score']}",
                )
            except Exception as exc:
                out["proxy_error"] = repr(exc)
                _log_metric_status(f"{row['method']}::{row['case_id']}", "proxy", ok=False, detail=repr(exc))

        if physics_iq_score_case is not None and source_video is not None:
            try:
                piq = physics_iq_score_case(str(video_path), source_video_path=str(source_video))
                out["physics_iq"] = piq
                out["physics_iq_score"] = None if piq is None else piq.get("score")
                _log_metric_status(
                    f"{row['method']}::{row['case_id']}",
                    "physics_iq",
                    ok=True,
                    detail=f"score={out['physics_iq_score']}",
                )
            except Exception as exc:
                out["physics_iq_error"] = repr(exc)
                _log_metric_status(f"{row['method']}::{row['case_id']}", "physics_iq", ok=False, detail=repr(exc))

        if phyground_score_case is not None and phyground_runner is not None:
            try:
                laws = [] if args.phyground_general_only else None
                phyground = phyground_score_case(
                    str(video_path),
                    caption=prompt,
                    laws=laws,
                    runner=phyground_runner,
                )
                out["phyground"] = phyground
                out["phyground_general_avg"] = None if phyground is None else phyground.get("general_avg")
                out["phyground_physical_avg"] = None if phyground is None else phyground.get("physical_avg")
                _log_metric_status(
                    f"{row['method']}::{row['case_id']}",
                    "phyground",
                    ok=True,
                    detail=f"general={out['phyground_general_avg']}",
                )
            except Exception as exc:
                out["phyground_error"] = repr(exc)
                _log_metric_status(f"{row['method']}::{row['case_id']}", "phyground", ok=False, detail=repr(exc))

        if videophy2_score_case is not None and videophy2_runner is not None:
            try:
                vp2 = videophy2_score_case(
                    str(video_path),
                    task=args.videophy2_task,
                    caption=prompt,
                    runner=videophy2_runner,
                )
                out["videophy2"] = vp2
                out["videophy2_task"] = args.videophy2_task
                out["videophy2_score"] = None if vp2 is None else vp2.get("score")
                _log_metric_status(
                    f"{row['method']}::{row['case_id']}",
                    "videophy2",
                    ok=True,
                    detail=f"score={out['videophy2_score']}",
                )
            except Exception as exc:
                out["videophy2_error"] = repr(exc)
                _log_metric_status(f"{row['method']}::{row['case_id']}", "videophy2", ok=False, detail=repr(exc))

        if cosmos_score_case is not None and cosmos_runner is not None:
            try:
                cosmos = cosmos_score_case(str(video_path), runner=cosmos_runner)
                out["cosmos_reason1"] = cosmos
                out["cosmos_reason1_score"] = None if cosmos is None else cosmos.get("score")
                _log_metric_status(
                    f"{row['method']}::{row['case_id']}",
                    "cosmos_reason1",
                    ok=True,
                    detail=f"score={out['cosmos_reason1_score']}",
                )
            except Exception as exc:
                out["cosmos_reason1_error"] = repr(exc)
                _log_metric_status(f"{row['method']}::{row['case_id']}", "cosmos_reason1", ok=False, detail=repr(exc))

        if pmf_score_case is not None and source_video is not None:
            try:
                pmf_case = dict(input_payload) if isinstance(input_payload, dict) else {}
                pmf_case["video"] = str(video_path)
                pmf_case["source_video"] = str(source_video)
                pmf = pmf_score_case(
                    pmf_case,
                    source_video_path=str(source_video),
                    device=args.pmf_device,
                )
                out["pmf"] = pmf
                out["pmf_score"] = None if pmf is None else pmf.get("score")
                _log_metric_status(
                    f"{row['method']}::{row['case_id']}",
                    "pmf",
                    ok=True,
                    detail=f"score={out['pmf_score']}",
                )
            except Exception as exc:
                out["pmf_error"] = repr(exc)
                _log_metric_status(f"{row['method']}::{row['case_id']}", "pmf", ok=False, detail=repr(exc))

        scored_rows.append(out)
        if args.save_every > 0 and (idx % args.save_every == 0 or idx == len(records)):
            payload = build_output_payload(
                scored_rows=scored_rows,
                method_dirs=method_dirs,
                baseline_label=args.baseline_label,
            )
            write_output_snapshot(args.out_json, payload)
            if args.out_md is not None:
                write_markdown_summary(args.out_md, payload)

    payload = build_output_payload(
        scored_rows=scored_rows,
        method_dirs=method_dirs,
        baseline_label=args.baseline_label,
    )
    write_output_snapshot(args.out_json, payload)
    if args.out_md is not None:
        write_markdown_summary(args.out_md, payload)

    print("\nMethod summary:", flush=True)
    for summary_row in payload["ranking_by_mean_delta_wmreward_surprise"]:
        print(
            f"{summary_row['method']:24s} "
            f"Δwmr={summary_row.get('mean_delta_wmreward_surprise_vs_baseline')} "
            f"Δpiq={summary_row.get('mean_delta_physics_iq_score_vs_baseline')} "
            f"Δvp2={summary_row.get('mean_delta_videophy2_score_vs_baseline')} "
            f"Δcosmos={summary_row.get('mean_delta_cosmos_reason1_score_vs_baseline')} "
            f"Δpmf={summary_row.get('mean_delta_pmf_score_vs_baseline')}",
            flush=True,
        )
    print(f"\nWrote {args.out_json}", flush=True)
    if args.out_md is not None:
        print(f"Wrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
