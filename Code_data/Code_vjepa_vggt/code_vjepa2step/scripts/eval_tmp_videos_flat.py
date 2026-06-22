#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


TMP_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa2step/tmp")
OUTPUT_DIR = TMP_ROOT / "eval_json_flat"
CODE_TRY_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
DEFAULT_METRICS = ["pdi", "wmreward", "proxy", "videophy2", "phyground", "cosmos"]


def find_videos() -> list[Path]:
    return sorted(TMP_ROOT.rglob("*.mp4"))


def extract_prompt(video_path: Path) -> str | None:
    log_path = video_path.with_suffix(".log")
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for pattern in [
        r"Input prompt:\s*(.+)",
        r"prompt='([^']+)'",
    ]:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return None


def base_record(video_path: Path, prompt: str | None) -> dict[str, Any]:
    return {
        "video_path": str(video_path),
        "video_name": video_path.name,
        "video_stem": video_path.stem,
        "prompt": prompt,
        "official_pdi": None,
        "scale_component": None,
        "traj_component": None,
        "epsilon_rigidity": None,
        "vp_component": None,
        "wmreward_surprise": None,
        "wmreward_similarity": None,
        "vjepa_proxy": None,
        "videophy2_auto_pc": None,
        "videophy2_auto_sa": None,
        "phyground_general_avg": None,
        "cosmos_reason1": None,
        "jepa_score": None,
        "fid": None,
        "fvd": None,
        "cse": None,
        "tse": None,
        "accuracy": None,
        "pearson_correlation": None,
        "official_pdi_error": None,
        "wmreward_error": None,
        "vjepa_proxy_error": None,
        "videophy2_auto_pc_error": None,
        "videophy2_auto_sa_error": None,
        "phyground_error": None,
        "cosmos_reason1_error": None,
        "jepa_error": None,
        "fid_note": "requires GT/reference set",
        "fvd_note": "requires GT/reference set",
        "cse_note": "not run for these single-view videos",
        "tse_note": "not run for these single-view videos",
        "accuracy_note": "requires labeled benchmark results",
        "pearson_correlation_note": "requires labeled benchmark results",
    }


def sanitize_error(exc: Exception) -> str:
    return str(exc).strip().replace("\n", " ")[:1000]


def write_record(record: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{record['video_stem']}.json"
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_record(video_path: Path) -> dict[str, Any]:
    output_path = OUTPUT_DIR / f"{video_path.stem}.json"
    if output_path.is_file():
        return json.loads(output_path.read_text(encoding="utf-8"))
    record = base_record(video_path, extract_prompt(video_path))
    write_record(record)
    return record


def flatten_pdi(record: dict[str, Any], result: dict[str, Any]) -> None:
    record["official_pdi"] = result.get("pdi_score")
    record["scale_component"] = result.get("scale_component")
    record["traj_component"] = result.get("traj_component")
    record["epsilon_rigidity"] = result.get("epsilon_rigidity")
    record["vp_component"] = result.get("vp_component")
    record["official_pdi_error"] = None


def flatten_wmreward(record: dict[str, Any], result: dict[str, Any]) -> None:
    record["wmreward_surprise"] = result.get("surprise")
    record["wmreward_similarity"] = result.get("similarity")
    record["wmreward_error"] = None


def flatten_proxy(record: dict[str, Any], result: dict[str, Any]) -> None:
    score = result.get("score")
    record["vjepa_proxy"] = score
    record["jepa_score"] = score
    record["vjepa_proxy_error"] = None
    record["jepa_error"] = None


def maybe_cleanup_torch() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_pdi(records: dict[Path, dict[str, Any]], gpu: str) -> None:
    import sys

    if str(CODE_TRY_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_TRY_ROOT))
    from physv_eval.official_pdi import OfficialPDIRunner

    runner = OfficialPDIRunner(cuda_visible_devices=gpu)
    for video_path, record in records.items():
        text_query = record.get("prompt") or "ball"
        try:
            flatten_pdi(record, runner.run(video_path, text_query, refresh=False))
        except Exception as exc:
            record["official_pdi_error"] = sanitize_error(exc)
        write_record(record)
    del runner
    maybe_cleanup_torch()


def run_wmreward(records: dict[Path, dict[str, Any]], gpu: str) -> None:
    import sys

    if str(CODE_TRY_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_TRY_ROOT))
    from physv_eval.wmreward_official import WMRewardRunner

    runner = WMRewardRunner(cuda_visible_devices=gpu)
    for video_path, record in records.items():
        try:
            flatten_wmreward(record, runner.score(video_path))
        except Exception as exc:
            record["wmreward_error"] = sanitize_error(exc)
        write_record(record)
    del runner
    maybe_cleanup_torch()


def run_proxy(records: dict[Path, dict[str, Any]], gpu: str) -> None:
    import sys

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if str(CODE_TRY_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_TRY_ROOT))
    from physv_eval.proxy_runner import ProxyRunner

    runner = ProxyRunner(device="cuda:0")
    for video_path, record in records.items():
        try:
            result = runner.score(video_path)
            if result is None:
                record["vjepa_proxy_error"] = "proxy runner returned None"
                record["jepa_error"] = "proxy runner returned None"
            else:
                flatten_proxy(record, result)
        except Exception as exc:
            error = sanitize_error(exc)
            record["vjepa_proxy_error"] = error
            record["jepa_error"] = error
        write_record(record)
    del runner
    maybe_cleanup_torch()


def run_videophy2(records: dict[Path, dict[str, Any]], gpu: str) -> None:
    import sys

    if str(CODE_TRY_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_TRY_ROOT))
    from physv_eval.videophy2_auto import VideoPhy2Runner

    runner = VideoPhy2Runner(device=f"cuda:{gpu}" if gpu.isdigit() else "cuda")
    for video_path, record in records.items():
        try:
            pc_result = runner.score_video(video_path, task="pc")
            record["videophy2_auto_pc"] = pc_result.get("score")
        except Exception as exc:
            record["videophy2_auto_pc_error"] = sanitize_error(exc)
        write_record(record)

    for video_path, record in records.items():
        try:
            caption = record.get("prompt") or "ball"
            sa_result = runner.score_video(video_path, task="sa", caption=caption)
            record["videophy2_auto_sa"] = sa_result.get("score")
        except Exception as exc:
            record["videophy2_auto_sa_error"] = sanitize_error(exc)
        write_record(record)
    del runner
    maybe_cleanup_torch()


def run_phyground(records: dict[Path, dict[str, Any]], gpu: str) -> None:
    import sys

    if str(CODE_TRY_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_TRY_ROOT))
    from physv_eval.phyground_official import GENERAL_METRICS, OfficialPhyGroundRunner

    runner = OfficialPhyGroundRunner(cuda_visible_devices=gpu)
    for video_path, record in records.items():
        caption = record.get("prompt") or video_path.stem
        try:
            result = runner.score_bundle(video_path, caption, metrics=list(GENERAL_METRICS), laws=[])
            record["phyground_general_avg"] = result.get("general_avg")
        except Exception as exc:
            record["phyground_error"] = sanitize_error(exc)
        write_record(record)
    del runner
    maybe_cleanup_torch()


def run_cosmos_reason1(records: dict[Path, dict[str, Any]], gpu: str) -> None:
    import sys

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if str(CODE_TRY_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_TRY_ROOT))
    from physv_eval.cosmos_reason1_official import OfficialCosmosReason1Runner

    runner = OfficialCosmosReason1Runner()
    for video_path, record in records.items():
        try:
            result = runner.score(video_path)
            record["cosmos_reason1"] = result.get("score")
            record["cosmos_reason1_error"] = None
        except Exception as exc:
            record["cosmos_reason1_error"] = sanitize_error(exc)
        write_record(record)
    del runner
    maybe_cleanup_torch()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="+",
        choices=DEFAULT_METRICS,
        help="Run only selected metrics.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep existing per-video JSON files and update them in place.",
    )
    args = parser.parse_args()

    videos = find_videos()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        for stale_json in OUTPUT_DIR.glob("*.json"):
            stale_json.unlink()
    records: dict[Path, dict[str, Any]] = {}
    for video_path in videos:
        record = load_record(video_path) if args.keep_existing else base_record(video_path, extract_prompt(video_path))
        records[video_path] = record
        write_record(record)

    selected_metrics = args.only or DEFAULT_METRICS

    # Use mostly idle GPUs to avoid interfering with existing workloads on 0/1.
    if "pdi" in selected_metrics:
        run_pdi(records, gpu="2")
    if "wmreward" in selected_metrics:
        run_wmreward(records, gpu="2")
    if "proxy" in selected_metrics:
        run_proxy(records, gpu="3")
    if "videophy2" in selected_metrics:
        run_videophy2(records, gpu="4")
    if "phyground" in selected_metrics:
        run_phyground(records, gpu="5")
    if "cosmos" in selected_metrics:
        run_cosmos_reason1(records, gpu="6")

    manifest = {
        "video_count": len(videos),
        "video_root": str(TMP_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "videos": [str(path) for path in videos],
    }
    (OUTPUT_DIR / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
