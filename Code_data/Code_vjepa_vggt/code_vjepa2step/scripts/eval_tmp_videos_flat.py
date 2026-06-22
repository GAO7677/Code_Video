#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
from textwrap import dedent
from pathlib import Path
from typing import Any


TMP_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa2step/tmp")
OUTPUT_DIR = TMP_ROOT / "eval_json_flat"
CODE_TRY_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
DEFAULT_METRICS = [
    "pdi",
    "wmreward",
    "proxy",
    "videophy2",
    "phyground",
    "cosmos",
    "fidfvd",
    "sampson",
    "summary",
]
GT_VIDEO = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/ball_block/e07_mu05_m1.mp4")
VPHY_PYTHON = Path("/data/gaoya/miniconda3/envs/vphy/bin/python")
FLUX_PYTHON = Path("/home/gaoya/miniconda3/envs/flux/bin/python")
SAM_PYTHON = Path("/data/gaoya/home_miniconda3/envs/sam/bin/python")


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
        "fid_error": None,
        "fvd_error": None,
        "cse_error": None,
        "tse_error": None,
        "fid_note": f"paired against GT reference {GT_VIDEO.name}",
        "fvd_note": f"paired against GT reference {GT_VIDEO.name}",
        "cse_note": f"sampson proxy against GT reference {GT_VIDEO.name}",
        "tse_note": "single-view temporal sampson proxy across consecutive frames",
        "accuracy_note": "dataset-level exact-match accuracy vs PDI-ranked pseudo labels",
        "pearson_correlation_note": "dataset-level Pearson correlation vs PDI-ranked pseudo labels",
    }


def sanitize_error(exc: Exception) -> str:
    return str(exc).strip().replace("\n", " ")[:1000]


def write_record(record: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{record['video_stem']}.json"
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_record(record: dict[str, Any], video_path: Path) -> dict[str, Any]:
    merged = base_record(video_path, extract_prompt(video_path))
    merged.update(record)
    return merged


def load_record(video_path: Path) -> dict[str, Any]:
    output_path = OUTPUT_DIR / f"{video_path.stem}.json"
    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        normalized = normalize_record(existing, video_path)
        write_record(normalized)
        return normalized
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


def _tmp_dir() -> Path:
    path = OUTPUT_DIR / "_tmp_batch"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_batch_input(records: dict[Path, dict[str, Any]], *, include_prompt: bool = False) -> Path:
    payload = []
    for video_path, record in records.items():
        row = {
            "video_path": str(video_path),
            "video_stem": record["video_stem"],
        }
        if include_prompt:
            row["prompt"] = record.get("prompt") or "ball"
        payload.append(row)
    input_path = _tmp_dir() / "batch_input.json"
    input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return input_path


def _run_subprocess(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or f"command failed: {' '.join(cmd)}"
        raise RuntimeError(detail[:2000])
    return completed


def _pearson(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    num = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b))
    den_a = sum((a - mean_a) ** 2 for a in values_a)
    den_b = sum((b - mean_b) ** 2 for b in values_b)
    den = (den_a * den_b) ** 0.5
    if den == 0:
        return None
    return float(num / den)


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
    input_path = _write_batch_input(records, include_prompt=True)
    output_path = _tmp_dir() / "videophy2_results.json"
    script = dedent(
        """
        import json, sys
        from pathlib import Path
        sys.path.insert(0, sys.argv[1])
        from physv_eval.videophy2_auto import VideoPhy2Runner

        input_path = Path(sys.argv[2])
        output_path = Path(sys.argv[3])
        items = json.loads(input_path.read_text(encoding="utf-8"))
        runner = VideoPhy2Runner(device="cpu", dtype="float32", num_frames=8)
        results = {}
        for item in items:
            video = Path(item["video_path"])
            caption = item.get("prompt") or "ball"
            row = {
                "videophy2_auto_pc": None,
                "videophy2_auto_sa": None,
                "videophy2_auto_pc_error": None,
                "videophy2_auto_sa_error": None,
            }
            try:
                row["videophy2_auto_pc"] = runner.score_video(video, task="pc").get("score")
            except Exception as exc:
                row["videophy2_auto_pc_error"] = str(exc).replace("\\n", " ")[:1000]
            try:
                row["videophy2_auto_sa"] = runner.score_video(video, task="sa", caption=caption).get("score")
            except Exception as exc:
                row["videophy2_auto_sa_error"] = str(exc).replace("\\n", " ")[:1000]
            results[str(video)] = row
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
        """
    )
    _run_subprocess([str(VPHY_PYTHON), "-c", script, str(CODE_TRY_ROOT), str(input_path), str(output_path)])
    results = json.loads(output_path.read_text(encoding="utf-8"))
    for video_path, record in records.items():
        row = results.get(str(video_path), {})
        record["videophy2_auto_pc"] = row.get("videophy2_auto_pc")
        record["videophy2_auto_sa"] = row.get("videophy2_auto_sa")
        record["videophy2_auto_pc_error"] = row.get("videophy2_auto_pc_error")
        record["videophy2_auto_sa_error"] = row.get("videophy2_auto_sa_error")
        if record["videophy2_auto_pc"] is not None:
            record["videophy2_auto_pc_error"] = None
        if record["videophy2_auto_sa"] is not None:
            record["videophy2_auto_sa_error"] = None
        write_record(record)


def run_phyground(records: dict[Path, dict[str, Any]], gpu: str) -> None:
    input_path = _write_batch_input(records, include_prompt=True)
    output_path = _tmp_dir() / "phyground_results.json"
    script = dedent(
        """
        import json, os, sys
        from pathlib import Path
        sys.path.insert(0, sys.argv[1])
        from physv_eval.phyground_official import GENERAL_METRICS, OfficialPhyGroundRunner

        input_path = Path(sys.argv[2])
        output_path = Path(sys.argv[3])
        gpu = sys.argv[4]
        items = json.loads(input_path.read_text(encoding="utf-8"))
        runner = OfficialPhyGroundRunner(
            dtype="float16",
            device_map="auto",
            cuda_visible_devices=gpu,
            max_new_tokens=256,
        )
        results = {}
        for item in items:
            video = Path(item["video_path"])
            caption = item.get("prompt") or video.stem
            row = {"phyground_general_avg": None, "phyground_error": None}
            try:
                out = runner.score_bundle(video, caption, metrics=list(GENERAL_METRICS), laws=[])
                row["phyground_general_avg"] = out.get("general_avg")
            except Exception as exc:
                row["phyground_error"] = str(exc).replace("\\n", " ")[:1000]
            results[str(video)] = row
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
        """
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _run_subprocess(
        [str(FLUX_PYTHON), "-c", script, str(CODE_TRY_ROOT), str(input_path), str(output_path), str(gpu)],
        env=env,
    )
    results = json.loads(output_path.read_text(encoding="utf-8"))
    for video_path, record in records.items():
        row = results.get(str(video_path), {})
        record["phyground_general_avg"] = row.get("phyground_general_avg")
        record["phyground_error"] = row.get("phyground_error")
        if record["phyground_general_avg"] is not None:
            record["phyground_error"] = None
        write_record(record)


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


def run_fid_fvd(records: dict[Path, dict[str, Any]]) -> None:
    input_path = _write_batch_input(records, include_prompt=False)
    output_path = _tmp_dir() / "fid_fvd_results.json"
    script = dedent(
        """
        import gc, json, sys
        from pathlib import Path

        import decord
        import numpy as np
        import torch
        from cdfvd import fvd
        from einops import rearrange
        from torch.nn.functional import interpolate
        from torchmetrics.image.fid import FID

        input_path = Path(sys.argv[1])
        output_path = Path(sys.argv[2])
        gt_video = sys.argv[3]
        device = "cpu"
        decord.bridge.set_bridge("torch")

        def load_images(video_paths, num_frames=32):
            all_images = []
            for video_path in video_paths:
                vr = decord.VideoReader(video_path)
                frame_idxs = np.arange(0, len(vr))
                raw_video = vr.get_batch(frame_idxs)
                video = raw_video[:num_frames] if num_frames is not None else raw_video
                video = video.permute(0, 3, 1, 2).float().to(device)
                video = interpolate(video, (299, 299), mode="bilinear", align_corners=False)
                video = video.clamp(0, 255).to(torch.uint8)
                all_images.extend([frame.unsqueeze(0) for frame in video])
                del video, vr
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return all_images

        def compute_fid(pred_video, gt_video):
            metric = FID(feature=2048).to(device)
            metric.reset()
            gt_images = load_images([gt_video], num_frames=32)
            pred_images = load_images([pred_video], num_frames=32)
            chunk = 64
            for i in range(0, len(gt_images), chunk):
                metric.update(torch.cat(gt_images[i:i + chunk]), real=True)
            for i in range(0, len(pred_images), chunk):
                metric.update(torch.cat(pred_images[i:i + chunk]), real=False)
            value = float(metric.compute().item())
            del metric, gt_images, pred_images
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return value

        def load_videos_for_fvd(video_paths, num_frames=32, target_size=(224, 224)):
            videos = []
            for video_path in video_paths:
                vr = decord.VideoReader(video_path)
                frame_idxs = np.arange(0, len(vr))
                raw_video = vr.get_batch(frame_idxs)
                video = raw_video[:num_frames] if num_frames is not None else raw_video
                video = video.permute(0, 3, 1, 2).float().to(device)
                video = interpolate(video, target_size, mode="bilinear", align_corners=False)
                videos.append(rearrange(video.cpu(), "t c h w -> c t h w") / 255.0)
                del video, vr
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return videos

        def compute_fvd(pred_video, gt_video):
            videos_real = load_videos_for_fvd([gt_video], num_frames=32, target_size=(224, 224))
            videos_fake = load_videos_for_fvd([pred_video], num_frames=32, target_size=(224, 224))
            evaluator = fvd.cdfvd("i3d", n_real="full", n_fake="full", ckpt_path=None, device="cpu")
            evaluator.compute_real_stats([{"video": torch.stack(videos_real)}])
            evaluator.compute_fake_stats([{"video": torch.stack(videos_fake)}])
            value = float(evaluator.compute_fvd_from_stats())
            del videos_real, videos_fake, evaluator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return value

        items = json.loads(input_path.read_text(encoding="utf-8"))
        results = {}
        for item in items:
            video_path = item["video_path"]
            row = {"fid": None, "fvd": None, "fid_error": None, "fvd_error": None}
            try:
                row["fid"] = compute_fid(video_path, gt_video)
            except Exception as exc:
                row["fid_error"] = str(exc).replace("\\n", " ")[:1000]
            try:
                row["fvd"] = compute_fvd(video_path, gt_video)
            except Exception as exc:
                row["fvd_error"] = str(exc).replace("\\n", " ")[:1000]
            results[video_path] = row
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
        """
    )
    _run_subprocess(
        [
            str(SAM_PYTHON),
            "-c",
            script,
            str(input_path),
            str(output_path),
            str(GT_VIDEO),
        ]
    )
    results = json.loads(output_path.read_text(encoding="utf-8"))
    for video_path, record in records.items():
        row = results.get(str(video_path), {})
        record["fid"] = row.get("fid")
        record["fvd"] = row.get("fvd")
        record["fid_error"] = row.get("fid_error")
        record["fvd_error"] = row.get("fvd_error")
        if record["fid"] is not None:
            record["fid_error"] = None
        if record["fvd"] is not None:
            record["fvd_error"] = None
        write_record(record)


def run_sampson(records: dict[Path, dict[str, Any]]) -> None:
    input_path = _write_batch_input(records, include_prompt=False)
    output_path = _tmp_dir() / "sampson_results.json"
    script = dedent(
        """
        import json, math, cv2, numpy as np, sys
        from pathlib import Path

        PENALTY = 1000.0

        def read_sampled_frames(video_path, n=8):
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()
            if not frames:
                return []
            if len(frames) <= n:
                return frames
            idxs = np.linspace(0, len(frames) - 1, n).round().astype(int).tolist()
            return [frames[i] for i in idxs]

        def sampson_mean(img1, img2):
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(nfeatures=1500)
            k1, d1 = orb.detectAndCompute(gray1, None)
            k2, d2 = orb.detectAndCompute(gray2, None)
            if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
                return PENALTY
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(matcher.match(d1, d2), key=lambda m: m.distance)[:256]
            if len(matches) < 8:
                return PENALTY
            pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([k2[m.trainIdx].pt for m in matches])
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
            if F is None or mask is None:
                return PENALTY
            if getattr(F, "ndim", 0) != 2 or F.shape[1] != 3 or F.shape[0] < 3:
                return PENALTY
            if F.shape[0] > 3:
                F = F[:3, :]
            mask = mask.ravel().astype(bool)
            pts1 = pts1[mask]
            pts2 = pts2[mask]
            if len(pts1) < 8:
                return PENALTY
            vals = []
            for p1, p2 in zip(pts1, pts2):
                x1 = np.array([p1[0], p1[1], 1.0], dtype=np.float64)
                x2 = np.array([p2[0], p2[1], 1.0], dtype=np.float64)
                Fx1 = F @ x1
                Ftx2 = F.T @ x2
                num = float((x2.T @ F @ x1) ** 2)
                den = float(Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2)
                if den > 1e-12:
                    vals.append(num / den)
            if not vals:
                return PENALTY
            return float(np.mean(vals))

        def compute_tse(frames):
            if len(frames) < 2:
                return PENALTY
            vals = [sampson_mean(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
            return float(np.mean(vals))

        def compute_cse(pred_frames, gt_frames):
            if not pred_frames or not gt_frames:
                return PENALTY
            count = min(len(pred_frames), len(gt_frames))
            pred_idxs = np.linspace(0, len(pred_frames) - 1, count).round().astype(int)
            gt_idxs = np.linspace(0, len(gt_frames) - 1, count).round().astype(int)
            vals = [sampson_mean(pred_frames[i], gt_frames[j]) for i, j in zip(pred_idxs, gt_idxs)]
            return float(np.mean(vals))

        input_path = Path(sys.argv[1])
        output_path = Path(sys.argv[2])
        gt_video = Path(sys.argv[3])
        items = json.loads(input_path.read_text(encoding="utf-8"))
        gt_frames = read_sampled_frames(gt_video, n=8)
        results = {}
        for item in items:
            video = Path(item["video_path"])
            row = {"tse": None, "cse": None, "tse_error": None, "cse_error": None}
            try:
                pred_frames = read_sampled_frames(video, n=8)
                row["tse"] = compute_tse(pred_frames)
                row["cse"] = compute_cse(pred_frames, gt_frames)
            except Exception as exc:
                msg = str(exc).replace("\\n", " ")[:1000]
                row["tse_error"] = msg
                row["cse_error"] = msg
            results[str(video)] = row
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
        """
    )
    _run_subprocess([str(SAM_PYTHON), "-c", script, str(input_path), str(output_path), str(GT_VIDEO)])
    results = json.loads(output_path.read_text(encoding="utf-8"))
    for video_path, record in records.items():
        row = results.get(str(video_path), {})
        record["tse"] = row.get("tse")
        record["cse"] = row.get("cse")
        record["tse_error"] = row.get("tse_error")
        record["cse_error"] = row.get("cse_error")
        if record["tse"] is not None:
            record["tse_error"] = None
        if record["cse"] is not None:
            record["cse_error"] = None
        write_record(record)


def run_summary(records: dict[Path, dict[str, Any]]) -> None:
    valid = [(path, record) for path, record in records.items() if record.get("official_pdi") is not None and record.get("videophy2_auto_pc") is not None]
    if len(valid) < 2:
        return
    ordered = sorted(valid, key=lambda item: float(item[1]["official_pdi"]))
    n = len(ordered)
    pseudo_labels: dict[Path, int] = {}
    for rank, (video_path, _) in enumerate(ordered):
        pseudo_labels[video_path] = int(round(5 - (rank * 4 / max(1, n - 1))))
        pseudo_labels[video_path] = max(1, min(5, pseudo_labels[video_path]))
    preds = [float(record["videophy2_auto_pc"]) for _, record in ordered]
    gts = [float(pseudo_labels[path]) for path, _ in ordered]
    exact = [1.0 if int(round(pred)) == int(gt) else 0.0 for pred, gt in zip(preds, gts)]
    dataset_accuracy = float(sum(exact) / len(exact))
    dataset_pearson = _pearson(gts, preds)
    for video_path, record in records.items():
        if video_path in pseudo_labels:
            record["accuracy"] = dataset_accuracy
            record["pearson_correlation"] = dataset_pearson
            record["accuracy_note"] = "dataset-level exact-match accuracy vs PDI-ranked pseudo labels"
            record["pearson_correlation_note"] = "dataset-level Pearson correlation vs PDI-ranked pseudo labels"
        write_record(record)


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
    if "fidfvd" in selected_metrics:
        run_fid_fvd(records)
    if "sampson" in selected_metrics:
        run_sampson(records)
    if "summary" in selected_metrics:
        run_summary(records)

    manifest = {
        "video_count": len(videos),
        "video_root": str(TMP_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "videos": [str(path) for path in videos],
    }
    (OUTPUT_DIR / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
