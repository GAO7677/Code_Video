from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_path_id(path: Path) -> str:
    resolved = path.expanduser().resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    return f"{resolved.stem}_{digest}"


def load_payload(json_path: Path) -> dict[str, Any]:
    return json.loads(json_path.read_text(encoding="utf-8"))


def save_payload(json_path: Path, payload: dict[str, Any]) -> None:
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_video_path(json_path: Path, payload: dict[str, Any]) -> Path:
    candidates = [
        payload.get("video"),
        payload.get("video_path"),
        json_path.with_suffix(".mp4"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            return path
    raise FileNotFoundError(f"No video found for {json_path}")


def get_metric_bucket(payload: dict[str, Any]) -> dict[str, Any]:
    bucket = payload.get("metric_results")
    if not isinstance(bucket, dict):
        bucket = {}
        payload["metric_results"] = bucket
    return bucket


def get_official_pdi(payload: dict[str, Any]) -> dict[str, Any] | None:
    bucket = payload.get("metric_results", {}).get("official_pdi")
    if isinstance(bucket, dict):
        return bucket
    bucket = payload.get("metrics")
    if isinstance(bucket, dict) and bucket.get("pdi_score") is not None:
        return bucket
    bucket = payload.get("pdi")
    if isinstance(bucket, dict) and bucket.get("pdi_score") is not None:
        return bucket
    if payload.get("pdi_score") is not None:
        return {"pdi_score": payload.get("pdi_score")}
    return None


def get_wmreward(payload: dict[str, Any]) -> dict[str, Any] | None:
    bucket = payload.get("metric_results", {}).get("wmreward_jepa")
    if isinstance(bucket, dict):
        return bucket
    value = payload.get("wmreward_jepa")
    if isinstance(value, dict):
        return value
    if value is None:
        return None
    return {"similarity": value}


def get_proxy(payload: dict[str, Any]) -> dict[str, Any] | None:
    bucket = payload.get("metric_results", {}).get("vjepa_proxy")
    if isinstance(bucket, dict):
        return bucket
    if payload.get("vjepa_proxy") is not None:
        return {"score": payload.get("vjepa_proxy")}
    legacy = payload.get("jepa")
    if isinstance(legacy, dict) and legacy.get("jepa_score") is not None:
        return {"score": legacy.get("jepa_score")}
    return None


def get_videophy2_auto(payload: dict[str, Any]) -> dict[str, Any] | None:
    bucket = payload.get("metric_results", {}).get("videophy2_auto")
    if isinstance(bucket, dict):
        return bucket
    value = payload.get("videophy2_auto")
    if isinstance(value, dict):
        return value
    if payload.get("videophy2_auto_pc") is not None:
        return {"pc_score": payload.get("videophy2_auto_pc")}
    return None


def metric_value(payload: dict[str, Any], name: str) -> float | None:
    if name == "official_pdi":
        bucket = get_official_pdi(payload)
        return None if bucket is None else bucket.get("pdi_score")
    if name == "scale_component":
        bucket = get_official_pdi(payload)
        return None if bucket is None else bucket.get("scale_component", bucket.get("scale_error"))
    if name == "traj_component":
        bucket = get_official_pdi(payload)
        return None if bucket is None else bucket.get("traj_component", bucket.get("traj_error"))
    if name == "epsilon_rigidity":
        bucket = get_official_pdi(payload)
        return None if bucket is None else bucket.get("epsilon_rigidity", bucket.get("rigidity_error"))
    if name == "vp_component":
        bucket = get_official_pdi(payload)
        return None if bucket is None else bucket.get("vp_component", bucket.get("vp_error"))
    if name == "wmreward_jepa":
        bucket = get_wmreward(payload)
        return None if bucket is None else bucket.get("similarity")
    if name == "wmreward_surprise":
        bucket = get_wmreward(payload)
        if bucket is None:
            return None
        surprise = bucket.get("surprise")
        if surprise is not None:
            return surprise
        similarity = bucket.get("similarity")
        return None if similarity is None else 1.0 - similarity
    if name == "vjepa_proxy":
        bucket = get_proxy(payload)
        return None if bucket is None else bucket.get("score")
    if name == "videophy2_auto_pc":
        bucket = get_videophy2_auto(payload)
        return None if bucket is None else bucket.get("pc_score")
    raise KeyError(name)


def set_official_pdi(payload: dict[str, Any], result: dict[str, Any]) -> None:
    bucket = {
        "pdi_score": result.get("pdi_score"),
        "grade": result.get("grade"),
        "scale_component": result.get("scale_component"),
        "traj_component": result.get("traj_component"),
        "epsilon_rigidity": result.get("epsilon_rigidity"),
        "rigidity_strategy": result.get("rigidity_strategy"),
        "vp_component": result.get("vp_component"),
        "ra_math_pass": result.get("ra_math_pass"),
        "ra_ground_rmse": result.get("ra_ground_rmse"),
        "ra_scale_jump": result.get("ra_scale_jump"),
        "ra_reproj_err": result.get("ra_reproj_err"),
        "ra_overall_pass": result.get("ra_overall_pass"),
    }
    get_metric_bucket(payload)["official_pdi"] = bucket
    payload["metrics"] = dict(bucket)
    payload["pdi"] = {
        "pdi_score": bucket["pdi_score"],
        "grade": bucket["grade"],
        "scale_error": bucket["scale_component"],
        "traj_error": bucket["traj_component"],
        "rigidity_error": bucket["epsilon_rigidity"],
        "vp_error": bucket["vp_component"],
    }
    payload["pdi_score"] = bucket["pdi_score"]
    payload["raw_pdi_report_path"] = result.get("raw_report_path")


def set_wmreward(payload: dict[str, Any], result: dict[str, Any]) -> None:
    bucket = {
        "surprise": result.get("surprise"),
        "similarity": result.get("similarity"),
        "window_size": result.get("window_size"),
        "context_frames": result.get("context_frames"),
        "stride": result.get("stride"),
    }
    get_metric_bucket(payload)["wmreward_jepa"] = bucket
    payload["wmreward_jepa"] = bucket["similarity"]


def set_proxy(payload: dict[str, Any], result: dict[str, Any]) -> None:
    bucket = {
        "score": result.get("score"),
        "context_frames": result.get("context_frames"),
        "future_frames": result.get("future_frames"),
    }
    get_metric_bucket(payload)["vjepa_proxy"] = bucket
    payload["jepa"] = {"jepa_score": bucket["score"]}
    payload["vjepa_proxy"] = bucket["score"]


def set_videophy2_auto(payload: dict[str, Any], result: dict[str, Any]) -> None:
    bucket = get_metric_bucket(payload).setdefault("videophy2_auto", {})
    task = result.get("task")
    if task == "pc":
        bucket["pc_score"] = result.get("score")
        payload["videophy2_auto_pc"] = result.get("score")
    elif task == "sa":
        bucket["sa_score"] = result.get("score")
    elif task == "rule":
        bucket["rule_score"] = result.get("score")
    bucket["raw_output"] = result.get("raw_output")
    bucket["num_frames"] = result.get("num_frames")
    bucket["checkpoint"] = result.get("checkpoint")
