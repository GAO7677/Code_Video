"""Audit valid-token, padding and deterministic-fill semantics on actual caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diagnostic_cpu_fit import load_case_rows, make_batch
from future_query_predictor import Predictor
from prepare_small_trial import sha

SEED = 20260919


def locate(root: Path, mode: str, seed: str, case: str) -> Path:
    options = [root / mode / seed / case, root / mode / mode / case,
               root / mode / case]
    for path in options:
        if (path / "scene_tokens.npz").exists():
            return path
    raise FileNotFoundError(f"cache not found: {root}/{mode}/{seed}/{case}")


def audit_npz(path: Path, report_path: Path | None = None):
    with np.load(path / "scene_tokens.npz", allow_pickle=False) as z:
        keys = z.files
        xyz = z["scene_xyz"].astype(np.float64)
        mask = z["scene_mask"].astype(bool)
        source_flat = z["source_flat_indices"][mask] if "source_flat_indices" in keys else None
        frame_yx = z["source_frame_y_x"][mask] if "source_frame_y_x" in keys else None
        task_source = z["source_task_pool_index"][mask] if "source_task_pool_index" in keys else None
        valid_xyz = xyz[mask]
    def duplicate_count(values):
        if values is None or len(values) == 0:
            return None
        unique = np.unique(values, axis=0) if values.ndim > 1 else np.unique(values)
        return int(len(values) - len(unique))
    result = {
        "path": str(path.resolve()), "npz_sha256": sha(path / "scene_tokens.npz"),
        "array_keys": keys, "total_token_slots": int(len(mask)),
        "raw_effective_point_count": int(len(valid_xyz)),
        "scene_mask_sum": int(mask.sum()), "mask_false_padding_count": int((~mask).sum()),
        "source_flat_unique_count": int(len(np.unique(source_flat))) if source_flat is not None else None,
        "source_flat_duplicate_count": duplicate_count(source_flat),
        "source_frame_yx_unique_count": int(len(np.unique(frame_yx, axis=0))) if frame_yx is not None else None,
        "source_frame_yx_duplicate_count": duplicate_count(frame_yx),
        "same_xy_across_frames_count": (int(len(frame_yx) - len(np.unique(frame_yx[:, 1:3], axis=0)))
                                         if frame_yx is not None else None),
        "task_source_unique_count": int(len(np.unique(task_source))) if task_source is not None else None,
        "task_source_duplicate_count": duplicate_count(task_source),
        "unique_xyz_rounded_1e-6_count": int(len(np.unique(np.round(valid_xyz, 6), axis=0))),
        "exact_xyz_duplicate_count": int(len(valid_xyz) - len(np.unique(valid_xyz, axis=0))),
        "artificial_fill_detected": bool(mask.sum() < len(mask) and np.any(mask)),
    }
    if report_path is not None:
        report = json.loads(report_path.read_text())
        result["report_stored_token_count"] = report.get("stored_token_count", report.get("stored_tokens"))
        result["report_native_root_tokens"] = report.get("native_root_tokens")
        result["report_encoder_input_points"] = report.get("point_audit", {}).get("encoder_input_points")
        result["report_padding_semantics"] = report.get("padding_semantics")
    return result


def valid_only_vs_padded(root: Path, cache_root: Path, case: str, weight: Path, train_cache: Path):
    train_cases = [f"train_door_{x}" for x in ("0712", "0660", "0366", "0437", "0955", "0891")]
    train_batch, _ = make_batch(load_case_rows(root / "small_trial_120", train_cache / "estimated_aabb", train_cases))
    mean = train_batch["motion"].mean((0, 1)); std = train_batch["motion"].std((0, 1), unbiased=False).clamp_min(1e-3)
    scene_root = cache_root / "estimated_aabb"
    rows = load_case_rows(root / "small_trial_120", scene_root, [case])
    padded_batch, _ = make_batch(rows)
    row = rows[0]
    valid = row["scene_mask"]
    compact_row = dict(row)
    compact_row["scene_xyz"] = row["scene_xyz"][valid]
    compact_row["scene_features"] = row["scene_features"][valid]
    compact_row["scene_mask"] = row["scene_mask"][valid]
    compact_batch, _ = make_batch([compact_row])
    model = Predictor(mean, std, scene_mode="geometry_only", seed=SEED, strict_future_grid=True).cpu()
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
    with torch.no_grad():
        padded = model(padded_batch); compact = model(compact_batch)
    delta = (padded - compact).abs()
    return {"case": case, "valid_count": int(valid.sum()), "padded_slots": int(len(valid)),
            "max_abs_output_delta": float(delta.max()), "mean_l2_output_delta": float(torch.linalg.vector_norm(padded - compact, dim=-1).mean()),
            "tolerance_m": 1e-6, "passed": bool(float(delta.max()) <= 1e-6),
            "checkpoint_sha256": sha(weight)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    project = args.project.resolve()
    records = []
    control = project / "controlled_scene_eval_12"
    cases = ["control_door_00", "control_door_01", "control_door_02", "control_door_03"]
    # Legacy cache: this is the path whose report says native_root_tokens=58740
    # and stored_tokens=1792.
    for case in cases:
        legacy = control / "scene_features" / case
        records.append({"variant": "legacy_control", "case": case,
                        **audit_npz(legacy, legacy / "report.json")})
    for mode in ("estimated_aabb", "estimated_sphere", "oracle_partial_visible_geometry"):
        for case in cases:
            path = locate(project / "validation_20260920" / "controlled_pair_caches", mode, "", case)
            records.append({"variant": f"control_{mode}", "case": case,
                            **audit_npz(path, path / "report.json")})
    for mode in ("sparse_critical_task_geometry", "dense_critical_task_geometry"):
        path = locate(project / "validation_20260920" / "task_geometry_caches", mode, "seed20260920", cases[0])
        records.append({"variant": f"task_{mode}", "case": cases[0],
                        **audit_npz(path, path / "report.json")})
    padded = valid_only_vs_padded(project, project / "validation_20260920" / "geometry_caches_matched",
                                  "train_door_0712", project / "validation_20260920" / "leaveout_geometry_500" / "geometry_estimated_aabb_step0500.pt",
                                  project / "validation_20260920" / "geometry_caches_matched")
    result = {"schema": "padding_semantics_audit_v1", "status": "EXECUTED",
              "records": records, "valid_only_vs_padded": padded,
              "interpretation": "mask=false padded slots are ignored; duplicate valid tokens, if any, would change attention weighting and are reported separately.",
              "task_padding_expected": "M=512 valid, 1280 mask=false slots"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "valid_only_vs_padded": padded,
                      "record_count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
