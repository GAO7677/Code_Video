"""Create common-support geometry-only caches for matched diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_small_trial import sha

TOKEN_COUNT = 1792
FEATURE_DIM = 1386
SEED = 20260920


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def make_case(case, a_dir, s_dir, o_dir, output):
    a = load(a_dir / case / "scene_tokens.npz")
    s = load(s_dir / case / "scene_tokens.npz")
    o = load(o_dir / case / "scene_tokens.npz")
    a_ids = a["source_flat_indices"][a["scene_mask"]].astype(np.int64)
    s_ids = set(s["source_flat_indices"][s["scene_mask"]].astype(np.int64).tolist())
    o_ids = set(o["source_flat_indices"][o["scene_mask"]].astype(np.int64).tolist())
    common = [int(x) for x in a_ids if int(x) in s_ids and int(x) in o_ids]
    if len(common) == 0:
        raise ValueError("empty common support")
    rows = {}
    for name, archive in (("estimated_aabb", a), ("estimated_sphere", s), ("oracle_partial_visible_geometry", o)):
        rows[name] = {int(idx): i for i, idx in enumerate(archive["source_flat_indices"][archive["scene_mask"]])}
    out_reports = []
    for mode, src, report_name in (("estimated_aabb", a, "estimated_aabb"),
                                   ("estimated_sphere", s, "estimated_sphere"),
                                   ("oracle_partial_visible_geometry", o, "oracle_partial_visible_geometry")):
        indices = [rows[mode][idx] for idx in common]
        xyz = src["scene_xyz"][indices].astype(np.float32)
        fyx = src["source_frame_y_x"][indices].astype(np.int64)
        ids = np.asarray(common, dtype=np.int64)
        n = len(ids)
        stored_xyz = np.zeros((TOKEN_COUNT, 3), dtype=np.float32); stored_xyz[:n] = xyz
        stored_fyx = np.full((TOKEN_COUNT, 3), -1, dtype=np.int64); stored_fyx[:n] = fyx
        stored_ids = np.full(TOKEN_COUNT, -1, dtype=np.int64); stored_ids[:n] = ids
        stored_mask = np.zeros(TOKEN_COUNT, dtype=bool); stored_mask[:n] = True
        stored_features = np.zeros((TOKEN_COUNT, FEATURE_DIM), dtype=np.float16)
        stored_conf = np.zeros(TOKEN_COUNT, dtype=np.float32); stored_conf[:n] = 1.
        out_dir = output / mode / case
        out_dir.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(out_dir / "scene_tokens.npz", scene_features=stored_features,
                            scene_xyz=stored_xyz, scene_mask=stored_mask,
                            scene_confidence=stored_conf, encoder_coords=stored_xyz.copy(),
                            source_flat_indices=stored_ids, source_frame_y_x=stored_fyx,
                            alignment_mode=np.asarray(mode))
        src_report = json.loads((({"estimated_aabb": a_dir, "estimated_sphere": s_dir,
                                  "oracle_partial_visible_geometry": o_dir}[mode] / case / "report.json").read_text()))
        report = dict(src_report)
        report.update({"schema": "matched_geometry_diagnostic_tokens_v1",
                       "cache_schema": "matched_geometry_diagnostic_tokens_v1",
                       "cache_version": 1, "alignment_mode": mode,
                       "matched_support": True, "privileged_support": True,
                       "common_support_definition": "intersection of source_frame_y_x ids from estimated_aabb, estimated_sphere and partial oracle",
                       "common_support_count": n, "sampling_seed": SEED,
                       "stored_token_count": n, "scene_features_placeholder": True,
                       "geometry_only_requires_feature_ignore": True,
                       "output_sha256": sha(out_dir / "scene_tokens.npz"),
                       "source_cache_hashes": {"estimated_aabb": sha(a_dir / case / "scene_tokens.npz"),
                                               "estimated_sphere": sha(s_dir / case / "scene_tokens.npz"),
                                               "oracle_partial_visible_geometry": sha(o_dir / case / "scene_tokens.npz")},
                       "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        (out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        out_reports.append(report)
    return {"case": case, "common_support_count": len(common),
            "aabb_sphere_ids_equal": np.array_equal(a_ids, s["source_flat_indices"][s["scene_mask"]]),
            "reports": out_reports}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aabb", type=Path, required=True)
    parser.add_argument("--sphere", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    m = json.loads(args.manifest.read_text())
    records = [r for r in m["records"] if r.get("family") == "door" and r.get("split") in {"train", "val"}]
    results, failures = [], []
    for row in records:
        try:
            results.append(make_case(row["key"], args.aabb, args.sphere, args.oracle, args.output))
            print("MATCHED_SUPPORT_READY", row["key"], results[-1]["common_support_count"], flush=True)
        except Exception as exc:
            failures.append({"case": row["key"], "reason": f"{type(exc).__name__}: {exc}"})
            print("MATCHED_SUPPORT_BLOCKED", row["key"], failures[-1]["reason"], flush=True)
    out = {"schema": "matched_geometry_diagnostic_manifest_v1", "status": "EXECUTED",
           "records_requested": len(records), "records_completed": len(results),
           "failures": failures, "privileged_support": True,
           "note": "Common support is a diagnostic protocol; it is not a deployment visibility gate."}
    (args.output / "manifest.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
