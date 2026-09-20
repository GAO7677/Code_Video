"""Decompose future sphere penetration by case, frame and static collider.

This evaluator deliberately reads simulator metadata and future labels only for
offline evaluation.  None of the geometry loaded here is passed to Predictor.
The ground is the legacy infinite z=0 proxy used by the existing door metric;
declared static boxes are evaluated individually with their true OBB pose.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from prepare_small_trial import sha
from trial_metrics import rotation_xyzw

THRESHOLDS_M = (0.0, 0.002, 0.005, 0.010, 0.020)
PRIMARY_M = 0.005


def _sdf_box(points, center, half, quat):
    # trial_metrics uses the same xyzw convention.  All current door fixtures
    # are axis aligned, but retaining the OBB transform avoids baking that in.
    local = (points - center) @ rotation_xyzw(quat)
    delta = np.abs(local) - half
    return np.linalg.norm(np.maximum(delta, 0.0), axis=-1) + np.minimum(delta.max(-1), 0.0)


def colliders_for(root: Path, case: str):
    sample = root / "samples" / case
    metadata = json.loads((sample / "metadata.json").read_text())
    with np.load(sample / "raw" / "states_xyzw.npz", allow_pickle=False) as data:
        names = data["object_names"].astype(str).tolist()
        colliders = [{"name": "ground", "category": "ground", "kind": "support"}]
        for i, name in enumerate(names):
            actor = metadata["actors"][name]
            if actor.get("dynamic") or actor.get("shape") != "box":
                continue
            half = np.array([actor["size_m"][k] for k in ("hx", "hy", "hz")], dtype=float)
            colliders.append({
                "name": name,
                "category": str(actor.get("object_id", "static_box")),
                "kind": "obstacle",
                "center": data["positions"][0, i].astype(float),
                "half": half,
                "quat": data["quats"][0, i].astype(float),
            })
        dynamic = [i for i, name in enumerate(names) if metadata["actors"][name].get("dynamic")]
        if len(dynamic) != 1:
            raise ValueError(f"expected one dynamic sphere in {case}, got {dynamic}")
        actor = metadata["actors"][names[dynamic[0]]]
        if actor.get("shape") != "sphere":
            raise ValueError(f"non-sphere door evaluator input: {case}")
        radius = float(actor["size_m"]["radius"])
    return colliders, radius


def distances(points, radius, colliders):
    """Return signed sphere separation [T,C]; negative means penetration."""
    points = np.asarray(points, dtype=float)
    values = [points[:, 2] - radius]
    for collider in colliders[1:]:
        values.append(_sdf_box(points, collider["center"], collider["half"], collider["quat"]) - radius)
    return np.stack(values, axis=1)


def longest_run(mask):
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def first_transition(mask):
    mask = np.asarray(mask, dtype=bool)
    prior = False
    for i, value in enumerate(mask):
        if value and not prior:
            return i
        prior = bool(value)
    return None


def depth_stats(distance):
    depth = np.maximum(0.0, -distance)
    positive = depth[depth > 0]
    return {
        "max_depth_m": float(depth.max(initial=0.0)),
        "mean_positive_depth_m": float(positive.mean()) if positive.size else 0.0,
        "p95_positive_depth_m": float(np.quantile(positive, 0.95)) if positive.size else 0.0,
        "positive_frame_count": int((depth > 0).sum()),
        "primary_frame_count": int((depth > PRIMARY_M).sum()),
        "primary_frame_rate": float((depth > PRIMARY_M).mean()),
        "primary_longest_continuous_frames": longest_run(depth > PRIMARY_M),
        "primary_first_frame_offset": first_transition(depth > PRIMARY_M),
        "threshold_frame_rates": {
            f"{int(t * 1000)}mm": float((depth > t).mean()) for t in THRESHOLDS_M
        },
    }


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text())
    split_cases = {
        "train": [r["key"] for r in manifest["records"] if r.get("family") == "door" and r.get("split") == "train"],
        "development_val": [r["key"] for r in manifest["records"] if r.get("family") == "door" and r.get("split") == "val"],
    }
    with np.load(args.predictions, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arms = sorted({k.rsplit("_", 1)[0] for k in arrays if k.endswith("_train")})
    if "target" not in arrays and "train_target" not in arrays:
        raise ValueError("prediction archive has no target")
    per_case, per_frame, per_collider = [], [], []
    summary = {"schema": "penetration_decomposition_v1", "primary_threshold_m": PRIMARY_M,
               "thresholds_m": list(THRESHOLDS_M), "ground_proxy": "infinite z=0 plane",
               "geometry_source": "offline simulator metadata; never Predictor input",
               "arms": arms, "splits": {}}
    for split, cases in split_cases.items():
        split_rows = []
        archive_split = "val" if split == "development_val" else "train"
        for case_index, case in enumerate(cases):
            target = arrays[f"{archive_split}_target"][case_index, 0]
            colliders, radius = colliders_for(args.root, case)
            gt_dist = distances(target, radius, colliders)
            for arm in arms:
                archive_name = "development_val" if split == "development_val" else "train"
                key = f"{arm}_{archive_name}"
                if key not in arrays:
                    raise ValueError(f"missing {key} in prediction archive")
                pred = arrays[key][case_index, 0]
                pred_dist = distances(pred, radius, colliders)
                for which, dist in (("pred", pred_dist), ("gt", gt_dist)):
                    all_depth = np.maximum(0.0, -dist)
                    min_ids = dist.argmin(axis=1)
                    min_depth = all_depth[np.arange(len(dist)), min_ids]
                    # Case rows keep all threshold rates and the collider that
                    # accounts for the largest penetration.
                    row = {"split": split, "case": case, "arm": arm, "truth": which,
                           "radius_m": radius, "collider_count": len(colliders),
                           **depth_stats(-min_depth),
                           "worst_collider": colliders[int(min_ids[np.argmax(min_depth)])]["name"]}
                    for category, use in (("obstacle", np.arange(1, len(colliders))),
                                          ("ground_support", np.array([0], dtype=int))):
                        category_depth = all_depth[:, use].max(axis=1)
                        row.update({f"{category}_{k}": v for k, v in depth_stats(-category_depth).items()})
                    per_case.append(row); split_rows.append(row)
                    for frame_offset in range(len(dist)):
                        cidx = int(min_ids[frame_offset])
                        frame_row = {
                            "split": split, "case": case, "arm": arm, "truth": which,
                            "future_frame": 8 + frame_offset, "collider": colliders[cidx]["name"],
                            "collider_category": colliders[cidx]["category"],
                            "collider_kind": colliders[cidx]["kind"],
                            "min_signed_clearance_m": float(dist[frame_offset, cidx]),
                            "min_penetration_depth_m": float(min_depth[frame_offset]),
                            "penetrates_0mm": bool(min_depth[frame_offset] > 0),
                            "penetrates_2mm": bool(min_depth[frame_offset] > .002),
                            "penetrates_5mm": bool(min_depth[frame_offset] > .005),
                            "penetrates_10mm": bool(min_depth[frame_offset] > .010),
                            "penetrates_20mm": bool(min_depth[frame_offset] > .020),
                        }
                        per_frame.append(frame_row)
                    for cidx, collider in enumerate(colliders):
                        cstats = depth_stats(dist[:, cidx])
                        per_collider.append({"split": split, "case": case, "arm": arm,
                                             "truth": which, "collider": collider["name"],
                                             "collider_category": collider["category"],
                                             "collider_kind": collider["kind"], **cstats})
        summary["splits"][split] = {"case_count": len(cases),
                                     "rows": len(split_rows),
                                     "primary_pred_rows": [r for r in split_rows if r["truth"] == "pred"]}
    write_csv(args.output / "per_case.csv", per_case)
    write_csv(args.output / "per_frame.csv", per_frame)
    write_csv(args.output / "per_collider.csv", per_collider)
    # Compact arm/split aggregate is easier to quote than the long per-case list.
    aggregates = {}
    for row in per_case:
        key = (row["split"], row["arm"], row["truth"])
        bucket = aggregates.setdefault(key, [])
        bucket.append(row)
    summary["aggregate"] = {}
    for key, rows in aggregates.items():
        split, arm, truth = key
        summary["aggregate"][f"{split}/{arm}/{truth}"] = {
            "case_count": len(rows),
            "mean_max_depth_m": float(np.mean([r["max_depth_m"] for r in rows])),
            "p95_case_max_depth_m": float(np.quantile([r["max_depth_m"] for r in rows], .95)),
            "case_fraction_any_primary_penetration": float(np.mean([r["primary_frame_count"] > 0 for r in rows])),
            "mean_primary_frame_rate": float(np.mean([r["primary_frame_rate"] for r in rows])),
            "mean_primary_longest_continuous_frames": float(np.mean([r["primary_longest_continuous_frames"] for r in rows])),
            "worst_case": max(rows, key=lambda r: r["max_depth_m"])["case"],
        }
    summary["input_sha256"] = sha(args.predictions)
    summary["manifest_sha256"] = sha(args.manifest)
    (args.output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "EXECUTED", "arms": arms, "outputs": ["per_case.csv", "per_frame.csv", "per_collider.csv", "report.json"]}, indent=2))


if __name__ == "__main__":
    main()
