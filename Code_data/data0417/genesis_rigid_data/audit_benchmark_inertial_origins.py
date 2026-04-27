#!/usr/bin/env python3
"""Audit benchmark PhysXNet rigid assets for suspicious inertial origins.

该脚本用于审计 benchmark 中 PhysXNet 刚体资产的惯性原点是否异常；输入为 benchmark 根目录及其 URDF/碰撞网格，输出为终端告警和可写出的审计报告 json。

The main check is simple:
1. Parse each cached URDF link inertial origin.
2. Parse the link collision mesh filenames referenced by the URDF.
3. Verify the inertial origin lies inside the combined collision-mesh AABB.

The script also maps suspicious object ids back to exported sample directories,
so downstream regeneration can target exactly the affected samples.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import trimesh


CURRENT_INERTIAL_POLICY_VERSION = "v2_bbox_fallback_for_nonvolume_meshes"
DEFAULT_BENCHMARK_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data_benchmark")


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_relpath(path: Path, start: Path) -> str:
    try:
        return str(path.relative_to(start))
    except Exception:
        return str(path)


def detect_benchmark_roots(root: Path) -> List[Path]:
    roots: set[Path] = set()
    marker_names = ("benchmark_manifest.json", "heldout_ids.txt")
    for marker_name in marker_names:
        for marker in root.rglob(marker_name):
            roots.add(marker.parent.resolve())
    if (root / "_asset_cache" / "physxnet_objects").exists():
        roots.add(root.resolve())
    return sorted(roots)


def detect_benchmark_type(root: Path) -> str:
    if (root / "heldout_ids.txt").exists():
        return "stage1_heldout"
    if (root / "benchmark_manifest.json").exists():
        return "physxnet_pool"
    return "unknown"


def iter_sample_metadata_paths(root: Path) -> Iterable[Path]:
    train_root = root / "train"
    if not train_root.exists():
        return []
    return sorted(train_root.rglob("metadata.json"))


def collect_sample_dirs_by_object(root: Path) -> Dict[str, List[str]]:
    grouped: Dict[str, set[str]] = defaultdict(set)
    for meta_path in iter_sample_metadata_paths(root):
        try:
            payload = _load_json(meta_path)
        except Exception:
            continue
        sample_dir = str(meta_path.parent)
        for obj in payload.get("objects", []) or []:
            if str(obj.get("dataset_source", "")).lower() != "physxnet":
                continue
            source_object_id = obj.get("source_object_id")
            if source_object_id is None:
                continue
            grouped[str(source_object_id)].add(sample_dir)
    return {key: sorted(value) for key, value in grouped.items()}


def resolve_object_asset_dirs(benchmark_root: Path, benchmark_type: str) -> List[Path]:
    dirs: List[Path] = []
    if benchmark_type == "stage1_heldout":
        cache_root = benchmark_root / "_asset_cache" / "physxnet_objects"
        if cache_root.exists():
            dirs.extend(path for path in sorted(cache_root.iterdir()) if (path / "meta" / "metadata.json").exists())
    elif benchmark_type == "physxnet_pool":
        manifest_path = benchmark_root / "benchmark_manifest.json"
        if manifest_path.exists():
            manifest = _load_json(manifest_path)
            work_root = Path(str(manifest.get("work_root", "")))
            cache_root = work_root / "_asset_cache" / "physxnet_objects"
            if cache_root.exists():
                dirs.extend(path for path in sorted(cache_root.iterdir()) if (path / "meta" / "metadata.json").exists())
            elif work_root.exists():
                dirs.extend(path for path in sorted(work_root.iterdir()) if (path / "meta" / "metadata.json").exists())
    else:
        cache_root = benchmark_root / "_asset_cache" / "physxnet_objects"
        if cache_root.exists():
            dirs.extend(path for path in sorted(cache_root.iterdir()) if (path / "meta" / "metadata.json").exists())
    return dirs


def _parse_xyz(text: str) -> np.ndarray:
    parts = [float(tok) for tok in str(text).strip().split()]
    if len(parts) != 3:
        raise ValueError(f"Expected xyz triplet, got: {text!r}")
    return np.asarray(parts, dtype=np.float64)


def _load_mesh_bounds(mesh_path: Path, cache: Dict[Path, Optional[Tuple[np.ndarray, np.ndarray]]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    cached = cache.get(mesh_path)
    if cached is not None or mesh_path in cache:
        return cached

    try:
        mesh = trimesh.load(mesh_path, force="mesh")
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            cache[mesh_path] = None
        else:
            cache[mesh_path] = (bounds[0], bounds[1])
    except Exception:
        cache[mesh_path] = None
    return cache[mesh_path]


def _combined_bounds(mesh_paths: Sequence[Path], cache: Dict[Path, Optional[Tuple[np.ndarray, np.ndarray]]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str], List[str]]:
    mins: List[np.ndarray] = []
    maxs: List[np.ndarray] = []
    missing: List[str] = []
    unloadable: List[str] = []
    for mesh_path in mesh_paths:
        if not mesh_path.exists():
            missing.append(str(mesh_path))
            continue
        bounds = _load_mesh_bounds(mesh_path, cache)
        if bounds is None:
            unloadable.append(str(mesh_path))
            continue
        mins.append(bounds[0])
        maxs.append(bounds[1])
    if not mins:
        return None, None, missing, unloadable
    return np.min(np.stack(mins, axis=0), axis=0), np.max(np.stack(maxs, axis=0), axis=0), missing, unloadable


def audit_object_asset(object_dir: Path, *, benchmark_root: Path, tol: float = 1e-5) -> Dict:
    metadata_path = object_dir / "meta" / "metadata.json"
    metadata = _load_json(metadata_path)
    object_id = str(metadata.get("object_id", object_dir.name))
    urdf_path = object_dir / "rigid" / f"{object_id}.urdf"

    result = {
        "benchmark_root": str(benchmark_root),
        "object_id": object_id,
        "object_dir": str(object_dir),
        "metadata_path": str(metadata_path),
        "urdf_path": str(urdf_path),
        "inertial_origin_policy_version": metadata.get("inertial_origin_policy_version"),
        "policy_matches_current": metadata.get("inertial_origin_policy_version") == CURRENT_INERTIAL_POLICY_VERSION,
        "issues": [],
        "links": [],
    }

    if not urdf_path.exists():
        result["issues"].append("missing_urdf")
        return result

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    bounds_cache: Dict[Path, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
    rigid_dir = urdf_path.parent

    for link in root.findall("link"):
        link_name = str(link.attrib.get("name", ""))
        inertial = link.find("inertial")
        if inertial is None:
            continue
        origin_node = inertial.find("origin")
        if origin_node is None:
            continue

        xyz_text = origin_node.attrib.get("xyz", "0 0 0")
        try:
            inertial_xyz = _parse_xyz(xyz_text)
        except Exception:
            result["issues"].append("bad_inertial_xyz")
            result["links"].append(
                {
                    "link_name": link_name,
                    "status": "bad_inertial_xyz",
                    "inertial_xyz": xyz_text,
                }
            )
            continue

        mesh_paths: List[Path] = []
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            mesh = geometry.find("mesh") if geometry is not None else None
            if mesh is None:
                continue
            filename = mesh.attrib.get("filename")
            if filename:
                mesh_paths.append((rigid_dir / filename).resolve())

        if not mesh_paths:
            result["links"].append(
                {
                    "link_name": link_name,
                    "status": "no_collision_mesh_ref",
                    "inertial_xyz": inertial_xyz.tolist(),
                }
            )
            continue

        bounds_min, bounds_max, missing_meshes, unloadable_meshes = _combined_bounds(mesh_paths, bounds_cache)
        status = "ok"
        issue_payload: Dict[str, object] = {
            "link_name": link_name,
            "inertial_xyz": inertial_xyz.tolist(),
            "mesh_paths": [str(path) for path in mesh_paths],
        }
        if not np.isfinite(inertial_xyz).all():
            status = "nonfinite_inertial_xyz"
            result["issues"].append(status)
        elif missing_meshes:
            status = "missing_collision_mesh"
            issue_payload["missing_meshes"] = missing_meshes
            result["issues"].append(status)
        elif unloadable_meshes:
            status = "unloadable_collision_mesh"
            issue_payload["unloadable_meshes"] = unloadable_meshes
            result["issues"].append(status)
        elif bounds_min is None or bounds_max is None:
            status = "missing_mesh_bounds"
            result["issues"].append(status)
        else:
            issue_payload["bounds_min"] = bounds_min.tolist()
            issue_payload["bounds_max"] = bounds_max.tolist()
            inside = np.all(inertial_xyz >= (bounds_min - tol)) and np.all(inertial_xyz <= (bounds_max + tol))
            if not inside:
                status = "inertial_outside_collision_aabb"
                delta_lo = np.maximum(0.0, bounds_min - inertial_xyz)
                delta_hi = np.maximum(0.0, inertial_xyz - bounds_max)
                issue_payload["distance_outside_aabb"] = (delta_lo + delta_hi).tolist()
                result["issues"].append(status)

        issue_payload["status"] = status
        result["links"].append(issue_payload)

    result["issues"] = sorted(set(result["issues"]))
    result["flagged"] = (not result["policy_matches_current"]) or bool(result["issues"])
    return result


def build_report(benchmark_root: Path, *, tol: float = 1e-5) -> Dict:
    benchmark_roots = detect_benchmark_roots(benchmark_root)
    report_roots: List[Dict] = []

    for root in benchmark_roots:
        benchmark_type = detect_benchmark_type(root)
        sample_dirs_by_object = collect_sample_dirs_by_object(root)
        object_reports: List[Dict] = []
        for object_dir in resolve_object_asset_dirs(root, benchmark_type):
            object_report = audit_object_asset(object_dir=object_dir, benchmark_root=root, tol=tol)
            object_report["sample_dirs"] = sample_dirs_by_object.get(object_report["object_id"], [])
            object_reports.append(object_report)

        flagged = [item for item in object_reports if bool(item.get("flagged", False))]
        report_roots.append(
            {
                "benchmark_root": str(root),
                "benchmark_type": benchmark_type,
                "sample_count": int(sum(len(paths) for paths in sample_dirs_by_object.values())),
                "sample_object_count": int(len(sample_dirs_by_object)),
                "asset_object_count": int(len(object_reports)),
                "flagged_object_count": int(len(flagged)),
                "flagged_object_ids": [str(item["object_id"]) for item in flagged],
                "objects": object_reports,
            }
        )

    total_assets = sum(int(item["asset_object_count"]) for item in report_roots)
    total_flagged = sum(int(item["flagged_object_count"]) for item in report_roots)
    return {
        "benchmark_root": str(benchmark_root),
        "current_inertial_policy_version": CURRENT_INERTIAL_POLICY_VERSION,
        "benchmark_root_count": int(len(report_roots)),
        "total_asset_objects": int(total_assets),
        "total_flagged_objects": int(total_flagged),
        "benchmark_roots": report_roots,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit benchmark URDF inertial origins against collision-mesh AABBs.")
    parser.add_argument("--benchmark_root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--tol", type=float, default=1e-5, help="AABB containment tolerance in meters.")
    args = parser.parse_args()

    report = build_report(Path(args.benchmark_root), tol=float(args.tol))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"benchmark_roots={report['benchmark_root_count']} "
        f"asset_objects={report['total_asset_objects']} "
        f"flagged={report['total_flagged_objects']}"
    )
    for root_report in report["benchmark_roots"]:
        print(
            f"[{root_report['benchmark_type']}] root={root_report['benchmark_root']} "
            f"assets={root_report['asset_object_count']} flagged={root_report['flagged_object_count']}"
        )
        for object_report in root_report["objects"]:
            if not object_report.get("flagged", False):
                continue
            sample_count = len(object_report.get("sample_dirs", []))
            print(
                f"  object_id={object_report['object_id']} "
                f"policy={object_report.get('inertial_origin_policy_version')} "
                f"issues={object_report.get('issues', [])} "
                f"samples={sample_count}"
            )


if __name__ == "__main__":
    main()
