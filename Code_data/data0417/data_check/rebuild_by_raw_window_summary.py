#!/usr/bin/env python3
"""Rebuild data_summary/by_raw_window from current sample metadata only.

重建规则：
- 不读取旧类别文件名，只把旧的 `all_sample_dirs.json/txt` 当作样本全集来源
- `raw/window` 由样本当前 `meta.json` / `metadata.json` 的 `view_type` 推断
- `train/test/benchmark` 先按真实路径组织推断，再按 metadata 兜底
- 复杂度叶子类别 = `物体数量分组 + 简化碰撞类型`
- 对 window 样本，若自身缺少 `collision_type_bucket`，优先回看 `source_sample_dir` 的 raw meta
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_ROOT = Path("/home/gaoya/Code_Video/Code_data/data0417/data_summary/by_raw_window")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_txt(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def find_meta_path(sample_dir: Path) -> Path | None:
    for name in ("meta.json", "metadata.json"):
        path = sample_dir / name
        if path.exists():
            return path
    return None


def load_meta(sample_dir: Path) -> dict[str, Any]:
    meta_path = find_meta_path(sample_dir)
    if meta_path is None:
        return {}
    data = load_json(meta_path)
    return data if isinstance(data, dict) else {}


def load_source_meta(sample_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    source_paths = meta.get("source_paths", {}) if isinstance(meta.get("source_paths"), dict) else {}
    source_dir = str(meta.get("source_sample_dir") or source_paths.get("source_sample_dir") or "").strip()
    if not source_dir:
        return {}
    source_path = Path(source_dir)
    if not source_path.exists() or source_path == sample_dir:
        return {}
    return load_meta(source_path)


def read_all_sample_dirs(summary_root: Path) -> list[Path]:
    all_json = summary_root / "all_sample_dirs.json"
    if all_json.exists():
        data = load_json(all_json)
        if isinstance(data, list):
            return sorted({Path(str(item)) for item in data})

    sample_dirs: set[Path] = set()
    for path in summary_root.rglob("*.json"):
        if path.name in {"summary.json", "all_sample_dirs.json"}:
            continue
        data = load_json(path)
        if not isinstance(data, list):
            continue
        for item in data:
            sample_dirs.add(Path(str(item)))
    return sorted(sample_dirs)


def infer_view_type(sample_dir: Path, meta: dict[str, Any], source_meta: dict[str, Any]) -> str:
    for payload in (meta, source_meta):
        value = str(payload.get("view_type") or "").strip().lower()
        if value in {"raw", "window"}:
            return value
    paths = meta.get("paths", {}) if isinstance(meta.get("paths"), dict) else {}
    if any(key in paths for key in ("context_video_path", "future_gt_video_path", "full_video_path")):
        return "window"
    if (sample_dir / "videos" / "rgb.mp4").exists() or (sample_dir / "rgb.mp4").exists():
        return "raw"
    if (sample_dir / "context_video.mp4").exists() or (sample_dir / "full_video.mp4").exists():
        return "window"
    return "raw"


def infer_split_parts(sample_dir: Path, meta: dict[str, Any], source_meta: dict[str, Any], view_type: str) -> tuple[str, ...]:
    parts = sample_dir.parts
    if "stage1adapter" in parts:
        idx = parts.index("stage1adapter")
        if idx + 1 < len(parts):
            head = parts[idx + 1]
            if head == "benchmark" and idx + 2 < len(parts):
                return ("benchmark", parts[idx + 2])
            if head in {"train", "test"}:
                return (head,)

    split = str(meta.get("split") or source_meta.get("split") or "").strip().lower()
    if split == "train":
        return ("train",)
    if split == "test":
        return ("test",)
    if split in {"heldout", "train_eval"}:
        return ("benchmark", split)

    return ("train",) if view_type == "raw" else ("test",)


def infer_num_objects(meta: dict[str, Any], source_meta: dict[str, Any]) -> int | None:
    for payload in (meta, source_meta):
        value = payload.get("num_objects")
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
        objects = payload.get("objects")
        if isinstance(objects, list) and objects:
            return len(objects)
    return None


def simplify_count_bucket(meta: dict[str, Any], source_meta: dict[str, Any]) -> str:
    raw_bucket = ""
    for payload in (meta, source_meta):
        raw_bucket = str(payload.get("object_count_bucket") or "").strip().lower()
        if raw_bucket and raw_bucket != "unknown":
            break
    if raw_bucket:
        if raw_bucket in {"count_01", "count_1"}:
            return "single_1"
        if raw_bucket in {"count_02", "count_2"}:
            return "pair_2"
        if raw_bucket in {"count_03", "count_04", "count_03_04", "count_3", "count_4"}:
            return "few_3_4"
        if raw_bucket.startswith("count_"):
            digits = "".join(ch for ch in raw_bucket[6:] if ch.isdigit())
            if digits:
                try:
                    value = int(digits)
                except ValueError:
                    value = -1
                if value >= 5:
                    return "many_5plus"

    num_objects = infer_num_objects(meta, source_meta)
    if num_objects == 1:
        return "single_1"
    if num_objects == 2:
        return "pair_2"
    if num_objects in {3, 4}:
        return "few_3_4"
    if num_objects is not None and num_objects >= 5:
        return "many_5plus"
    return "unknown"


def raw_collision_bucket(meta: dict[str, Any], source_meta: dict[str, Any]) -> str:
    for payload in (meta, source_meta):
        value = payload.get("collision_type_bucket")
        if value is not None:
            text = str(value).strip().lower()
            if text and text != "unknown" and text != "none":
                return text
            if text == "none":
                return "none"

    for payload in (meta, source_meta):
        obj_obj = payload.get("obj_obj_event_count")
        obj_env = payload.get("obj_env_event_count")
        if obj_obj is None and obj_env is None:
            continue
        try:
            obj_obj_i = int(obj_obj or 0)
            obj_env_i = int(obj_env or 0)
        except Exception:
            continue
        if obj_obj_i <= 0 and obj_env_i <= 0:
            return "none"
        if obj_obj_i <= 0:
            return "env_only"
        if obj_env_i <= 0:
            return "obj_obj_only"
        return "mixed"

    return "unknown"


def simplify_collision_bucket(meta: dict[str, Any], source_meta: dict[str, Any]) -> str:
    raw_bucket = raw_collision_bucket(meta, source_meta)
    if raw_bucket == "none":
        return "none"
    if raw_bucket == "env_only":
        return "env_only"
    if raw_bucket in {"mixed", "obj_obj_only"}:
        return "collision"
    return "unknown"


def classify_leaf(meta: dict[str, Any], source_meta: dict[str, Any]) -> str:
    count_bucket = simplify_count_bucket(meta, source_meta)
    collision_bucket = simplify_collision_bucket(meta, source_meta)
    if count_bucket != "single_1" and collision_bucket == "env_only":
        collision_bucket = "collision"
    return f"{count_bucket}_{collision_bucket}"


def build_readme(
    total_samples: int,
    by_view: dict[str, int],
    by_split: dict[str, int],
    by_leaf: dict[str, int],
) -> str:
    lines = [
        "# by_raw_window",
        "",
        "这个目录记录按 `raw/window -> train/test/benchmark -> 复杂度叶子类别` 重建后的样本路径清单。",
        "",
        "分类依据：",
        "- 第一层按当前 metadata 的 `view_type` 划分：`raw` / `window`。",
        "- 第二层优先按真实路径组织划分：`train` / `test` / `benchmark`。",
        "- 第三层按 `物体数量分组 + 简化碰撞类型` 划分。",
        "- `物体数量分组` 统一映射为：`single_1` / `pair_2` / `few_3_4` / `many_5plus` / `unknown`。",
        "- `简化碰撞类型` 统一映射为：`none` / `env_only` / `collision` / `unknown`。",
        "- 其中 `pair_2`、`few_3_4`、`many_5plus` 不再区分 `obj-obj` 与 `obj-env`，所有已知碰撞统一并到 `*_collision`。",
        "- `single_1` 仍保留 `single_1_env_only`，因为单物体与环境接触是主要碰撞形式。",
        "- 对 window 样本，如果自身缺少 `collision_type_bucket`，会优先回看 `source_sample_dir` 对应 raw 样本的 metadata。",
        "- 本次重建不复用旧类别文件名中的历史判定，只使用当前 `meta.json` / `metadata.json` 和 source raw metadata。",
        "",
        f"总样本数：`{total_samples}`",
        "",
        "按视图统计：",
    ]
    for key in sorted(by_view):
        lines.append(f"- `{key}`：{by_view[key]}")
    lines.extend(["", "按目录统计："])
    for key in sorted(by_split):
        lines.append(f"- `{key}`：{by_split[key]}")
    lines.extend(["", "按叶子类别统计："])
    for key in sorted(by_leaf):
        lines.append(f"- `{key}`：{by_leaf[key]}")
    lines.extend(["", "说明：", "- 每个 `json/txt` 文件都只保存样本文件夹绝对路径。", "- `_all_samples` 表示该目录下的全量路径合集。"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild data_summary/by_raw_window from current metadata")
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    args = parser.parse_args()

    summary_root = args.summary_root.resolve()
    sample_dirs = read_all_sample_dirs(summary_root)
    if not sample_dirs:
        raise RuntimeError(f"no sample dirs found under {summary_root}")

    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    all_paths: list[str] = []
    by_view = Counter()
    by_split = Counter()
    by_leaf = Counter()
    by_dataset = Counter()

    for sample_dir in sample_dirs:
        meta = load_meta(sample_dir)
        source_meta = load_source_meta(sample_dir, meta)
        view_type = infer_view_type(sample_dir, meta, source_meta)
        split_parts = infer_split_parts(sample_dir, meta, source_meta, view_type)
        leaf = classify_leaf(meta, source_meta)
        rel_group = (view_type, *split_parts, leaf)
        rel_all = (view_type, *split_parts, "_all_samples")

        sample_str = str(sample_dir)
        groups[rel_group].append(sample_str)
        groups[rel_all].append(sample_str)
        all_paths.append(sample_str)

        by_view[view_type] += 1
        by_split["/".join((view_type, *split_parts))] += 1
        by_leaf["/".join((view_type, *split_parts, leaf))] += 1
        by_dataset[str(meta.get("dataset") or source_meta.get("dataset") or "Unknown")] += 1

    if (summary_root / "raw").exists():
        shutil.rmtree(summary_root / "raw")
    if (summary_root / "window").exists():
        shutil.rmtree(summary_root / "window")

    all_paths = sorted(set(all_paths))
    write_json(summary_root / "all_sample_dirs.json", all_paths)
    write_txt(summary_root / "all_sample_dirs.txt", all_paths)

    for rel_parts, entries in sorted(groups.items()):
        dst = summary_root.joinpath(*rel_parts)
        ordered = sorted(set(entries))
        write_json(dst.with_suffix(".json"), ordered)
        write_txt(dst.with_suffix(".txt"), ordered)

    summary = {
        "total_samples": len(all_paths),
        "by_view": dict(sorted(by_view.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_leaf": dict(sorted(by_leaf.items())),
        "by_dataset": dict(sorted(by_dataset.items())),
    }
    write_json(summary_root / "summary.json", summary)
    (summary_root / "README.md").write_text(
        build_readme(
            total_samples=len(all_paths),
            by_view=dict(sorted(by_view.items())),
            by_split=dict(sorted(by_split.items())),
            by_leaf=dict(sorted(by_leaf.items())),
        ),
        encoding="utf-8",
    )

    print(summary_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
