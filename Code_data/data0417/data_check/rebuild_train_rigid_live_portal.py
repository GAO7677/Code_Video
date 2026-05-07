#!/usr/bin/env python3
from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATA_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid")
PORTAL_ROOT = Path("/home/gaoya/portal_hub_sim/train_rigid_live_portal")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_sample_dirs(root: Path) -> list[Path]:
    sample_dirs: set[Path] = set()
    for meta_name in ("meta.json", "metadata.json"):
        for meta_path in root.rglob(meta_name):
            sample_dirs.add(meta_path.parent)
    return sorted(sample_dirs)


def sample_record(sample_dir: Path) -> dict[str, Any] | None:
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        meta_path = sample_dir / "metadata.json"
    if not meta_path.exists():
        return None

    rgb_video = sample_dir / "videos" / "rgb.mp4"
    depth_video = sample_dir / "videos" / "depth.mp4"
    if not rgb_video.exists():
        return None

    try:
        meta = load_json(meta_path)
    except Exception:
        meta = {}

    rel_parts = sample_dir.relative_to(DATA_ROOT).parts
    scene_composition = rel_parts[0] if len(rel_parts) >= 1 else "unknown"
    count_bucket = rel_parts[1] if len(rel_parts) >= 2 else "unknown"
    case_name = str(meta.get("case_name") or "")
    if not case_name:
        parts = sample_dir.name.split("__")
        case_name = parts[1] if len(parts) >= 2 else "unknown_case"

    return {
        "sample_name": sample_dir.name,
        "sample_dir": str(sample_dir),
        "scene_composition": scene_composition,
        "count_bucket": count_bucket,
        "case_name": case_name,
        "caption": str(meta.get("caption") or ""),
        "detail_caption": str(meta.get("detail_caption") or ""),
        "rgb_video": str(rgb_video),
        "depth_video": str(depth_video) if depth_video.exists() else "",
        "meta_path": str(meta_path),
        "has_qa_metrics": (sample_dir / "qa_metrics.json").exists(),
        "has_pair_meta": (sample_dir / "pair_meta.json").exists(),
    }


def build_group_pages(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["scene_composition"]),
            str(record["count_bucket"]),
            str(record["case_name"]),
        )
        grouped[key].append(record)

    groups_out: list[dict[str, Any]] = []
    groups_root = PORTAL_ROOT / "groups"
    groups_root.mkdir(parents=True, exist_ok=True)

    for scene_composition, count_bucket, case_name in sorted(grouped):
        samples = sorted(grouped[(scene_composition, count_bucket, case_name)], key=lambda x: x["sample_name"])
        slug = f"{scene_composition}__{count_bucket}__{case_name}".replace("/", "_")
        page_path = groups_root / f"{slug}.html"

        cards: list[str] = []
        for item in samples:
            rgb_src = html.escape(item["rgb_video"])
            depth_src = html.escape(item["depth_video"])
            caption = html.escape(item["caption"])
            detail_caption = html.escape(item["detail_caption"])
            cards.append(
                f"""
                <div class="card">
                  <div class="title">{html.escape(item["sample_name"])}</div>
                  <div class="path">{html.escape(item["sample_dir"])}</div>
                  <div class="meta">
                    <div>scene={html.escape(item["scene_composition"])}</div>
                    <div>count={html.escape(item["count_bucket"])}</div>
                    <div>case={html.escape(item["case_name"])}</div>
                    <div>qa_metrics={'yes' if item['has_qa_metrics'] else 'no'} | pair_meta={'yes' if item['has_pair_meta'] else 'no'}</div>
                  </div>
                  <div class="caption"><strong>Caption:</strong> {caption or '(empty)'}</div>
                  <div class="caption"><strong>Detail:</strong> {detail_caption or '(empty)'}</div>
                  <div class="videos">
                    <div>
                      <div class="label">RGB</div>
                      <video src="{rgb_src}" controls preload="metadata"></video>
                    </div>
                    <div>
                      <div class="label">Depth</div>
                      {f'<video src="{depth_src}" controls preload="metadata"></video>' if depth_src else '<div class="missing">missing</div>'}
                    </div>
                  </div>
                </div>
                """
            )

        page_path.write_text(
            f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(case_name)}</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f6f8; color: #111; }}
    .wrap {{ max-width: 1800px; margin: 0 auto; padding: 18px; }}
    .back {{ margin-bottom: 14px; }}
    .summary {{ background: #fff; border: 1px solid #d8dde6; border-radius: 10px; padding: 12px; margin-bottom: 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(560px, 1fr)); gap: 12px; }}
    .card {{ background: #fff; border: 1px solid #d8dde6; border-radius: 10px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
    .title {{ font-weight: 700; margin-bottom: 6px; }}
    .path {{ color: #555; font-size: 12px; word-break: break-all; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .meta {{ font-size: 13px; line-height: 1.5; margin-top: 8px; }}
    .caption {{ font-size: 13px; line-height: 1.5; margin-top: 8px; }}
    .videos {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }}
    .label {{ font-size: 12px; color: #555; margin-bottom: 4px; }}
    video {{ width: 100%; background: #000; border-radius: 8px; }}
    .missing {{ height: 200px; display: flex; align-items: center; justify-content: center; background: #eceff4; border-radius: 8px; color: #666; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="back"><a href="../index.html">Back to train/rigid portal</a></div>
    <div class="summary">
      <h1>{html.escape(case_name)}</h1>
      <div>scene={html.escape(scene_composition)} | count={html.escape(count_bucket)} | samples={len(samples)}</div>
    </div>
    <div class="grid">
      {''.join(cards)}
    </div>
  </div>
</body>
</html>
""",
            encoding="utf-8",
        )

        groups_out.append(
            {
                "scene_composition": scene_composition,
                "count_bucket": count_bucket,
                "case_name": case_name,
                "count": len(samples),
                "page": f"groups/{slug}.html",
            }
        )

    return groups_out


def build_index(records: list[dict[str, Any]], groups: list[dict[str, Any]]) -> str:
    scene_counts = Counter(item["scene_composition"] for item in records)
    count_counts = Counter(item["count_bucket"] for item in records)
    case_counts = Counter(item["case_name"] for item in records)

    grouped_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        grouped_by_scene[group["scene_composition"]].append(group)

    sections: list[str] = []
    for scene_name in sorted(grouped_by_scene):
        rows = []
        for group in sorted(grouped_by_scene[scene_name], key=lambda x: (x["count_bucket"], x["case_name"])):
            rows.append(
                f"""
                <tr>
                  <td>{html.escape(group['count_bucket'])}</td>
                  <td>{html.escape(group['case_name'])}</td>
                  <td>{group['count']}</td>
                  <td><a href="{html.escape(group['page'])}">Open</a></td>
                </tr>
                """
            )
        sections.append(
            f"""
            <section class="panel">
              <h2>{html.escape(scene_name)}</h2>
              <table>
                <thead><tr><th>count</th><th>case</th><th>samples</th><th>page</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
              </table>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>train/rigid live portal</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f6f8; color: #111; }}
    .wrap {{ max-width: 1680px; margin: 0 auto; padding: 18px; }}
    .stats {{ display: grid; grid-template-columns: repeat(3, minmax(260px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .panel {{ background: #fff; border: 1px solid #d8dde6; border-radius: 10px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 14px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; border-bottom: 1px solid #e5e8ef; padding: 8px 6px; font-size: 14px; }}
    a {{ color: #1155cc; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>train/rigid Full Live Portal</h1>
    <div class="stats">
      <div class="panel">
        <h2>总样本数</h2>
        <pre>{html.escape(json.dumps({"total_samples": len(records), "groups": len(groups)}, ensure_ascii=False, indent=2))}</pre>
      </div>
      <div class="panel">
        <h2>按 scene 统计</h2>
        <pre>{html.escape(json.dumps(dict(scene_counts), ensure_ascii=False, indent=2))}</pre>
      </div>
      <div class="panel">
        <h2>按 count 统计</h2>
        <pre>{html.escape(json.dumps(dict(count_counts), ensure_ascii=False, indent=2))}</pre>
      </div>
    </div>
    <div class="panel">
      <h2>按 case 统计</h2>
      <pre>{html.escape(json.dumps(dict(case_counts.most_common()), ensure_ascii=False, indent=2))}</pre>
    </div>
    {''.join(sections)}
  </div>
</body>
</html>
"""


def main() -> None:
    PORTAL_ROOT.mkdir(parents=True, exist_ok=True)
    records = [rec for rec in (sample_record(sample_dir) for sample_dir in iter_sample_dirs(DATA_ROOT)) if rec is not None]
    groups = build_group_pages(records)
    summary = {
        "total_samples": len(records),
        "total_groups": len(groups),
    }
    (PORTAL_ROOT / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (PORTAL_ROOT / "index.html").write_text(build_index(records, groups), encoding="utf-8")
    print(PORTAL_ROOT / "index.html")


if __name__ == "__main__":
    main()
