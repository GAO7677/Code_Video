#!/usr/bin/env python3
from __future__ import annotations

import html
import json
from pathlib import Path
from collections import Counter, defaultdict


DATA_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid")
PORTAL_ROOT = Path("/home/gaoya/portal_hub_sim/count01_live_portal")


def collect_samples() -> tuple[list[dict], list[dict]]:
    valid_root = DATA_ROOT / "single_object_preview" / "count_01"
    invalid_root = DATA_ROOT / "single_object_preview" / "invalid_by_qa" / "count_01"
    valid: list[dict] = []
    invalid: list[dict] = []

    for base, bucket_name, target in [
        (valid_root, "valid", valid),
        (invalid_root, "invalid_by_qa", invalid),
    ]:
        if not base.exists():
            continue
        for sample_dir in sorted(base.glob("*__rs0*")):
            if not sample_dir.is_dir():
                continue
            parts = sample_dir.name.split("__")
            if len(parts) < 3:
                continue
            object_id, case_name, rs_tag = parts[0], parts[1], parts[2]
            target.append(
                {
                    "sample_name": sample_dir.name,
                    "object_id": object_id,
                    "case_name": case_name,
                    "rs_tag": rs_tag,
                    "bucket": bucket_name,
                    "sample_dir": str(sample_dir),
                    "rgb_video": str(sample_dir / "videos" / "rgb.mp4"),
                    "depth_video": str(sample_dir / "videos" / "depth.mp4"),
                    "meta": str(sample_dir / "meta.json"),
                }
            )
    return valid, invalid


def build_html(valid: list[dict], invalid: list[dict]) -> str:
    valid_case_counts = Counter(item["case_name"] for item in valid)
    valid_rs_counts = Counter(item["rs_tag"] for item in valid)
    invalid_case_counts = Counter(item["case_name"] for item in invalid)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in valid:
        grouped[item["rs_tag"]].append(item)

    sections: list[str] = []
    for rs_tag in sorted(grouped):
        cards: list[str] = []
        for item in grouped[rs_tag]:
            rgb_src = html.escape(item["rgb_video"])
            depth_src = html.escape(item["depth_video"])
            sample_name = html.escape(item["sample_name"])
            sample_dir = html.escape(item["sample_dir"])
            case_name = html.escape(item["case_name"])
            cards.append(
                f"""
                <div class="card">
                  <div class="meta">
                    <div><strong>{sample_name}</strong></div>
                    <div>{case_name} | {html.escape(item["rs_tag"])}</div>
                    <div class="path">{sample_dir}</div>
                  </div>
                  <div class="videos">
                    <div>
                      <div class="label">RGB</div>
                      <video src="{rgb_src}" controls preload="metadata"></video>
                    </div>
                    <div>
                      <div class="label">Depth</div>
                      <video src="{depth_src}" controls preload="metadata"></video>
                    </div>
                  </div>
                </div>
                """
            )
        sections.append(
            f"""
            <section>
              <h2>{html.escape(rs_tag)} ({len(grouped[rs_tag])})</h2>
              <div class="grid">
                {''.join(cards)}
              </div>
            </section>
            """
        )

    summary = {
        "valid_total": len(valid),
        "invalid_total": len(invalid),
        "valid_case_counts": dict(valid_case_counts),
        "valid_rs_counts": dict(valid_rs_counts),
        "invalid_case_counts": dict(invalid_case_counts),
    }
    (PORTAL_ROOT / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>count01 live portal</title>
  <style>
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f6f8;
      color: #111;
    }}
    .wrap {{
      max-width: 1800px;
      margin: 0 auto;
      padding: 20px;
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(260px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .panel, .card {{
      background: #fff;
      border: 1px solid #d8dde6;
      border-radius: 10px;
      padding: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(540px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .videos {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 10px;
    }}
    video {{
      width: 100%;
      background: #000;
      border-radius: 8px;
    }}
    .meta {{
      font-size: 14px;
      line-height: 1.45;
    }}
    .path {{
      color: #555;
      word-break: break-all;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      margin-top: 4px;
    }}
    .label {{
      font-size: 12px;
      color: #555;
      margin-bottom: 4px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Count01 Live Portal</h1>
    <div class="stats">
      <div class="panel">
        <h2>有效样本</h2>
        <pre>{html.escape(json.dumps(dict(valid_case_counts), ensure_ascii=False, indent=2))}</pre>
      </div>
      <div class="panel">
        <h2>按 Resample 统计</h2>
        <pre>{html.escape(json.dumps(dict(valid_rs_counts), ensure_ascii=False, indent=2))}</pre>
      </div>
      <div class="panel">
        <h2>Invalid By QA</h2>
        <pre>{html.escape(json.dumps(dict(invalid_case_counts), ensure_ascii=False, indent=2))}</pre>
      </div>
      <div class="panel">
        <h2>总数</h2>
        <pre>{html.escape(json.dumps({"valid_total": len(valid), "invalid_total": len(invalid)}, ensure_ascii=False, indent=2))}</pre>
      </div>
    </div>
    {''.join(sections)}
  </div>
</body>
</html>
"""


def main() -> None:
    PORTAL_ROOT.mkdir(parents=True, exist_ok=True)
    valid, invalid = collect_samples()
    (PORTAL_ROOT / "index.html").write_text(build_html(valid, invalid), encoding="utf-8")
    print(PORTAL_ROOT / "index.html")


if __name__ == "__main__":
    main()
