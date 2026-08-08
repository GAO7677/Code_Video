#!/usr/bin/env python3
"""Build a compact dashboard for original and controlled ball-block videos."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path


ORIGINAL_DIR = Path(
    "/home/gaoya/data/AAA_test_video/Dataset_physV/0526dp/videos/ball_block"
)
MOTION_DIR = Path(
    "/home/gaoya/data/agent-data/datasets/physv-ball-block-motion-controlled"
)
DASHBOARD_DIR = Path(
    "/home/gaoya/data/agent-data/outputs/physv-ball-block-motion-controlled-dashboard"
)
BASELINE = "e07_mu05_m1"
MOTION_ORDER = (
    "motion_speed_050x",
    "motion_speed_075x",
    "motion_speed_125x",
    "motion_speed_150x",
    "motion_direction_yaw_m10",
    "motion_direction_yaw_p10",
    "motion_distance_050x",
    "motion_distance_075x",
    "motion_distance_125x",
    "motion_distance_150x",
)
ORIGINAL_ORDER = (
    "e03_mu05_m1",
    "e05_mu05_m1",
    "e09_mu05_m1",
    "e07_mu01_m1",
    "e07_mu10_m1",
    "e07_mu05_m01",
    "e07_mu05_m5",
)


def load_case(root: Path, name: str) -> dict[str, object]:
    return json.loads((root / f"{name}.json").read_text(encoding="utf-8"))


def speed_and_yaw(velocity: list[float]) -> tuple[float, float]:
    speed = math.sqrt(sum(value * value for value in velocity))
    yaw = math.degrees(math.atan2(velocity[1], velocity[0]))
    return speed, yaw


def render_card(
    metadata: dict[str, object],
    source_dir: str,
    changed_variable: str,
) -> str:
    name = str(metadata["scenario"])
    parameters = metadata["parameters"]
    conditions = metadata["initial_conditions"]
    velocity = conditions["ball_velocity_ms"]
    ball_start = conditions["ball_start_xyz"]
    block_start = conditions["block_start_xyz"]
    center_distance_x = float(block_start[0]) - float(ball_start[0])
    speed, yaw = speed_and_yaw(velocity)
    vector = ", ".join(f"{value:.3f}" for value in velocity)
    return f"""
    <article class="card">
      <div class="heading">
        <h3>{html.escape(name)}</h3>
        <span>{html.escape(changed_variable)}</span>
      </div>
      <video controls muted loop preload="metadata" src="{source_dir}/{html.escape(name)}.mp4"></video>
      <dl>
        <div><dt>e</dt><dd>{float(parameters['restitution']):g}</dd></div>
        <div><dt>μ</dt><dd>{float(parameters['lateral_friction']):g}</dd></div>
        <div><dt>ball mass</dt><dd>{float(parameters['ball_mass_kg']):g} kg</dd></div>
        <div><dt>ball start x</dt><dd>{float(ball_start[0]):.3f} m</dd></div>
        <div><dt>x distance</dt><dd>{center_distance_x:.3f} m</dd></div>
        <div><dt>velocity</dt><dd>[{vector}] m/s</dd></div>
        <div><dt>|v|</dt><dd>{speed:.3f} m/s</dd></div>
        <div><dt>yaw</dt><dd>{yaw:+.1f}°</dd></div>
      </dl>
      <a href="{source_dir}/{html.escape(name)}.json">JSON metadata</a>
    </article>
    """


def ensure_source_link(name: str, target: Path) -> None:
    link = DASHBOARD_DIR / name
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Unexpected existing dashboard path: {link}")
    link.symlink_to(target, target_is_directory=True)


def main() -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    ensure_source_link("original", ORIGINAL_DIR)
    ensure_source_link("motion", MOTION_DIR)

    baseline = load_case(ORIGINAL_DIR, BASELINE)
    motion_manifest = json.loads((MOTION_DIR / "manifest.json").read_text(encoding="utf-8"))
    changed_by_name = {
        item["scenario"]: item["changed_variable"] for item in motion_manifest["cases"]
    }
    controlled_cards = [render_card(baseline, "original", "baseline")]
    controlled_cards.extend(
        render_card(load_case(MOTION_DIR, name), "motion", changed_by_name[name])
        for name in MOTION_ORDER
    )

    changed_original = {
        "e03_mu05_m1": "restitution",
        "e05_mu05_m1": "restitution",
        "e09_mu05_m1": "restitution",
        "e07_mu01_m1": "friction",
        "e07_mu10_m1": "friction",
        "e07_mu05_m01": "ball mass",
        "e07_mu05_m5": "ball mass",
    }
    original_cards = [
        render_card(load_case(ORIGINAL_DIR, name), "original", changed_original[name])
        for name in ORIGINAL_ORDER
    ]

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ball-block 单变量仿真对比</title>
  <style>
    :root {{ --ink:#18211b; --muted:#657068; --line:#ccd5ce; --canvas:#f3f5f1; --surface:#fff; --accent:#087f5b; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--canvas); font-family:Arial,"Noto Sans SC",sans-serif; line-height:1.4; }}
    header {{ padding:30px max(24px,calc((100vw - 1480px)/2)); color:#fff; background:var(--ink); border-bottom:5px solid #c4511a; }}
    h1,h2,h3,p {{ margin-top:0; }} h1 {{ margin-bottom:8px; }} header p {{ margin:0; color:#d9e5db; }}
    main {{ max-width:1480px; margin:auto; padding:28px 24px 56px; }}
    .notice {{ padding:14px 17px; background:#e9f5ef; border-left:5px solid var(--accent); }}
    section {{ padding-top:30px; }} section > p {{ color:var(--muted); }}
    .grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:18px; }}
    .card {{ min-width:0; padding:14px; background:var(--surface); border:1px solid var(--line); }}
    .heading {{ display:flex; justify-content:space-between; gap:12px; align-items:baseline; margin-bottom:10px; }}
    h3 {{ margin:0; font-size:17px; overflow-wrap:anywhere; }}
    .heading span {{ color:var(--accent); font-size:12px; font-weight:700; text-transform:uppercase; }}
    video {{ display:block; width:100%; aspect-ratio:16/9; background:#111; object-fit:contain; }}
    dl {{ margin:12px 0; font-size:13px; }} dl div {{ display:grid; grid-template-columns:90px 1fr; gap:8px; padding:4px 0; border-bottom:1px solid #edf0ed; }}
    dt {{ color:var(--muted); }} dd {{ margin:0; font-family:"Courier New",monospace; overflow-wrap:anywhere; }}
    a {{ color:var(--accent); font-size:13px; font-weight:700; text-decoration:none; }}
    @media (max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header><h1>Ball-block 单变量仿真对比</h1><p>原始 8 个材料参数视频 + 10 个速度/方向/距离控制视频 · 60 FPS · 150 帧 · 1280×720</p></header>
  <main>
    <div class="notice"><strong>控制原则：</strong>速度实验只缩放速度模长；方向实验只旋转水平 yaw；距离实验只改变小球初始 x 坐标。其余材料、质量、几何、重力、阻尼、相机和渲染参数保持固定。</div>
    <section><h2>运动与初始位置控制组</h2><p>基准 e07_mu05_m1 与十个新增单变量 case。</p><div class="grid">{''.join(controlled_cards)}</div></section>
    <section><h2>原始材料参数组</h2><p>原有其余七个 restitution、friction 和 ball-mass 变化 case；基准已在上方展示。</p><div class="grid">{''.join(original_cards)}</div></section>
  </main>
</body>
</html>
"""
    (DASHBOARD_DIR / "index.html").write_text(page, encoding="utf-8")
    (DASHBOARD_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "baseline": BASELINE,
                "motion_cases": list(MOTION_ORDER),
                "original_cases": [BASELINE, *ORIGINAL_ORDER],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {DASHBOARD_DIR / 'index.html'} (18 videos)")


if __name__ == "__main__":
    main()
