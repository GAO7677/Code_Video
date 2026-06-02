#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import html
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601")
DEFAULT_PREVIEW_ROOT = DEFAULT_PROJECT_ROOT / "preview_v1"
DEFAULT_INDUSTRIAL_ROOT = DEFAULT_PREVIEW_ROOT / "industrial"
DEFAULT_DAILY_ROOT = DEFAULT_PREVIEW_ROOT / "daily"
DEFAULT_OUTPUT_ROOT = DEFAULT_PREVIEW_ROOT / "overview"
DEFAULT_PORT = 18827


def load_manifest(root: Path) -> list[dict]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src, target_is_directory=True)


def object_summary(item: dict) -> str:
    parts = []
    for obj in item["objects"]:
        role = obj["role"]
        role_text = "遮挡物" if role == "occluder" else ("支撑物" if role == "support" else "动态物体")
        tex = obj.get("texture_asset") or obj.get("texture_style")
        parts.append(f"{obj['name']} / {obj['shape']} / {role_text} / {tex}")
    return "；".join(parts)


def summarize_counts(items: list[dict]) -> tuple[int, dict[str, int], dict[str, int]]:
    family_counter: collections.Counter[str] = collections.Counter()
    shape_counter: collections.Counter[str] = collections.Counter()
    for item in items:
        family_counter[item["family"]] += 1
        dynamic_objects = [obj for obj in item["objects"] if obj.get("role") == "dynamic"]
        main_shape = dynamic_objects[0]["shape"] if dynamic_objects else "unknown"
        shape_counter[main_shape] += 1
    return len(items), dict(family_counter), dict(shape_counter)


def format_counter(title: str, counts: dict[str, int]) -> str:
    parts = [f"{name} {count}" for name, count in counts.items()]
    body = "；".join(parts) if parts else "无"
    return f"{title}：{body}"


def build_page(industrial: list[dict], daily: list[dict], output_root: Path, port: int) -> Path:
    industrial_by_key = {item["key"]: item for item in industrial}
    daily_by_key = {item["key"]: item for item in daily}
    ordered_keys = [item["key"] for item in industrial if item["key"] in daily_by_key]

    capsule_keys = [key for key in ordered_keys if key.startswith("simple_f1_capsule")]
    industrial_total, industrial_family, industrial_shape = summarize_counts(industrial)
    daily_total, daily_family, daily_shape = summarize_counts(daily)
    paired_cards: list[str] = []
    for key in ordered_keys:
        item_i = industrial_by_key[key]
        item_d = daily_by_key[key]
        paired_cards.append(
            f"""
            <article class="card" id="{html.escape(key)}">
              <div class="card-head">
                <div>
                  <div class="eyebrow">{html.escape(item_i['family'])} · {html.escape(key)}</div>
                  <h3>{html.escape(item_i['title'])}</h3>
                  <p class="desc">{html.escape(item_i['description'])}</p>
                  <p class="desc alt">{html.escape(item_d['title'])}：{html.escape(item_d['description'])}</p>
                </div>
                <div class="meta">
                  <span>g={item_i['gravity']}</span>
                  <span>floor_mu={item_i['floor_friction']}</span>
                  <span>pre-roll={item_i['pre_roll_s']}s</span>
                  <span>objects={len(item_i['objects'])}</span>
                </div>
              </div>
              <div class="video-grid">
                <section class="video-panel">
                  <div class="panel-title">工业训练数据版</div>
                  <video controls muted loop playsinline preload="metadata">
                    <source src="industrial/videos/{html.escape(key)}.mp4" type="video/mp4">
                  </video>
                  <p class="panel-note">{html.escape(object_summary(item_i))}</p>
                </section>
                <section class="video-panel">
                  <div class="panel-title">日常物体版</div>
                  <video controls muted loop playsinline preload="metadata">
                    <source src="daily/videos/{html.escape(key)}.mp4" type="video/mp4">
                  </video>
                  <p class="panel-note">{html.escape(object_summary(item_d))}</p>
                </section>
              </div>
            </article>
            """
        )

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>仿真数据总页面</title>
  <style>
    :root {{
      --bg: #0f1214;
      --panel: rgba(25, 29, 32, 0.94);
      --line: rgba(255, 255, 255, 0.10);
      --text: #eef1f3;
      --muted: #b6c0c8;
      --accent: #f2994a;
      --accent2: #64b6d9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(242, 153, 74, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(100, 182, 217, 0.15), transparent 26%),
        linear-gradient(180deg, #121517 0%, #0c0f11 100%);
    }}
    .page {{
      max-width: 1700px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero, .section, .card {{
      border: 1px solid var(--line);
      border-radius: 20px;
      background: var(--panel);
      backdrop-filter: blur(10px);
    }}
    .hero {{
      padding: 24px 26px;
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
    }}
    .lead {{
      margin: 0;
      color: var(--muted);
      line-height: 1.8;
      font-size: 15px;
      max-width: 1200px;
    }}
    .pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }}
    .pill {{
      padding: 8px 14px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.04);
      color: var(--text);
      font-size: 13px;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .stats-card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px 18px;
      background: rgba(255, 255, 255, 0.03);
    }}
    .stats-card h3 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    .stats-card p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.8;
      font-size: 13px;
    }}
    .stats-card p + p {{
      margin-top: 8px;
    }}
    .section {{
      padding: 20px 22px;
      margin-bottom: 18px;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 24px;
    }}
    .section p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.8;
      font-size: 14px;
    }}
    .section p + p {{
      margin-top: 10px;
    }}
    .anchor-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}
    .anchor {{
      display: inline-flex;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--text);
      text-decoration: none;
      font-size: 13px;
      background: rgba(255, 255, 255, 0.03);
    }}
    .cards {{
      display: grid;
      gap: 18px;
    }}
    .card {{
      padding: 18px;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 14px;
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    h3 {{
      margin: 0 0 8px;
      font-size: 20px;
    }}
    .desc {{
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
      font-size: 14px;
      max-width: 1000px;
    }}
    .desc.alt {{
      margin-top: 6px;
      color: #d8e1e8;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }}
    .meta span {{
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      font-size: 12px;
      color: #dde5eb;
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .video-panel {{
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      background: rgba(255, 255, 255, 0.02);
    }}
    .panel-title {{
      padding: 12px 14px 0;
      font-size: 14px;
      color: #e7edf2;
      font-weight: 600;
    }}
    video {{
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #000;
      margin-top: 10px;
    }}
    .panel-note {{
      margin: 0;
      padding: 12px 14px 14px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.65;
    }}
    .footer {{
      margin-top: 24px;
      color: var(--muted);
      font-size: 13px;
    }}
    a {{ color: var(--accent2); }}
    @media (max-width: 980px) {{
      .stats-grid {{
        grid-template-columns: 1fr;
      }}
      .video-grid {{
        grid-template-columns: 1fr;
      }}
      .card-head {{
        flex-direction: column;
      }}
      .meta {{
        justify-content: flex-start;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>仿真数据总页面</h1>
      <p class="lead">
        这一页是 `phys_state_video` 仿真数据集的统一可视化入口。当前主方案是基于 PyBullet 和 Pyrender 的简单刚体仿真，
        优先使用 `sphere / box / cylinder / capsule / puck`，固定地球重力 `9.81 m/s²`，显式随机化地面摩擦，避免中途出生，
        并保留逐帧 object-level 状态真值。页面内部把同一组物理 case 的工业训练数据版和日常物体版并排展示，方便直接对比外观风格和运动结果。
      </p>
      <div class="pills">
        <span class="pill">配对案例数 {len(ordered_keys)}</span>
        <span class="pill">工业版总数 {industrial_total}</span>
        <span class="pill">日常版总数 {daily_total}</span>
        <span class="pill">Capsule 变体 {len(capsule_keys)}</span>
        <span class="pill">默认生成主题 = 工业训练数据版</span>
        <span class="pill">工业版目录 {html.escape(str(DEFAULT_INDUSTRIAL_ROOT))}</span>
        <span class="pill">日常版目录 {html.escape(str(DEFAULT_DAILY_ROOT))}</span>
        <span class="pill">总页面端口 {port}</span>
      </div>
      <div class="stats-grid">
        <section class="stats-card">
          <h3>工业训练数据版统计</h3>
          <p>总数据量：{industrial_total} 个 case。</p>
          <p>{html.escape(format_counter("分类别", industrial_family))}</p>
          <p>{html.escape(format_counter("分类型", industrial_shape))}</p>
        </section>
        <section class="stats-card">
          <h3>日常物体版统计</h3>
          <p>总数据量：{daily_total} 个 case。</p>
          <p>{html.escape(format_counter("分类别", daily_family))}</p>
          <p>{html.escape(format_counter("分类型", daily_shape))}</p>
        </section>
      </div>
    </section>

    <section class="section">
      <h2>方案说明</h2>
      <p>
        当前仿真数据方案的核心目标，是先构造一套“物理碰撞干净、状态监督完整、视觉外观可控”的训练源，再逐步扩到更大规模。
        现阶段优先覆盖五类现象：单物体运动、双体碰撞、多体连锁、遮挡重现、支撑与跌落，并重点增加 `capsule` 在不同初始角度、线速度、角速度下的滚滑和翻滚变体。
      </p>
      <p>
        外观上目前同时维护两条主题：工业训练数据版偏规整喷涂、标签、警示条和真实木纹；日常物体版偏玩具球、收纳盒、杯罐和路障柱。
        两条主题共用同一套物理参数，只在材质、纹理和命名语义上做切换，便于后续做外观泛化测试。
      </p>
      <div class="anchor-row">
        <a class="anchor" href="industrial/index.html">工业训练数据版单独入口</a>
        <a class="anchor" href="daily/index.html">日常物体版单独入口</a>
        <a class="anchor" href="#simple_f1_capsule_slide_spin">跳到 Capsule 组</a>
        <a class="anchor" href="#simple_f3_capsule_box_cylinder_chain">跳到 Capsule 连锁组</a>
        <a class="anchor" href="#simple_f5_cylinder_topple">跳到支撑与跌落组</a>
      </div>
    </section>

    <section class="section">
      <h2>可视化说明</h2>
      <p>
        每个 case 卡片内左侧是工业版，右侧是日常物体版；两侧共享相同的重力、摩擦、时长和运动参数。
        对比时优先看三个维度：一是运动是否连续、是否有不合理插地或漂浮；二是不同外观主题下同一物理运动是否仍然清晰可辨；三是带真实木纹的盒体和支撑体是否比早期程序材质更自然。
      </p>
    </section>

    <div class="cards">
      {''.join(paired_cards)}
    </div>

    <div class="footer">
      本地访问地址：
      <a href="http://127.0.0.1:{port}">http://127.0.0.1:{port}</a>
    </div>
  </div>
</body>
</html>
"""
    html_path = output_root / "index.html"
    html_path.write_text(page, encoding="utf-8")
    return html_path


def start_server(output_root: Path, port: int) -> int:
    log_path = output_root / f"http_{port}.log"
    pid_path = output_root / f"http_{port}.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)
            return pid
        except Exception:
            pid_path.unlink(missing_ok=True)
    with open(log_path, "wb") as handle:
        proc = subprocess.Popen(
            ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1"],
            cwd=str(output_root),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    time.sleep(1.0)
    return proc.pid


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the unified simulation overview page.")
    parser.add_argument("--industrial-root", type=Path, default=DEFAULT_INDUSTRIAL_ROOT)
    parser.add_argument("--daily-root", type=Path, default=DEFAULT_DAILY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--serve-only", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for name in ["industrial", "daily"]:
            path = args.output_root / name
            if path.is_symlink() or path.exists():
                if path.is_symlink() or path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)

    ensure_symlink(args.industrial_root, args.output_root / "industrial")
    ensure_symlink(args.daily_root, args.output_root / "daily")

    industrial = load_manifest(args.industrial_root)
    daily = load_manifest(args.daily_root)
    html_path = build_page(industrial, daily, args.output_root, args.port)
    pid = start_server(args.output_root, args.port)
    print(f"overview: {html_path}")
    print(f"server: http://127.0.0.1:{args.port}")
    print(f"pid: {pid}")


if __name__ == "__main__":
    main()
