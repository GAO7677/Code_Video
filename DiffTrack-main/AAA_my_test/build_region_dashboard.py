#!/usr/bin/env python3
"""Build a dependency-free dashboard for region-level DiffTrack results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REGION_META = {
    "object_a": {
        "short": "物体 A",
        "title": "高速运动体 · Wheel",
        "accent": "#e85d34",
        "finding": "frame-wise VAE 后，多数可见帧能够跟随高速物体；主要误差集中在快速位移、遮挡与碰撞附近。",
    },
    "object_b": {
        "short": "物体 B",
        "title": "受撞目标 · Crate Box",
        "accent": "#d69b2d",
        "finding": "逐帧直接匹配后，受撞目标在整段视频中保持稳定对应，碰撞阶段仅出现短暂误差波动。",
    },
    "background": {
        "short": "背景",
        "title": "静态场景区域",
        "accent": "#178c72",
        "finding": "Q/K 与 CoTracker 仍近似重合；背景结果不受 VAE 时间压缩协议变化影响。",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.result_dir / "index.html"
    run_manifest = json.loads((args.result_dir / "run_manifest.json").read_text())
    regions = {}
    for name, meta in REGION_META.items():
        metrics = json.loads((args.result_dir / name / "metrics.json").read_text())
        regions[name] = {**meta, "metrics": metrics}

    payload = {
        "run": {
            "case_key": run_manifest["case_key"],
            "sample_type": run_manifest["sample_type"],
            "mask_source": run_manifest["mask_source"],
            "query_frame": run_manifest["query_frame"],
            "points_per_region": run_manifest["points_per_region"],
            "layer": run_manifest["layer"],
            "inverse_step": run_manifest["inverse_step"],
            "matching_timestep": run_manifest.get("matching_timestep", run_manifest["inverse_step"]),
            "protocol": run_manifest.get("protocol", "temporal_vae_interpolated"),
            "frame_as_latent": run_manifest.get("frame_as_latent", False),
            "temporal_interpolation": run_manifest.get("temporal_interpolation", True),
            "chunk_len": run_manifest.get("chunk_len"),
        },
        "regions": regions,
        "comparison": (
            json.loads((args.result_dir / "protocol_comparison.json").read_text())
            if (args.result_dir / "protocol_comparison.json").exists()
            else None
        ),
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    html = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RegionTrack · Q/K correspondence audit</title>
  <style>
    :root {
      --ink: #18201f;
      --muted: #68716e;
      --paper: #f5f0e6;
      --panel: rgba(255, 253, 247, 0.9);
      --line: #d6ccba;
      --orange: #e85d34;
      --gold: #d69b2d;
      --green: #178c72;
      --blue: #277da1;
      --shadow: 0 18px 60px rgba(44, 49, 42, 0.11);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 0%, rgba(232, 93, 52, 0.12), transparent 31rem),
        radial-gradient(circle at 91% 19%, rgba(23, 140, 114, 0.13), transparent 30rem),
        linear-gradient(135deg, #f8f3e9 0%, #eee7d9 100%);
      font-family: "Trebuchet MS", "Noto Sans CJK SC", sans-serif;
      min-height: 100vh;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: 0.22;
      background-image: linear-gradient(rgba(24, 32, 31, 0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(24, 32, 31, 0.06) 1px, transparent 1px);
      background-size: 38px 38px;
      mask-image: linear-gradient(to bottom, black, transparent 72%);
    }
    main { width: min(1500px, calc(100% - 40px)); margin: 0 auto; padding: 40px 0 80px; position: relative; }
    .hero { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(300px, 0.6fr); gap: 34px; align-items: end; padding: 28px 0 34px; }
    .eyebrow { font: 700 12px/1.2 "Trebuchet MS", sans-serif; letter-spacing: 0.18em; text-transform: uppercase; color: var(--orange); }
    h1, h2, h3 { font-family: Georgia, "Noto Serif CJK SC", serif; margin: 0; font-weight: 600; }
    h1 { margin-top: 10px; font-size: clamp(46px, 6vw, 94px); line-height: 0.92; letter-spacing: -0.055em; max-width: 920px; }
    .hero p { color: var(--muted); font-size: 16px; line-height: 1.75; margin: 18px 0 0; max-width: 790px; }
    .run-card { border-left: 1px solid var(--line); padding-left: 28px; display: grid; gap: 13px; }
    .run-row { display: flex; justify-content: space-between; gap: 18px; font-size: 13px; }
    .run-row span:first-child { color: var(--muted); }
    .run-row code { color: var(--ink); font-weight: 700; }
    .verdict {
      background: #1f2927;
      color: #fffaf0;
      border-radius: 4px 28px 4px 4px;
      padding: 26px 30px;
      box-shadow: var(--shadow);
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 22px;
      align-items: center;
      margin-bottom: 28px;
    }
    .verdict-mark { width: 62px; height: 62px; display: grid; place-items: center; border-radius: 50%; background: var(--orange); font: 700 25px/1 Georgia, serif; }
    .verdict h2 { font-size: 24px; margin-bottom: 6px; }
    .verdict p { margin: 0; color: #cbd2ce; line-height: 1.65; }
    .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }
    .metric-card {
      --accent: var(--orange);
      background: var(--panel);
      border: 1px solid rgba(214, 204, 186, 0.9);
      border-top: 5px solid var(--accent);
      border-radius: 4px 4px 18px 4px;
      padding: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }
    .metric-card h3 { font-size: 23px; }
    .metric-card .sub { color: var(--muted); font-size: 13px; margin: 5px 0 22px; }
    .big-number { display: flex; align-items: baseline; gap: 5px; color: var(--accent); }
    .big-number strong { font: 600 54px/1 Georgia, serif; letter-spacing: -0.05em; }
    .big-number span { font-size: 15px; font-weight: 700; }
    .metric-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 18px; }
    .mini { padding-top: 10px; border-top: 1px solid var(--line); }
    .mini b { display: block; font-size: 16px; }
    .mini span { display: block; color: var(--muted); font-size: 11px; margin-top: 3px; }
    .finding { margin: 18px 0 0; min-height: 66px; line-height: 1.65; font-size: 14px; color: #414b48; }
    .section-head { display: flex; justify-content: space-between; align-items: end; gap: 20px; margin: 62px 0 20px; }
    .section-head h2 { font-size: clamp(30px, 4vw, 49px); letter-spacing: -0.035em; }
    .section-head p { max-width: 560px; color: var(--muted); margin: 0; line-height: 1.6; text-align: right; }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; }
    button {
      border: 1px solid var(--ink);
      background: transparent;
      color: var(--ink);
      border-radius: 999px;
      padding: 9px 15px;
      font: 700 12px/1 "Trebuchet MS", sans-serif;
      cursor: pointer;
      transition: 150ms ease;
    }
    button:hover { background: var(--ink); color: white; transform: translateY(-1px); }
    .region-block { margin-top: 22px; background: var(--panel); border: 1px solid var(--line); border-radius: 4px 26px 4px 4px; padding: 22px; box-shadow: var(--shadow); }
    .region-title { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
    .region-title i { width: 13px; height: 13px; border-radius: 50%; background: var(--accent); }
    .region-title h3 { font-size: 26px; }
    .media-grid { display: grid; grid-template-columns: 0.75fr 1.25fr; gap: 14px; }
    .media-card { background: #151a19; border-radius: 3px 14px 3px 3px; overflow: hidden; min-width: 0; }
    .media-card img, .media-card video { display: block; width: 100%; aspect-ratio: 3 / 2; object-fit: cover; background: #171b1a; }
    .media-card figcaption { color: #d9dedb; padding: 10px 13px; font-size: 12px; display: flex; justify-content: space-between; }
    .triptych { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 14px; }
    .chart-panel { background: var(--panel); border: 1px solid var(--line); padding: 24px; border-radius: 4px 22px 4px 4px; box-shadow: var(--shadow); }
    .chart-panel h3 { font-size: 22px; margin-bottom: 3px; }
    .chart-panel > p { margin: 0 0 18px; color: var(--muted); font-size: 13px; }
    .chart-wrap { width: 100%; min-height: 370px; }
    svg { display: block; width: 100%; height: 370px; overflow: visible; }
    .axis { stroke: #a69e8e; stroke-width: 1; }
    .grid-line { stroke: #dcd4c5; stroke-width: 1; stroke-dasharray: 3 5; }
    .tick { fill: #737b78; font: 11px "Trebuchet MS", sans-serif; }
    .chart-label { fill: #26302e; font: 700 12px "Trebuchet MS", sans-serif; }
    .legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; font-size: 12px; color: var(--muted); }
    .legend i { display: inline-block; width: 22px; height: 3px; vertical-align: middle; margin-right: 6px; }
    .note { margin-top: 24px; padding: 18px 22px; border-left: 4px solid var(--gold); background: rgba(255, 253, 247, 0.66); color: #4e5754; line-height: 1.65; font-size: 14px; }
    footer { margin-top: 58px; padding-top: 18px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; display: flex; justify-content: space-between; }
    @media (max-width: 980px) {
      main { width: min(100% - 22px, 760px); padding-top: 22px; }
      .hero, .metric-grid, .media-grid, .triptych { grid-template-columns: 1fr; }
      .hero { align-items: start; }
      .run-card { border-left: 0; border-top: 1px solid var(--line); padding: 18px 0 0; }
      .section-head { align-items: start; flex-direction: column; }
      .section-head p { text-align: left; }
      .metric-card .finding { min-height: 0; }
      .media-card img, .media-card video { aspect-ratio: 3 / 2; }
      .verdict { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<main>
  <header class="hero">
    <div>
      <div class="eyebrow">DiffTrack · Region-level audit</div>
      <h1>谁在撑起<br>96% 的 PCK？</h1>
      <p>将同一帧拆成高速运动物体、受撞目标与静态背景，每组独立采样 32 个 query。圆点和粗线代表 CoTracker，方框和细线代表 CogVideoX Q/K correspondence。</p>
    </div>
    <aside class="run-card" id="run-meta"></aside>
  </header>

  <section class="verdict">
    <div class="verdict-mark">!</div>
    <div>
      <h2>逐帧 VAE 编码修正了运动物体的时间错位。</h2>
      <p>高速 wheel 的 PCK@8 从旧协议的 11.38% 升至 61.90%，受撞箱体从 75.26% 升至 94.73%，而背景保持 96.83%。旧结果混入了时间压缩与插值误差，不能解释为逐帧 Q/K correspondence。</p>
    </div>
  </section>

  <section class="metric-grid" id="metric-grid"></section>

  <div class="section-head" id="protocol-head">
    <div><div class="eyebrow">Protocol correction</div><h2>编码协议改变了什么</h2></div>
    <p>同一批 query points、同一份 CoTracker 轨迹和同一层；对比旧 analyze-real 路径与论文 tracking 路径。核心差异是整段时序 VAE + 插值，改为逐帧独立 VAE + chunk 直接匹配。</p>
  </div>
  <section class="metric-grid" id="protocol-grid"></section>

  <div class="section-head">
    <div><div class="eyebrow">Visual evidence</div><h2>轨迹直接对照</h2></div>
    <div class="controls">
      <button id="play-all">同步播放</button>
      <button id="pause-all">全部暂停</button>
      <button id="reset-all">归零</button>
    </div>
  </div>
  <div id="region-visuals"></div>

  <div class="section-head">
    <div><div class="eyebrow">Temporal diagnosis</div><h2>误差发生在哪一帧</h2></div>
    <p>曲线仅统计 CoTracker 判定可见的 query；断点表示该帧没有可见点。PCK@8 越高越好，像素误差越低越好。</p>
  </div>
  <section class="chart-panel">
    <h3>逐帧平均位置误差</h3>
    <p>Q/K 预测坐标相对 CoTracker 坐标的欧氏距离，单位为像素。</p>
    <div class="chart-wrap" id="error-chart"></div>
    <div class="legend" id="error-legend"></div>
  </section>
  <section class="chart-panel" style="margin-top:18px">
    <h3>逐帧 PCK@8</h3>
    <p>误差小于 8 像素的可见 query 比例。</p>
    <div class="chart-wrap" id="pck-chart"></div>
    <div class="legend" id="pck-legend"></div>
  </section>

  <aside class="note"><strong>解释边界：</strong>物体 A 的 CoTracker 可见率只有 56.95%，所有 PCK 只在 CoTracker 判定可见的位置统计。frame-wise VAE 显著改善结果，但剩余 38.10% 的失败仍混合了 Q/K correspondence 误差与 CoTracker 在高速、遮挡阶段的不确定性。</aside>
  <footer><span>RegionTrack / CogVideoX-2B</span><span>Frame-wise VAE · No temporal interpolation · 49 direct predictions</span></footer>
</main>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
  const data = JSON.parse(document.getElementById('payload').textContent);
  const order = ['object_a', 'object_b', 'background'];
  const fmt = (value, digits = 2) => Number(value).toFixed(digits);
  const visibility = name => data.regions[name].metrics.visible_comparisons / (48 * data.run.points_per_region) * 100;

  document.getElementById('run-meta').innerHTML = [
    ['CASE', data.run.case_key], ['SAMPLE', data.run.sample_type], ['QUERY', `frame ${data.run.query_frame} · ${data.run.points_per_region} points/region`],
    ['PROBE', `layer ${data.run.layer} · matching ${data.run.matching_timestep}`],
    ['PROTOCOL', data.run.frame_as_latent ? `frame-wise VAE · chunks of ${data.run.chunk_len}` : 'temporal VAE · interpolated'],
    ['MASK', 'lossless renderer instance IDs']
  ].map(([k,v]) => `<div class="run-row"><span>${k}</span><code>${v}</code></div>`).join('');

  document.getElementById('metric-grid').innerHTML = order.map(name => {
    const r = data.regions[name], m = r.metrics;
    return `<article class="metric-card" style="--accent:${r.accent}">
      <h3>${r.title}</h3><div class="sub">${r.short} · CoTracker visible ${fmt(visibility(name), 1)}%</div>
      <div class="big-number"><strong>${fmt(m.pck8, 1)}</strong><span>% PCK@8</span></div>
      <div class="metric-strip">
        <div class="mini"><b>${fmt(m.mean_error_px, 1)} px</b><span>平均误差</span></div>
        <div class="mini"><b>${fmt(m.median_error_px, 1)} px</b><span>中位误差</span></div>
        <div class="mini"><b>${fmt(m.pck16, 1)}%</b><span>PCK@16</span></div>
      </div><p class="finding">${r.finding}</p></article>`;
  }).join('');

  if (data.comparison) {
    document.getElementById('protocol-grid').innerHTML = order.map(name => {
      const r = data.regions[name], c = data.comparison[name];
      const deltaClass = c.pck8_delta >= 0 ? '+' : '';
      return `<article class="metric-card" style="--accent:${r.accent}">
        <h3>${r.title}</h3><div class="sub">PCK@8 · protocol-only ablation</div>
        <div class="metric-strip">
          <div class="mini"><b>${fmt(c.temporal_vae_interpolated.pck8, 1)}%</b><span>时序 VAE + 插值</span></div>
          <div class="mini"><b>${fmt(c.framewise_vae.pck8, 1)}%</b><span>逐帧 VAE</span></div>
          <div class="mini"><b>${deltaClass}${fmt(c.pck8_delta, 1)} pp</b><span>PCK@8 变化</span></div>
        </div>
        <p class="finding">平均误差 ${fmt(c.temporal_vae_interpolated.mean_error_px, 1)} px → ${fmt(c.framewise_vae.mean_error_px, 1)} px，变化 ${fmt(c.mean_error_delta_px, 1)} px。</p>
      </article>`;
    }).join('');
  } else {
    document.getElementById('protocol-head').style.display = 'none';
    document.getElementById('protocol-grid').style.display = 'none';
  }

  document.getElementById('region-visuals').innerHTML = order.map(name => {
    const r = data.regions[name];
    return `<article class="region-block" style="--accent:${r.accent}">
      <div class="region-title"><i></i><h3>${r.title}</h3></div>
      <div class="media-grid">
        <figure class="media-card"><img src="${name}/mask_points.png" alt="${r.short} mask and query points"><figcaption><span>Mask 内采样</span><span>32 points</span></figcaption></figure>
        <figure class="media-card"><video controls muted loop playsinline preload="metadata" src="${name}/overlay_comparison.mp4"></video><figcaption><span>叠加比较</span><span>circle = CoTracker · square = Q/K</span></figcaption></figure>
      </div>
      <div class="triptych">
        <figure class="media-card"><video controls muted loop playsinline preload="metadata" src="${name}/cotracker_tracks.mp4"></video><figcaption><span>CoTracker</span><span>pseudo GT</span></figcaption></figure>
        <figure class="media-card"><video controls muted loop playsinline preload="metadata" src="${name}/qk_tracks.mp4"></video><figcaption><span>Q/K matching</span><span>L${data.run.layer} · M${data.run.matching_timestep}</span></figcaption></figure>
        <figure class="media-card"><video controls muted loop playsinline preload="metadata" src="${name}/overlay_comparison.mp4"></video><figcaption><span>误差连线</span><span>visible points only</span></figcaption></figure>
      </div></article>`;
  }).join('');

  const videos = () => [...document.querySelectorAll('video')];
  document.getElementById('play-all').onclick = () => { videos().forEach(v => { v.currentTime = 0; v.play(); }); };
  document.getElementById('pause-all').onclick = () => videos().forEach(v => v.pause());
  document.getElementById('reset-all').onclick = () => videos().forEach(v => { v.pause(); v.currentTime = 0; });

  function renderChart(target, key, maxY, suffix) {
    const width = 1240, height = 360, left = 62, right = 24, top = 24, bottom = 45;
    const plotW = width - left - right, plotH = height - top - bottom;
    const x = i => left + i / 48 * plotW;
    const y = v => top + (1 - Math.min(maxY, Math.max(0, v)) / maxY) * plotH;
    const parts = [`<svg viewBox="0 0 ${width} ${height}" role="img">`];
    for (let i = 0; i <= 5; i++) {
      const value = maxY * i / 5, yy = y(value);
      parts.push(`<line class="grid-line" x1="${left}" y1="${yy}" x2="${width-right}" y2="${yy}"/>`);
      parts.push(`<text class="tick" x="${left-10}" y="${yy+4}" text-anchor="end">${Math.round(value)}${suffix}</text>`);
    }
    [0, 8, 16, 24, 32, 40, 48].forEach(frame => {
      parts.push(`<line class="grid-line" x1="${x(frame)}" y1="${top}" x2="${x(frame)}" y2="${height-bottom}"/>`);
      parts.push(`<text class="tick" x="${x(frame)}" y="${height-bottom+22}" text-anchor="middle">${frame}</text>`);
    });
    parts.push(`<line class="axis" x1="${left}" y1="${height-bottom}" x2="${width-right}" y2="${height-bottom}"/>`);
    parts.push(`<text class="chart-label" x="${width-right}" y="${height-6}" text-anchor="end">frame</text>`);
    order.forEach(name => {
      const r = data.regions[name], values = r.metrics[key];
      let path = '', drawing = false;
      values.forEach((value, index) => {
        if (value === null) { drawing = false; return; }
        path += `${drawing ? 'L' : 'M'} ${x(index).toFixed(1)} ${y(value).toFixed(1)} `;
        drawing = true;
      });
      parts.push(`<path d="${path}" fill="none" stroke="${r.accent}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>`);
    });
    parts.push('</svg>');
    document.getElementById(target).innerHTML = parts.join('');
    document.getElementById(target.replace('chart', 'legend')).innerHTML = order.map(name => `<span><i style="background:${data.regions[name].accent}"></i>${data.regions[name].title}</span>`).join('');
  }
  renderChart('error-chart', 'per_frame_mean_error_px', 140, '');
  renderChart('pck-chart', 'per_frame_pck8', 100, '%');
</script>
</body>
</html>'''.replace("__PAYLOAD__", payload_json)

    output.write_text(html)
    print(output)


if __name__ == "__main__":
    main()
