#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import shutil
from collections import Counter
from pathlib import Path


DEFAULT_RIGID_ROOT = Path("/data/gaoya/agent-data/outputs/dataset_new_0705/AAA_check_0710")
DEFAULT_MPM_ROOT = Path("/data/gaoya/agent-data/outputs/dataset_new_0705/mpm_preview_batch_20260710")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/dataset_new_0705/unified_overview_20260710")
DEFAULT_PORT = 18830
RIGID_SIM_HZ = 240
RIGID_SIM_DT = 1.0 / float(RIGID_SIM_HZ)
RIGID_SIM_DURATION_S = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified local overview page for rigid and MPM dataset_new_0705 outputs.")
    parser.add_argument("--rigid-root", type=Path, default=DEFAULT_RIGID_ROOT)
    parser.add_argument("--mpm-root", type=Path, default=DEFAULT_MPM_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--page-title", default="dataset_new_0705 刚体 + MPM 总览")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src, target_is_directory=True)


def _case_object_summary(objects: list[dict], *, max_items: int = 3) -> str:
    parts: list[str] = []
    for obj in objects[:max_items]:
        texture = obj.get("texture_asset") or obj.get("texture_style") or obj.get("material_key") or "-"
        parts.append(f"{obj.get('shape', '-')} / {obj.get('role', '-')} / {texture}")
    return "；".join(parts) if parts else "-"


def _rigid_cases(rigid_root: Path) -> tuple[list[dict], dict[str, object]]:
    manifest_path = rigid_root / "manifest.json"
    manifest = _read_json(manifest_path)
    cases: list[dict] = []
    family_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    for item in manifest:
        meta = _read_json(Path(item["meta"]))
        dynamic_objects = [obj for obj in meta.get("objects", []) if obj.get("role") == "dynamic"]
        main_shape = dynamic_objects[0].get("shape", "unknown") if dynamic_objects else "unknown"
        family_key = str(item["family_key"])
        duration_s = float(meta.get("duration_s", RIGID_SIM_DURATION_S))
        sim_steps = int(round(duration_s * float(RIGID_SIM_HZ)))
        rel_video = Path("rigid_source") / Path(item["video"]).resolve().relative_to(rigid_root.resolve())
        cases.append(
            {
                "dataset": "rigid",
                "dataset_label": "Rigid / PyBullet",
                "family_key": family_key,
                "family_filter": f"rigid:{family_key}",
                "case_id": item["case_id"],
                "title": meta.get("title", item["case_id"]),
                "description": meta.get("description", ""),
                "video_rel": rel_video.as_posix(),
                "solver_type": "PyBullet rigid-body + Pyrender",
                "sim_type": meta.get("sim_type", "rigid_realism_v2"),
                "resolution": meta.get("resolution", [1280, 720]),
                "fps": int(meta.get("fps", 30)),
                "duration_s": duration_s,
                "pre_roll_s": float(meta.get("pre_roll_s", 0.0)),
                "dt": RIGID_SIM_DT,
                "sim_hz": RIGID_SIM_HZ,
                "sim_steps": sim_steps,
                "substeps": "-",
                "gravity": float(meta.get("gravity", 9.81)),
                "surface_key": meta.get("surface_key", ""),
                "lighting_key": meta.get("lighting_key", ""),
                "camera_key": meta.get("blueprint", {}).get("camera_key", ""),
                "vis_mode": "mesh",
                "objects_summary": _case_object_summary(meta.get("objects", [])),
                "tags": meta.get("tags", []),
                "family_label": meta.get("family", family_key),
                "param_lines": [
                    f"solver={meta.get('sim_type', 'rigid_realism_v2')}",
                    f"dt={RIGID_SIM_DT:.6f}s",
                    f"sim_hz={RIGID_SIM_HZ}",
                    f"steps={sim_steps}",
                    f"fps={int(meta.get('fps', 30))}",
                    f"res={meta.get('resolution', [1280, 720])[0]}x{meta.get('resolution', [1280, 720])[1]}",
                    f"pre_roll={float(meta.get('pre_roll_s', 0.0)):.3f}s",
                    f"floor_mu={float(meta.get('floor_friction', 0.0)):.3f}",
                ],
            }
        )
        family_counts[family_key] += 1
        shape_counts[main_shape] += 1
    cases.sort(key=lambda item: (item["family_key"], item["case_id"]))
    summary = {
        "root": str(rigid_root),
        "case_count": len(cases),
        "family_count": len(family_counts),
        "family_counts": dict(sorted(family_counts.items())),
        "shape_counts": dict(sorted(shape_counts.items())),
        "solver_type": "PyBullet rigid-body + Pyrender",
        "defaults": {
            "sim_hz": RIGID_SIM_HZ,
            "dt": RIGID_SIM_DT,
            "duration_s": RIGID_SIM_DURATION_S,
            "steps": int(RIGID_SIM_DURATION_S * RIGID_SIM_HZ),
            "fps": 30,
        },
    }
    return cases, summary


def _mpm_cases(mpm_root: Path) -> tuple[list[dict], dict[str, object]]:
    manifest_paths = sorted(mpm_root.glob("*/manifest.json"))
    cases: list[dict] = []
    family_counts: Counter[str] = Counter()
    motion_counts: Counter[str] = Counter()
    for manifest_path in manifest_paths:
        meta = _read_json(manifest_path)
        family_key = str(meta["family"])
        camera_res = meta.get("camera", {}).get("res", [960, 544])
        sim = meta.get("sim", {})
        rel_video = Path("mpm_source") / manifest_path.parent.resolve().relative_to(mpm_root.resolve()) / "video" / "preview.mp4"
        objects_summary = []
        if "soft_block" in meta.get("initial_state", {}):
            soft_block = meta["initial_state"]["soft_block"]
            objects_summary.append(f"soft_block / particles={soft_block.get('n_particles', '-')}")
        for key in sorted(meta.get("initial_state", {}).keys()):
            if key == "soft_block":
                continue
            objects_summary.append(f"{key} / rigid")
        cases.append(
            {
                "dataset": "mpm",
                "dataset_label": "MPM / Genesis",
                "family_key": family_key,
                "family_filter": f"mpm:{family_key}",
                "case_id": meta["case_key"],
                "title": meta.get("title", meta["case_key"]),
                "description": meta.get("description", ""),
                "video_rel": rel_video.as_posix(),
                "solver_type": "Genesis MPM Elastic + rigid coupling",
                "sim_type": "mpm_preview_case",
                "resolution": camera_res,
                "fps": int(meta.get("fps", 30)),
                "duration_s": float(meta.get("video_duration_s", 0.0)),
                "pre_roll_s": 0.0,
                "dt": float(sim.get("dt", 0.0)),
                "sim_hz": round(1.0 / float(sim.get("dt", 1.0))) if sim.get("dt") else "-",
                "sim_steps": int(sim.get("horizon", 0)),
                "substeps": int(sim.get("substeps", 0)),
                "gravity": float(sim.get("gravity", [0.0, 0.0, -9.81])[-1]) if sim.get("gravity") else -9.81,
                "surface_key": meta.get("scene_theme", ""),
                "lighting_key": "",
                "camera_key": "",
                "vis_mode": meta.get("mpm_vis_mode", "visual"),
                "objects_summary": "；".join(objects_summary) if objects_summary else "-",
                "tags": [meta.get("motion_profile", ""), meta.get("scene_theme", ""), meta.get("mpm_vis_mode", "")],
                "family_label": family_key,
                "param_lines": [
                    "solver=Genesis MPM Elastic + rigid coupling",
                    f"dt={float(sim.get('dt', 0.0)):.6f}s",
                    f"substeps={int(sim.get('substeps', 0))}",
                    f"horizon={int(sim.get('horizon', 0))}",
                    f"grid={int(sim.get('grid_density', 0))}",
                    f"fps={int(meta.get('fps', 30))}",
                    f"res={camera_res[0]}x{camera_res[1]}",
                    f"vis={meta.get('mpm_vis_mode', 'visual')}",
                ],
            }
        )
        family_counts[family_key] += 1
        motion_counts[str(meta.get("motion_profile", "unknown"))] += 1
    cases.sort(key=lambda item: (item["family_key"], item["case_id"]))
    summary = {
        "root": str(mpm_root),
        "case_count": len(cases),
        "family_count": len(family_counts),
        "family_counts": dict(sorted(family_counts.items())),
        "motion_counts": dict(sorted(motion_counts.items())),
        "solver_type": "Genesis MPM Elastic + rigid coupling",
        "defaults": {
            "fps": 30,
        },
    }
    return cases, summary


def _family_button(label: str, value: str, count: int | None = None) -> str:
    suffix = f" · {count}" if count is not None else ""
    return f'<button class="chip" data-filter-value="{html.escape(value)}">{html.escape(label + suffix)}</button>'


def _build_page(
    *,
    rigid_cases: list[dict],
    rigid_summary: dict[str, object],
    mpm_cases: list[dict],
    mpm_summary: dict[str, object],
    output_root: Path,
    page_title: str,
    port: int,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    _ensure_symlink(Path(rigid_summary["root"]), output_root / "rigid_source")
    _ensure_symlink(Path(mpm_summary["root"]), output_root / "mpm_source")

    all_cases = rigid_cases + mpm_cases
    all_cases.sort(key=lambda item: (item["dataset"], item["family_key"], item["case_id"]))
    all_family_buttons = [
        '<button class="chip active" data-filter-value="ALL">ALL</button>',
    ]
    for family_key, count in sorted(rigid_summary["family_counts"].items()):
        all_family_buttons.append(_family_button(f"Rigid {family_key}", f"rigid:{family_key}", count))
    for family_key, count in sorted(mpm_summary["family_counts"].items()):
        all_family_buttons.append(_family_button(f"MPM {family_key}", f"mpm:{family_key}", count))

    dataset_buttons = [
        '<button class="chip active" data-dataset-value="ALL">ALL</button>',
        f'<button class="chip" data-dataset-value="rigid">Rigid / PyBullet · {rigid_summary["case_count"]}</button>',
        f'<button class="chip" data-dataset-value="mpm">MPM / Genesis · {mpm_summary["case_count"]}</button>',
    ]

    cards: list[str] = []
    for case in all_cases:
        tags = "".join(f"<span>{html.escape(str(tag))}</span>" for tag in case["tags"] if str(tag))
        param_lines = "".join(f"<li>{html.escape(line)}</li>" for line in case["param_lines"])
        res_text = "x".join(str(v) for v in case["resolution"])
        cards.append(
            f"""
            <article class="card" data-dataset="{html.escape(case['dataset'])}" data-family="{html.escape(case['family_filter'])}">
              <div class="card-top">
                <div>
                  <div class="eyebrow">{html.escape(case['dataset_label'])} · {html.escape(case['family_label'])} · {html.escape(case['case_id'])}</div>
                  <h2>{html.escape(case['title'])}</h2>
                  <p class="desc">{html.escape(case['description'])}</p>
                </div>
                <div class="metrics">
                  <span>{html.escape(case['solver_type'])}</span>
                  <span>{html.escape(res_text)}</span>
                  <span>{html.escape(str(case['fps']))} fps</span>
                  <span>{html.escape(str(case['vis_mode']))}</span>
                </div>
              </div>
              <video controls preload="metadata" playsinline>
                <source src="{html.escape(case['video_rel'])}" type="video/mp4">
              </video>
              <div class="meta-grid">
                <div><strong>Solver</strong><span>{html.escape(case['solver_type'])}</span></div>
                <div><strong>dt</strong><span>{case['dt'] if isinstance(case['dt'], str) else f"{case['dt']:.6f}s"}</span></div>
                <div><strong>Steps</strong><span>{html.escape(str(case['sim_steps']))}</span></div>
                <div><strong>Substeps</strong><span>{html.escape(str(case['substeps']))}</span></div>
                <div><strong>Duration</strong><span>{float(case['duration_s']):.3f}s</span></div>
                <div><strong>Gravity</strong><span>{float(case['gravity']):.2f}</span></div>
                <div><strong>Surface</strong><span>{html.escape(str(case['surface_key']))}</span></div>
                <div><strong>Light</strong><span>{html.escape(str(case['lighting_key'])) or "-"}</span></div>
              </div>
              <div class="tags">{tags}</div>
              <p class="object-note">{html.escape(case['objects_summary'])}</p>
              <ul class="param-list">{param_lines}</ul>
            </article>
            """
        )

    collect_command = (
        "PYTHONPATH=/home/gaoya/Code_Video/phys_state_video/scripts:/home/gaoya/Code_Video "
        "/data/gaoya/miniconda3/envs/physxnet_mpm_env/bin/python "
        "/home/gaoya/Code_Video/phys_state_video/scripts/dataset_new_0705/build_unified_overview_page.py "
        f"--rigid-root {rigid_summary['root']} "
        f"--mpm-root {mpm_summary['root']} "
        f"--output-root {output_root} "
        f"--port {port}"
    )
    serve_command = (
        f"cd {output_root}\n"
        f"/data/gaoya/miniconda3/envs/physxnet_mpm_env/bin/python -m http.server {port} --bind 127.0.0.1"
    )
    oneshot_command = (
        f"/home/gaoya/Code_Video/phys_state_video/scripts/dataset_new_0705/run_unified_overview_20260710.sh"
    )

    overview_data = {
        "rigid_summary": rigid_summary,
        "mpm_summary": mpm_summary,
        "case_count": len(all_cases),
        "port": port,
        "output_root": str(output_root),
    }
    (output_root / "overview_data.json").write_text(json.dumps(overview_data, ensure_ascii=False, indent=2), encoding="utf-8")

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(page_title)}</title>
  <style>
    :root {{
      --bg: #ece7dd;
      --ink: #1f252b;
      --muted: #64717b;
      --panel: rgba(255, 250, 243, 0.92);
      --line: rgba(20, 27, 34, 0.10);
      --warm: #b45c37;
      --cool: #4d6c80;
      --soft: #e8d6bd;
      --shadow: 0 22px 44px rgba(48, 33, 18, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(180, 92, 55, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(77, 108, 128, 0.16), transparent 26%),
        linear-gradient(180deg, #f7f2e9 0%, #f0eadf 42%, #e9e3d8 100%);
    }}
    .shell {{
      width: min(1540px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 18px 0 50px;
    }}
    .hero {{
      position: sticky;
      top: 0;
      z-index: 10;
      padding: 22px 22px 18px;
      border-radius: 28px;
      background: rgba(247, 242, 233, 0.82);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(255,255,255,0.5);
      box-shadow: var(--shadow);
      margin-bottom: 22px;
    }}
    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--cool);
      margin-bottom: 8px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(30px, 4vw, 54px);
      letter-spacing: -0.04em;
      line-height: 1.02;
    }}
    .subtitle {{
      margin: 10px 0 0;
      max-width: 1040px;
      color: var(--muted);
      line-height: 1.65;
      font-size: 15px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .summary-box {{
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.74), rgba(255,247,239,0.88));
      border: 1px solid var(--line);
      padding: 14px 16px;
    }}
    .summary-box strong {{
      display: block;
      font-size: 11px;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      color: var(--cool);
      margin-bottom: 6px;
    }}
    .summary-box span {{
      display: block;
      font-size: 25px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .solver-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .solver-box {{
      border-radius: 20px;
      padding: 16px;
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
    }}
    .solver-box h3 {{
      margin: 0 0 8px;
      font-size: 18px;
    }}
    .solver-box p {{
      margin: 0 0 10px;
      color: var(--muted);
      line-height: 1.6;
      font-size: 14px;
    }}
    .solver-box ul {{
      margin: 0;
      padding-left: 18px;
      line-height: 1.6;
      color: var(--muted);
      font-size: 13px;
    }}
    .cmd-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .cmd-box {{
      background: #1d242a;
      color: #edf3f7;
      border-radius: 18px;
      padding: 14px 16px;
      box-shadow: var(--shadow);
    }}
    .cmd-box strong {{
      display: block;
      margin-bottom: 8px;
      color: #9fd6f2;
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .cmd-box pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.5;
      font-size: 13px;
      font-family: "IBM Plex Mono", "Source Code Pro", monospace;
    }}
    .filters {{
      margin-top: 18px;
      display: grid;
      gap: 12px;
    }}
    .filter-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .chip {{
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      cursor: pointer;
      background: rgba(255,255,255,0.74);
      color: var(--ink);
      box-shadow: inset 0 0 0 1px var(--line);
      transition: transform 160ms ease, background 160ms ease, color 160ms ease;
      font: inherit;
    }}
    .chip:hover {{ transform: translateY(-1px); background: #fff; }}
    .chip.active {{ background: var(--warm); color: #fff8f1; box-shadow: none; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.45);
      border-radius: 24px;
      padding: 18px;
      box-shadow: var(--shadow);
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .card.hidden {{ display: none; }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
    }}
    .card h2 {{
      margin: 4px 0 6px;
      font-size: 22px;
      line-height: 1.1;
      letter-spacing: -0.03em;
    }}
    .desc {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
      font-size: 14px;
    }}
    .metrics {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
      align-content: flex-start;
      min-width: 180px;
    }}
    .metrics span, .tags span {{
      display: inline-flex;
      align-items: center;
      padding: 6px 9px;
      border-radius: 999px;
      font-size: 12px;
      background: rgba(232, 214, 189, 0.56);
      color: #684429;
    }}
    video {{
      width: 100%;
      aspect-ratio: 16 / 9;
      border-radius: 16px;
      background: #000;
      border: 1px solid rgba(0,0,0,0.08);
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }}
    .meta-grid div {{
      background: rgba(255,255,255,0.68);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 10px 12px;
    }}
    .meta-grid strong {{
      display: block;
      font-size: 11px;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      color: var(--cool);
      margin-bottom: 6px;
    }}
    .meta-grid span {{
      font-size: 13px;
      font-weight: 600;
      line-height: 1.45;
    }}
    .tags {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .object-note {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }}
    .param-list {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.55;
      font-size: 13px;
    }}
    @media (max-width: 860px) {{
      .shell {{ width: min(100vw, calc(100vw - 14px)); padding-top: 10px; }}
      .hero {{ padding: 18px 16px; border-radius: 20px; }}
      .card-top {{ flex-direction: column; }}
      .metrics {{ justify-content: flex-start; min-width: 0; }}
      .meta-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="eyebrow">dataset_new_0705 · unified local entry</div>
      <h1>{html.escape(page_title)}</h1>
      <p class="subtitle">
        这个入口把刚体 batch 和 MPM batch 放到了同一页里。上半部分直接写出两套仿真的基础配置，
        下半部分可以按 `Rigid / MPM` 和 family 过滤视频卡片，统一检查求解器差异、分辨率、dt、步数、材质和可视化模式。
      </p>
      <div class="summary">
        <div class="summary-box"><strong>总视频数</strong><span>{len(all_cases)}</span></div>
        <div class="summary-box"><strong>Rigid Cases</strong><span>{rigid_summary['case_count']}</span></div>
        <div class="summary-box"><strong>MPM Cases</strong><span>{mpm_summary['case_count']}</span></div>
        <div class="summary-box"><strong>本地端口</strong><span>127.0.0.1:{port}</span></div>
      </div>
      <div class="solver-grid">
        <section class="solver-box">
          <h3>Rigid Solver</h3>
          <p>{html.escape(rigid_summary['solver_type'])}</p>
          <ul>
            <li>默认 `sim_hz = {RIGID_SIM_HZ}`，`dt = {RIGID_SIM_DT:.6f}s`，`duration = {RIGID_SIM_DURATION_S:.1f}s`，`steps = {int(RIGID_SIM_DURATION_S * RIGID_SIM_HZ)}`。</li>
            <li>当前这批 `AAA_check_0710` 输出分辨率是 `1280x720`，导出视频 `fps = 30`。</li>
            <li>family 分布：{html.escape("；".join(f"{k}={v}" for k, v in rigid_summary['family_counts'].items()))}</li>
          </ul>
        </section>
        <section class="solver-box">
          <h3>MPM Solver</h3>
          <p>{html.escape(mpm_summary['solver_type'])}</p>
          <ul>
            <li>每个 case 自带 `dt / substeps / horizon / grid_density`，页面卡片里逐条展开。</li>
            <li>当前 `F1-F13` 结果分辨率以 case manifest 为准，当前大多是 `960x544`，导出视频 `fps = 30`。</li>
            <li>family 分布：{html.escape("；".join(f"{k}={v}" for k, v in mpm_summary['family_counts'].items()))}</li>
          </ul>
        </section>
      </div>
      <div class="cmd-grid">
        <section class="cmd-box">
          <strong>Collect Command</strong>
          <pre>{html.escape(collect_command)}</pre>
        </section>
        <section class="cmd-box">
          <strong>Serve Command</strong>
          <pre>{html.escape(serve_command)}</pre>
        </section>
        <section class="cmd-box">
          <strong>One-Click Build + Serve</strong>
          <pre>{html.escape(oneshot_command)}</pre>
        </section>
      </div>
      <div class="filters">
        <div class="filter-row" id="dataset-filters">
          {"".join(dataset_buttons)}
        </div>
        <div class="filter-row" id="family-filters">
          {"".join(all_family_buttons)}
        </div>
      </div>
    </section>
    <section class="grid" id="case-grid">
      {"".join(cards)}
    </section>
  </main>
  <script>
    const datasetButtons = Array.from(document.querySelectorAll('[data-dataset-value]'));
    const familyButtons = Array.from(document.querySelectorAll('[data-filter-value]'));
    const cards = Array.from(document.querySelectorAll('.card'));
    let activeDataset = 'ALL';
    let activeFamily = 'ALL';

    function renderFilters() {{
      datasetButtons.forEach((btn) => btn.classList.toggle('active', btn.dataset.datasetValue === activeDataset));
      familyButtons.forEach((btn) => btn.classList.toggle('active', btn.dataset.filterValue === activeFamily));
      cards.forEach((card) => {{
        const datasetOk = activeDataset === 'ALL' || card.dataset.dataset === activeDataset;
        const familyOk = activeFamily === 'ALL' || card.dataset.family === activeFamily;
        card.classList.toggle('hidden', !(datasetOk && familyOk));
      }});
    }}

    datasetButtons.forEach((btn) => {{
      btn.addEventListener('click', () => {{
        activeDataset = btn.dataset.datasetValue;
        renderFilters();
      }});
    }});
    familyButtons.forEach((btn) => {{
      btn.addEventListener('click', () => {{
        activeFamily = btn.dataset.filterValue;
        renderFilters();
      }});
    }});
  </script>
</body>
</html>
"""
    output_path = output_root / "index.html"
    output_path.write_text(page, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    rigid_cases, rigid_summary = _rigid_cases(args.rigid_root)
    mpm_cases, mpm_summary = _mpm_cases(args.mpm_root)
    output_path = _build_page(
        rigid_cases=rigid_cases,
        rigid_summary=rigid_summary,
        mpm_cases=mpm_cases,
        mpm_summary=mpm_summary,
        output_root=args.output_root,
        page_title=args.page_title,
        port=args.port,
    )
    print(
        json.dumps(
            {
                "rigid_cases": len(rigid_cases),
                "mpm_cases": len(mpm_cases),
                "index_html": str(output_path),
                "output_root": str(args.output_root),
                "port": args.port,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
