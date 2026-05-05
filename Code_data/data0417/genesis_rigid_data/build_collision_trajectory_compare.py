#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_COLLISION_CASE = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/"
    "train/rigid/interaction_pair_plus_dynamic/count_02/10054__case005_entry_left"
)
DEFAULT_NO_COLLISION_CASE = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/"
    "train/rigid/single_object_preview/count_01/10054__case005_entry_left__cf_no_collision_neg"
)
DEFAULT_PORTAL_ROOT = Path("/home/gaoya/portal_hub_sim")


def _load_case(case_dir: Path) -> Dict[str, Any]:
    meta_path = case_dir / "meta.json"
    if not meta_path.exists():
        meta_path = case_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    scene_input = json.loads((case_dir / "scene_input.json").read_text(encoding="utf-8"))
    physics_dir = case_dir / "physics"
    event_path = physics_dir / "event_windows.json"
    if not event_path.exists():
        event_path = physics_dir / "collision_events.json"
    events = json.loads(event_path.read_text(encoding="utf-8"))
    kin = np.load(case_dir / "physics" / "rigid_kinematics.npz")

    dt = float(metadata.get("simulation", {}).get("dt", 0.003))
    steps_per_frame = int(metadata.get("simulation", {}).get("steps_per_frame", 1))
    frame_dt = dt * max(1, steps_per_frame)
    frames = int(kin["com_pos"].shape[0])
    t = np.arange(frames, dtype=np.float64) * frame_dt

    main_pos = np.asarray(kin["com_pos"], dtype=np.float64)[:, 0, :]
    main_vel = np.asarray(kin["linear_vel"], dtype=np.float64)[:, 0, :]
    main_speed = np.linalg.norm(main_vel, axis=1)

    obj_obj_events = [e for e in events if -1 not in e.get("participants", [])]
    obj_env_events = [e for e in events if -1 in e.get("participants", [])]
    collision_spans = [
        (
            int(event.get("start_frame", event.get("frame_idx", -1))),
            int(event.get("end_frame", event.get("frame_idx", -1))),
        )
        for event in obj_obj_events
    ]
    collision_spans = [span for span in collision_spans if span[0] >= 0 and span[1] >= 0]

    return {
        "case_dir": case_dir,
        "metadata": metadata,
        "scene_input": scene_input,
        "events": events,
        "obj_obj_events": obj_obj_events,
        "obj_env_events": obj_env_events,
        "collision_spans": collision_spans,
        "frames": frames,
        "frame_dt": frame_dt,
        "time_s": t,
        "main_pos": main_pos,
        "main_vel": main_vel,
        "main_speed": main_speed,
        "video_path": case_dir / "videos" / "rgb.mp4",
    }


def _summary(case: Dict[str, Any]) -> Dict[str, Any]:
    pos = case["main_pos"]
    speed = case["main_speed"]
    return {
        "case_name": case["case_dir"].name,
        "motion_category": case["metadata"].get("motion_category"),
        "num_objects": int(case["metadata"].get("num_objects", 0)),
        "frames": int(case["frames"]),
        "duration_s": float(case["time_s"][-1]) if case["frames"] > 1 else 0.0,
        "start_pos": [float(v) for v in pos[0].tolist()],
        "end_pos": [float(v) for v in pos[-1].tolist()],
        "max_speed_mps": float(np.max(speed)),
        "mean_speed_mps": float(np.mean(speed)),
        "obj_obj_events": int(len(case["obj_obj_events"])),
        "obj_env_events": int(len(case["obj_env_events"])),
        "collision_spans": [[int(a), int(b)] for a, b in case["collision_spans"]],
        "counterfactual": case["metadata"].get("counterfactual"),
    }


def _shade_collision_spans(ax: plt.Axes, t: np.ndarray, spans: Sequence[Tuple[int, int]]) -> None:
    used_label = False
    for start_idx, end_idx in spans:
        start_idx = max(0, min(start_idx, len(t) - 1))
        end_idx = max(0, min(end_idx, len(t) - 1))
        label = "collision window" if not used_label else None
        ax.axvspan(t[start_idx], t[end_idx], color="#f3c46b", alpha=0.28, label=label)
        used_label = True


def _plot_xyz(collision_case: Dict[str, Any], no_collision_case: Dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    labels = ["x (m)", "y (m)", "z (m)"]
    colors = {"collision": "#b33a3a", "no_collision": "#1d6fa5"}
    for axis_idx, ax in enumerate(axes):
        ax.plot(collision_case["time_s"], collision_case["main_pos"][:, axis_idx], color=colors["collision"], lw=2.2, label="with collision")
        ax.plot(no_collision_case["time_s"], no_collision_case["main_pos"][:, axis_idx], color=colors["no_collision"], lw=2.2, label="without collision")
        _shade_collision_spans(ax, collision_case["time_s"], collision_case["collision_spans"])
        ax.set_ylabel(labels[axis_idx])
        ax.grid(alpha=0.25)
        if axis_idx == 0:
            ax.legend(loc="best")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Main Object COM Trajectory")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_speed(collision_case: Dict[str, Any], no_collision_case: Dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    colors = {"collision": "#b33a3a", "no_collision": "#1d6fa5"}

    axes[0].plot(collision_case["time_s"], collision_case["main_speed"], color=colors["collision"], lw=2.2, label="with collision")
    axes[0].plot(no_collision_case["time_s"], no_collision_case["main_speed"], color=colors["no_collision"], lw=2.2, label="without collision")
    _shade_collision_spans(axes[0], collision_case["time_s"], collision_case["collision_spans"])
    axes[0].set_ylabel("|v| (m/s)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    vel_labels = ["vx", "vz"]
    vel_indices = [0, 2]
    for idx, ax in enumerate(axes[1:]):
        pass
    axes[1].plot(collision_case["time_s"], collision_case["main_vel"][:, vel_indices[0]], color="#d95f02", lw=1.8, label="with collision vx")
    axes[1].plot(no_collision_case["time_s"], no_collision_case["main_vel"][:, vel_indices[0]], color="#7570b3", lw=1.8, label="without collision vx")
    axes[1].plot(collision_case["time_s"], collision_case["main_vel"][:, vel_indices[1]], color="#1b9e77", lw=1.8, ls="--", label="with collision vz")
    axes[1].plot(no_collision_case["time_s"], no_collision_case["main_vel"][:, vel_indices[1]], color="#66a61e", lw=1.8, ls="--", label="without collision vz")
    _shade_collision_spans(axes[1], collision_case["time_s"], collision_case["collision_spans"])
    axes[1].set_ylabel("velocity (m/s)")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", ncol=2)

    fig.suptitle("Main Object Speed and Key Velocity Components")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_phase(collision_case: Dict[str, Any], no_collision_case: Dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = {"collision": "#b33a3a", "no_collision": "#1d6fa5"}
    specs = [
        (0, 1, "XY trajectory", "x (m)", "y (m)"),
        (0, 2, "XZ trajectory", "x (m)", "z (m)"),
    ]
    for ax, (ix, iy, title, xlabel, ylabel) in zip(axes, specs):
        ax.plot(collision_case["main_pos"][:, ix], collision_case["main_pos"][:, iy], color=colors["collision"], lw=2.2, label="with collision")
        ax.plot(no_collision_case["main_pos"][:, ix], no_collision_case["main_pos"][:, iy], color=colors["no_collision"], lw=2.2, label="without collision")
        ax.scatter(collision_case["main_pos"][0, ix], collision_case["main_pos"][0, iy], color=colors["collision"], marker="o", s=40)
        ax.scatter(no_collision_case["main_pos"][0, ix], no_collision_case["main_pos"][0, iy], color=colors["no_collision"], marker="o", s=40)
        ax.scatter(collision_case["main_pos"][-1, ix], collision_case["main_pos"][-1, iy], color=colors["collision"], marker="x", s=60)
        ax.scatter(no_collision_case["main_pos"][-1, ix], no_collision_case["main_pos"][-1, iy], color=colors["no_collision"], marker="x", s=60)
        for start_idx, end_idx in collision_case["collision_spans"]:
            hit_idx = max(0, min(start_idx, collision_case["frames"] - 1))
            ax.scatter(
                collision_case["main_pos"][hit_idx, ix],
                collision_case["main_pos"][hit_idx, iy],
                color="#f3c46b",
                edgecolor="#7d5a14",
                s=70,
                zorder=5,
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    fig.suptitle("Main Object Path Projection")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _rel_link(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _copy_or_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_html(
    out_dir: Path,
    collision_case: Dict[str, Any],
    no_collision_case: Dict[str, Any],
    summary: Dict[str, Any],
) -> None:
    collision_video = _rel_link(out_dir / "with_collision.mp4", out_dir)
    no_collision_video = _rel_link(out_dir / "without_collision.mp4", out_dir)
    xyz_plot = _rel_link(out_dir / "compare_xyz.png", out_dir)
    speed_plot = _rel_link(out_dir / "compare_speed.png", out_dir)
    phase_plot = _rel_link(out_dir / "compare_phase.png", out_dir)

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Collision vs No-Collision Trajectory Compare</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --ink: #1d2328;
      --muted: #5d6670;
      --line: #d6c8b7;
      --card: rgba(255,255,255,0.92);
      --red: #b33a3a;
      --blue: #1d6fa5;
      --gold: #f3c46b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(179,58,58,0.12), transparent 28%),
        linear-gradient(180deg, #fcf9f4 0%, var(--bg) 100%);
    }}
    main {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px 20px 56px;
    }}
    .hero, .card {{
      border: 1px solid var(--line);
      background: var(--card);
      border-radius: 18px;
      box-shadow: 0 18px 50px rgba(47, 38, 27, 0.08);
    }}
    .hero {{
      padding: 22px 24px;
      margin-bottom: 20px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-family: "IBM Plex Serif", "Noto Serif SC", serif;
      font-size: clamp(28px, 3vw, 40px);
    }}
    .hero p {{
      margin: 8px 0;
      color: var(--muted);
      line-height: 1.55;
    }}
    .grid2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 18px;
      margin-bottom: 18px;
    }}
    .card {{
      padding: 16px;
    }}
    .card h2 {{
      margin: 0 0 8px;
      font-size: 20px;
    }}
    .tag {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      color: #fff;
      font-size: 13px;
      margin-bottom: 10px;
    }}
    .red {{ background: var(--red); }}
    .blue {{ background: var(--blue); }}
    video, img {{
      display: block;
      width: 100%;
      border-radius: 14px;
      background: #111;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-top: 12px;
    }}
    .metric {{
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,0.72);
    }}
    .metric strong {{
      display: block;
      margin-bottom: 6px;
    }}
    pre {{
      margin: 0;
      padding: 14px;
      border-radius: 14px;
      overflow: auto;
      background: #1d1f21;
      color: #f8f3eb;
      font-size: 12px;
      line-height: 1.45;
    }}
    .plots {{
      display: grid;
      gap: 18px;
    }}
    @media (max-width: 640px) {{
      main {{ padding: 16px 12px 36px; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Collision vs No-Collision Trajectory Compare</h1>
      <p>对比样本选择为同一对象 `10054` 的 `case005_entry_left` factual 碰撞样本，以及它的 `cf_no_collision_neg` 无碰撞反事实样本。</p>
      <p>曲线全部取主物体质心轨迹和线速度。黄色阴影表示“有碰撞”样本中的物体间接触窗口。</p>
    </section>

    <section class="grid2">
      <article class="card">
        <div class="tag red">with collision</div>
        <h2>{html.escape(collision_case["case_dir"].name)}</h2>
        <video controls preload="metadata" playsinline src="{collision_video}"></video>
      </article>
      <article class="card">
        <div class="tag blue">without collision</div>
        <h2>{html.escape(no_collision_case["case_dir"].name)}</h2>
        <video controls preload="metadata" playsinline src="{no_collision_video}"></video>
      </article>
    </section>

    <section class="card" style="margin-bottom: 18px;">
      <h2>Summary</h2>
      <div class="metrics">
        <div class="metric"><strong>with collision</strong>max speed: {summary["with_collision"]["max_speed_mps"]:.3f} m/s<br>obj-obj events: {summary["with_collision"]["obj_obj_events"]}<br>obj-env events: {summary["with_collision"]["obj_env_events"]}</div>
        <div class="metric"><strong>without collision</strong>max speed: {summary["without_collision"]["max_speed_mps"]:.3f} m/s<br>obj-obj events: {summary["without_collision"]["obj_obj_events"]}<br>obj-env events: {summary["without_collision"]["obj_env_events"]}</div>
        <div class="metric"><strong>collision windows</strong>{html.escape(str(summary["with_collision"]["collision_spans"]))}</div>
      </div>
    </section>

    <section class="plots">
      <article class="card">
        <h2>Main Object XYZ</h2>
        <img src="{xyz_plot}" alt="xyz compare">
      </article>
      <article class="card">
        <h2>Speed Compare</h2>
        <img src="{speed_plot}" alt="speed compare">
      </article>
      <article class="card">
        <h2>Path Projection</h2>
        <img src="{phase_plot}" alt="phase compare">
      </article>
      <article class="card">
        <h2>Raw Summary JSON</h2>
        <pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
      </article>
    </section>
  </main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local HTML page comparing collision vs no-collision trajectory curves.")
    parser.add_argument("--collision_case", type=str, default=str(DEFAULT_COLLISION_CASE))
    parser.add_argument("--no_collision_case", type=str, default=str(DEFAULT_NO_COLLISION_CASE))
    parser.add_argument("--portal_root", type=str, default=str(DEFAULT_PORTAL_ROOT))
    parser.add_argument("--slug", type=str, default="trajectory_compare_collision_vs_no_collision_10054_case005")
    args = parser.parse_args()

    collision_case = _load_case(Path(args.collision_case).resolve())
    no_collision_case = _load_case(Path(args.no_collision_case).resolve())

    out_dir = Path(args.portal_root).resolve() / args.slug
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _copy_or_symlink(collision_case["video_path"], out_dir / "with_collision.mp4")
    _copy_or_symlink(no_collision_case["video_path"], out_dir / "without_collision.mp4")

    _plot_xyz(collision_case, no_collision_case, out_dir / "compare_xyz.png")
    _plot_speed(collision_case, no_collision_case, out_dir / "compare_speed.png")
    _plot_phase(collision_case, no_collision_case, out_dir / "compare_phase.png")

    summary = {
        "with_collision": _summary(collision_case),
        "without_collision": _summary(no_collision_case),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_html(out_dir, collision_case, no_collision_case, summary)
    print(json.dumps({"portal_dir": str(out_dir), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
