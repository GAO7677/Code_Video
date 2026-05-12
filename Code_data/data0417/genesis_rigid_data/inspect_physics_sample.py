# 用途：检查单个 physics 样本并导出静态图。
"""该脚本用于检查单个 Genesis 刚体样本并生成静态摘要图；输入为 sample_dir 下的 metadata、rgb 帧和 physics 数据，输出为 sample_dir/visualizations 下的 summary_frames.png、summary_state.png 和 contact_timeline.png。"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils_io import colorize_label_map, depth_to_vis, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create static summary visualizations for one Genesis rigid sample.")
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--num_preview_frames", type=int, default=4, help="How many frames to include in the overview grid.")
    return parser.parse_args()


def load_rgb_frames(sample_dir: Path) -> list[np.ndarray]:
    rgb_dir = sample_dir / "rgb"
    frame_paths = sorted(rgb_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No RGB frames found under {rgb_dir}")
    return [imageio.imread(path) for path in frame_paths]


def pick_frame_indices(num_frames: int, num_preview_frames: int) -> np.ndarray:
    count = max(1, min(num_preview_frames, num_frames))
    return np.unique(np.linspace(0, num_frames - 1, count, dtype=int))


def save_overview_figure(
    sample_dir: Path,
    rgb_frames: list[np.ndarray],
    depth_metric: np.ndarray,
    seg: np.ndarray,
    frame_indices: np.ndarray,
    near: float,
    far: float,
) -> Path:
    vis_dir = sample_dir / "visualizations"
    ensure_dir(vis_dir)
    out_path = vis_dir / "summary_frames.png"

    fig, axes = plt.subplots(len(frame_indices), 3, figsize=(12, 3.6 * len(frame_indices)), squeeze=False)
    for row, frame_idx in enumerate(frame_indices):
        axes[row, 0].imshow(rgb_frames[frame_idx])
        axes[row, 0].set_title(f"RGB t={frame_idx}")
        axes[row, 1].imshow(depth_to_vis(depth_metric[frame_idx], near=near, far=far))
        axes[row, 1].set_title(f"Depth t={frame_idx}")
        axes[row, 2].imshow(colorize_label_map(seg[frame_idx]))
        axes[row, 2].set_title(f"Seg t={frame_idx}")
        for col in range(3):
            axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_state_figure(
    sample_dir: Path,
    meta: dict,
    rigid: dict[str, np.ndarray],
    contact_graph: np.ndarray,
    contact_impulse: np.ndarray,
) -> Path:
    vis_dir = sample_dir / "visualizations"
    ensure_dir(vis_dir)
    out_path = vis_dir / "summary_state.png"

    object_ids = rigid["object_ids"]
    com_pos = rigid["com_pos"]
    linear_vel = rigid["linear_vel"]
    angular_vel = rigid["angular_vel"]
    kinetic = rigid["kinetic_energy"]
    potential = rigid["potential_energy"]
    total = rigid["total_energy"]

    t = np.arange(com_pos.shape[0], dtype=np.float32) / max(float(meta.get("video_fps", meta.get("fps", 1))), 1.0)
    lin_speed = np.linalg.norm(linear_vel, axis=-1)
    ang_speed = np.linalg.norm(angular_vel, axis=-1)
    contact_count = contact_graph.sum(axis=(1, 2)) / 2.0
    max_impulse = contact_impulse.max(axis=(1, 2)) if contact_impulse.size > 0 else np.zeros((contact_graph.shape[0],), dtype=np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]
    for idx, object_id in enumerate(object_ids):
        ax.plot(com_pos[:, idx, 0], com_pos[:, idx, 1], label=f"obj{int(object_id)}")
    ax.set_title("Top-Down COM Trajectory (x-y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")

    ax = axes[0, 1]
    for idx, object_id in enumerate(object_ids):
        ax.plot(t, com_pos[:, idx, 2], label=f"obj{int(object_id)}")
    ax.set_title("COM Height")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("z (m)")
    ax.legend(loc="best")

    ax = axes[0, 2]
    for idx, object_id in enumerate(object_ids):
        ax.plot(t, lin_speed[:, idx], label=f"obj{int(object_id)}")
    ax.set_title("Linear Speed")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("m/s")
    ax.legend(loc="best")

    ax = axes[1, 0]
    for idx, object_id in enumerate(object_ids):
        ax.plot(t, ang_speed[:, idx], label=f"obj{int(object_id)}")
    ax.set_title("Angular Speed")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rad/s")
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.plot(t, kinetic, label="kinetic")
    ax.plot(t, potential, label="potential")
    ax.plot(t, total, label="total")
    ax.set_title("Energy")
    ax.set_xlabel("time (s)")
    ax.legend(loc="best")

    ax = axes[1, 2]
    ax.plot(t, contact_count, label="contact_count")
    ax.plot(t, max_impulse, label="max_impulse")
    ax.set_title("Contacts")
    ax.set_xlabel("time (s)")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_contact_timeline_figure(
    sample_dir: Path,
    meta: dict,
    rigid: dict[str, np.ndarray],
    contact_graph: np.ndarray,
    contact_impulse: np.ndarray,
) -> Path:
    vis_dir = sample_dir / "visualizations"
    ensure_dir(vis_dir)
    out_path = vis_dir / "contact_timeline.png"

    object_ids = [int(v) for v in rigid["object_ids"].tolist()]
    pair_indices = []
    pair_labels = []
    for i in range(len(object_ids)):
        for j in range(i + 1, len(object_ids)):
            pair_indices.append((i, j))
            pair_labels.append(f"obj{object_ids[i]}-obj{object_ids[j]}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    if not pair_indices:
        for ax in axes:
            ax.axis("off")
            ax.text(0.5, 0.5, "No object-object pairs in this sample", ha="center", va="center", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return out_path

    contact_pair_series = np.stack([contact_graph[:, i, j] for i, j in pair_indices], axis=0).astype(np.float32)
    impulse_pair_series = np.stack([contact_impulse[:, i, j] for i, j in pair_indices], axis=0).astype(np.float32)
    fps = max(float(meta.get("video_fps", meta.get("fps", 1))), 1.0)
    time_edges = np.arange(contact_graph.shape[0] + 1, dtype=np.float32) / fps
    y_edges = np.arange(len(pair_indices) + 1, dtype=np.float32)

    im0 = axes[0].pcolormesh(time_edges, y_edges, contact_pair_series, shading="auto", cmap="Greys", vmin=0.0, vmax=1.0)
    axes[0].set_title("Contact Graph Timeline")
    axes[0].set_ylabel("object pair")
    axes[0].set_yticks(np.arange(len(pair_labels), dtype=np.float32) + 0.5)
    axes[0].set_yticklabels(pair_labels)
    fig.colorbar(im0, ax=axes[0], label="contact")

    vmax = float(np.max(impulse_pair_series)) if impulse_pair_series.size > 0 else 0.0
    im1 = axes[1].pcolormesh(
        time_edges,
        y_edges,
        impulse_pair_series,
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=max(vmax, 1e-6),
    )
    axes[1].set_title("Contact Impulse Timeline")
    axes[1].set_ylabel("object pair")
    axes[1].set_xlabel("time (s)")
    axes[1].set_yticks(np.arange(len(pair_labels), dtype=np.float32) + 0.5)
    axes[1].set_yticklabels(pair_labels)
    fig.colorbar(im1, ax=axes[1], label="impulse")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    meta_path = sample_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    physics_dir = sample_dir / "physics"
    rgb_frames = load_rgb_frames(sample_dir)
    depth_metric = np.load(physics_dir / "depth_metric.npy")
    seg = np.load(physics_dir / "seg.npy")
    contact_graph = np.load(physics_dir / "contact_graph.npy")
    contact_impulse = np.load(physics_dir / "contact_impulse.npy")
    rigid_npz = np.load(physics_dir / "rigid_kinematics.npz")
    rigid = {key: rigid_npz[key] for key in rigid_npz.files}

    frame_indices = pick_frame_indices(depth_metric.shape[0], args.num_preview_frames)
    camera_intrinsics = meta["camera_intrinsics"]
    overview_path = save_overview_figure(
        sample_dir=sample_dir,
        rgb_frames=rgb_frames,
        depth_metric=depth_metric,
        seg=seg,
        frame_indices=frame_indices,
        near=float(camera_intrinsics["near"]),
        far=float(camera_intrinsics["far"]),
    )
    state_path = save_state_figure(
        sample_dir=sample_dir,
        meta=meta,
        rigid=rigid,
        contact_graph=contact_graph,
        contact_impulse=contact_impulse,
    )
    contact_path = save_contact_timeline_figure(
        sample_dir=sample_dir,
        meta=meta,
        rigid=rigid,
        contact_graph=contact_graph,
        contact_impulse=contact_impulse,
    )

    print(f"[DONE] wrote: {overview_path}")
    print(f"[DONE] wrote: {state_path}")
    print(f"[DONE] wrote: {contact_path}")


if __name__ == "__main__":
    main()
