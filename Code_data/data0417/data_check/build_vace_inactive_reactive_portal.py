#!/usr/bin/env python3
"""Build a local portal that compares VACE inactive/reactive inputs across train and inference paths."""

from __future__ import annotations

import html
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image


PORTAL_ROOT = Path("/home/gaoya/portal_hub_sim/vace_inactive_reactive_portal")
ASSET_ROOT = PORTAL_ROOT / "assets"
DATASET_SAMPLES = [
    Path(
        "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1adapter/train/genesis/rigid/single_object_preview/count_01/10096__case006_entry_right__ratio12"
    ),
    Path(
        "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1adapter/train/genesis/rigid/multi_object_free_motion/count_02/10007__case210_multi2_projectile_nocollision__ratio12"
    ),
    Path(
        "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1adapter/train/genesis/rigid/interaction_pair_plus_dynamic/count_02/10007__case003_static_highdrop__cf_no_collision_neg__ratio12"
    ),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_frames(frame_paths: list[str], size: tuple[int, int]) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for frame_path in frame_paths:
        frame = Image.open(frame_path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
        frames.append(frame)
    return frames


def solid_frame(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, color)


def build_train_condition_video(
    context_frames: list[Image.Image],
    total_frames: int,
    size: tuple[int, int],
) -> tuple[list[Image.Image], list[Image.Image]]:
    placeholder = solid_frame(size, (128, 128, 128))
    black = solid_frame(size, (0, 0, 0))
    white = solid_frame(size, (255, 255, 255))
    known = len(context_frames)
    video = list(context_frames) + [placeholder.copy() for _ in range(total_frames - known)]
    mask = [black.copy() for _ in range(known)] + [white.copy() for _ in range(total_frames - known)]
    return video, mask


def build_infer_v2v_condition_video(
    context_frames: list[Image.Image],
    total_frames: int,
    size: tuple[int, int],
) -> tuple[list[Image.Image], list[Image.Image]]:
    return build_train_condition_video(context_frames, total_frames, size)


def build_infer_ti2v_condition_video(
    context_frames: list[Image.Image],
    total_frames: int,
    size: tuple[int, int],
) -> tuple[list[Image.Image], list[Image.Image]]:
    placeholder = solid_frame(size, (128, 128, 128))
    black = solid_frame(size, (0, 0, 0))
    white = solid_frame(size, (255, 255, 255))
    first = context_frames[:1]
    video = list(first) + [placeholder.copy() for _ in range(total_frames - len(first))]
    mask = [black.copy() for _ in range(len(first))] + [white.copy() for _ in range(total_frames - len(first))]
    return video, mask


def apply_mask(video: list[Image.Image], mask: list[Image.Image], reactive: bool) -> list[Image.Image]:
    out: list[Image.Image] = []
    for frame, mask_frame in zip(video, mask):
        frame_rgb = frame.convert("RGB")
        mask_l = mask_frame.convert("L")
        src = frame_rgb.load()
        m = mask_l.load()
        result = Image.new("RGB", frame_rgb.size, (0, 0, 0))
        dst = result.load()
        width, height = frame_rgb.size
        for y in range(height):
            for x in range(width):
                mask_on = m[x, y] >= 128
                if reactive:
                    dst[x, y] = src[x, y] if mask_on else (0, 0, 0)
                else:
                    dst[x, y] = (0, 0, 0) if mask_on else src[x, y]
        out.append(result)
    return out


def save_gif(frames: list[Image.Image], path: Path, duration_ms: int = 150) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def save_mp4_from_frames(frames: list[Image.Image], path: Path, fps: int = 8) -> None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return
    frames_dir = path.parent / f"{path.stem}_frames"
    ensure_clean_dir(frames_dir)
    for idx, frame in enumerate(frames):
        frame.save(frames_dir / f"{idx:04d}.png")
    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "%04d.png"),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    shutil.rmtree(frames_dir)


def mean_abs_diff(frames_a: list[Image.Image], frames_b: list[Image.Image]) -> float:
    total = 0.0
    count = 0
    for frame_a, frame_b in zip(frames_a, frames_b):
        pixels_a = frame_a.convert("RGB")
        pixels_b = frame_b.convert("RGB")
        a = pixels_a.load()
        b = pixels_b.load()
        width, height = pixels_a.size
        for y in range(height):
            for x in range(width):
                pa = a[x, y]
                pb = b[x, y]
                total += sum(abs(int(pa[c]) - int(pb[c])) for c in range(3)) / 3.0
                count += 1
    return float(total / max(count, 1))


def build_sample_record(sample_dir: Path) -> dict[str, Any]:
    meta = load_json(sample_dir / "pair_meta.json")
    context_len = int(meta["context_len"])
    future_len = int(meta["future_len"])
    total_len = context_len + future_len
    size = (720, 544)

    context_frames = load_frames(meta["x_frame_paths"], size)
    future_frames = load_frames(meta["y_frame_paths"], size)
    full_frames = context_frames + future_frames

    train_video, train_mask = build_train_condition_video(context_frames, total_len, size)
    infer_v2v_video, infer_v2v_mask = build_infer_v2v_condition_video(context_frames, total_len, size)
    infer_ti2v_video, infer_ti2v_mask = build_infer_ti2v_condition_video(context_frames, total_len, size)

    variants = {
        "train": (train_video, train_mask),
        "infer_v2v": (infer_v2v_video, infer_v2v_mask),
        "infer_ti2v": (infer_ti2v_video, infer_ti2v_mask),
    }

    sample_asset_dir = ASSET_ROOT / sample_dir.name
    sample_asset_dir.mkdir(parents=True, exist_ok=True)

    save_gif(context_frames, sample_asset_dir / "context.gif")
    save_mp4_from_frames(context_frames, sample_asset_dir / "context.mp4")
    save_gif(full_frames, sample_asset_dir / "full.gif")
    save_mp4_from_frames(full_frames, sample_asset_dir / "full.mp4")

    exported_variants: dict[str, dict[str, str]] = {}
    for variant_name, (video, mask) in variants.items():
        inactive_frames = apply_mask(video, mask, reactive=False)
        reactive_frames = apply_mask(video, mask, reactive=True)
        save_gif(video, sample_asset_dir / f"{variant_name}_input.gif")
        save_mp4_from_frames(video, sample_asset_dir / f"{variant_name}_input.mp4")
        save_gif(mask, sample_asset_dir / f"{variant_name}_mask.gif")
        save_mp4_from_frames(mask, sample_asset_dir / f"{variant_name}_mask.mp4")
        save_gif(inactive_frames, sample_asset_dir / f"{variant_name}_inactive.gif")
        save_mp4_from_frames(inactive_frames, sample_asset_dir / f"{variant_name}_inactive.mp4")
        save_gif(reactive_frames, sample_asset_dir / f"{variant_name}_reactive.gif")
        save_mp4_from_frames(reactive_frames, sample_asset_dir / f"{variant_name}_reactive.mp4")
        exported_variants[variant_name] = {
            "input_gif": f"assets/{sample_dir.name}/{variant_name}_input.gif",
            "mask_gif": f"assets/{sample_dir.name}/{variant_name}_mask.gif",
            "inactive_gif": f"assets/{sample_dir.name}/{variant_name}_inactive.gif",
            "reactive_gif": f"assets/{sample_dir.name}/{variant_name}_reactive.gif",
            "input_mp4": f"assets/{sample_dir.name}/{variant_name}_input.mp4",
            "mask_mp4": f"assets/{sample_dir.name}/{variant_name}_mask.mp4",
            "inactive_mp4": f"assets/{sample_dir.name}/{variant_name}_inactive.mp4",
            "reactive_mp4": f"assets/{sample_dir.name}/{variant_name}_reactive.mp4",
        }

    comparisons = {
        "train_vs_infer_v2v_inactive": mean_abs_diff(
            apply_mask(train_video, train_mask, reactive=False),
            apply_mask(infer_v2v_video, infer_v2v_mask, reactive=False),
        ),
        "train_vs_infer_v2v_reactive": mean_abs_diff(
            apply_mask(train_video, train_mask, reactive=True),
            apply_mask(infer_v2v_video, infer_v2v_mask, reactive=True),
        ),
        "train_vs_infer_ti2v_inactive": mean_abs_diff(
            apply_mask(train_video, train_mask, reactive=False),
            apply_mask(infer_ti2v_video, infer_ti2v_mask, reactive=False),
        ),
        "train_vs_infer_ti2v_reactive": mean_abs_diff(
            apply_mask(train_video, train_mask, reactive=True),
            apply_mask(infer_ti2v_video, infer_ti2v_mask, reactive=True),
        ),
    }

    return {
        "sample_name": sample_dir.name,
        "sample_dir": str(sample_dir),
        "prompt": str(meta.get("prompt") or ""),
        "context_len": context_len,
        "future_len": future_len,
        "full_gif": f"assets/{sample_dir.name}/full.gif",
        "context_gif": f"assets/{sample_dir.name}/context.gif",
        "full_mp4": f"assets/{sample_dir.name}/full.mp4",
        "context_mp4": f"assets/{sample_dir.name}/context.mp4",
        "variants": exported_variants,
        "comparisons": comparisons,
    }


def card_html(title: str, media_path: str, kind: str, description: str) -> str:
    media = (
        f"<img src='{html.escape(media_path)}' alt='{html.escape(title)}'>"
        if kind == "gif"
        else f"<video src='{html.escape(media_path)}' controls preload='metadata' muted playsinline></video>"
    )
    return f"""
    <div class="card">
      <div class="card-title">{html.escape(title)}</div>
      <div class="card-desc">{html.escape(description)}</div>
      {media}
    </div>
    """


def build_html(records: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for record in records:
        variants_html: list[str] = []
        for variant_name, variant in record["variants"].items():
            label = {
                "train": "训练代码路径",
                "infer_v2v": "推理代码路径 v2v_clipref",
                "infer_ti2v": "推理代码路径 ti2v_firstframe",
            }[variant_name]
            variants_html.append(
                f"""
                <h3>{html.escape(label)}</h3>
                <div class="grid">
                  {card_html(f"{label} Input", variant["input_gif"], "gif", "送入 VACE 的 RGB 条件视频。")}
                  {card_html(f"{label} Mask", variant["mask_gif"], "gif", "黑色表示 inactive，白色表示 reactive。")}
                  {card_html(f"{label} Inactive", variant["inactive_gif"], "gif", "inactive = vace_video * (1 - mask)。")}
                  {card_html(f"{label} Reactive", variant["reactive_gif"], "gif", "reactive = vace_video * mask。")}
                </div>
                """
            )

        compare = record["comparisons"]
        sections.append(
            f"""
            <section class="sample">
              <h2>{html.escape(record["sample_name"])}</h2>
              <div class="sample-meta">
                <div><strong>路径：</strong>{html.escape(record["sample_dir"])}</div>
                <div><strong>Prompt：</strong>{html.escape(record["prompt"])}</div>
                <div><strong>ctx/future：</strong>{record["context_len"]} / {record["future_len"]}</div>
                <div><strong>差异统计：</strong>
                  train vs infer_v2v inactive = {compare["train_vs_infer_v2v_inactive"]:.4f},
                  reactive = {compare["train_vs_infer_v2v_reactive"]:.4f};
                  train vs infer_ti2v inactive = {compare["train_vs_infer_ti2v_inactive"]:.4f},
                  reactive = {compare["train_vs_infer_ti2v_reactive"]:.4f}
                </div>
              </div>
              <div class="grid">
                {card_html("Context", record["context_gif"], "gif", "原样本的 context 视频。")}
                {card_html("Full Video", record["full_gif"], "gif", "原样本的完整 GT 视频。")}
              </div>
              {''.join(variants_html)}
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VACE inactive/reactive compare</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f6f8; color: #111; }}
    .wrap {{ max-width: 1800px; margin: 0 auto; padding: 20px; }}
    .sample {{ background: #fff; border: 1px solid #d8dde6; border-radius: 12px; padding: 16px; margin-bottom: 18px; }}
    .sample-meta {{ font-size: 14px; line-height: 1.6; margin-bottom: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; margin-bottom: 12px; }}
    .card {{ background: #fafbfc; border: 1px solid #dde3ea; border-radius: 10px; padding: 10px; }}
    .card-title {{ font-weight: 700; margin-bottom: 6px; }}
    .card-desc {{ font-size: 13px; color: #555; margin-bottom: 8px; line-height: 1.5; }}
    img, video {{ width: 100%; display: block; border-radius: 8px; background: #000; }}
    h1, h2, h3 {{ margin-top: 0; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>VACE inactive / reactive 对比</h1>
    <p>说明：这里对比的是进入 VACE 之前的条件视频拆分结果，不是最终生成视频。训练路径和推理路径在 `v2v_clipref` 下应当基本一致；`ti2v_firstframe` 则只保留第一帧作为 inactive，其余都进入 reactive 区域。</p>
    {''.join(sections)}
  </div>
</body>
</html>
"""


def main() -> None:
    ensure_clean_dir(PORTAL_ROOT)
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    records = [build_sample_record(sample_dir) for sample_dir in DATASET_SAMPLES]
    (PORTAL_ROOT / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (PORTAL_ROOT / "index.html").write_text(build_html(records), encoding="utf-8")
    print(PORTAL_ROOT / "index.html")


if __name__ == "__main__":
    main()
