from pathlib import Path
import json

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW")
SOURCE_VIDEO = ROOT / "tdw_cloth_drop_real_scene_from_baseline" / "sphere_on_ground_real_scene.mp4"
OUTPUT_DIR = ROOT / "tdw_axis_export_compare"
OUTPUT_VIDEO = OUTPUT_DIR / "axis_export_compare_sphere_on_ground.mp4"
OUTPUT_JSON = OUTPUT_DIR / "axis_export_compare_sphere_on_ground.json"
OUTPUT_POSTER = OUTPUT_DIR / "axis_export_compare_sphere_on_ground_poster.png"


def yup_to_zup_vec(v):
    x, y, z = [float(a) for a in v]
    return [x, -z, y]


def load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def draw_multiline(draw: ImageDraw.ImageDraw, xy, lines, font, fill, line_gap=6):
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y = bbox[3] + line_gap


def make_canvas(frame: np.ndarray, idx: int, fps: float, sample_world_tdw, sample_velocity_tdw):
    h, w = frame.shape[:2]
    footer_h = 300
    gap = 24
    canvas = Image.new("RGB", (w * 2 + gap * 3, h + footer_h + gap * 3), (245, 240, 232))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(32)
    head_font = load_font(24)
    body_font = load_font(18)
    mono_font = load_font(20)

    left_x = gap
    top_y = gap * 2
    right_x = left_x + w + gap
    footer_y = top_y + h + gap

    frame_img = Image.fromarray(frame)
    canvas.paste(frame_img, (left_x, top_y))
    canvas.paste(frame_img, (right_x, top_y))

    draw.text((gap, 14), "TDW axis export comparison: same rendered video, different coordinate convention",
              font=title_font, fill=(24, 22, 19))

    # Left panel.
    draw.rounded_rectangle((left_x + 18, top_y + 18, left_x + 360, top_y + 110), radius=14, fill=(255, 255, 255))
    draw.text((left_x + 34, top_y + 34), "Y-up export (TDW native)", font=head_font, fill=(133, 95, 60))
    draw.text((left_x + 34, top_y + 70), "up axis = +Y, gravity = [0, -9.81, 0]", font=body_font, fill=(60, 58, 54))

    # Right panel.
    draw.rounded_rectangle((right_x + 18, top_y + 18, right_x + 368, top_y + 110), radius=14, fill=(255, 255, 255))
    draw.text((right_x + 34, top_y + 34), "Z-up export (Genesis-style)", font=head_font, fill=(70, 107, 93))
    draw.text((right_x + 34, top_y + 70), "up axis = +Z, gravity = [0, 0, -9.81]", font=body_font, fill=(60, 58, 54))

    # Axis marker.
    def draw_axis(origin_x, origin_y, mode):
        draw.line((origin_x, origin_y, origin_x + 54, origin_y), fill=(180, 70, 70), width=4)
        draw.text((origin_x + 60, origin_y - 12), "X", font=body_font, fill=(180, 70, 70))
        if mode == "yup":
            draw.line((origin_x, origin_y, origin_x, origin_y - 54), fill=(70, 140, 80), width=4)
            draw.text((origin_x - 10, origin_y - 86), "Y up", font=body_font, fill=(70, 140, 80))
            draw.line((origin_x, origin_y, origin_x - 34, origin_y + 34), fill=(70, 110, 180), width=4)
            draw.text((origin_x - 92, origin_y + 34), "Z", font=body_font, fill=(70, 110, 180))
        else:
            draw.line((origin_x, origin_y, origin_x, origin_y - 54), fill=(70, 110, 180), width=4)
            draw.text((origin_x - 10, origin_y - 86), "Z up", font=body_font, fill=(70, 110, 180))
            draw.line((origin_x, origin_y, origin_x - 34, origin_y + 34), fill=(70, 140, 80), width=4)
            draw.text((origin_x - 92, origin_y + 34), "Y", font=body_font, fill=(70, 140, 80))

    draw_axis(left_x + w - 120, top_y + 110, "yup")
    draw_axis(right_x + w - 120, top_y + 110, "zup")

    sample_world_zup = yup_to_zup_vec(sample_world_tdw)
    sample_velocity_zup = yup_to_zup_vec(sample_velocity_tdw)
    time_sec = idx / fps

    draw.rounded_rectangle((gap, footer_y, canvas.width - gap, canvas.height - gap), radius=18, fill=(255, 255, 255))
    draw.text((gap + 24, footer_y + 20), f"Frame {idx:03d}   t = {time_sec:.2f}s", font=head_font, fill=(24, 22, 19))

    left_lines = [
        "What stays the same:",
        "1. RGB video is identical because the TDW simulation and camera did not change.",
        "2. Image-space appearance does not depend on how you name the vertical axis in the exported metadata.",
        "",
        "Sample TDW world values:",
        f"position_yup = {np.round(sample_world_tdw, 3).tolist()}",
        f"linear_vel_yup = {np.round(sample_velocity_tdw, 3).tolist()}",
    ]
    right_lines = [
        "What changes after offline axis conversion:",
        "1. gravity axis",
        "2. object positions / velocities / quaternions",
        "3. camera extrinsics if you export them",
        "4. any downstream 3D replay, projection, or learning target",
        "",
        "Converted example (x, y, z) -> (x, -z, y):",
        f"position_zup = {np.round(sample_world_zup, 3).tolist()}",
        f"linear_vel_zup = {np.round(sample_velocity_zup, 3).tolist()}",
    ]
    draw_multiline(draw, (gap + 24, footer_y + 64), left_lines, body_font, (60, 58, 54))
    draw_multiline(draw, (canvas.width // 2 + 24, footer_y + 64), right_lines, body_font, (60, 58, 54))

    draw.text((gap + 24, canvas.height - 48),
              "Conclusion: if you only watch the MP4, there is effectively no difference; the difference is in exported 3D truth.",
              font=mono_font, fill=(24, 22, 19))
    return np.asarray(canvas)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(str(SOURCE_VIDEO))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 24.0))
    frames = []
    sample_world_tdw = np.array([0.25, 1.10, -0.40], dtype=np.float32)
    sample_velocity_tdw = np.array([0.00, -1.80, 0.65], dtype=np.float32)
    limit = 72
    for idx, frame in enumerate(reader):
        if idx >= limit:
            break
        frames.append(make_canvas(np.asarray(frame), idx, fps, sample_world_tdw, sample_velocity_tdw))
    reader.close()
    imageio.mimwrite(str(OUTPUT_VIDEO), frames, fps=fps, quality=8)
    imageio.imwrite(str(OUTPUT_POSTER), frames[0])
    payload = {
        "source_video": str(SOURCE_VIDEO),
        "output_video": str(OUTPUT_VIDEO),
        "output_poster": str(OUTPUT_POSTER),
        "axis_conversion": {
            "name": "yup_to_zup",
            "formula": "(x, y, z) -> (x, -z, y)",
            "gravity_tdw_yup": [0.0, -9.81, 0.0],
            "gravity_export_zup": [0.0, 0.0, -9.81],
        },
        "conclusion": "Rendered RGB video is unchanged. Differences only appear in exported coordinates, velocities, quaternions, camera extrinsics, and any downstream 3D reconstruction.",
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT_VIDEO)
    print(OUTPUT_POSTER)
    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()
